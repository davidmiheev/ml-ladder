"""H2. Semantic segmentation with a U-Net — skip connections, Dice vs cross-entropy loss

Lab reference: [open-mmlab/mmocr — detection and recognition reference implementations](https://github.com/open-mmlab/mmocr)

H1 asked *where* the objects are and answered with boxes. This rung asks the
same question at pixel resolution, which changes two things.

First, the architecture. Predicting a label per pixel needs the output at input
resolution, so an encoder that downsamples has to be undone by a decoder that
upsamples — and the fine detail destroyed by pooling cannot be recovered by
upsampling alone. That is what U-Net's skip connections are for, and the rung
ablates them to measure it rather than assert it. It is B4's residual argument
in a second setting: give the signal a path that does not pass through the
bottleneck.

Second, the loss. Cross-entropy averages over pixels, so a class occupying 3%
of the image contributes 3% of the gradient; Dice normalizes by class area and
does not. With small objects on a large background, those two objectives
disagree about what "good" means — and since the *metric* (IoU per class) is
itself area-normalized, the disagreement is measurable.

Run with ``ladder run H2`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, experiment, plotting, synth, torchkit, vision

# -- the metric --------------------------------------------------------------


def confusion(
    prediction: NDArray[np.int64],
    truth: NDArray[np.int64],
    *,
    num_classes: int,
    where: NDArray[np.bool_] | None = None,
) -> NDArray[np.int64]:
    """Pixel confusion matrix, accumulated with a single bincount.

    ``where`` restricts the count to a subset of pixels, which is how the
    boundary-band score below is computed from the same forward pass.
    """
    if where is not None:
        prediction, truth = prediction[where], truth[where]
    flat = truth.ravel() * num_classes + prediction.ravel()
    return np.bincount(flat, minlength=num_classes**2).reshape(num_classes, num_classes)


def iou_per_class(matrix: NDArray[np.int64]) -> NDArray[np.float64]:
    r"""Intersection over union per class, from the confusion matrix.

    For class $c$, with $T_c$ its true pixels and $P_c$ its predicted pixels,

    $$\mathrm{IoU}_c = \frac{|T_c \cap P_c|}{|T_c \cup P_c|} = \frac{M_{cc}}{\sum_j M_{cj} + \sum_i M_{ic} - M_{cc}},$$

    the diagonal entry over the row plus column minus the double-counted
    diagonal. A class absent from both truth and prediction gives 0/0 and is
    returned as NaN, so that averaging can skip it rather than score it 0.
    """
    intersection = np.diag(matrix).astype(np.float64)
    union = matrix.sum(axis=1) + matrix.sum(axis=0) - intersection
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(union > 0, intersection / union, np.nan)


def dice_per_class(matrix: NDArray[np.int64]) -> NDArray[np.float64]:
    r"""Dice (F1) per class: $2|T \cap P| / (|T| + |P|)$.

    Dice and IoU are monotone in each other ($D = 2I/(1+I)$), so they rank
    models identically — Dice is reported only because it is the loss, and
    seeing the loss and the metric side by side is the point of the ablation.
    """
    intersection = np.diag(matrix).astype(np.float64)
    totals = matrix.sum(axis=1) + matrix.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(totals > 0, 2 * intersection / totals, np.nan)


def boundary_band(masks: NDArray[np.int64], *, band: int) -> NDArray[np.bool_]:
    """Pixels within ``band`` of a label change — where the skip claim lives.

    Whole-image mIoU is dominated by object interiors, which are easy: a blob
    roughly in the right place scores well. But the argument for skip
    connections is specifically that upsampling cannot recover the high
    frequencies pooling destroyed, and high frequency means *edges*. Scoring
    only a thin band around each true boundary tests that mechanism instead of
    averaging it away.

    Implemented without scipy: a label change is a disagreement with a shifted
    copy, and the band is that indicator dilated by repeated shifting.
    """
    edges = np.zeros(masks.shape, dtype=bool)
    edges[:, 1:, :] |= masks[:, 1:, :] != masks[:, :-1, :]
    edges[:, :-1, :] |= masks[:, 1:, :] != masks[:, :-1, :]
    edges[:, :, 1:] |= masks[:, :, 1:] != masks[:, :, :-1]
    edges[:, :, :-1] |= masks[:, :, 1:] != masks[:, :, :-1]

    dilated = edges.copy()
    for _ in range(max(band - 1, 0)):
        shifted = dilated.copy()
        shifted[:, 1:, :] |= dilated[:, :-1, :]
        shifted[:, :-1, :] |= dilated[:, 1:, :]
        shifted[:, :, 1:] |= dilated[:, :, :-1]
        shifted[:, :, :-1] |= dilated[:, :, 1:]
        dilated = shifted
    return dilated


def make_unet(*, num_classes: int, width: int, skips: bool):
    """A two-level U-Net, with the skip connections switchable.

    ``skips=False`` keeps every parameter of the decoder and only stops the
    encoder features from being concatenated in — the decoder's input channel
    count is adjusted so the two arms have comparable capacity. Without that
    adjustment the ablation would be confounded by size, which is the usual way
    this comparison gets made badly.
    """
    torch = vision.require_torch()
    from torch import nn

    def block(inputs: int, outputs: int) -> nn.Module:
        return nn.Sequential(
            nn.Conv2d(inputs, outputs, 3, padding=1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
            nn.Conv2d(outputs, outputs, 3, padding=1),
            nn.BatchNorm2d(outputs),
            nn.ReLU(inplace=True),
        )

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.down1 = block(1, width)
            self.down2 = block(width, width * 2)
            self.bottom = block(width * 2, width * 4)
            self.pool = nn.MaxPool2d(2)
            self.up2 = nn.ConvTranspose2d(width * 4, width * 2, 2, stride=2)
            self.up1 = nn.ConvTranspose2d(width * 2, width, 2, stride=2)
            # With skips the decoder sees its own features *and* the encoder's,
            # so the input channel count doubles.
            self.merge2 = block(width * 4 if skips else width * 2, width * 2)
            self.merge1 = block(width * 2 if skips else width, width)
            self.head = nn.Conv2d(width, num_classes, 1)
            self.skips = skips

        def forward(self, images):
            first = self.down1(images)
            second = self.down2(self.pool(first))
            bottom = self.bottom(self.pool(second))

            up = self.up2(bottom)
            up = self.merge2(torch.cat([up, second], dim=1) if self.skips else up)
            up = self.up1(up)
            up = self.merge1(torch.cat([up, first], dim=1) if self.skips else up)
            return self.head(up)

    return UNet


def dice_loss(logits, targets, *, num_classes: int, eps: float = 1.0):
    r"""Soft Dice over classes, computed on probabilities so it is differentiable.

    $$\mathcal{L}_{\text{Dice}} = 1 - \frac{1}{C}\sum_c \frac{2\sum_i p_{ic} y_{ic} + \epsilon}{\sum_i p_{ic} + \sum_i y_{ic} + \epsilon}$$

    The hard Dice score uses an argmax and has zero gradient; replacing the
    indicator with the softmax probability $p_{ic}$ is what makes it trainable.
    Each class contributes one term regardless of how many pixels it owns —
    that normalization is the entire difference from cross-entropy, and the
    reason it is expected to help on small objects.

    ``eps`` smooths the ratio so a class absent from a batch contributes
    $\epsilon/\epsilon = 1$ (a perfect score) rather than $0/0$.
    """
    torch = vision.require_torch()
    probabilities = torch.softmax(logits, dim=1)
    one_hot = torch.nn.functional.one_hot(targets, num_classes).permute(0, 3, 1, 2).float()
    dims = (0, 2, 3)
    intersection = (probabilities * one_hot).sum(dims)
    totals = probabilities.sum(dims) + one_hot.sum(dims)
    return 1.0 - ((2 * intersection + eps) / (totals + eps)).mean()


def segmentation_loss(logits, targets, *, kind: str, num_classes: int):
    """Cross-entropy, Dice, or their sum — the three arms of the loss ablation."""
    torch = vision.require_torch()
    cross_entropy = torch.nn.functional.cross_entropy(logits, targets)
    if kind == "cross_entropy":
        return cross_entropy
    dice = dice_loss(logits, targets, num_classes=num_classes)
    if kind == "dice":
        return dice
    if kind == "sum":
        return cross_entropy + dice
    raise ValueError(f"unknown loss {kind!r}")


@experiment(
    "h2_unet_segmentation",
    defaults={
        "train_images": 2000,
        "test_images": 500,
        "size": 64,
        "width": 12,
        "epochs": 10,
        "batch_size": 32,
        "lr": 2e-3,
        "losses": ["cross_entropy", "dice", "sum"],
        "reference_loss": "cross_entropy",
        # Width of the band around each true boundary that the boundary score
        # is restricted to. 2 px on a 64 px image is ~12% of the pixels.
        "boundary_band": 2,
    },
    description="A U-Net on synthetic shape masks, ablating skip connections and the segmentation loss.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Semantic segmentation with a U-Net — skip connections, Dice vs cross-entropy loss"""
    torch = vision.require_torch()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)
    print(f"[h2] device: {device}")

    size = exp["size"]
    num_classes = len(synth.SHAPE_CLASSES) + 1  # + background
    train_images, train_masks = synth.stack_masks(synth.shapes(exp["train_images"], size=size, rng=rng))
    test_images, test_masks = synth.stack_masks(synth.shapes(exp["test_images"], size=size, rng=rng))

    def to_tensors(images, masks):
        x = torch.from_numpy(images.astype(np.float32) / 255.0).unsqueeze(1)
        y = torch.from_numpy(masks.astype(np.int64))
        return x.to(device), y.to(device)

    x_train, y_train = to_tensors(train_images, train_masks)
    x_test, _ = to_tensors(test_images, test_masks)  # masks are scored as numpy, via the confusion matrix
    band = boundary_band(test_masks, band=exp["boundary_band"])
    print(f"[h2] boundary band covers {band.mean():.1%} of test pixels")

    # The class imbalance is the reason the loss ablation is interesting, so
    # report it rather than leave it implicit.
    fractions = np.bincount(train_masks.ravel(), minlength=num_classes) / train_masks.size
    names = ("background", *synth.SHAPE_CLASSES)
    print("[h2] pixel share: " + "  ".join(f"{name}={share:.3f}" for name, share in zip(names, fractions, strict=True)))

    summary: dict[str, float] = {
        f"pixel_share/{name}": float(share) for name, share in zip(names, fractions, strict=True)
    }
    curves: dict[str, list[float]] = {}
    predictions_for_figure: dict[str, NDArray[np.int64]] = {}

    def arm(name: str, *, skips: bool, loss_kind: str):
        torchkit.seed_torch(exp.seed)
        model = make_unet(num_classes=num_classes, width=exp["width"], skips=skips)().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=1e-4)
        losses: list[float] = []
        for _epoch in range(exp["epochs"]):
            model.train()
            permutation = torch.randperm(len(x_train), device=device)
            epoch_losses = []
            for start in range(0, len(x_train), exp["batch_size"]):
                index = permutation[start : start + exp["batch_size"]]
                loss = segmentation_loss(model(x_train[index]), y_train[index], kind=loss_kind, num_classes=num_classes)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            losses.append(float(np.mean(epoch_losses)))

        model.eval()
        matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        band_matrix = np.zeros((num_classes, num_classes), dtype=np.int64)
        predicted_chunks = []
        with torch.no_grad():
            for start in range(0, len(x_test), 128):
                logits = model(x_test[start : start + 128])
                predicted = logits.argmax(dim=1).cpu().numpy()
                predicted_chunks.append(predicted)
                chunk_truth = test_masks[start : start + 128]
                matrix += confusion(predicted, chunk_truth, num_classes=num_classes)
                band_matrix += confusion(
                    predicted, chunk_truth, num_classes=num_classes, where=band[start : start + 128]
                )

        ious, dices = iou_per_class(matrix), dice_per_class(matrix)
        parameters = torchkit.count_parameters(model)
        curves[name] = losses
        predictions_for_figure[name] = np.concatenate(predicted_chunks)

        # mIoU over the foreground only: including background — 85% of pixels
        # and trivially easy — flatters every arm and hides the differences
        # the rung is looking for. Both are reported.
        summary[f"{name}/mIoU"] = float(np.nanmean(ious))
        summary[f"{name}/mIoU_foreground"] = float(np.nanmean(ious[1:]))
        summary[f"{name}/mDice_foreground"] = float(np.nanmean(dices[1:]))
        summary[f"{name}/pixel_accuracy"] = float(np.diag(matrix).sum() / matrix.sum())
        summary[f"{name}/parameters"] = float(parameters)
        # The same predictions, scored only near a true boundary.
        band_ious = iou_per_class(band_matrix)
        summary[f"{name}/mIoU_boundary"] = float(np.nanmean(band_ious[1:]))
        summary[f"{name}/pixel_accuracy_boundary"] = float(np.diag(band_matrix).sum() / band_matrix.sum())
        for class_id, iou in enumerate(ious):
            if not np.isnan(iou):
                summary[f"{name}/IoU/{names[class_id]}"] = float(iou)
        print(
            f"[h2] {name:<26} mIoU={summary[f'{name}/mIoU']:.4f}  "
            f"fg-mIoU={summary[f'{name}/mIoU_foreground']:.4f}  "
            f"boundary-mIoU={summary[f'{name}/mIoU_boundary']:.4f}  "
            f"pixel-acc={summary[f'{name}/pixel_accuracy']:.4f}  ({parameters:,} params)"
        )
        return matrix

    # -- the skip-connection ablation, at the reference loss -----------------
    reference = exp["reference_loss"]
    arm(f"skips/{reference}", skips=True, loss_kind=reference)
    arm(f"no_skips/{reference}", skips=False, loss_kind=reference)

    # -- the loss ablation, with skips on --------------------------------------
    reference_matrix = None
    for loss_kind in exp["losses"]:
        if loss_kind == reference:
            continue
        matrix = arm(f"skips/{loss_kind}", skips=True, loss_kind=loss_kind)
        if reference_matrix is None:
            reference_matrix = matrix

    # -- derived comparisons, guarded on what was actually swept -------------
    skip_key, no_skip_key = f"skips/{reference}", f"no_skips/{reference}"
    summary["skip_gain/mIoU_foreground"] = (
        summary[f"{skip_key}/mIoU_foreground"] - summary[f"{no_skip_key}/mIoU_foreground"]
    )
    summary["skip_gain/pixel_accuracy"] = (
        summary[f"{skip_key}/pixel_accuracy"] - summary[f"{no_skip_key}/pixel_accuracy"]
    )
    # The comparison the mechanism predicts: skips should buy more at the
    # boundary than in the interior. If these two are equal, whatever the
    # skips are doing is not what the textbook says they do.
    summary["skip_gain/mIoU_boundary"] = summary[f"{skip_key}/mIoU_boundary"] - summary[f"{no_skip_key}/mIoU_boundary"]
    summary["skip_gain/boundary_over_whole"] = summary["skip_gain/mIoU_boundary"] - summary["skip_gain/mIoU_foreground"]
    print(
        f"[h2] skip gain: whole={summary['skip_gain/mIoU_foreground']:+.4f}  "
        f"boundary={summary['skip_gain/mIoU_boundary']:+.4f}"
    )
    swept = [kind for kind in exp["losses"] if f"skips/{kind}/mIoU_foreground" in summary]
    if swept:
        best = max(swept, key=lambda kind: summary[f"skips/{kind}/mIoU_foreground"])
        summary["best/mIoU_foreground"] = summary[f"skips/{best}/mIoU_foreground"]
        summary["best/loss_is_dice"] = float(best == "dice")
        spread = [summary[f"skips/{kind}/mIoU_foreground"] for kind in swept]
        summary["loss_spread/mIoU_foreground"] = max(spread) - min(spread)
        print(f"[h2] best loss by foreground mIoU: {best} ({summary['best/mIoU_foreground']:.4f})")

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "training",
        plotting.plot_curves(
            curves,
            x=list(range(1, exp["epochs"] + 1)),
            title="Segmentation loss by arm (different objectives are not comparable in value)",
            xlabel="epoch",
            ylabel="loss",
        ),
    )
    exp.log_figure(
        "arms",
        plotting.plot_comparison(
            {name.replace("skips/", "skip+"): summary[f"{name}/mIoU_foreground"] for name in curves},
            title="Foreground mIoU by arm",
            ylabel="mIoU (shapes only)",
        ),
    )

    shown = 6
    panels = [test_images[:shown], test_masks[:shown]]
    titles = ["image", "truth"]
    for name in (skip_key, no_skip_key):
        panels.append(predictions_for_figure[name][:shown])
        titles.append(name)
    exp.log_figure(
        "masks",
        plotting.plot_panel_rows(
            panels,
            titles,
            vmax=num_classes - 1,
            title="Predicted masks (rows: image, truth, then each arm)",
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
