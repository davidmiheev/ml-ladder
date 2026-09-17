"""H1. Object detection from scratch — IoU, non-max suppression and mAP implemented by hand

Lab reference: [open-mmlab/mmocr — detection and recognition reference implementations](https://github.com/open-mmlab/mmocr)

Every rung before this one had a metric handed to it. Accuracy needs no
argument: the prediction either equals the label or it does not. Detection has
no such luxury — a model emits a variable number of boxes, each with a score and
a class, and *nothing* says which prediction corresponds to which object. The
metric has to be constructed, and the construction involves at least three
choices (how much overlap counts as a match, how duplicates are removed, how a
precision-recall curve becomes one number) that change the answer.

So this rung writes IoU, non-max suppression and mean average precision from
scratch, then measures how much the score moves when only those choices change
and the detector stays fixed. The detector itself is deliberately the simplest
thing that works: one YOLO-style prediction per cell of an 8x8 grid.

Run with ``ladder run H1`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, experiment, plotting, synth, torchkit, vision

# -- the metric, built from nothing ------------------------------------------


def iou_matrix(a: NDArray[np.float32], b: NDArray[np.float32]) -> NDArray[np.float32]:
    r"""Pairwise intersection-over-union between two sets of boxes.

    For boxes $A$ and $B$,

    $$\mathrm{IoU}(A, B) = \frac{|A \cap B|}{|A \cup B|} = \frac{|A \cap B|}{|A| + |B| - |A \cap B|}.$$

    The intersection is a box itself — the coordinate-wise max of the corners
    and min of the far corners — which is why this is cheap enough to run on
    every prediction-truth pair. Returns an ``(len(a), len(b))`` matrix.
    """
    if len(a) == 0 or len(b) == 0:
        return np.zeros((len(a), len(b)), dtype=np.float32)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    intersection = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return (intersection / np.maximum(area_a + area_b - intersection, 1e-9)).astype(np.float32)


def nms(boxes: NDArray[np.float32], scores: NDArray[np.float32], threshold: float) -> NDArray[np.int64]:
    """Greedy non-max suppression: keep the best box, drop what it overlaps.

    The algorithm is three lines and one assumption — that two boxes
    overlapping by more than ``threshold`` describe the same object. That
    assumption is wrong whenever real objects overlap that much, which is
    exactly why the threshold is swept below rather than set to a default.
    """
    order = np.argsort(-scores)
    keep: list[int] = []
    while len(order):
        best = int(order[0])
        keep.append(best)
        if len(order) == 1:
            break
        overlaps = iou_matrix(boxes[best][None], boxes[order[1:]])[0]
        order = order[1:][overlaps <= threshold]
    return np.array(keep, dtype=np.int64)


def average_precision(
    predictions: list[tuple[int, float, NDArray[np.float32]]],
    truths: list[tuple[int, NDArray[np.float32]]],
    *,
    iou_threshold: float,
) -> float:
    r"""Average precision for one class, by the all-points interpolated rule.

    Predictions are ``(image_id, score, box)``, truths are ``(image_id, box)``.
    Sort every prediction by descending score and walk down the list: a
    prediction is a true positive if it clears ``iou_threshold`` against a truth
    in the same image *that nothing better has already claimed*, and a false
    positive otherwise. The "already claimed" rule is what makes duplicates
    cost something, and therefore what makes NMS worth running.

    Each prefix of the walk gives a (recall, precision) point. AP is the area
    under the monotone envelope of that curve:

    $$\mathrm{AP} = \sum_k (r_k - r_{k-1}) \max_{\tilde{r} \ge r_k} p(\tilde{r}).$$

    Taking the running maximum of precision from the right is the "interpolated"
    part: it ignores the wiggles that come from one lucky detection and is the
    convention COCO and VOC (2010 onward) both use.
    """
    if len(truths) == 0:
        return float("nan")  # undefined, not zero — the class is absent
    by_image: dict[int, list[NDArray[np.float32]]] = {}
    for image_id, box in truths:
        by_image.setdefault(image_id, []).append(box)
    claimed = {image_id: np.zeros(len(boxes), dtype=bool) for image_id, boxes in by_image.items()}

    true_positives = np.zeros(len(predictions), dtype=np.float64)
    false_positives = np.zeros(len(predictions), dtype=np.float64)
    for rank, (image_id, _score, box) in enumerate(sorted(predictions, key=lambda item: -item[1])):
        candidates = by_image.get(image_id)
        if not candidates:
            false_positives[rank] = 1.0
            continue
        overlaps = iou_matrix(box[None], np.stack(candidates))[0]
        best = int(np.argmax(overlaps))
        if overlaps[best] >= iou_threshold and not claimed[image_id][best]:
            claimed[image_id][best] = True
            true_positives[rank] = 1.0
        else:
            false_positives[rank] = 1.0

    cumulative_tp = np.cumsum(true_positives)
    recall = cumulative_tp / len(truths)
    precision = cumulative_tp / np.maximum(cumulative_tp + np.cumsum(false_positives), 1e-9)

    # Pad so the curve starts at recall 0, then take the precision envelope.
    recall = np.concatenate([[0.0], recall, [recall[-1] if len(recall) else 0.0]])
    precision = np.concatenate([[1.0], precision, [0.0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]
    return float(np.sum(np.diff(recall) * precision[1:]))


def mean_average_precision(
    predictions: list[tuple[int, int, float, NDArray[np.float32]]],
    truths: list[tuple[int, int, NDArray[np.float32]]],
    *,
    num_classes: int,
    iou_threshold: float = 0.5,
) -> tuple[float, dict[int, float]]:
    """mAP: average precision per class, averaged over classes that exist.

    Classes with no ground truth are skipped rather than scored 0 — averaging
    in a zero for an absent class is a silent way to make a detector look bad
    on a dataset that simply never showed it that class.
    """
    per_class: dict[int, float] = {}
    for class_id in range(num_classes):
        class_predictions = [(image, score, box) for image, label, score, box in predictions if label == class_id]
        class_truths = [(image, box) for image, label, box in truths if label == class_id]
        score = average_precision(class_predictions, class_truths, iou_threshold=iou_threshold)
        if not np.isnan(score):
            per_class[class_id] = score
    return (float(np.mean(list(per_class.values()))) if per_class else float("nan")), per_class


# -- the detector ------------------------------------------------------------


def make_detector(*, grid: int, width: int, num_classes: int):
    """A one-box-per-cell detector: conv stack down to ``grid``, then a head.

    Each cell predicts objectness, a box, and a class distribution. The box is
    parameterised relative to its cell — centre as a sigmoid offset inside the
    cell, size as a log-ratio of the image — which is YOLO's trick and matters
    for a reason worth stating: predicting absolute pixel coordinates asks
    every cell to learn where it is, while predicting offsets lets one shared
    convolution work everywhere.
    """
    vision.require_torch()
    from itertools import pairwise

    from torch import nn

    class Detector(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            channels = [1, width, width * 2, width * 4]
            blocks: list[nn.Module] = []
            for inp, out in pairwise(channels):
                blocks += [
                    nn.Conv2d(inp, out, 3, padding=1),
                    nn.BatchNorm2d(out),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out, out, 3, padding=1),
                    nn.BatchNorm2d(out),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2),  # three pools: 64 -> 8
                ]
            self.backbone = nn.Sequential(*blocks)
            # 1 objectness + 4 box + num_classes, per cell.
            self.head = nn.Conv2d(width * 4, 5 + num_classes, 1)
            self.grid = grid
            self.num_classes = num_classes

        def forward(self, images):
            features = self.backbone(images)
            raw = self.head(features)  # (n, 5+c, g, g)
            return raw.permute(0, 2, 3, 1)  # (n, g, g, 5+c)

    return Detector


def encode_targets(scenes: list[synth.ShapeScene], *, grid: int, size: int, num_classes: int):
    """Assign each ground-truth box to the cell holding its centre.

    This is YOLOv1's assignment rule and it has a consequence worth measuring:
    two objects whose centres land in the same cell cannot both be predicted,
    because the cell has one slot. The rung reports how often that happens so
    the ceiling it imposes is visible rather than mysterious.
    """
    torch = vision.require_torch()
    cell = size / grid
    objectness = torch.zeros(len(scenes), grid, grid)
    boxes = torch.zeros(len(scenes), grid, grid, 4)
    labels = torch.zeros(len(scenes), grid, grid, dtype=torch.long)
    collisions = 0
    for index, scene in enumerate(scenes):
        for (x1, y1, x2, y2), label in zip(scene.boxes, scene.labels, strict=True):
            centre_x, centre_y = (x1 + x2) / 2, (y1 + y2) / 2
            column, row = int(centre_x // cell), int(centre_y // cell)
            column, row = min(column, grid - 1), min(row, grid - 1)
            if objectness[index, row, column] == 1:
                collisions += 1
            objectness[index, row, column] = 1.0
            boxes[index, row, column] = torch.tensor(
                [
                    centre_x / cell - column,  # offset inside the cell, in [0, 1)
                    centre_y / cell - row,
                    np.log(max(x2 - x1, 1.0) / size),  # log-size, so scale is additive
                    np.log(max(y2 - y1, 1.0) / size),
                ]
            )
            labels[index, row, column] = int(label)
    return objectness, boxes, labels, collisions


def decode(raw, *, grid: int, size: int, score_threshold: float):
    """Turn the grid tensor back into per-image boxes, scores and classes."""
    torch = vision.require_torch()
    cell = size / grid
    objectness = torch.sigmoid(raw[..., 0])
    offsets = torch.sigmoid(raw[..., 1:3])
    log_size = raw[..., 3:5]
    class_scores = torch.softmax(raw[..., 5:], dim=-1)
    best_class = class_scores.argmax(dim=-1)

    rows = torch.arange(grid, device=raw.device).view(1, grid, 1)
    columns = torch.arange(grid, device=raw.device).view(1, 1, grid)
    centre_x = (columns + offsets[..., 0]) * cell
    centre_y = (rows + offsets[..., 1]) * cell
    box_width = torch.exp(log_size[..., 0]) * size
    box_height = torch.exp(log_size[..., 1]) * size
    boxes = torch.stack(
        [centre_x - box_width / 2, centre_y - box_height / 2, centre_x + box_width / 2, centre_y + box_height / 2],
        dim=-1,
    )

    # The score is objectness times class confidence: a cell that is sure
    # something is there but unsure what it is should rank below one that is
    # sure of both, and multiplying is how YOLO expresses that.
    scores = objectness * class_scores.max(dim=-1).values
    results = []
    for index in range(len(raw)):
        flat_scores = scores[index].reshape(-1)
        keep = flat_scores >= score_threshold
        results.append(
            (
                boxes[index].reshape(-1, 4)[keep].detach().cpu().numpy(),
                flat_scores[keep].detach().cpu().numpy(),
                best_class[index].reshape(-1)[keep].detach().cpu().numpy(),
            )
        )
    return results


def detection_loss(raw, objectness, boxes, labels, *, no_object_weight: float, box_weight: float):
    r"""Objectness everywhere, box and class only where an object is.

    $$\mathcal{L} = \mathcal{L}_{\text{obj}} + \lambda_{\text{box}}\,\mathcal{L}_{\text{box}}\!\restriction_{\text{obj}} + \mathcal{L}_{\text{cls}}\!\restriction_{\text{obj}}$$

    The restriction is not an optimization — it is a correctness requirement.
    An empty cell has no box to regress toward, so including it would train the
    head to predict the mean of nothing. And because an 8x8 grid holds 64 cells
    for typically two objects, the negatives outnumber positives ~30:1, which
    is what ``no_object_weight`` is for: unweighted, predicting "nothing here"
    everywhere is already a good loss.
    """
    torch = vision.require_torch()
    import torch.nn.functional as F

    object_mask = objectness > 0.5
    weight = torch.where(object_mask, torch.ones_like(objectness), torch.full_like(objectness, no_object_weight))
    loss_obj = F.binary_cross_entropy_with_logits(raw[..., 0], objectness, weight=weight)

    if object_mask.any():
        predicted = torch.cat([torch.sigmoid(raw[..., 1:3]), raw[..., 3:5]], dim=-1)[object_mask]
        loss_box = F.mse_loss(predicted, boxes[object_mask])
        loss_cls = F.cross_entropy(raw[..., 5:][object_mask], labels[object_mask])
    else:  # pragma: no cover - only with an empty batch
        loss_box = loss_cls = torch.zeros((), device=raw.device)
    return loss_obj + box_weight * loss_box + loss_cls, loss_obj, loss_box, loss_cls


def collect(results, scenes, *, nms_threshold: float | None):
    """Flatten per-image detections into the (image, class, score, box) lists mAP wants."""
    predictions: list[tuple[int, int, float, NDArray[np.float32]]] = []
    truths: list[tuple[int, int, NDArray[np.float32]]] = []
    for image_id, ((boxes, scores, classes), scene) in enumerate(zip(results, scenes, strict=True)):
        if nms_threshold is not None and len(boxes):
            # NMS is per class: two different objects of different classes may
            # legitimately sit on top of each other.
            keep: list[int] = []
            for class_id in np.unique(classes):
                mask = np.flatnonzero(classes == class_id)
                keep.extend(mask[nms(boxes[mask], scores[mask], nms_threshold)].tolist())
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        for box, score, class_id in zip(boxes, scores, classes, strict=True):
            predictions.append((image_id, int(class_id), float(score), box.astype(np.float32)))
        for box, label in zip(scene.boxes, scene.labels, strict=True):
            truths.append((image_id, int(label), box.astype(np.float32)))
    return predictions, truths


@experiment(
    "h1_detection_from_scratch",
    defaults={
        "train_images": 3000,
        "test_images": 600,
        "size": 64,
        "grid": 8,
        "width": 16,
        "epochs": 12,
        "batch_size": 64,
        "lr": 2e-3,
        "no_object_weight": 0.2,
        "box_weight": 5.0,
        "score_threshold": 0.25,
        "nms_thresholds": [0.1, 0.3, 0.5, 0.7, 0.9],
        "iou_thresholds": [0.5, 0.75],
        "reference_nms": 0.5,
    },
    description="A YOLO-style detector on synthetic shapes, scored with hand-written IoU, NMS and mAP.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Object detection from scratch — IoU, non-max suppression and mAP implemented by hand"""
    torch = vision.require_torch()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)
    print(f"[h1] device: {device}")

    size, grid = exp["size"], exp["grid"]
    num_classes = len(synth.SHAPE_CLASSES)
    train_scenes = synth.shapes(exp["train_images"], size=size, rng=rng)
    test_scenes = synth.shapes(exp["test_images"], size=size, rng=rng)

    def to_tensor(scenes):
        images = np.stack([scene.image for scene in scenes]).astype(np.float32) / 255.0
        return torch.from_numpy(images).unsqueeze(1)

    x_train, x_test = to_tensor(train_scenes).to(device), to_tensor(test_scenes).to(device)
    objectness, boxes, labels, collisions = encode_targets(train_scenes, grid=grid, size=size, num_classes=num_classes)
    objectness, boxes, labels = objectness.to(device), boxes.to(device), labels.to(device)

    total_objects = sum(len(scene.boxes) for scene in train_scenes)
    collision_rate = collisions / max(total_objects, 1)
    print(
        f"[h1] {len(train_scenes)} train scenes, {total_objects} objects; "
        f"{collisions} ({collision_rate:.2%}) lost to cell collisions at grid {grid}"
    )

    torchkit.seed_torch(exp.seed)
    model = make_detector(grid=grid, width=exp["width"], num_classes=num_classes)().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=1e-4)
    print(f"[h1] detector: {torchkit.count_parameters(model):,} parameters")

    losses: list[float] = []
    for epoch in range(1, exp["epochs"] + 1):
        model.train()
        permutation = torch.randperm(len(x_train), device=device)
        epoch_losses = []
        for start in range(0, len(x_train), exp["batch_size"]):
            index = permutation[start : start + exp["batch_size"]]
            raw = model(x_train[index])
            loss, loss_obj, loss_box, loss_cls = detection_loss(
                raw,
                objectness[index],
                boxes[index],
                labels[index],
                no_object_weight=exp["no_object_weight"],
                box_weight=exp["box_weight"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(float(np.mean(epoch_losses)))
        if epoch % max(exp["epochs"] // 4, 1) == 0:
            print(
                f"[h1] epoch {epoch:>3}  loss={losses[-1]:.4f}  "
                f"(obj={loss_obj.item():.3f} box={loss_box.item():.3f} cls={loss_cls.item():.3f})"
            )

    # -- evaluate: the detector is now fixed, only the metric's knobs move ----
    model.eval()
    with torch.no_grad():
        raw_test = torch.cat([model(chunk) for chunk in x_test.split(256)])
    results = decode(raw_test, grid=grid, size=size, score_threshold=exp["score_threshold"])
    raw_count = float(np.mean([len(boxes_) for boxes_, _, _ in results]))

    summary: dict[str, float] = {
        "cell_collision_rate": collision_rate,
        "train_loss": losses[-1],
        "boxes_per_image/no_nms": raw_count,
    }

    # The headline comparison: the same predictions, with and without NMS.
    for nms_threshold in [None, *exp["nms_thresholds"]]:
        predictions, truths = collect(results, test_scenes, nms_threshold=nms_threshold)
        name = "no_nms" if nms_threshold is None else f"nms@{nms_threshold}"
        kept = len(predictions) / len(test_scenes)
        if nms_threshold is not None:
            summary[f"boxes_per_image/{name}"] = kept
        for iou_threshold in exp["iou_thresholds"]:
            score, _per_class = mean_average_precision(
                predictions, truths, num_classes=num_classes, iou_threshold=iou_threshold
            )
            summary[f"mAP@{iou_threshold}/{name}"] = score
        print(
            f"[h1] {name:<9} boxes/image={kept:5.2f}  "
            + "  ".join(f"mAP@{t}={summary[f'mAP@{t}/{name}']:.4f}" for t in exp["iou_thresholds"])
        )

    # The derived metrics below are the point of the rung, but every one of
    # them names a specific threshold that ``--iou-thresholds`` and
    # ``--nms-thresholds`` are free to exclude. Part C lost a run to exactly
    # this (a summary key that existed only for the default sweep), so each is
    # guarded on the values actually swept rather than assumed.
    reference = f"nms@{exp['reference_nms']}"
    loose, strict = min(exp["iou_thresholds"]), max(exp["iou_thresholds"])
    if exp["reference_nms"] in exp["nms_thresholds"]:
        best_nms = max(exp["nms_thresholds"], key=lambda t: summary[f"mAP@{loose}/nms@{t}"])
        summary["best/nms_threshold"] = float(best_nms)
        summary[f"best/mAP@{loose}"] = summary[f"mAP@{loose}/nms@{best_nms}"]
        summary[f"nms_gain@{loose}"] = summary[f"mAP@{loose}/{reference}"] - summary[f"mAP@{loose}/no_nms"]
        if strict != loose:
            # How much of the score is localization rather than detection: the
            # same predictions, scored against a stricter idea of "correct".
            summary["localization_penalty"] = summary[f"mAP@{loose}/{reference}"] - summary[f"mAP@{strict}/{reference}"]
            # How much the *metric's* knob moves the answer, with the detector
            # frozen — the comparison the rung exists to make.
            spread = [summary[f"mAP@{loose}/nms@{t}"] for t in exp["nms_thresholds"]]
            summary[f"nms_spread@{loose}"] = max(spread) - min(spread)
            strict_spread = [summary[f"mAP@{strict}/nms@{t}"] for t in exp["nms_thresholds"]]
            summary[f"nms_spread@{strict}"] = max(strict_spread) - min(strict_spread)

    # -- per-class AP at the reference setting -------------------------------
    predictions, truths = collect(results, test_scenes, nms_threshold=exp["reference_nms"])
    for iou_threshold in exp["iou_thresholds"]:
        _, per_class = mean_average_precision(predictions, truths, num_classes=num_classes, iou_threshold=iou_threshold)
        for class_id, score in per_class.items():
            summary[f"AP@{iou_threshold}/{synth.SHAPE_CLASSES[class_id]}"] = score
        print(
            f"[h1] per-class AP@{iou_threshold}: "
            + "  ".join(f"{synth.SHAPE_CLASSES[k]}={v:.3f}" for k, v in per_class.items())
        )

    # -- where the localization ceiling actually is --------------------------
    # mAP@0.75 being far below mAP@0.5 says the boxes are loose, but not by
    # how much or in which direction. This measures it directly: for every
    # ground-truth object, the IoU of the best surviving detection.
    matched: list[float] = []
    size_error: list[float] = []
    for (boxes_, scores_, _classes), scene in zip(results, test_scenes, strict=True):
        if len(boxes_) == 0 or len(scene.boxes) == 0:
            matched.extend([0.0] * len(scene.boxes))
            continue
        keep = nms(boxes_, scores_, exp["reference_nms"])
        overlaps = iou_matrix(boxes_[keep], scene.boxes)
        for column in range(len(scene.boxes)):
            row = int(np.argmax(overlaps[:, column]))
            matched.append(float(overlaps[row, column]))
            predicted, truth = boxes_[keep][row], scene.boxes[column]
            size_error.append(float((predicted[2] - predicted[0]) - (truth[2] - truth[0])))
    matched_iou = np.array(matched)
    summary["matched_iou/mean"] = float(matched_iou.mean())
    summary["matched_iou/median"] = float(np.median(matched_iou))
    for level in (0.5, 0.75, 0.9):
        summary[f"matched_iou/above_{level}"] = float((matched_iou >= level).mean())
    if size_error:  # empty when the detector emits nothing, e.g. undertrained
        summary["width_error_px/bias"] = float(np.mean(size_error))
        summary["width_error_px/abs"] = float(np.mean(np.abs(size_error)))
    print(
        f"[h1] matched IoU mean={matched_iou.mean():.3f}  "
        f">=0.5:{(matched_iou >= 0.5).mean():.3f}  >=0.75:{(matched_iou >= 0.75).mean():.3f}  "
        f">=0.9:{(matched_iou >= 0.9).mean():.3f}"
        + (f"  |width error|={np.mean(np.abs(size_error)):.2f}px" if size_error else "  (no detections)")
    )

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "training",
        plotting.plot_curves(
            {"total": losses},
            x=list(range(1, exp["epochs"] + 1)),
            title="Detection loss",
            xlabel="epoch",
            ylabel="loss",
            logy=True,
        ),
    )
    exp.log_figure(
        "nms_sweep",
        plotting.plot_curves(
            {f"mAP@{t}": [summary[f"mAP@{t}/nms@{n}"] for n in exp["nms_thresholds"]] for t in exp["iou_thresholds"]},
            x=exp["nms_thresholds"],
            title="The detector is fixed; only the NMS threshold moves",
            xlabel="NMS IoU threshold",
            ylabel="mAP",
            marker="o",
        ),
    )
    exp.log_figure(
        "boxes_per_image",
        plotting.plot_curves(
            {"kept": [summary[f"boxes_per_image/nms@{n}"] for n in exp["nms_thresholds"]]},
            x=exp["nms_thresholds"],
            title=f"Boxes kept per image (before NMS: {raw_count:.1f})",
            xlabel="NMS IoU threshold",
            ylabel="boxes per image",
            marker="o",
        ),
    )

    exp.log_figure(
        "matched_iou",
        plotting.plot_histogram(
            matched_iou,
            bins=np.linspace(0, 1, 26),
            markers={f"IoU {level}": level for level in exp["iou_thresholds"]},
            title="IoU of the best detection per ground-truth object",
            xlabel="IoU with ground truth",
            ylabel="objects",
        ),
    )

    shown = 8
    kept_results = []
    for boxes_, scores_, classes_ in results[:shown]:
        if len(boxes_):
            keep = nms(boxes_, scores_, exp["reference_nms"])
            kept_results.append((boxes_[keep], scores_[keep], classes_[keep]))
        else:
            kept_results.append((boxes_, scores_, classes_))
    exp.log_figure(
        "detections",
        plotting.plot_boxes(
            np.stack([scene.image for scene in test_scenes[:shown]]),
            [item[0] for item in kept_results],
            truths=[scene.boxes for scene in test_scenes[:shown]],
            labels=[item[2] for item in kept_results],
            scores=[item[1] for item in kept_results],
            class_names=synth.SHAPE_CLASSES,
            nrow=4,
            title=f"Detections at NMS {exp['reference_nms']} (dashed grey = ground truth)",
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
