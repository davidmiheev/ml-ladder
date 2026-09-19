"""F1. Rotation-equivariant CNN on a rotated-MNIST task, compared against a standard CNN

Lab reference: [escnn — equivariant CNN library, and PyTorch Geometric tutorials](https://github.com/QUVA-Lab/escnn)

Every convolution in parts B and H was already equivariant to one group:
translation. That is the whole reason convolution is used instead of a dense
layer — shifting the input shifts the feature map, so a feature learned in one
place is available everywhere, and the weights are shared rather than relearned.

Nothing about that argument is special to translation. It goes through for any
group whose action commutes with the layer, and the cost is the same: weights
shared across the group's orbits instead of learned per orbit. This rung builds
that for $p4$ — translations plus the four 90-degree rotations — and measures
the thing that makes it different in kind from augmentation.

**Augmentation buys invariance by showing the network every orbit. Equivariance
buys it by construction, so the network cannot fail at it even on inputs it has
never seen.** The first is a statement about the training distribution and the
second is a statement about the function. That distinction is either worth
parameters or it isn't, and it is measurable: an exactly invariant network has
*zero* invariance error at the group elements, to floating-point, on any input
at all -- including ones drawn from a distribution it was never trained on.

Run with ``ladder run F1`` (``--help`` for flags).
"""

from __future__ import annotations

import math

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, experiment, plotting, torchkit, vision

# The four rotations of p4, as quarter turns.
ROTATIONS = 4


def rotate_batch(images, degrees: float):
    r"""Rotate a batch of images about their centre by ``degrees``.

    Bilinear resampling on an affine grid, so this is defined for every angle
    rather than only the multiples of 90 that ``rot90`` covers. That matters:
    the point of this rung is to evaluate a $p4$-equivariant network *off* its
    group, where the guarantee does not apply and only interpolation error and
    genuine approximation are left.

    The grid is built in normalized ``[-1, 1]`` coordinates, where a rotation
    about the image centre is just a rotation of the sampling grid by the
    inverse angle.
    """
    torch = vision.require_torch()
    if degrees % 360 == 0:
        return images.clone()
    theta = math.radians(degrees)
    cos, sin = math.cos(theta), math.sin(theta)
    # Sampling grid uses the inverse rotation: to read the output at x we look
    # up the input at R^{-1}x.
    matrix = torch.tensor([[cos, sin, 0.0], [-sin, cos, 0.0]], dtype=images.dtype, device=images.device)
    matrix = matrix.unsqueeze(0).expand(images.shape[0], -1, -1)
    grid = torch.nn.functional.affine_grid(matrix, list(images.shape), align_corners=False)
    return torch.nn.functional.grid_sample(images, grid, align_corners=False, padding_mode="zeros")


def make_lifting_conv(in_channels: int, out_channels: int, kernel: int):
    r"""$\mathbb{Z}^2 \to p4$: one filter, applied in four orientations.

    The first layer of a group CNN cannot be a group convolution, because its
    input has no orientation axis yet -- an image is a function on
    $\mathbb{Z}^2$, not on $p4$. The lifting layer creates that axis: it
    convolves with the *same* filter rotated by each $r \in C_4$, so

    $$[f \star_{\text{lift}} \psi](x, r) = \sum_{y} f(y)\,\psi\!\left(r^{-1}(y - x)\right),$$

    and the output is a function on the group. The parameter count is that of
    one planar filter bank; the *feature* count is four times larger. Rotating
    the input by a quarter turn permutes those four responses cyclically and
    rotates each spatially, which is exactly the regular representation of
    $C_4$ acting on the new axis -- the property the next layer relies on.
    """
    torch = vision.require_torch()

    class Lifting(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, kernel, kernel))
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))

        def forward(self, x):
            # rot90 on the *weights*, not the input: the four orientations of
            # one filter, stacked into the output-channel axis.
            filters = torch.cat([torch.rot90(self.weight, r, dims=(2, 3)) for r in range(ROTATIONS)], dim=0)
            bias = self.bias.repeat(ROTATIONS)
            out = torch.nn.functional.conv2d(x, filters, bias, padding=kernel // 2)
            n, _, h, w = out.shape
            # [n, r * out, h, w] -> [n, out, r, h, w]; r is the group axis.
            return out.view(n, ROTATIONS, out_channels, h, w).permute(0, 2, 1, 3, 4)

    return Lifting


def make_group_conv(in_channels: int, out_channels: int, kernel: int):
    r"""$p4 \to p4$: rotate the filter *and* cycle the orientation axis.

    Once the feature map lives on the group, equivariance requires the filter
    to transform with it. For output orientation $r$ the filter is rotated
    spatially by $r$ **and** its input-orientation axis is cyclically shifted
    by $r$:

    $$[f \star \psi](x, r) = \sum_{y}\sum_{s \in C_4} f(y, s)\,\psi\!\left(r^{-1}(y-x),\, r^{-1}s\right).$$

    The spatial rotation alone is the mistake that is easy to make and hard to
    see: it gives a network that looks equivariant on a single layer and drifts
    over depth, because the orientation channels stop corresponding to the
    orientations they are named after. The ``roll`` is what keeps the bookkeeping
    honest, and a test pins it.
    """
    torch = vision.require_torch()

    class GroupConv(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.empty(out_channels, in_channels, ROTATIONS, kernel, kernel))
            torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
            self.bias = torch.nn.Parameter(torch.zeros(out_channels))

        def forward(self, x):
            n, _, _, h, w = x.shape
            filters = []
            for r in range(ROTATIONS):
                rotated = torch.rot90(self.weight, r, dims=(3, 4))
                filters.append(torch.roll(rotated, shifts=r, dims=2))
            # [out * r, in * r, k, k]
            stacked = torch.cat(filters, dim=0).reshape(
                out_channels * ROTATIONS, in_channels * ROTATIONS, kernel, kernel
            )
            bias = self.bias.repeat(ROTATIONS)
            flat = x.reshape(n, in_channels * ROTATIONS, h, w)
            out = torch.nn.functional.conv2d(flat, stacked, bias, padding=kernel // 2)
            return out.view(n, ROTATIONS, out_channels, h, w).permute(0, 2, 1, 3, 4)

    return GroupConv


def make_p4_cnn(width: int = 16, num_classes: int = 10, kernel: int = 3):
    r"""Lift, two group convolutions, then pool the group away.

    Invariance is *not* what the convolutions provide -- they provide
    equivariance, which is the useful thing to keep through depth. Invariance
    comes at the end, from a max over the orientation axis:

    $$h(f) = \max_{r \in C_4} f(\cdot, r),$$

    which is constant under the $C_4$ action because the action only permutes
    the terms of the max. Pool earlier and the network throws away the
    orientation information the later layers could still use; never pool and
    the classifier sees a representation that moves when the input rotates.
    """
    torch = vision.require_torch()
    Lifting, GroupA, GroupB = (
        make_lifting_conv(1, width, kernel),
        make_group_conv(width, width, kernel),
        make_group_conv(width, 2 * width, kernel),
    )

    class P4CNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.lift = Lifting()
            self.conv1 = GroupA()
            self.conv2 = GroupB()
            self.norm1 = torch.nn.BatchNorm3d(width)
            self.norm2 = torch.nn.BatchNorm3d(width)
            self.norm3 = torch.nn.BatchNorm3d(2 * width)
            self.head = torch.nn.Linear(2 * width, num_classes)

        def feature_maps(self, x):
            """The spatial map after the group axis is pooled away.

            Still a function on the plane, so it *rotates* with the input --
            this is the equivariant quantity, and the one worth measuring.
            """
            h = torch.relu(self.norm1(self.lift(x)))
            h = torch.nn.functional.avg_pool3d(h, (1, 2, 2))
            h = torch.relu(self.norm2(self.conv1(h)))
            h = torch.nn.functional.avg_pool3d(h, (1, 2, 2))
            h = torch.relu(self.norm3(self.conv2(h)))
            return h.amax(dim=2)  # max over C_4: invariant to orientation

        def features(self, x):
            return self.feature_maps(x).mean(dim=(2, 3))  # global average pool

        def forward(self, x):
            return self.head(self.features(x))

    return P4CNN


def make_plain_cnn(width: int = 16, num_classes: int = 10, kernel: int = 3):
    """The control: the same shape without the group structure.

    Matched on *feature width* rather than parameter count, so the comparison
    is "same representation size, with and without the constraint". The
    parameter counts are reported either way, because the constraint is
    supposed to cost something.
    """
    torch = vision.require_torch()

    class PlainCNN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = torch.nn.Conv2d(1, width * 2, kernel, padding=kernel // 2)
            self.conv2 = torch.nn.Conv2d(width * 2, width * 2, kernel, padding=kernel // 2)
            self.conv3 = torch.nn.Conv2d(width * 2, 2 * width, kernel, padding=kernel // 2)
            self.norm1 = torch.nn.BatchNorm2d(width * 2)
            self.norm2 = torch.nn.BatchNorm2d(width * 2)
            self.norm3 = torch.nn.BatchNorm2d(2 * width)
            self.head = torch.nn.Linear(2 * width, num_classes)

        def feature_maps(self, x):
            h = torch.relu(self.norm1(self.conv1(x)))
            h = torch.nn.functional.avg_pool2d(h, 2)
            h = torch.relu(self.norm2(self.conv2(h)))
            h = torch.nn.functional.avg_pool2d(h, 2)
            return torch.relu(self.norm3(self.conv3(h)))

        def features(self, x):
            return self.feature_maps(x).mean(dim=(2, 3))

        def forward(self, x):
            return self.head(self.features(x))

    return PlainCNN


def invariance_error(model, images, degrees: float) -> float:
    r"""Relative change in the pooled representation under a rotation.

    $$\varepsilon(\theta) = \frac{\lVert h(R_\theta x) - h(x)\rVert}{\lVert h(x)\rVert},$$

    measured on the representation the classifier actually sees. For an exactly
    $C_4$-invariant network this is zero at $\theta \in \{0, 90, 180, 270\}$ up
    to floating point *and up to the resampling* -- ``rotate_batch``
    interpolates, so even a quarter turn is not bit-exact through the grid
    sampler. Reporting the number rather than asserting the theorem is the
    point: the theorem is about the operator, the number is about the program.
    """
    torch = vision.require_torch()
    model.eval()
    with torch.no_grad():
        base = model.features(images)
        rotated = model.features(rotate_batch(images, degrees))
        return float((rotated - base).norm() / base.norm().clamp_min(1e-12))


def equivariance_error(model, images, degrees: float) -> float:
    r"""How far the feature maps are from rotating with the input.

    $$\varepsilon_{\text{eq}}(\theta) = \frac{\lVert F(R_\theta x) - R_\theta F(x)\rVert}{\lVert F(x)\rVert}$$

    This, not the invariance of the pooled vector, is the quantity the
    architecture actually promises. **Measuring the pooled vector instead
    flatters every network**: a global average over spatial positions cannot
    distinguish a permutation of those positions, so a plain CNN ending in one
    scores 0.019 at a quarter turn -- not because its features are equivariant
    but because the pooling threw away the evidence. On the feature maps the
    same network scores an order of magnitude worse, which is the truth about
    it.

    The floor here is the resampler, not the network: ``rotate_batch``
    interpolates, and four quarter turns do not return an image to itself
    exactly (about 3% relative error), so no architecture can score zero.
    """
    torch = vision.require_torch()
    model.eval()
    with torch.no_grad():
        base = model.feature_maps(images)
        moved = rotate_batch(base, degrees)
        rotated = model.feature_maps(rotate_batch(images, degrees))
        return float((rotated - moved).norm() / base.norm().clamp_min(1e-12))


def rotated_copy(images, rng, *, discrete: bool = False):
    """A randomly rotated copy of a batch, for training augmentation.

    ``discrete`` restricts the angles to the group itself, which is the
    augmentation that *could* in principle teach exactly what the architecture
    enforces. The continuous version is the one practitioners actually use.
    """
    torch = vision.require_torch()
    out = torch.empty_like(images)
    if discrete:
        angles = 90.0 * rng.integers(0, ROTATIONS, size=len(images))
    else:
        angles = rng.uniform(0.0, 360.0, size=len(images))
    for index, angle in enumerate(angles):
        out[index : index + 1] = rotate_batch(images[index : index + 1], float(angle))
    return out


def train(model, images, labels, *, exp, device, augment: str | None, rng):
    """One arm's training loop. ``augment`` is None, "discrete" or "continuous"."""
    torch = vision.require_torch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=1e-4)
    losses: list[float] = []
    n = len(images)
    for _ in range(exp["epochs"]):
        model.train()
        order = torch.from_numpy(rng.permutation(n)).to(device)
        epoch: list[float] = []
        for start in range(0, n, exp["batch_size"]):
            index = order[start : start + exp["batch_size"]]
            batch, target = images[index], labels[index]
            if augment is not None:
                batch = rotated_copy(batch, rng, discrete=augment == "discrete")
            loss = torch.nn.functional.cross_entropy(model(batch), target)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch.append(loss.item())
        losses.append(float(np.mean(epoch)))
    return losses


def accuracy(model, images, labels, *, batch_size: int = 512) -> float:
    torch = vision.require_torch()
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            logits = model(images[start : start + batch_size])
            correct += int((logits.argmax(dim=1) == labels[start : start + batch_size]).sum())
    return correct / len(images)


def rotated_accuracy(model, images, labels, degrees: float, *, batch_size: int = 512) -> float:
    torch = vision.require_torch()
    model.eval()
    correct = 0
    with torch.no_grad():
        for start in range(0, len(images), batch_size):
            batch = rotate_batch(images[start : start + batch_size], degrees)
            correct += int((model(batch).argmax(dim=1) == labels[start : start + batch_size]).sum())
    return correct / len(images)


@experiment(
    "f1_equivariant_cnn",
    defaults={
        "train_samples": 8000,
        "test_samples": 2000,
        "width": 16,
        "kernel": 3,
        "epochs": 8,
        "batch_size": 128,
        "lr": 2e-3,
        # Angles the invariance curve is measured at. 0/90/180/270 are the
        # group; 45 and 135 are deliberately off it, where the guarantee stops.
        "angles": [0, 45, 90, 135, 180, 225, 270, 315],
        "eval_batch": 256,
        "arms": ["plain", "plain_augmented", "p4", "p4_augmented"],
    },
    description="A p4 group-equivariant CNN against a plain CNN and against rotation augmentation.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Rotation-equivariant CNN on a rotated-MNIST task, compared against a standard CNN"""
    torch = vision.require_torch()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)
    print(f"[f1] device: {device}")

    raw_train, y_train = vision.load("mnist", split="train", limit=exp["train_samples"], rng=rng)
    raw_test, y_test = vision.load("mnist", split="test", limit=exp["test_samples"], rng=rng)
    x_train = torch.from_numpy((raw_train.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)).to(device)
    x_test = torch.from_numpy((raw_test.astype(np.float32) / 255.0).transpose(0, 3, 1, 2)).to(device)
    y_train_t = torch.from_numpy(y_train).to(device)
    y_test_t = torch.from_numpy(y_test).to(device)
    print(f"[f1] {len(x_train)} train / {len(x_test)} test, upright MNIST; the test set is rotated at evaluation")

    summary: dict[str, float] = {}
    curves: dict[str, list[float]] = {}
    invariance: dict[str, list[float]] = {}
    equivariance: dict[str, list[float]] = {}
    accuracies: dict[str, list[float]] = {}
    probe = x_test[: exp["eval_batch"]]

    for arm in exp["arms"]:
        torchkit.seed_torch(exp.seed)
        equivariant = arm.startswith("p4")
        factory = make_p4_cnn if equivariant else make_plain_cnn
        model = factory(width=exp["width"], kernel=exp["kernel"])().to(device)
        augment = "continuous" if arm.endswith("augmented") else None
        parameters = torchkit.count_parameters(model)
        summary[f"{arm}/parameters"] = float(parameters)

        curves[arm] = train(
            model, x_train, y_train_t, exp=exp, device=device, augment=augment, rng=np.random.default_rng(exp.seed)
        )
        summary[f"{arm}/train_loss"] = curves[arm][-1]
        summary[f"{arm}/accuracy_upright"] = accuracy(model, x_test, y_test_t)

        # Both quantities, because the difference between them is itself a
        # result: the pooled one is flattered by the global average.
        equivariance[arm] = [equivariance_error(model, probe, float(a)) for a in exp["angles"]]
        invariance[arm] = [invariance_error(model, probe, float(a)) for a in exp["angles"]]
        accuracies[arm] = [rotated_accuracy(model, x_test, y_test_t, float(a)) for a in exp["angles"]]
        for angle, error, acc, eq in zip(
            exp["angles"], invariance[arm], accuracies[arm], equivariance[arm], strict=True
        ):
            summary[f"{arm}/invariance/{angle}"] = error
            summary[f"{arm}/equivariance/{angle}"] = eq
            summary[f"{arm}/accuracy/{angle}"] = acc

        group_angles = [i for i, a in enumerate(exp["angles"]) if a % 90 == 0]
        off_angles = [i for i, a in enumerate(exp["angles"]) if a % 90 != 0]
        summary[f"{arm}/invariance/on_group"] = float(np.mean([invariance[arm][i] for i in group_angles]))
        summary[f"{arm}/invariance/off_group"] = float(np.mean([invariance[arm][i] for i in off_angles]))
        summary[f"{arm}/equivariance/on_group"] = float(np.mean([equivariance[arm][i] for i in group_angles]))
        summary[f"{arm}/equivariance/off_group"] = float(np.mean([equivariance[arm][i] for i in off_angles]))
        # How much the global average pool understates the difference.
        summary[f"{arm}/pooling_flattery"] = summary[f"{arm}/equivariance/on_group"] / max(
            summary[f"{arm}/invariance/on_group"], 1e-12
        )
        summary[f"{arm}/accuracy/on_group"] = float(np.mean([accuracies[arm][i] for i in group_angles]))
        summary[f"{arm}/accuracy/off_group"] = float(np.mean([accuracies[arm][i] for i in off_angles]))
        summary[f"{arm}/accuracy/worst_angle"] = float(min(accuracies[arm]))
        print(
            f"[f1] {arm:<16} {parameters:>7,} params  upright {summary[f'{arm}/accuracy_upright']:.3f}  "
            f"on-group acc {summary[f'{arm}/accuracy/on_group']:.3f} (equiv err {summary[f'{arm}/equivariance/on_group']:.2e})  "
            f"off-group acc {summary[f'{arm}/accuracy/off_group']:.3f} (equiv err {summary[f'{arm}/equivariance/off_group']:.2e})"
        )

    # -- the comparison the rung exists to make -------------------------------
    # Augmentation and equivariance both aim at invariance. The difference is
    # where it holds: one on the training distribution, the other on the group.
    if "p4" in invariance and "plain_augmented" in invariance:
        summary["exactness/p4_over_augmented"] = summary["plain_augmented/equivariance/on_group"] / max(
            summary["p4/equivariance/on_group"], 1e-12
        )
        print(
            f"[f1] on the group, p4 is {summary['exactness/p4_over_augmented']:.0f}x more invariant than the "
            f"augmented plain CNN -- and augmentation saw those rotations in training while p4 never did"
        )
    if "p4" in accuracies and "plain" in accuracies:
        summary["generalization/p4_over_plain_on_group"] = (
            summary["p4/accuracy/on_group"] - summary["plain/accuracy/on_group"]
        )
        summary["generalization/p4_over_plain_off_group"] = (
            summary["p4/accuracy/off_group"] - summary["plain/accuracy/off_group"]
        )
    if "p4" in summary_keys(summary, "parameters") and "plain" in summary_keys(summary, "parameters"):
        summary["parameters/p4_over_plain"] = summary["p4/parameters"] / summary["plain/parameters"]

    # -- figures --------------------------------------------------------------
    exp.log_figure(
        "equivariance",
        plotting.plot_curves(
            equivariance,
            x=exp["angles"],
            title="Equivariance error of the feature maps (the floor is the resampler)",
            xlabel="rotation (degrees)",
            ylabel="relative error",
            logy=True,
            marker="o",
        ),
    )
    exp.log_figure(
        "invariance_pooled",
        plotting.plot_curves(
            invariance,
            x=exp["angles"],
            title="Invariance of the pooled vector -- flattered by the global average",
            xlabel="rotation (degrees)",
            ylabel="relative error",
            logy=True,
            marker="o",
        ),
    )
    exp.log_figure(
        "accuracy_by_angle",
        plotting.plot_curves(
            accuracies,
            x=exp["angles"],
            title="Test accuracy against rotation, trained on upright digits",
            xlabel="rotation (degrees)",
            ylabel="accuracy",
            marker="o",
        ),
    )
    exp.log_figure(
        "training",
        plotting.plot_curves(curves, title="Training loss", xlabel="epoch", ylabel="cross-entropy"),
    )
    exp.log_figure(
        "parameters",
        plotting.plot_comparison(
            {arm: summary[f"{arm}/parameters"] for arm in exp["arms"]},
            title="Parameters",
            ylabel="count",
        ),
    )
    return summary


def summary_keys(summary: dict[str, float], suffix: str) -> set[str]:
    """Arm names that have recorded ``suffix``, so derived metrics stay guarded."""
    return {key.rsplit("/", 1)[0] for key in summary if key.endswith("/" + suffix)}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
