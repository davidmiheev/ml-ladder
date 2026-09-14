"""B4. ResNet block implementation, trained on CIFAR-10/100

Lab reference: [karpathy/micrograd — autodiff reference implementation](https://github.com/karpathy/micrograd)

The residual block's claim is not that it is a better feature extractor, but
that it removes a *failure*: stacking more plain layers makes a network harder
to optimize, so a deeper plain net can end up worse than a shallower one on the
training set — degradation, not overfitting. The fix is to let each block learn
a residual around an identity path.

So this rung trains the same architecture at increasing depth with the skip
connections switched on and off, and measures the mechanism directly: the
gradient norm reaching the earliest layers.

Run with ``ladder run B4`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np

from mlladder import Experiment, experiment, plotting, torchkit, vision


def make_block_class():
    """Define the block against the torch imported at call time."""
    vision.require_torch()
    from torch import nn

    class ResidualBlock(nn.Module):
        """``y = ReLU(F(x) + shortcut(x))`` with ``F`` two 3x3 convolutions.

        With ``residual=False`` the same stack runs without the skip, which is
        the plain baseline the ResNet paper compares against.
        """

        def __init__(self, in_channels: int, out_channels: int, *, stride: int = 1, residual: bool = True) -> None:
            super().__init__()
            self.residual = residual
            self.body = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
                nn.BatchNorm2d(out_channels),
            )
            # A projection shortcut only where the identity cannot be used:
            # changed shape. Everywhere else the skip is literally identity.
            needs_projection = stride != 1 or in_channels != out_channels
            self.shortcut = (
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels, 1, stride=stride, bias=False), nn.BatchNorm2d(out_channels)
                )
                if needs_projection
                else nn.Identity()
            )
            self.relu = nn.ReLU(inplace=True)

        def forward(self, x):
            out = self.body(x)
            if self.residual:
                out = out + self.shortcut(x)
            return self.relu(out)

    return ResidualBlock


def build_network(depth: int, *, residual: bool, classes: int = 10, width: int = 16):
    """``depth`` blocks in three stages, CIFAR-ResNet style, then global pooling."""
    vision.require_torch()
    from torch import nn

    ResidualBlock = make_block_class()
    stages = [(width, 1), (width * 2, 2), (width * 4, 2)]

    layers: list[nn.Module] = [
        nn.Conv2d(3, width, 3, padding=1, bias=False),
        nn.BatchNorm2d(width),
        nn.ReLU(inplace=True),
    ]
    in_channels = width
    for index in range(depth):
        out_channels, stride = stages[min(index * len(stages) // max(depth, 1), len(stages) - 1)]
        layers.append(
            ResidualBlock(
                in_channels, out_channels, stride=stride if in_channels != out_channels else 1, residual=residual
            )
        )
        in_channels = out_channels
    layers += [nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Linear(in_channels, classes)]
    return nn.Sequential(*layers)


def first_layer_gradient_norms(model, loader) -> list[float]:
    """Gradient norm per parameter tensor after one backward pass.

    The residual identity path means $\\partial L/\\partial x_\\ell$ keeps a term that
    does not pass through any weight matrix, so early layers should see larger
    gradients than in the plain stack.
    """
    torch = vision.require_torch()
    features, labels = next(iter(loader))
    model.zero_grad(set_to_none=True)
    torch.nn.functional.cross_entropy(model(features), labels).backward()
    return [float(parameter.grad.norm()) for _, parameter in model.named_parameters() if parameter.grad is not None]


@experiment(
    "b4_resnet_blocks",
    defaults={
        "train_samples": 6000,
        "test_samples": 2000,
        "depths": [2, 4, 8],
        "epochs": 8,
        "lr": 0.05,
        "batch_size": 128,
        "width": 16,
        "cifar100_depth": 4,
        "cifar100_epochs": 8,
    },
    description="Plain vs residual networks at increasing depth on CIFAR-10, plus gradient flow and a CIFAR-100 run.",
)
def run(exp: Experiment) -> dict[str, float]:
    """ResNet block implementation, trained on CIFAR-10/100."""
    torch = vision.require_torch()
    rng = np.random.default_rng(exp.seed)

    raw_train, y_train = vision.load("cifar10", split="train", limit=exp["train_samples"], rng=rng)
    raw_test, y_test = vision.load("cifar10", split="test", limit=exp["test_samples"], rng=rng)
    x_train = vision.normalize(raw_train, "cifar10")
    x_test = vision.normalize(raw_test, "cifar10")
    print(f"[b4] CIFAR-10: {len(x_train)} train / {len(x_test)} test")

    curves: dict[str, list[float]] = {}
    summary: dict[str, float] = {}
    train_accuracy_by_arm: dict[str, float] = {}

    for depth in exp["depths"]:
        for residual in (False, True):
            arm = f"{'residual' if residual else 'plain'}-{depth}"
            torchkit.seed_torch(exp.seed)
            model = build_network(depth, residual=residual, width=exp["width"])
            train_loader, test_loader = torchkit.loaders(
                x_train, y_train, x_test, y_test, batch_size=exp["batch_size"], seed=exp.seed
            )
            optimizer = torch.optim.SGD(model.parameters(), lr=exp["lr"], momentum=0.9, weight_decay=5e-4)

            history = torchkit.fit(model, train_loader, test_loader, optimizer, epochs=exp["epochs"])
            _, train_accuracy = torchkit.evaluate(model, torchkit.eval_loader(x_train, y_train))
            curves[arm] = history["test_accuracy"]
            train_accuracy_by_arm[arm] = train_accuracy

            summary |= {
                f"{arm}/test_accuracy": history["test_accuracy"][-1],
                f"{arm}/train_accuracy": train_accuracy,
                f"{arm}/parameters": float(torchkit.count_parameters(model)),
                f"{arm}/seconds_per_epoch": float(np.mean(history["seconds"])),
            }
            print(
                f"[b4] {arm:<14} params={torchkit.count_parameters(model):>7,}  "
                f"train={train_accuracy:.4f}  test={history['test_accuracy'][-1]:.4f}  "
                f"{np.mean(history['seconds']):.1f}s/epoch"
            )
            exp.log({f"{arm}/test_accuracy": history["test_accuracy"][-1], "depth": depth})

    # -- the mechanism: gradient reaching the early layers ------------------
    deepest = max(exp["depths"])
    gradient_curves: dict[str, list[float]] = {}
    for residual in (False, True):
        torchkit.seed_torch(exp.seed)
        model = build_network(deepest, residual=residual, width=exp["width"])
        train_loader, _ = torchkit.loaders(
            x_train, y_train, x_test, y_test, batch_size=exp["batch_size"], seed=exp.seed
        )
        norms = first_layer_gradient_norms(model, train_loader)
        name = "residual" if residual else "plain"
        gradient_curves[name] = norms
        summary[f"gradient/{name}_first_tensor"] = norms[0]
        summary[f"gradient/{name}_ratio_first_to_last"] = norms[0] / max(norms[-1], 1e-12)
        print(
            f"[b4] depth-{deepest} {name:<8} first-tensor grad norm={norms[0]:.3e}  first/last ratio={norms[0] / max(norms[-1], 1e-12):.3f}"
        )

    # -- the same block on a harder label set -------------------------------
    raw_train_100, y_train_100 = vision.load("cifar100", split="train", limit=exp["train_samples"], rng=rng)
    raw_test_100, y_test_100 = vision.load("cifar100", split="test", limit=exp["test_samples"], rng=rng)
    torchkit.seed_torch(exp.seed)
    model_100 = build_network(exp["cifar100_depth"], residual=True, classes=100, width=exp["width"])
    train_loader, test_loader = torchkit.loaders(
        vision.normalize(raw_train_100, "cifar100"),
        y_train_100,
        vision.normalize(raw_test_100, "cifar100"),
        y_test_100,
        batch_size=exp["batch_size"],
        seed=exp.seed,
    )
    optimizer = torch.optim.SGD(model_100.parameters(), lr=exp["lr"], momentum=0.9, weight_decay=5e-4)
    history_100 = torchkit.fit(model_100, train_loader, test_loader, optimizer, epochs=exp["cifar100_epochs"])
    chance = 1 / 100
    print(
        f"[b4] CIFAR-100 residual-{exp['cifar100_depth']}: test={history_100['test_accuracy'][-1]:.4f} "
        f"({history_100['test_accuracy'][-1] / chance:.0f}x chance)"
    )

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "depth_curves",
        plotting.plot_curves(
            curves,
            x=list(range(1, exp["epochs"] + 1)),
            title="CIFAR-10 subset — test accuracy by depth and skip connection",
            xlabel="epoch",
            ylabel="test accuracy",
        ),
    )
    exp.log_figure(
        "depth_summary",
        plotting.plot_comparison(
            {arm: summary[f"{arm}/test_accuracy"] for arm in curves},
            title="Final test accuracy — plain vs residual",
            ylabel="test accuracy",
        ),
    )
    exp.log_figure(
        "train_accuracy",
        plotting.plot_comparison(
            train_accuracy_by_arm,
            title="Train accuracy — degradation shows here first",
            ylabel="train accuracy",
        ),
    )
    exp.log_figure(
        "gradient_flow",
        plotting.plot_curves(
            gradient_curves,
            title=f"Gradient norm per parameter tensor (depth {deepest}, one batch)",
            xlabel="parameter tensor (input side to output side)",
            ylabel="gradient norm",
            logy=True,
        ),
    )
    exp.log_figure(
        "cifar100",
        plotting.plot_curves(
            {"CIFAR-100 test accuracy": history_100["test_accuracy"]},
            x=list(range(1, exp["cifar100_epochs"] + 1)),
            title=f"Residual-{exp['cifar100_depth']} on CIFAR-100 (chance = 0.01)",
            xlabel="epoch",
            ylabel="test accuracy",
            marker="o",
        ),
    )

    for depth in exp["depths"]:
        summary[f"residual_minus_plain/depth{depth}"] = (
            summary[f"residual-{depth}/test_accuracy"] - summary[f"plain-{depth}/test_accuracy"]
        )
    summary["cifar100/test_accuracy"] = history_100["test_accuracy"][-1]
    summary["cifar100/times_chance"] = history_100["test_accuracy"][-1] * 100
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
