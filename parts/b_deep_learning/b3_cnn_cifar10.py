"""B3. CNN on CIFAR-10 with a batchnorm/dropout ablation

Lab reference: [karpathy/micrograd — autodiff reference implementation](https://github.com/karpathy/micrograd)

Batch normalization and dropout are both sold as "regularizers", but they do
different jobs: BN rescales activations so the loss surface is better
conditioned (and only incidentally regularizes), while dropout injects noise
that directly attacks overfitting. A 2x2 ablation separates them, and a
learning-rate probe tests the claim BN is actually famous for — that it lets
you train at step sizes a plain network cannot survive.

Sized for CPU: a subset of CIFAR-10 and a four-conv network. The point is the
*comparison*, which needs the four arms to share data, schedule and seed, not a
state-of-the-art number.

Run with ``ladder run B3`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np

from mlladder import Experiment, experiment, metrics, plotting, torchkit, vision


def build_cnn(*, use_batchnorm: bool, dropout: float, channels: tuple[int, int] = (32, 64), classes: int = 10):
    """Two VGG-style blocks, each ``conv-conv-pool``, then a classifier head.

    Bias is dropped from convolutions that feed a BN layer: BN's shift
    parameter subsumes it, so the bias would be an unused degree of freedom.
    """
    vision.require_torch()
    from torch import nn

    def block(in_channels: int, out_channels: int) -> list[nn.Module]:
        layers: list[nn.Module] = []
        for index in range(2):
            source = in_channels if index == 0 else out_channels
            layers.append(nn.Conv2d(source, out_channels, kernel_size=3, padding=1, bias=not use_batchnorm))
            if use_batchnorm:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
        layers.append(nn.MaxPool2d(2))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        return layers

    first, second = channels
    head: list[nn.Module] = [nn.Flatten(), nn.Linear(second * 8 * 8, 128), nn.ReLU(inplace=True)]
    if dropout > 0:
        head.append(nn.Dropout(dropout))
    head.append(nn.Linear(128, classes))
    return nn.Sequential(*block(3, first), *block(first, second), *head)


ARMS = (
    ("plain", False, 0.0),
    ("batchnorm", True, 0.0),
    ("dropout", False, 0.25),
    ("both", True, 0.25),
)


@experiment(
    "b3_cnn_cifar10",
    defaults={
        "train_samples": 5000,
        "test_samples": 2000,
        "epochs": 8,
        "lr": 0.05,
        "batch_size": 128,
        "dropout": 0.25,
        "probe_lr": 0.5,
        "probe_epochs": 4,
    },
    description="A 2x2 batchnorm/dropout ablation on a CIFAR-10 subset, plus a high-learning-rate probe.",
)
def run(exp: Experiment) -> dict[str, float]:
    """CNN on CIFAR-10 with a batchnorm/dropout ablation."""
    torch = vision.require_torch()
    rng = np.random.default_rng(exp.seed)

    raw_train, y_train = vision.load("cifar10", split="train", limit=exp["train_samples"], rng=rng)
    raw_test, y_test = vision.load("cifar10", split="test", limit=exp["test_samples"], rng=rng)
    x_train = vision.normalize(raw_train, "cifar10")
    x_test = vision.normalize(raw_test, "cifar10")
    print(f"[b3] {len(x_train)} train / {len(x_test)} test images of shape {x_train.shape[1:]}")

    histories: dict[str, dict[str, list[float]]] = {}
    summary: dict[str, float] = {}
    best_arm, best_accuracy, best_model = "", 0.0, None

    for name, use_batchnorm, dropout_rate in ARMS:
        torchkit.seed_torch(exp.seed)
        dropout = exp["dropout"] if dropout_rate > 0 else 0.0
        model = build_cnn(use_batchnorm=use_batchnorm, dropout=dropout)
        train_loader, test_loader = torchkit.loaders(
            x_train, y_train, x_test, y_test, batch_size=exp["batch_size"], seed=exp.seed
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=exp["lr"], momentum=0.9)

        history = torchkit.fit(model, train_loader, test_loader, optimizer, epochs=exp["epochs"])
        histories[name] = history

        # Train loss is measured while dropout is active, so compare like with
        # like: re-evaluate the training set in eval mode for the gap.
        _, train_accuracy = torchkit.evaluate(model, torchkit.eval_loader(x_train, y_train))
        final_accuracy = history["test_accuracy"][-1]
        summary |= {
            f"{name}/test_accuracy": final_accuracy,
            f"{name}/best_test_accuracy": max(history["test_accuracy"]),
            f"{name}/train_accuracy": train_accuracy,
            f"{name}/generalization_gap": train_accuracy - final_accuracy,
            f"{name}/seconds_per_epoch": float(np.mean(history["seconds"])),
        }
        print(
            f"[b3] {name:<10} test={final_accuracy:.4f} (best {max(history['test_accuracy']):.4f})  "
            f"train={train_accuracy:.4f}  gap={train_accuracy - final_accuracy:+.4f}  "
            f"{np.mean(history['seconds']):.1f}s/epoch"
        )
        exp.log({f"{name}/{key}": value[-1] for key, value in history.items()})
        if final_accuracy > best_accuracy:
            best_arm, best_accuracy, best_model = name, final_accuracy, model

    # -- what BN is actually for: surviving a large step size ---------------
    probe: dict[str, list[float]] = {}
    for name, use_batchnorm in (("plain", False), ("batchnorm", True)):
        torchkit.seed_torch(exp.seed)
        model = build_cnn(use_batchnorm=use_batchnorm, dropout=0.0)
        train_loader, test_loader = torchkit.loaders(
            x_train, y_train, x_test, y_test, batch_size=exp["batch_size"], seed=exp.seed
        )
        optimizer = torch.optim.SGD(model.parameters(), lr=exp["probe_lr"], momentum=0.9)
        history = torchkit.fit(model, train_loader, test_loader, optimizer, epochs=exp["probe_epochs"])
        probe[f"{name} @ lr={exp['probe_lr']}"] = history["test_accuracy"]
        summary[f"probe/{name}_test_accuracy"] = history["test_accuracy"][-1]
        diverged = not np.isfinite(history["train_loss"][-1]) or history["test_accuracy"][-1] < 0.15
        print(
            f"[b3] probe lr={exp['probe_lr']} {name:<10} accuracy={history['test_accuracy'][-1]:.4f}{'  (collapsed)' if diverged else ''}"
        )

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "ablation_accuracy",
        plotting.plot_curves(
            {name: history["test_accuracy"] for name, history in histories.items()},
            x=list(range(1, exp["epochs"] + 1)),
            title="CIFAR-10 subset — test accuracy by ablation arm",
            xlabel="epoch",
            ylabel="test accuracy",
            marker="o",
        ),
    )
    exp.log_figure(
        "ablation_loss",
        plotting.plot_curves(
            {f"{name} (train)": history["train_loss"] for name, history in histories.items()},
            x=list(range(1, exp["epochs"] + 1)),
            title="Train loss by ablation arm",
            xlabel="epoch",
            ylabel="cross-entropy",
        ),
    )
    exp.log_figure(
        "generalization_gap",
        plotting.plot_comparison(
            {name: summary[f"{name}/generalization_gap"] for name, _, _ in ARMS},
            title="Train accuracy minus test accuracy",
            ylabel="gap",
        ),
    )
    exp.log_figure(
        "cost_per_epoch",
        plotting.plot_comparison(
            {name: summary[f"{name}/seconds_per_epoch"] for name, _, _ in ARMS},
            title="Seconds per epoch (CPU)",
            ylabel="seconds",
        ),
    )
    exp.log_figure(
        "lr_probe",
        plotting.plot_curves(
            probe,
            x=list(range(1, exp["probe_epochs"] + 1)),
            title=f"Learning rate {exp['probe_lr']} — with and without batchnorm",
            xlabel="epoch",
            ylabel="test accuracy",
            marker="o",
        ),
    )
    if best_model is not None:
        _, test_loader = torchkit.loaders(x_train, y_train, x_test, y_test, batch_size=exp["batch_size"], seed=exp.seed)
        predicted, actual = torchkit.predictions(best_model, test_loader)
        exp.log_figure(
            "confusion",
            plotting.plot_confusion_matrix(
                metrics.confusion_matrix(actual, predicted, num_classes=10),
                labels=["plane", "car", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"],
                title=f"Best arm ({best_arm}) — test confusion",
                normalize=True,
            ),
        )

    summary["best_arm_accuracy"] = best_accuracy
    summary["batchnorm_minus_plain"] = summary["batchnorm/test_accuracy"] - summary["plain/test_accuracy"]
    summary["dropout_minus_plain"] = summary["dropout/test_accuracy"] - summary["plain/test_accuracy"]
    summary["probe/batchnorm_minus_plain"] = (
        summary["probe/batchnorm_test_accuracy"] - summary["probe/plain_test_accuracy"]
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
