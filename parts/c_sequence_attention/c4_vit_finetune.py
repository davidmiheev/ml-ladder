"""C4. Fine-tune a small ViT on a custom image-classification task

Lab reference: [Annotated Transformer / nanoGPT](https://nlp.seas.harvard.edu/annotated-transformer/)

C3 built a transformer for text; a ViT is the same block applied to 16x16 image
patches, so this rung is about what *pretraining* buys rather than what the
architecture is. Three arms on the same small task:

1. **from scratch** — the ViT-Tiny architecture, random init;
2. **linear probe** — frozen pretrained features, train only a linear head;
3. **full fine-tune** — the same pretrained weights, everything trainable.

A transformer has almost no image prior built in (unlike a CNN's locality and
translation equivariance), so on a few thousand images the from-scratch arm
should be bad and the gap to the probe should be large. That gap is the whole
argument for pretraining, and it is one number.

Run with ``ladder run C4`` (``--help`` for flags). CPU-sized by default;
``--device cuda`` and larger budgets if you have a GPU.
"""

from __future__ import annotations

import time

import numpy as np

from mlladder import Experiment, experiment, metrics, plotting, torchkit, vision

# A four-class subset makes this a "custom task" rather than a benchmark, and
# keeps the CPU budget honest.
TASK_CLASSES = (0, 1, 8, 9)  # plane, car, ship, truck — two natural pairs
TASK_NAMES = ("plane", "car", "ship", "truck")


def build_vit(*, pretrained: bool, num_classes: int, frozen: bool, model_name: str = "vit_tiny_patch16_224"):
    """A timm ViT with a fresh head, optionally with the backbone frozen."""
    vision.require_torch()
    import timm

    model = timm.create_model(model_name, pretrained=pretrained, num_classes=num_classes)
    if frozen:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("head")
    return model


def resize_batch(images, size: int):
    """Upsample CIFAR's 32x32 to the 224x224 the pretrained patches expect."""
    torch = vision.require_torch()
    return torch.nn.functional.interpolate(images, size=(size, size), mode="bilinear", align_corners=False)


def subset_task(images, labels, classes=TASK_CLASSES):
    """Keep only the chosen classes and relabel them 0..k-1."""
    mask = np.isin(labels, classes)
    remap = {original: index for index, original in enumerate(classes)}
    return images[mask], np.array([remap[int(label)] for label in labels[mask]], dtype=np.int64)


@experiment(
    "c4_vit_finetune",
    defaults={
        # CPU-sized: a full fine-tune of this ViT costs ~4 s/image-batch on four
        # cores. `--train-samples 8000 --epochs 5` is the GPU configuration.
        "train_samples": 1500,
        "test_samples": 1000,
        "image_size": 224,
        "epochs": 2,
        "batch_size": 32,
        "lr_head": 1e-3,
        "lr_finetune": 3e-5,
        "lr_scratch": 3e-4,
        "device": "auto",
        "arms": ["scratch", "probe", "finetune"],
    },
    description="ViT-Tiny on a 4-class CIFAR task: from scratch vs linear probe vs full fine-tune.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Fine-tune a small ViT on a custom image-classification task."""
    torch = vision.require_torch()
    device = torchkit.device() if exp["device"] == "auto" else torch.device(exp["device"])
    rng = np.random.default_rng(exp.seed)
    print(f"[c4] device: {device}")

    raw_train, y_train_all = vision.load("cifar10", split="train", limit=exp["train_samples"] * 3, rng=rng)
    raw_test, y_test_all = vision.load("cifar10", split="test", limit=exp["test_samples"] * 3, rng=rng)
    x_train, y_train = subset_task(vision.normalize(raw_train, "cifar10"), y_train_all)
    x_test, y_test = subset_task(vision.normalize(raw_test, "cifar10"), y_test_all)
    x_train, y_train = x_train[: exp["train_samples"]], y_train[: exp["train_samples"]]
    x_test, y_test = x_test[: exp["test_samples"]], y_test[: exp["test_samples"]]
    print(f"[c4] task {TASK_NAMES}: {len(x_train)} train / {len(x_test)} test")

    def run_epochs(model, *, lr: float, epochs: int, arm: str) -> tuple[list[float], float, float]:
        model = model.to(device)
        trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=0.01)
        train_loader, test_loader = torchkit.loaders(
            x_train, y_train, x_test, y_test, batch_size=exp["batch_size"], seed=exp.seed
        )

        accuracies = []
        started = time.perf_counter()
        for epoch in range(epochs):
            model.train()
            for images, labels in train_loader:
                images = resize_batch(images.to(device), exp["image_size"])
                loss = torch.nn.functional.cross_entropy(model(images), labels.to(device))
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

            model.eval()
            correct = total = 0
            with torch.no_grad():
                for images, labels in test_loader:
                    images = resize_batch(images.to(device), exp["image_size"])
                    correct += int((model(images).argmax(dim=1).cpu() == labels).sum())
                    total += len(labels)
            accuracies.append(correct / total)
            print(f"[c4] {arm:<9} epoch {epoch + 1}/{epochs}  test accuracy={accuracies[-1]:.4f}")
            exp.log({f"{arm}/test_accuracy": accuracies[-1]}, step=epoch)
        return accuracies, time.perf_counter() - started, float(sum(p.numel() for p in trainable))

    configurations = {
        "scratch": {"pretrained": False, "frozen": False, "lr": exp["lr_scratch"]},
        "probe": {"pretrained": True, "frozen": True, "lr": exp["lr_head"]},
        "finetune": {"pretrained": True, "frozen": False, "lr": exp["lr_finetune"]},
    }

    curves: dict[str, list[float]] = {}
    summary: dict[str, float] = {}
    best_model, best_accuracy, best_arm = None, 0.0, ""
    for arm in exp["arms"]:
        settings = configurations[arm]
        torchkit.seed_torch(exp.seed)
        model = build_vit(pretrained=settings["pretrained"], num_classes=len(TASK_CLASSES), frozen=settings["frozen"])
        accuracies, seconds, trainable = run_epochs(model, lr=settings["lr"], epochs=exp["epochs"], arm=arm)

        curves[arm] = accuracies
        summary |= {
            f"{arm}/test_accuracy": accuracies[-1],
            f"{arm}/best_test_accuracy": max(accuracies),
            f"{arm}/trainable_parameters": trainable,
            f"{arm}/seconds": seconds,
        }
        print(f"[c4] {arm:<9} final={accuracies[-1]:.4f}  trainable={trainable / 1e6:.3f}M  {seconds:.0f}s")
        if accuracies[-1] > best_accuracy:
            best_model, best_accuracy, best_arm = model, accuracies[-1], arm

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "arms",
        plotting.plot_curves(
            curves,
            x=list(range(1, exp["epochs"] + 1)),
            title=f"ViT-Tiny on {'/'.join(TASK_NAMES)} — test accuracy",
            xlabel="epoch",
            ylabel="test accuracy",
            marker="o",
        ),
    )
    exp.log_figure(
        "final_accuracy",
        plotting.plot_comparison(
            {arm: summary[f"{arm}/test_accuracy"] for arm in curves},
            title="Pretraining is the whole story",
            ylabel="test accuracy",
        ),
    )
    exp.log_figure(
        "trainable_parameters",
        plotting.plot_comparison(
            {arm: summary[f"{arm}/trainable_parameters"] / 1e6 for arm in curves},
            title="Trainable parameters (millions)",
            ylabel="millions",
        ),
    )
    if best_model is not None:
        best_model.eval()
        _, test_loader = torchkit.loaders(x_train, y_train, x_test, y_test, batch_size=exp["batch_size"], seed=exp.seed)
        predicted, actual = [], []
        with torch.no_grad():
            for images, labels in test_loader:
                predicted.append(
                    best_model(resize_batch(images.to(device), exp["image_size"])).argmax(dim=1).cpu().numpy()
                )
                actual.append(labels.numpy())
        predicted, actual = np.concatenate(predicted), np.concatenate(actual)
        exp.log_figure(
            "confusion",
            plotting.plot_confusion_matrix(
                metrics.confusion_matrix(actual, predicted, num_classes=len(TASK_CLASSES)),
                labels=list(TASK_NAMES),
                title=f"Best arm ({best_arm}) — test confusion",
                normalize=True,
            ),
        )

    if "probe" in curves and "scratch" in curves:
        summary["probe_minus_scratch"] = summary["probe/test_accuracy"] - summary["scratch/test_accuracy"]
    if "finetune" in curves and "probe" in curves:
        summary["finetune_minus_probe"] = summary["finetune/test_accuracy"] - summary["probe/test_accuracy"]
    summary["best_accuracy"] = best_accuracy
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
