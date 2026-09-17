"""D3. Linear probing of the learned representations against a supervised baseline

Lab reference: [lightly-ai — contrastive-learning reference implementations](https://github.com/lightly-ai/lightly)

D1 and D2 each produced a representation and scored it with a probe at one label
budget. That answers "did it learn something" but not the question self-supervised
learning is actually sold on: **when you have plenty of images and few labels, is
pretraining worth more than the labels you would have spent?**

So this rung sweeps the label budget. Every method sees the same unlabelled
images and the same labelled subsets, and is read two ways — a linear probe
(what a classifier can extract) and k-nearest-neighbours (what the metric
structure already encodes, with nothing trained at all).

Run with ``ladder run D3`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np

from mlladder import Experiment, experiment, metrics, plotting, torchkit, vision
from parts.d_self_supervised.d1_simclr_pretraining import (
    linear_probe,
    make_encoder,
    make_projection_head,
    pretrain,
)
from parts.d_self_supervised.d2_masked_autoencoder import make_mae, patchify, random_mask


def knn_accuracy(features_train, y_train, features_test, y_test, *, k: int = 20):
    r"""Cosine k-NN on frozen features — the no-training-at-all read.

    A linear probe can rescue a representation whose classes are separable but
    badly scaled; k-NN cannot, so the two together say whether the *geometry*
    is right or only the linear separability.
    """
    torch = vision.require_torch()
    train = torch.nn.functional.normalize(features_train, dim=1)
    test = torch.nn.functional.normalize(features_test, dim=1)
    labels = torch.as_tensor(y_train, device=train.device)

    predictions = []
    for chunk in test.split(256):
        similarity = chunk @ train.T
        neighbours = similarity.topk(min(k, len(train)), dim=1).indices
        votes = labels[neighbours]
        predictions.append(torch.mode(votes, dim=1).values)
    return metrics.accuracy(y_test, torch.cat(predictions).cpu().numpy())


def stratified_subset(labels, per_class: int, rng: np.random.Generator):
    """Indices with exactly ``per_class`` examples of each class, where available."""
    chosen = []
    for label in np.unique(labels):
        candidates = np.flatnonzero(labels == label)
        chosen.append(rng.permutation(candidates)[:per_class])
    return np.concatenate(chosen)


def supervised_encoder(
    x_train, y_train, *, device, depth: int, width: int, steps: int, batch_size: int, lr: float, seed: int
):
    """Train the same architecture with labels — the baseline SSL must beat."""
    torch = vision.require_torch()
    from torch import nn

    torchkit.seed_torch(seed)
    encoder, feature_dim = make_encoder(depth=depth, width=width)
    encoder = encoder.to(device)
    head = nn.Linear(feature_dim, int(y_train.max()) + 1).to(device)
    optimizer = torch.optim.AdamW([*encoder.parameters(), *head.parameters()], lr=lr, weight_decay=1e-4)

    rng = np.random.default_rng(seed)
    images = torch.from_numpy(x_train)
    labels = torch.from_numpy(y_train)
    for _ in range(steps):
        index = rng.integers(0, len(images), size=min(batch_size, len(images)))
        batch, target = images[index].to(device), labels[index].to(device)
        encoder.train()
        loss = torch.nn.functional.cross_entropy(head(encoder(batch)), target)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return encoder


@experiment(
    "d3_linear_probing",
    defaults={
        "pretrain_samples": 8000,
        "label_pool": 5000,
        "test_samples": 2000,
        "labels_per_class": [5, 25, 100, 400],
        "depth": 4,
        "width": 32,
        "steps": 600,
        "batch_size": 128,
        "lr": 1e-3,
        "temperature": 0.5,
        "strength": 1.0,
        "mask_ratio": 0.75,
        "patch": 4,
        "mae_width": 128,
        "mae_depth": 4,
        "probe_epochs": 15,
        "knn_k": 20,
    },
    description="Label-efficiency of SimCLR, MAE, supervised and random features under linear and k-NN probes.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Linear probing of the learned representations against a supervised baseline."""
    torch = vision.require_torch()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)
    print(f"[d3] device: {device}")

    raw_pretrain, _ = vision.load("cifar10", split="train", limit=exp["pretrain_samples"], rng=rng)
    raw_pool, y_pool = vision.load("cifar10", split="train", limit=exp["label_pool"], rng=rng)
    raw_test, y_test = vision.load("cifar10", split="test", limit=exp["test_samples"], rng=rng)
    x_pretrain = vision.normalize(raw_pretrain, "cifar10")
    x_pool = vision.normalize(raw_pool, "cifar10")
    x_test = vision.normalize(raw_test, "cifar10")
    print(f"[d3] {len(x_pretrain)} unlabelled, {len(x_pool)} labelled pool, {len(x_test)} test")

    encoders: dict[str, object] = {}

    # -- random: the architecture with no training at all --------------------
    torchkit.seed_torch(exp.seed)
    random_encoder, _ = make_encoder(depth=exp["depth"], width=exp["width"])
    encoders["random"] = random_encoder.to(device)

    # -- SimCLR (D1) ---------------------------------------------------------
    torchkit.seed_torch(exp.seed)
    simclr_encoder, feature_dim = make_encoder(depth=exp["depth"], width=exp["width"])
    simclr_encoder = simclr_encoder.to(device)
    head = make_projection_head(feature_dim).to(device)
    pretrain(
        simclr_encoder,
        head,
        x_pretrain,
        device=device,
        steps=exp["steps"],
        batch_size=exp["batch_size"],
        temperature=exp["temperature"],
        strength=exp["strength"],
        lr=exp["lr"],
        seed=exp.seed,
        log_every=exp["steps"],
        exp=exp,
        tag="simclr",
    )
    encoders["simclr"] = simclr_encoder

    # -- MAE (D2) ------------------------------------------------------------
    _, channels, height, width_px = x_pretrain.shape
    patch = exp["patch"]
    num_patches = (height // patch) * (width_px // patch)
    torchkit.seed_torch(exp.seed)
    mae = make_mae(
        num_patches=num_patches, patch_dim=channels * patch * patch, width=exp["mae_width"], depth=exp["mae_depth"]
    ).to(device)
    optimizer = torch.optim.AdamW(mae.parameters(), lr=exp["lr"], weight_decay=0.05)
    mae_rng = np.random.default_rng(exp.seed)
    tensor = torch.from_numpy(x_pretrain)
    for step in range(1, exp["steps"] + 1):
        index = mae_rng.integers(0, len(tensor), size=exp["batch_size"])
        images = tensor[index].to(device)
        keep_index, mask = random_mask(len(images), num_patches, exp["mask_ratio"], device=device)
        mae.train()
        loss, _ = mae(patchify(images, patch), keep_index, mask)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if step % exp["steps"] == 0:
            print(f"[d3] mae pretraining done, final masked MSE={loss.item():.4f}")

    class MAEFeatures(torch.nn.Module):
        def __init__(self, model) -> None:
            super().__init__()
            self.model = model

        def forward(self, images):
            return self.model.represent(images, patch=patch)

    encoders["mae"] = MAEFeatures(mae).to(device)

    # -- supervised on the full labelled pool: the ceiling -------------------
    encoders["supervised"] = supervised_encoder(
        x_pool,
        y_pool,
        device=device,
        depth=exp["depth"],
        width=exp["width"],
        steps=exp["steps"],
        batch_size=exp["batch_size"],
        lr=exp["lr"],
        seed=exp.seed,
    )

    def features_of(encoder, images):
        encoder.eval()
        with torch.no_grad():
            return torch.cat([encoder(batch.to(device)) for batch in torch.from_numpy(images).split(512)])

    cached = {name: (features_of(encoder, x_pool), features_of(encoder, x_test)) for name, encoder in encoders.items()}

    # -- the sweep -----------------------------------------------------------
    probe_curves: dict[str, list[float]] = {name: [] for name in encoders}
    knn_curves: dict[str, list[float]] = {name: [] for name in encoders}
    summary: dict[str, float] = {}

    for per_class in exp["labels_per_class"]:
        subset = stratified_subset(y_pool, per_class, np.random.default_rng(exp.seed))
        for name, encoder in encoders.items():
            accuracy = linear_probe(
                encoder, x_pool[subset], y_pool[subset], x_test, y_test, device=device, epochs=exp["probe_epochs"]
            )
            pool_features, test_features = cached[name]
            knn = knn_accuracy(pool_features[subset], y_pool[subset], test_features, y_test, k=exp["knn_k"])
            probe_curves[name].append(accuracy)
            knn_curves[name].append(knn)
            summary |= {f"{name}/probe@{per_class}": accuracy, f"{name}/knn@{per_class}": knn}
        print(
            f"[d3] {per_class:>3} labels/class  "
            + "  ".join(f"{name}={probe_curves[name][-1]:.3f}" for name in encoders)
        )

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "label_efficiency",
        plotting.plot_curves(
            probe_curves,
            x=exp["labels_per_class"],
            title="Linear probe accuracy vs. labels per class",
            xlabel="labels per class",
            ylabel="test accuracy",
            logx=True,
            marker="o",
        ),
    )
    exp.log_figure(
        "knn_efficiency",
        plotting.plot_curves(
            knn_curves,
            x=exp["labels_per_class"],
            title="k-NN accuracy vs. labels per class (nothing trained)",
            xlabel="labels per class",
            ylabel="test accuracy",
            logx=True,
            marker="o",
        ),
    )
    smallest, largest = exp["labels_per_class"][0], exp["labels_per_class"][-1]
    exp.log_figure(
        "low_label_regime",
        plotting.plot_comparison(
            {name: summary[f"{name}/probe@{smallest}"] for name in encoders},
            title=f"Probe accuracy with {smallest} labels per class",
            ylabel="test accuracy",
        ),
    )

    summary["simclr_minus_supervised@min_labels"] = (
        summary[f"simclr/probe@{smallest}"] - summary[f"supervised/probe@{smallest}"]
    )
    summary["simclr_minus_supervised@max_labels"] = (
        summary[f"simclr/probe@{largest}"] - summary[f"supervised/probe@{largest}"]
    )
    summary["simclr_minus_random@min_labels"] = (
        summary[f"simclr/probe@{smallest}"] - summary[f"random/probe@{smallest}"]
    )
    summary["mae_minus_random@min_labels"] = summary[f"mae/probe@{smallest}"] - summary[f"random/probe@{smallest}"]
    summary["best@min_labels"] = max(summary[f"{name}/probe@{smallest}"] for name in encoders)
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
