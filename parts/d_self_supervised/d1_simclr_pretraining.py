"""D1. SimCLR-style contrastive pretraining on CIFAR-10

Lab reference: [lightly-ai — contrastive-learning reference implementations](https://github.com/lightly-ai/lightly)

Contrastive learning trains a network with no labels by asserting one thing:
two augmented views of the same image should land close together, and views of
different images should not. Everything then depends on what "augmented" means —
if the augmentation is too weak the task is solved by matching colour
histograms, and the representation learns nothing worth having.

So the rung ablates exactly that, plus the temperature, and scores every arm the
only way an unsupervised method can be scored: by how linearly separable its
frozen features turn out to be.

The encoder is B4's residual network with its classifier removed — part B built
it, part D reuses it.

Run with ``ladder run D1`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np

from mlladder import Experiment, experiment, metrics, plotting, torchkit, vision
from parts.b_deep_learning.b4_resnet_blocks import build_network


def make_encoder(*, depth: int = 4, width: int = 32):
    """B4's residual stack with the classification head taken off.

    Returns ``(encoder, feature_dim)``; the encoder maps images to the pooled
    representation that everything downstream probes.
    """
    vision.require_torch()
    from torch import nn

    network = build_network(depth, residual=True, classes=10, width=width)
    layers = list(network)
    feature_dim = layers[-1].in_features
    return nn.Sequential(*layers[:-1]), feature_dim


def make_projection_head(feature_dim: int, output_dim: int = 64):
    """The 2-layer MLP SimCLR discards after pretraining.

    Its whole purpose is to absorb the invariances the contrastive loss demands,
    so the layer *below* it keeps information the loss would otherwise throw
    away — which is why probes read the encoder, not the projection.
    """
    vision.require_torch()
    from torch import nn

    return nn.Sequential(nn.Linear(feature_dim, feature_dim), nn.ReLU(inplace=True), nn.Linear(feature_dim, output_dim))


def augment(images, *, strength: float, generator=None):
    r"""Two independent views per image: crop, flip, colour jitter, grayscale.

    Written with tensor ops rather than a torchvision pipeline so the strength
    knob is one number the ablation can sweep. ``strength=0`` leaves the image
    untouched, which is the control arm that should fail.
    """
    torch = vision.require_torch()

    batch, channels, height, width = images.shape
    out = images
    if strength <= 0:
        return out

    # Random resized crop: scale in [1 - 0.9s, 1], then resize back.
    low = max(0.2, 1.0 - 0.9 * strength)
    scale = torch.empty(1, device=images.device).uniform_(low, 1.0).item()
    crop = max(8, int(height * scale))
    top = int(torch.randint(0, height - crop + 1, (1,), generator=generator).item())
    left = int(torch.randint(0, width - crop + 1, (1,), generator=generator).item())
    out = out[:, :, top : top + crop, left : left + crop]
    out = torch.nn.functional.interpolate(out, size=(height, width), mode="bilinear", align_corners=False)

    # Horizontal flip, per sample.
    flip = torch.rand(batch, device=images.device) < 0.5
    out = torch.where(flip[:, None, None, None], out.flip(-1), out)

    # Colour jitter: per-sample brightness, contrast and channel gain.
    jitter = 0.8 * strength
    brightness = 1 + (torch.rand(batch, 1, 1, 1, device=images.device) - 0.5) * 2 * jitter
    contrast = 1 + (torch.rand(batch, 1, 1, 1, device=images.device) - 0.5) * 2 * jitter
    gain = 1 + (torch.rand(batch, channels, 1, 1, device=images.device) - 0.5) * 2 * jitter
    mean = out.mean(dim=(2, 3), keepdim=True)
    out = ((out - mean) * contrast + mean) * brightness * gain

    # Random grayscale, the augmentation that stops the network solving the
    # task with colour alone.
    if channels == 3:
        gray_mask = (torch.rand(batch, device=images.device) < 0.2 * strength)[:, None, None, None]
        gray = out.mean(dim=1, keepdim=True).expand(-1, channels, -1, -1)
        out = torch.where(gray_mask, gray, out)
    return out


def nt_xent(features, *, temperature: float):
    r"""NT-Xent over $2N$ views, from the definition.

    With $z$ the L2-normalized projections and $\mathrm{sim}(u,v) = u^\top v$,

    $$\ell_{i,j} = -\log\frac{\exp(\mathrm{sim}(z_i, z_j)/\tau)}{\sum_{k \ne i}\exp(\mathrm{sim}(z_i,z_k)/\tau)},$$

    averaged over both directions of every positive pair. Views are stacked as
    ``[view_a; view_b]``, so the positive of row $i$ is row $i + N$ (mod $2N$).
    """
    torch = vision.require_torch()

    z = torch.nn.functional.normalize(features, dim=1)
    total = len(z)
    similarity = (z @ z.T) / temperature
    # A view is not its own positive; -inf removes it from the denominator.
    similarity.fill_diagonal_(float("-inf"))

    half = total // 2
    targets = torch.arange(total, device=z.device)
    targets = (targets + half) % total
    return torch.nn.functional.cross_entropy(similarity, targets)


def linear_probe(
    encoder, x_train, y_train, x_test, y_test, *, device, epochs: int = 12, batch_size: int = 256, lr: float = 1e-2
):
    """Train a linear classifier on frozen features; return test accuracy.

    Features are extracted once — the encoder never moves, so recomputing them
    every epoch would only waste time.
    """
    torch = vision.require_torch()

    encoder.eval()
    with torch.no_grad():
        train_features = torch.cat([encoder(batch.to(device)) for batch in torch.from_numpy(x_train).split(512)])
        test_features = torch.cat([encoder(batch.to(device)) for batch in torch.from_numpy(x_test).split(512)])

    classifier = torch.nn.Linear(train_features.shape[1], int(y_train.max()) + 1).to(device)
    optimizer = torch.optim.AdamW(classifier.parameters(), lr=lr, weight_decay=1e-4)
    labels = torch.from_numpy(y_train).to(device)

    for _ in range(epochs):
        permutation = torch.randperm(len(train_features), device=device)
        for start in range(0, len(train_features), batch_size):
            index = permutation[start : start + batch_size]
            loss = torch.nn.functional.cross_entropy(classifier(train_features[index]), labels[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    with torch.no_grad():
        predicted = classifier(test_features).argmax(dim=1).cpu().numpy()
    return metrics.accuracy(y_test, predicted)


def pretrain(
    encoder,
    head,
    images,
    *,
    device,
    steps: int,
    batch_size: int,
    temperature: float,
    strength: float,
    lr: float,
    seed: int = 0,
    log_every: int = 0,
    exp: Experiment | None = None,
    tag: str = "",
):
    """Run SimCLR pretraining; returns the loss curve."""
    torch = vision.require_torch()

    parameters = [*encoder.parameters(), *head.parameters()]
    optimizer = torch.optim.AdamW(parameters, lr=lr, weight_decay=1e-4)
    rng = np.random.default_rng(seed)
    tensor = torch.from_numpy(images)

    losses = []
    for step in range(1, steps + 1):
        index = rng.integers(0, len(tensor), size=batch_size)
        batch = tensor[index].to(device)
        views = torch.cat([augment(batch, strength=strength), augment(batch, strength=strength)])

        encoder.train()
        head.train()
        loss = nt_xent(head(encoder(views)), temperature=temperature)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

        if log_every and step % log_every == 0:
            print(f"[d1] {tag} step {step:>4}/{steps}  contrastive loss={np.mean(losses[-log_every:]):.4f}")
            if exp is not None:
                exp.log({f"{tag}/contrastive_loss": float(np.mean(losses[-log_every:]))}, step=step)
    return losses


@experiment(
    "d1_simclr_pretraining",
    defaults={
        "pretrain_samples": 8000,
        "probe_samples": 4000,
        "test_samples": 2000,
        "depth": 4,
        "width": 32,
        "steps": 600,
        "batch_size": 128,
        "lr": 1e-3,
        "temperature": 0.5,
        "strength": 1.0,
        "temperatures": [0.1, 0.5, 1.0],
        "strengths": [0.0, 0.5, 1.0],
        "probe_epochs": 12,
    },
    description="SimCLR on CIFAR-10 with NT-Xent from scratch, ablated over temperature and augmentation strength.",
)
def run(exp: Experiment) -> dict[str, float]:
    """SimCLR-style contrastive pretraining on CIFAR-10."""
    torch = vision.require_torch()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)
    print(f"[d1] device: {device}")

    raw_pretrain, _ = vision.load("cifar10", split="train", limit=exp["pretrain_samples"], rng=rng)
    raw_probe, y_probe = vision.load("cifar10", split="train", limit=exp["probe_samples"], rng=rng)
    raw_test, y_test = vision.load("cifar10", split="test", limit=exp["test_samples"], rng=rng)
    x_pretrain = vision.normalize(raw_pretrain, "cifar10")
    x_probe = vision.normalize(raw_probe, "cifar10")
    x_test = vision.normalize(raw_test, "cifar10")
    print(f"[d1] {len(x_pretrain)} unlabelled / {len(x_probe)} probe / {len(x_test)} test")

    summary: dict[str, float] = {}
    curves: dict[str, list[float]] = {}

    def arm(name: str, *, temperature: float, strength: float) -> tuple[float, list[float], object]:
        torchkit.seed_torch(exp.seed)
        encoder, feature_dim = make_encoder(depth=exp["depth"], width=exp["width"])
        encoder = encoder.to(device)
        head = make_projection_head(feature_dim).to(device)
        losses = pretrain(
            encoder,
            head,
            x_pretrain,
            device=device,
            steps=exp["steps"],
            batch_size=exp["batch_size"],
            temperature=temperature,
            strength=strength,
            lr=exp["lr"],
            seed=exp.seed,
            log_every=exp["steps"] // 3,
            exp=exp,
            tag=name,
        )
        accuracy = linear_probe(encoder, x_probe, y_probe, x_test, y_test, device=device, epochs=exp["probe_epochs"])
        return accuracy, losses, encoder

    # -- the untrained control: how much is the architecture alone worth? ----
    torchkit.seed_torch(exp.seed)
    random_encoder, _ = make_encoder(depth=exp["depth"], width=exp["width"])
    random_accuracy = linear_probe(
        random_encoder.to(device), x_probe, y_probe, x_test, y_test, device=device, epochs=exp["probe_epochs"]
    )
    summary["random_encoder/probe_accuracy"] = random_accuracy
    print(f"[d1] random (untrained) encoder probe accuracy={random_accuracy:.4f}")

    # -- temperature ablation ------------------------------------------------
    temperature_scores = []
    best_encoder, best_accuracy, best_name = None, 0.0, ""
    for temperature in exp["temperatures"]:
        name = f"tau={temperature}"
        accuracy, losses, encoder = arm(name, temperature=temperature, strength=exp["strength"])
        temperature_scores.append(accuracy)
        curves[name] = losses
        summary[f"temperature/{temperature}/probe_accuracy"] = accuracy
        print(f"[d1] temperature={temperature:<5} probe accuracy={accuracy:.4f}")
        if accuracy > best_accuracy:
            best_encoder, best_accuracy, best_name = encoder, accuracy, name

    # -- augmentation ablation -----------------------------------------------
    strength_scores = []
    for strength in exp["strengths"]:
        name = f"strength={strength}"
        accuracy, losses, _ = arm(name, temperature=exp["temperature"], strength=strength)
        strength_scores.append(accuracy)
        curves[name] = losses
        summary[f"strength/{strength}/probe_accuracy"] = accuracy
        print(f"[d1] strength={strength:<5}    probe accuracy={accuracy:.4f}")

    # -- what the representation looks like ----------------------------------
    if best_encoder is not None:
        from parts.a_classical.a3_kmeans_and_pca import PCA

        best_encoder.eval()
        with torch.no_grad():
            features = (
                torch.cat([best_encoder(batch.to(device)) for batch in torch.from_numpy(x_test).split(512)])
                .cpu()
                .numpy()
            )
        projected = PCA(n_components=2).fit(features).transform(features)
        exp.log_figure(
            "feature_space",
            plotting.plot_scatter(
                projected, y_test, title=f"SimCLR features ({best_name}), PCA to 2-D", xlabel="PC1", ylabel="PC2"
            ),
        )

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "contrastive_loss",
        plotting.plot_curves(
            {name: values for name, values in curves.items() if name.startswith("tau")},
            title="NT-Xent loss by temperature",
            xlabel="step",
            ylabel="contrastive loss",
        ),
    )
    exp.log_figure(
        "temperature_ablation",
        plotting.plot_curves(
            {"linear probe": temperature_scores},
            x=exp["temperatures"],
            title="Probe accuracy vs. temperature",
            xlabel="temperature",
            ylabel="test accuracy",
            marker="o",
        ),
    )
    exp.log_figure(
        "augmentation_ablation",
        plotting.plot_curves(
            {"linear probe": strength_scores, "untrained encoder": [random_accuracy] * len(exp["strengths"])},
            x=exp["strengths"],
            title="Probe accuracy vs. augmentation strength",
            xlabel="augmentation strength",
            ylabel="test accuracy",
            marker="o",
        ),
    )

    summary["best/probe_accuracy"] = best_accuracy
    summary["best_minus_random"] = best_accuracy - random_accuracy
    # Only defined when the sweep actually ran both ends; --strengths takes any
    # subset, and a missing key here would crash the rung after all the work.
    weakest, strongest = (
        f"strength/{min(exp['strengths'])}/probe_accuracy",
        f"strength/{max(exp['strengths'])}/probe_accuracy",
    )
    if weakest in summary and strongest in summary and weakest != strongest:
        summary["augmentation_gain"] = summary[strongest] - summary[weakest]
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
