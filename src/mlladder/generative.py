"""Shared machinery for part E: how do you score a model that has no labels?

A classifier is scored by counting correct answers. A generative model has no
correct answer — it proposes a distribution, and the question is whether that
distribution resembles the data. Part E's four rungs all need the same two
things, so they live here rather than in whichever rung happened to need them
first:

- a **feature space** to compare distributions in, because comparing images
  pixel-wise says a blurry mean is close to everything;
- a **distance between distributions** in that space.

The distance is Fréchet: fit a Gaussian to each set of features and compute the
Wasserstein-2 distance between them in closed form. That is exactly what FID
is. What makes FID *FID* is the feature extractor — an ImageNet Inception —
and using a small classifier trained on the dataset instead makes these numbers
**incomparable with any published FID**. They are comparable with each other,
which is all part E asks of them, and the module says so rather than letting a
reader assume otherwise.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mlladder import torchkit, vision


def frechet_distance(a: NDArray[np.float64], b: NDArray[np.float64], *, eps: float = 1e-6) -> float:
    r"""Wasserstein-2 distance between Gaussians fitted to two feature sets.

    $$d^2 = \lVert \mu_a - \mu_b \rVert^2 + \operatorname{tr}\!\left(\Sigma_a + \Sigma_b - 2(\Sigma_a^{1/2}\Sigma_b\Sigma_a^{1/2})^{1/2}\right)$$

    The awkward term is the matrix square root. It is computed here without
    scipy by noting that $\Sigma_a^{1/2}\Sigma_b\Sigma_a^{1/2}$ is symmetric
    positive semi-definite whenever both covariances are, so a symmetric
    eigendecomposition gives its square root's trace as the sum of the square
    roots of its eigenvalues. Eigenvalues are clipped at zero because finite
    samples make the smallest ones drift very slightly negative.
    """
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    mu_a, mu_b = a.mean(axis=0), b.mean(axis=0)
    # The ridge keeps the covariances strictly positive definite, which matters
    # when a mode-collapsed generator produces features with no variance at all.
    sigma_a = np.cov(a, rowvar=False) + eps * np.eye(a.shape[1])
    sigma_b = np.cov(b, rowvar=False) + eps * np.eye(b.shape[1])

    values, vectors = np.linalg.eigh(sigma_a)
    root_a = vectors @ np.diag(np.sqrt(np.clip(values, 0, None))) @ vectors.T
    middle = root_a @ sigma_b @ root_a
    cross = np.sqrt(np.clip(np.linalg.eigvalsh(middle), 0, None)).sum()
    return float(((mu_a - mu_b) ** 2).sum() + np.trace(sigma_a) + np.trace(sigma_b) - 2 * cross)


def make_feature_extractor(*, channels: int = 16, num_classes: int = 10, image_channels: int = 1):
    """A small CNN whose penultimate activations are the feature space."""
    vision.require_torch()
    from torch import nn

    class Features(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(image_channels, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(channels, channels * 2, 3, padding=1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),
                nn.Conv2d(channels * 2, channels * 4, 3, padding=1),
                nn.BatchNorm2d(channels * 4),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
            )
            self.head = nn.Linear(channels * 4, num_classes)

        def forward(self, images, *, features_only: bool = False):
            features = self.body(images)
            return features if features_only else self.head(features)

    return Features


def train_feature_extractor(
    images, labels, *, device, seed: int, epochs: int = 2, batch_size: int = 256, lr: float = 2e-3
):
    """Train the scorer on real data, deterministically.

    Seeded and retrained per run rather than cached to disk: a feature space
    that silently differed between two rungs would make their scores
    incomparable, which is the one thing this module exists to prevent.
    """
    torch = vision.require_torch()

    torchkit.seed_torch(seed)
    model = make_feature_extractor(num_classes=int(labels.max()) + 1, image_channels=images.shape[1])().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    x = torch.from_numpy(images).to(device)
    y = torch.from_numpy(labels).to(device)
    generator = np.random.default_rng(seed)
    for _ in range(epochs):
        model.train()
        order = torch.from_numpy(generator.permutation(len(x))).to(device)
        for start in range(0, len(x), batch_size):
            index = order[start : start + batch_size]
            loss = torch.nn.functional.cross_entropy(model(x[index]), y[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    model.eval()
    return model


def extract(model, images, *, device, batch_size: int = 512) -> NDArray[np.float64]:
    """Penultimate features for a batch of images, as numpy."""
    torch = vision.require_torch()
    tensor = torch.from_numpy(images) if isinstance(images, np.ndarray) else images
    model.eval()
    chunks = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            batch = tensor[start : start + batch_size].to(device)
            chunks.append(model(batch, features_only=True).cpu().numpy())
    return np.concatenate(chunks).astype(np.float64)


def distribution_distance(model, real, fake, *, device) -> float:
    """Fréchet distance between real and generated images in the scorer's space.

    Reported throughout part E as ``fd`` rather than ``FID`` — the number is a
    Fréchet distance, but not *the* Fréchet Inception Distance, and the name
    should not imply a comparison to the literature that is not available.
    """
    return frechet_distance(extract(model, real, device=device), extract(model, fake, device=device))


def nearest_neighbour_distance(fake, real, *, sample: int = 256, rng=None) -> float:
    """Mean L2 from each generated image to its closest real one.

    A memorisation check. A model that has learned the training set rather than
    the distribution scores *too* well here, and a Fréchet distance alone cannot
    tell that apart from having learned the distribution properly.
    """
    generator = np.random.default_rng(rng)
    fake_flat = np.asarray(fake, dtype=np.float32).reshape(len(fake), -1)
    real_flat = np.asarray(real, dtype=np.float32).reshape(len(real), -1)
    if len(fake_flat) > sample:
        fake_flat = fake_flat[generator.permutation(len(fake_flat))[:sample]]
    distances = []
    for start in range(0, len(fake_flat), 64):
        block = fake_flat[start : start + 64]
        gaps = ((block[:, None, :] - real_flat[None, :, :]) ** 2).sum(axis=2)
        distances.append(np.sqrt(gaps.min(axis=1)))
    return float(np.concatenate(distances).mean())


def class_coverage(model, images, *, device, num_classes: int | None = None) -> dict[str, float]:
    r"""What fraction of the data's classes a sample set actually covers.

    A Fréchet distance is a single number over a fitted Gaussian, and a
    generator that has collapsed onto three of ten digits can still land a
    respectable one — the mean and covariance of its features are simply wrong
    in a way that averages out. Classifying the samples and looking at the
    histogram catches that directly.

    Reported as the normalized entropy

    $$H_{\text{norm}} = \frac{-\sum_c p_c \log p_c}{\log C},$$

    which is 1 when the sample set is spread evenly over all $C$ classes and
    falls toward 0 as it concentrates, plus the count of classes that appear at
    all. The classifier is the same one the Fréchet distance uses, so this costs
    one extra forward pass and no extra assumptions.
    """
    torch = vision.require_torch()
    tensor = torch.from_numpy(images) if isinstance(images, np.ndarray) else images
    model.eval()
    predictions = []
    with torch.no_grad():
        for start in range(0, len(tensor), 512):
            logits = model(tensor[start : start + 512].to(device))
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    predicted = np.concatenate(predictions)
    classes = num_classes or int(predicted.max()) + 1
    counts = np.bincount(predicted, minlength=classes).astype(np.float64)
    probabilities = counts / counts.sum()
    nonzero = probabilities[probabilities > 0]
    # ``+ 0.0`` normalises the negative zero a fully collapsed sample set
    # produces (p = 1 gives -0.0), which otherwise prints as "-0.000".
    entropy = float(-(nonzero * np.log(nonzero)).sum() / np.log(classes)) + 0.0
    return {
        "coverage": float((counts > 0).sum() / classes),
        "entropy": entropy,
        "max_class_share": float(probabilities.max()),
    }
