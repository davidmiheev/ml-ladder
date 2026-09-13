"""Synthetic datasets, numpy only.

Part A forbids sklearn, and that ban is easiest to keep if the data generators
are ours too. Everything here is deterministic given a seed and small enough to
fit in memory many times over.
"""

from __future__ import annotations

import numpy as np
from numpy.random import Generator
from numpy.typing import NDArray

Array = NDArray[np.float64]
Labels = NDArray[np.int64]


def as_rng(seed: int | Generator | None = None) -> Generator:
    """Accept a seed, a generator, or nothing."""
    return seed if isinstance(seed, Generator) else np.random.default_rng(seed)


def make_regression(
    n_samples: int = 200,
    n_features: int = 3,
    *,
    noise: float = 0.1,
    bias: float = 0.0,
    rng: int | Generator | None = None,
) -> tuple[Array, Array, Array]:
    """Linear targets with gaussian noise.

    Returns ``(X, y, true_weights)`` — the true weights make it possible to
    check an optimizer against the answer, not just against its own loss.
    """
    generator = as_rng(rng)
    weights = generator.normal(size=n_features)
    features = generator.normal(size=(n_samples, n_features))
    targets = features @ weights + bias + generator.normal(scale=noise, size=n_samples)
    return features, targets, weights


def make_blobs(
    n_samples: int = 300,
    *,
    centers: int = 3,
    n_features: int = 2,
    cluster_std: float = 1.0,
    center_box: tuple[float, float] = (-6.0, 6.0),
    rng: int | Generator | None = None,
) -> tuple[Array, Labels, Array]:
    """Isotropic gaussian blobs. Returns ``(X, y, centers)``."""
    generator = as_rng(rng)
    middles = generator.uniform(center_box[0], center_box[1], size=(centers, n_features))
    counts = np.full(centers, n_samples // centers)
    counts[: n_samples % centers] += 1

    features = np.concatenate(
        [
            generator.normal(loc=middle, scale=cluster_std, size=(count, n_features))
            for middle, count in zip(middles, counts, strict=True)
        ]
    )
    labels = np.repeat(np.arange(centers), counts)
    shuffled_features, shuffled_labels = shuffle(features, labels, rng=generator)
    return shuffled_features, shuffled_labels, middles


def make_moons(n_samples: int = 300, *, noise: float = 0.1, rng: int | Generator | None = None) -> tuple[Array, Labels]:
    """Two interleaving half-circles — linearly inseparable on purpose."""
    generator = as_rng(rng)
    outer = n_samples // 2
    inner = n_samples - outer

    outer_angles = np.linspace(0, np.pi, outer)
    inner_angles = np.linspace(0, np.pi, inner)
    features = np.vstack(
        [
            np.column_stack([np.cos(outer_angles), np.sin(outer_angles)]),
            np.column_stack([1 - np.cos(inner_angles), 0.5 - np.sin(inner_angles)]),
        ]
    )
    features += generator.normal(scale=noise, size=features.shape)
    labels = np.concatenate([np.zeros(outer, dtype=np.int64), np.ones(inner, dtype=np.int64)])
    return shuffle(features, labels, rng=generator)


def make_spirals(
    n_samples: int = 300, *, classes: int = 3, noise: float = 0.15, rng: int | Generator | None = None
) -> tuple[Array, Labels]:
    """Interleaved spiral arms — hard for axis-aligned splits, easy for depth."""
    generator = as_rng(rng)
    per_class = n_samples // classes
    features, labels = [], []
    for label in range(classes):
        radius = np.linspace(0.1, 1.0, per_class)
        theta = np.linspace(label * 2 * np.pi / classes, label * 2 * np.pi / classes + 3.0, per_class)
        theta = theta + generator.normal(scale=noise, size=per_class)
        features.append(np.column_stack([radius * np.sin(theta), radius * np.cos(theta)]))
        labels.append(np.full(per_class, label, dtype=np.int64))
    return shuffle(np.concatenate(features), np.concatenate(labels), rng=generator)


def make_friedman1(
    n_samples: int = 500, *, n_features: int = 10, noise: float = 1.0, rng: int | Generator | None = None
) -> tuple[Array, Array]:
    """Friedman's non-linear regression benchmark.

    ``y = 10 sin(pi x0 x1) + 20 (x2 - 0.5)^2 + 10 x3 + 5 x4 + noise``, with the
    remaining features pure distractors — a fair test for trees and boosting.
    """
    generator = as_rng(rng)
    if n_features < 5:
        raise ValueError("friedman1 needs at least 5 features")
    features = generator.uniform(size=(n_samples, n_features))
    targets = (
        10 * np.sin(np.pi * features[:, 0] * features[:, 1])
        + 20 * (features[:, 2] - 0.5) ** 2
        + 10 * features[:, 3]
        + 5 * features[:, 4]
        + generator.normal(scale=noise, size=n_samples)
    )
    return features, targets


def shuffle(features: Array, labels: Labels, *, rng: int | Generator | None = None) -> tuple[Array, Labels]:
    generator = as_rng(rng)
    order = generator.permutation(len(features))
    return features[order], labels[order]


def train_test_split(
    features: Array,
    targets: Array,
    *,
    test_size: float = 0.25,
    rng: int | Generator | None = None,
) -> tuple[Array, Array, Array, Array]:
    """Random split. Returns ``(X_train, X_test, y_train, y_test)``."""
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be in (0, 1)")
    generator = as_rng(rng)
    order = generator.permutation(len(features))
    cut = round(len(features) * (1 - test_size))
    train, test = order[:cut], order[cut:]
    return features[train], features[test], targets[train], targets[test]


def standardize(train: Array, *others: Array) -> tuple[Array, ...]:
    """Zero-mean, unit-variance using the training statistics only."""
    mean = train.mean(axis=0)
    scale = train.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return tuple((array - mean) / scale for array in (train, *others))
