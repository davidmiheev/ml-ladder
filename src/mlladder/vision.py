"""Image datasets for parts B onward, loaded through torchvision.

Kept small on purpose: the rungs run on CPU, so every loader takes a ``limit``
and returns plain numpy. Raw pixels stay ``uint8`` (a float32 copy of CIFAR-10
is 600 MB); normalization happens on the subset you actually train on.

Downloads land in ``paths.data_dir()`` (git-ignored, ``MLLADDER_DATA_DIR`` to
relocate) and are cached by torchvision after the first run.
"""

from __future__ import annotations

import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mlladder import paths
from mlladder.datasets import as_rng

# The canonical CIFAR host (cs.toronto.edu) serves at a few hundred bytes/s
# through some proxies, which turns a 170 MB download into hours. The same
# images are mirrored on the Hugging Face hub as parquet; these are the files.
CIFAR_PARQUET = {
    "cifar10": ("uoft-cs/cifar10", "plain_text", "img", "label"),
    "cifar100": ("uoft-cs/cifar100", "cifar100", "img", "fine_label"),
}


@dataclass(frozen=True)
class DatasetSpec:
    """Per-dataset normalization statistics and shape, as the papers use them."""

    name: str
    mean: tuple[float, ...]
    std: tuple[float, ...]
    classes: int
    shape: tuple[int, int, int]  # H, W, C


SPECS = {
    "mnist": DatasetSpec("MNIST", (0.1307,), (0.3081,), 10, (28, 28, 1)),
    "fashion_mnist": DatasetSpec("FashionMNIST", (0.2860,), (0.3530,), 10, (28, 28, 1)),
    "cifar10": DatasetSpec("CIFAR10", (0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616), 10, (32, 32, 3)),
    "cifar100": DatasetSpec("CIFAR100", (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), 100, (32, 32, 3)),
}


def require_torch():
    """Import torch with an actionable message when the extra is not installed."""
    try:
        import torch
    except ImportError as error:  # pragma: no cover - exercised by the message itself
        raise SystemExit("this rung needs PyTorch — run `uv sync --extra torch`") from error
    return torch


def load(
    name: str,
    *,
    split: str = "train",
    limit: int | None = None,
    rng: int | np.random.Generator | None = None,
) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    """Return ``(images, labels)`` as ``uint8 (N, H, W, C)`` and ``int64 (N,)``.

    ``limit`` takes a *random* subset rather than the first N, so class balance
    survives; pass ``rng`` to make the subset reproducible.
    """
    if name not in SPECS:
        raise ValueError(f"unknown dataset {name!r}; known: {sorted(SPECS)}")

    images, labels = _load_cifar(name, split) if name in CIFAR_PARQUET else _load_torchvision(name, split)
    if images.ndim == 3:  # MNIST-style, add the channel axis
        images = images[..., None]

    if limit is not None and limit < len(images):
        index = as_rng(rng).choice(len(images), size=limit, replace=False)
        images, labels = images[index], labels[index]
    return np.ascontiguousarray(images, dtype=np.uint8), labels


def _load_torchvision(name: str, split: str) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    require_torch()
    import torchvision

    dataset = getattr(torchvision.datasets, SPECS[name].name)(
        root=str(paths.data_dir()), train=split == "train", download=True
    )
    return np.asarray(dataset.data), np.asarray(dataset.targets, dtype=np.int64)


def _load_cifar(name: str, split: str) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    """CIFAR from the hub mirror, decoded once and cached as an ``.npz``."""
    cache = paths.data_dir() / "cifar" / f"{name}-{split}.npz"
    if cache.exists():
        stored = np.load(cache)
        return stored["images"], stored["labels"]

    repo, config, image_column, label_column = CIFAR_PARQUET[name]
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{config}/{split}-00000-of-00001.parquet"
    parquet = paths.data_dir() / "cifar" / f"{name}-{split}.parquet"
    parquet.parent.mkdir(parents=True, exist_ok=True)
    if not parquet.exists():
        print(f"[vision] downloading {name} {split} from {repo}")
        _download(url, parquet)

    import io

    import pyarrow.parquet as pq
    from PIL import Image

    table = pq.read_table(parquet)
    raw_images = table.column(image_column).to_pylist()
    images = np.stack([np.asarray(Image.open(io.BytesIO(item["bytes"])).convert("RGB")) for item in raw_images])
    labels = np.asarray(table.column(label_column).to_pylist(), dtype=np.int64)

    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache, images=images, labels=labels)
    parquet.unlink()  # the npz is the cache from here on
    return images, labels


def _download(url: str, destination: Path) -> None:
    """Stream a URL to disk, honouring the environment's proxy settings."""
    import urllib.request

    request = urllib.request.Request(url, headers={"User-Agent": "ml-ladder"})
    with urllib.request.urlopen(request) as response, destination.open("wb") as handle:
        shutil.copyfileobj(response, handle)


def normalize(images: NDArray[np.uint8], name: str, *, channels_first: bool = True) -> NDArray[np.float32]:
    """Scale to [0, 1] and standardize with the dataset's channel statistics."""
    spec = SPECS[name]
    scaled = images.astype(np.float32) / 255.0
    scaled = (scaled - np.array(spec.mean, dtype=np.float32)) / np.array(spec.std, dtype=np.float32)
    return np.ascontiguousarray(scaled.transpose(0, 3, 1, 2)) if channels_first else scaled


def flatten(images: NDArray[np.float32]) -> NDArray[np.float32]:
    return images.reshape(len(images), -1)


def as_tensors(images: NDArray[np.float32], labels: NDArray[np.int64]):
    """Numpy to torch, without a copy where possible."""
    torch = require_torch()
    return torch.from_numpy(images), torch.from_numpy(labels)


def batches(n_samples: int, batch_size: int, rng: np.random.Generator, *, shuffle: bool = True):
    """Index batches for one epoch, shared by the numpy and torch rungs."""
    order = rng.permutation(n_samples) if shuffle else np.arange(n_samples)
    for start in range(0, n_samples, batch_size):
        yield order[start : start + batch_size]
