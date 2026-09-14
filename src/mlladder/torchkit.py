"""The small amount of PyTorch plumbing every torch rung repeats.

Deliberately thin: loaders from numpy arrays, one training epoch, one
evaluation pass, and a ``fit`` that records history. Anything a rung is
*about* — the model, the optimizer comparison, the ablation — stays in the
rung.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Iterator
from typing import Any

import numpy as np
from numpy.typing import NDArray

from mlladder.vision import require_torch


def device() -> Any:
    """CUDA when there is a GPU, else CPU. The rungs are sized for CPU."""
    torch = require_torch()
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def count_parameters(model: Any, *, trainable_only: bool = True) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)


def loaders(
    x_train: NDArray[np.float32],
    y_train: NDArray[np.int64],
    x_test: NDArray[np.float32],
    y_test: NDArray[np.int64],
    *,
    batch_size: int = 128,
    seed: int = 0,
) -> tuple[Any, Any]:
    """Train/test ``DataLoader`` pair over in-memory numpy arrays."""
    torch = require_torch()
    from torch.utils.data import DataLoader, TensorDataset

    generator = torch.Generator().manual_seed(seed)
    train = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    test = TensorDataset(torch.from_numpy(x_test), torch.from_numpy(y_test))
    return (
        DataLoader(train, batch_size=batch_size, shuffle=True, generator=generator, drop_last=False),
        DataLoader(test, batch_size=512, shuffle=False),
    )


def eval_loader(features: NDArray[np.float32], labels: NDArray[np.int64], *, batch_size: int = 512) -> Any:
    """A shuffle-free loader over one array pair, for measuring in eval mode."""
    require_torch()
    import torch
    from torch.utils.data import DataLoader, TensorDataset

    dataset = TensorDataset(torch.from_numpy(features), torch.from_numpy(labels))
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


def train_epoch(model: Any, loader: Any, optimizer: Any, *, target: Any = None) -> float:
    """One pass over the data. Returns the mean loss, weighted by batch size."""
    torch = require_torch()
    criterion = target or torch.nn.functional.cross_entropy

    model.train()
    total, count = 0.0, 0
    for features, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        loss = criterion(model(features), labels)
        loss.backward()
        optimizer.step()
        total += loss.item() * len(labels)
        count += len(labels)
    return total / max(count, 1)


def evaluate(model: Any, loader: Any) -> tuple[float, float]:
    """Mean loss and accuracy over a loader, with gradients off."""
    torch = require_torch()
    model.eval()
    total_loss, correct, count = 0.0, 0, 0
    with torch.no_grad():
        for features, labels in loader:
            logits = model(features)
            total_loss += torch.nn.functional.cross_entropy(logits, labels, reduction="sum").item()
            correct += int((logits.argmax(dim=1) == labels).sum())
            count += len(labels)
    return total_loss / max(count, 1), correct / max(count, 1)


def fit(
    model: Any,
    train_loader: Any,
    test_loader: Any,
    optimizer: Any,
    *,
    epochs: int,
    scheduler: Any = None,
    on_epoch: Callable[[int, dict[str, float]], None] | None = None,
) -> dict[str, list[float]]:
    """Train for ``epochs``, recording loss/accuracy and wall-clock per epoch."""
    history: dict[str, list[float]] = {"train_loss": [], "test_loss": [], "test_accuracy": [], "lr": [], "seconds": []}
    for epoch in range(epochs):
        started = time.perf_counter()
        train_loss = train_epoch(model, train_loader, optimizer)
        test_loss, accuracy = evaluate(model, test_loader)
        if scheduler is not None:
            scheduler.step()

        history["train_loss"].append(train_loss)
        history["test_loss"].append(test_loss)
        history["test_accuracy"].append(accuracy)
        history["lr"].append(float(optimizer.param_groups[0]["lr"]))
        history["seconds"].append(time.perf_counter() - started)
        if on_epoch is not None:
            on_epoch(epoch, {key: values[-1] for key, values in history.items()})
    return history


def predictions(model: Any, loader: Any) -> tuple[NDArray[np.int64], NDArray[np.int64]]:
    """Predicted and true labels over a loader, for confusion matrices."""
    torch = require_torch()
    model.eval()
    predicted: list[NDArray[np.int64]] = []
    actual: list[NDArray[np.int64]] = []
    with torch.no_grad():
        for features, labels in loader:
            predicted.append(model(features).argmax(dim=1).numpy())
            actual.append(labels.numpy())
    return np.concatenate(predicted), np.concatenate(actual)


def seed_torch(seed: int) -> None:
    """Seed torch's generators; numpy/random are handled by the harness."""
    torch = require_torch()
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def iter_parameters(model: Any) -> Iterator[tuple[str, Any]]:
    yield from model.named_parameters()
