"""Small, dependency-light metrics shared across parts.

Deliberately numpy-only: part A forbids sklearn, and the later parts should be
scored by the same code that scored the from-scratch implementations.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike, NDArray


def _as_1d(values: ArrayLike) -> NDArray[np.float64]:
    return np.asarray(values, dtype=np.float64).ravel()


def accuracy(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    true, pred = _as_1d(y_true), _as_1d(y_pred)
    return float((true == pred).mean())


def top_k_accuracy(y_true: ArrayLike, scores: ArrayLike, k: int = 5) -> float:
    """``scores`` is (n_samples, n_classes) of logits or probabilities."""
    true = np.asarray(y_true).ravel()
    ranked = np.argsort(np.asarray(scores), axis=1)[:, ::-1][:, :k]
    return float(np.mean([label in row for label, row in zip(true, ranked, strict=True)]))


def confusion_matrix(y_true: ArrayLike, y_pred: ArrayLike, num_classes: int | None = None) -> NDArray[np.int64]:
    true = np.asarray(y_true).ravel().astype(int)
    pred = np.asarray(y_pred).ravel().astype(int)
    n = num_classes or int(max(true.max(), pred.max())) + 1
    matrix = np.zeros((n, n), dtype=np.int64)
    np.add.at(matrix, (true, pred), 1)
    return matrix


def precision_recall_f1(
    y_true: ArrayLike, y_pred: ArrayLike, num_classes: int | None = None
) -> dict[str, NDArray[np.float64]]:
    """Per-class precision/recall/F1, with 0.0 where the denominator vanishes."""
    matrix = confusion_matrix(y_true, y_pred, num_classes)
    tp = np.diag(matrix).astype(np.float64)
    predicted = matrix.sum(axis=0).astype(np.float64)
    actual = matrix.sum(axis=1).astype(np.float64)
    precision = np.divide(tp, predicted, out=np.zeros_like(tp), where=predicted > 0)
    recall = np.divide(tp, actual, out=np.zeros_like(tp), where=actual > 0)
    denominator = precision + recall
    f1 = np.divide(2 * precision * recall, denominator, out=np.zeros_like(tp), where=denominator > 0)
    return {"precision": precision, "recall": recall, "f1": f1}


def cluster_purity(y_true: ArrayLike, cluster_ids: ArrayLike) -> float:
    """Fraction of points in the majority class of their own cluster.

    Label-permutation invariant, so it can score clusterings against ground
    truth — but it rises with the number of clusters, so read it next to the
    adjusted Rand index rather than alone.
    """
    matrix = _contingency(y_true, cluster_ids)
    return float(matrix.max(axis=0).sum() / matrix.sum())


def adjusted_rand_index(y_true: ArrayLike, cluster_ids: ArrayLike) -> float:
    """Rand index corrected for chance: 1.0 is exact, 0.0 is a random split."""
    matrix = _contingency(y_true, cluster_ids).astype(np.float64)
    pairs = _choose_two(matrix).sum()
    rows = _choose_two(matrix.sum(axis=1)).sum()
    columns = _choose_two(matrix.sum(axis=0)).sum()
    total = _choose_two(np.array([matrix.sum()]))[0]
    expected = rows * columns / total
    maximum = (rows + columns) / 2
    return float((pairs - expected) / (maximum - expected)) if maximum != expected else 0.0


def _contingency(y_true: ArrayLike, cluster_ids: ArrayLike) -> NDArray[np.int64]:
    true = np.asarray(y_true).ravel()
    clusters = np.asarray(cluster_ids).ravel()
    true_labels, true_index = np.unique(true, return_inverse=True)
    _, cluster_index = np.unique(clusters, return_inverse=True)
    matrix = np.zeros((len(true_labels), cluster_index.max() + 1), dtype=np.int64)
    np.add.at(matrix, (true_index, cluster_index), 1)
    return matrix


def _choose_two(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    values = np.asarray(counts, dtype=np.float64)
    return values * (values - 1) / 2


def mse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    true, pred = _as_1d(y_true), _as_1d(y_pred)
    return float(np.mean((true - pred) ** 2))


def rmse(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    return float(np.sqrt(mse(y_true, y_pred)))


def mae(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    true, pred = _as_1d(y_true), _as_1d(y_pred)
    return float(np.mean(np.abs(true - pred)))


def r2_score(y_true: ArrayLike, y_pred: ArrayLike) -> float:
    true, pred = _as_1d(y_true), _as_1d(y_pred)
    residual = np.sum((true - pred) ** 2)
    total = np.sum((true - true.mean()) ** 2)
    return float(1.0 - residual / total) if total > 0 else 0.0


def cross_entropy(y_true: ArrayLike, probs: ArrayLike, eps: float = 1e-12) -> float:
    """Mean negative log-likelihood of the true class."""
    true = np.asarray(y_true).ravel().astype(int)
    p = np.clip(np.asarray(probs, dtype=np.float64), eps, 1.0)
    return float(-np.mean(np.log(p[np.arange(len(true)), true])))
