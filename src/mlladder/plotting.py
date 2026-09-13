"""House plotting style plus the handful of figures every part needs.

All helpers return the ``Figure`` so a caller can keep tweaking it, and
``save_fig`` is the only thing that touches disk.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import ArrayLike

matplotlib.use("Agg")  # figures are written to disk, never shown

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

# Muted, colour-blind-safe cycle; readable in print and on a projector.
PALETTE = ("#33628E", "#A9812F", "#7B5EA7", "#3F8F6B", "#B4553C", "#5C645F")

STYLE: dict[str, object] = {
    "figure.figsize": (7.0, 4.2),
    "figure.dpi": 120,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
    "axes.prop_cycle": matplotlib.cycler(color=PALETTE),
    "axes.grid": True,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "grid.alpha": 0.25,
    "legend.frameon": False,
    "font.size": 10,
}


def use_house_style() -> None:
    """Apply the shared style globally. Called for you by ``Experiment``."""
    plt.rcParams.update(STYLE)


def save_fig(fig: Figure, path: str | Path, *, close: bool = True) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    if close:
        plt.close(fig)
    return destination


def plot_curves(
    series: Mapping[str, Sequence[float]],
    *,
    title: str = "",
    xlabel: str = "step",
    ylabel: str = "value",
    logy: bool = False,
) -> Figure:
    """Training/validation curves — one line per key."""
    use_house_style()
    fig, ax = plt.subplots()
    for label, values in series.items():
        ax.plot(range(len(values)), values, label=label, linewidth=1.8)
    if logy:
        ax.set_yscale("log")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    if len(series) > 1:
        ax.legend()
    return fig


def plot_confusion_matrix(
    matrix: ArrayLike, *, labels: Sequence[str] | None = None, title: str = "Confusion matrix", normalize: bool = False
) -> Figure:
    use_house_style()
    data = np.asarray(matrix, dtype=np.float64)
    if normalize:
        totals = data.sum(axis=1, keepdims=True)
        data = np.divide(data, totals, out=np.zeros_like(data), where=totals > 0)

    fig, ax = plt.subplots(figsize=(5.2, 4.6))
    image = ax.imshow(data, cmap="Blues")
    fig.colorbar(image, ax=ax, shrink=0.85)
    ticks = labels if labels is not None else [str(i) for i in range(data.shape[0])]
    ax.set(
        title=title,
        xlabel="predicted",
        ylabel="true",
        xticks=range(len(ticks)),
        yticks=range(len(ticks)),
    )
    ax.set_xticklabels(ticks, rotation=45, ha="right")
    ax.set_yticklabels(ticks)
    ax.grid(False)
    return fig


def plot_comparison(scores: Mapping[str, float], *, title: str = "", ylabel: str = "score") -> Figure:
    """Bar chart for the bake-off rungs (B5, E4, F1, A4)."""
    use_house_style()
    fig, ax = plt.subplots()
    names = list(scores)
    ax.bar(names, [scores[name] for name in names], color=PALETTE[0], width=0.6)
    ax.set(title=title, ylabel=ylabel)
    ax.tick_params(axis="x", rotation=20)
    return fig


def plot_image_grid(images: ArrayLike, *, nrow: int = 8, title: str = "", cmap: str | None = None) -> Figure:
    """Sample grids for the generative and self-supervised parts.

    Accepts ``(n, h, w)``, ``(n, h, w, c)`` or torch-style ``(n, c, h, w)``.
    """
    use_house_style()
    batch = np.asarray(images)
    if batch.ndim == 4 and batch.shape[1] in (1, 3) and batch.shape[-1] not in (1, 3):
        batch = np.transpose(batch, (0, 2, 3, 1))
    if batch.ndim == 4 and batch.shape[-1] == 1:
        batch = batch[..., 0]

    count = len(batch)
    ncol = min(nrow, count)
    nrows = int(np.ceil(count / ncol))
    fig, axes = plt.subplots(nrows, ncol, figsize=(ncol * 1.1, nrows * 1.1))
    for ax, image in zip(np.atleast_1d(axes).ravel(), batch, strict=False):
        ax.imshow(image, cmap=cmap or ("gray" if image.ndim == 2 else None))
    for ax in np.atleast_1d(axes).ravel():
        ax.axis("off")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig
