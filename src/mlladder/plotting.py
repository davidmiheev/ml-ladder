"""House plotting style plus the handful of figures every part needs.

All helpers return the ``Figure`` so a caller can keep tweaking it, and
``save_fig`` is the only thing that touches disk.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import matplotlib
import numpy as np
from numpy.typing import ArrayLike, NDArray

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
    x: Sequence[float] | None = None,
    title: str = "",
    xlabel: str = "step",
    ylabel: str = "value",
    logy: bool = False,
    logx: bool = False,
    marker: str | None = None,
) -> Figure:
    """Training/validation curves — one line per key.

    ``x`` gives explicit x values (a depth or size sweep); without it the
    series are plotted against their index, as training curves want.
    """
    use_house_style()
    fig, ax = plt.subplots()
    for label, values in series.items():
        positions = list(x) if x is not None else list(range(len(values)))
        ax.plot(positions, values, label=label, linewidth=1.8, marker=marker)
    if logy:
        ax.set_yscale("log")
    if logx:
        ax.set_xscale("log")
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


def plot_attention(
    weights: ArrayLike,
    *,
    title: str = "",
    source_labels: Sequence[str] | None = None,
    target_labels: Sequence[str] | None = None,
    xlabel: str = "source position (key)",
    ylabel: str = "output step (query)",
    max_ticks: int = 40,
) -> Figure:
    """An attention map, with the axes attention actually has.

    Rows are decoding steps (queries), columns are the positions attended to
    (keys) — not a confusion matrix, whose "true"/"predicted" axes say something
    quite different. Character labels are drawn only while they still fit.
    """
    use_house_style()
    matrix = np.asarray(weights, dtype=np.float64)
    fig, ax = plt.subplots(figsize=(6.0, 4.4))
    image = ax.imshow(matrix, cmap="Blues", aspect="auto", vmin=0.0, interpolation="nearest")
    fig.colorbar(image, ax=ax, shrink=0.85, label="attention weight")

    if source_labels is not None and len(source_labels) <= max_ticks:
        ax.set_xticks(range(len(source_labels)))
        ax.set_xticklabels(source_labels, fontsize=7)
    if target_labels is not None and len(target_labels) <= max_ticks:
        ax.set_yticks(range(len(target_labels)))
        ax.set_yticklabels(target_labels, fontsize=7)
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
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


def plot_scatter(
    features: ArrayLike,
    labels: ArrayLike | None = None,
    *,
    title: str = "",
    xlabel: str = "x0",
    ylabel: str = "x1",
    centers: ArrayLike | None = None,
) -> Figure:
    """2-D scatter, coloured by label, with optional cluster centres marked."""
    use_house_style()
    points = np.asarray(features)
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    if labels is None:
        ax.scatter(points[:, 0], points[:, 1], s=14, alpha=0.85, color=PALETTE[0])
    else:
        tags = np.asarray(labels)
        for index, value in enumerate(np.unique(tags)):
            mask = tags == value
            ax.scatter(
                points[mask, 0],
                points[mask, 1],
                s=14,
                alpha=0.85,
                color=PALETTE[index % len(PALETTE)],
                label=str(value),
            )
        ax.legend(title="class", fontsize=8)
    if centers is not None:
        marks = np.asarray(centers)
        ax.scatter(
            marks[:, 0], marks[:, 1], s=180, marker="X", edgecolor="black", linewidth=1.2, color="white", zorder=3
        )
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    return fig


def plot_decision_boundary(
    predict: Callable[[NDArray[np.float64]], ArrayLike],
    features: ArrayLike,
    labels: ArrayLike,
    *,
    title: str = "",
    resolution: int = 200,
    margin: float = 0.5,
) -> Figure:
    """Shade a 2-D classifier's decision regions behind the training points.

    ``predict`` takes an ``(n, 2)`` array and returns hard labels.
    """
    use_house_style()
    points = np.asarray(features, dtype=np.float64)
    tags = np.asarray(labels)
    x_min, y_min = points.min(axis=0) - margin
    x_max, y_max = points.max(axis=0) + margin
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, resolution), np.linspace(y_min, y_max, resolution))
    grid = np.column_stack([xx.ravel(), yy.ravel()])
    regions = np.asarray(predict(grid)).reshape(xx.shape)

    classes = np.unique(tags)
    cmap = matplotlib.colors.ListedColormap([PALETTE[i % len(PALETTE)] for i in range(len(classes))])
    fig, ax = plt.subplots(figsize=(5.4, 4.6))
    ax.contourf(xx, yy, regions, levels=len(classes) - 1 if len(classes) > 1 else 1, cmap=cmap, alpha=0.18)
    for index, value in enumerate(classes):
        mask = tags == value
        ax.scatter(
            points[mask, 0],
            points[mask, 1],
            s=16,
            color=PALETTE[index % len(PALETTE)],
            edgecolor="white",
            linewidth=0.4,
            label=str(value),
        )
    ax.set(title=title, xlabel="x0", ylabel="x1")
    ax.legend(title="class", fontsize=8)
    ax.grid(False)
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


def plot_boxes(
    images: ArrayLike,
    predictions: Sequence[ArrayLike],
    *,
    truths: Sequence[ArrayLike] | None = None,
    labels: Sequence[ArrayLike] | None = None,
    class_names: Sequence[str] | None = None,
    scores: Sequence[ArrayLike] | None = None,
    nrow: int = 4,
    title: str = "",
) -> Figure:
    """Detections over images, with ground truth underneath for comparison.

    Part H's detection rungs are the first ones whose output is not a number
    per image, and a table of mAP says nothing about *how* a detector is wrong
    — duplicate boxes, boxes off by a stride, the right box with the wrong
    class all read the same. Truth is drawn dashed and grey, predictions solid
    and coloured by class, so the failure mode is visible rather than inferred.

    ``predictions[i]`` is an ``(n, 4)`` array of ``(x1, y1, x2, y2)`` for image
    ``i``; every other per-image sequence is optional and indexed the same way.
    """
    from matplotlib.patches import Rectangle

    use_house_style()
    batch = np.asarray(images)
    if batch.ndim == 4 and batch.shape[1] in (1, 3):
        batch = np.transpose(batch, (0, 2, 3, 1))
    if batch.ndim == 4 and batch.shape[-1] == 1:
        batch = batch[..., 0]

    count = len(batch)
    ncol = min(nrow, count)
    nrows = int(np.ceil(count / ncol))
    fig, axes = plt.subplots(nrows, ncol, figsize=(ncol * 2.0, nrows * 2.1), squeeze=False)
    flat = axes.ravel()
    for index, ax in enumerate(flat):
        ax.axis("off")
        ax.grid(False)
        if index >= count:
            continue
        image = batch[index]
        ax.imshow(
            image, cmap="gray" if image.ndim == 2 else None, vmin=0, vmax=255 if image.dtype == np.uint8 else None
        )

        if truths is not None:
            for x1, y1, x2, y2 in np.asarray(truths[index]).reshape(-1, 4):
                ax.add_patch(
                    Rectangle(
                        (x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor="#888888", linewidth=1.6, linestyle="--"
                    )
                )
        boxes = np.asarray(predictions[index]).reshape(-1, 4)
        for order, (x1, y1, x2, y2) in enumerate(boxes):
            class_id = int(np.asarray(labels[index]).reshape(-1)[order]) if labels is not None else 0
            colour = PALETTE[class_id % len(PALETTE)]
            ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, edgecolor=colour, linewidth=1.4))
            caption = class_names[class_id][:4] if class_names is not None else ""
            if scores is not None:
                score = float(np.asarray(scores[index]).reshape(-1)[order])
                caption = f"{caption} {score:.2f}".strip()
            if caption:
                ax.text(x1, y1 - 1.5, caption, color=colour, fontsize=6.5, va="bottom")
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    return fig


def plot_histogram(
    values: ArrayLike,
    *,
    bins: int | Sequence[float] = 25,
    markers: Mapping[str, float] | None = None,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "count",
) -> Figure:
    """A distribution with optional vertical reference lines.

    Summary statistics hide shape: a mean IoU of 0.79 is equally consistent
    with every box being mediocre and with most boxes being excellent beside a
    tail of misses, and those two call for different fixes. ``markers`` draws
    the thresholds a metric is evaluated at, so how much of the mass sits on
    either side is visible rather than computed.
    """
    use_house_style()
    fig, ax = plt.subplots()
    ax.hist(np.asarray(values).ravel(), bins=bins, color=PALETTE[0])
    for index, (label, position) in enumerate((markers or {}).items()):
        ax.axvline(position, color=PALETTE[(index + 1) % len(PALETTE)], linestyle="--", label=label)
    if markers:
        ax.legend()
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
    fig.tight_layout()
    return fig
