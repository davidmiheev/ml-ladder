"""Figure for part I's executor run, from docs/writeups/data/i-whisper-executor.json.

uv run python tools/real_models/plot_whisper_executor.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from mlladder import plotting  # noqa: E402

ARMS = [
    ("tiny.en/cuda-eager", "candle executor\ntiny.en"),
    ("hf-greedy/tiny.en/executor_prompt_auto", "transformers, executor's\nprompt, tiny.en"),
    ("hf-generate/tiny.en", "transformers\ntiny.en"),
    ("turbo/cuda-eager-en", "candle executor\nturbo, en"),
    ("turbo/cuda-eager-auto", "candle executor\nturbo, auto"),
    ("hf-generate/large-v3-turbo", "transformers\nturbo, en"),
]


def main() -> None:
    data = json.loads((ROOT / "docs/writeups/data/i-whisper-executor.json").read_text())["arms"]
    plotting.use_house_style()
    palette = plotting.PALETTE
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw={"width_ratios": [1.2, 1.0, 1.0]})
    labels = [label.replace("\n", " ") for _, label in ARMS]
    colours = [palette[0] if key.startswith(("tiny", "turbo")) else palette[1] for key, _ in ARMS]
    for ax, metric, title, xlabel in (
        (axes[0], "digit_exact", "isolated digits read correctly (50 clips)", "exact match"),
        (axes[1], "sequence_wer", "four-digit sequences (10 clips)", "word error rate (lower is better)"),
    ):
        values = [data[key]["scores"][metric] for key, _ in ARMS]
        ax.barh(range(len(ARMS)), values, color=colours)
        for index, value in enumerate(values):
            ax.text(
                value + 0.005,
                index,
                f"{value:.3f}" if metric == "sequence_wer" else f"{value:.2f}",
                va="center",
                fontsize=8,
            )
        ax.set_yticks(range(len(ARMS)), labels if ax is axes[0] else [], fontsize=8)
        ax.invert_yaxis()
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(xlabel)
    axes[0].set_xlim(0, 1)
    ax = axes[2]
    timed = [
        ("tiny.en/cuda-eager", "tiny.en, T4"),
        ("tiny.en/cpu", "tiny.en, CPU"),
        ("turbo/cuda-eager-en", "turbo, en, T4"),
        ("turbo/cuda-eager-auto", "turbo, auto, T4"),
        ("turbo/cuda-static-graph-auto", "turbo, auto,\nCUDA graph"),
    ]
    seconds = [data[key]["scores"]["median_s"] for key, _ in timed]
    ax.barh(range(len(timed)), seconds, color=palette[0])
    for index, value in enumerate(seconds):
        ax.text(value + 0.05, index, f"{value:.2f} s", va="center", fontsize=8)
    ax.set_yticks(range(len(timed)), [label for _, label in timed], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("median seconds per request (model load included)")
    ax.set_title("the executor loads the model per request", fontsize=9.5)
    fig.suptitle(
        "Whisper through ignite-catalog's candle-transcription executor (blue) and transformers (gold), same bytes, held-out speaker",
        fontsize=10,
    )
    fig.tight_layout()
    path = plotting.save_fig(fig, ROOT / "docs/writeups/figures/i/i_whisper_executor.png")
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
