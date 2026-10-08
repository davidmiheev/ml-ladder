"""Figure for part I's whisper-example diagnosis, from docs/writeups/data/i-whisper-example-rules.json.

uv run python tools/real_models/plot_whisper_example.py
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
    ("upstream", "upstream example"),
    ("fixed", "pairing rule fixed"),
    ("prompt", ".en prompt fixed"),
    ("patched", "both fixed"),
    ("hf-timestamps", "transformers,\ntimestamps on"),
]


def main() -> None:
    data = json.loads((ROOT / "docs/writeups/data/i-whisper-example-rules.json").read_text())
    arms, agree = data["arms"], data["agreement_with_transformers"]
    plotting.use_house_style()
    palette = plotting.PALETTE
    colours = [palette[1] if key == "hf-timestamps" else palette[0] for key, _ in ARMS]
    labels = [label for _, label in ARMS]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    panels = [
        (
            lambda k: arms[k]["scores"]["clips_with_empty_segment"],
            "clips with an empty segment (of 61)",
            "lower is better",
            61,
            "{:.0f}",
        ),
        (
            lambda k: arms[k]["scores"]["digit"]["exact"],
            "isolated digits read correctly (50)",
            "exact match",
            1.0,
            "{:.2f}",
        ),
        (lambda k: agree[k], "transcripts equal to transformers' (of 61)", "clips", 61, "{:.0f}"),
    ]
    for ax, (value, title, xlabel, xmax, fmt) in zip(axes, panels, strict=True):
        values = [value(key) for key, _ in ARMS]
        ax.barh(range(len(ARMS)), values, color=colours)
        for index, v in enumerate(values):
            ax.text(v + xmax * 0.01, index, fmt.format(v), va="center", fontsize=8)
        ax.set_yticks(range(len(ARMS)), labels if ax is axes[0] else [], fontsize=8)
        ax.invert_yaxis()
        ax.set_xlim(0, xmax * 1.12)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlabel(xlabel)
    fig.suptitle(
        "candle's whisper example, tiny.en: two defects, each harmless-looking alone (blue), against transformers (gold)",
        fontsize=10,
    )
    fig.tight_layout()
    path = plotting.save_fig(fig, ROOT / "docs/writeups/figures/i/i_whisper_example_rules.png")
    print(path.relative_to(ROOT))


if __name__ == "__main__":
    main()
