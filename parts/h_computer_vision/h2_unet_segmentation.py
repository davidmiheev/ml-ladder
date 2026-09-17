"""H2. Semantic segmentation with a U-Net — skip connections, Dice vs cross-entropy loss

Lab reference: [open-mmlab/mmocr — detection and recognition reference implementations](https://github.com/open-mmlab/mmocr)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run H2`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "h2_unet_segmentation",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """Semantic segmentation with a U-Net — skip connections, Dice vs cross-entropy loss"""
    todo("H2 — Semantic segmentation with a U-Net — skip connections, Dice vs cross-entropy loss")
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
