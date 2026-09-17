"""H1. Object detection from scratch — IoU, non-max suppression and mAP implemented by hand

Lab reference: [open-mmlab/mmocr — detection and recognition reference implementations](https://github.com/open-mmlab/mmocr)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run H1`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "h1_detection_from_scratch",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """Object detection from scratch — IoU, non-max suppression and mAP implemented by hand"""
    todo("H1 — Object detection from scratch — IoU, non-max suppression and mAP implemented by hand")
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
