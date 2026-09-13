"""D2. Masked autoencoder (MAE) on a small image dataset

Lab reference: [lightly-ai — contrastive-learning reference implementations](https://github.com/lightly-ai/lightly)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run D2`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "d2_masked_autoencoder",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """Masked autoencoder (MAE) on a small image dataset"""
    todo("D2 — Masked autoencoder (MAE) on a small image dataset")
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
