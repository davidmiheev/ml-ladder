"""D1. SimCLR-style contrastive pretraining on CIFAR-10

Lab reference: [lightly-ai — contrastive-learning reference implementations](https://github.com/lightly-ai/lightly)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run D1`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "d1_simclr_pretraining",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """SimCLR-style contrastive pretraining on CIFAR-10"""
    todo("D1 — SimCLR-style contrastive pretraining on CIFAR-10")
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
