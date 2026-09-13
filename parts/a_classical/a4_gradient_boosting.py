"""A4. Minimal gradient-boosting implementation, benchmarked against XGBoost

Lab reference: [Stanford CS229 problem sets](https://cs229.stanford.edu/)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run A4`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "a4_gradient_boosting",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """Minimal gradient-boosting implementation, benchmarked against XGBoost"""
    todo("A4 — Minimal gradient-boosting implementation, benchmarked against XGBoost")
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
