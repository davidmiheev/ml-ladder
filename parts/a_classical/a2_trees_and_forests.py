"""A2. Decision tree + random forest from scratch (Gini/entropy splits, bagging)

Lab reference: [Stanford CS229 problem sets](https://cs229.stanford.edu/)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run A2`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "a2_trees_and_forests",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """Decision tree + random forest from scratch (Gini/entropy splits, bagging)"""
    todo("A2 — Decision tree + random forest from scratch (Gini/entropy splits, bagging)")
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
