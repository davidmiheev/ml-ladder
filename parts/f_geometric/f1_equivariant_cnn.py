"""F1. Rotation-equivariant CNN on a rotated-MNIST task, compared against a standard CNN

Lab reference: [escnn — equivariant CNN library, and PyTorch Geometric tutorials](https://github.com/QUVA-Lab/escnn)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run F1`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "f1_equivariant_cnn",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """Rotation-equivariant CNN on a rotated-MNIST task, compared against a standard CNN"""
    todo("F1 — Rotation-equivariant CNN on a rotated-MNIST task, compared against a standard CNN")
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
