"""B2. MLP on MNIST — manual backprop, then repeat with PyTorch autograd and diff the gradients

Lab reference: [karpathy/micrograd — autodiff reference implementation](https://github.com/karpathy/micrograd)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run B2`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "b2_mlp_mnist",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """MLP on MNIST — manual backprop, then repeat with PyTorch autograd and diff the gradients"""
    todo("B2 — MLP on MNIST — manual backprop, then repeat with PyTorch autograd and diff the gradients")
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
