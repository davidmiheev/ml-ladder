"""G3. Rust/Candle port of one transformer-block forward pass, as a comprehension exercise for [[ignite-catalog]]

Lab reference: [PyTorch mixed-precision & profiler docs](https://pytorch.org/docs/stable/notes/amp_examples.html)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run G3`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "g3_candle_transformer_port",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """Rust/Candle port of one transformer-block forward pass, as a comprehension exercise for [[ignite-catalog]]"""
    todo(
        "G3 — Rust/Candle port of one transformer-block forward pass, as a comprehension exercise for [[ignite-catalog]]"
    )
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
