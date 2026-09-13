"""C3. Transformer from scratch (attention, positional encoding, causal masking)

Lab reference: [Annotated Transformer / nanoGPT](https://nlp.seas.harvard.edu/annotated-transformer/)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run C3`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "c3_transformer_from_scratch",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """Transformer from scratch (attention, positional encoding, causal masking)"""
    todo("C3 — Transformer from scratch (attention, positional encoding, causal masking)")
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
