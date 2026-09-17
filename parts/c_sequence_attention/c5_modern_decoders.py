"""C5. Modern decoder architectures read from source — Qwen3.5, Gemma 4 and DeepSeek-V2 against C3

Lab reference: [Annotated Transformer / nanoGPT](https://nlp.seas.harvard.edu/annotated-transformer/)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run C5`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "c5_modern_decoders",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """Modern decoder architectures read from source — Qwen3.5, Gemma 4 and DeepSeek-V2 against C3"""
    todo("C5 — Modern decoder architectures read from source — Qwen3.5, Gemma 4 and DeepSeek-V2 against C3")
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
