"""E4. Compare sample quality and training stability across VAE, GAN, and diffusion

Lab reference: [lucidrains/denoising-diffusion-pytorch — diffusion reference](https://github.com/lucidrains/denoising-diffusion-pytorch)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run E4`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "e4_generative_comparison",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """Compare sample quality and training stability across VAE, GAN, and diffusion"""
    todo("E4 — Compare sample quality and training stability across VAE, GAN, and diffusion")
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
