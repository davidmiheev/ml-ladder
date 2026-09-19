"""I1. Audio front end from scratch — STFT, mel filterbank and MFCCs, against torchaudio's

Lab reference: [torchaudio tutorials — pipelines, feature extraction and forced alignment](https://docs.pytorch.org/audio/stable/tutorials/)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run I1`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "i1_audio_frontend",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """Audio front end from scratch — STFT, mel filterbank and MFCCs, against torchaudio's"""
    todo("I1 — Audio front end from scratch — STFT, mel filterbank and MFCCs, against torchaudio's")
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
