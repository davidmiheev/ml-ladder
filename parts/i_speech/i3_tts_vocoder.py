"""I3. Text-to-speech: a mel-spectrogram predictor plus a vocoder, evaluated without listening

Lab reference: [torchaudio tutorials — pipelines, feature extraction and forced alignment](https://docs.pytorch.org/audio/stable/tutorials/)

Plan
----
- [ ] sketch the approach and the baseline it is measured against
- [ ] implement
- [ ] log metrics and at least one figure through the harness
- [ ] write down what surprised you in the run summary

Run with ``ladder run I3`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, experiment, todo


@experiment(
    "i3_tts_vocoder",
    # Placeholder flags — replace with the knobs this rung actually has.
    defaults={"epochs": 10, "lr": 1e-3, "batch_size": 128},
)
def run(exp: Experiment) -> dict[str, float]:
    """Text-to-speech: a mel-spectrogram predictor plus a vocoder, evaluated without listening"""
    todo("I3 — Text-to-speech: a mel-spectrogram predictor plus a vocoder, evaluated without listening")
    return {}


main = run

if __name__ == "__main__":
    raise SystemExit(main())
