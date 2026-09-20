r"""I3. Text-to-speech: a mel-spectrogram predictor plus a vocoder, evaluated without listening

Lab reference: [torchaudio tutorials — pipelines, feature extraction and forced alignment](https://docs.pytorch.org/audio/stable/tutorials/)

Synthesis has part E's problem in a harder form. A generative image model at
least has FID; the field's default for speech is a **mean opinion score** — a
listening test — which cannot be run in CI, cannot be reproduced, and is not
available to me at all. So the question this rung is really about is not how to
build a small TTS system. It is what you are allowed to conclude from one when
you cannot hear it.

The system is the standard two stages, because the split is the interesting
part:

1. **An acoustic model** maps text to a mel spectrogram. This is where prosody,
   duration and speaker identity live.
2. **A vocoder** maps the mel back to a waveform. Here it is Griffin-Lim, which
   is not learned at all: it iteratively guesses the phase that a magnitude
   spectrogram is missing.

That second stage is the reason this rung can exist on a CPU, and it is also a
*measurement* rather than an assumption — Griffin-Lim puts a ceiling on quality
that has nothing to do with the acoustic model, and the rung measures the
ceiling before it measures anything else.

**Three metrics, none of which is a listening test.**

- **Mel distance** to the reference. Cheap, and it correlates with nothing in
  particular: a synthesiser that produces a perfectly intelligible utterance
  with different timing scores badly, and one that produces a blurry average of
  the training set scores well. Reported because it is what the model optimises,
  and therefore says whether training worked — not whether the output is good.
- **An ASR round trip.** Feed the synthesised audio to the recogniser from I2
  and measure its error rate. This asks the only question that has an objective
  answer: *is the intended word recoverable from the audio?* Its weakness is
  that it is blind to everything ASR is invariant to, which includes most of
  what makes speech sound human.
- **A speaker/prosody check.** Duration error against the reference, and whether
  the output's spectral envelope is closer to the target speaker than to the
  others. This catches the failure mel distance hides: averaging every voice
  into one.

The honest summary is that these three cover intelligibility and coverage and
say nothing about naturalness, and this rung says so rather than implying its
numbers mean more than they do.

Run with ``ladder run I3`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, audio, experiment, plotting, torchkit, vision


def griffin_lim(
    magnitude: NDArray[np.float64], *, n_fft: int, hop: int, iterations: int = 64, rng=None
) -> NDArray[np.float32]:
    r"""Recover a waveform from a magnitude spectrogram by iterating on the phase.

    An STFT magnitude has thrown away the phase, and phase is most of what makes
    a waveform. Griffin-Lim starts from random phase and alternately projects
    onto the two constraint sets — "has the right magnitude" and "is the STFT of
    some real signal" — which converges to a local minimum of the distance
    between them:

    $$x_{k+1} = \operatorname{ISTFT}\!\left(|S| \cdot \frac{\operatorname{STFT}(x_k)}{|\operatorname{STFT}(x_k)|}\right).$$

    It is not a good vocoder and nothing here pretends otherwise. It is a
    *measurable* one: run it on the reference's own spectrogram and whatever
    quality is lost is the ceiling every downstream number inherits. Both
    constraint sets are non-convex, so the iteration converges to a fixed point
    that need not be the global one — on a pure tone the tests measure it
    plateauing at ~5% reconstruction error however long it runs.

    The output is peak-normalised, so the absolute level is not recovered. That
    is fine for a vocoder and fatal for a comparison that forgets it.
    """
    from parts.i_speech.i1_audio_frontend import hann_window

    generator = np.random.default_rng(rng)
    window = hann_window(n_fft)
    angles = np.exp(2j * np.pi * generator.random(magnitude.shape))
    spectrum = magnitude * angles
    signal = _istft(spectrum, window=window, hop=hop)
    for _ in range(iterations):
        rebuilt = _stft(signal, window=window, hop=hop, frames=magnitude.shape[1])
        angles = rebuilt / np.maximum(np.abs(rebuilt), 1e-12)
        signal = _istft(magnitude * angles, window=window, hop=hop)
    peak = np.abs(signal).max()
    return (signal / peak if peak > 0 else signal).astype(np.float32)


def _stft(signal: NDArray[np.float64], *, window: NDArray[np.float64], hop: int, frames: int):
    n_fft = len(window)
    padded = np.pad(signal, n_fft // 2, mode="reflect")
    indices = np.arange(n_fft)[None, :] + hop * np.arange(frames)[:, None]
    indices = np.clip(indices, 0, len(padded) - 1)
    return np.fft.rfft(padded[indices] * window[None, :], axis=1).T


def _istft(spectrum: NDArray[np.complex128], *, window: NDArray[np.float64], hop: int) -> NDArray[np.float64]:
    """Overlap-add with window-squared normalisation.

    Dividing by the summed squared window is what makes this an inverse rather
    than an approximation: overlap-add of windowed frames multiplies the signal
    by $\\sum_t w^2[n - t\\,H]$, and undoing that is the difference between a
    reconstruction and a signal with a periodic amplitude ripple at the hop rate.
    """
    n_fft = len(window)
    frames = np.fft.irfft(spectrum.T, n=n_fft, axis=1)
    length = hop * (frames.shape[0] - 1) + n_fft
    signal = np.zeros(length)
    weights = np.zeros(length)
    for index, row in enumerate(frames):
        start = index * hop
        signal[start : start + n_fft] += row * window
        weights[start : start + n_fft] += window**2
    signal = signal / np.maximum(weights, 1e-8)
    return signal[n_fft // 2 : len(signal) - n_fft // 2]


def make_acoustic_model(*, n_symbols: int, n_mels: int, hidden: int, frames: int):
    """Text to mel, by the smallest architecture that can express duration.

    An embedding of the whole word, then a GRU that unrolls it over a *fixed*
    number of output frames. Fixed because duration prediction is its own
    research problem and a ten-word vocabulary does not need it — but that is a
    choice with a cost, and the rung measures it as a duration error rather than
    leaving it in the architecture unremarked.
    """
    vision.require_torch()
    from torch import nn

    class Acoustic(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(n_symbols, hidden)
            self.recurrent = nn.GRU(hidden, hidden, num_layers=2, batch_first=True)
            self.head = nn.Linear(hidden, n_mels)
            self.frames = frames

        def forward(self, word_ids):
            context = self.embedding(word_ids).unsqueeze(1).expand(-1, self.frames, -1)
            hidden_states, _ = self.recurrent(context)
            return self.head(hidden_states).transpose(1, 2)  # (batch, mels, frames)

    return Acoustic


def mel_distance(a: NDArray[np.float64], b: NDArray[np.float64]) -> float:
    """Mean absolute difference of log-mels over the overlapping frames.

    Truncating to the shorter of the two rather than padding, because padding
    compares synthesised speech against silence and scores a short utterance as
    though it had got the ending wrong.
    """
    frames = min(a.shape[1], b.shape[1])
    return float(np.abs(a[:, :frames] - b[:, :frames]).mean())


def voiced_frames(log_mel: NDArray[np.float64], *, threshold: float = 0.25) -> int:
    """Frames whose energy exceeds a fraction of the clip's peak — a crude VAD.

    Crude on purpose: the duration metric only needs to know where the word is,
    and a trained voice-activity detector would be a second model with its own
    failure modes sitting inside a metric.
    """
    energy = log_mel.mean(axis=0)
    span = energy.max() - energy.min()
    if span <= 0:
        return 0
    return int(((energy - energy.min()) / span > threshold).sum())


@experiment(
    "i3_tts_vocoder",
    defaults={
        "per_combination": 20,
        "hidden": 128,
        "frames": 48,  # 384 ms at an 8 ms hop
        "epochs": 400,
        "lr": 3e-3,
        "griffin_lim_iterations": 64,
        "asr_epochs": 60,
        "asr_channels": 32,
        "asr_hidden": 128,
        "target_speaker": "george",
    },
    description="A digit synthesiser scored by mel distance, an ASR round trip and a duration check -- no listening.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Text-to-speech: a mel-spectrogram predictor plus a vocoder, evaluated without listening"""
    torch = vision.require_torch()
    device = torchkit.device()
    config = audio.FrontEndConfig()
    from parts.i_speech.i1_audio_frontend import log_compress, mel_filterbank, mel_spectrogram

    print(f"[i3] device: {device}, {exp['frames']} frames = {exp['frames'] * config.hop_ms:.0f} ms per utterance")

    clips = audio.load_fsdd(per_combination=exp["per_combination"])
    voice = clips.select(np.asarray(clips.speakers) == exp["target_speaker"])
    summary: dict[str, float] = {"corpus/clips": float(len(voice)), "corpus/frames": float(exp["frames"])}
    print(f"[i3] training on {len(voice)} clips from one speaker ({exp['target_speaker']})")

    def padded_log_mel(waveform) -> NDArray[np.float64]:
        mel = log_compress(mel_spectrogram(waveform, config=config))
        out = np.full((config.n_mels, exp["frames"]), mel.min(), dtype=np.float64)
        width = min(mel.shape[1], exp["frames"])
        out[:, :width] = mel[:, :width]
        return out

    targets = np.stack([padded_log_mel(w) for w in voice.waveforms])
    word_ids = voice.digits.copy()

    # -- the vocoder's ceiling, measured before anything is trained ----------
    # Griffin-Lim on the *reference's own* magnitudes. Whatever it loses here is
    # inherited by every number below, so it is established first.
    filters = mel_filterbank(
        n_fft=config.n_fft, n_mels=config.n_mels, rate=config.rate, fmin=config.fmin, fmax=config.fmax
    )
    pseudo_inverse = np.linalg.pinv(filters.T)

    def vocode(log_mel: NDArray[np.float64]) -> NDArray[np.float32]:
        """log-mel -> linear magnitude -> Griffin-Lim.

        The mel filterbank is not invertible — 40 bands from 129 bins — so the
        pseudo-inverse is a least-squares guess at the spectrum that produced
        them. That approximation sits *underneath* Griffin-Lim's phase problem,
        and the ceiling measured here is both of them together.
        """
        power = np.maximum(pseudo_inverse @ np.exp(log_mel), 0.0)
        return griffin_lim(
            np.sqrt(power), n_fft=config.n_fft, hop=config.hop, iterations=exp["griffin_lim_iterations"], rng=exp.seed
        )

    ceiling = []
    for waveform in voice.waveforms[:20]:
        reference = log_compress(mel_spectrogram(waveform, config=config))
        rebuilt = log_compress(mel_spectrogram(vocode(reference), config=config))
        ceiling.append(mel_distance(rebuilt, reference))
    summary["vocoder/ceiling_mel_distance"] = float(np.mean(ceiling))
    print(
        f"[i3] vocoder ceiling: re-analysing Griffin-Lim's output of a *real* clip already costs "
        f"{summary['vocoder/ceiling_mel_distance']:.3f} mel distance -- no synthesiser can score below this"
    )

    # -- train the acoustic model ---------------------------------------------
    torchkit.seed_torch(exp.seed)
    model = make_acoustic_model(n_symbols=10, n_mels=config.n_mels, hidden=exp["hidden"], frames=exp["frames"])().to(
        device
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"])
    ids = torch.from_numpy(word_ids).to(device)
    mel_targets = torch.from_numpy(targets.astype(np.float32)).to(device)
    curve: list[float] = []
    for _ in range(exp["epochs"]):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        loss = torch.nn.functional.l1_loss(model(ids), mel_targets)
        loss.backward()
        optimizer.step()
        curve.append(float(loss.item()))
    summary["acoustic/train_l1"] = curve[-1]
    print(f"[i3] acoustic model trained: L1 {curve[0]:.3f} -> {curve[-1]:.3f}")

    # -- synthesise every digit ------------------------------------------------
    model.eval()
    with torch.no_grad():
        synthesised = model(torch.arange(10, device=device)).cpu().numpy().astype(np.float64)
    waveforms = [vocode(synthesised[digit]) for digit in range(10)]

    # -- metric 1: mel distance, against two baselines ------------------------
    references = {digit: targets[word_ids == digit].mean(axis=0) for digit in range(10)}
    distances = [mel_distance(synthesised[digit], references[digit]) for digit in range(10)]
    summary["mel/distance"] = float(np.mean(distances))
    # The control that makes it readable: predict the *same* average spectrogram
    # for every word. A synthesiser that beats this has learned which word it
    # was asked for; one that does not has learned the speaker's average voice.
    grand_mean = targets.mean(axis=0)
    summary["mel/constant_baseline"] = float(np.mean([mel_distance(grand_mean, references[d]) for d in range(10)]))
    summary["mel/gain_over_constant"] = summary["mel/constant_baseline"] - summary["mel/distance"]
    # And the other end: how far apart two real recordings of the same word are.
    within = []
    for digit in range(10):
        members = targets[word_ids == digit]
        if len(members) >= 2:
            within.extend(mel_distance(members[i], members[i + 1]) for i in range(len(members) - 1))
    summary["mel/within_word_spread"] = float(np.mean(within)) if within else float("nan")
    print(
        f"[i3] mel distance {summary['mel/distance']:.3f}; a constant spectrogram scores "
        f"{summary['mel/constant_baseline']:.3f} and two real recordings of the same digit differ by "
        f"{summary['mel/within_word_spread']:.3f} -- the metric's own noise floor"
    )

    # -- metric 2: the ASR round trip ----------------------------------------
    from parts.i_speech.i2_asr_ctc_whisper import (
        character_error_rate,
        prepare,
        train_ctc,
        transcribe,
    )

    train_clips, validation_clips, _ = audio.split_by_speaker(clips)
    asr_exp = Experiment(
        name="i3-asr",
        config={
            "channels": exp["asr_channels"],
            "hidden": exp["asr_hidden"],
            "epochs": exp["asr_epochs"],
            "batch_size": 32,
            "lr": 3e-3,
            "eval_every": 10,
        },
        run_dir=exp.run_dir,
        tracker=exp.tracker,
        seed=exp.seed,
    )
    train_split = prepare(train_clips, config=config)
    mean, std = train_split[-2], train_split[-1]
    validation_split = (
        *prepare(validation_clips, config=config, mean=mean, std=std)[:4],
        validation_clips.transcripts,
    )
    recognizer, _, asr_val_cer, _ = train_ctc(train_split, validation_split, exp=asr_exp, config=config, device=device)
    summary["roundtrip/recognizer_val_cer"] = asr_val_cer

    synthetic_clips = audio.Clips(
        waveforms=waveforms, digits=np.arange(10, dtype=np.int64), speakers=["synthetic"] * 10, rate=config.rate
    )
    synthetic_split = (*prepare(synthetic_clips, config=config, mean=mean, std=std)[:4], synthetic_clips.transcripts)
    heard = transcribe(recognizer, synthetic_split, device=device)
    summary["roundtrip/cer"] = character_error_rate(heard, synthetic_clips.transcripts)
    summary["roundtrip/exact"] = float(
        np.mean([p == t for p, t in zip(heard, synthetic_clips.transcripts, strict=True)])
    )
    # The control: the same recognizer on *real* audio from the same speaker,
    # vocoded the same way. Anything worse than this is the synthesiser's fault;
    # anything at this level is Griffin-Lim's.
    vocoded_real = audio.Clips(
        waveforms=[vocode(log_compress(mel_spectrogram(w, config=config))) for w in voice.waveforms[:10]],
        digits=voice.digits[:10],
        speakers=["vocoded"] * 10,
        rate=config.rate,
    )
    vocoded_split = (*prepare(vocoded_real, config=config, mean=mean, std=std)[:4], vocoded_real.transcripts)
    summary["roundtrip/vocoded_real_cer"] = character_error_rate(
        transcribe(recognizer, vocoded_split, device=device), vocoded_real.transcripts
    )
    print(
        f"[i3] round trip: the recognizer reads the synthesiser at CER {summary['roundtrip/cer']:.3f} "
        f"(exact {summary['roundtrip/exact']:.1f}); it reads *real* audio through the same vocoder at "
        f"CER {summary['roundtrip/vocoded_real_cer']:.3f}, and held-out human speech at {asr_val_cer:.3f}"
    )
    for digit, text in enumerate(heard):
        summary[f"roundtrip/{audio.DIGIT_WORDS[digit]}/correct"] = float(text == audio.DIGIT_WORDS[digit])

    # -- metric 3: duration and speaker ---------------------------------------
    predicted_frames = [voiced_frames(synthesised[digit]) for digit in range(10)]
    reference_frames = [voiced_frames(references[digit]) for digit in range(10)]
    errors = [abs(p - r) * config.hop_ms for p, r in zip(predicted_frames, reference_frames, strict=True)]
    summary["duration/mean_error_ms"] = float(np.mean(errors))
    summary["duration/reference_mean_ms"] = float(np.mean(reference_frames) * config.hop_ms)
    summary["duration/relative_error"] = summary["duration/mean_error_ms"] / max(
        summary["duration/reference_mean_ms"], 1e-9
    )
    print(
        f"[i3] duration: {summary['duration/mean_error_ms']:.0f} ms mean error against a "
        f"{summary['duration/reference_mean_ms']:.0f} ms reference "
        f"({summary['duration/relative_error'] * 100:.0f}%)"
    )

    envelopes = {}
    for speaker in sorted(set(clips.speakers)):
        member = clips.select(np.asarray(clips.speakers) == speaker)
        envelopes[speaker] = np.mean(
            [log_compress(mel_spectrogram(w, config=config)).mean(axis=1) for w in member.waveforms], axis=0
        )
    synthetic_envelope = np.mean(
        [log_compress(mel_spectrogram(w, config=config)).mean(axis=1) for w in waveforms], axis=0
    )
    gaps = {speaker: float(np.abs(synthetic_envelope - envelope).mean()) for speaker, envelope in envelopes.items()}
    closest = min(gaps, key=gaps.get)
    summary["speaker/matches_target"] = float(closest == exp["target_speaker"])
    summary["speaker/gap_to_target"] = gaps[exp["target_speaker"]]
    summary["speaker/gap_to_nearest_other"] = min(v for k, v in gaps.items() if k != exp["target_speaker"])
    summary["speaker/margin"] = summary["speaker/gap_to_nearest_other"] - summary["speaker/gap_to_target"]
    print(
        f"[i3] speaker: the synthesised voice is closest to '{closest}' "
        f"(trained on '{exp['target_speaker']}'), margin {summary['speaker/margin']:+.3f}"
    )

    # -- figures ---------------------------------------------------------------
    exp.log_figure(
        "training",
        plotting.plot_curves({"L1 to target mel": curve}, title="Acoustic model", xlabel="epoch", ylabel="L1"),
    )
    exp.log_figure(
        "synthesis",
        plotting.plot_spectrograms(
            {
                'reference "seven"': references[7],
                'synthesised "seven"': synthesised[7],
                "after Griffin-Lim": log_compress(mel_spectrogram(waveforms[7], config=config)),
            },
            title="Target, prediction, and what survives the vocoder",
            rate=config.rate,
            hop=config.hop,
        ),
    )
    exp.log_figure(
        "metrics",
        plotting.plot_comparison(
            {
                "two real clips": summary["mel/within_word_spread"],
                "the synthesiser": summary["mel/distance"],
                "one constant mel": summary["mel/constant_baseline"],
            },
            title="Mel distance, with the floor and the ceiling that make it readable",
            ylabel="mel distance",
        ),
    )
    exp.log_figure(
        "roundtrip",
        plotting.plot_comparison(
            {
                "human speech": summary["roundtrip/recognizer_val_cer"],
                "real, vocoded": summary["roundtrip/vocoded_real_cer"],
                "synthesised": summary["roundtrip/cer"],
            },
            title="What the recognizer makes of each -- the vocoder's share is the middle bar",
            ylabel="CER",
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
