r"""I2. Speech recognition with CTC on a small corpus, and the same audio through Whisper

Lab reference: [torchaudio tutorials — pipelines, feature extraction and forced alignment](https://docs.pytorch.org/audio/stable/tutorials/)

Rung H3 built a CTC recogniser for text lines. The loss, the decoders and the
error metrics all transfer to speech unchanged, which is the point of putting
this rung after that one: **CTC does not know what a frame is.** It needs a
monotonic alignment between an input sequence and a shorter output sequence, and
whether the input is image columns or 8 ms audio frames is not its concern.

What does not transfer is the split. An image dataset can be shuffled; a speech
dataset cannot, because the thing a model most easily memorises is a *voice*.
Every speaker here says each digit fifty times, so a random split hands the
model forty-nine examples of this person saying "seven" and asks about the
fiftieth. That number is much better than the one anyone wants, and this rung
measures both so the size of the leak is on the record rather than in a caveat.

The pretrained baseline is Whisper, and it is not here to win. It is here to say
what the small model's errors are *made of*: the same audio, a model three
orders of magnitude larger trained on 680k hours, and the gap between them is
the part attributable to scale rather than to anything this rung did. It also
carries a constraint worth measuring — Whisper expects 16 kHz and this corpus is
8 kHz, so everything above 4 kHz in its input is interpolation, and a model
trained on wideband speech is being handed a band-limited signal.

Run with ``ladder run I2`` (``--help`` for flags).
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, audio, experiment, plotting, torchkit, vision

#: Blank is index 0, every letter shifted up by one — torch's CTCLoss default.
BLANK = 0
ALPHABET = "".join(audio.ALPHABET)


def encode(text: str) -> list[int]:
    return [ALPHABET.index(character) + 1 for character in text]


def log_mel_features(waveform, *, config: audio.FrontEndConfig) -> NDArray[np.float32]:
    """The front end from I1, reused verbatim.

    Not torchaudio's, and not a second implementation: I1 established that the
    two agree to 1e-06, so using the hand-written one here keeps the whole part
    on one front end whose conventions are written down — which is what makes
    I3's round-trip metric comparable with this rung's error rates.
    """
    from parts.i_speech.i1_audio_frontend import log_compress, mel_spectrogram

    return log_compress(mel_spectrogram(waveform, config=config)).astype(np.float32)


def normalize(features: NDArray[np.float32], mean, std, *, mode: str = "global"):
    """Per-mel-band standardisation, by corpus statistics or by utterance.

    An earlier version of this rung did only the first and justified it in
    writing: per-utterance normalisation "would quietly remove exactly the
    speaker differences this rung is trying to measure generalisation across".
    That reasoning is backwards. The rung is not trying to *measure* speaker
    differences, it is trying to generalise across them, and subtracting each
    utterance's own long-term spectrum is the oldest trick in speech recognition
    for exactly that: **cepstral mean normalisation** removes the channel and
    the speaker's average vocal-tract shape, which are nuisance, and leaves the
    variation over time, which is the signal.

    So it is an arm rather than a decision, and the sweep says which is right.
    """
    if mode == "utterance":
        # The clip's own statistics: what a streaming system can compute from
        # the utterance it is currently holding, and nothing else.
        mean = features.mean(axis=1)
        std = features.std(axis=1) + 1e-5
    return (features - mean[:, None]) / std[:, None]


def make_speech_crnn(*, n_mels: int, num_symbols: int, channels: int, hidden: int):
    """Conv over frequency, then a BiGRU along time.

    The convolution is 2-D but deliberately strides only in frequency: pooling
    along time would shorten the sequence CTC aligns against, and at 8 ms per
    frame a 400 ms digit is already only ~50 frames against a 5-character
    target. CTC needs input length >= output length, and halving the time axis
    twice would make "three" unrepresentable in a short clip.
    """
    vision.require_torch()
    from torch import nn

    class SpeechCRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.MaxPool2d((2, 1)),  # frequency only
                nn.Conv2d(channels, channels * 2, 3, padding=1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(),
                nn.MaxPool2d((2, 1)),
            )
            reduced = n_mels // 4
            self.recurrent = nn.GRU(
                channels * 2 * reduced, hidden, num_layers=2, bidirectional=True, batch_first=True, dropout=0.1
            )
            self.head = nn.Linear(2 * hidden, num_symbols)

        def forward(self, x):  # x: (batch, mels, frames)
            h = self.features(x.unsqueeze(1))
            batch, channel, mel, frames = h.shape
            h = h.permute(0, 3, 1, 2).reshape(batch, frames, channel * mel)
            h, _ = self.recurrent(h)
            return self.head(h)  # (batch, frames, symbols)

    return SpeechCRNN


def prepare(clips: audio.Clips, *, config: audio.FrontEndConfig, mean=None, std=None, mode: str = "global"):
    """Features, targets and both length vectors for a whole split."""
    features = [log_mel_features(w, config=config) for w in clips.waveforms]
    if mean is None or std is None:
        stacked = np.concatenate(features, axis=1)
        mean, std = stacked.mean(axis=1), stacked.std(axis=1) + 1e-5
    features = [normalize(f, mean, std, mode=mode) for f in features]
    lengths = np.asarray([f.shape[1] for f in features], dtype=np.int64)
    padded = np.zeros((len(features), config.n_mels, int(lengths.max())), dtype=np.float32)
    for row, item in enumerate(features):
        padded[row, :, : item.shape[1]] = item
    targets = [encode(text) for text in clips.transcripts]
    target_lengths = np.asarray([len(t) for t in targets], dtype=np.int64)
    flat = np.asarray([symbol for target in targets for symbol in target], dtype=np.int64)
    return padded, lengths, flat, target_lengths, mean, std


def train_ctc(train, validation, *, exp: Experiment, config: audio.FrontEndConfig, device, progress: str = ""):
    """Minibatch training loop with validation-CER model selection.

    ``progress`` names the arm in the per-eval log line. A speech rung on CPU
    runs for long enough that a silent loop is indistinguishable from a hung
    one, which cost twenty minutes of not knowing.
    """
    torch = vision.require_torch()

    torchkit.seed_torch(exp.seed)
    model = make_speech_crnn(
        n_mels=config.n_mels, num_symbols=len(ALPHABET) + 1, channels=exp["channels"], hidden=exp["hidden"]
    )().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=1e-4)
    loss_fn = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)

    x, lengths, targets, target_lengths, mean, std = train
    inputs = torch.from_numpy(x).to(device)
    input_lengths = torch.from_numpy(lengths)
    target_length_tensor = torch.from_numpy(target_lengths)

    rng = np.random.default_rng(exp.seed)
    order = np.arange(len(lengths))
    curve: list[float] = []
    started = time.time()
    best_state, best_cer = None, float("inf")
    for epoch in range(exp["epochs"]):
        model.train()
        rng.shuffle(order)
        total = 0.0
        batches = 0
        for start in range(0, len(order), exp["batch_size"]):
            index = order[start : start + exp["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs[index])
            log_probabilities = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
            # CTC needs the *unpadded* input length per item, or the padding
            # becomes alignable frames and the model learns to emit into silence.
            offsets = np.concatenate([[0], np.cumsum(target_lengths)])
            flat = np.concatenate([targets[offsets[i] : offsets[i + 1]] for i in index]) if len(index) else targets
            loss = loss_fn(
                log_probabilities,
                torch.from_numpy(flat).to(device),
                input_lengths[index].to(device),
                target_length_tensor[index].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total += float(loss.item())
            batches += 1
        curve.append(total / max(batches, 1))

        if (epoch + 1) % exp["eval_every"] == 0 or epoch == exp["epochs"] - 1:
            predictions = transcribe(model, validation, device=device)
            cer = character_error_rate(predictions, validation[-1])
            if cer < best_cer:
                best_cer, best_state = cer, {k: v.detach().clone() for k, v in model.state_dict().items()}
            if progress:
                print(
                    f"[{progress}] epoch {epoch + 1:>3}/{exp['epochs']}  loss {curve[-1]:.3f}  "
                    f"val CER {cer:.3f}  ({time.time() - started:.0f}s)",
                    flush=True,
                )
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, curve, best_cer, (mean, std)


def transcribe(model, split, *, device, beam_width: int = 0) -> list[str]:
    """Greedy or beam CTC decode for a whole split."""
    from parts.h_computer_vision.h3_crnn_ctc import beam_decode, greedy_decode

    torch = vision.require_torch()
    x, lengths = split[0], split[1]
    model.eval()
    outputs: list[str] = []
    with torch.no_grad():
        logits = model(torch.from_numpy(x).to(device))
        log_probabilities = torch.log_softmax(logits, dim=-1).cpu().numpy()
    for row, length in enumerate(lengths):
        frames = log_probabilities[row, : int(length)]
        outputs.append(
            beam_decode(frames, beam_width=beam_width, alphabet=ALPHABET)
            if beam_width
            else greedy_decode(frames, alphabet=ALPHABET)
        )
    return outputs


def character_error_rate(predictions: list[str], truths: list[str]) -> float:
    from parts.h_computer_vision.h3_crnn_ctc import character_error_rate as cer

    return cer(predictions, truths)


def word_error_rate(predictions: list[str], truths: list[str]) -> float:
    from parts.h_computer_vision.h3_crnn_ctc import word_error_rate as wer

    return wer(predictions, truths)


#: Orthographic variants that are the *same sounds* as a digit word. A speech
#: model that emits "too" for "two" has done its acoustic job and failed a
#: spelling test that only a language model can pass, so scoring both ways
#: separates the two failures instead of charging the first for the second.
HOMOPHONES = {
    "too": "two",
    "to": "two",
    "for": "four",
    "fore": "four",
    "won": "one",
    "ate": "eight",
    "ait": "eight",
    "sicks": "six",
    "tree": "three",
    "free": "three",
    "nein": "nine",
    "zeero": "zero",
}


def normalize_homophones(text: str) -> str:
    """Map a known homophone of a digit onto the digit's spelling."""
    return HOMOPHONES.get(text, text)


def whisper_transcribe(clips: audio.Clips, *, model_name: str, config: audio.FrontEndConfig, limit: int):
    """The pretrained baseline, on the same clips, resampled to what it expects.

    Returns ``None`` when transformers or the weights are unavailable, so the
    rung degrades to "the baseline did not run" rather than to silence.
    """
    try:
        import torch
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
    except ImportError:  # pragma: no cover - optional dependency
        return None
    try:
        processor = WhisperProcessor.from_pretrained(model_name)
        model = WhisperForConditionalGeneration.from_pretrained(model_name)
    except Exception as error:  # pragma: no cover - network-dependent
        print(f"[i2] whisper unavailable ({type(error).__name__}: {error}); baseline skipped")
        return None
    model.eval()

    outputs: list[str] = []
    for waveform in clips.waveforms[:limit]:
        wideband = audio.resample_poly(waveform, config.rate, 16000)
        inputs = processor(wideband, sampling_rate=16000, return_tensors="pt")
        with torch.no_grad():
            ids = model.generate(inputs.input_features, max_new_tokens=16, language="en", task="transcribe")
        text = processor.batch_decode(ids, skip_special_tokens=True)[0]
        outputs.append("".join(character for character in text.lower() if character.isalpha()))
    return outputs


@experiment(
    "i2_asr_ctc_whisper",
    defaults={
        "per_combination": 20,
        "channels": 32,
        "hidden": 128,
        "epochs": 40,
        "batch_size": 32,
        "lr": 3e-3,
        "eval_every": 5,
        "beam_width": 8,
        "normalizations": ["global", "utterance"],
        "whisper": "openai/whisper-tiny",
        "whisper_clips": 40,
    },
    description="A CTC speech recogniser on spoken digits, split by speaker, against Whisper on the same audio.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Speech recognition with CTC on a small corpus, and the same audio through Whisper"""
    vision.require_torch()
    device = torchkit.device()
    config = audio.FrontEndConfig()
    print(f"[i2] device: {device}, alphabet '{ALPHABET}' ({len(ALPHABET)} symbols + blank)")

    clips = audio.load_fsdd(per_combination=exp["per_combination"])
    train_clips, validation_clips, test_clips = audio.split_by_speaker(clips)
    summary: dict[str, float] = {
        "corpus/clips": float(len(clips)),
        "corpus/train": float(len(train_clips)),
        "corpus/test": float(len(test_clips)),
        "corpus/train_speakers": float(len(set(train_clips.speakers))),
    }
    print(
        f"[i2] {len(clips)} clips: train {len(train_clips)} ({sorted(set(train_clips.speakers))}), "
        f"val {len(validation_clips)} ({sorted(set(validation_clips.speakers))}), "
        f"test {len(test_clips)} ({sorted(set(test_clips.speakers))})"
    )

    # -- normalisation, which an earlier version of this rung decided by
    # -- argument and got wrong. Both arms, same everything else.
    arms = {}
    for mode in exp["normalizations"]:
        split = prepare(train_clips, config=config, mode=mode)
        arm_mean, arm_std = split[-2], split[-1]
        arm_validation = (
            *prepare(validation_clips, config=config, mean=arm_mean, std=arm_std, mode=mode)[:4],
            validation_clips.transcripts,
        )
        arm_test = (
            *prepare(test_clips, config=config, mean=arm_mean, std=arm_std, mode=mode)[:4],
            test_clips.transcripts,
        )
        arm_model, arm_curve, arm_cer, _ = train_ctc(
            split, arm_validation, exp=exp, config=config, device=device, progress=f"i2 {mode}"
        )
        arm_predictions = transcribe(arm_model, arm_test, device=device)
        arms[mode] = {
            "model": arm_model,
            "curve": arm_curve,
            "val_cer": arm_cer,
            "test_cer": character_error_rate(arm_predictions, test_clips.transcripts),
            "split": split,
            "test": arm_test,
            "validation": arm_validation,
            "stats": (arm_mean, arm_std),
        }
        summary[f"normalization/{mode}/val_cer"] = arm_cer
        summary[f"normalization/{mode}/test_cer"] = arms[mode]["test_cer"]
        print(f"[i2] normalisation '{mode}': val CER {arm_cer:.3f}, test CER {arms[mode]['test_cer']:.3f}")

    best_mode = min(arms, key=lambda mode: arms[mode]["val_cer"])
    summary["normalization/best_is_utterance"] = float(best_mode == "utterance")
    if len(arms) > 1:
        other = next(mode for mode in arms if mode != best_mode)
        summary["normalization/gain"] = arms[other]["test_cer"] - arms[best_mode]["test_cer"]
        if abs(summary["normalization/gain"]) < 1e-9:
            print(f"[i2] the two normalisations are indistinguishable on this budget ({best_mode} chosen on val)")
        else:
            print(
                f"[i2] '{best_mode}' wins by {summary['normalization/gain']:.3f} test CER, "
                f"with everything else held fixed"
            )

    chosen = arms[best_mode]
    model, curve = chosen["model"], chosen["curve"]
    test = chosen["test"]
    summary["ctc/normalization"] = float(best_mode == "utterance")
    summary["ctc/val_cer"] = chosen["val_cer"]
    summary["ctc/train_loss"] = curve[-1]

    greedy = transcribe(model, test, device=device)
    beam = transcribe(model, test, device=device, beam_width=exp["beam_width"])
    truths = test_clips.transcripts
    for name, predictions in (("greedy", greedy), ("beam", beam)):
        summary[f"ctc/{name}/cer"] = character_error_rate(predictions, truths)
        summary[f"ctc/{name}/wer"] = word_error_rate(predictions, truths)
        summary[f"ctc/{name}/exact"] = float(np.mean([p == t for p, t in zip(predictions, truths, strict=True)]))
        print(
            f"[i2] CTC {name:<6} CER {summary[f'ctc/{name}/cer']:.3f}  WER {summary[f'ctc/{name}/wer']:.3f}  "
            f"exact {summary[f'ctc/{name}/exact']:.3f}"
        )
    summary["ctc/beam_gain_cer"] = summary["ctc/greedy/cer"] - summary["ctc/beam/cer"]

    # -- the leak: what a shuffled split would have reported -------------------
    # Same model, same budget, same number of clips; the only change is that the
    # test voices are now also training voices.
    rng = np.random.default_rng(exp.seed)
    order = rng.permutation(len(clips))
    cut = len(train_clips)
    shuffled_train = clips.select(np.isin(np.arange(len(clips)), order[:cut]))
    shuffled_test = clips.select(np.isin(np.arange(len(clips)), order[cut : cut + len(test_clips)]))
    shuffled_validation = clips.select(
        np.isin(np.arange(len(clips)), order[cut + len(test_clips) : cut + len(test_clips) + len(validation_clips)])
    )
    leaky_train = prepare(shuffled_train, config=config, mode=best_mode)
    leaky_mean, leaky_std = leaky_train[-2], leaky_train[-1]
    leaky_validation = (
        *prepare(shuffled_validation, config=config, mean=leaky_mean, std=leaky_std, mode=best_mode)[:4],
        shuffled_validation.transcripts,
    )
    leaky_test = (
        *prepare(shuffled_test, config=config, mean=leaky_mean, std=leaky_std, mode=best_mode)[:4],
        shuffled_test.transcripts,
    )
    leaky_model, _, _, _ = train_ctc(
        leaky_train, leaky_validation, exp=exp, config=config, device=device, progress="i2 shuffled"
    )
    leaky_predictions = transcribe(leaky_model, leaky_test, device=device)
    summary["leak/shuffled_cer"] = character_error_rate(leaky_predictions, shuffled_test.transcripts)
    summary["leak/speaker_held_out_cer"] = summary["ctc/greedy/cer"]
    summary["leak/ratio"] = summary["leak/speaker_held_out_cer"] / max(summary["leak/shuffled_cer"], 1e-9)
    print(
        f"[i2] the same model on a shuffled split: CER {summary['leak/shuffled_cer']:.3f} against "
        f"{summary['leak/speaker_held_out_cer']:.3f} speaker-held-out -- "
        f"{summary['leak/ratio']:.1f}x, and only the split changed"
    )

    # -- the pretrained baseline ----------------------------------------------
    whisper = whisper_transcribe(test_clips, model_name=exp["whisper"], config=config, limit=exp["whisper_clips"])
    summary["whisper/available"] = float(whisper is not None)
    if whisper is not None:
        subset = truths[: len(whisper)]
        summary["whisper/cer"] = character_error_rate(whisper, subset)
        summary["whisper/wer"] = word_error_rate(whisper, subset)
        summary["whisper/exact"] = float(np.mean([p == t for p, t in zip(whisper, subset, strict=True)]))
        summary["whisper/clips"] = float(len(whisper))
        matched = character_error_rate(greedy[: len(whisper)], subset)
        summary["whisper/ctc_on_same_clips_cer"] = matched
        summary["whisper/advantage_cer"] = matched - summary["whisper/cer"]
        print(
            f"[i2] whisper-tiny on {len(whisper)} of the same clips: CER {summary['whisper/cer']:.3f}, "
            f"WER {summary['whisper/wer']:.3f}, exact {summary['whisper/exact']:.3f} "
            f"(the CTC model scores CER {matched:.3f} on those clips)"
        )

        # How much of that error is spelling rather than hearing.
        folded = [normalize_homophones(text) for text in whisper]
        summary["whisper/cer_homophone_folded"] = character_error_rate(folded, subset)
        summary["whisper/exact_homophone_folded"] = float(
            np.mean([p == t for p, t in zip(folded, subset, strict=True)])
        )
        summary["whisper/homophone_share"] = float(
            np.mean([f == t and w != t for f, w, t in zip(folded, whisper, subset, strict=True)])
        )
        print(
            f"[i2] folding digit homophones (two/too, four/for, one/won): exact rises "
            f"{summary['whisper/exact']:.3f} -> {summary['whisper/exact_homophone_folded']:.3f}; "
            f"{summary['whisper/homophone_share'] * 100:.0f}% of clips were heard correctly and spelled wrong"
        )

    # -- figures ---------------------------------------------------------------
    exp.log_figure(
        "training",
        plotting.plot_curves({"CTC loss": curve}, title="CTC training loss", xlabel="epoch", ylabel="loss"),
    )
    scores = {"CTC greedy": summary["ctc/greedy/cer"], "CTC beam": summary["ctc/beam/cer"]}
    if whisper is not None:
        scores["whisper-tiny"] = summary["whisper/cer"]
    exp.log_figure(
        "error_rates",
        plotting.plot_comparison(scores, title="Character error rate on held-out speakers", ylabel="CER"),
    )
    exp.log_figure(
        "leak",
        plotting.plot_comparison(
            {"speaker held out": summary["leak/speaker_held_out_cer"], "clips shuffled": summary["leak/shuffled_cer"]},
            title="The same model, the same clips, a different split",
            ylabel="CER",
        ),
    )
    # What CTC actually learned: the posterior over symbols against time.
    torch = vision.require_torch()
    model.eval()
    with torch.no_grad():
        probabilities = torch.softmax(model(torch.from_numpy(test[0][:1]).to(device)), dim=-1)[0].cpu().numpy().T
    exp.log_figure(
        "alignment",
        plotting.plot_spectrograms(
            {"log mel": test[0][0], f'p(symbol | t), "{truths[0]}"': probabilities[:, : int(test[1][0])]},
            title="CTC learns an alignment nobody supplied",
            rate=config.rate,
            hop=config.hop,
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
