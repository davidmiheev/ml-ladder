r"""I4. Streaming and latency — chunked inference against full-utterance, with the accuracy it costs

Lab reference: [torchaudio tutorials — pipelines, feature extraction and forced alignment](https://docs.pytorch.org/audio/stable/tutorials/)

Every number in I2 was computed with the whole utterance in hand. That is a
reasonable way to measure a model and a useless way to deploy one: a system that
answers after you have finished speaking has a latency equal to the length of
what you said, and a system that answers *while* you speak has to commit to
frames it cannot yet see the end of.

This rung is where speech stops being sequence modelling and becomes the systems
problem part G was accounting for. The quantity is not accuracy and it is not
latency; it is the **exchange rate between them**, and it has a shape worth
measuring rather than assuming.

**What makes it a trade.** I2's recogniser is *bidirectional* — its GRU reads
each utterance forwards and backwards, so the representation of frame $t$
depends on every frame after it. That is free at evaluation time and impossible
in a stream. Three things can be done about it, and the rung measures all three:

1. **Chunk the bidirectional model anyway**, running it independently on
   windows of $C$ frames. Cheap, wrong at every boundary, and the error is a
   direct measurement of how much context the model was using.
2. **Allow lookahead**: emit frame $t$ only after $L$ further frames have
   arrived. Latency becomes $L$ frames, and the question is how much of the
   bidirectional advantage comes back per frame of delay.
3. **Train a causal model** whose GRU only reads forwards. It pays a fixed
   accuracy cost and has no latency beyond the frame itself, which is the
   design real streaming systems actually use.

Algorithmic latency is reported separately from wall-clock, because they are
different claims: the first is a property of the architecture and reproduces
anywhere, the second is a property of this CPU on this day.

Run with ``ladder run I4`` (``--help`` for flags).
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, audio, experiment, plotting, torchkit, vision


def chunked_logits(model, features: NDArray[np.float32], *, chunk: int, lookahead: int, device):
    """Run a model over a stream, emitting ``chunk`` frames at a time.

    For each output block the model sees the block plus ``lookahead`` frames
    beyond it, then the lookahead is **discarded** — it exists to inform the
    block, not to be emitted. That is what a streaming system does, and it is
    why lookahead costs latency rather than compute.

    Left context is *not* truncated here: each window starts from the beginning
    of the utterance. That makes this an upper bound on chunked accuracy and
    isolates the variable the rung is about, which is how much of the model's
    advantage came from seeing the *future*.
    """
    torch = vision.require_torch()
    frames = features.shape[-1]
    pieces = []
    with torch.no_grad():
        for start in range(0, frames, chunk):
            stop = min(start + chunk + lookahead, frames)
            window = torch.from_numpy(features[None, :, :stop]).to(device)
            logits = model(window)[0].cpu().numpy()
            pieces.append(logits[start : min(start + chunk, frames)])
    return np.concatenate(pieces, axis=0)


def make_causal_crnn(*, n_mels: int, num_symbols: int, channels: int, hidden: int):
    """I2's recogniser with the backward pass removed.

    One flag — ``bidirectional=False`` — and a doubled hidden size, so that
    "causal is worse" is not partly "causal is smaller". Doubling does not make
    the counts equal: a unidirectional GRU of width $2h$ has more parameters
    than a bidirectional one of width $h$, because the recurrent matrix is
    $(2h)^2$ against two of $h^2$. Measured here it is **1.22x** the
    bidirectional model, which is reported as ``causal/parameter_ratio`` rather
    than described as matched. The direction of the bias is the useful part:
    the causal arm is the larger one, so any accuracy it still gives up is a
    cost of causality and not of capacity.

    The convolution is already causal in effect: it strides only in frequency,
    so no time step ever mixes with its neighbours there.
    """
    vision.require_torch()
    from torch import nn

    class CausalCRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(),
                nn.MaxPool2d((2, 1)),
                nn.Conv2d(channels, channels * 2, 3, padding=1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(),
                nn.MaxPool2d((2, 1)),
            )
            reduced = n_mels // 4
            self.recurrent = nn.GRU(
                channels * 2 * reduced, 2 * hidden, num_layers=2, bidirectional=False, batch_first=True, dropout=0.1
            )
            self.head = nn.Linear(2 * hidden, num_symbols)

        def forward(self, x):
            h = self.features(x.unsqueeze(1))
            batch, channel, mel, frames = h.shape
            h = h.permute(0, 3, 1, 2).reshape(batch, frames, channel * mel)
            h, _ = self.recurrent(h)
            return self.head(h)

    return CausalCRNN


@experiment(
    "i4_streaming_latency",
    defaults={
        "per_combination": 30,
        "channels": 32,
        "hidden": 128,
        "epochs": 40,
        "batch_size": 32,
        "lr": 3e-3,
        "eval_every": 10,
        "chunks": [4, 8, 16, 32],
        "lookaheads": [0, 2, 4, 8, 16],
        "chunk_for_lookahead": 8,
        "timing_repeats": 20,
    },
    description="Chunked and causal streaming inference against full-utterance decoding, and what each costs.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Streaming and latency — chunked inference against full-utterance, with the accuracy it costs"""
    torch = vision.require_torch()
    device = torchkit.device()
    config = audio.FrontEndConfig()
    from parts.h_computer_vision.h3_crnn_ctc import greedy_decode
    from parts.i_speech.i2_asr_ctc_whisper import (
        ALPHABET,
        BLANK,
        character_error_rate,
        prepare,
        train_ctc,
        transcribe,
    )

    print(f"[i4] device: {device}, {config.hop_ms:.0f} ms per frame")

    clips = audio.load_fsdd(per_combination=exp["per_combination"])
    train_clips, validation_clips, test_clips = audio.split_by_speaker(clips)
    train_split = prepare(train_clips, config=config)
    mean, std = train_split[-2], train_split[-1]
    validation_split = (
        *prepare(validation_clips, config=config, mean=mean, std=std)[:4],
        validation_clips.transcripts,
    )
    test_split = (*prepare(test_clips, config=config, mean=mean, std=std)[:4], test_clips.transcripts)
    truths = test_clips.transcripts

    summary: dict[str, float] = {
        "corpus/test": float(len(test_clips)),
        "frame_ms": config.hop_ms,
    }

    model, _, _, _ = train_ctc(train_split, validation_split, exp=exp, config=config, device=device)
    offline = transcribe(model, test_split, device=device)
    summary["offline/cer"] = character_error_rate(offline, truths)
    print(f"[i4] full-utterance, bidirectional: CER {summary['offline/cer']:.3f}")

    features, lengths = test_split[0], test_split[1]

    def stream_cer(*, chunk: int, lookahead: int) -> float:
        predictions = []
        for row, length in enumerate(lengths):
            logits = chunked_logits(
                model, features[row, :, : int(length)], chunk=chunk, lookahead=lookahead, device=device
            )
            log_probabilities = torch.log_softmax(torch.from_numpy(logits), dim=-1).numpy()
            predictions.append(greedy_decode(log_probabilities, alphabet=ALPHABET))
        return character_error_rate(predictions, truths)

    # -- 1. chunk size, with no lookahead ------------------------------------
    chunk_curve: list[float] = []
    for chunk in exp["chunks"]:
        cer = stream_cer(chunk=chunk, lookahead=0)
        chunk_curve.append(cer)
        summary[f"chunk/{chunk}/cer"] = cer
        summary[f"chunk/{chunk}/latency_ms"] = chunk * config.hop_ms
        print(
            f"[i4] chunk {chunk:>3} frames ({chunk * config.hop_ms:>5.0f} ms): CER {cer:.3f} "
            f"({cer - summary['offline/cer']:+.3f} against full-utterance)"
        )
    summary["chunk/cost_at_smallest"] = chunk_curve[0] - summary["offline/cer"]
    summary["chunk/cost_at_largest"] = chunk_curve[-1] - summary["offline/cer"]

    # -- 2. buying accuracy back with latency --------------------------------
    lookahead_curve: list[float] = []
    for lookahead in exp["lookaheads"]:
        cer = stream_cer(chunk=exp["chunk_for_lookahead"], lookahead=lookahead)
        lookahead_curve.append(cer)
        summary[f"lookahead/{lookahead}/cer"] = cer
        summary[f"lookahead/{lookahead}/latency_ms"] = (exp["chunk_for_lookahead"] + lookahead) * config.hop_ms
        print(f"[i4] lookahead {lookahead:>3} frames (+{lookahead * config.hop_ms:>4.0f} ms latency): CER {cer:.3f}")
    recovered = lookahead_curve[0] - lookahead_curve[-1]
    summary["lookahead/recovered_cer"] = recovered
    summary["lookahead/remaining_gap"] = lookahead_curve[-1] - summary["offline/cer"]
    summary["lookahead/cer_per_10ms"] = recovered / max(exp["lookaheads"][-1] * config.hop_ms / 10.0, 1e-9)

    # -- 3. a model that never needed the future -----------------------------
    torchkit.seed_torch(exp.seed)
    causal = make_causal_crnn(
        n_mels=config.n_mels, num_symbols=len(ALPHABET) + 1, channels=exp["channels"], hidden=exp["hidden"]
    )().to(device)
    summary["causal/parameters"] = float(torchkit.count_parameters(causal))
    summary["bidirectional/parameters"] = float(torchkit.count_parameters(model))
    summary["causal/parameter_ratio"] = summary["causal/parameters"] / summary["bidirectional/parameters"]

    optimizer = torch.optim.AdamW(causal.parameters(), lr=exp["lr"], weight_decay=1e-4)
    loss_fn = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    x, train_lengths, targets, target_lengths, _, _ = train_split
    inputs = torch.from_numpy(x).to(device)
    offsets = np.concatenate([[0], np.cumsum(target_lengths)])
    rng = np.random.default_rng(exp.seed)
    order = np.arange(len(train_lengths))
    for _ in range(exp["epochs"]):
        causal.train()
        rng.shuffle(order)
        for start in range(0, len(order), exp["batch_size"]):
            index = order[start : start + exp["batch_size"]]
            optimizer.zero_grad(set_to_none=True)
            log_probabilities = torch.log_softmax(causal(inputs[index]), dim=-1).permute(1, 0, 2)
            flat = np.concatenate([targets[offsets[i] : offsets[i + 1]] for i in index])
            loss = loss_fn(
                log_probabilities,
                torch.from_numpy(flat).to(device),
                torch.from_numpy(train_lengths[index]).to(device),
                torch.from_numpy(target_lengths[index]).to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(causal.parameters(), 5.0)
            optimizer.step()
    summary["causal/cer"] = character_error_rate(transcribe(causal, test_split, device=device), truths)
    summary["causal/cost"] = summary["causal/cer"] - summary["offline/cer"]
    print(
        f"[i4] causal model ({int(summary['causal/parameters']):,} parameters against "
        f"{int(summary['bidirectional/parameters']):,}): CER {summary['causal/cer']:.3f} "
        f"({summary['causal/cost']:+.3f}) -- and zero lookahead latency"
    )

    # -- wall clock, kept separate from algorithmic latency -------------------
    single = features[:1, :, : int(lengths[0])]
    model.eval()
    with torch.no_grad():
        for _ in range(3):
            model(torch.from_numpy(single).to(device))
        started = time.time()
        for _ in range(exp["timing_repeats"]):
            model(torch.from_numpy(single).to(device))
        offline_ms = (time.time() - started) / exp["timing_repeats"] * 1000.0
    started = time.time()
    for _ in range(exp["timing_repeats"]):
        chunked_logits(
            model, features[0, :, : int(lengths[0])], chunk=exp["chunk_for_lookahead"], lookahead=0, device=device
        )
    chunked_ms = (time.time() - started) / exp["timing_repeats"] * 1000.0
    summary["wallclock/offline_ms"] = offline_ms
    summary["wallclock/chunked_ms"] = chunked_ms
    summary["wallclock/overhead"] = chunked_ms / max(offline_ms, 1e-9)
    summary["algorithmic/offline_latency_ms"] = float(lengths[0]) * config.hop_ms
    summary["algorithmic/streaming_latency_ms"] = exp["chunk_for_lookahead"] * config.hop_ms
    print(
        f"[i4] algorithmic latency {summary['algorithmic/offline_latency_ms']:.0f} ms offline against "
        f"{summary['algorithmic/streaming_latency_ms']:.0f} ms streaming; wall clock costs "
        f"{summary['wallclock/overhead']:.1f}x more compute for the same audio, because each chunk "
        f"re-reads the utterance from the start"
    )

    # -- figures ---------------------------------------------------------------
    exp.log_figure(
        "chunk",
        plotting.plot_curves(
            {"chunked": chunk_curve, "full utterance": [summary["offline/cer"]] * len(exp["chunks"])},
            x=[c * config.hop_ms for c in exp["chunks"]],
            title="Chunked decoding against chunk size, no lookahead",
            xlabel="chunk (ms)",
            ylabel="CER",
            marker="o",
        ),
    )
    exp.log_figure(
        "lookahead",
        plotting.plot_curves(
            {
                "chunked + lookahead": lookahead_curve,
                "full utterance": [summary["offline/cer"]] * len(exp["lookaheads"]),
                "causal model": [summary["causal/cer"]] * len(exp["lookaheads"]),
            },
            x=[a * config.hop_ms for a in exp["lookaheads"]],
            title="What a frame of delay buys back",
            xlabel="lookahead (ms)",
            ylabel="CER",
            marker="o",
        ),
    )
    exp.log_figure(
        "designs",
        plotting.plot_comparison(
            {
                "full utterance": summary["offline/cer"],
                f"chunked {exp['chunk_for_lookahead']}f": summary["lookahead/0/cer"],
                "causal": summary["causal/cer"],
            },
            title="Three answers to the same constraint",
            ylabel="CER",
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
