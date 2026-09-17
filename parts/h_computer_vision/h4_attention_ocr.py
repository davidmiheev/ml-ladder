"""H4. Attention-decoder OCR against CTC on the same data, with a pretrained model as the baseline

Lab reference: [open-mmlab/mmocr — detection and recognition reference implementations](https://github.com/open-mmlab/mmocr)

H3 read text with CTC: one prediction per image column, alignment marginalized
out, everything emitted in one parallel forward pass. The alternative is the
decoder C2 built — an autoregressive RNN that attends over the encoder states
and emits one character at a time, conditioned on what it has already written.

The two make opposite bets. CTC assumes the output is **monotonic in the input**
and conditionally independent given the columns; it cannot look back at what it
wrote, so it cannot spell. An attention decoder gives up the monotonicity
assumption and gains a language model over its own output for free — which helps
when the image is ambiguous and hurts when the model starts believing itself.

That trade has a specific, testable consequence: CTC's monotonic alignment is
length-agnostic, while an attention decoder has to *learn* when to stop. So this
rung trains both on one and two words and then tests on **three**, which neither
has seen, with a same-length-wider-canvas control so that "longer text" is not
confounded with "smaller glyphs".

TrOCR is the third arm and is not trained here at all: it is a pretrained
reference, and the interesting question it answers is whether the held-out-font
gap H3 found survives contact with a model trained on real printed text.

Needs ``uv sync --extra torch --extra ocr``. Run with ``ladder run H4``.
"""

from __future__ import annotations

import time
from functools import lru_cache

import numpy as np

from mlladder import Experiment, experiment, plotting, synth, torchkit, vision
from parts.h_computer_vision.h3_crnn_ctc import (
    BLANK,
    character_error_rate,
    encode_text,
    greedy_decode,
    make_crnn,
    prepare,
    word_error_rate,
)

# The attention decoder needs two symbols CTC does not: something to start from
# and something to stop at. Reusing H3's 1-based symbol ids keeps the two arms'
# vocabularies aligned, so a decoded string means the same thing in both.
SOS = 0
EOS = len(synth.ALPHABET) + 2


def decode_symbols(ids: list[int]) -> str:
    alphabet = synth.ALPHABET + " "
    return "".join(alphabet[index - 1] for index in ids if 1 <= index <= len(alphabet))


def make_attention_recognizer(*, num_symbols: int, channels: int, hidden: int):
    """H3's convolutional encoder, C2's attention decoder bolted on.

    Deliberately the same encoder as the CTC arm, and deliberately the same
    dot-product attention as C2's translation model: if the encoder or the
    attention mechanism differed, the comparison would be about those and not
    about CTC versus autoregressive decoding, which is the one thing this rung
    is trying to isolate.
    """
    torch = vision.require_torch()
    from torch import nn

    class AttentionRecognizer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            # Same stack as the CRNN, minus the CTC head.
            crnn = make_crnn(num_symbols=num_symbols, channels=channels, hidden=hidden)()
            self.features = crnn.features
            self.encoder = crnn.rnn
            self.embedding = nn.Embedding(num_symbols, hidden)
            # The decoder state is 2*hidden wide to match the bidirectional
            # encoder's memory, so dot-product attention needs no extra
            # projection. C2 sized its decoder the same way and for the same
            # reason.
            self.decoder = nn.LSTM(hidden + hidden * 2, hidden * 2, batch_first=True)
            self.combine = nn.Linear(hidden * 4, hidden * 2)
            self.head = nn.Linear(hidden * 2, num_symbols)
            self.hidden = hidden
            self.memory_width = hidden * 2

        def encode(self, images):
            features = self.features(images)
            sequence = features.squeeze(2).permute(0, 2, 1)
            memory, _ = self.encoder(sequence)  # (n, columns, 2*hidden)
            return memory

        def step(self, memory, token, state, previous_context):
            """One decoder step: attend, then emit.

            The context vector from the previous step is fed back in alongside
            the embedded token — "input feeding", which is what lets the
            decoder know where it just looked.
            """
            embedded = self.embedding(token)  # (n, 1, hidden)
            decoder_input = torch.cat([embedded, previous_context], dim=-1)
            output, state = self.decoder(decoder_input, state)
            scores = torch.bmm(output, memory.transpose(1, 2)) / np.sqrt(self.memory_width)
            weights = torch.softmax(scores, dim=-1)  # (n, 1, columns)
            context = torch.bmm(weights, memory)  # (n, 1, 2*hidden)
            combined = torch.tanh(self.combine(torch.cat([output, context], dim=-1)))
            return self.head(combined), state, context, weights

        def forward(self, images, targets_in):
            """Teacher-forced training pass: the whole target sequence at once."""
            memory = self.encode(images)
            state = None
            context = memory.new_zeros(len(images), 1, self.memory_width)
            logits = []
            for position in range(targets_in.shape[1]):
                step_logits, state, context, _ = self.step(
                    memory, targets_in[:, position : position + 1], state, context
                )
                logits.append(step_logits)
            return torch.cat(logits, dim=1)

        @torch.no_grad()
        def transcribe(self, images, *, max_length: int):
            """Greedy autoregressive decode, feeding each emission back in."""
            self.eval()
            memory = self.encode(images)
            state = None
            context = memory.new_zeros(len(images), 1, self.memory_width)
            token = torch.full((len(images), 1), SOS, dtype=torch.long, device=images.device)
            finished = torch.zeros(len(images), dtype=torch.bool, device=images.device)
            emitted, attention = [], []
            for _ in range(max_length):
                step_logits, state, context, weights = self.step(memory, token, state, context)
                token = step_logits.argmax(dim=-1)
                attention.append(weights[:, 0])
                # Once a row has emitted EOS, pin it there so the rest of the
                # batch can keep decoding without corrupting it.
                token = torch.where(finished.unsqueeze(1), torch.full_like(token, EOS), token)
                finished |= token[:, 0] == EOS
                emitted.append(token[:, 0])
                if bool(finished.all()):
                    break
            return torch.stack(emitted, dim=1), torch.stack(attention, dim=1)

    return AttentionRecognizer


def error_breakdown(predictions: list[str], truths: list[str]) -> dict[str, float]:
    """Split the error into length and content, which fail differently.

    An attention decoder that loops emits far too many characters; one that
    stops early emits too few. Both show up in CER as one number, and the sign
    of the length error is what tells them apart — so it is reported separately
    rather than inferred from examples.
    """
    predicted_lengths = np.array([len(p) for p in predictions], dtype=np.float64)
    true_lengths = np.array([len(t) for t in truths], dtype=np.float64)
    return {
        "length_bias": float(np.mean(predicted_lengths - true_lengths)),
        "length_abs": float(np.mean(np.abs(predicted_lengths - true_lengths))),
        "empty_rate": float(np.mean(predicted_lengths == 0)),
        "overlong_rate": float(np.mean(predicted_lengths > true_lengths + 2)),
    }


@lru_cache(maxsize=2)
def load_trocr(model_name: str, device_key: str):
    """Load TrOCR once per (model, device) and keep it.

    Cached because this rung reads three splits per seed and a sweep runs three
    seeds: without it the 62M-parameter checkpoint is re-read from disk nine
    times for no benefit. ``device_key`` is a string rather than a
    ``torch.device`` so the cache key stays hashable and stable.
    """
    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

    processor = TrOCRProcessor.from_pretrained(model_name)
    model = VisionEncoderDecoderModel.from_pretrained(model_name).to(device_key)
    model.eval()
    return processor, model


def trocr_transcribe(lines: list[synth.TextLine], *, model_name: str, batch_size: int, device):
    """Zero-shot TrOCR. Returns the transcriptions, upper-cased for comparison.

    Case folding is not cheating: the synthetic corpus is upper case only, so a
    model that reads "Token" has read the glyphs correctly and merely disagrees
    about a convention this dataset does not test. Scoring it case-sensitively
    would measure the dataset's quirk rather than the model's OCR.

    The pixel values have to be moved to the model's device explicitly. Leaving
    them on the CPU does not fail — ``transformers`` raises nothing and the
    model simply runs where its inputs are — so the first version of this
    silently did all 62M parameters' worth of inference on CPU while sitting on
    an idle T4.
    """
    from PIL import Image

    processor, model = load_trocr(model_name, str(device))

    outputs: list[str] = []
    for start in range(0, len(lines), batch_size):
        batch = lines[start : start + batch_size]
        images = [Image.fromarray(line.image).convert("RGB") for line in batch]
        pixel_values = processor(images=images, return_tensors="pt").pixel_values.to(device)
        generated = model.generate(pixel_values, max_new_tokens=32)
        outputs.extend(text.strip().upper() for text in processor.batch_decode(generated, skip_special_tokens=True))
    return outputs, sum(parameter.numel() for parameter in model.parameters())


@experiment(
    "h4_attention_ocr",
    defaults={
        "train_lines": 6000,
        "test_lines": 800,
        "length_test_lines": 400,
        "height": 32,
        "width": 128,
        # The unseen-length arm needs room for a third word; 192 px keeps the
        # glyphs at roughly the training scale instead of squeezing them.
        "wide_width": 192,
        "channels": 16,
        "hidden": 96,
        "epochs": 12,
        "batch_size": 64,
        "lr": 2e-3,
        "max_words": 2,
        "trocr_model": "microsoft/trocr-small-printed",
        "trocr_batch": 32,
        "with_trocr": 1,
    },
    description="An attention decoder against CTC on identical data, with zero-shot TrOCR as the pretrained reference.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Attention-decoder OCR against CTC on the same data, with a pretrained model as the baseline"""
    torch = vision.require_torch()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)
    print(f"[h4] device: {device}")

    num_symbols = EOS + 1  # H3's symbols, plus EOS
    shared = {"height": exp["height"], "max_words": exp["max_words"]}
    train_lines = synth.text_lines(exp["train_lines"], fonts=synth.TRAIN_FONTS, width=exp["width"], rng=rng, **shared)
    seen_lines = synth.text_lines(exp["test_lines"], fonts=synth.TRAIN_FONTS, width=exp["width"], rng=rng, **shared)
    held_out_lines = synth.text_lines(
        exp["test_lines"], fonts=synth.HELD_OUT_FONTS, width=exp["width"], rng=rng, **shared
    )
    # Two extra sets that separate "more words" from "wider canvas".
    two_word_wide = synth.text_lines(
        exp["length_test_lines"],
        fonts=synth.TRAIN_FONTS,
        height=exp["height"],
        width=exp["wide_width"],
        min_words=2,
        max_words=2,
        rng=rng,
    )
    three_word_wide = synth.text_lines(
        exp["length_test_lines"],
        fonts=synth.TRAIN_FONTS,
        height=exp["height"],
        width=exp["wide_width"],
        min_words=3,
        max_words=3,
        rng=rng,
    )
    splits = {
        "seen": seen_lines,
        "held_out": held_out_lines,
        "2word_wide": two_word_wide,
        "3word_wide": three_word_wide,
    }

    summary: dict[str, float] = {}
    curves: dict[str, list[float]] = {}

    # -- arm 1: CTC, as in H3 ------------------------------------------------
    x_train, ctc_targets, ctc_lengths = prepare(train_lines, device)
    torchkit.seed_torch(exp.seed)
    ctc_model = make_crnn(num_symbols=num_symbols, channels=exp["channels"], hidden=exp["hidden"])().to(device)
    optimizer = torch.optim.AdamW(ctc_model.parameters(), lr=exp["lr"], weight_decay=1e-4)
    criterion = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    offsets = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long, device=device), ctc_lengths]), dim=0)

    ctc_losses: list[float] = []
    for _epoch in range(exp["epochs"]):
        ctc_model.train()
        permutation = torch.randperm(len(x_train), device=device)
        epoch_losses = []
        for start in range(0, len(x_train), exp["batch_size"]):
            index = permutation[start : start + exp["batch_size"]]
            logits = ctc_model(x_train[index])
            log_probabilities = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
            batch_targets = torch.cat([ctc_targets[offsets[i] : offsets[i + 1]] for i in index])
            input_lengths = torch.full((len(index),), logits.shape[1], dtype=torch.long, device=device)
            loss = criterion(log_probabilities, batch_targets, input_lengths, ctc_lengths[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(ctc_model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        ctc_losses.append(float(np.mean(epoch_losses)))
    curves["ctc"] = ctc_losses
    summary["ctc/parameters"] = float(torchkit.count_parameters(ctc_model))
    print(f"[h4] CTC arm trained: {summary['ctc/parameters']:,.0f} parameters, final loss {ctc_losses[-1]:.4f}")

    # -- arm 2: attention decoder -------------------------------------------
    longest = int(ctc_lengths.max())
    target_width = longest + 1  # room for EOS
    attention_in = torch.full((len(train_lines), target_width), SOS, dtype=torch.long)
    attention_out = torch.full((len(train_lines), target_width), EOS, dtype=torch.long)
    for row, line in enumerate(train_lines):
        ids = encode_text(line.text)
        attention_in[row, 1 : len(ids) + 1] = torch.tensor(ids)
        attention_out[row, : len(ids)] = torch.tensor(ids)
    attention_in, attention_out = attention_in.to(device), attention_out.to(device)

    torchkit.seed_torch(exp.seed)
    attention_model = make_attention_recognizer(
        num_symbols=num_symbols, channels=exp["channels"], hidden=exp["hidden"]
    )().to(device)
    optimizer = torch.optim.AdamW(attention_model.parameters(), lr=exp["lr"], weight_decay=1e-4)

    attention_losses: list[float] = []
    for epoch in range(1, exp["epochs"] + 1):
        attention_model.train()
        permutation = torch.randperm(len(x_train), device=device)
        epoch_losses = []
        for start in range(0, len(x_train), exp["batch_size"]):
            index = permutation[start : start + exp["batch_size"]]
            logits = attention_model(x_train[index], attention_in[index])
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, num_symbols), attention_out[index].reshape(-1))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(attention_model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        attention_losses.append(float(np.mean(epoch_losses)))
        if epoch % max(exp["epochs"] // 4, 1) == 0:
            print(f"[h4] epoch {epoch:>3}  ctc={ctc_losses[epoch - 1]:.4f}  attention={attention_losses[-1]:.4f}")
    curves["attention"] = attention_losses
    summary["attention/parameters"] = float(torchkit.count_parameters(attention_model))
    print(f"[h4] attention arm trained: {summary['attention/parameters']:,.0f} parameters")

    # -- evaluate every arm on every split -----------------------------------
    attention_maps = None
    for split_name, lines in splits.items():
        truths = [line.text for line in lines]
        x, _, _ = prepare(lines, device)

        started = time.perf_counter()
        ctc_model.eval()
        with torch.no_grad():
            chunks = [torch.log_softmax(ctc_model(chunk), dim=-1).cpu().numpy() for chunk in x.split(128)]
        ctc_predictions = [greedy_decode(row) for row in np.concatenate(chunks)]
        ctc_seconds = time.perf_counter() - started

        started = time.perf_counter()
        emitted_chunks, map_chunks = [], []
        for chunk in x.split(128):
            emitted, maps = attention_model.transcribe(chunk, max_length=target_width + 8)
            emitted_chunks.append(emitted.cpu().numpy())
            map_chunks.append(maps.cpu().numpy())
        attention_predictions = [
            decode_symbols([int(token) for token in row[: np.argmax(row == EOS) if (row == EOS).any() else len(row)]])
            for chunk in emitted_chunks
            for row in chunk
        ]
        attention_seconds = time.perf_counter() - started
        if split_name == "seen":
            attention_maps = (map_chunks[0], lines)
        if split_name == "held_out":
            held_out_examples = list(zip(ctc_predictions, attention_predictions, truths, strict=True))[:4]

        for arm, predictions, seconds in (
            ("ctc", ctc_predictions, ctc_seconds),
            ("attention", attention_predictions, attention_seconds),
        ):
            summary[f"{split_name}/{arm}/cer"] = character_error_rate(predictions, truths)
            summary[f"{split_name}/{arm}/wer"] = word_error_rate(predictions, truths)
            summary[f"{split_name}/{arm}/exact"] = float(
                np.mean([p == t for p, t in zip(predictions, truths, strict=True)])
            )
            summary[f"{split_name}/{arm}/seconds"] = seconds
            for key, value in error_breakdown(predictions, truths).items():
                summary[f"{split_name}/{arm}/{key}"] = value
            print(
                f"[h4] {split_name:<11} {arm:<10} CER={summary[f'{split_name}/{arm}/cer']:.4f}  "
                f"WER={summary[f'{split_name}/{arm}/wer']:.4f}  "
                f"exact={summary[f'{split_name}/{arm}/exact']:.4f}  "
                f"len-bias={summary[f'{split_name}/{arm}/length_bias']:+.2f}  ({seconds:.1f}s)"
            )

    # -- arm 3: the pretrained reference, zero-shot --------------------------
    if exp["with_trocr"]:
        for split_name in ("seen", "held_out", "3word_wide"):
            lines = splits[split_name]
            truths = [line.text for line in lines]
            started = time.perf_counter()
            predictions, parameters = trocr_transcribe(
                lines, model_name=exp["trocr_model"], batch_size=exp["trocr_batch"], device=device
            )
            summary[f"{split_name}/trocr/cer"] = character_error_rate(predictions, truths)
            summary[f"{split_name}/trocr/wer"] = word_error_rate(predictions, truths)
            summary[f"{split_name}/trocr/exact"] = float(
                np.mean([p == t for p, t in zip(predictions, truths, strict=True)])
            )
            summary[f"{split_name}/trocr/seconds"] = time.perf_counter() - started
            summary["trocr/parameters"] = float(parameters)
            for key, value in error_breakdown(predictions, truths).items():
                summary[f"{split_name}/trocr/{key}"] = value
            print(
                f"[h4] {split_name:<11} {'trocr':<10} CER={summary[f'{split_name}/trocr/cer']:.4f}  "
                f"WER={summary[f'{split_name}/trocr/wer']:.4f}  "
                f"exact={summary[f'{split_name}/trocr/exact']:.4f}  "
                f"({summary[f'{split_name}/trocr/seconds']:.1f}s, {parameters / 1e6:.0f}M params)"
            )

    # -- the comparisons the rung exists to make -----------------------------
    arms = ["ctc", "attention"] + (["trocr"] if exp["with_trocr"] else [])
    for arm in arms:
        if f"held_out/{arm}/cer" in summary:
            summary[f"font_gap/{arm}"] = summary[f"held_out/{arm}/cer"] - summary[f"seen/{arm}/cer"]
    for arm in ("ctc", "attention"):
        # Unseen length, with the canvas width held fixed between the two —
        # so this is the effect of a third word and nothing else.
        summary[f"length_gap/{arm}"] = summary[f"3word_wide/{arm}/cer"] - summary[f"2word_wide/{arm}/cer"]
        # And the canvas change on its own, as the control.
        summary[f"width_gap/{arm}"] = summary[f"2word_wide/{arm}/cer"] - summary[f"seen/{arm}/cer"]
    summary["attention_minus_ctc/seen_cer"] = summary["seen/attention/cer"] - summary["seen/ctc/cer"]
    summary["attention_minus_ctc/length_gap"] = summary["length_gap/attention"] - summary["length_gap/ctc"]
    summary["attention_over_ctc/seconds"] = summary["seen/attention/seconds"] / max(summary["seen/ctc/seconds"], 1e-9)
    print(
        f"[h4] unseen third word: CTC {summary['length_gap/ctc']:+.4f} CER  vs  "
        f"attention {summary['length_gap/attention']:+.4f}  (control, wider canvas only: "
        f"CTC {summary['width_gap/ctc']:+.4f}, attention {summary['width_gap/attention']:+.4f})"
    )
    print(f"[h4] attention decode is {summary['attention_over_ctc/seconds']:.1f}x CTC's wall clock")

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "training",
        plotting.plot_curves(
            curves,
            x=list(range(1, exp["epochs"] + 1)),
            title="CTC and cross-entropy are different objectives; only the trend is comparable",
            xlabel="epoch",
            ylabel="loss",
            logy=True,
        ),
    )
    # The length experiment, isolated: same canvas width, one more word.
    exp.log_figure(
        "length_generalization",
        plotting.plot_curves(
            {arm: [summary[f"2word_wide/{arm}/cer"], summary[f"3word_wide/{arm}/cer"]] for arm in ("ctc", "attention")},
            x=[2, 3],
            title="Words per line at a fixed canvas width (trained on 1-2)",
            xlabel="words in the line",
            ylabel="CER",
            marker="o",
        ),
    )
    comparison = {
        f"{arm}\n{split}": summary[f"{split}/{arm}/cer"]
        for split in ("seen", "held_out", "3word_wide")
        for arm in arms
        if f"{split}/{arm}/cer" in summary
    }
    exp.log_figure(
        "arms",
        plotting.plot_comparison(comparison, title="CER by arm and split", ylabel="CER"),
    )

    if attention_maps is not None:
        maps, lines = attention_maps
        # The shortest line, so the alignment is legible rather than a band —
        # the same choice C2's attention figure had to make.
        shortest = int(np.argmin([len(line.text) for line in lines[: len(maps)]]))
        steps = len(lines[shortest].text) + 1
        exp.log_figure(
            "attention_map",
            plotting.plot_attention(
                maps[shortest][:steps],
                source_labels=[str(column) for column in range(maps.shape[2])],
                target_labels=[*list(lines[shortest].text), "EOS"],
                xlabel="image column",
                ylabel="emitted character",
                title=f"Decoder attention: {lines[shortest].text!r}",
            ),
        )

    print("[h4] held-out examples (ctc | attention | truth):")
    for ctc_text, attention_text, truth in held_out_examples:
        print(f"  {ctc_text!r:<22} {attention_text!r:<22} {truth!r}")
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
