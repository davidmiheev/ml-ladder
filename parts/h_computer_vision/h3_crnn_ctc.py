"""H3. Text recognition with CRNN + CTC — synthetic text lines, greedy vs beam decoding, CER/WER

Lab reference: [open-mmlab/mmocr — detection and recognition reference implementations](https://github.com/open-mmlab/mmocr)

C2 read a sequence and wrote a sequence, and knew how they lined up: teacher
forcing gave the decoder the previous target at every step. Reading text off an
image does not come with that alignment. A 128-pixel-wide line holds a variable
number of characters at unknown positions, and labelling every column by hand is
exactly the annotation cost OCR exists to avoid.

CTC is the trick that removes the requirement. Instead of predicting one
character per column, the network predicts a distribution over the alphabet plus
a *blank* at every column, and the loss sums the probability of **every** column
labelling that collapses to the target string. Alignment becomes a latent
variable that gets marginalized out rather than supplied.

The rung then measures the two things that follow from that design: how much a
better decoder recovers (greedy versus beam search over the same probabilities)
and how the model fares on letterforms it never trained on (held-out fonts).

Run with ``ladder run H3`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, experiment, plotting, synth, torchkit, vision

# The blank is index 0 and is not a character; every real symbol is shifted up
# by one. This is the convention torch's CTCLoss defaults to.
BLANK = 0


def encode_text(text: str) -> list[int]:
    """Characters to CTC targets. Spaces are a symbol like any other."""
    alphabet = synth.ALPHABET + " "
    return [alphabet.index(character) + 1 for character in text]


def decode_ids(ids: list[int]) -> str:
    alphabet = synth.ALPHABET + " "
    return "".join(alphabet[index - 1] for index in ids if index != BLANK)


# -- the metrics -------------------------------------------------------------


def edit_distance(a: str, b: str) -> int:
    """Levenshtein distance by the standard two-row dynamic program.

    OCR error is reported as an edit distance rather than a per-position
    mismatch because a recognizer that drops one character would otherwise
    score as if it got *everything after it* wrong too. One deletion should
    cost one, and that is what this measures.
    """
    if a == b:
        return 0
    previous = list(range(len(b) + 1))
    for i, character_a in enumerate(a, start=1):
        current = [i]
        for j, character_b in enumerate(b, start=1):
            current.append(
                min(
                    previous[j] + 1,  # deletion
                    current[j - 1] + 1,  # insertion
                    previous[j - 1] + (character_a != character_b),  # substitution
                )
            )
        previous = current
    return previous[-1]


def character_error_rate(predictions: list[str], truths: list[str]) -> float:
    r"""$\mathrm{CER} = \sum_i d(\hat{y}_i, y_i) / \sum_i |y_i|$.

    Pooled over the corpus rather than averaged per line: a per-line average
    lets a short line's single error count as much as a long line's, and the
    denominator is what the metric is normalising by.

    Note CER is not bounded by 1 — a model that hallucinates can have more
    insertions than the reference has characters.
    """
    distance = sum(edit_distance(p, t) for p, t in zip(predictions, truths, strict=True))
    return distance / max(sum(len(t) for t in truths), 1)


def word_error_rate(predictions: list[str], truths: list[str]) -> float:
    """Edit distance over whitespace-separated tokens rather than characters.

    WER is the stricter metric and the one an application usually cares about:
    a single wrong character ruins the word it sits in. The gap between CER and
    WER therefore says how *concentrated* the errors are — many errors in few
    words, or a scattering across many.
    """
    distance = 0
    total = 0
    for prediction, truth in zip(predictions, truths, strict=True):
        predicted_words, true_words = prediction.split(), truth.split()
        # The same dynamic program, over token sequences. Mapping each distinct
        # token to one character lets the character routine do the work.
        vocabulary = {word: chr(index) for index, word in enumerate(dict.fromkeys(predicted_words + true_words))}
        distance += edit_distance(
            "".join(vocabulary[word] for word in predicted_words),
            "".join(vocabulary[word] for word in true_words),
        )
        total += len(true_words)
    return distance / max(total, 1)


# -- decoding ----------------------------------------------------------------


def greedy_decode(log_probabilities: NDArray[np.float32]) -> str:
    """Best symbol per column, then collapse repeats and drop blanks.

    This is the cheap decoder, and it is *not* the most likely string: it is the
    most likely **alignment**, collapsed. Those differ whenever the probability
    mass for one string is spread over several alignments, which is the gap beam
    search is supposed to close.
    """
    best = log_probabilities.argmax(axis=-1)
    collapsed: list[int] = []
    previous = -1
    for index in best:
        if index != previous and index != BLANK:
            collapsed.append(int(index))
        previous = int(index)
    return decode_ids(collapsed)


def beam_decode(log_probabilities: NDArray[np.float32], *, beam_width: int) -> str:
    r"""Prefix beam search: score *strings*, not alignments.

    The distinction is the whole point. A string's probability is the sum over
    every alignment that collapses to it,

    $$p(y \mid x) = \sum_{\pi \in \mathcal{B}^{-1}(y)} \prod_t p(\pi_t \mid x),$$

    so the search has to keep two numbers per prefix — the probability that it
    ends in a blank and the probability that it ends in its final symbol —
    because which one applies decides whether the next repeated character
    extends the prefix or merges into it. Collapsing them (as a naive beam over
    alignments does) is what makes greedy decoding wrong.

    Kept deliberately free of a language model: with one, the decoder starts
    fixing errors using knowledge the *recognizer* never had, and the
    comparison stops being about decoding.
    """
    # beams maps prefix -> (log p ending in blank, log p ending in the symbol)
    beams: dict[tuple[int, ...], tuple[float, float]] = {(): (0.0, -np.inf)}
    for column in log_probabilities:
        next_beams: dict[tuple[int, ...], tuple[float, float]] = {}

        def add(prefix, blank_score, symbol_score, store=next_beams):
            existing = store.get(prefix, (-np.inf, -np.inf))
            store[prefix] = (
                float(np.logaddexp(existing[0], blank_score)),
                float(np.logaddexp(existing[1], symbol_score)),
            )

        # Only the top candidates per column are worth extending; the rest
        # contribute nothing measurable and cost |alphabet| times the work.
        candidates = np.argsort(-column)[: beam_width + 1]
        for prefix, (blank_log, symbol_log) in beams.items():
            total = float(np.logaddexp(blank_log, symbol_log))
            for symbol in candidates:
                symbol = int(symbol)
                score = float(column[symbol])
                if symbol == BLANK:
                    # A blank keeps the prefix but moves its mass to "ends in blank".
                    add(prefix, total + score, -np.inf)
                elif prefix and symbol == prefix[-1]:
                    # Repeating the last symbol: extends the prefix only if the
                    # previous column was a blank, otherwise it merges.
                    add(prefix, -np.inf, symbol_log + score)
                    add((*prefix, symbol), -np.inf, blank_log + score)
                else:
                    add((*prefix, symbol), -np.inf, total + score)
        beams = dict(sorted(next_beams.items(), key=lambda item: -np.logaddexp(*item[1]))[:beam_width])
    best = max(beams.items(), key=lambda item: np.logaddexp(*item[1]))[0]
    return decode_ids(list(best))


def make_crnn(*, num_symbols: int, channels: int, hidden: int):
    """CNN over the image, then a BiLSTM along the width.

    The shape discipline is the interesting part. The convolutions pool height
    all the way to 1 but pool width only twice, so a 32x128 image becomes a
    sequence of 32 columns — one "time step" per 4 input pixels. That ratio is
    a hyperparameter with a hard constraint attached: CTC cannot emit more
    characters than it has columns, and needs a spare column between repeated
    characters, so the sequence must be comfortably longer than the longest
    label.
    """
    vision.require_torch()
    from torch import nn

    class CRNN(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.features = nn.Sequential(
                nn.Conv2d(1, channels, 3, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # 32x128 -> 16x64
                nn.Conv2d(channels, channels * 2, 3, padding=1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2),  # -> 8x32
                nn.Conv2d(channels * 2, channels * 4, 3, padding=1),
                nn.BatchNorm2d(channels * 4),
                nn.ReLU(inplace=True),
                # Pool height only: the width axis is the sequence and must survive.
                nn.MaxPool2d((2, 1)),  # -> 4x32
                nn.Conv2d(channels * 4, channels * 4, 3, padding=1),
                nn.BatchNorm2d(channels * 4),
                nn.ReLU(inplace=True),
                # Collapse whatever height remains to 1 while leaving the width
                # untouched. A fixed MaxPool2d((4, 1)) would silently only work
                # at height 32 and fail the squeeze below at any other size.
                nn.AdaptiveMaxPool2d((1, None)),  # -> 1 x width/4
            )
            self.rnn = nn.LSTM(channels * 4, hidden, batch_first=True, bidirectional=True, num_layers=2)
            self.head = nn.Linear(hidden * 2, num_symbols)

        def forward(self, images):
            features = self.features(images)  # (n, c, 1, w)
            assert features.shape[2] == 1, f"height did not collapse: {tuple(features.shape)}"
            sequence = features.squeeze(2).permute(0, 2, 1)  # (n, w, c)
            outputs, _ = self.rnn(sequence)
            return self.head(outputs)  # (n, w, symbols)

    return CRNN


def prepare(lines: list[synth.TextLine], device):
    """Images to a normalized tensor, texts to a flat CTC target buffer."""
    torch = vision.require_torch()
    images = np.stack([line.image for line in lines]).astype(np.float32) / 255.0
    # Ink is dark on light; invert so that "there is a stroke here" is a large
    # activation rather than a small one. Nothing depends on this, but it makes
    # the first convolution's job the obvious one.
    tensor = torch.from_numpy(1.0 - images).unsqueeze(1).to(device)
    targets = [encode_text(line.text) for line in lines]
    lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)
    flat = torch.tensor([symbol for target in targets for symbol in target], dtype=torch.long)
    return tensor, flat.to(device), lengths.to(device)


@experiment(
    "h3_crnn_ctc",
    defaults={
        "train_lines": 6000,
        "test_lines": 800,
        "height": 32,
        "width": 128,
        "channels": 16,
        "hidden": 96,
        "epochs": 12,
        "batch_size": 64,
        "lr": 2e-3,
        "beam_widths": [1, 5, 15],
        "max_words": 2,
    },
    description="A CRNN trained with CTC on rendered text, decoded greedily and with a prefix beam search.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Text recognition with CRNN + CTC — synthetic text lines, greedy vs beam decoding, CER/WER"""
    torch = vision.require_torch()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)
    print(f"[h3] device: {device}")

    num_symbols = len(synth.ALPHABET) + 2  # alphabet + space + blank
    shared = {"height": exp["height"], "width": exp["width"], "max_words": exp["max_words"]}
    train_lines = synth.text_lines(exp["train_lines"], fonts=synth.TRAIN_FONTS, rng=rng, **shared)
    test_lines = synth.text_lines(exp["test_lines"], fonts=synth.TRAIN_FONTS, rng=rng, **shared)
    held_out_lines = synth.text_lines(exp["test_lines"], fonts=synth.HELD_OUT_FONTS, rng=rng, **shared)

    x_train, targets, lengths = prepare(train_lines, device)
    longest = int(lengths.max())
    print(
        f"[h3] {len(train_lines)} train lines, longest label {longest} symbols; {exp['width'] // 4} columns available"
    )
    if exp["width"] // 4 <= longest:
        # CTC's constraint, stated loudly rather than discovered as a silent
        # inf in the loss: the input sequence must be longer than the label.
        raise SystemExit(f"width {exp['width']} gives {exp['width'] // 4} columns for labels up to {longest} symbols")

    torchkit.seed_torch(exp.seed)
    model = make_crnn(num_symbols=num_symbols, channels=exp["channels"], hidden=exp["hidden"])().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=1e-4)
    criterion = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    print(f"[h3] CRNN: {torchkit.count_parameters(model):,} parameters")

    offsets = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long, device=device), lengths]), dim=0)
    losses: list[float] = []
    for epoch in range(1, exp["epochs"] + 1):
        model.train()
        permutation = torch.randperm(len(x_train), device=device)
        epoch_losses = []
        for start in range(0, len(x_train), exp["batch_size"]):
            index = permutation[start : start + exp["batch_size"]]
            logits = model(x_train[index])
            log_probabilities = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)  # CTC wants (T, N, C)
            batch_targets = torch.cat([targets[offsets[i] : offsets[i + 1]] for i in index])
            input_lengths = torch.full((len(index),), logits.shape[1], dtype=torch.long, device=device)
            loss = criterion(log_probabilities, batch_targets, input_lengths, lengths[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(float(np.mean(epoch_losses)))
        if epoch % max(exp["epochs"] // 4, 1) == 0:
            print(f"[h3] epoch {epoch:>3}  ctc loss={losses[-1]:.4f}")

    # -- decode both test sets, every beam width -----------------------------
    summary: dict[str, float] = {"train_loss": losses[-1], "columns": float(exp["width"] // 4)}
    examples: dict[str, list[tuple[str, str]]] = {}

    def evaluate(name: str, lines: list[synth.TextLine]):
        x, _, _ = prepare(lines, device)
        model.eval()
        with torch.no_grad():
            chunks = [torch.log_softmax(model(chunk), dim=-1).cpu().numpy() for chunk in x.split(128)]
        log_probabilities = np.concatenate(chunks)
        truths = [line.text for line in lines]

        greedy = [greedy_decode(row) for row in log_probabilities]
        summary[f"{name}/greedy/cer"] = character_error_rate(greedy, truths)
        summary[f"{name}/greedy/wer"] = word_error_rate(greedy, truths)
        summary[f"{name}/greedy/exact"] = float(np.mean([p == t for p, t in zip(greedy, truths, strict=True)]))
        print(
            f"[h3] {name:<9} greedy      CER={summary[f'{name}/greedy/cer']:.4f}  "
            f"WER={summary[f'{name}/greedy/wer']:.4f}  exact={summary[f'{name}/greedy/exact']:.4f}"
        )
        examples[f"{name}/greedy"] = list(zip(greedy, truths, strict=True))[:8]
        # Per-font CER over the whole split: which letterforms actually fail,
        # rather than which happened to land in the first few samples.
        for font in sorted({line.font for line in lines}):
            chosen = [(p, t) for p, t, line in zip(greedy, truths, lines, strict=True) if line.font == font]
            summary[f"{name}/font/{font.replace('.ttf', '')}/cer"] = character_error_rate(
                [p for p, _ in chosen], [t for _, t in chosen]
            )

        for beam_width in exp["beam_widths"]:
            decoded = [beam_decode(row, beam_width=beam_width) for row in log_probabilities]
            summary[f"{name}/beam{beam_width}/cer"] = character_error_rate(decoded, truths)
            summary[f"{name}/beam{beam_width}/wer"] = word_error_rate(decoded, truths)
            summary[f"{name}/beam{beam_width}/exact"] = float(
                np.mean([p == t for p, t in zip(decoded, truths, strict=True)])
            )
            print(
                f"[h3] {name:<9} beam {beam_width:<6} CER={summary[f'{name}/beam{beam_width}/cer']:.4f}  "
                f"WER={summary[f'{name}/beam{beam_width}/wer']:.4f}  "
                f"exact={summary[f'{name}/beam{beam_width}/exact']:.4f}"
            )
        return log_probabilities

    train_font_probabilities = evaluate("seen", test_lines)
    evaluate("held_out", held_out_lines)

    # -- derived comparisons -------------------------------------------------
    summary["font_gap/cer"] = summary["held_out/greedy/cer"] - summary["seen/greedy/cer"]
    summary["font_gap/wer"] = summary["held_out/greedy/wer"] - summary["seen/greedy/wer"]
    widest = max(exp["beam_widths"])
    summary["beam_gain/cer"] = summary["seen/greedy/cer"] - summary[f"seen/beam{widest}/cer"]
    summary["beam_gain/wer"] = summary["seen/greedy/wer"] - summary[f"seen/beam{widest}/wer"]
    # CER counts characters and WER counts words, so WER/CER > 1 always; how
    # much greater says whether errors cluster into few words or scatter.
    if summary["seen/greedy/cer"] > 0:
        summary["seen/wer_over_cer"] = summary["seen/greedy/wer"] / summary["seen/greedy/cer"]
    print(
        f"[h3] font gap (greedy): CER {summary['seen/greedy/cer']:.4f} -> "
        f"{summary['held_out/greedy/cer']:.4f} ({summary['font_gap/cer']:+.4f})"
    )
    print(f"[h3] beam {widest} over greedy: CER {summary['beam_gain/cer']:+.5f}  WER {summary['beam_gain/wer']:+.5f}")

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "training",
        plotting.plot_curves(
            {"CTC": losses},
            x=list(range(1, exp["epochs"] + 1)),
            title="CTC loss",
            xlabel="epoch",
            ylabel="loss",
            logy=True,
        ),
    )
    exp.log_figure(
        "decoders",
        plotting.plot_comparison(
            {
                f"{split} {decoder}": summary[f"{split}/{decoder}/cer"]
                for split in ("seen", "held_out")
                for decoder in ("greedy", f"beam{widest}")
            },
            title="Character error rate by split and decoder",
            ylabel="CER",
        ),
    )
    per_font = {key.split("/font/")[1].removesuffix("/cer"): value for key, value in summary.items() if "/font/" in key}
    exp.log_figure(
        "error_by_font",
        plotting.plot_comparison(
            per_font,
            title="CER per font (left of the gap: trained on; right: held out)",
            ylabel="CER",
        ),
    )

    # The frame-level posteriors are the object CTC actually produces, and
    # seeing the blank dominate is the clearest way to understand the collapse.
    exp.log_figure(
        "ctc_posteriors",
        plotting.plot_attention(
            np.exp(train_font_probabilities[0]).T,
            source_labels=[str(column) for column in range(train_font_probabilities.shape[1])],
            target_labels=["blank", *list(synth.ALPHABET), "space"],
            xlabel="image column (time step)",
            ylabel="symbol",
            title=f"CTC posteriors: {test_lines[0].text!r} -> {greedy_decode(train_font_probabilities[0])!r}",
        ),
    )
    exp.log_figure(
        "samples",
        plotting.plot_image_grid(
            np.stack([line.image for line in test_lines[:8]]),
            nrow=2,
            title="Test lines (seen fonts)",
        ),
    )

    print("[h3] examples (prediction | truth):")
    for prediction, truth in examples["held_out/greedy"][:5]:
        flag = " " if prediction == truth else "x"
        print(f"  {flag} {prediction!r:<28} | {truth!r}")
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
