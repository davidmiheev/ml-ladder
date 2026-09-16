"""C2. Seq2seq with attention on a toy translation task

Lab reference: [Annotated Transformer / nanoGPT](https://nlp.seas.harvard.edu/annotated-transformer/)

The task is date normalization — "august 15, 2021" to "2021-08-15" — chosen
because every output character is a copy of some input span, so the alignment
the attention *should* learn is known in advance and can be scored rather than
eyeballed.

The comparison is the one the attention paper was written to make: an
encoder-decoder whose entire source sentence must fit through one fixed-size
vector, against the same model allowed to look back at the encoder states.

The length that matters is *content*, not padding. The first version of this
rung swept the pad width and found nothing — zeros cost a fixed-size vector
nothing to carry, so both arms scored 0.99+ everywhere. Here the date is buried
in filler words instead, so the distance the bottleneck must actually carry
information really does grow.

Run with ``ladder run C2`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np

from mlladder import Experiment, experiment, plotting, text, torchkit, vision


def make_models():
    torch = vision.require_torch()
    from torch import nn

    class Encoder(nn.Module):
        """A bidirectional LSTM over the source characters."""

        def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            self.rnn = nn.LSTM(embedding_size, hidden_size, batch_first=True, bidirectional=True)

        def forward(self, source):
            outputs, (h, c) = self.rnn(self.embedding(source))
            # Concatenate the two directions into one initial decoder state.
            state = (torch.cat([h[0], h[1]], dim=1).unsqueeze(0), torch.cat([c[0], c[1]], dim=1).unsqueeze(0))
            return outputs, state

    class BottleneckDecoder(nn.Module):
        """The pre-attention design: one vector carries the whole source."""

        def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            self.rnn = nn.LSTM(embedding_size, hidden_size * 2, batch_first=True)
            self.head = nn.Linear(hidden_size * 2, vocab_size)

        def forward(self, encoder_outputs, state, target_in, source_mask=None):
            outputs, _ = self.rnn(self.embedding(target_in), state)
            return self.head(outputs), None

    class AttentionDecoder(nn.Module):
        r"""Luong (dot-product) attention over the encoder states.

        At each output step, $\alpha = \mathrm{softmax}(h_t^\top H_{\text{enc}})$ and the
        context $\sum_i \alpha_i H_i$ is concatenated to the decoder state before
        the output projection.
        """

        def __init__(self, vocab_size: int, embedding_size: int, hidden_size: int) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_size, padding_idx=0)
            self.rnn = nn.LSTM(embedding_size, hidden_size * 2, batch_first=True)
            self.combine = nn.Linear(hidden_size * 4, hidden_size * 2)
            self.head = nn.Linear(hidden_size * 2, vocab_size)

        def forward(self, encoder_outputs, state, target_in, source_mask=None):
            decoder_outputs, _ = self.rnn(self.embedding(target_in), state)
            scores = torch.bmm(decoder_outputs, encoder_outputs.transpose(1, 2))
            if source_mask is not None:
                # Padding must not receive attention mass, or the softmax
                # spends probability on positions that carry no information.
                scores = scores.masked_fill(~source_mask.unsqueeze(1), float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            context = torch.bmm(weights, encoder_outputs)
            combined = torch.tanh(self.combine(torch.cat([decoder_outputs, context], dim=-1)))
            return self.head(combined), weights

    class Seq2Seq(nn.Module):
        def __init__(
            self, source_vocab: int, target_vocab: int, embedding_size: int, hidden_size: int, *, attention: bool
        ) -> None:
            super().__init__()
            self.encoder = Encoder(source_vocab, embedding_size, hidden_size)
            decoder_class = AttentionDecoder if attention else BottleneckDecoder
            self.decoder = decoder_class(target_vocab, embedding_size, hidden_size)
            self.attention = attention

        def forward(self, source, target_in):
            encoder_outputs, state = self.encoder(source)
            return self.decoder(encoder_outputs, state, target_in, source != 0)

        @torch.no_grad()
        def translate(self, source, *, max_length: int, start_id: int = 0):
            """Greedy decode, feeding each prediction back in."""
            self.eval()
            encoder_outputs, state = self.encoder(source)
            tokens = torch.full((len(source), 1), start_id, dtype=torch.long, device=source.device)
            weights_trace = []
            for _ in range(max_length):
                logits, weights = self.decoder(encoder_outputs, state, tokens, source != 0)
                next_token = logits[:, -1:].argmax(dim=-1)
                tokens = torch.cat([tokens, next_token], dim=1)
                if weights is not None:
                    weights_trace.append(weights[:, -1])
            return tokens[:, 1:], (torch.stack(weights_trace, dim=1) if weights_trace else None)

    return Seq2Seq


def alignment_score(weights, sources: list[str], targets: list[str]) -> float:
    """Fraction of output characters whose attention peak lands on a matching input character.

    The date task makes this well defined: an output digit should attend to a
    position holding the same digit (for the month, to the month name instead,
    which this scorer counts as a miss — so a perfect model does not score 1.0).
    """
    hits = total = 0
    for row, (source, target) in enumerate(zip(sources, targets, strict=True)):
        for position, character in enumerate(target):
            if character == "-" or position >= weights.shape[1]:
                continue
            peak = int(weights[row, position].argmax())
            total += 1
            hits += int(peak < len(source) and source[peak] == character)
    return hits / max(total, 1)


@experiment(
    "c2_seq2seq_attention",
    defaults={
        "train_samples": 12000,
        "test_samples": 2000,
        "embedding": 48,
        # Deliberately small: the decoder's entire view of the source is one
        # 2*hidden vector, and that vector is the quantity under test.
        "hidden": 64,
        "epochs": 6,
        "batch_size": 128,
        "lr": 3e-3,
        "context_words": [0, 10, 24],
    },
    description="Encoder-decoder with and without attention on date normalization, across content lengths.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Seq2seq with attention on a toy translation task."""
    torch = vision.require_torch()
    Seq2Seq = make_models()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)

    summary: dict[str, float] = {}
    curves: dict[str, list[float]] = {}
    accuracy_by_length: dict[str, list[float]] = {"attention": [], "bottleneck": []}
    example_weights = None
    example_pairs = None

    source_lengths: list[int] = []
    for context_words in exp["context_words"]:
        train_pairs = text.make_date_pairs(exp["train_samples"], context_words=context_words, rng=rng)
        test_pairs = text.make_date_pairs(exp["test_samples"], context_words=context_words, rng=rng)
        # Pad to whatever this content length actually needs, no further.
        source_length = max(len(pair.source) for pair in (*train_pairs, *test_pairs))
        source_lengths.append(source_length)
        x_train, y_train, source_tokenizer, target_tokenizer = text.encode_pairs(
            train_pairs, source_length=source_length
        )
        x_test, y_test, _, _ = text.encode_pairs(test_pairs, source_length=source_length)

        for name in ("bottleneck", "attention"):
            torchkit.seed_torch(exp.seed)
            model = Seq2Seq(
                source_tokenizer.vocab_size,
                target_tokenizer.vocab_size,
                exp["embedding"],
                exp["hidden"],
                attention=name == "attention",
            ).to(device)
            optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"])

            sources = torch.from_numpy(x_train).to(device)
            targets = torch.from_numpy(y_train).to(device)
            # Teacher forcing: the decoder input is the target shifted right,
            # with the pad id doubling as the start symbol.
            decoder_in = torch.cat([torch.zeros_like(targets[:, :1]), targets[:, :-1]], dim=1)

            losses = []
            for _epoch in range(exp["epochs"]):
                model.train()
                permutation = torch.randperm(len(sources), device=device)
                epoch_losses = []
                for start in range(0, len(sources), exp["batch_size"]):
                    index = permutation[start : start + exp["batch_size"]]
                    logits, _ = model(sources[index], decoder_in[index])
                    loss = torch.nn.functional.cross_entropy(
                        logits.reshape(-1, target_tokenizer.vocab_size), targets[index].reshape(-1)
                    )
                    optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    epoch_losses.append(loss.item())
                losses.append(float(np.mean(epoch_losses)))

            # Exact-match accuracy under greedy decoding — no teacher forcing.
            predicted, weights = model.translate(torch.from_numpy(x_test).to(device), max_length=y_test.shape[1])
            exact = float((predicted.cpu().numpy() == y_test).all(axis=1).mean())
            character = float((predicted.cpu().numpy() == y_test).mean())

            arm = f"{name}@{context_words}"
            curves[arm] = losses
            accuracy_by_length[name].append(exact)
            summary |= {
                f"{arm}/exact_match": exact,
                f"{arm}/character_accuracy": character,
                f"{arm}/train_loss": losses[-1],
            }
            print(
                f"[c2] context={context_words:<3} (len {source_length:>3}) {name:<11} "
                f"exact={exact:.4f}  char={character:.4f}  loss={losses[-1]:.4f}"
            )

            # Keep the shortest source for the figure: the alignment is the
            # point, and at 193 characters it is a thin unreadable band. The
            # long-context behaviour is already quantified in the table above.
            if name == "attention" and context_words == exp["context_words"][0] and weights is not None:
                example_weights = weights[:6].cpu().numpy()
                example_pairs = test_pairs[:6]

    # -- does the attention align where it should? --------------------------
    if example_weights is not None and example_pairs is not None:
        score = alignment_score(
            example_weights, [pair.source for pair in example_pairs], [pair.target for pair in example_pairs]
        )
        summary["attention/alignment_score"] = score
        print(f"[c2] attention alignment: {score:.4f} of non-dash output characters peak on a matching input character")

        best = example_pairs[0]
        exp.log_figure(
            "attention_map",
            plotting.plot_attention(
                example_weights[0][:, : len(best.source)],
                source_labels=list(best.source),
                target_labels=list(best.target),
                title=f"Attention: {best.source[:40]!r} -> {best.target!r}",
            ),
        )

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "accuracy_by_length",
        plotting.plot_curves(
            accuracy_by_length,
            x=source_lengths,
            title="Exact-match accuracy vs. source length (filler words around the date)",
            xlabel="source characters",
            ylabel="exact match",
            marker="o",
        ),
    )
    exp.log_figure(
        "training",
        plotting.plot_curves(
            {arm: values for arm, values in curves.items()},
            x=list(range(1, exp["epochs"] + 1)),
            title="Training loss by arm",
            xlabel="epoch",
            ylabel="cross-entropy",
            logy=True,
        ),
    )

    for index, context_words in enumerate(exp["context_words"]):
        summary[f"source_length@{context_words}"] = float(source_lengths[index])
        summary[f"attention_minus_bottleneck@{context_words}"] = (
            accuracy_by_length["attention"][index] - accuracy_by_length["bottleneck"][index]
        )
    summary["attention/mean_exact"] = float(np.mean(accuracy_by_length["attention"]))
    summary["bottleneck/mean_exact"] = float(np.mean(accuracy_by_length["bottleneck"]))
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
