"""C3. Transformer from scratch (attention, positional encoding, causal masking)

Lab reference: [Annotated Transformer / nanoGPT](https://nlp.seas.harvard.edu/annotated-transformer/)

Attention is written out rather than taken from ``nn.MultiheadAttention``: the
shapes and the mask are the parts that are easy to get subtly wrong, and a
causal model that leaks one position into the past still trains beautifully
while being worthless. There is a test for exactly that leak.

Trained on the same corpus, split and budget as C1's recurrent models, so the
comparison is like for like — and ablated against its own positional encoding,
because without one the architecture is permutation-equivariant and that should
be visible as a number, not just asserted.

Run with ``ladder run C3`` (``--help`` for flags).
"""

from __future__ import annotations

import math

import numpy as np

from mlladder import Experiment, experiment, plotting, text, torchkit, vision


def make_transformer():
    torch = vision.require_torch()
    from torch import nn

    def sinusoidal_encoding(length: int, width: int):
        r"""$PE_{(t,2i)} = \sin(t/10000^{2i/d})$, $PE_{(t,2i+1)} = \cos(\cdot)$.

        Each dimension is a sinusoid of a different wavelength, so a fixed
        offset $k$ acts on the pair $(\sin, \cos)$ as a rotation that does not
        depend on $t$ — which is what lets attention express "k steps back".
        """
        position = torch.arange(length).unsqueeze(1).float()
        index = torch.arange(0, width, 2).float()
        divisor = torch.exp(-index * math.log(10000.0) / width)
        encoding = torch.zeros(length, width)
        encoding[:, 0::2] = torch.sin(position * divisor)
        encoding[:, 1::2] = torch.cos(position * divisor)
        return encoding

    class MultiHeadSelfAttention(nn.Module):
        r"""$\mathrm{softmax}\!\left(QK^\top/\sqrt{d_k} + M\right)V$, over $h$ heads in parallel."""

        def __init__(self, width: int, heads: int, *, dropout: float = 0.0) -> None:
            super().__init__()
            if width % heads:
                raise ValueError(f"width {width} must divide into {heads} heads")
            self.heads = heads
            self.head_width = width // heads
            self.qkv = nn.Linear(width, 3 * width)
            self.projection = nn.Linear(width, width)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x, *, causal: bool = True, return_weights: bool = False):
            batch, length, width = x.shape
            q, k, v = self.qkv(x).chunk(3, dim=-1)
            # (batch, heads, length, head_width)
            q, k, v = (t.view(batch, length, self.heads, self.head_width).transpose(1, 2) for t in (q, k, v))

            # The 1/sqrt(d_k) keeps the logits' variance independent of head
            # width; without it softmax saturates and the gradient dies.
            scores = (q @ k.transpose(-2, -1)) / math.sqrt(self.head_width)
            if causal:
                mask = torch.ones(length, length, dtype=torch.bool, device=x.device).triu(1)
                scores = scores.masked_fill(mask, float("-inf"))
            weights = torch.softmax(scores, dim=-1)
            out = (self.dropout(weights) @ v).transpose(1, 2).reshape(batch, length, width)
            out = self.projection(out)
            return (out, weights) if return_weights else (out, None)

    class Block(nn.Module):
        """Pre-LN block: LayerNorm inside the residual branch, not around it.

        Post-LN (the 2017 paper's order) needs learning-rate warmup to train at
        all; pre-LN is what every modern implementation uses, and it is the
        same argument as B4's identity path — keep the skip clean.
        """

        def __init__(self, width: int, heads: int, *, dropout: float = 0.0) -> None:
            super().__init__()
            self.norm1 = nn.LayerNorm(width)
            self.attention = MultiHeadSelfAttention(width, heads, dropout=dropout)
            self.norm2 = nn.LayerNorm(width)
            self.mlp = nn.Sequential(
                nn.Linear(width, 4 * width), nn.GELU(), nn.Linear(4 * width, width), nn.Dropout(dropout)
            )

        def forward(self, x, *, return_weights: bool = False):
            attended, weights = self.attention(self.norm1(x), return_weights=return_weights)
            x = x + attended
            x = x + self.mlp(self.norm2(x))
            return x, weights

    class CharTransformer(nn.Module):
        """A decoder-only transformer over characters."""

        def __init__(
            self,
            vocab_size: int,
            width: int = 192,
            heads: int = 4,
            depth: int = 4,
            block_size: int = 128,
            *,
            positional: str = "sinusoidal",
            dropout: float = 0.0,
        ) -> None:
            super().__init__()
            self.block_size = block_size
            self.positional = positional
            self.token_embedding = nn.Embedding(vocab_size, width)
            if positional == "learned":
                self.position_embedding = nn.Embedding(block_size, width)
            elif positional == "sinusoidal":
                self.register_buffer("position_table", sinusoidal_encoding(block_size, width), persistent=False)
            elif positional != "none":
                raise ValueError(f"unknown positional encoding {positional!r}")
            self.blocks = nn.ModuleList([Block(width, heads, dropout=dropout) for _ in range(depth)])
            self.norm = nn.LayerNorm(width)
            self.head = nn.Linear(width, vocab_size, bias=False)

        def forward(self, ids, *, return_weights: bool = False):
            _, length = ids.shape
            x = self.token_embedding(ids)
            if self.positional == "learned":
                x = x + self.position_embedding(torch.arange(length, device=ids.device))
            elif self.positional == "sinusoidal":
                x = x + self.position_table[:length]

            weights = None
            for block in self.blocks:
                x, block_weights = block(x, return_weights=return_weights)
                if block_weights is not None:
                    weights = block_weights
            return self.head(self.norm(x)), weights

        @torch.no_grad()
        def sample(self, tokenizer, *, length: int = 240, prompt: str = "\n", temperature: float = 0.8, device=None):
            self.eval()
            ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
            for _ in range(length):
                logits, _ = self(ids[:, -self.block_size :])
                probabilities = torch.softmax(logits[0, -1] / temperature, dim=-1)
                ids = torch.cat([ids, torch.multinomial(probabilities, num_samples=1).unsqueeze(0)], dim=1)
            return tokenizer.decode(ids[0].tolist())

    return CharTransformer, sinusoidal_encoding


@experiment(
    "c3_transformer_from_scratch",
    defaults={
        "corpus_chars": 250000,
        "width": 192,
        "heads": 4,
        "depth": 4,
        "block_size": 64,
        "batch_size": 48,
        "steps": 500,
        "lr": 3e-3,
        "eval_every": 50,
        "eval_batches": 8,
        "sample_chars": 240,
        "encodings": ["sinusoidal", "learned", "none"],
        # A second, much smaller model sized to C1's LSTM (~0.32 M parameters),
        # so "transformer vs LSTM" compares like with like instead of
        # comparing 5x the parameter budget.
        "matched_width": 96,
        "matched_depth": 2,
    },
    description="A decoder-only transformer written from scratch, ablated against its positional encoding.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Transformer from scratch (attention, positional encoding, causal masking)."""
    torch = vision.require_torch()
    CharTransformer, sinusoidal_encoding = make_transformer()
    device = torchkit.device()

    corpus = text.tiny_shakespeare(limit=exp["corpus_chars"])
    tokenizer = text.CharTokenizer.from_text(corpus)
    train_ids, val_ids = text.split_corpus(tokenizer.encode(corpus))
    print(f"[c3] {len(corpus)} chars, vocab {tokenizer.vocab_size}")

    def evaluate(model) -> float:
        model.eval()
        losses = []
        eval_rng = np.random.default_rng(12345)  # the same windows C1 evaluates on
        with torch.no_grad():
            for _ in range(exp["eval_batches"]):
                inputs, targets = text.lm_batch(
                    val_ids, batch_size=exp["batch_size"], block_size=exp["block_size"], rng=eval_rng
                )
                logits, _ = model(torch.from_numpy(inputs).to(device))
                losses.append(
                    torch.nn.functional.cross_entropy(
                        logits.reshape(-1, tokenizer.vocab_size), torch.from_numpy(targets).to(device).reshape(-1)
                    ).item()
                )
        return float(np.mean(losses))

    curves: dict[str, list[float]] = {}
    summary: dict[str, float] = {}
    models = {}
    for encoding in exp["encodings"]:
        torchkit.seed_torch(exp.seed)
        model = CharTransformer(
            tokenizer.vocab_size,
            width=exp["width"],
            heads=exp["heads"],
            depth=exp["depth"],
            block_size=exp["block_size"],
            positional=encoding,
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=0.1)
        batch_rng = np.random.default_rng(exp.seed)

        losses = []
        for step in range(1, exp["steps"] + 1):
            model.train()
            inputs, targets = text.lm_batch(
                train_ids, batch_size=exp["batch_size"], block_size=exp["block_size"], rng=batch_rng
            )
            logits, _ = model(torch.from_numpy(inputs).to(device))
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, tokenizer.vocab_size), torch.from_numpy(targets).to(device).reshape(-1)
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % exp["eval_every"] == 0 or step == 1:
                validation = evaluate(model)
                losses.append(validation)
                exp.log({f"{encoding}/train_loss": loss.item(), f"{encoding}/val_loss": validation}, step=step)

        curves[encoding] = losses
        models[encoding] = model
        bits_per_char = losses[-1] / math.log(2)
        summary |= {
            f"{encoding}/val_loss": losses[-1],
            f"{encoding}/bits_per_char": bits_per_char,
            f"{encoding}/parameters": float(torchkit.count_parameters(model)),
        }
        print(
            f"[c3] positional={encoding:<11} val loss={losses[-1]:.4f}  bits/char={bits_per_char:.4f}  "
            f"params={torchkit.count_parameters(model):,}"
        )

    # -- the same architecture at C1's parameter budget ---------------------
    torchkit.seed_torch(exp.seed)
    matched = CharTransformer(
        tokenizer.vocab_size,
        width=exp["matched_width"],
        heads=exp["heads"],
        depth=exp["matched_depth"],
        block_size=exp["block_size"],
        positional="learned",
    ).to(device)
    optimizer = torch.optim.AdamW(matched.parameters(), lr=exp["lr"], weight_decay=0.1)
    batch_rng = np.random.default_rng(exp.seed)
    matched_losses = []
    for step in range(1, exp["steps"] + 1):
        matched.train()
        inputs, targets = text.lm_batch(
            train_ids, batch_size=exp["batch_size"], block_size=exp["block_size"], rng=batch_rng
        )
        logits, _ = matched(torch.from_numpy(inputs).to(device))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, tokenizer.vocab_size), torch.from_numpy(targets).to(device).reshape(-1)
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(matched.parameters(), 1.0)
        optimizer.step()
        if step % exp["eval_every"] == 0 or step == 1:
            matched_losses.append(evaluate(matched))
    curves["matched"] = matched_losses
    models["matched"] = matched
    matched_parameters = torchkit.count_parameters(matched)
    summary |= {
        "matched/val_loss": matched_losses[-1],
        "matched/bits_per_char": matched_losses[-1] / math.log(2),
        "matched/parameters": float(matched_parameters),
    }
    print(
        f"[c3] parameter-matched  val loss={matched_losses[-1]:.4f}  "
        f"bits/char={matched_losses[-1] / math.log(2):.4f}  params={matched_parameters:,}"
    )

    best = min(curves, key=lambda key: curves[key][-1])
    sample = models[best].sample(tokenizer, length=exp["sample_chars"], device=device)
    print(f"\n[c3] --- sample ({best}) ---\n{sample}\n")
    exp.path("sample.txt").write_text(sample)

    # -- the causal mask, checked rather than assumed ------------------------
    model = models[best]
    model.eval()
    probe_rng = np.random.default_rng(7)
    inputs, _ = text.lm_batch(val_ids, batch_size=2, block_size=exp["block_size"], rng=probe_rng)
    ids = torch.from_numpy(inputs).to(device)
    with torch.no_grad():
        original, weights = model(ids, return_weights=True)
        tampered_ids = ids.clone()
        tampered_ids[:, -1] = (tampered_ids[:, -1] + 1) % tokenizer.vocab_size  # change the future
        tampered, _ = model(tampered_ids)
    leak = float((original[:, :-1] - tampered[:, :-1]).abs().max())
    upper_mass = float(weights.triu(1).abs().max())
    print(
        f"[c3] causal check: changing the last token moves earlier logits by {leak:.2e}; mass above the diagonal {upper_mass:.2e}"
    )
    summary["causal/leak"] = leak
    summary["causal/upper_triangular_mass"] = upper_mass

    # -- figures -------------------------------------------------------------
    steps_axis = [1, *range(exp["eval_every"], exp["steps"] + 1, exp["eval_every"])][: len(curves[best])]
    exp.log_figure(
        "positional_ablation",
        plotting.plot_curves(
            curves,
            x=steps_axis,
            title="Transformer — validation loss by positional encoding",
            xlabel="step",
            ylabel="cross-entropy",
        ),
    )
    exp.log_figure(
        "bits_per_char",
        plotting.plot_comparison(
            {encoding: summary[f"{encoding}/bits_per_char"] for encoding in curves},
            title="Validation bits per character",
            ylabel="bits/char",
        ),
    )
    exp.log_figure(
        "attention_map",
        plotting.plot_attention(
            weights[0, 0, :48, :48].cpu().numpy(),
            title="Head 0, last block — causal self-attention",
            xlabel="key position (attended to)",
            ylabel="query position",
        ),
    )
    table = sinusoidal_encoding(exp["block_size"], exp["width"])
    similarity = (torch.nn.functional.normalize(table, dim=1) @ torch.nn.functional.normalize(table, dim=1).T).numpy()
    exp.log_figure(
        "positional_similarity",
        plotting.plot_confusion_matrix(
            similarity, title="Sinusoidal encoding — cosine similarity between positions", normalize=False
        ),
    )

    # Every comparison below is only defined when both of its arms were run,
    # and --encodings lets a caller run any subset.
    def difference(name: str, left: str, right: str) -> None:
        if f"{left}/bits_per_char" in summary and f"{right}/bits_per_char" in summary:
            summary[name] = summary[f"{left}/bits_per_char"] - summary[f"{right}/bits_per_char"]

    difference("none_minus_sinusoidal_bpc", "none", "sinusoidal")
    difference("learned_minus_sinusoidal_bpc", "learned", "sinusoidal")
    difference("matched_minus_full_bpc", "matched", "learned")
    summary["best_bits_per_char"] = min(summary[f"{encoding}/bits_per_char"] for encoding in curves)
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
