"""C1. Char-level RNN/LSTM language model

Lab reference: [Annotated Transformer / nanoGPT](https://nlp.seas.harvard.edu/annotated-transformer/)

Both cells are written out rather than taken from ``nn.RNN``/``nn.LSTM``, because
the whole point of the rung is the difference between them: a vanilla recurrent
cell multiplies its state by the same matrix at every step, and an LSTM adds a
path along which the state is *added to* rather than multiplied. That is the
same trick as B4's residual connection, one year earlier and in time rather
than depth.

The measurement that separates them is not the loss — it is how far back a
gradient still carries.

Run with ``ladder run C1`` (``--help`` for flags).
"""

from __future__ import annotations

import math

import numpy as np

from mlladder import Experiment, experiment, plotting, text, torchkit, vision


def make_cells():
    """Define the two cells against the torch imported at call time."""
    torch = vision.require_torch()
    from torch import nn

    class VanillaRNNCell(nn.Module):
        r"""$h_t = \tanh(x_t W_x + h_{t-1} W_h + b)$.

        The state passes through $W_h$ and a saturating nonlinearity at *every*
        step, so a gradient reaching back $k$ steps carries a factor
        $\prod \mathrm{diag}(1-h^2) W_h^\top$ — geometric in $k$.
        """

        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__()
            self.input_to_hidden = nn.Linear(input_size, hidden_size, bias=False)
            self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size)
            self.hidden_size = hidden_size

        def forward(self, x, state):
            h = state[0]
            h = torch.tanh(self.input_to_hidden(x) + self.hidden_to_hidden(h))
            return h, (h,)

        def initial_state(self, batch_size: int, device):
            return (torch.zeros(batch_size, self.hidden_size, device=device),)

    class LSTMCell(nn.Module):
        r"""The four gates, computed in one matrix multiply.

        $i, f, o = \sigma(\cdot)$, $g = \tanh(\cdot)$, then
        $c_t = f \odot c_{t-1} + i \odot g$ and $h_t = o \odot \tanh(c_t)$.
        The cell state's recurrence is *additive* with a multiplier $f$ the
        network controls, which is where the long-range gradient survives.
        """

        def __init__(self, input_size: int, hidden_size: int) -> None:
            super().__init__()
            self.gates = nn.Linear(input_size + hidden_size, 4 * hidden_size)
            self.hidden_size = hidden_size
            # Forget-gate bias at 1: start by remembering, and let the network
            # learn to forget. Standard trick, and it matters at this scale.
            with torch.no_grad():
                self.gates.bias[hidden_size : 2 * hidden_size].fill_(1.0)

        def forward(self, x, state):
            h, c = state
            i, f, g, o = self.gates(torch.cat([x, h], dim=1)).chunk(4, dim=1)
            c = torch.sigmoid(f) * c + torch.sigmoid(i) * torch.tanh(g)
            h = torch.sigmoid(o) * torch.tanh(c)
            return h, (h, c)

        def initial_state(self, batch_size: int, device):
            zeros = torch.zeros(batch_size, self.hidden_size, device=device)
            return (zeros, zeros.clone())

    class CharLanguageModel(nn.Module):
        """Embedding, a stack of recurrent cells, and a tied-free output head."""

        def __init__(self, vocab_size: int, hidden_size: int, cell: str = "lstm", layers: int = 1) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, hidden_size)
            cell_class = LSTMCell if cell == "lstm" else VanillaRNNCell
            self.cells = nn.ModuleList([cell_class(hidden_size, hidden_size) for _ in range(layers)])
            self.head = nn.Linear(hidden_size, vocab_size)
            self.cell_name = cell

        def forward(self, ids, states=None, *, return_hidden: bool = False):
            batch, length = ids.shape
            embedded = self.embedding(ids)
            if states is None:
                states = [cell.initial_state(batch, ids.device) for cell in self.cells]

            outputs, hidden_trace = [], []
            for step in range(length):
                x = embedded[:, step]
                for index, cell in enumerate(self.cells):
                    x, states[index] = cell(x, states[index])
                outputs.append(x)
                if return_hidden:
                    hidden_trace.append(x)
            logits = self.head(torch.stack(outputs, dim=1))
            return (logits, states, hidden_trace) if return_hidden else (logits, states)

        @torch.no_grad()
        def sample(self, tokenizer, *, length: int = 200, prompt: str = "\n", temperature: float = 0.8, device=None):
            self.eval()
            ids = torch.tensor(tokenizer.encode(prompt), device=device).unsqueeze(0)
            states = None
            generated = list(prompt)
            for _ in range(length):
                logits, states = self(ids[:, -1:], states)
                probabilities = torch.softmax(logits[0, -1] / temperature, dim=-1)
                next_id = torch.multinomial(probabilities, num_samples=1)
                generated.append(tokenizer.characters[int(next_id)])
                ids = next_id.unsqueeze(0)
            return "".join(generated)

    return VanillaRNNCell, LSTMCell, CharLanguageModel


def gradient_reach(model, ids, *, device) -> list[float]:
    r"""How much of the gradient at the last step survives back to step $t$.

    Differentiates the final position's loss with respect to each *embedding*
    along the sequence, giving $\lVert\partial L_T/\partial x_t\rVert$ as a
    function of distance — the empirical version of the vanishing-gradient
    product above.
    """
    vision.require_torch()
    model.zero_grad(set_to_none=True)

    embedded = model.embedding(ids).detach().requires_grad_(True)
    batch, length, _ = embedded.shape
    states = [cell.initial_state(batch, device) for cell in model.cells]
    for step in range(length):
        x = embedded[:, step]
        for index, cell in enumerate(model.cells):
            x, states[index] = cell(x, states[index])
    loss = model.head(x).pow(2).sum()  # a scalar that depends on the final state only
    loss.backward()

    norms = embedded.grad.norm(dim=(0, 2))
    return [float(value) for value in norms]


@experiment(
    "c1_char_rnn",
    defaults={
        "corpus_chars": 250000,
        "hidden": 192,
        "layers": 1,
        "block_size": 64,
        "batch_size": 48,
        "steps": 500,
        "lr": 3e-3,
        "eval_every": 50,
        "eval_batches": 8,
        "sample_chars": 240,
    },
    description="Vanilla RNN vs LSTM as char-level language models, compared on loss and on gradient reach.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Char-level RNN/LSTM language model."""
    torch = vision.require_torch()
    _, _, CharLanguageModel = make_cells()
    device = torchkit.device()

    corpus = text.tiny_shakespeare(limit=exp["corpus_chars"])
    tokenizer = text.CharTokenizer.from_text(corpus)
    train_ids, val_ids = text.split_corpus(tokenizer.encode(corpus))
    print(f"[c1] {len(corpus)} chars, vocab {tokenizer.vocab_size}, {len(train_ids)} train / {len(val_ids)} val")

    def evaluate(model) -> float:
        model.eval()
        losses = []
        eval_rng = np.random.default_rng(12345)  # same windows for every model
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

    # An LSTM cell has four gates, so at equal hidden size it carries ~4x the
    # parameters of a vanilla cell. Doubling the RNN's width matches the budget
    # (both scale as hidden^2), which is the only way the comparison is fair.
    arms = (
        ("rnn", exp["hidden"]),
        ("rnn-wide", exp["hidden"] * 2),
        ("lstm", exp["hidden"]),
    )

    curves: dict[str, list[float]] = {}
    summary: dict[str, float] = {}
    models = {}
    for arm, hidden in arms:
        cell = "lstm" if arm == "lstm" else "rnn"
        torchkit.seed_torch(exp.seed)
        model = CharLanguageModel(tokenizer.vocab_size, hidden, cell=cell, layers=exp["layers"]).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"])
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
            # Recurrent nets need this: one bad batch can otherwise blow the
            # state up through the whole unrolled product.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            if step % exp["eval_every"] == 0 or step == 1:
                validation = evaluate(model)
                losses.append(validation)
                exp.log({f"{arm}/train_loss": loss.item(), f"{arm}/val_loss": validation}, step=step)

        curves[arm] = losses
        models[arm] = model
        bits_per_char = losses[-1] / math.log(2)
        summary |= {
            f"{arm}/val_loss": losses[-1],
            f"{arm}/bits_per_char": bits_per_char,
            f"{arm}/parameters": float(torchkit.count_parameters(model)),
        }
        print(
            f"[c1] {arm:<9} hidden={hidden:<4} val loss={losses[-1]:.4f}  bits/char={bits_per_char:.4f}  "
            f"params={torchkit.count_parameters(model):,}"
        )

    # -- how far back does a gradient survive? ------------------------------
    reach_rng = np.random.default_rng(999)
    inputs, _ = text.lm_batch(val_ids, batch_size=16, block_size=exp["block_size"], rng=reach_rng)
    ids = torch.from_numpy(inputs).to(device)
    reaches: dict[str, list[float]] = {}
    for arm, model in models.items():
        norms = gradient_reach(model, ids, device=device)
        reaches[arm] = norms
        # Normalize by the gradient at the final position: the question is the
        # shape of the decay, not the scale.
        relative = norms[0] / max(norms[-1], 1e-30)
        summary[f"{arm}/gradient_reach_ratio"] = relative
        summary[f"{arm}/gradient_at_max_distance"] = norms[0]
        # How far back does the gradient stay above a millionth of its value at
        # the final position? That is the cell's effective memory, measured.
        threshold = norms[-1] * 1e-6
        survived = [index for index, value in enumerate(norms) if value >= threshold]
        horizon = float(len(norms) - 1 - min(survived)) if survived else 0.0
        summary[f"{arm}/effective_horizon"] = horizon
        print(
            f"[c1] {arm:<9} gradient at distance {exp['block_size'] - 1}: {norms[0]:.3e}  "
            f"(final step {norms[-1]:.3e}, ratio {relative:.2e}, horizon {horizon:.0f} steps)"
        )

    for arm, model in models.items():
        sample = model.sample(tokenizer, length=exp["sample_chars"], device=device, prompt="\n")
        print(f"\n[c1] --- {arm} sample ---\n{sample}\n")
        exp.path(f"sample_{arm}.txt").write_text(sample)

    # -- figures -------------------------------------------------------------
    steps_axis = [1, *range(exp["eval_every"], exp["steps"] + 1, exp["eval_every"])][: len(curves["lstm"])]
    exp.log_figure(
        "validation_loss",
        plotting.plot_curves(
            curves, x=steps_axis, title="Char LM — validation loss", xlabel="step", ylabel="cross-entropy"
        ),
    )
    exp.log_figure(
        "gradient_reach",
        plotting.plot_curves(
            dict(reaches),
            x=list(range(-(exp["block_size"] - 1), 1)),
            title="Gradient norm at the final step w.r.t. earlier inputs",
            xlabel="distance from the final position",
            ylabel="gradient norm",
            logy=True,
        ),
    )
    exp.log_figure(
        "bits_per_char",
        plotting.plot_comparison(
            {arm: summary[f"{arm}/bits_per_char"] for arm in curves},
            title="Validation bits per character",
            ylabel="bits/char",
        ),
    )

    summary["lstm_minus_rnn_bpc"] = summary["lstm/bits_per_char"] - summary["rnn/bits_per_char"]
    summary["reach_ratio_lstm_over_rnn"] = summary["lstm/gradient_reach_ratio"] / max(
        summary["rnn/gradient_reach_ratio"], 1e-30
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
