r"""G1. Fit your own scaling-law curve (loss vs. params/data/compute) on a small model family

Lab reference: [PyTorch mixed-precision & profiler docs](https://pytorch.org/docs/stable/notes/amp_examples.html)

Every rung so far trained *a* model and asked how well it did. A scaling law
asks a different question: given a family of models and a knob, how does the
loss move when you turn it? The answer, empirically, is a power law with a
floor,

$$L(x) = E + \frac{A}{x^{\alpha}},$$

and the reason that shape matters is that it **extrapolates**. A curve fitted on
models you can afford predicts the loss of a model you cannot, which is the only
reason anyone spends eight figures on a training run before seeing it work.

This rung fits that curve three times over, on the char transformer built in
rung C3, because the interesting content is not the fit itself but the three
things that have to be true for a fit to mean anything:

1. **The protocol has to hold the other variables still.** Every run here gets
   a cosine schedule matched to its own token budget and exactly one pass over
   fresh data, because a model trained twice as long is not a bigger model.
2. **The compute axis has to be counted, not assumed.** Everyone quotes
   $C = 6ND$. That formula drops the attention term, and this rung measures how
   much it drops at these shapes -- which is not a rounding error.
3. **The claim that only $N$ matters is testable.** Kaplan et al. report that
   loss depends on parameter count and barely on how those parameters are
   arranged. Two models with the same $N$ and different (width, depth) settle
   the question here in one comparison.

Then the compute-optimal question: at a fixed budget $C$, is it better to train
a big model briefly or a small model for longer? Sweeping $N$ along an IsoFLOP
contour locates the minimum, and how that minimum moves with $C$ is the
Chinchilla exponent.

Run with ``ladder run G1`` (``--help`` for flags).
"""

from __future__ import annotations

import math
import time

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, experiment, plotting, text, torchkit, vision


def transformer_flops(*, width: int, depth: int, block_size: int, vocab_size: int) -> dict[str, float]:
    r"""FLOPs per *token* of one forward pass, counted term by term.

    Convention: a multiply-accumulate is 2 FLOPs. Per layer, for a sequence of
    length $L$ and width $d$, the matrix multiplications are

    | term | FLOPs per sequence |
    | --- | --- |
    | QKV projection | $6Ld^2$ |
    | $QK^\top$ | $2L^2d$ |
    | $AV$ | $2L^2d$ |
    | output projection | $2Ld^2$ |
    | MLP (4x expansion) | $16Ld^2$ |

    The $d^2$ terms sum to $24Ld^2 = 2L \cdot 12d^2$, and $12d^2$ is exactly the
    layer's parameter count -- which is where "$2N$ per token" comes from, and
    why it is *only* the $d^2$ terms. What is left over is $4L^2d$ per layer, or
    $4Ld$ per token: the attention term the $6ND$ rule silently drops.

    Backward is taken as twice forward (one gradient for the input, one for the
    weights), giving the familiar $6N$ per token, so

    $$C \approx 6ND + 12 \cdot n_{\text{layer}} \cdot L \cdot d \cdot D .$$

    The second term is not negligible here and the rung reports its share.
    """
    per_layer_params = 12 * width * width
    dense = 2.0 * depth * per_layer_params
    attention = 4.0 * depth * block_size * width
    head = 2.0 * vocab_size * width
    return {
        "dense": dense,
        "attention": attention,
        "head": head,
        "forward": dense + attention + head,
        # What the 6ND rule counts, and what it misses.
        "rule_of_thumb": dense,
        "attention_share": attention / (dense + attention),
    }


def non_embedding_parameters(model) -> int:
    """Parameters excluding the token embedding, the output head, and *positions*.

    $N$ in a scaling law is conventionally the non-embedding count, and the
    reason is a scaling argument rather than a bookkeeping one: the body grows
    as $d^2$ while every embedding table grows as $d$, so including them bends
    the curve for no modelling reason.

    The positional table is easy to miss and it matters here. At width 32 with a
    64-token context it is 2,048 of 27,520 parameters — 7.4% — and its share
    shrinks as the model widens, which is exactly the shape of contamination
    that would flatten a fitted exponent.
    """
    total = sum(p.numel() for p in model.parameters())
    embedding = model.token_embedding.weight.numel() + model.head.weight.numel()
    position = getattr(model, "position_embedding", None)
    if position is not None:
        embedding += position.weight.numel()
    return total - embedding


def fit_power_law(x: NDArray[np.float64], y: NDArray[np.float64]) -> dict[str, float]:
    r"""Least squares for $y = E + A x^{-\alpha}$, without scipy.

    Nonlinear in $E$ and linear in everything else once $E$ is fixed: for a
    candidate floor, $\log(y - E) = \log A - \alpha \log x$ is an ordinary
    regression. So profile over $E$ on a grid below $\min y$ and keep the one
    whose residual is smallest.

    Writing the fit out rather than importing it is not only about dependencies.
    **The floor is the parameter that matters and the one that is least
    identified**: over a narrow range of $x$, a power law with a floor and a
    pure power law are nearly indistinguishable, and the fitted $E$ moves a lot
    for very little residual. The grid makes that visible instead of handing
    back one confident number.
    """
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    log_x = np.log(x)
    total = float(np.sum((y - y.mean()) ** 2))

    floors = np.linspace(0.0, float(y.min()) * 0.995, 400)
    records: list[dict[str, float]] = []
    for floor in floors:
        residuals_y = y - floor
        if np.any(residuals_y <= 0):
            continue
        slope, intercept = np.polyfit(log_x, np.log(residuals_y), 1)
        predicted = floor + np.exp(intercept) * x**slope
        records.append(
            {
                "irreducible": float(floor),
                "coefficient": float(np.exp(intercept)),
                "exponent": float(-slope),
                "residual": float(np.sum((y - predicted) ** 2)),
            }
        )
    if not records:
        return {"irreducible": 0.0, "coefficient": float("nan"), "exponent": float("nan"), "r2": float("nan")}

    best = min(records, key=lambda record: record["residual"])
    best["r2"] = 1.0 - best["residual"] / total if total > 0 else float("nan")

    # How badly identified the floor is: every floor whose fit leaves at most
    # 20% more squared residual than the best one. Scale-free on purpose, so
    # the band collapses when the points really do pin the floor down and opens
    # when they do not -- which, over the half-decade of $x$ a sweep you can
    # afford actually covers, is most of the time.
    threshold = best["residual"] * 1.2 + 1e-9 * max(total, 1e-12)
    plausible = [record for record in records if record["residual"] <= threshold]
    best["irreducible_low"] = min(record["irreducible"] for record in plausible)
    best["irreducible_high"] = max(record["irreducible"] for record in plausible)
    best["exponent_low"] = min(record["exponent"] for record in plausible)
    best["exponent_high"] = max(record["exponent"] for record in plausible)
    return best


def power_law_curve(fit: dict[str, float], x: NDArray[np.float64]) -> NDArray[np.float64]:
    return fit["irreducible"] + fit["coefficient"] * np.asarray(x, dtype=np.float64) ** (-fit["exponent"])


def cosine_schedule(step: int, total: int, *, warmup: float = 0.05, floor: float = 0.1) -> float:
    """Warmup then cosine decay, as a multiplier on the base learning rate.

    Matched to each run's own budget, which is the part that makes the sweep a
    sweep. A shared schedule would leave the short runs stopped mid-decay at a
    high learning rate and the long runs decayed to nothing early, and the
    resulting curve would be a picture of the schedule.
    """
    warmup_steps = max(1, int(total * warmup))
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, total - warmup_steps)
    return floor + (1 - floor) * 0.5 * (1 + math.cos(math.pi * progress))


def train_to_budget(
    *,
    exp: Experiment,
    tokens: int,
    width: int,
    depth: int,
    train_ids,
    val_ids,
    vocab_size: int,
    device,
    seed: int,
) -> dict[str, float]:
    """Train one member of the family on exactly ``tokens`` tokens.

    One pass' worth of *fresh* windows rather than epochs over a small slice:
    a scaling law in $D$ is about how much data the model saw, and repeating
    data measures memorization instead. tiny-shakespeare is 1.1M characters, so
    budgets above that do repeat -- which the rung reports as ``epochs_of_data``
    rather than hiding.
    """
    torch = vision.require_torch()
    from parts.c_sequence_attention.c3_transformer_from_scratch import make_transformer

    CharTransformer, _ = make_transformer()
    torchkit.seed_torch(seed)
    model = CharTransformer(
        vocab_size,
        width=width,
        heads=exp["heads"],
        depth=depth,
        block_size=exp["block_size"],
        positional="learned",
    ).to(device)
    parameters = non_embedding_parameters(model)

    tokens_per_step = exp["batch_size"] * exp["block_size"]
    steps = max(1, round(tokens / tokens_per_step))
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=0.1, betas=(0.9, 0.95))
    rng = np.random.default_rng(seed)

    started = time.time()
    model.train()
    for step in range(steps):
        scale = cosine_schedule(step, steps)
        for group in optimizer.param_groups:
            group["lr"] = exp["lr"] * scale
        inputs, targets = text.lm_batch(train_ids, batch_size=exp["batch_size"], block_size=exp["block_size"], rng=rng)
        x = torch.from_numpy(inputs).to(device)
        y = torch.from_numpy(targets).to(device)
        optimizer.zero_grad(set_to_none=True)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
    elapsed = time.time() - started

    validation = evaluate(model, val_ids, exp=exp, vocab_size=vocab_size, device=device, seed=seed)
    flops = transformer_flops(width=width, depth=depth, block_size=exp["block_size"], vocab_size=vocab_size)
    realized = steps * tokens_per_step
    return {
        "width": float(width),
        "depth": float(depth),
        "parameters": float(parameters),
        "tokens": float(realized),
        "steps": float(steps),
        "val_loss": validation,
        "compute": 3.0 * flops["forward"] * realized,
        "compute_rule": 3.0 * flops["rule_of_thumb"] * realized,
        "attention_share": flops["attention_share"],
        "epochs_of_data": realized / len(train_ids),
        "seconds": elapsed,
    }


def evaluate(model, ids, *, exp: Experiment, vocab_size: int, device, seed: int) -> float:
    """Held-out loss, averaged over a fixed set of windows.

    Fixed because the comparison across family members has to see the same
    validation windows; a freshly drawn evaluation set would add noise on the
    axis the fit is most sensitive to.
    """
    torch = vision.require_torch()
    model.eval()
    rng = np.random.default_rng(seed + 10_000)
    total = 0.0
    with torch.no_grad():
        for _ in range(exp["eval_batches"]):
            inputs, targets = text.lm_batch(ids, batch_size=exp["batch_size"], block_size=exp["block_size"], rng=rng)
            x = torch.from_numpy(inputs).to(device)
            y = torch.from_numpy(targets).to(device)
            logits, _ = model(x)
            total += float(torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1)))
    return total / exp["eval_batches"]


@experiment(
    "g1_scaling_law_fit",
    defaults={
        "heads": 4,
        "block_size": 64,
        "batch_size": 32,
        "lr": 3e-3,
        "eval_batches": 16,
        # The N sweep: one depth, five widths, one token budget.
        "widths": [32, 48, 64, 96, 128],
        "depth": 2,
        # One pass over the 1.0M-token training split. Budgets beyond it repeat
        # data, and a model that has seen a token twice is not in the same
        # regime as one that has not.
        "budget_tokens": 1_000_000,
        # The D sweep: one width, five budgets.
        "data_width": 64,
        "data_budgets": [125_000, 250_000, 500_000, 1_000_000],
        # "Only N matters": same parameter count, different arrangement.
        # 2 x 12 x 96^2 and 8 x 12 x 48^2 are the same number: identical dense
        # parameter counts, four times the depth at half the width.
        "shape_pairs": [[96, 2], [48, 8]],
        # IsoFLOP contours, in FLOPs. Each is swept over the widths below.
        "isoflop_budgets": [3e11, 9e11, 2.7e12],
        "isoflop_widths": [32, 48, 64, 96],
        # Block sizes for the FLOP-accounting figure (no training).
        "flop_block_sizes": [32, 64, 128, 256, 512, 1024],
    },
    description="Power-law fits of loss against parameters, data and compute on a char-transformer family.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Fit your own scaling-law curve (loss vs. params/data/compute) on a small model family"""
    vision.require_torch()
    device = torchkit.device()
    print(f"[g1] device: {device}")

    corpus = text.tiny_shakespeare()
    tokenizer = text.CharTokenizer.from_text(corpus)
    ids = tokenizer.encode(corpus)
    train_ids, val_ids = text.split_corpus(ids)
    vocab_size = tokenizer.vocab_size
    print(f"[g1] {len(ids):,} characters, vocab {vocab_size}, {len(train_ids):,} train / {len(val_ids):,} val")

    summary: dict[str, float] = {"corpus/tokens": float(len(ids)), "corpus/vocab": float(vocab_size)}

    def fit(**kwargs) -> dict[str, float]:
        return train_to_budget(
            exp=exp,
            train_ids=train_ids,
            val_ids=val_ids,
            vocab_size=vocab_size,
            device=device,
            seed=exp.seed,
            **kwargs,
        )

    # -- L(N) at a fixed token budget -----------------------------------------
    parameter_sweep = [fit(tokens=exp["budget_tokens"], width=w, depth=exp["depth"]) for w in exp["widths"]]
    for record in parameter_sweep:
        summary[f"N/{int(record['width'])}/parameters"] = record["parameters"]
        summary[f"N/{int(record['width'])}/val_loss"] = record["val_loss"]
        print(
            f"[g1] N sweep width {int(record['width']):>4}  {int(record['parameters']):>8,} params  "
            f"loss {record['val_loss']:.4f}  ({record['seconds']:.0f}s)"
        )
    n_values = np.array([r["parameters"] for r in parameter_sweep])
    n_losses = np.array([r["val_loss"] for r in parameter_sweep])
    n_fit = fit_power_law(n_values, n_losses)
    summary["law/N/exponent"] = n_fit["exponent"]
    summary["law/N/irreducible"] = n_fit["irreducible"]
    summary["law/N/r2"] = n_fit["r2"]
    summary["law/N/irreducible_low"] = n_fit["irreducible_low"]
    summary["law/N/irreducible_high"] = n_fit["irreducible_high"]
    summary["law/N/exponent_low"] = n_fit["exponent_low"]
    summary["law/N/exponent_high"] = n_fit["exponent_high"]
    print(
        f"[g1] L(N) = {n_fit['irreducible']:.3f} + {n_fit['coefficient']:.3g}/N^{n_fit['exponent']:.3f}  "
        f"(R^2 {n_fit['r2']:.4f}); floors from {n_fit['irreducible_low']:.2f} to "
        f"{n_fit['irreducible_high']:.2f} fit essentially as well, with exponents "
        f"{n_fit['exponent_low']:.3f} to {n_fit['exponent_high']:.3f}"
    )

    # -- L(D) at a fixed model size -------------------------------------------
    data_sweep = [fit(tokens=budget, width=exp["data_width"], depth=exp["depth"]) for budget in exp["data_budgets"]]
    for record in data_sweep:
        summary[f"D/{int(record['tokens'])}/val_loss"] = record["val_loss"]
        summary[f"D/{int(record['tokens'])}/epochs_of_data"] = record["epochs_of_data"]
        print(
            f"[g1] D sweep {int(record['tokens']):>9,} tokens ({record['epochs_of_data']:.1f} epochs)  "
            f"loss {record['val_loss']:.4f}"
        )
    d_values = np.array([r["tokens"] for r in data_sweep])
    d_losses = np.array([r["val_loss"] for r in data_sweep])
    d_fit = fit_power_law(d_values, d_losses)
    summary["law/D/exponent"] = d_fit["exponent"]
    summary["law/D/irreducible"] = d_fit["irreducible"]
    summary["law/D/r2"] = d_fit["r2"]
    summary["law/D/irreducible_low"] = d_fit["irreducible_low"]
    summary["law/D/irreducible_high"] = d_fit["irreducible_high"]
    summary["law/D/exponent_low"] = d_fit["exponent_low"]
    summary["law/D/exponent_high"] = d_fit["exponent_high"]
    print(
        f"[g1] L(D) = {d_fit['irreducible']:.3f} + {d_fit['coefficient']:.3g}/D^{d_fit['exponent']:.3f}  "
        f"(R^2 {d_fit['r2']:.4f}); floors from {d_fit['irreducible_low']:.2f} to "
        f"{d_fit['irreducible_high']:.2f} fit essentially as well"
    )

    # -- does the arrangement matter, or only the count? ----------------------
    shapes = [fit(tokens=exp["budget_tokens"], width=w, depth=d) for w, d in exp["shape_pairs"]]
    for record in shapes:
        summary[f"shape/{int(record['width'])}x{int(record['depth'])}/parameters"] = record["parameters"]
        summary[f"shape/{int(record['width'])}x{int(record['depth'])}/val_loss"] = record["val_loss"]
    if len(shapes) == 2:
        summary["shape/parameter_ratio"] = shapes[0]["parameters"] / shapes[1]["parameters"]
        summary["shape/loss_gap"] = abs(shapes[0]["val_loss"] - shapes[1]["val_loss"])
        # The same gap read against the curve: how much of a parameter change
        # would produce it, if only N mattered.
        summary["shape/gap_in_law_terms"] = summary["shape/loss_gap"] / max(
            abs(n_losses[0] - n_losses[-1]) / max(len(n_losses) - 1, 1), 1e-9
        )
        print(
            f"[g1] shape: {int(shapes[0]['width'])}x{int(shapes[0]['depth'])} "
            f"({int(shapes[0]['parameters']):,} params) loss {shapes[0]['val_loss']:.4f} vs "
            f"{int(shapes[1]['width'])}x{int(shapes[1]['depth'])} "
            f"({int(shapes[1]['parameters']):,} params) loss {shapes[1]['val_loss']:.4f} -- "
            f"gap {summary['shape/loss_gap']:.4f}"
        )

    # -- IsoFLOP: the compute-optimal question --------------------------------
    contours: dict[str, list[float]] = {}
    contour_x: dict[str, list[float]] = {}
    optimal_params: list[float] = []
    optimal_tokens: list[float] = []
    max_epochs = 0.0
    for budget in exp["isoflop_budgets"]:
        losses: list[float] = []
        sizes: list[float] = []
        tokens_used: list[float] = []
        for width in exp["isoflop_widths"]:
            # Solve C = 3 * flops_per_token * D for the tokens this width gets.
            per_token = transformer_flops(
                width=width, depth=exp["depth"], block_size=exp["block_size"], vocab_size=vocab_size
            )["forward"]
            tokens = budget / (3.0 * per_token)
            record = fit(tokens=tokens, width=width, depth=exp["depth"])
            losses.append(record["val_loss"])
            sizes.append(record["parameters"])
            tokens_used.append(record["tokens"])
            max_epochs = max(max_epochs, record["epochs_of_data"])
            summary[f"isoflop/{budget:.3g}/{width}/val_loss"] = record["val_loss"]
            summary[f"isoflop/{budget:.3g}/{width}/tokens"] = record["tokens"]
        best = int(np.argmin(losses))
        optimal_params.append(sizes[best])
        optimal_tokens.append(tokens_used[best])
        label = f"C={budget:.1g}"
        contours[label] = losses
        contour_x[label] = sizes
        summary[f"isoflop/{budget:.3g}/optimal_parameters"] = sizes[best]
        summary[f"isoflop/{budget:.3g}/optimal_tokens"] = tokens_used[best]
        summary[f"isoflop/{budget:.3g}/optimal_width"] = float(exp["isoflop_widths"][best])
        summary[f"isoflop/{budget:.3g}/best_loss"] = losses[best]
        # An interior minimum is what makes the contour informative; one that
        # sits at an end means the answer is off the grid.
        summary[f"isoflop/{budget:.3g}/interior_minimum"] = float(0 < best < len(losses) - 1)
        print(
            f"[g1] isoflop C={budget:.2g}: best width {exp['isoflop_widths'][best]} "
            f"({int(sizes[best]):,} params, {int(tokens_used[best]):,} tokens) loss {losses[best]:.4f}"
            f"{'' if 0 < best < len(losses) - 1 else '  [minimum is at the edge of the grid]'}"
        )

    summary["isoflop/max_epochs_of_data"] = max_epochs
    if max_epochs > 1.5:
        print(
            f"[g1] note: the largest IsoFLOP point repeated the training split {max_epochs:.1f} times -- "
            f"at this corpus size the compute axis and the data axis cannot be separated at the top end"
        )

    if len(optimal_params) >= 2:
        # Chinchilla: N* ~ C^a with a ~ 0.5, and D* ~ C^(1-a).
        slope_n = float(np.polyfit(np.log(exp["isoflop_budgets"]), np.log(optimal_params), 1)[0])
        slope_d = float(np.polyfit(np.log(exp["isoflop_budgets"]), np.log(optimal_tokens), 1)[0])
        summary["isoflop/params_exponent"] = slope_n
        summary["isoflop/tokens_exponent"] = slope_d
        summary["isoflop/exponents_sum"] = slope_n + slope_d
        summary["isoflop/tokens_per_parameter"] = optimal_tokens[-1] / optimal_params[-1]
        print(
            f"[g1] compute-optimal: N* ~ C^{slope_n:.3f}, D* ~ C^{slope_d:.3f} "
            f"(Chinchilla: 0.5 and 0.5; the two must sum to 1 by construction, measured {slope_n + slope_d:.3f})"
        )

    # -- what the 6ND rule leaves out -----------------------------------------
    shares: dict[str, list[float]] = {}
    for width in (64, 128):
        shares[f"width {width}"] = [
            transformer_flops(width=width, depth=exp["depth"], block_size=block, vocab_size=vocab_size)[
                "attention_share"
            ]
            for block in exp["flop_block_sizes"]
        ]
    at_default = transformer_flops(
        width=exp["data_width"], depth=exp["depth"], block_size=exp["block_size"], vocab_size=vocab_size
    )
    summary["flops/attention_share"] = at_default["attention_share"]
    summary["flops/compute_over_rule"] = at_default["forward"] / at_default["rule_of_thumb"]
    print(
        f"[g1] at width {exp['data_width']}, block {exp['block_size']}: attention is "
        f"{at_default['attention_share'] * 100:.1f}% of forward FLOPs, so 6ND undercounts by "
        f"{(at_default['forward'] / at_default['rule_of_thumb'] - 1) * 100:.1f}%"
    )

    # -- figures --------------------------------------------------------------
    grid = np.geomspace(n_values.min() * 0.7, n_values.max() * 1.4, 64)
    exp.log_figure(
        "loss_vs_parameters",
        plotting.plot_fit(
            n_values,
            n_losses,
            curve=(grid, power_law_curve(n_fit, grid)),
            title=f"L(N) = {n_fit['irreducible']:.2f} + A/N^{n_fit['exponent']:.2f}",
            xlabel="non-embedding parameters",
            ylabel="held-out loss",
            logx=True,
            logy=True,
            curve_label=f"power law (R^2 {n_fit['r2']:.3f})",
            annotate=[f"w{w}" for w in exp["widths"]],
        ),
    )
    data_grid = np.geomspace(d_values.min() * 0.7, d_values.max() * 1.4, 64)
    exp.log_figure(
        "loss_vs_tokens",
        plotting.plot_fit(
            d_values,
            d_losses,
            curve=(data_grid, power_law_curve(d_fit, data_grid)),
            title=f"L(D) = {d_fit['irreducible']:.2f} + B/D^{d_fit['exponent']:.2f}",
            xlabel="training tokens",
            ylabel="held-out loss",
            logx=True,
            logy=True,
            curve_label=f"power law (R^2 {d_fit['r2']:.3f})",
            annotate=[f"{r['epochs_of_data']:.1f}ep" for r in data_sweep],
        ),
    )
    exp.log_figure(
        "isoflop",
        plotting.plot_curves(
            contours,
            x=contour_x[next(iter(contour_x))],
            title="IsoFLOP contours: same compute, different model size",
            xlabel="non-embedding parameters",
            ylabel="held-out loss",
            logx=True,
            marker="o",
        ),
    )
    exp.log_figure(
        "attention_share",
        plotting.plot_curves(
            shares,
            x=exp["flop_block_sizes"],
            title="Share of forward FLOPs the 6ND rule drops",
            xlabel="context length",
            ylabel="attention fraction",
            logx=True,
            marker="o",
        ),
    )
    exp.log_figure(
        "flop_accounting",
        plotting.plot_architecture(
            {
                "One pre-LN block": [
                    ("input (B, L, d)", "residual stream"),
                    ("LayerNorm"),
                    ("QKV projection", "6Ld^2 -- counted by 6ND"),
                    ("QK^T then AV", "4L^2 d -- NOT counted"),
                    ("output projection", "2Ld^2 -- counted"),
                    ("LayerNorm + MLP 4x", "16Ld^2 -- counted"),
                ],
            },
            highlight={"One pre-LN block": [3]},
            title="Where 6ND comes from, and the one term it omits",
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
