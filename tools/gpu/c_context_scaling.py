"""Part C at a scale where context can matter: the out-of-band GPU study.

C1 and C3 compare recurrence and attention at 64 characters of context, 250k
characters of text and 500 steps, and there the wide vanilla RNN wins. C3 says
why that is not the end of the story -- no warmup for the transformer, and a
regime with neither the context nor the data attention is built for -- and
lists "warmup" and "longer context, where the recurrent models must lose" as
open threads. This script runs those threads.

Three studies, each over seeds 0-2:

``shakespeare``
    C1/C3's exact regime (same corpus slice, context, batch, steps, peak
    learning rate, the hand-written C1 cells and the C3 transformer), with and
    without linear warmup + cosine decay. Answers: was C3's loss to the RNN the
    missing warmup?

``enwik8``
    The standard byte-level benchmark (first 90M bytes train, next 5M
    validation). Five models at ~4M parameters (vanilla RNN, GRU, LSTM,
    transformer) plus a ~25M transformer, each trained at 64, 256 and 1024
    bytes of context on the same number of *tokens* per step. The recurrences
    use cuDNN (``nn.RNN/GRU/LSTM`` -- the same equations as C1's cells, which
    would take hours as a Python loop at 1024 steps). Reports bits/byte and
    the loss at every position in the window, which is the direct measure of
    how much a model gains from each extra byte of context.

``copy``
    A sequence of n random symbols followed by the same n symbols again.
    The second half is predictable only by looking exactly n positions back,
    so accuracy there measures usable memory and nothing else.

Writes one JSON file (``--out``); ``--plot`` turns it into the figures.
"""

from __future__ import annotations

import argparse
import io
import itertools
import json
import math
import sys
import time
import types
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "src"), str(ROOT)]

ENWIK8_URL = "http://mattmahoney.net/dc/enwik8.zip"

# ---------------------------------------------------------------------------
# models


def recurrent_lm(kind: str, hidden: int, layers: int, vocab: int = 256):
    """Embedding -> cuDNN recurrence -> linear head."""
    import torch
    from torch import nn

    class RecurrentLM(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = nn.Embedding(vocab, hidden)
            cls = {"rnn": nn.RNN, "gru": nn.GRU, "lstm": nn.LSTM}[kind]
            self.rnn = cls(hidden, hidden, num_layers=layers, batch_first=True)
            self.head = nn.Linear(hidden, vocab)
            if kind == "lstm":  # C1's forget-gate bias of 1: start by remembering
                with torch.no_grad():
                    for name, parameter in self.rnn.named_parameters():
                        if name.startswith("bias_ih"):
                            parameter[hidden : 2 * hidden].fill_(1.0)
                        elif name.startswith("bias_hh"):
                            parameter[hidden : 2 * hidden].fill_(0.0)

        def forward(self, ids):
            out, _ = self.rnn(self.embedding(ids))
            return self.head(out), None

    return RecurrentLM()


def transformer_lm(width: int, depth: int, heads: int, block_size: int, vocab: int = 256, dropout: float = 0.0):
    """C3's CharTransformer, learned positions, with attention through SDPA.

    Same computation as C3's explicit softmax(QK^T/sqrt(d) + M)V -- checked
    numerically by ``attention_equivalence`` -- but through PyTorch's fused
    kernel, which does not materialise the T x T score matrix and is what makes
    1024-byte windows fit on a 16 GB T4.
    """
    from parts.c_sequence_attention.c3_transformer_from_scratch import make_transformer

    CharTransformer, _ = make_transformer()
    model = CharTransformer(
        vocab, width=width, heads=heads, depth=depth, block_size=block_size, positional="learned", dropout=dropout
    )
    for module in model.modules():
        if type(module).__name__ == "MultiHeadSelfAttention":
            module.explicit_forward = module.forward
            module.forward = types.MethodType(_sdpa_forward, module)
    return model


def _sdpa_forward(self, x, *, causal: bool = True, return_weights: bool = False):
    import torch.nn.functional as F

    batch, length, width = x.shape
    q, k, v = self.qkv(x).chunk(3, dim=-1)
    q, k, v = (t.view(batch, length, self.heads, self.head_width).transpose(1, 2) for t in (q, k, v))
    out = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
    return self.projection(out.transpose(1, 2).reshape(batch, length, width)), None


def attention_equivalence(device) -> float:
    """Largest |explicit - SDPA| over one random batch, in fp32."""
    import torch

    model = transformer_lm(64, 1, 4, 32).to(device).eval()
    attention = next(m for m in model.modules() if type(m).__name__ == "MultiHeadSelfAttention")
    x = torch.randn(3, 32, 64, device=device)
    with torch.no_grad():
        fused, _ = attention(x)
        explicit, _ = attention.explicit_forward(x)
    return float((fused - explicit).abs().max())


def c1_lm(kind: str, hidden: int, vocab: int):
    """C1's hand-written cells, for the regime C1 itself ran in."""
    from parts.c_sequence_attention.c1_char_rnn import make_cells

    _, _, CharLanguageModel = make_cells()
    return CharLanguageModel(vocab, hidden, cell=kind, layers=1)


# ---------------------------------------------------------------------------
# training


def lr_at(step: int, total: int, peak: float, schedule: str, warmup_fraction: float = 0.05) -> float:
    """Constant (C1/C3's setting) or linear warmup then cosine to 10% of peak."""
    if schedule == "constant":
        return peak
    warmup = max(1, int(total * warmup_fraction))
    if step < warmup:
        return peak * (step + 1) / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return peak * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * progress)))


def train(
    model, sample_batch, *, steps, peak_lr, schedule, device, weight_decay=0.1, amp=True, log_every=250, label=""
):
    import torch

    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=(0.9, 0.95), weight_decay=weight_decay)
    use_amp = amp and device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
    started = time.time()
    history = []
    for step in range(steps):
        for group in optimizer.param_groups:
            group["lr"] = lr_at(step, steps, peak_lr, schedule)
        inputs, targets = sample_batch()
        model.train()
        with torch.autocast(device.type, dtype=torch.float16, enabled=use_amp):
            logits, _ = model(inputs)
            loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
        optimizer.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        if step % log_every == 0 or step == steps - 1:
            history.append((step, loss.item()))
            print(f"  [{label}] step {step:5d} loss {loss.item():.4f} ({time.time() - started:.0f}s)", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize()
    seconds = time.time() - started
    peak = torch.cuda.max_memory_allocated() / 2**20 if device.type == "cuda" else 0.0
    return {"train_seconds": seconds, "peak_memory_mb": peak, "train_history": history}


def per_position_loss(model, windows, *, device, batch_size) -> np.ndarray:
    """Mean cross-entropy (nats) at every position of fixed validation windows."""
    import torch

    model.eval()
    total = None
    count = 0
    with torch.no_grad():
        for start in range(0, len(windows), batch_size):
            chunk = torch.from_numpy(windows[start : start + batch_size]).to(device)
            inputs, targets = chunk[:, :-1], chunk[:, 1:]
            with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == "cuda"):
                logits, _ = model(inputs)
            losses = torch.nn.functional.cross_entropy(
                logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="none"
            ).view(targets.shape)
            summed = losses.sum(dim=0).double().cpu().numpy()
            total = summed if total is None else total + summed
            count += len(chunk)
    return total / count


def count(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# studies


def study_shakespeare(args, device, report):
    """C1/C3's regime, with and without warmup + cosine."""
    import torch

    from mlladder import text

    corpus = text.tiny_shakespeare(limit=250000)
    tokenizer = text.CharTokenizer.from_text(corpus)
    train_ids, val_ids = text.split_corpus(tokenizer.encode(corpus))
    block, batch, steps = 64, 48, args.shakespeare_steps
    eval_rng = np.random.default_rng(12345)  # C1/C3's evaluation windows
    windows = []
    for _ in range(8):
        inputs, targets = text.lm_batch(val_ids, batch_size=batch, block_size=block, rng=eval_rng)
        windows.append(np.concatenate([inputs, targets[:, -1:]], axis=1))
    windows = np.concatenate(windows)

    arms = {
        "rnn-wide": (lambda: c1_lm("rnn", 384, tokenizer.vocab_size), 0.0),
        "lstm": (lambda: c1_lm("lstm", 192, tokenizer.vocab_size), 0.0),
        "transformer": (
            lambda: transformer_lm(192, 4, 4, block, vocab=tokenizer.vocab_size),
            0.1,
        ),
    }
    rows = report.setdefault("shakespeare", [])
    for seed in args.seeds:
        for schedule in ("constant", "warmup-cosine"):
            for name, (build, weight_decay) in arms.items():
                torch.manual_seed(seed)
                model = build()
                rng = np.random.default_rng(seed)

                def sample_batch(rng=rng):
                    inputs, targets = text.lm_batch(train_ids, batch_size=batch, block_size=block, rng=rng)
                    return torch.from_numpy(inputs).to(device), torch.from_numpy(targets).to(device)

                # C1 trains its cells without weight decay; C3 with 0.1. Kept.
                info = train(
                    model,
                    sample_batch,
                    steps=steps,
                    peak_lr=3e-3,
                    schedule=schedule,
                    device=device,
                    weight_decay=weight_decay,
                    amp=False,
                    log_every=max(steps // 2, 1),
                    label=f"shk {name} {schedule} s{seed}",
                )
                curve = per_position_loss(model, windows, device=device, batch_size=96)
                rows.append(
                    {
                        "seed": seed,
                        "model": name,
                        "schedule": schedule,
                        "parameters": count(model),
                        "bits_per_char": float(curve.mean() / math.log(2)),
                        "train_seconds": info["train_seconds"],
                    }
                )
                print(
                    f"[shakespeare] s{seed} {name:<11} {schedule:<13} {rows[-1]['bits_per_char']:.4f} bits/char",
                    flush=True,
                )
                save(args, report)


def load_enwik8(args) -> np.ndarray:
    path = Path(args.data_dir) / "enwik8"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        print("[enwik8] downloading", flush=True)
        with urllib.request.urlopen(ENWIK8_URL, timeout=300) as response:
            archive = zipfile.ZipFile(io.BytesIO(response.read()))
        path.write_bytes(archive.read("enwik8"))
    return np.frombuffer(path.read_bytes(), dtype=np.uint8).astype(np.int64)


ENWIK8_MODELS = {
    # name: (builder(block), peak_lr, weight_decay). ~4M parameters each except
    # transformer-25M; widths chosen so the counts match (see "parameters").
    "rnn": (lambda block: recurrent_lm("rnn", 1024, 2), 1e-3, 0.0),
    "gru": (lambda block: recurrent_lm("gru", 592, 2), 2e-3, 0.0),
    "lstm": (lambda block: recurrent_lm("lstm", 512, 2), 2e-3, 0.0),
    "transformer-4M": (lambda block: transformer_lm(256, 5, 4, block), 1e-3, 0.1),
    "transformer-25M": (lambda block: transformer_lm(512, 8, 8, block), 6e-4, 0.1),
}


def study_enwik8(args, device, report):
    import torch

    data = load_enwik8(args)
    train_ids, val_ids = data[:90_000_000], data[90_000_000:95_000_000]
    rows = report.setdefault("enwik8", [])
    done = {(r["seed"], r["model"], r["context"]) for r in rows}
    for seed in args.seeds:
        for context in args.contexts:
            batch = args.tokens_per_step // context
            # Non-overlapping validation windows, the same bytes for every model
            # at a given context: 256 KiB of held-out text.
            n_windows = args.eval_tokens // context
            starts = np.arange(n_windows) * (context + 1)
            windows = np.stack([val_ids[s : s + context + 1] for s in starts])
            for name in args.models:
                if (seed, name, context) in done:
                    continue
                build, peak_lr, weight_decay = ENWIK8_MODELS[name]
                torch.manual_seed(seed)
                model = build(context)
                rng = np.random.default_rng(seed)

                def sample_batch(rng=rng, context=context, batch=batch):
                    starts = rng.integers(0, len(train_ids) - context - 1, size=batch)
                    chunk = np.stack([train_ids[s : s + context + 1] for s in starts])
                    tensor = torch.from_numpy(chunk).to(device)
                    return tensor[:, :-1], tensor[:, 1:]

                info = train(
                    model,
                    sample_batch,
                    steps=args.enwik8_steps,
                    peak_lr=peak_lr,
                    schedule="warmup-cosine",
                    device=device,
                    weight_decay=weight_decay,
                    label=f"enwik8 {name} T={context} s{seed}",
                )
                curve = per_position_loss(model, windows, device=device, batch_size=max(1, 65536 // context))
                bits = curve / math.log(2)
                rows.append(
                    {
                        "seed": seed,
                        "model": name,
                        "context": context,
                        "batch": batch,
                        "parameters": count(model),
                        "steps": args.enwik8_steps,
                        "tokens_seen": args.enwik8_steps * batch * context,
                        "bits_per_byte": float(bits.mean()),
                        "bits_last_quarter": float(bits[-max(1, context // 4) :].mean()),
                        "bits_by_position": [round(float(b), 4) for b in bits],
                        **{k: v for k, v in info.items()},
                        "tokens_per_second": args.enwik8_steps * batch * context / info["train_seconds"],
                    }
                )
                print(
                    f"[enwik8] s{seed} T={context:<5} {name:<16} {rows[-1]['bits_per_byte']:.4f} bits/byte "
                    f"(last quarter {rows[-1]['bits_last_quarter']:.4f}) params={count(model):,} "
                    f"{info['train_seconds']:.0f}s {info['peak_memory_mb']:.0f}MB",
                    flush=True,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                save(args, report)


COPY_VOCAB = 32


def study_copy(args, device, report):
    import torch

    rows = report.setdefault("copy", [])
    builders = {
        "lstm": (lambda block: recurrent_lm("lstm", 512, 2, vocab=COPY_VOCAB), 2e-3, 0.0),
        "transformer-4M": (lambda block: transformer_lm(256, 5, 4, block, vocab=COPY_VOCAB), 1e-3, 0.1),
    }
    for seed in args.seeds:
        for n in args.copy_lengths:
            length = 2 * n
            batch = max(8, args.tokens_per_step // length)
            for name, (build, peak_lr, weight_decay) in builders.items():
                torch.manual_seed(seed)
                model = build(length - 1)
                rng = np.random.default_rng(seed)

                def sample_batch(size=batch, rng=rng, n=n):
                    half = rng.integers(0, COPY_VOCAB, size=(size, n))
                    tensor = torch.from_numpy(np.concatenate([half, half], axis=1)).to(device)
                    return tensor[:, :-1], tensor[:, 1:]

                info = train(
                    model,
                    sample_batch,
                    steps=args.copy_steps,
                    peak_lr=peak_lr,
                    schedule="warmup-cosine",
                    device=device,
                    weight_decay=weight_decay,
                    log_every=max(args.copy_steps // 4, 1),
                    label=f"copy {name} n={n} s{seed}",
                )
                model.eval()
                eval_rng_state = rng
                with torch.no_grad():
                    inputs, targets = sample_batch(512)
                    logits, _ = model(inputs)
                    correct = (logits.argmax(-1) == targets).float()
                # Targets n-1 .. 2n-2 are the second half: each is a copy of the
                # symbol exactly n positions earlier.
                second = float(correct[:, n - 1 :].mean())
                first = float(correct[:, : n - 1].mean())
                del eval_rng_state
                rows.append(
                    {
                        "seed": seed,
                        "model": name,
                        "n": n,
                        "parameters": count(model),
                        "second_half_accuracy": second,
                        "first_half_accuracy": first,
                        "chance": 1 / COPY_VOCAB,
                        "train_seconds": info["train_seconds"],
                    }
                )
                print(
                    f"[copy] s{seed} n={n:<4} {name:<15} second-half acc {second:.3f} (first half {first:.3f}, chance {1 / COPY_VOCAB:.3f})",
                    flush=True,
                )
                save(args, report)


def save(args, report) -> None:
    Path(args.out).write_text(json.dumps(report, indent=1))


# ---------------------------------------------------------------------------
# figures


def plot(json_path: Path, figures: Path) -> None:
    import matplotlib.pyplot as plt

    from mlladder import plotting

    plotting.use_house_style()
    report = json.loads(json_path.read_text())
    figures.mkdir(parents=True, exist_ok=True)
    palette = plotting.PALETTE

    rows = report.get("enwik8", [])
    if rows:
        models = [m for m in ENWIK8_MODELS if any(r["model"] == m for r in rows)]
        contexts = sorted({r["context"] for r in rows})

        def median(model, context, key):
            values = [r[key] for r in rows if r["model"] == model and r["context"] == context]
            return (float(np.median(values)), float(min(values)), float(max(values))) if values else None

        fig, ax = plt.subplots(figsize=(7, 4.2))
        for index, model in enumerate(models):
            points = [(c, median(model, c, "bits_per_byte")) for c in contexts]
            points = [(c, m) for c, m in points if m]
            xs = [c for c, _ in points]
            ax.errorbar(
                xs,
                [m[0] for _, m in points],
                yerr=[[m[0] - m[1] for _, m in points], [m[2] - m[0] for _, m in points]],
                marker="o",
                capsize=3,
                label=model,
                color=palette[index % len(palette)],
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(contexts, [str(c) for c in contexts])
        ax.set_xlabel("training context (bytes)")
        ax.set_ylabel("validation bits/byte (lower is better)")
        ax.set_title("enwik8: what each architecture does with more context")
        ax.legend(fontsize=8)
        plotting.save_fig(fig, figures / "c_gpu_enwik8_context.png")

        widest = max(contexts)
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for index, model in enumerate(models):
            curves = [r["bits_by_position"] for r in rows if r["model"] == model and r["context"] == widest]
            if not curves:
                continue
            curve = np.median(np.array(curves), axis=0)
            np.arange(1, len(curve) + 1)
            # Smooth in log-spaced bins: a single position is noisy at 256 KiB.
            edges = np.unique(np.round(np.logspace(0, np.log10(len(curve)), 40)).astype(int))
            centres = [np.sqrt(a * b) for a, b in itertools.pairwise(edges)]
            values = [curve[a - 1 : b - 1].mean() for a, b in itertools.pairwise(edges)]
            ax.plot(centres, values, label=model, color=palette[index % len(palette)])
        ax.set_xscale("log")
        ax.set_xlabel("position in the window = bytes of context available")
        ax.set_ylabel("bits/byte at that position")
        ax.set_title(f"Loss by position, models trained at T={widest}")
        ax.legend(fontsize=8)
        plotting.save_fig(fig, figures / "c_gpu_enwik8_by_position.png")

        fig, ax = plt.subplots(figsize=(7, 4.2))
        for index, model in enumerate(models):
            points = [(c, median(model, c, "tokens_per_second")) for c in contexts]
            points = [(c, m[0]) for c, m in points if m]
            ax.plot(
                [c for c, _ in points],
                [v / 1e3 for _, v in points],
                marker="o",
                label=model,
                color=palette[index % len(palette)],
            )
        ax.set_xscale("log", base=2)
        ax.set_xticks(contexts, [str(c) for c in contexts])
        ax.set_xlabel("training context (bytes)")
        ax.set_ylabel("training throughput (k tokens/s, T4, fp16)")
        ax.set_title("What each architecture costs per token as context grows")
        ax.legend(fontsize=8)
        plotting.save_fig(fig, figures / "c_gpu_enwik8_throughput.png")

    rows = report.get("copy", [])
    if rows:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        lengths = sorted({r["n"] for r in rows})
        for index, model in enumerate(sorted({r["model"] for r in rows})):
            med = [
                np.median([r["second_half_accuracy"] for r in rows if r["model"] == model and r["n"] == n])
                for n in lengths
            ]
            ax.plot(lengths, med, marker="o", label=model, color=palette[index])
        ax.axhline(1 / COPY_VOCAB, color="#888", linestyle=":", label="chance")
        ax.set_xscale("log", base=2)
        ax.set_xticks(lengths, [str(n) for n in lengths])
        ax.set_ylim(0, 1.05)
        ax.set_xlabel("n: symbols to remember before they repeat")
        ax.set_ylabel("accuracy on the repeated half")
        ax.set_title("Copy task: memory that has to reach exactly n back")
        ax.legend(fontsize=8)
        plotting.save_fig(fig, figures / "c_gpu_copy.png")

    rows = report.get("shakespeare", [])
    if rows:
        fig, ax = plt.subplots(figsize=(6.5, 4))
        models = ["rnn-wide", "lstm", "transformer"]
        width = 0.36
        for offset, schedule in enumerate(("constant", "warmup-cosine")):
            values = [
                [r["bits_per_char"] for r in rows if r["model"] == m and r["schedule"] == schedule] for m in models
            ]
            med = [np.median(v) for v in values]
            lo = [m - min(v) for m, v in zip(med, values, strict=False)]
            hi = [max(v) - m for m, v in zip(med, values, strict=False)]
            ax.bar(
                np.arange(len(models)) + (offset - 0.5) * width,
                med,
                width,
                yerr=[lo, hi],
                capsize=3,
                label=schedule,
                color=palette[offset],
            )
        ax.set_xticks(range(len(models)), models)
        ax.set_ylim(2.0, None)
        ax.set_ylabel("validation bits/char")
        ax.set_title("C1/C3's regime: does warmup rescue the transformer?")
        ax.legend(fontsize=8)
        plotting.save_fig(fig, figures / "c_gpu_shakespeare_warmup.png")


# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default="c-gpu.json")
    parser.add_argument("--data-dir", default="data/text")
    parser.add_argument("--studies", nargs="+", default=["shakespeare", "copy", "enwik8"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--contexts", nargs="+", type=int, default=[64, 256, 1024])
    parser.add_argument("--models", nargs="+", default=list(ENWIK8_MODELS))
    parser.add_argument("--tokens-per-step", type=int, default=16384)
    parser.add_argument("--enwik8-steps", type=int, default=3000)
    parser.add_argument("--eval-tokens", type=int, default=262144)
    parser.add_argument("--shakespeare-steps", type=int, default=500)
    parser.add_argument("--copy-lengths", nargs="+", type=int, default=[32, 128, 512])
    parser.add_argument("--copy-steps", type=int, default=1500)
    parser.add_argument("--plot", metavar="JSON", help="only draw figures from an existing report")
    parser.add_argument("--figures", default=str(ROOT / "docs/writeups/figures/c"))
    args = parser.parse_args()

    if args.plot:
        plot(Path(args.plot), Path(args.figures))
        return

    import torch

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    report = json.loads(Path(args.out).read_text()) if Path(args.out).exists() else {}
    report["device"] = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
    report["torch"] = torch.__version__
    report["settings"] = {k: v for k, v in vars(args).items() if k not in ("plot", "figures")}
    report["sdpa_vs_explicit_max_abs_diff"] = attention_equivalence(device)
    print("SDPA vs explicit attention, max |diff|:", report["sdpa_vs_explicit_max_abs_diff"], flush=True)
    for study in args.studies:
        {"shakespeare": study_shakespeare, "enwik8": study_enwik8, "copy": study_copy}[study](args, device, report)
    report["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    save(args, report)


if __name__ == "__main__":
    main()
