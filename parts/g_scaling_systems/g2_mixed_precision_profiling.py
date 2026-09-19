r"""G2. Mixed-precision + gradient-accumulation training, profiled for throughput and memory

Lab reference: [PyTorch mixed-precision & profiler docs](https://pytorch.org/docs/stable/notes/amp_examples.html)

Two techniques that everybody turns on and few people measure. Both trade one
resource for another, and both have a failure mode that does not announce
itself.

**Mixed precision** runs the matmuls in 16 bits and keeps a 32-bit copy of the
weights. The speedup comes from tensor cores, not from the narrower type, so it
appears only when the matrices are large enough to reach them — which makes
"AMP is faster" a claim about your shapes, not about AMP. The failure mode is
underflow: fp16's smallest normal is $6\times10^{-5}$, and gradients live below
it, so without loss scaling they quietly become zero. This rung counts how
often that happens rather than asserting that it does.

$$\text{fp16: } 1\text{ sign} + 5\text{ exp} + 10\text{ mantissa},\qquad
\text{bf16: } 1 + 8 + 7,\qquad \text{fp32: } 1 + 8 + 23.$$

Read the exponent fields and the whole story is there. **bf16 has fp32's
exponent**, so it cannot underflow where fp32 would not and needs no scaler at
all; it pays with three fewer mantissa bits. fp16 keeps precision and loses
range, which is exactly the wrong trade for gradients — hence `GradScaler`
existing for one of the two and not the other. Whether you get the choice is a
property of the silicon, and asking the library is not the same as asking the
silicon. bf16 tensor cores arrived with Ampere (sm_80), but on a Turing T4
``torch.cuda.is_bf16_supported()`` returns **True** — the dtype works, it is
simply emulated. Measured on that T4, fp16 runs at 2.59x fp32 and bf16 at
**0.78x**: slower than the format it was meant to replace. Only the throughput
number distinguishes "supported" from "accelerated", which is why this rung
measures the arm instead of trusting the capability flag.

**Gradient accumulation** buys effective batch size with time instead of
memory: run $k$ micro-batches, sum their gradients, step once. Its failure mode
is arithmetic. The loss of a micro-batch is a mean over its own tokens, so
summing $k$ of them gives $k$ times the gradient of the large batch, and the
run silently trains at $k$ times the learning rate. Dividing by $k$ is the fix
and the rung measures the gradient norms both ways rather than trusting it.

Run with ``ladder run G2`` (``--help`` for flags). Most of what it measures
needs a GPU; on CPU it reports that plainly instead of producing numbers that
look like results.
"""

from __future__ import annotations

import time

import numpy as np

from mlladder import Experiment, experiment, plotting, text, torchkit, vision


def device_report() -> dict[str, object]:
    """What the hardware can actually do, asked rather than assumed.

    ``bf16_supported`` is the load-bearing entry: Ampere and later have bf16
    tensor cores, Turing does not, and the difference decides whether the
    scaler-free path in this rung is a measurement or a skipped arm.
    """
    torch = vision.require_torch()
    if not torch.cuda.is_available():
        return {"name": "cpu", "capability": (0, 0), "cuda": False, "bf16_supported": False, "fp16_tensor_cores": False}
    index = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(index)
    return {
        "name": torch.cuda.get_device_name(index),
        "capability": (major, minor),
        "cuda": True,
        # Worth cross-reading against the capability: torch answers True on
        # sm_75, where bf16 is emulated rather than accelerated.
        "bf16_supported": bool(torch.cuda.is_bf16_supported()),
        "fp16_tensor_cores": major >= 7,
    }


def float_ranges() -> dict[str, dict[str, float]]:
    """The exponent-range facts the rest of the rung turns on."""
    torch = vision.require_torch()
    return {
        name: {
            "smallest_normal": float(torch.finfo(dtype).tiny),
            "max": float(torch.finfo(dtype).max),
            "eps": float(torch.finfo(dtype).eps),
        }
        for name, dtype in (("fp32", torch.float32), ("fp16", torch.float16), ("bf16", torch.bfloat16))
    }


def build_model(exp: Experiment, vocab_size: int, device, *, width: int | None = None, heads: int | None = None):
    torch = vision.require_torch()
    from parts.c_sequence_attention.c3_transformer_from_scratch import make_transformer

    CharTransformer, _ = make_transformer()
    torchkit.seed_torch(exp.seed)
    model = CharTransformer(
        vocab_size,
        width=width if width is not None else exp["width"],
        heads=heads if heads is not None else exp["heads"],
        depth=exp["depth"],
        block_size=exp["block_size"],
        positional="learned",
    ).to(device)
    return model, torch


def train_arm(
    *,
    exp: Experiment,
    ids,
    vocab_size: int,
    device,
    precision: str,
    accumulate: int = 1,
    micro_batch: int | None = None,
    use_scaler: bool = True,
    width: int | None = None,
    heads: int | None = None,
    steps: int | None = None,
) -> dict[str, float]:
    """One training configuration, timed and profiled.

    ``micro_batch`` times ``accumulate`` is held at the arm's effective batch,
    so every configuration sees the same number of tokens per optimizer step and
    the comparison is about *how* they were fed, not how many.

    Timing excludes a warmup and synchronizes the device before reading the
    clock. Without the synchronize, CUDA's asynchrony means the loop measures
    how fast Python can enqueue work, which on a small model is a very fast and
    completely wrong number.
    """
    torch = vision.require_torch()
    model, _ = build_model(exp, vocab_size, device, width=width, heads=heads)
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"])
    rng = np.random.default_rng(exp.seed)
    measured_steps = steps if steps is not None else exp["steps"]

    effective = exp["batch_size"]
    micro = micro_batch if micro_batch is not None else effective // accumulate
    assert micro * accumulate == effective, "micro-batch times accumulation must equal the effective batch"

    autocast_dtype = {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]
    on_cuda = device.type == "cuda" if hasattr(device, "type") else str(device).startswith("cuda")
    wanted = bool(on_cuda and precision == "fp16" and use_scaler)
    # torch.amp.GradScaler is the 2.3+ spelling; the older one still ships on
    # some managed runtimes (Kaggle's image among them), so both are accepted
    # rather than pinning a torch version this rung does not otherwise need.
    try:
        scaler = torch.amp.GradScaler("cuda", enabled=wanted)
    except (AttributeError, TypeError):  # pragma: no cover - runtime-dependent
        scaler = torch.cuda.amp.GradScaler(enabled=wanted)

    if on_cuda:
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()

    losses: list[float] = []
    skipped = 0
    nonfinite_micro = 0
    total_micro = 0
    started = None
    for step in range(measured_steps + exp["warmup_steps"]):
        if step == exp["warmup_steps"]:
            if on_cuda:
                torch.cuda.synchronize()
            started = time.time()
        optimizer.zero_grad(set_to_none=True)
        step_loss = 0.0
        for _ in range(accumulate):
            inputs, targets = text.lm_batch(ids, batch_size=micro, block_size=exp["block_size"], rng=rng)
            x = torch.from_numpy(inputs).to(device)
            y = torch.from_numpy(targets).to(device)
            with torch.autocast(
                device_type="cuda" if on_cuda else "cpu",
                dtype=autocast_dtype,
                enabled=autocast_dtype is not None and on_cuda,
            ):
                logits, _ = model(x)
                # Divided by the accumulation count so the summed gradient is
                # the mean over the effective batch and not k times it.
                loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1)) / accumulate
            scaler.scale(loss).backward()
            step_loss += float(loss.item()) * accumulate
            total_micro += 1

        if scaler.is_enabled():
            before = scaler.get_scale()
            scaler.step(optimizer)
            scaler.update()
            # A scale that shrank means the step was skipped for non-finite
            # gradients: the underflow guard doing its job, counted.
            skipped += int(scaler.get_scale() < before)
        else:
            finite = all(bool(torch.isfinite(p.grad).all()) for p in model.parameters() if p.grad is not None)
            nonfinite_micro += int(not finite)
            optimizer.step()
        losses.append(step_loss / accumulate)

    if on_cuda:
        torch.cuda.synchronize()
    elapsed = time.time() - (started or time.time())
    tokens = measured_steps * effective * exp["block_size"]
    peak = float(torch.cuda.max_memory_allocated()) / 2**20 if on_cuda else float("nan")
    return {
        "tokens_per_second": tokens / max(elapsed, 1e-9),
        "seconds": elapsed,
        "peak_memory_mb": peak,
        "final_loss": float(np.mean(losses[-10:])),
        "skipped_steps": float(skipped),
        "nonfinite_steps": float(nonfinite_micro),
        "micro_batches": float(total_micro),
        "micro_batch_size": float(micro),
    }


def gradient_norm(*, exp: Experiment, ids, vocab_size: int, device, accumulate: int, average: bool) -> float:
    """Gradient norm after one optimizer step's worth of micro-batches.

    The whole point: with ``average=False`` the gradients are summed, so the
    norm comes out about ``accumulate`` times larger than the equivalent large
    batch — the model then trains at ``accumulate`` times the learning rate it
    was configured with, which looks like a tuning problem rather than a bug.
    """
    torch = vision.require_torch()
    model, _ = build_model(exp, vocab_size, device)
    rng = np.random.default_rng(exp.seed)
    micro = exp["batch_size"] // accumulate
    model.zero_grad(set_to_none=True)
    for _ in range(accumulate):
        inputs, targets = text.lm_batch(ids, batch_size=micro, block_size=exp["block_size"], rng=rng)
        x = torch.from_numpy(inputs).to(device)
        y = torch.from_numpy(targets).to(device)
        logits, _ = model(x)
        loss = torch.nn.functional.cross_entropy(logits.reshape(-1, vocab_size), y.reshape(-1))
        (loss / accumulate if average else loss).backward()
    total = sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None)
    return float(np.sqrt(total))


@experiment(
    "g2_mixed_precision_profiling",
    defaults={
        "width": 384,
        "heads": 6,
        "depth": 4,
        "block_size": 128,
        "batch_size": 32,
        "lr": 3e-4,
        "steps": 40,
        "warmup_steps": 10,
        "precisions": ["fp32", "fp16", "bf16"],
        "accumulation_steps": [1, 2, 4, 8],
        # AMP's speedup is a claim about matrix shapes, so the widths are swept.
        "amp_widths": [64, 128, 256, 512],
        "amp_steps": 20,
        "amp_heads": 4,
    },
    description="Mixed precision and gradient accumulation, profiled for throughput, memory and correctness.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Mixed-precision + gradient-accumulation training, profiled for throughput and memory"""
    vision.require_torch()
    device = torchkit.device()
    hardware = device_report()
    print(f"[g2] device: {hardware['name']} (capability {hardware['capability']}), bf16 {hardware['bf16_supported']}")

    summary: dict[str, float] = {
        "hardware/cuda": float(bool(hardware["cuda"])),
        "hardware/capability_major": float(hardware["capability"][0]),
        "hardware/capability_minor": float(hardware["capability"][1]),
        "hardware/bf16_supported": float(bool(hardware["bf16_supported"])),
        "hardware/fp16_tensor_cores": float(bool(hardware["fp16_tensor_cores"])),
    }
    ranges = float_ranges()
    for name, values in ranges.items():
        summary[f"format/{name}/smallest_normal"] = values["smallest_normal"]
        summary[f"format/{name}/eps"] = values["eps"]
    # bf16 buys range with mantissa bits: its smallest normal matches fp32's,
    # and its epsilon is correspondingly coarser. Both directions recorded,
    # because the trade is the entire reason both formats exist.
    summary["format/bf16_range_over_fp16"] = ranges["bf16"]["smallest_normal"] / ranges["fp16"]["smallest_normal"]
    summary["format/bf16_eps_over_fp16"] = ranges["bf16"]["eps"] / ranges["fp16"]["eps"]
    print(
        f"[g2] smallest normal: fp32 {ranges['fp32']['smallest_normal']:.2e}, "
        f"fp16 {ranges['fp16']['smallest_normal']:.2e}, bf16 {ranges['bf16']['smallest_normal']:.2e} "
        f"-- bf16 matches fp32's range exactly and pays "
        f"{summary['format/bf16_eps_over_fp16']:.0f}x coarser resolution for it"
    )

    corpus = text.tiny_shakespeare()
    tokenizer = text.CharTokenizer.from_text(corpus)
    ids = tokenizer.encode(corpus)
    vocab_size = tokenizer.vocab_size

    if not hardware["cuda"]:
        print(
            "[g2] no CUDA device. Autocast on CPU does not reach tensor cores and GradScaler is a no-op, "
            "so the throughput and memory arms would measure nothing; they are skipped rather than faked."
        )
        summary["skipped/needs_gpu"] = 1.0
        summary["accumulation/mean_vs_sum"] = gradient_norm(
            exp=exp, ids=ids, vocab_size=vocab_size, device=device, accumulate=4, average=False
        ) / max(
            gradient_norm(exp=exp, ids=ids, vocab_size=vocab_size, device=device, accumulate=4, average=True), 1e-12
        )
        print(
            f"[g2] the one arm that needs no GPU: summing 4 micro-batch gradients instead of averaging them "
            f"gives a gradient {summary['accumulation/mean_vs_sum']:.2f}x larger"
        )
        exp.log_figure(
            "formats",
            plotting.plot_comparison(
                {name: values["smallest_normal"] for name, values in ranges.items()},
                title="Smallest normal value (log scale is the point)",
                ylabel="smallest normal",
            ),
        )
        return summary
    summary["skipped/needs_gpu"] = 0.0

    # -- precision arms -------------------------------------------------------
    throughput: dict[str, float] = {}
    memory: dict[str, float] = {}
    for precision in exp["precisions"]:
        if precision == "bf16" and not hardware["bf16_supported"]:
            print(f"[g2] bf16 unsupported on {hardware['name']} (needs sm_80); arm skipped and recorded as such")
            summary["precision/bf16/skipped"] = 1.0
            continue
        record = train_arm(exp=exp, ids=ids, vocab_size=vocab_size, device=device, precision=precision)
        throughput[precision] = record["tokens_per_second"]
        memory[precision] = record["peak_memory_mb"]
        for key, value in record.items():
            summary[f"precision/{precision}/{key}"] = value
        print(
            f"[g2] {precision:<5} {record['tokens_per_second'] / 1e3:8.1f}k tok/s  "
            f"peak {record['peak_memory_mb']:7.1f} MB  loss {record['final_loss']:.4f}  "
            f"scaler skipped {int(record['skipped_steps'])} steps"
        )

    if "fp32" in throughput:
        for precision in ("fp16", "bf16"):
            if precision in throughput:
                summary[f"precision/{precision}/speedup"] = throughput[precision] / throughput["fp32"]
                summary[f"precision/{precision}/memory_ratio"] = memory[precision] / max(memory["fp32"], 1e-9)
                # "Supported" and "accelerated" are different questions, and
                # only this ratio answers the second one.
                accelerated = summary[f"precision/{precision}/speedup"] > 1.0
                summary[f"precision/{precision}/accelerated"] = float(accelerated)
                print(
                    f"[g2] {precision} is {summary[f'precision/{precision}/speedup']:.2f}x fp32's throughput "
                    f"at {summary[f'precision/{precision}/memory_ratio']:.2f}x its peak memory"
                    + ("" if accelerated else "  <- supported by the library, not accelerated by the hardware")
                )

    # -- does fp16 actually need the scaler here? -----------------------------
    unscaled = train_arm(exp=exp, ids=ids, vocab_size=vocab_size, device=device, precision="fp16", use_scaler=False)
    summary["scaler/without/nonfinite_steps"] = unscaled["nonfinite_steps"]
    summary["scaler/without/final_loss"] = unscaled["final_loss"]
    summary["scaler/with/final_loss"] = summary.get("precision/fp16/final_loss", float("nan"))
    summary["scaler/loss_penalty"] = unscaled["final_loss"] - summary["scaler/with/final_loss"]
    print(
        f"[g2] fp16 without a scaler: {int(unscaled['nonfinite_steps'])} non-finite steps, "
        f"final loss {unscaled['final_loss']:.4f} against {summary['scaler/with/final_loss']:.4f} with one"
    )

    # -- AMP's speedup is a claim about shapes --------------------------------
    speedups: dict[str, list[float]] = {"fp16 speedup": []}
    for width in exp["amp_widths"]:
        # Four heads, not the rung's default six: the swept widths have to be
        # divisible by the head count, and 64 is not divisible by 6.
        shared = {
            "exp": exp,
            "ids": ids,
            "vocab_size": vocab_size,
            "device": device,
            "width": width,
            "heads": exp["amp_heads"],
        }
        reference = train_arm(**shared, precision="fp32", steps=exp["amp_steps"])
        mixed = train_arm(**shared, precision="fp16", steps=exp["amp_steps"])
        ratio = mixed["tokens_per_second"] / max(reference["tokens_per_second"], 1e-9)
        speedups["fp16 speedup"].append(ratio)
        summary[f"amp_width/{width}/speedup"] = ratio
        summary[f"amp_width/{width}/fp32_tokens_per_second"] = reference["tokens_per_second"]
        print(f"[g2] width {width:>4}: fp16 is {ratio:.2f}x fp32")
    summary["amp_width/smallest"] = speedups["fp16 speedup"][0]
    summary["amp_width/largest"] = speedups["fp16 speedup"][-1]

    # -- gradient accumulation ------------------------------------------------
    accumulation_throughput: dict[str, float] = {}
    accumulation_memory: dict[str, float] = {}
    for steps in exp["accumulation_steps"]:
        record = train_arm(exp=exp, ids=ids, vocab_size=vocab_size, device=device, precision="fp16", accumulate=steps)
        label = f"{steps}x"
        accumulation_throughput[label] = record["tokens_per_second"]
        accumulation_memory[label] = record["peak_memory_mb"]
        for key, value in record.items():
            summary[f"accumulation/{steps}/{key}"] = value
        print(
            f"[g2] accumulate {steps}x (micro-batch {int(record['micro_batch_size'])}): "
            f"{record['tokens_per_second'] / 1e3:8.1f}k tok/s  peak {record['peak_memory_mb']:7.1f} MB  "
            f"loss {record['final_loss']:.4f}"
        )
    first, last = exp["accumulation_steps"][0], exp["accumulation_steps"][-1]
    summary["accumulation/memory_saving"] = summary[f"accumulation/{first}/peak_memory_mb"] / max(
        summary[f"accumulation/{last}/peak_memory_mb"], 1e-9
    )
    summary["accumulation/throughput_cost"] = summary[f"accumulation/{first}/tokens_per_second"] / max(
        summary[f"accumulation/{last}/tokens_per_second"], 1e-9
    )
    summary["accumulation/loss_gap"] = abs(
        summary[f"accumulation/{last}/final_loss"] - summary[f"accumulation/{first}/final_loss"]
    )
    print(
        f"[g2] accumulating {last}x instead of {first}x: {summary['accumulation/memory_saving']:.2f}x less peak "
        f"memory for {summary['accumulation/throughput_cost']:.2f}x the time, loss gap "
        f"{summary['accumulation/loss_gap']:.4f}"
    )

    summed = gradient_norm(exp=exp, ids=ids, vocab_size=vocab_size, device=device, accumulate=4, average=False)
    averaged = gradient_norm(exp=exp, ids=ids, vocab_size=vocab_size, device=device, accumulate=4, average=True)
    summary["accumulation/mean_vs_sum"] = summed / max(averaged, 1e-12)
    print(
        f"[g2] summing 4 micro-batch gradients instead of averaging: gradient norm "
        f"{summary['accumulation/mean_vs_sum']:.2f}x larger -- a 4x learning rate nobody configured"
    )

    # -- figures --------------------------------------------------------------
    exp.log_figure(
        "throughput",
        plotting.plot_comparison(
            {k: v / 1e3 for k, v in throughput.items()}, title="Training throughput", ylabel="thousand tokens/s"
        ),
    )
    exp.log_figure(
        "memory",
        plotting.plot_comparison(memory, title="Peak allocated memory", ylabel="MB"),
    )
    exp.log_figure(
        "amp_by_width",
        plotting.plot_curves(
            speedups,
            x=exp["amp_widths"],
            title="fp16 speedup against model width -- AMP is a claim about your shapes",
            xlabel="model width",
            ylabel="fp16 / fp32 throughput",
            logx=True,
            marker="o",
        ),
    )
    exp.log_figure(
        "accumulation",
        plotting.plot_comparison(accumulation_memory, title="Peak memory against accumulation steps", ylabel="MB"),
    )
    exp.log_figure(
        "accumulation_throughput",
        plotting.plot_comparison(
            {k: v / 1e3 for k, v in accumulation_throughput.items()},
            title="Throughput against accumulation steps (same effective batch)",
            ylabel="thousand tokens/s",
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
