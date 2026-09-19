r"""G3. Rust/Candle port of one transformer-block forward pass

Lab reference: [PyTorch mixed-precision & profiler docs](https://pytorch.org/docs/stable/notes/amp_examples.html)

A forward pass you can only describe is a forward pass you do not know. This
rung takes the pre-LN block from rung C3 and rewrites it against Candle, in
Rust, then checks the two against each other numerically — because "it looks
right" is not a result and `max |a - b|` is.

The port lives in ``rust/crates/ignite-port``; this module is the driver that
exports weights, runs both sides and diffs them. Nothing here is about speed.
Rust is not faster than PyTorch at this size and the rung measures that too,
rather than implying otherwise.

What the exercise is actually for is the set of decisions PyTorch makes on your
behalf and Candle does not:

- **`nn.GELU()` has two definitions** and the libraries default to different
  ones. PyTorch computes the exact $x\Phi(x)$; Candle's `gelu()` is the $\tanh$
  approximation and `gelu_erf()` is the exact form. Choosing wrong does not
  crash — it produces a quiet disagreement that reads like a precision problem,
  and this rung measures how big it is.
- **`matmul` needs contiguous tensors.** `transpose` in Candle permutes strides
  and returns a view; PyTorch hides the copy, Candle makes you write it. The
  batched attention path is where forgetting it bites, and where it will not
  show up at batch 1.
- **Weight orientation.** Both libraries store `Linear` as `[out, in]` and
  transpose on use, so a weight loaded the wrong way round is a shape error when
  the dimensions differ and silently wrong numbers when they do not.

Run with ``ladder run G3`` (``--help`` for flags). Needs a Rust toolchain; the
rung builds the crate itself and reports honestly if it cannot.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from mlladder import Experiment, experiment, plotting, torchkit, vision

ROOT = Path(__file__).resolve().parents[2]
CRATE = ROOT / "rust"
BINARY = CRATE / "target" / "release" / "block-forward"

#: The weight names both sides agree on — exactly ``Block.state_dict()`` from C3.
EXPECTED_WEIGHTS = (
    "norm1.weight",
    "norm1.bias",
    "attention.qkv.weight",
    "attention.qkv.bias",
    "attention.projection.weight",
    "attention.projection.bias",
    "norm2.weight",
    "norm2.bias",
    "mlp.0.weight",
    "mlp.0.bias",
    "mlp.2.weight",
    "mlp.2.bias",
)


def cargo_available() -> bool:
    return shutil.which("cargo") is not None


def build_port(*, quiet: bool = True) -> tuple[bool, str]:
    """Build the Rust crate in release mode. Returns ``(ok, message)``.

    Release rather than debug is not about speed here: Candle's debug build is
    slow enough that a forward pass over a handful of tokens takes long enough
    to look broken.
    """
    if not cargo_available():
        return False, "cargo is not on PATH"
    started = time.time()
    result = subprocess.run(
        ["cargo", "build", "--release", "--quiet" if quiet else "--verbose"],
        cwd=CRATE,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return False, result.stderr.strip()[-2000:]
    return True, f"built in {time.time() - started:.1f}s"


def export(block, x, directory: Path) -> tuple[Path, Path]:
    """Write the block's weights and one input to ``.npz``/``.npy``.

    numpy's formats rather than safetensors, because Candle reads both and this
    way the rung adds no Python dependency to check a Rust port — the exchange
    format should not be the interesting part.
    """
    directory.mkdir(parents=True, exist_ok=True)
    arrays = {name: tensor.detach().cpu().numpy().astype(np.float32) for name, tensor in block.state_dict().items()}
    missing = [name for name in EXPECTED_WEIGHTS if name not in arrays]
    if missing:
        raise RuntimeError(f"the Python block no longer exports {missing}; the Rust side reads these by name")
    weights_path = directory / "weights.npz"
    input_path = directory / "input.npy"
    np.savez(weights_path, **arrays)
    np.save(input_path, x.detach().cpu().numpy().astype(np.float32))
    return weights_path, input_path


def run_port(weights: Path, inputs: Path, output: Path, *, heads: int, gelu: str) -> tuple[bool, str]:
    result = subprocess.run(
        [str(BINARY), str(weights), str(inputs), str(output), str(heads), gelu],
        capture_output=True,
        text=True,
    )
    return result.returncode == 0, (result.stderr or result.stdout).strip()[-2000:]


def spin(count: int) -> list[subprocess.Popen]:
    """Start ``count`` processes that do nothing but burn a core.

    Crude on purpose. The point is not to simulate a realistic workload but to
    take the cores away, which is the condition under which a spin-waiting
    thread pool stops being free.
    """
    return [
        subprocess.Popen(
            [sys.executable, "-c", "x = 0.0\nwhile True:\n    x = (x * 1.0000001 + 1.0) % 1e9"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        for _ in range(count)
    ]


def disagreement(reference: np.ndarray, ported: np.ndarray) -> dict[str, float]:
    r"""How far apart two implementations of the same function are.

    Both an absolute and a relative number, because neither alone is readable:
    $10^{-6}$ absolute is excellent on activations of size 1 and meaningless on
    activations of size $10^{6}$. The relative figure is
    $\lVert a - b\rVert / \lVert a \rVert$, which is the quantity that stays
    comparable as the block gets wider.
    """
    difference = reference - ported
    return {
        "max_abs": float(np.abs(difference).max()),
        "relative": float(np.linalg.norm(difference) / max(np.linalg.norm(reference), 1e-12)),
    }


@experiment(
    "g3_candle_transformer_port",
    defaults={
        "width": 64,
        "heads": 4,
        # Batch > 1 on purpose: the contiguity bug in a batched attention path
        # is invisible at batch 1, which is the only shape a single-sequence
        # demo would ever exercise.
        "batches": [1, 4],
        "length": 16,
        "timing_repeats": 50,
        # Not a tuning knob: the spread across these, on both an idle and a
        # loaded machine, is the measurement.
        "thread_counts": [1, 2, 4],
        "contention_processes": 4,
        "build": True,
    },
    description="A transformer block ported to Rust/Candle and diffed against the PyTorch original.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Rust/Candle port of one transformer-block forward pass"""
    torch = vision.require_torch()
    from parts.c_sequence_attention.c3_transformer_from_scratch import make_transformer

    summary: dict[str, float] = {}
    scratch = Path(os.environ.get("MLLADDER_SCRATCH", "/tmp")) / f"g3-port-{exp.seed}"

    if exp["build"]:
        ok, message = build_port()
        summary["build/ok"] = float(ok)
        print(f"[g3] cargo build: {'ok' if ok else 'FAILED'} -- {message}")
        if not ok:
            print("[g3] no Rust toolchain or the crate does not compile; the numerical comparison cannot run")
            return summary
    if not BINARY.exists():
        print(f"[g3] {BINARY} is missing; run `cargo build --release` in rust/")
        summary["build/ok"] = 0.0
        return summary
    summary["build/ok"] = 1.0

    # The block under test, built by the same factory rung C3 trains.
    CharTransformer, _ = make_transformer()
    torchkit.seed_torch(exp.seed)
    model = CharTransformer(65, width=exp["width"], heads=exp["heads"], depth=1, block_size=exp["length"])
    block = model.blocks[0].eval()
    summary["block/parameters"] = float(sum(p.numel() for p in block.parameters()))
    print(f"[g3] block: width {exp['width']}, {exp['heads']} heads, {int(summary['block/parameters']):,} parameters")

    errors: dict[str, list[float]] = {"exact GELU": [], "tanh GELU": []}
    for batch in exp["batches"]:
        generator = torch.Generator().manual_seed(exp.seed + batch)
        x = torch.randn(batch, exp["length"], exp["width"], generator=generator)
        with torch.no_grad():
            reference, _ = block(x)
        reference_np = reference.numpy()

        weights_path, input_path = export(block, x, scratch / f"b{batch}")
        for gelu, label in (("exact", "exact GELU"), ("tanh", "tanh GELU")):
            output_path = scratch / f"b{batch}" / f"rust-{gelu}.npy"
            ok, message = run_port(weights_path, input_path, output_path, heads=exp["heads"], gelu=gelu)
            if not ok:
                raise RuntimeError(f"the ported block failed at batch {batch} with {gelu} GELU: {message}")
            measured = disagreement(reference_np, np.load(output_path))
            errors[label].append(measured["relative"])
            summary[f"batch{batch}/{gelu}/max_abs"] = measured["max_abs"]
            summary[f"batch{batch}/{gelu}/relative"] = measured["relative"]
            print(
                f"[g3] batch {batch}, {gelu:<5} GELU: max |diff| {measured['max_abs']:.3e}, "
                f"relative {measured['relative']:.3e}"
            )

    # -- the two findings the numbers support ---------------------------------
    summary["agreement/relative"] = max(summary[f"batch{b}/exact/relative"] for b in exp["batches"])
    summary["gelu/tanh_over_exact"] = max(errors["tanh GELU"]) / max(max(errors["exact GELU"]), 1e-12)
    summary["gelu/tanh_relative"] = max(errors["tanh GELU"])
    print(
        f"[g3] with the matching GELU the port agrees to {summary['agreement/relative']:.2e} relative; "
        f"with Candle's default it is {summary['gelu/tanh_over_exact']:.0f}x worse "
        f"({summary['gelu/tanh_relative']:.2e}) -- same code, one function swapped"
    )
    # Batch 1 cannot see a batched-layout bug, so state that it was tested.
    summary["agreement/batches_checked"] = float(len(exp["batches"]))

    # -- speed, and two traps in measuring it ---------------------------------
    # The first version of this compared "PyTorch" against "the ported binary"
    # and made Rust look 10x faster; it was measuring process startup. The
    # replacement measured PyTorch at 1/2/4 threads and found an 84 ms outlier
    # at 4 -- 192x the single-threaded time -- which got written up as the cost
    # of a thread pool. It was not: that run shared a 4-core box with a sweep
    # that was using all of it. On an idle machine the same sweep spreads by
    # 1.2x. What is real is the *interaction*, so the contention is now created
    # on purpose and measured on both sides of it.
    batch = exp["batches"][-1]
    generator = torch.Generator().manual_seed(exp.seed)
    x = torch.randn(batch, exp["length"], exp["width"], generator=generator)

    def time_threads(threads: int) -> float:
        torch.set_num_threads(threads)
        with torch.no_grad():
            for _ in range(20):
                block(x)
            started = time.time()
            for _ in range(exp["timing_repeats"]):
                block(x)
            return (time.time() - started) / exp["timing_repeats"] * 1e3

    restore = torch.get_num_threads()
    idle: dict[int, float] = {}
    busy: dict[int, float] = {}
    try:
        for threads in exp["thread_counts"]:
            idle[threads] = time_threads(threads)
        load = spin(exp["contention_processes"])
        try:
            for threads in exp["thread_counts"]:
                busy[threads] = time_threads(threads)
        finally:
            for process in load:
                process.kill()
            for process in load:
                process.wait(timeout=10)
    finally:
        torch.set_num_threads(restore)

    for threads in exp["thread_counts"]:
        summary[f"speed/idle/{threads}_threads_ms"] = idle[threads]
        summary[f"speed/busy/{threads}_threads_ms"] = busy[threads]
    summary["speed/idle_spread"] = max(idle.values()) / max(min(idle.values()), 1e-12)
    summary["speed/busy_spread"] = max(busy.values()) / max(min(busy.values()), 1e-12)
    widest = max(exp["thread_counts"])
    summary["speed/contention_cost_multithreaded"] = busy[widest] / max(idle[widest], 1e-12)
    summary["speed/contention_cost_singlethreaded"] = busy[1] / max(idle[1], 1e-12)
    print(
        "[g3] idle:  "
        + ", ".join(f"{n}t {idle[n]:.3f} ms" for n in exp["thread_counts"])
        + f"  (spread {summary['speed/idle_spread']:.1f}x)"
    )
    print(
        f"[g3] with {exp['contention_processes']} competing processes: "
        + ", ".join(f"{n}t {busy[n]:.3f} ms" for n in exp["thread_counts"])
        + f"  (spread {summary['speed/busy_spread']:.1f}x)"
    )
    print(
        f"[g3] contention costs the {widest}-thread pool "
        f"{summary['speed/contention_cost_multithreaded']:.1f}x and the single thread "
        f"{summary['speed/contention_cost_singlethreaded']:.2f}x -- the pool spins for cores it "
        f"cannot get, and one thread simply waits its turn"
    )

    # The ported binary, measured honestly: a whole process, npz parse included.
    weights_path, input_path = export(block, x, scratch / "timing")
    output_path = scratch / "timing" / "out.npy"
    started = time.time()
    for _ in range(exp["timing_repeats"]):
        run_port(weights_path, input_path, output_path, heads=exp["heads"], gelu="exact")
    rust_seconds = (time.time() - started) / exp["timing_repeats"]
    summary["speed/rust_process_ms"] = rust_seconds * 1e3
    print(
        f"[g3] the ported binary takes {rust_seconds * 1e3:.1f} ms per call, but that is process startup "
        f"and npz parsing around a {min(idle.values()):.3f} ms kernel -- it is not a comparison and is "
        f"not reported as one"
    )

    exp.log_figure(
        "threads",
        plotting.plot_curves(
            {
                "idle machine": [idle[n] for n in exp["thread_counts"]],
                f"{exp['contention_processes']} competing processes": [busy[n] for n in exp["thread_counts"]],
            },
            x=exp["thread_counts"],
            title=f"PyTorch forward pass, {int(summary['block/parameters']):,} parameters",
            xlabel="torch intra-op threads",
            ylabel="milliseconds",
            logy=True,
            marker="o",
        ),
    )
    exp.log_figure(
        "agreement",
        plotting.plot_comparison(
            {
                "exact GELU (matching)": max(errors["exact GELU"]),
                "tanh GELU (Candle default)": max(errors["tanh GELU"]),
            },
            title="Relative disagreement with the PyTorch block",
            ylabel="relative error",
        ),
    )
    exp.log_figure(
        "port",
        plotting.plot_architecture(
            {
                "PyTorch (rung C3)": [
                    ("x", "(B, L, d)"),
                    ("LayerNorm(eps=1e-5)"),
                    ("qkv -> chunk(3)", "view/transpose are free"),
                    ("softmax(QK^T/sqrt(d_k) + mask)V"),
                    ("x + proj(attn)"),
                    ("x + mlp(LayerNorm(x))", "nn.GELU() = exact"),
                ],
                "Rust / Candle": [
                    ("x", "read from .npy"),
                    ("LayerNorm::new(w, b, 1e-5)", "eps is explicit"),
                    ("qkv -> narrow x3", "transpose then .contiguous()"),
                    ("softmax_last_dim(masked scores)", "-1e30, not -inf"),
                    ("x + proj(attn)"),
                    ("x + mlp(LayerNorm(x))", "gelu_erf(), NOT gelu()"),
                ],
            },
            highlight={"Rust / Candle": [1, 2, 3, 5]},
            title="The same block twice: highlighted rows are what PyTorch had decided for you",
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
