"""E5. Production text-to-image architectures read from source — Stable Diffusion's U-Net against FLUX's MMDiT

Lab reference: [lucidrains/denoising-diffusion-pytorch — diffusion reference](https://github.com/lucidrains/denoising-diffusion-pytorch)

E3 built a DDPM: a U-Net that predicts noise, trained on pixels, sampled by
walking a fixed schedule backwards. Every production text-to-image model is a
descendant of that, and each generation changed one of its three parts. This
rung reads two of them out of the accompanying candle fork and prices the
changes, the way C5 does for decoders and H6 for OCR.

**Where the diffusion happens.** E3 diffuses pixels, so its cost scales with
resolution directly. Stable Diffusion diffuses in the latent space of a trained
autoencoder at an 8x downsample, which is 64x fewer positions to denoise — the
single change that made megapixel generation affordable, and the reason E1's
VAE is not a dead end but a component.

**How the text gets in.** SD's U-Net accepts the prompt through cross-attention:
image queries, text keys and values, one direction only. FLUX's MMDiT drops the
U-Net for a transformer and makes both streams first-class, concatenating image
and text tokens and running *joint* attention over them — so the text
representation is updated by the image as well as the reverse.

**What the network is asked to predict.** E3 predicts the noise added by a
hand-designed schedule. Flow matching instead learns a velocity field along a
straight path between noise and data, which is why its samplers need far fewer
steps than E3's sweep found necessary.

Run with ``ladder run E5`` (``--help`` for flags). ``--candle-root`` points at
the checkout used to verify the recorded shapes.
"""

from __future__ import annotations

from pathlib import Path

from mlladder import Experiment, experiment, plotting
from mlladder.archspec import Transformer

# Recorded from the published configs and cross-checked against candle's
# implementations in candle-transformers/src/models/{stable_diffusion,flux,mmdit}.
# Sources are named so a reader can re-derive rather than trust.
ARCHITECTURES = {
    "SD-1.5 U-Net": {
        "source": "runwayml/stable-diffusion-v1-5 (unet)",
        "candle": "stable_diffusion/unet_2d.rs",
        "kind": "unet",
        "latent_channels": 4,
        "downsample": 8,
        "cross_attention_dim": 768,  # CLIP ViT-L/14 text width
        "block_channels": (320, 640, 1280, 1280),
        "layers_per_block": 2,
        "attention_head_dim": 8,
    },
    "FLUX.1 MMDiT": {
        "source": "black-forest-labs/FLUX.1-schnell",
        "candle": "flux/model.rs",
        "kind": "mmdit",
        "latent_channels": 16,
        "downsample": 8,
        "hidden_size": 3072,
        "num_heads": 24,
        "depth": 19,  # double-stream blocks: image and text updated jointly
        "depth_single_blocks": 38,  # single-stream: the two streams concatenated
        "mlp_ratio": 4.0,
        "context_in_dim": 4096,  # T5-XXL text width
        "patch_size": 2,
    },
}


def latent_positions(*, height: int, width: int, downsample: int, patch: int = 1) -> int:
    """Positions the denoiser actually operates on, after VAE and patching.

    Two compressions in series, and they multiply: the autoencoder's spatial
    downsample and, for a transformer denoiser, a further patchify. FLUX's
    2x2 patching on top of an 8x VAE means one token per 16x16 pixels.
    """
    return (height // (downsample * patch)) * (width // (downsample * patch))


def mmdit_parameters(config: dict) -> dict[str, int]:
    r"""Parameter budget of a double/single-stream MMDiT stack.

    A **double-stream** block carries two complete transformer blocks — one for
    image tokens, one for text — whose attention is computed jointly over the
    concatenation. That is the "MM" in MMDiT: the modalities have separate
    weights but a shared attention operation, so each can read the other. A
    **single-stream** block runs one set of weights over the concatenated
    sequence, which is cheaper and is where FLUX spends most of its depth.
    """
    hidden = config["hidden_size"]
    intermediate = int(hidden * config["mlp_ratio"])
    block = Transformer(
        name="mmdit block",
        layers=1,
        hidden=hidden,
        intermediate=intermediate,
        heads=config["num_heads"],
        gated=False,
        biases=True,
    ).block_parameters()
    # Modulation: each block is conditioned on timestep and pooled text through
    # adaptive layer norm, which costs six vectors of scale/shift per stream.
    modulation = 6 * hidden * hidden
    double = config["depth"] * (2 * block + 2 * modulation)
    single = config["depth_single_blocks"] * (block + modulation)
    projections = config["context_in_dim"] * hidden + config["latent_channels"] * config["patch_size"] ** 2 * hidden
    return {
        "double_stream": double,
        "single_stream": single,
        "projections": projections,
        "total": double + single + projections,
    }


def unet_parameters(config: dict) -> dict[str, int]:
    """Rough parameter budget of an SD-style U-Net.

    Counted as: two 3x3 convolutions per residual layer at each resolution, plus
    self- and cross-attention blocks at the resolutions that carry them. This is
    an estimate, not an exact count — the real module has embedding projections,
    shortcut convolutions and a decoder that mirrors the encoder — so the rung
    reports it as such rather than implying the precision the MMDiT count has.
    """
    channels = config["block_channels"]
    convolutions = 0
    attention = 0
    for index, width in enumerate(channels):
        previous = channels[index - 1] if index else width
        for layer in range(config["layers_per_block"]):
            inputs = previous if layer == 0 else width
            convolutions += 9 * inputs * width + 9 * width * width
        if index > 0:  # attention lives at the lower resolutions
            # self-attention (4 projections) + cross-attention (q, o on image;
            # k, v from the text width)
            attention += 4 * width * width + 2 * width * width + 2 * config["cross_attention_dim"] * width
    # The decoder mirrors the encoder and additionally consumes skip channels.
    return {"encoder": convolutions + attention, "total": 2 * (convolutions + attention)}


@experiment(
    "e5_flux_and_sd",
    defaults={
        "candle_root": "/home/user/davidmiheev/candle",
        "resolutions": [256, 512, 1024, 2048],
        "headline_resolution": 1024,
        "prompt_tokens": 512,
        "pixel_baseline_channels": 3,
    },
    description="Latent vs pixel diffusion, cross- vs joint attention, and what each costs at resolution.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Production text-to-image architectures read from source — Stable Diffusion's U-Net against FLUX's MMDiT"""
    summary: dict[str, float] = {}

    candle = Path(exp["candle_root"]) / "candle-transformers/src/models"
    present = {name: (candle / config["candle"]).exists() for name, config in ARCHITECTURES.items()}
    summary["candle/modules_found"] = float(sum(present.values()))
    print(f"[e5] candle modules found: {present}")

    sd = ARCHITECTURES["SD-1.5 U-Net"]
    flux = ARCHITECTURES["FLUX.1 MMDiT"]

    # -- what latent diffusion bought ----------------------------------------
    headline = exp["headline_resolution"]
    for resolution in exp["resolutions"]:
        pixels = resolution * resolution
        sd_positions = latent_positions(height=resolution, width=resolution, downsample=sd["downsample"])
        flux_tokens = latent_positions(
            height=resolution, width=resolution, downsample=flux["downsample"], patch=flux["patch_size"]
        )
        summary[f"positions/{resolution}/pixel"] = float(pixels)
        summary[f"positions/{resolution}/sd_latent"] = float(sd_positions)
        summary[f"positions/{resolution}/flux_tokens"] = float(flux_tokens)
        summary[f"positions/{resolution}/latent_saving"] = pixels / sd_positions
        print(
            f"[e5] {resolution:>5}px: {pixels:>9,} pixels -> {sd_positions:>7,} latent positions "
            f"({pixels / sd_positions:.0f}x) -> {flux_tokens:>6,} MMDiT tokens"
        )

    # FLUX's joint attention runs over image AND text tokens together, so the
    # sequence the quadratic applies to is longer than the image alone.
    image_tokens = latent_positions(
        height=headline, width=headline, downsample=flux["downsample"], patch=flux["patch_size"]
    )
    joint = image_tokens + exp["prompt_tokens"]
    summary["flux/image_tokens"] = float(image_tokens)
    summary["flux/joint_sequence"] = float(joint)
    summary["flux/text_share_of_sequence"] = exp["prompt_tokens"] / joint
    # Cross-attention costs image x text; joint attention costs (image+text)^2.
    cross_cost = image_tokens * exp["prompt_tokens"]
    joint_cost = joint * joint
    summary["attention/joint_over_cross"] = joint_cost / cross_cost
    print(
        f"[e5] at {headline}px FLUX attends over {image_tokens:,} image + {exp['prompt_tokens']} text = "
        f"{joint:,} tokens; joint attention costs {summary['attention/joint_over_cross']:.1f}x "
        f"the image-to-text cross-attention it replaces"
    )

    # -- parameter budgets ---------------------------------------------------
    flux_budget = mmdit_parameters(flux)
    sd_budget = unet_parameters(sd)
    for part, value in flux_budget.items():
        summary[f"params/flux/{part}"] = float(value)
    summary["params/sd_unet_estimate"] = float(sd_budget["total"])
    summary["params/flux_over_sd"] = flux_budget["total"] / sd_budget["total"]
    summary["flux/single_stream_share"] = flux_budget["single_stream"] / flux_budget["total"]
    print(
        f"[e5] FLUX MMDiT {flux_budget['total'] / 1e9:.2f}B "
        f"(double-stream {flux_budget['double_stream'] / 1e9:.2f}B, "
        f"single-stream {flux_budget['single_stream'] / 1e9:.2f}B = "
        f"{summary['flux/single_stream_share']:.0%} of it)"
    )
    print(
        f"[e5] SD-1.5 U-Net ~{sd_budget['total'] / 1e6:.0f}M (estimate), so FLUX is ~{summary['params/flux_over_sd']:.0f}x"
    )

    # -- how far can these estimates be trusted? -----------------------------
    # FLUX.1 is published at ~12B and SD-1.5's U-Net at ~860M. Stating both
    # gaps is the point: the MMDiT count is a real calculation over a uniform
    # stack, while the U-Net estimate counts convolutions and attention and
    # misses time embeddings, shortcut convolutions and the up-blocks' extra
    # skip channels. One of these numbers is load-bearing and the other is an
    # order-of-magnitude sanity check, and the difference should not be buried.
    published = {"flux": 12.0e9, "sd_unet": 860e6}
    summary["validation/flux_error"] = abs(flux_budget["total"] - published["flux"]) / published["flux"]
    summary["validation/sd_unet_error"] = abs(sd_budget["total"] - published["sd_unet"]) / published["sd_unet"]
    print(
        f"[e5] against published sizes: FLUX {flux_budget['total'] / 1e9:.2f}B vs ~12B "
        f"({summary['validation/flux_error']:.0%} off) — the MMDiT stack is uniform enough to count; "
        f"SD U-Net {sd_budget['total'] / 1e6:.0f}M vs ~860M ({summary['validation/sd_unet_error']:.0%} off), "
        f"so treat that one as a lower bound, not a count"
    )

    # -- what E3 would have cost at these resolutions ------------------------
    # E3 diffuses pixels. Stating its cost at production resolution is the
    # cleanest way to say what the latent space is worth.
    e3_positions = summary[f"positions/{headline}/pixel"]
    summary["pixel_diffusion/positions_at_headline"] = e3_positions
    summary["pixel_diffusion/attention_ratio"] = (e3_positions / image_tokens) ** 2
    print(
        f"[e3->e5] E3's pixel-space denoiser at {headline}px would work on {e3_positions:,.0f} positions "
        f"against FLUX's {image_tokens:,}; quadratic attention over them would cost "
        f"{summary['pixel_diffusion/attention_ratio']:,.0f}x as much"
    )

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "positions",
        plotting.plot_curves(
            {
                "pixels (E3's space)": [summary[f"positions/{r}/pixel"] for r in exp["resolutions"]],
                "SD latent positions": [summary[f"positions/{r}/sd_latent"] for r in exp["resolutions"]],
                "FLUX tokens (latent + 2x2 patch)": [summary[f"positions/{r}/flux_tokens"] for r in exp["resolutions"]],
            },
            x=exp["resolutions"],
            title="What the denoiser actually operates on",
            xlabel="image side (px)",
            ylabel="positions",
            marker="o",
            logx=True,
            logy=True,
        ),
    )
    exp.log_figure(
        "flux_budget",
        plotting.plot_comparison(
            {
                "double-stream\n(19 blocks)": flux_budget["double_stream"] / 1e9,
                "single-stream\n(38 blocks)": flux_budget["single_stream"] / 1e9,
                "projections": flux_budget["projections"] / 1e9,
            },
            title="Where FLUX spends its parameters",
            ylabel="billions",
        ),
    )
    exp.log_figure(
        "attention_cost",
        plotting.plot_curves(
            {
                "cross-attention (image x text)": [
                    latent_positions(height=r, width=r, downsample=8, patch=2) * exp["prompt_tokens"]
                    for r in exp["resolutions"]
                ],
                "joint attention ((image+text)^2)": [
                    (latent_positions(height=r, width=r, downsample=8, patch=2) + exp["prompt_tokens"]) ** 2
                    for r in exp["resolutions"]
                ],
            },
            x=exp["resolutions"],
            title="Conditioning cost: one-directional cross-attention against joint streams",
            xlabel="image side (px)",
            ylabel="attention entries",
            marker="o",
            logx=True,
            logy=True,
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
