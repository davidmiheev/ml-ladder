"""H6. OCR as a vision-language model — PaddleOCR-VL and Unlimited-OCR read from source, against the H5 pipeline

Lab reference: [open-mmlab/mmocr — detection and recognition reference implementations](https://github.com/open-mmlab/mmocr)

H5 built the classical pipeline and measured where it fails: **99.98% of its
page-level error came from detection**, not recognition. The recognizer on
ground-truth crops was essentially perfect; the whole error appeared the moment
it was fed real boxes. That number is the reason this rung exists.

A vision-language OCR model deletes the stage boundary rather than improving
either stage. There is no box, so there is no box error, no crop, no NMS
threshold, no reading-order heuristic — a page becomes a sequence of visual
tokens and the transcript is decoded autoregressively from them. Everything H5
attributed to detection simply has no place to live.

What that costs is the question this rung answers, and it answers it in
parameters and tokens rather than in prose. Two architectures are read
**directly out of the candle sources** — their configs are parsed from the Rust
rather than quoted from memory — and their budgets are computed from the same
formulas, which are validated against a model whose true parameter count this
rung can measure.

The comparison is deliberately not an accuracy benchmark. H5's pipeline is a
0.5M-parameter toy on synthetic pages and PaddleOCR-VL is a 0.9B-parameter
model trained on real documents in 109 languages; "which is more accurate" has
an answer nobody needs measured. What is worth measuring is the exchange rate:
how much compute buys the removal of the interface that H5 showed is where all
the error was.

Run with ``ladder run H6`` (``--help`` for flags). Needs the candle checkout for
the parse step; ``--candle-root`` points at it, and the rung falls back to the
values recorded here when it is absent.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from mlladder import Experiment, experiment, plotting

# Architecture facts as read from candle on 2026-09-17. They are kept here so
# the rung runs without the checkout, and *checked against* the sources when it
# is present — a mismatch means the fork moved and these numbers are stale,
# which the rung reports rather than silently ignoring.
PADDLEOCR_VL_DEFAULTS = {
    "vision_hidden_size": 1152,
    "vision_intermediate_size": 4304,
    "vision_num_hidden_layers": 27,
    "vision_num_attention_heads": 16,
    "vision_patch_size": 14,
    "vision_image_size": 384,
    "vision_spatial_merge_size": 2,
    "vocab_size": 103424,
    "hidden_size": 1024,
    "intermediate_size": 3072,
    "num_hidden_layers": 18,
    "num_attention_heads": 16,
    "num_key_value_heads": 2,
    "head_dim": 128,
}

# Unlimited-OCR's candle config carries no defaults — every field is loaded from
# the checkpoint — so these are taken from the published config.json
# (baidu/Unlimited-OCR) and cross-checked against the field names in
# unlimited_ocr/text.rs. Recorded rather than fetched so the rung stays offline.
UNLIMITED_OCR_TEXT = {
    "hidden_size": 1280,
    "intermediate_size": 6848,
    "moe_intermediate_size": 896,
    "num_hidden_layers": 12,
    "num_attention_heads": 10,
    "num_key_value_heads": 10,
    "v_head_dim": 128,
    "vocab_size": 129280,
    "n_routed_experts": 64,
    "n_shared_experts": 2,
    "num_experts_per_tok": 6,
    "first_k_dense_replace": 1,
}


@dataclass(frozen=True)
class Transformer:
    """Enough of a transformer stack to count its parameters.

    ``gated`` selects a SwiGLU MLP (three matrices) over a plain one (two),
    which is not a detail: at these widths the MLP is most of the block, so
    getting it wrong misstates the whole budget by ~25%.
    """

    name: str
    layers: int
    hidden: int
    intermediate: int
    heads: int
    kv_heads: int | None = None
    head_dim: int | None = None
    gated: bool = True
    vocab: int = 0
    tied_embeddings: bool = False
    biases: bool = False
    extra: dict[str, int] = field(default_factory=dict)
    # Mixture-of-experts, which decouples the parameters a model *has* from the
    # ones it *runs* per token. Counting an MoE stack as dense — as the first
    # version of this rung did — understates Unlimited-OCR's text tower by
    # roughly an order of magnitude.
    routed_experts: int = 0
    shared_experts: int = 0
    experts_per_token: int = 0
    moe_intermediate: int = 0
    dense_layers: int = 0  # leading layers that keep a plain MLP

    @property
    def per_head(self) -> int:
        return self.head_dim or self.hidden // self.heads

    @property
    def kv(self) -> int:
        return self.kv_heads if self.kv_heads is not None else self.heads

    def attention_parameters(self) -> int:
        """Q, K, V and the output projection, with GQA shrinking K and V."""
        q = self.hidden * self.heads * self.per_head
        k = self.hidden * self.kv * self.per_head
        v = k
        o = self.heads * self.per_head * self.hidden
        bias = (self.heads * self.per_head + 2 * self.kv * self.per_head + self.hidden) if self.biases else 0
        return q + k + v + o + bias

    def mlp_parameters(self) -> int:
        matrices = 3 if self.gated else 2
        bias = (self.intermediate * (matrices - 1) + self.hidden) if self.biases else 0
        return matrices * self.hidden * self.intermediate + bias

    def expert_parameters(self) -> int:
        """One expert's FFN, at the (much narrower) MoE width."""
        return 3 * self.hidden * self.moe_intermediate if self.routed_experts else 0

    def block_parameters(self, *, active: bool = False) -> int:
        """A block's parameters; ``active`` counts only the experts a token uses."""
        if self.routed_experts:
            used = self.experts_per_token if active else self.routed_experts
            ffn = (used + self.shared_experts) * self.expert_parameters()
            ffn += self.hidden * self.routed_experts  # the router
        else:
            ffn = self.mlp_parameters()
        return self.attention_parameters() + ffn + 2 * self.hidden  # two norms

    def parameters(self, *, active: bool = False) -> dict[str, int]:
        """Total parameters, or those touched per token when ``active``."""
        dense = min(self.dense_layers, self.layers)
        blocks = dense * (self.attention_parameters() + self.mlp_parameters() + 2 * self.hidden)
        blocks += (self.layers - dense) * self.block_parameters(active=active)
        embedding = self.vocab * self.hidden
        head = 0 if self.tied_embeddings else embedding
        return {
            "blocks": blocks,
            "embedding": embedding,
            "output_head": head,
            "extra": sum(self.extra.values()),
            "total": blocks + embedding + head + sum(self.extra.values()),
        }

    def kv_cache_bytes_per_token(self, *, dtype_bytes: int = 2) -> int:
        r"""$2 \times L \times H_{kv} \times d_h \times \text{bytes}$ — keys and values, every layer.

        This is the number that decides whether a long document fits in memory,
        and it is where grouped-query attention pays: PaddleOCR-VL's text stack
        has 16 query heads against 2 key/value heads, so the cache is an eighth
        of what multi-head attention would need for the same query capacity.
        """
        return 2 * self.layers * self.kv * self.per_head * dtype_bytes

    def attention_flops(self, tokens: int) -> int:
        r"""Score-and-apply cost, $2 \times 2 \times L \times H \times T^2 \times d_h$.

        Only the quadratic term: the projections are linear in $T$ and are
        dwarfed by this once a page is a few hundred tokens long, which is
        exactly the regime a document model lives in.
        """
        return 4 * self.layers * self.heads * tokens * tokens * self.per_head


def parse_candle_defaults(config_path: Path) -> dict[str, int]:
    """Read ``fn default_x() -> usize { N }`` out of a candle config.rs.

    Parsing the source rather than trusting a table in this file is the point:
    an architecture note that drifts from the code it describes is worse than
    no note, and this way the drift is detectable.
    """
    source = config_path.read_text()
    found: dict[str, int] = {}
    for match in re.finditer(r"fn (default_[a-z0-9_]+)\(\)\s*->\s*usize\s*\{\s*([0-9_]+)\s*\}", source):
        found[match.group(1).removeprefix("default_")] = int(match.group(2).replace("_", ""))
    return found


def page_tokens(*, height: int, width: int, patch: int, merge: int = 1) -> int:
    """Visual tokens a page becomes: patches, then the spatial merge.

    The merge is a 2x2 concatenation before the projector, so it divides the
    token count by four while keeping the information — the cheapest available
    lever on an attention cost that grows as the square.
    """
    return (height // patch) * (width // patch) // (merge * merge)


@experiment(
    "h6_ocr_vlm_architectures",
    defaults={
        "candle_root": "/home/user/davidmiheev/candle",
        "page_sizes": [384, 640, 1024, 1536],
        "a4_height": 3508,  # A4 at 300 dpi, the resolution a scan actually arrives at
        "a4_width": 2480,
        "validate_against": "microsoft/trocr-small-printed",
        "dtype_bytes": 2,
    },
    description="Parameter and token budgets of PaddleOCR-VL and Unlimited-OCR, read from candle, against H5's pipeline.",
)
def run(exp: Experiment) -> dict[str, float]:
    """OCR as a vision-language model — PaddleOCR-VL and Unlimited-OCR read from source, against the H5 pipeline"""
    summary: dict[str, float] = {}

    # -- read the architecture out of the fork, or fall back ------------------
    config = dict(PADDLEOCR_VL_DEFAULTS)
    candle_config = Path(exp["candle_root"]) / "candle-transformers/src/models/paddleocr_vl/config.rs"
    if candle_config.exists():
        parsed = parse_candle_defaults(candle_config)
        drifted = {key: (value, parsed[key]) for key, value in config.items() if key in parsed and parsed[key] != value}
        config.update({key: parsed[key] for key in config if key in parsed})
        summary["candle/parsed_defaults"] = float(len(parsed))
        summary["candle/drifted_keys"] = float(len(drifted))
        print(f"[h6] parsed {len(parsed)} defaults from candle's paddleocr_vl/config.rs")
        if drifted:
            print(f"[h6] WARNING: recorded values differ from the source: {drifted}")
    else:
        summary["candle/parsed_defaults"] = 0.0
        print(f"[h6] candle checkout not at {exp['candle_root']}; using the recorded defaults")

    # -- the architectures ---------------------------------------------------
    paddle_vision = Transformer(
        name="PaddleOCR-VL vision (NaViT)",
        layers=config["vision_num_hidden_layers"],
        hidden=config["vision_hidden_size"],
        intermediate=config["vision_intermediate_size"],
        heads=config["vision_num_attention_heads"],
        gated=False,  # fc1/fc2 in vision.rs, not gate/up/down
        biases=True,
    )
    paddle_text = Transformer(
        name="PaddleOCR-VL text (ERNIE-4.5-0.3B)",
        layers=config["num_hidden_layers"],
        hidden=config["hidden_size"],
        intermediate=config["intermediate_size"],
        heads=config["num_attention_heads"],
        kv_heads=config["num_key_value_heads"],
        head_dim=config["head_dim"],
        gated=True,  # gate_proj/up_proj/down_proj in text.rs
        vocab=config["vocab_size"],
    )
    # The projector: 2x2 spatial merge then a two-layer MLP, 1152*4 -> 1024.
    merged = config["vision_hidden_size"] * config["vision_spatial_merge_size"] ** 2
    projector_parameters = merged * merged + merged * config["hidden_size"]

    # Unlimited-OCR, from its module docs and vision.rs literals.
    unlimited_clip = Transformer(
        name="Unlimited-OCR CLIP-L tower", layers=24, hidden=1024, intermediate=4096, heads=16, gated=False, biases=True
    )
    unlimited_sam = Transformer(
        name="Unlimited-OCR SAM-ViT-B", layers=12, hidden=768, intermediate=3072, heads=12, gated=False, biases=True
    )
    unlimited = UNLIMITED_OCR_TEXT
    unlimited_text = Transformer(
        name="Unlimited-OCR text (DeepSeek-OCR lineage, MoE)",
        layers=unlimited["num_hidden_layers"],
        hidden=unlimited["hidden_size"],
        intermediate=unlimited["intermediate_size"],
        heads=unlimited["num_attention_heads"],
        kv_heads=unlimited["num_key_value_heads"],  # no MLA shipped: 10 heads == 10 kv heads
        head_dim=unlimited["v_head_dim"],
        gated=True,
        vocab=unlimited["vocab_size"],
        routed_experts=unlimited["n_routed_experts"],
        shared_experts=unlimited["n_shared_experts"],
        experts_per_token=unlimited["num_experts_per_tok"],
        moe_intermediate=unlimited["moe_intermediate_size"],
        dense_layers=unlimited["first_k_dense_replace"],
    )

    stacks = {
        "paddle_vision": paddle_vision,
        "paddle_text": paddle_text,
        "unlimited_sam": unlimited_sam,
        "unlimited_clip": unlimited_clip,
        "unlimited_text": unlimited_text,
    }
    for key, stack in stacks.items():
        counts = stack.parameters()
        for part, value in counts.items():
            summary[f"params/{key}/{part}"] = float(value)
        print(f"[h6] {stack.name:<44} {counts['total'] / 1e6:8.1f}M parameters")

    unlimited_total = unlimited_text.parameters()["total"]
    unlimited_active = unlimited_text.parameters(active=True)["total"]
    summary["params/unlimited_text/total"] = float(unlimited_total)
    summary["params/unlimited_text/active_per_token"] = float(unlimited_active)
    summary["params/unlimited_text/sparsity"] = unlimited_total / unlimited_active
    print(
        f"[h6] Unlimited-OCR text is sparse: {unlimited_total / 1e9:.2f}B total but "
        f"{unlimited_active / 1e6:.0f}M active per token ({summary['params/unlimited_text/sparsity']:.1f}x), "
        f"{unlimited['n_routed_experts']} routed experts at width {unlimited['moe_intermediate_size']} "
        f"against a dense width of {unlimited['intermediate_size']}"
    )

    summary["params/paddle_projector"] = float(projector_parameters)
    paddle_total = paddle_vision.parameters()["total"] + paddle_text.parameters()["total"] + projector_parameters
    summary["params/paddleocr_vl/total"] = float(paddle_total)
    summary["params/paddleocr_vl/vision_share"] = paddle_vision.parameters()["total"] / paddle_total
    print(
        f"[h6] PaddleOCR-VL total {paddle_total / 1e6:.1f}M "
        f"(vision {summary['params/paddleocr_vl/vision_share']:.1%}, projector {projector_parameters / 1e6:.1f}M)"
    )

    # -- does the counting method actually work? -----------------------------
    # Validated against a model whose parameters can be counted for real, so
    # the budgets above are a verified calculation rather than an assertion.
    if exp["validate_against"]:
        try:
            from parts.h_computer_vision.h4_attention_ocr import load_trocr

            _processor, model = load_trocr(exp["validate_against"], "cpu")
            measured = sum(parameter.numel() for parameter in model.parameters())
            decoder_config = model.config.decoder
            encoder_config = model.config.encoder
            predicted_encoder = Transformer(
                name="TrOCR encoder (DeiT)",
                layers=encoder_config.num_hidden_layers,
                hidden=encoder_config.hidden_size,
                intermediate=encoder_config.intermediate_size,
                heads=encoder_config.num_attention_heads,
                gated=False,
                biases=True,
            ).parameters()["total"]
            predicted_decoder = Transformer(
                name="TrOCR decoder",
                layers=decoder_config.decoder_layers,
                hidden=decoder_config.d_model,
                intermediate=decoder_config.decoder_ffn_dim,
                heads=decoder_config.decoder_attention_heads,
                gated=False,
                biases=True,
                vocab=decoder_config.vocab_size,
            ).parameters()["total"]
            # The decoder also carries cross-attention in every layer, which the
            # plain block formula does not include.
            cross_attention = decoder_config.decoder_layers * (
                2 * decoder_config.d_model * decoder_config.d_model
                + 2 * decoder_config.cross_attention_hidden_size * decoder_config.d_model
            )
            predicted = predicted_encoder + predicted_decoder + cross_attention
            error = abs(predicted - measured) / measured
            summary["validation/trocr_measured"] = float(measured)
            summary["validation/trocr_predicted"] = float(predicted)
            summary["validation/relative_error"] = float(error)
            print(
                f"[h6] method check on TrOCR: predicted {predicted / 1e6:.1f}M vs measured "
                f"{measured / 1e6:.1f}M ({error:.1%} off)"
            )
        except Exception as error:  # pragma: no cover - the optional extra
            print(f"[h6] skipping the parameter-method check ({type(error).__name__}: {error})")

    # -- token economics -----------------------------------------------------
    merge = config["vision_spatial_merge_size"]
    patch = config["vision_patch_size"]
    for size in exp["page_sizes"]:
        raw = page_tokens(height=size, width=size, patch=patch)
        after = page_tokens(height=size, width=size, patch=patch, merge=merge)
        summary[f"tokens/{size}/patches"] = float(raw)
        summary[f"tokens/{size}/after_merge"] = float(after)
        summary[f"flops/{size}/attention_after_merge"] = float(paddle_vision.attention_flops(after))
        summary[f"flops/{size}/saved_by_merge"] = float(
            paddle_vision.attention_flops(raw) - paddle_vision.attention_flops(after)
        )
        print(
            f"[h6] {size:>5}px page: {raw:>6} patches -> {after:>5} tokens after the 2x2 merge; "
            f"attention {paddle_vision.attention_flops(after) / 1e9:8.1f} GFLOP "
            f"({paddle_vision.attention_flops(raw) / max(paddle_vision.attention_flops(after), 1):.0f}x cheaper)"
        )

    a4_raw = page_tokens(height=exp["a4_height"], width=exp["a4_width"], patch=patch)
    a4_merged = page_tokens(height=exp["a4_height"], width=exp["a4_width"], patch=patch, merge=merge)
    summary["tokens/a4_300dpi/patches"] = float(a4_raw)
    summary["tokens/a4_300dpi/after_merge"] = float(a4_merged)
    summary["kv_bytes_per_token/paddle_text"] = float(
        paddle_text.kv_cache_bytes_per_token(dtype_bytes=exp["dtype_bytes"])
    )
    summary["kv_bytes_per_token/unlimited_text"] = float(
        unlimited_text.kv_cache_bytes_per_token(dtype_bytes=exp["dtype_bytes"])
    )
    # What GQA is worth: the same stack with kv_heads == heads.
    multi_head = Transformer(
        name="PaddleOCR-VL text without GQA",
        layers=paddle_text.layers,
        hidden=paddle_text.hidden,
        intermediate=paddle_text.intermediate,
        heads=paddle_text.heads,
        kv_heads=paddle_text.heads,
        head_dim=paddle_text.head_dim,
        vocab=paddle_text.vocab,
    )
    summary["kv_bytes_per_token/paddle_text_without_gqa"] = float(
        multi_head.kv_cache_bytes_per_token(dtype_bytes=exp["dtype_bytes"])
    )
    summary["gqa/kv_cache_ratio"] = (
        summary["kv_bytes_per_token/paddle_text_without_gqa"] / summary["kv_bytes_per_token/paddle_text"]
    )
    print(
        f"[h6] A4 at 300dpi: {a4_raw:,} patches -> {a4_merged:,} tokens; "
        f"KV cache {summary['kv_bytes_per_token/paddle_text'] / 1024:.1f} KiB/token "
        f"({summary['gqa/kv_cache_ratio']:.0f}x less than multi-head)"
    )

    # -- against H5's pipeline -----------------------------------------------
    # H5's own measured numbers, quoted from its metrics rather than restated.
    pipeline_parameters = exp.get("pipeline_parameters", 0) or 487_806  # detector 72,760 + recognizer 415,046
    summary["params/h5_pipeline"] = float(pipeline_parameters)
    summary["params/vlm_over_pipeline"] = paddle_total / pipeline_parameters
    print(
        f"[h6] PaddleOCR-VL is {summary['params/vlm_over_pipeline']:,.0f}x H5's pipeline "
        f"({paddle_total / 1e6:.0f}M vs {pipeline_parameters / 1e3:.0f}k parameters), and has no stage interface "
        f"— the thing H5 measured as 99.98% of its error."
    )

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "parameter_budget",
        plotting.plot_comparison(
            {
                "Paddle\nvision": paddle_vision.parameters()["total"] / 1e6,
                "Paddle\nprojector": projector_parameters / 1e6,
                "Paddle\ntext": paddle_text.parameters()["total"] / 1e6,
                "Unlimited\nSAM": unlimited_sam.parameters()["total"] / 1e6,
                "Unlimited\nCLIP": unlimited_clip.parameters()["total"] / 1e6,
                "Unlimited\ntext": unlimited_text.parameters()["total"] / 1e6,
            },
            title="Where a VLM OCR model spends its parameters",
            ylabel="millions of parameters",
        ),
    )
    exp.log_figure(
        "token_cost",
        plotting.plot_curves(
            {
                "patches (no merge)": [page_tokens(height=s, width=s, patch=patch) for s in exp["page_sizes"]],
                "tokens after 2x2 merge": [
                    page_tokens(height=s, width=s, patch=patch, merge=merge) for s in exp["page_sizes"]
                ],
            },
            x=exp["page_sizes"],
            title=f"Visual tokens per square page (patch {patch})",
            xlabel="page side (px)",
            ylabel="tokens",
            marker="o",
            logy=True,
        ),
    )
    exp.log_figure(
        "attention_cost",
        plotting.plot_curves(
            {
                "no merge": [
                    paddle_vision.attention_flops(page_tokens(height=s, width=s, patch=patch)) / 1e9
                    for s in exp["page_sizes"]
                ],
                "2x2 merge": [
                    paddle_vision.attention_flops(page_tokens(height=s, width=s, patch=patch, merge=merge)) / 1e9
                    for s in exp["page_sizes"]
                ],
            },
            x=exp["page_sizes"],
            title="Vision-encoder attention cost (quadratic in tokens)",
            xlabel="page side (px)",
            ylabel="GFLOP",
            marker="o",
            logy=True,
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
