"""C5. Modern decoder architectures read from source — Qwen3.5, Gemma 4 and DeepSeek-V2 against C3

Lab reference: [Annotated Transformer / nanoGPT](https://nlp.seas.harvard.edu/annotated-transformer/)

C3 built a transformer decoder by hand and C1 built a recurrent one. The
recurrent model lost: its state is a fixed-size vector, so information has to be
squeezed through it, and C1 measured the resulting horizon at 27 steps against
the LSTM's 63. Attention won by refusing to compress — it keeps every past
position and pays $O(T^2)$ for the privilege.

That victory has a bill, and it is not the quadratic term. At **generation**
time each new token attends over everything already produced, so the model must
*retain* every past key and value: a cache growing linearly in context, read in
full for every single token. Decode is memory-bound, and the cache is the
memory. Every decoder architecture since C3 is an attack on that cache:

- **grouped-query attention** shrinks the head count the cache is stored at,
  leaving query capacity untouched;
- **sliding-window attention** caps how far back a layer looks, making its cache
  constant in context and pushing long-range reach into depth instead;
- **multi-head latent attention** (DeepSeek-V2) stores one low-rank latent per
  token rather than per-head keys and values;
- **gated delta networks** (Qwen3.5 / Qwen3-Next) keep a fixed-size recurrent
  state and drop the dependence on context entirely — **which is C1's
  recurrence, returned, with the compression problem solved differently.**

So this rung does not train anything. It reads five real configurations — C3's
own, measured from this repo, and four published ones — and computes the single
quantity they are all competing on: **bytes of decode state as a function of
context length**. The architectures are named from candle's implementations in
the accompanying fork, and their numbers come from published configs with the
provenance recorded.

Run with ``ladder run C5`` (``--help`` for flags).
"""

from __future__ import annotations

from mlladder import Experiment, archspec, experiment, plotting
from mlladder.archspec import AttentionLayer, Decoder, Transformer, uniform_attention

# Configs as published on the hub, fetched and recorded on 2026-09-18. Recorded
# rather than fetched at run time so the rung is offline and deterministic; the
# mechanisms each one exercises are implemented in candle under the module named
# beside it.
CONFIGS = {
    # candle: candle-transformers/src/models/qwen3.rs — plain GQA, the baseline
    # modern decoder against which the rest are refinements.
    "Qwen3-8B": {
        "source": "Qwen/Qwen3-8B",
        "hidden_size": 4096,
        "num_hidden_layers": 36,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 12288,
        "vocab_size": 151936,
    },
    # candle: mistral.rs — the canonical sliding-window stack.
    "Mistral-7B": {
        "source": "mistralai/Mistral-7B-v0.1",
        "hidden_size": 4096,
        "num_hidden_layers": 32,
        "num_attention_heads": 32,
        "num_key_value_heads": 8,
        "head_dim": 128,
        "intermediate_size": 14336,
        "vocab_size": 32000,
        "sliding_window": 4096,
    },
    # candle: deepseek2.rs — MLA plus fine-grained MoE.
    "DeepSeek-V2-Lite": {
        "source": "deepseek-ai/DeepSeek-V2-Lite",
        "hidden_size": 2048,
        "num_hidden_layers": 27,
        "num_attention_heads": 16,
        "num_key_value_heads": 16,
        "intermediate_size": 10944,
        "vocab_size": 102400,
        "kv_lora_rank": 512,
        "qk_nope_head_dim": 128,
        "qk_rope_head_dim": 64,
        "v_head_dim": 128,
        "n_routed_experts": 64,
        "n_shared_experts": 2,
        "num_experts_per_tok": 6,
        "moe_intermediate_size": 1408,
        "first_k_dense_replace": 1,
    },
    # candle: qwen3_5.rs — the hybrid Gated-DeltaNet stack, three linear layers
    # for every full-attention one.
    "Qwen3-Next-80B-A3B": {
        "source": "Qwen/Qwen3-Next-80B-A3B-Instruct",
        "hidden_size": 2048,
        "num_hidden_layers": 48,
        "num_attention_heads": 16,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "intermediate_size": 5120,
        "vocab_size": 151936,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 32,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "moe_intermediate_size": 512,
        "num_experts_per_tok": 10,
    },
}


def deltanet_state_elements(config: dict) -> int:
    r"""The fixed state one Gated-DeltaNet layer carries.

    Two pieces, neither depending on context: the delta-rule matrix
    $S \in \mathbb{R}^{H_v \times d_k \times d_v}$, which is the outer-product
    memory the rule updates, and the short causal convolution's window of
    $k-1$ past activations over the concatenated query, key and value streams.

    The matrix dominates, and it is the whole trade: a *fixed* budget of memory
    whose size is chosen by the architect rather than by how much text has been
    read. C1's hidden state was the same bargain at 1/1000th the size.
    """
    key_dim = config["linear_num_key_heads"] * config["linear_key_head_dim"]
    value_dim = config["linear_num_value_heads"] * config["linear_value_head_dim"]
    recurrent = config["linear_num_value_heads"] * config["linear_key_head_dim"] * config["linear_value_head_dim"]
    convolution = (2 * key_dim + value_dim) * (config["linear_conv_kernel_dim"] - 1)
    return recurrent + convolution


def build_decoder(name: str, config: dict) -> Decoder:
    """Turn a published config into the layer list its decode state follows."""
    layers: list[AttentionLayer] = []
    if "linear_num_key_heads" in config:
        # Hybrid: every full_attention_interval-th layer keeps real attention,
        # the rest are linear. Reach comes from the full layers; capacity from
        # the linear ones.
        state = deltanet_state_elements(config)
        for index in range(config["num_hidden_layers"]):
            if (index + 1) % config["full_attention_interval"] == 0:
                layers.append(
                    AttentionLayer(
                        kind="attention",
                        heads=config["num_attention_heads"],
                        kv_heads=config["num_key_value_heads"],
                        head_dim=config["head_dim"],
                    )
                )
            else:
                layers.append(AttentionLayer(kind="linear", state_elements=state))
    elif "kv_lora_rank" in config:
        layers = [
            AttentionLayer(
                kind="attention",
                heads=config["num_attention_heads"],
                kv_heads=config["num_attention_heads"],
                head_dim=config["v_head_dim"],
                kv_lora_rank=config["kv_lora_rank"],
                rope_head_dim=config["qk_rope_head_dim"],
            )
            for _ in range(config["num_hidden_layers"])
        ]
    else:
        layers = list(
            uniform_attention(
                layers=config["num_hidden_layers"],
                heads=config["num_attention_heads"],
                kv_heads=config["num_key_value_heads"],
                head_dim=config.get("head_dim", config["hidden_size"] // config["num_attention_heads"]),
                window=config.get("sliding_window"),
            )
        )
    return Decoder(name=name, layers=tuple(layers))


@experiment(
    "c5_modern_decoders",
    defaults={
        # C3's own transformer, so the comparison starts from a model this repo
        # actually trained rather than from a hypothetical baseline.
        "c3_width": 192,
        "c3_heads": 4,
        "c3_depth": 4,
        "c3_block_size": 64,
        "contexts": [1024, 4096, 16384, 65536, 262144],
        "headline_context": 65536,
        "dtype_bytes": 2,
    },
    description="Decode-state and parameter accounting for GQA, sliding-window, MLA and delta-rule decoders against C3.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Modern decoder architectures read from source — Qwen3.5, Gemma 4 and DeepSeek-V2 against C3"""
    summary: dict[str, float] = {}

    decoders: dict[str, Decoder] = {}
    # C3's transformer: plain multi-head attention, no GQA, no window.
    c3_head_dim = exp["c3_width"] // exp["c3_heads"]
    decoders["C3 (this repo)"] = Decoder(
        name="C3 (this repo)",
        layers=uniform_attention(
            layers=exp["c3_depth"], heads=exp["c3_heads"], kv_heads=exp["c3_heads"], head_dim=c3_head_dim
        ),
        dtype_bytes=exp["dtype_bytes"],
    )
    for name, config in CONFIGS.items():
        decoders[name] = Decoder(name=name, layers=build_decoder(name, config).layers, dtype_bytes=exp["dtype_bytes"])

    for name, decoder in decoders.items():
        composition = decoder.composition()
        print(f"[c5] {name:<22} {len(decoder.layers):>3} layers {composition}")
        for kind, count in composition.items():
            summary[f"layers/{name}/{kind}"] = float(count)

    # -- the quantity they are all competing on ------------------------------
    print("[c5] decode state (MiB) at context:  " + "  ".join(f"{t:>8}" for t in exp["contexts"]))
    curves: dict[str, list[float]] = {}
    for name, decoder in decoders.items():
        row = []
        for tokens in exp["contexts"]:
            megabytes = decoder.state_bytes_at(tokens) / 2**20
            summary[f"state_mib/{name}/{tokens}"] = megabytes
            row.append(megabytes)
        curves[name] = row
        print(f"[c5] {name:<22}              " + "  ".join(f"{value:8.1f}" for value in row))

    headline = exp["headline_context"]
    reference = decoders["Qwen3-8B"].state_bytes_at(headline)
    for name, decoder in decoders.items():
        if name == "C3 (this repo)":
            continue
        summary[f"state_vs_gqa/{name}"] = decoder.state_bytes_at(headline) / reference

    # -- each mechanism, isolated against its own baseline --------------------
    # The point of an ablation: change one thing about a real config and price
    # the change, rather than compare models that differ in ten ways.
    qwen = CONFIGS["Qwen3-8B"]
    as_multi_head = Decoder(
        name="Qwen3-8B without GQA",
        layers=uniform_attention(
            layers=qwen["num_hidden_layers"],
            heads=qwen["num_attention_heads"],
            kv_heads=qwen["num_attention_heads"],
            head_dim=qwen["head_dim"],
        ),
        dtype_bytes=exp["dtype_bytes"],
    )
    summary["gqa/cache_ratio"] = as_multi_head.state_bytes_at(headline) / decoders["Qwen3-8B"].state_bytes_at(headline)
    summary["gqa/heads_ratio"] = qwen["num_attention_heads"] / qwen["num_key_value_heads"]
    print(
        f"[c5] GQA: {qwen['num_attention_heads']} query heads over {qwen['num_key_value_heads']} kv heads shrinks "
        f"the cache {summary['gqa/cache_ratio']:.0f}x at unchanged query capacity"
    )

    mistral = CONFIGS["Mistral-7B"]
    without_window = Decoder(
        name="Mistral-7B without the window",
        layers=uniform_attention(
            layers=mistral["num_hidden_layers"],
            heads=mistral["num_attention_heads"],
            kv_heads=mistral["num_key_value_heads"],
            head_dim=mistral["head_dim"],
        ),
        dtype_bytes=exp["dtype_bytes"],
    )
    summary["sliding/cache_ratio"] = without_window.state_bytes_at(headline) / decoders["Mistral-7B"].state_bytes_at(
        headline
    )
    # Reach is recovered through depth: each layer sees `window` back, so a
    # stack of L sliding layers has a receptive field of L * window.
    summary["sliding/receptive_field"] = float(mistral["sliding_window"] * mistral["num_hidden_layers"])
    print(
        f"[c5] sliding window {mistral['sliding_window']} caps the cache "
        f"{summary['sliding/cache_ratio']:.1f}x below full attention at {headline} tokens, and "
        f"{mistral['num_hidden_layers']} layers still reach {summary['sliding/receptive_field']:,.0f} tokens through depth"
    )

    deepseek = CONFIGS["DeepSeek-V2-Lite"]
    as_plain_mha = Decoder(
        name="DeepSeek-V2-Lite without MLA",
        layers=uniform_attention(
            layers=deepseek["num_hidden_layers"],
            heads=deepseek["num_attention_heads"],
            kv_heads=deepseek["num_attention_heads"],
            head_dim=deepseek["qk_nope_head_dim"] + deepseek["qk_rope_head_dim"],
        ),
        dtype_bytes=exp["dtype_bytes"],
    )
    summary["mla/cache_ratio"] = as_plain_mha.state_bytes_at(headline) / decoders["DeepSeek-V2-Lite"].state_bytes_at(
        headline
    )
    summary["mla/latent_rank"] = float(deepseek["kv_lora_rank"])
    print(
        f"[c5] MLA: one {deepseek['kv_lora_rank']}-wide latent plus a {deepseek['qk_rope_head_dim']}-wide rope key "
        f"replaces {deepseek['num_attention_heads']} heads of K and V — {summary['mla/cache_ratio']:.1f}x smaller"
    )

    # -- the C1 callback: constant state, at a price -------------------------
    hybrid = decoders["Qwen3-Next-80B-A3B"]
    qwen_next = CONFIGS["Qwen3-Next-80B-A3B"]
    linear_state = deltanet_state_elements(qwen_next) * exp["dtype_bytes"]
    summary["deltanet/state_bytes_per_layer"] = float(linear_state)
    summary["deltanet/state_mib_total"] = (
        linear_state * sum(1 for layer in hybrid.layers if layer.kind == "linear") / 2**20
    )
    # Isolated the same way as the others: the same stack with every linear
    # layer replaced by the full-attention layer it alternates with. Comparing
    # the hybrid to Qwen3-8B instead would credit the delta rule with savings
    # that are really its much more aggressive GQA (2 kv heads against 8).
    all_attention = Decoder(
        name="Qwen3-Next with no linear layers",
        layers=uniform_attention(
            layers=qwen_next["num_hidden_layers"],
            heads=qwen_next["num_attention_heads"],
            kv_heads=qwen_next["num_key_value_heads"],
            head_dim=qwen_next["head_dim"],
        ),
        dtype_bytes=exp["dtype_bytes"],
    )
    summary["deltanet/cache_ratio"] = all_attention.state_bytes_at(headline) / hybrid.state_bytes_at(headline)
    own_crossover = archspec.crossover_tokens(hybrid, all_attention)
    summary["deltanet/crossover_vs_own_attention"] = float(own_crossover) if own_crossover else -1.0
    crossover = archspec.crossover_tokens(hybrid, decoders["Qwen3-8B"])
    summary["deltanet/crossover_vs_gqa"] = float(crossover) if crossover else -1.0
    print(
        f"[c5] delta-rule state is {linear_state / 2**20:.2f} MiB per linear layer and constant in context; "
        f"the hybrid carries {summary['deltanet/state_mib_total']:.0f} MiB before reading anything"
    )
    print(
        f"[c5] against its own all-attention twin the delta rule is worth "
        f"{summary['deltanet/cache_ratio']:.1f}x at {headline:,} tokens, paying for its fixed state "
        f"from {own_crossover:,} tokens onward"
        if own_crossover
        else "[c5] delta rule never pays back its fixed state"
    )
    if crossover:
        print(f"[c5] (against Qwen3-8B it crosses at {crossover:,} tokens, but that comparison also credits its GQA)")

    # -- parameters held against parameters run ------------------------------
    for name in ("DeepSeek-V2-Lite", "Qwen3-Next-80B-A3B"):
        config = CONFIGS[name]
        stack = Transformer(
            name=name,
            layers=config["num_hidden_layers"],
            hidden=config["hidden_size"],
            intermediate=config["intermediate_size"],
            heads=config["num_attention_heads"],
            kv_heads=config.get("num_key_value_heads"),
            head_dim=config.get("head_dim"),
            vocab=config["vocab_size"],
            routed_experts=config.get("n_routed_experts", 0),
            shared_experts=config.get("n_shared_experts", 0),
            experts_per_token=config.get("num_experts_per_tok", 0),
            moe_intermediate=config.get("moe_intermediate_size", 0),
            dense_layers=config.get("first_k_dense_replace", 0),
        )
        if not stack.routed_experts:
            continue
        held = stack.parameters()["total"]
        active = stack.parameters(active=True)["total"]
        summary[f"params/{name}/held"] = float(held)
        summary[f"params/{name}/active"] = float(active)
        summary[f"params/{name}/sparsity"] = held / active
        print(f"[c5] {name:<22} holds {held / 1e9:5.1f}B, runs {active / 1e9:4.2f}B per token ({held / active:.1f}x)")

    # -- does the MoE accounting match what the authors report? --------------
    # DeepSeek-V2-Lite is published at 15.7B total and 2.4B activated, so the
    # count here can be checked rather than trusted. Activated is expected to come
    # out high: the published figure counts transformer parameters, while this
    # one includes the embedding and output head, which every token does use.
    published = {"total": 15.7e9, "active": 2.4e9}
    if "params/DeepSeek-V2-Lite/held" in summary:
        summary["validation/deepseek_total_error"] = (
            abs(summary["params/DeepSeek-V2-Lite/held"] - published["total"]) / published["total"]
        )
        embedding_free_active = summary["params/DeepSeek-V2-Lite/active"] - 2 * (
            CONFIGS["DeepSeek-V2-Lite"]["vocab_size"] * CONFIGS["DeepSeek-V2-Lite"]["hidden_size"]
        )
        summary["validation/deepseek_active_error"] = (
            abs(embedding_free_active - published["active"]) / published["active"]
        )
        print(
            f"[c5] check against DeepSeek's published figures: total "
            f"{summary['params/DeepSeek-V2-Lite/held'] / 1e9:.1f}B vs 15.7B "
            f"({summary['validation/deepseek_total_error']:.1%} off), active excluding embeddings "
            f"{embedding_free_active / 1e9:.2f}B vs 2.4B ({summary['validation/deepseek_active_error']:.1%} off)"
        )

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "decode_state",
        plotting.plot_curves(
            curves,
            x=exp["contexts"],
            title="Decode state vs. context — the cost every variant since C3 attacks",
            xlabel="context (tokens)",
            ylabel="state (MiB)",
            marker="o",
            logx=True,
            logy=True,
        ),
    )
    exp.log_figure(
        "mechanism_savings",
        plotting.plot_comparison(
            {
                "GQA\n(32:8 heads)": summary["gqa/cache_ratio"],
                "sliding window\n(4096)": summary["sliding/cache_ratio"],
                "MLA\n(rank 512)": summary["mla/cache_ratio"],
            },
            title=f"Cache shrink at {headline:,} tokens, each against its own config without the mechanism",
            ylabel="times smaller",
        ),
    )
    per_token = {
        name: [decoder.state_bytes_per_token(tokens) / 1024 for tokens in exp["contexts"]]
        for name, decoder in decoders.items()
    }
    exp.log_figure(
        "state_per_token",
        plotting.plot_curves(
            per_token,
            x=exp["contexts"],
            title="State per token — flat for a cache, falling for a fixed recurrent state",
            xlabel="context (tokens)",
            ylabel="KiB per token",
            marker="o",
            logx=True,
            logy=True,
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
