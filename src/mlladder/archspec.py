"""Architecture accounting: what a stack holds, runs, and must remember.

Parts C and H both need to read a published architecture and say what it costs,
so the arithmetic lives here rather than in either rung. Two questions get
asked of every stack:

- **parameters**, split into what is *held* and what is *run* per token, which
  a mixture-of-experts layer makes different numbers;
- **decode state**, the bytes that must be kept per generated token, which is
  what every attention variant since the original transformer has been trying
  to shrink.

Nothing here loads weights. These are closed-form counts from a config, which
makes them checkable — :mod:`parts.h_computer_vision.h6_ocr_vlm_architectures`
validates them against a model it can actually load and count.
"""

from __future__ import annotations

from dataclasses import dataclass, field


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


# -- decode state: the quantity every attention variant attacks ---------------


@dataclass(frozen=True)
class AttentionLayer:
    r"""One layer's contribution to the state a decoder carries per token.

    C3 builds multi-head attention, whose cache grows as $2 L H d_h T$ — linear
    in context, with the full head count as the multiplier. Everything since is
    an attack on one of those factors:

    - **grouped-query** shrinks $H$ to $H_{kv}$ while leaving query capacity alone;
    - **sliding-window** caps $T$ at the window, making the layer's cache constant;
    - **MLA** replaces the per-head keys and values with one low-rank latent plus
      a shared rotary key, so the multiplier stops scaling with heads at all;
    - **linear attention** (a delta rule) keeps a fixed-size recurrent state and
      drops the dependence on $T$ entirely — which is C1's recurrence, returned.
    """

    kind: str  # "attention" | "linear"
    heads: int = 0
    kv_heads: int = 0
    head_dim: int = 0
    window: int | None = None
    # MLA: the compressed latent and the decoupled rotary key that DeepSeek
    # caches instead of per-head K and V.
    kv_lora_rank: int = 0
    rope_head_dim: int = 0
    # Linear attention: a recurrent matrix plus a short convolution window.
    state_elements: int = 0

    def state_elements_at(self, tokens: int) -> int:
        """Elements of state this layer holds after ``tokens`` generated tokens."""
        if self.kind == "linear":
            return self.state_elements  # constant in context, by construction
        effective = min(tokens, self.window) if self.window else tokens
        if self.kv_lora_rank:
            # One latent per token, plus the rope key shared across heads.
            return effective * (self.kv_lora_rank + self.rope_head_dim)
        return 2 * effective * self.kv_heads * self.head_dim


@dataclass(frozen=True)
class Decoder:
    """A decoder as a sequence of layers, for state accounting.

    Modern stacks are not homogeneous — Gemma alternates sliding and full
    attention, Qwen3-Next runs three linear layers for every full one — so the
    layer list is explicit rather than a count times a type.
    """

    name: str
    layers: tuple[AttentionLayer, ...]
    dtype_bytes: int = 2

    def state_bytes_at(self, tokens: int) -> int:
        return self.dtype_bytes * sum(layer.state_elements_at(tokens) for layer in self.layers)

    def state_bytes_per_token(self, tokens: int) -> float:
        return self.state_bytes_at(tokens) / max(tokens, 1)

    def composition(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for layer in self.layers:
            key = layer.kind
            if layer.kind == "attention":
                key = "sliding" if layer.window else ("mla" if layer.kv_lora_rank else "full")
            counts[key] = counts.get(key, 0) + 1
        return counts


def uniform_attention(
    *, layers: int, heads: int, kv_heads: int, head_dim: int, window: int | None = None
) -> tuple[AttentionLayer, ...]:
    """``layers`` identical attention layers — the C3-style homogeneous stack."""
    return tuple(
        AttentionLayer(kind="attention", heads=heads, kv_heads=kv_heads, head_dim=head_dim, window=window)
        for _ in range(layers)
    )


def crossover_tokens(a: Decoder, b: Decoder, *, limit: int = 1 << 22) -> int | None:
    """Smallest context at which ``a`` carries less state than ``b``.

    A constant-state decoder starts out *larger* than a cache that grows from
    nothing, so "which is cheaper" has a context length attached to it. Binary
    search over the crossing, or ``None`` when it does not cross below
    ``limit``.
    """
    if a.state_bytes_at(1) < b.state_bytes_at(1):
        return 1
    if a.state_bytes_at(limit) >= b.state_bytes_at(limit):
        return None
    low, high = 1, limit
    while low < high:
        middle = (low + high) // 2
        if a.state_bytes_at(middle) < b.state_bytes_at(middle):
            high = middle
        else:
            low = middle + 1
    return low
