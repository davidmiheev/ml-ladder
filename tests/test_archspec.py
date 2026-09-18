"""Architecture accounting, checked against cases with known answers.

These formulas are the entire content of C5 and H6 — nothing is trained, so a
wrong constant produces a confident wrong plot rather than a failure. The tests
below are hand-computable cases plus the identities the mechanisms are supposed
to satisfy.
"""

from __future__ import annotations

import pytest

from mlladder.archspec import AttentionLayer, Decoder, Transformer, crossover_tokens, uniform_attention

# -- parameters --------------------------------------------------------------


def test_block_parameters_by_hand():
    stack = Transformer(name="toy", layers=1, hidden=8, intermediate=16, heads=2, gated=False)
    assert stack.attention_parameters() == 4 * 8 * 8  # q, k, v, o
    assert stack.mlp_parameters() == 2 * 8 * 16
    assert stack.block_parameters() == 256 + 256 + 16


def test_moe_holds_more_than_it_runs():
    stack = Transformer(
        name="moe",
        layers=3,
        hidden=64,
        intermediate=256,
        heads=8,
        routed_experts=16,
        shared_experts=2,
        experts_per_token=4,
        moe_intermediate=32,
        dense_layers=1,
    )
    held, active = stack.parameters()["total"], stack.parameters(active=True)["total"]
    assert held - active == 2 * (16 - 4) * (3 * 64 * 32)


# -- decode state ------------------------------------------------------------


def test_multi_head_cache_is_linear_in_context():
    layer = AttentionLayer(kind="attention", heads=8, kv_heads=8, head_dim=64)
    assert layer.state_elements_at(100) == 2 * 100 * 8 * 64
    assert layer.state_elements_at(200) == 2 * layer.state_elements_at(100)


def test_grouped_query_divides_the_cache_by_the_head_ratio():
    shared = {"kind": "attention", "heads": 32, "head_dim": 128}
    multi = AttentionLayer(**shared, kv_heads=32)
    grouped = AttentionLayer(**shared, kv_heads=8)
    assert multi.state_elements_at(1000) == 4 * grouped.state_elements_at(1000)


def test_sliding_window_caps_the_cache():
    layer = AttentionLayer(kind="attention", heads=8, kv_heads=8, head_dim=64, window=128)
    assert layer.state_elements_at(64) == 2 * 64 * 8 * 64  # below the window: linear
    assert layer.state_elements_at(1_000_000) == layer.state_elements_at(128), "past the window it must be constant"


def test_mla_caches_one_latent_per_token_not_per_head():
    """Independent of head count, which is the whole point of the latent."""
    few = AttentionLayer(kind="attention", heads=8, kv_heads=8, head_dim=128, kv_lora_rank=512, rope_head_dim=64)
    many = AttentionLayer(kind="attention", heads=128, kv_heads=128, head_dim=128, kv_lora_rank=512, rope_head_dim=64)
    assert few.state_elements_at(50) == many.state_elements_at(50) == 50 * (512 + 64)


def test_linear_attention_state_is_constant_in_context():
    layer = AttentionLayer(kind="linear", state_elements=1000)
    assert layer.state_elements_at(1) == layer.state_elements_at(10**6) == 1000


def test_decoder_sums_its_layers_and_reports_composition():
    hybrid = Decoder(
        name="hybrid",
        layers=(
            AttentionLayer(kind="linear", state_elements=100),
            AttentionLayer(kind="attention", heads=4, kv_heads=2, head_dim=8),
            AttentionLayer(kind="attention", heads=4, kv_heads=2, head_dim=8, window=16),
        ),
        dtype_bytes=2,
    )
    expected = 100 + 2 * 10 * 2 * 8 + 2 * 10 * 2 * 8
    assert hybrid.state_bytes_at(10) == 2 * expected
    assert hybrid.composition() == {"linear": 1, "full": 1, "sliding": 1}


def test_state_per_token_falls_for_a_constant_state_and_is_flat_for_a_cache():
    """The shape that distinguishes the two designs, not just their size."""
    cache = Decoder(name="cache", layers=uniform_attention(layers=2, heads=4, kv_heads=4, head_dim=16))
    recurrent = Decoder(name="recurrent", layers=(AttentionLayer(kind="linear", state_elements=10_000),) * 2)
    assert cache.state_bytes_per_token(100) == pytest.approx(cache.state_bytes_per_token(10_000))
    assert recurrent.state_bytes_per_token(10_000) < recurrent.state_bytes_per_token(100) / 50


# -- the crossover -----------------------------------------------------------


def test_crossover_finds_where_a_fixed_state_starts_winning():
    cache = Decoder(name="cache", layers=uniform_attention(layers=1, heads=1, kv_heads=1, head_dim=1))
    # cache holds 2*T elements; the fixed state holds 200, so it wins past T=100.
    recurrent = Decoder(name="recurrent", layers=(AttentionLayer(kind="linear", state_elements=200),))
    assert crossover_tokens(recurrent, cache) == 101


def test_crossover_is_none_when_it_never_pays_back():
    cache = Decoder(name="cache", layers=uniform_attention(layers=1, heads=1, kv_heads=1, head_dim=1))
    huge = Decoder(name="huge", layers=(AttentionLayer(kind="linear", state_elements=10**9),))
    assert crossover_tokens(huge, cache, limit=1000) is None


def test_crossover_is_one_when_it_wins_immediately():
    big_cache = Decoder(name="cache", layers=uniform_attention(layers=8, heads=32, kv_heads=32, head_dim=128))
    tiny = Decoder(name="tiny", layers=(AttentionLayer(kind="linear", state_elements=8),))
    assert crossover_tokens(tiny, big_cache) == 1


# -- architecture diagrams ---------------------------------------------------


def test_architecture_diagram_renders_every_block():
    """The diagram is generated, not checked in, so it must not silently drop rows."""
    from mlladder.plotting import plot_architecture

    figure = plot_architecture(
        {"A": ["input", ("middle", "annotated"), "output"], "B": ["input", "output"]},
        highlight={"A": [1]},
        links=[(0, 1, "replaces")],
        title="test",
    )
    texts = {text.get_text() for axis in figure.axes for text in axis.texts}
    for label in ("input", "middle", "annotated", "output", "replaces"):
        assert label in texts, f"{label} missing from the diagram"
    figure.clf()


def test_architecture_diagram_accepts_uneven_columns():
    from mlladder.plotting import plot_architecture

    figure = plot_architecture({"tall": ["a", "b", "c", "d"], "short": ["a"]})
    assert figure.axes, "a figure with no axes would render blank"
    figure.clf()
