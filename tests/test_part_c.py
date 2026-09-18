"""Part C — sequence models and attention.

The load-bearing test is the causal-mask check: a decoder that leaks one
position from the future still trains beautifully and is worthless, and nothing
else in the rung would notice.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlladder import text

torch = pytest.importorskip("torch", reason="uv sync --extra torch")


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("MLLADDER_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MLLADDER_TRACKER", "none")


# -- the text plumbing ------------------------------------------------------


def test_char_tokenizer_roundtrips():
    tokenizer = text.CharTokenizer.from_text("hello world")
    assert tokenizer.vocab_size == len(set("hello world"))
    assert tokenizer.decode(tokenizer.encode("hello")) == "hello"


def test_lm_batch_targets_are_inputs_shifted_by_one():
    ids = np.arange(200, dtype=np.int64)
    inputs, targets = text.lm_batch(ids, batch_size=4, block_size=16, rng=np.random.default_rng(0))
    assert inputs.shape == targets.shape == (4, 16)
    assert np.array_equal(inputs[:, 1:], targets[:, :-1])
    assert np.array_equal(targets, inputs + 1)  # this corpus is the integers


def test_corpus_split_is_contiguous():
    ids = np.arange(100, dtype=np.int64)
    train, validation = text.split_corpus(ids, val_fraction=0.2)
    assert len(train) == 80 and len(validation) == 20
    assert train[-1] + 1 == validation[0], "a shuffled split would leak the continuation"


@pytest.mark.parametrize("context_words", [0, 6, 20])
def test_date_pairs_carry_the_answer_and_grow_with_context(context_words):
    pairs = text.make_date_pairs(40, context_words=context_words, rng=0)
    for pair in pairs:
        year, month, day = pair.target.split("-")
        assert len(pair.target) == 10 and year.isdigit() and 1 <= int(month) <= 12 and 1 <= int(day) <= 28
    lengths = [len(pair.source) for pair in pairs]
    assert min(lengths) >= 8 + context_words  # filler really is in there


def test_recall_task_labels_come_from_the_first_symbol_only():
    sequences, labels = text.make_recall_task(64, length=12, n_symbols=8, rng=0)
    assert sequences.shape == (64, 12)
    assert np.array_equal(sequences[:, 0] - 1, labels)
    assert (sequences[:, 1:] == 0).all(), "nothing after position 0 may carry signal"


# -- C1: the cells ----------------------------------------------------------


def test_cells_produce_the_right_shapes_and_lstm_starts_by_remembering():
    from parts.c_sequence_attention.c1_char_rnn import make_cells

    VanillaRNNCell, LSTMCell, CharLanguageModel = make_cells()

    rnn = VanillaRNNCell(8, 16)
    state = rnn.initial_state(4, torch.device("cpu"))
    out, state = rnn(torch.randn(4, 8), state)
    assert out.shape == (4, 16) and len(state) == 1

    lstm = LSTMCell(8, 16)
    h, c = lstm.initial_state(4, torch.device("cpu"))
    assert h.shape == c.shape == (4, 16)
    # Forget-gate bias initialized to 1 -- the standard "remember by default".
    assert torch.allclose(lstm.gates.bias[16:32], torch.ones(16))

    model = CharLanguageModel(20, 16, cell="lstm")
    logits, _ = model(torch.randint(0, 20, (3, 7)))
    assert logits.shape == (3, 7, 20)


def test_gradient_reach_decays_further_for_the_vanilla_cell():
    """The rung's central claim, at a size small enough for a unit test."""
    from parts.c_sequence_attention.c1_char_rnn import gradient_reach, make_cells

    _, _, CharLanguageModel = make_cells()
    ids = torch.randint(0, 12, (8, 48))

    torch.manual_seed(0)
    rnn = gradient_reach(CharLanguageModel(12, 32, cell="rnn"), ids, device=torch.device("cpu"))
    torch.manual_seed(0)
    lstm = gradient_reach(CharLanguageModel(12, 32, cell="lstm"), ids, device=torch.device("cpu"))

    assert len(rnn) == len(lstm) == 48
    assert rnn[-1] > 0 and lstm[-1] > 0, "the final position always gets gradient"
    # Both decay backwards; the vanilla cell decays much harder.
    assert rnn[0] / rnn[-1] < lstm[0] / lstm[-1]


# -- C2: attention ----------------------------------------------------------


def test_attention_weights_are_a_distribution_that_ignores_padding():
    from parts.c_sequence_attention.c2_seq2seq_attention import make_models

    Seq2Seq = make_models()
    model = Seq2Seq(source_vocab=10, target_vocab=6, embedding_size=8, hidden_size=8, attention=True)

    source = torch.tensor([[3, 4, 5, 0, 0], [6, 7, 0, 0, 0]])
    logits, weights = model(source, torch.tensor([[0, 1, 2], [0, 1, 2]]))
    assert logits.shape == (2, 3, 6)
    assert torch.allclose(weights.sum(dim=-1), torch.ones(2, 3), atol=1e-5)
    assert weights[0, :, 3:].max() == 0, "pad positions must receive no attention mass"
    assert weights[1, :, 2:].max() == 0


def test_bottleneck_decoder_returns_no_weights():
    from parts.c_sequence_attention.c2_seq2seq_attention import make_models

    Seq2Seq = make_models()
    model = Seq2Seq(source_vocab=10, target_vocab=6, embedding_size=8, hidden_size=8, attention=False)
    logits, weights = model(torch.tensor([[3, 4, 5, 0]]), torch.tensor([[0, 1]]))
    assert logits.shape == (1, 2, 6) and weights is None


def test_alignment_score_rewards_peaks_on_matching_characters():
    from parts.c_sequence_attention.c2_seq2seq_attention import alignment_score

    # Output "12" with peaks on the input positions holding '1' and '2'.
    weights = np.zeros((1, 2, 4))
    weights[0, 0, 2] = 1.0  # position 2 of "ab12" holds '1'
    weights[0, 1, 3] = 1.0
    assert alignment_score(weights, ["ab12"], ["12"]) == pytest.approx(1.0)

    weights[0, 1, 0] = 2.0  # now peaks on 'a' instead
    assert alignment_score(weights, ["ab12"], ["12"]) == pytest.approx(0.5)


# -- C3: the transformer ----------------------------------------------------


def test_the_causal_mask_does_not_leak_the_future():
    """Change the last token; every earlier position's logits must not move."""
    from parts.c_sequence_attention.c3_transformer_from_scratch import make_transformer

    CharTransformer, _ = make_transformer()
    torch.manual_seed(0)
    model = CharTransformer(16, width=32, heads=4, depth=2, block_size=12).eval()

    ids = torch.randint(0, 16, (2, 12))
    tampered = ids.clone()
    tampered[:, -1] = (tampered[:, -1] + 1) % 16

    with torch.no_grad():
        original, weights = model(ids, return_weights=True)
        changed, _ = model(tampered)

    assert torch.equal(original[:, :-1], changed[:, :-1])
    assert weights.triu(1).abs().max() == 0, "no attention mass above the diagonal"
    assert torch.allclose(weights.sum(dim=-1), torch.ones_like(weights.sum(dim=-1)), atol=1e-5)


def test_sinusoidal_encoding_is_bounded_and_position_specific():
    from parts.c_sequence_attention.c3_transformer_from_scratch import make_transformer

    _, sinusoidal_encoding = make_transformer()
    table = sinusoidal_encoding(64, 32)
    assert table.shape == (64, 32)
    assert table.abs().max() <= 1.0
    # Neighbouring positions are similar, distant ones are not: that ordering
    # is the entire signal the model can use.
    normalized = torch.nn.functional.normalize(table, dim=1)
    similarity = normalized @ normalized.T
    assert similarity[0, 1] > similarity[0, 32]


def test_transformer_rejects_bad_configurations():
    from parts.c_sequence_attention.c3_transformer_from_scratch import make_transformer

    CharTransformer, _ = make_transformer()
    with pytest.raises(ValueError, match="must divide"):
        CharTransformer(16, width=30, heads=4, depth=1, block_size=8)
    with pytest.raises(ValueError, match="unknown positional"):
        CharTransformer(16, width=32, heads=4, depth=1, block_size=8, positional="nope")


# -- C4: the ViT ------------------------------------------------------------


def test_task_subset_relabels_to_a_dense_range():
    from parts.c_sequence_attention.c4_vit_finetune import TASK_CLASSES, subset_task

    images = np.zeros((10, 3, 4, 4), dtype=np.float32)
    labels = np.array([0, 1, 2, 3, 8, 9, 0, 9, 5, 1], dtype=np.int64)
    kept_images, kept_labels = subset_task(images, labels)
    assert len(kept_images) == len(kept_labels) == 7  # 2, 3 and 5 are dropped
    assert set(kept_labels.tolist()) <= set(range(len(TASK_CLASSES)))


def test_frozen_backbone_leaves_only_the_head_trainable():
    from parts.c_sequence_attention.c4_vit_finetune import build_vit, resize_batch

    model = build_vit(pretrained=False, num_classes=4, frozen=True)
    trainable = [name for name, parameter in model.named_parameters() if parameter.requires_grad]
    assert trainable and all(name.startswith("head") for name in trainable)

    resized = resize_batch(torch.randn(2, 3, 32, 32), 224)
    assert resized.shape == (2, 3, 224, 224)


# -- the rungs end to end ---------------------------------------------------


def test_c1_runs():
    from parts.c_sequence_attention.c1_char_rnn import main

    assert (
        main(
            [
                "--corpus-chars",
                "6000",
                "--steps",
                "3",
                "--eval-every",
                "3",
                "--block-size",
                "16",
                "--batch-size",
                "8",
                "--hidden",
                "16",
                "--eval-batches",
                "1",
                "--sample-chars",
                "20",
                "--recall-lengths",
                "4",
                "--recall-steps",
                "2",
            ]
        )
        == 0
    )


def test_c2_runs():
    from parts.c_sequence_attention.c2_seq2seq_attention import main

    assert (
        main(
            [
                "--train-samples",
                "128",
                "--test-samples",
                "64",
                "--epochs",
                "1",
                "--hidden",
                "16",
                "--embedding",
                "8",
                "--context-words",
                "0",
            ]
        )
        == 0
    )


def test_c3_runs():
    from parts.c_sequence_attention.c3_transformer_from_scratch import main

    assert (
        main(
            [
                "--corpus-chars",
                "6000",
                "--steps",
                "3",
                "--eval-every",
                "3",
                "--block-size",
                "16",
                "--batch-size",
                "8",
                "--width",
                "32",
                "--depth",
                "1",
                "--eval-batches",
                "1",
                "--sample-chars",
                "20",
                "--encodings",
                "learned",
                "--matched-width",
                "32",
                "--matched-depth",
                "1",
            ]
        )
        == 0
    )


def test_c4_runs():
    from parts.c_sequence_attention.c4_vit_finetune import main

    assert main(["--train-samples", "32", "--test-samples", "32", "--epochs", "1", "--arms", "probe"]) == 0


# -- C5: modern decoder accounting -------------------------------------------


def test_c5_configs_carry_their_provenance():
    """Every recorded config names where it came from, or it is unverifiable."""
    from parts.c_sequence_attention.c5_modern_decoders import CONFIGS

    for name, config in CONFIGS.items():
        assert "source" in config, f"{name} has no source repo recorded"
        assert "/" in config["source"], f"{name}'s source is not a hub id"


def test_build_decoder_picks_the_right_mechanism_per_config():
    from parts.c_sequence_attention.c5_modern_decoders import CONFIGS, build_decoder

    assert build_decoder("Qwen3-8B", CONFIGS["Qwen3-8B"]).composition() == {"full": 36}
    assert build_decoder("Mistral-7B", CONFIGS["Mistral-7B"]).composition() == {"sliding": 32}
    assert build_decoder("DeepSeek-V2-Lite", CONFIGS["DeepSeek-V2-Lite"]).composition() == {"mla": 27}
    # Three linear layers for every full one, at interval 4 over 48 layers.
    hybrid = build_decoder("Qwen3-Next-80B-A3B", CONFIGS["Qwen3-Next-80B-A3B"]).composition()
    assert hybrid == {"linear": 36, "full": 12}


def test_deltanet_state_is_dominated_by_the_recurrent_matrix():
    """The conv window is a rounding error next to the outer-product memory."""
    from parts.c_sequence_attention.c5_modern_decoders import CONFIGS, deltanet_state_elements

    config = CONFIGS["Qwen3-Next-80B-A3B"]
    matrix = config["linear_num_value_heads"] * config["linear_key_head_dim"] * config["linear_value_head_dim"]
    total = deltanet_state_elements(config)
    assert total > matrix
    assert matrix / total > 0.9


def test_sliding_window_stack_stops_growing_but_full_attention_does_not():
    from parts.c_sequence_attention.c5_modern_decoders import CONFIGS, build_decoder

    mistral = build_decoder("Mistral-7B", CONFIGS["Mistral-7B"])
    qwen = build_decoder("Qwen3-8B", CONFIGS["Qwen3-8B"])
    assert mistral.state_bytes_at(65536) == mistral.state_bytes_at(262144)
    assert qwen.state_bytes_at(262144) == 4 * qwen.state_bytes_at(65536)


def test_moe_accounting_reproduces_deepseeks_published_size():
    """15.7B total is public, so the formulas can be checked rather than trusted."""
    from parts.c_sequence_attention.c5_modern_decoders import CONFIGS

    from mlladder.archspec import Transformer

    config = CONFIGS["DeepSeek-V2-Lite"]
    stack = Transformer(
        name="DeepSeek-V2-Lite",
        layers=config["num_hidden_layers"],
        hidden=config["hidden_size"],
        intermediate=config["intermediate_size"],
        heads=config["num_attention_heads"],
        kv_heads=config["num_key_value_heads"],
        vocab=config["vocab_size"],
        routed_experts=config["n_routed_experts"],
        shared_experts=config["n_shared_experts"],
        experts_per_token=config["num_experts_per_tok"],
        moe_intermediate=config["moe_intermediate_size"],
        dense_layers=config["first_k_dense_replace"],
    )
    held = stack.parameters()["total"]
    assert abs(held - 15.7e9) / 15.7e9 < 0.05, f"{held / 1e9:.1f}B against a published 15.7B"


def test_c5_runs():
    from parts.c_sequence_attention.c5_modern_decoders import main

    assert main(["--contexts", "1024", "65536"]) == 0


def test_the_two_qwen_generations_are_one_architecture():
    """C5's method is reading configs; this is where reading configs runs out.

    Qwen3.6-27B and Qwen3.8-27B are a generation apart and share a config, so
    the rung carries them as one entry. If a future edit splits them into two,
    this fails and forces the split to be justified by an actual difference.
    """
    from parts.c_sequence_attention.c5_modern_decoders import CONFIGS, IDENTICAL_BY_CONFIG

    entry = CONFIGS["Qwen3.6-27B / Qwen3.8-27B"]
    assert all(name in entry["source"] for name in IDENTICAL_BY_CONFIG)
    assert entry["num_hidden_layers"] == 64
    assert entry["full_attention_interval"] == 4
    # 48 linear layers to 16 full ones, which is the hybrid ratio the rung prices.
    assert entry["num_hidden_layers"] - entry["num_hidden_layers"] // 4 == 48


def test_the_qwen_27b_pair_builds_a_hybrid_stack():
    from parts.c_sequence_attention.c5_modern_decoders import CONFIGS, build_decoder

    decoder = build_decoder("Qwen3.6-27B / Qwen3.8-27B", CONFIGS["Qwen3.6-27B / Qwen3.8-27B"])
    kinds = [layer.kind for layer in decoder.layers]
    assert len(decoder.layers) == 64
    assert kinds.count("linear") == 48
    assert kinds.count("attention") == 16
