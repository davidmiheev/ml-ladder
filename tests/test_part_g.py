"""Part G: the compute accounting and the curve fitting, checked rather than trusted."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from mlladder import vision

torch = pytest.importorskip("torch")


def test_flop_accounting_reproduces_the_2n_rule():
    """The $d^2$ terms are exactly $2N$ per token, which is where $6ND$ comes from.

    Not a tautology: the rung derives the dense term from the five matrix
    multiplications in a block, and the parameter count comes from PyTorch. If
    the two agree to within the bias and LayerNorm overhead, the derivation in
    the docstring is the one the code implements.
    """
    from parts.c_sequence_attention.c3_transformer_from_scratch import make_transformer
    from parts.g_scaling_systems.g1_scaling_law_fit import non_embedding_parameters, transformer_flops

    CharTransformer, _ = make_transformer()
    for width, depth in ((64, 2), (96, 2), (48, 8)):
        model = CharTransformer(65, width=width, heads=4, depth=depth, block_size=64, positional="learned")
        parameters = non_embedding_parameters(model)
        dense = transformer_flops(width=width, depth=depth, block_size=64, vocab_size=65)["dense"]
        # Biases and LayerNorms are parameters that carry no matmul, so the
        # measured ratio sits just under 2.
        ratio = dense / parameters
        assert 1.9 < ratio <= 2.0, (width, depth, ratio)


def test_the_attention_term_the_rule_drops_grows_with_context():
    """$4L^2d$ against $24Ld^2$: the share is linear in $L/d$.

    At the rung's own shape it is 14%; quadrupling the context roughly
    quadruples it. This is the term the 6ND rule of thumb silently omits, and
    the reason the omission is safe for GPT-3-shaped models and not for these.
    """
    from parts.g_scaling_systems.g1_scaling_law_fit import transformer_flops

    shares = [
        transformer_flops(width=64, depth=2, block_size=block, vocab_size=65)["attention_share"]
        for block in (64, 256, 1024)
    ]
    assert shares == sorted(shares)
    assert 0.10 < shares[0] < 0.20
    assert shares[-1] > 0.6
    # Wider models hide it: at the same context, doubling the width halves it.
    narrow = transformer_flops(width=64, depth=2, block_size=256, vocab_size=65)["attention_share"]
    wide = transformer_flops(width=128, depth=2, block_size=256, vocab_size=65)["attention_share"]
    assert wide < narrow


def test_non_embedding_parameters_drops_the_positional_table_too():
    """The one that is easy to forget, and scales the wrong way.

    Token embedding and head are obvious. The positional table is not, and it
    grows as $d$ while the body grows as $d^2$, so leaving it in would inflate
    the smallest models most — flattening the fitted exponent.
    """
    from parts.c_sequence_attention.c3_transformer_from_scratch import make_transformer
    from parts.g_scaling_systems.g1_scaling_law_fit import non_embedding_parameters

    CharTransformer, _ = make_transformer()
    learned = CharTransformer(65, width=64, heads=4, depth=2, block_size=64, positional="learned")
    sinusoidal = CharTransformer(65, width=64, heads=4, depth=2, block_size=64, positional="sinusoidal")
    assert non_embedding_parameters(learned) == non_embedding_parameters(sinusoidal)

    big_vocab = CharTransformer(5000, width=64, heads=4, depth=2, block_size=64, positional="learned")
    assert non_embedding_parameters(big_vocab) == non_embedding_parameters(learned)


def test_power_law_fit_recovers_parameters_it_was_given():
    from parts.g_scaling_systems.g1_scaling_law_fit import fit_power_law

    x = np.geomspace(1e4, 1e7, 12)
    y = 1.4 + 90.0 * x ** (-0.25)
    fit = fit_power_law(x, y)
    assert fit["exponent"] == pytest.approx(0.25, abs=0.02)
    assert fit["irreducible"] == pytest.approx(1.4, abs=0.05)
    assert fit["r2"] > 0.999


def test_the_floor_is_the_parameter_a_short_sweep_cannot_pin_down():
    """Why the fit reports a band and not just a number.

    Over three decades of $x$ the floor is identified; over half a decade — which
    is what a sweep you can afford looks like — a wide range of floors fits
    essentially as well, and each one comes with its own exponent. Quoting the
    argmin alone would be quoting noise.
    """
    from parts.g_scaling_systems.g1_scaling_law_fit import fit_power_law

    def band(lo, hi, n=6):
        # With a little noise, as any real sweep has: three seeds of a training
        # run do not agree to five decimals.
        rng = np.random.default_rng(0)
        x = np.geomspace(lo, hi, n)
        y = 1.4 + 90.0 * x ** (-0.25) + rng.normal(0.0, 0.01, size=n)
        fit = fit_power_law(x, y)
        return fit["irreducible_high"] - fit["irreducible_low"], fit["exponent_high"] - fit["exponent_low"]

    wide_floor, wide_exponent = band(1e4, 1e7)
    narrow_floor, narrow_exponent = band(1e5, 3e5)
    assert narrow_floor > 3 * wide_floor, (narrow_floor, wide_floor)
    assert narrow_exponent > wide_exponent


def test_cosine_schedule_warms_up_then_decays_to_its_floor():
    from parts.g_scaling_systems.g1_scaling_law_fit import cosine_schedule

    total = 1000
    values = [cosine_schedule(step, total) for step in range(total)]
    warmup = int(total * 0.05)
    assert values[0] < 0.1
    assert values[warmup - 1] == pytest.approx(1.0)
    # Monotone decay after the peak, ending just above the floor.
    after = values[warmup:]
    assert all(b <= a + 1e-12 for a, b in itertools.pairwise(after))
    assert 0.1 <= values[-1] < 0.12


def test_schedule_is_matched_to_the_budget_not_shared():
    """A short run and a long run must trace the *same* shape, not a prefix of one.

    If the schedule were shared, the short runs would stop at a high learning
    rate and the L(D) curve would be a picture of the schedule rather than of
    the data.
    """
    from parts.g_scaling_systems.g1_scaling_law_fit import cosine_schedule

    short = [cosine_schedule(step, 100) for step in range(100)]
    long = [cosine_schedule(step, 1000) for step in range(1000)]
    assert short[-1] == pytest.approx(long[-1], abs=1e-3)
    assert short[len(short) // 2] == pytest.approx(long[len(long) // 2], abs=0.02)


def test_g1_runs():
    from parts.g_scaling_systems.g1_scaling_law_fit import main

    vision.require_torch()
    assert (
        main(
            [
                "--widths",
                "32",
                "48",
                "--budget-tokens",
                "20000",
                "--data-budgets",
                "10000",
                "20000",
                "--isoflop-budgets",
                "2e10",
                "6e10",
                "--isoflop-widths",
                "32",
                "48",
                "--eval-batches",
                "2",
                "--shape-pairs",
                "48,2",
                "32,4",
            ]
        )
        == 0
    )
