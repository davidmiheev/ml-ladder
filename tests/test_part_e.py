"""Part E — generative models.

The load-bearing tests are on the ELBO's two terms: a KL that is wrong by a
constant still trains, still goes down, and silently moves the beta the sweep
concludes with.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="uv sync --extra torch")


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("MLLADDER_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MLLADDER_TRACKER", "none")


# -- E1: the ELBO ------------------------------------------------------------


def test_kl_is_zero_for_the_prior_itself():
    r"""$\mathcal{N}(0, I)$ against $\mathcal{N}(0, I)$ must cost nothing."""
    from parts.e_generative.e1_vae import kl_to_standard_normal

    mean = torch.zeros(32, 8)
    log_variance = torch.zeros(32, 8)  # log 1 = 0
    assert kl_to_standard_normal(mean, log_variance).item() == pytest.approx(0.0, abs=1e-6)


def test_kl_matches_the_closed_form_on_a_hand_case():
    from parts.e_generative.e1_vae import kl_to_standard_normal

    # One dimension, mu = 2, sigma^2 = 1: KL = -0.5 * (1 + 0 - 4 - 1) = 2.
    mean = torch.full((1, 1), 2.0)
    assert kl_to_standard_normal(mean, torch.zeros(1, 1)).item() == pytest.approx(2.0)


def test_kl_grows_as_the_posterior_leaves_the_prior():
    from parts.e_generative.e1_vae import kl_to_standard_normal

    log_variance = torch.zeros(16, 4)
    near = kl_to_standard_normal(torch.full((16, 4), 0.1), log_variance)
    far = kl_to_standard_normal(torch.full((16, 4), 2.0), log_variance)
    assert near.item() < far.item()


def test_per_dimension_kl_sums_to_the_total():
    """Posterior collapse is read off the per-dimension split, so it must agree."""
    from parts.e_generative.e1_vae import kl_to_standard_normal

    torch.manual_seed(0)
    mean, log_variance = torch.randn(64, 6), torch.randn(64, 6) * 0.3
    total = kl_to_standard_normal(mean, log_variance)
    split = kl_to_standard_normal(mean, log_variance, per_dimension=True)
    assert split.shape == (6,)
    assert split.sum().item() == pytest.approx(total.item(), rel=1e-5)


def test_a_collapsed_dimension_has_near_zero_kl():
    """What the active-unit count is actually detecting."""
    from parts.e_generative.e1_vae import kl_to_standard_normal

    mean = torch.zeros(128, 3)
    log_variance = torch.zeros(128, 3)
    mean[:, 0] = 2.0  # one informative dimension, two at the prior
    per_dimension = kl_to_standard_normal(mean, log_variance, per_dimension=True)
    assert per_dimension[0].item() > 1.0
    assert per_dimension[1].item() == pytest.approx(0.0, abs=1e-6)


def test_reparameterize_is_differentiable_through_the_moments():
    """The trick's entire purpose: gradients must reach mu and log-sigma."""
    from parts.e_generative.e1_vae import reparameterize

    torch.manual_seed(0)
    mean = torch.zeros(8, 4, requires_grad=True)
    log_variance = torch.zeros(8, 4, requires_grad=True)
    reparameterize(mean, log_variance).sum().backward()
    assert mean.grad is not None and mean.grad.abs().sum() > 0
    assert log_variance.grad is not None and log_variance.grad.abs().sum() > 0


def test_reparameterize_has_the_requested_moments():
    from parts.e_generative.e1_vae import reparameterize

    torch.manual_seed(0)
    mean = torch.full((20000, 1), 3.0)
    log_variance = torch.full((20000, 1), np.log(4.0))  # sigma = 2
    drawn = reparameterize(mean, log_variance)
    assert drawn.mean().item() == pytest.approx(3.0, abs=0.05)
    assert drawn.std().item() == pytest.approx(2.0, abs=0.05)


def test_reconstruction_is_summed_over_pixels_not_averaged():
    r"""Averaging over pixels would silently rescale $\beta$ by the pixel count."""
    from parts.e_generative.e1_vae import elbo_terms

    images = torch.zeros(4, 1, 8, 8)
    logits = torch.zeros(4, 1, 8, 8)  # sigmoid(0) = 0.5 everywhere
    reconstruction, _ = elbo_terms(logits, images, torch.zeros(4, 2), torch.zeros(4, 2))
    # 64 pixels, each costing -log(0.5) = log 2.
    assert reconstruction.item() == pytest.approx(64 * np.log(2), rel=1e-5)


def test_vae_round_trips_its_shapes():
    from parts.e_generative.e1_vae import make_vae

    model = make_vae(latent=8, channels=16, image_size=28, image_channels=1)()
    logits, mean, log_variance = model(torch.randn(3, 1, 28, 28))
    assert logits.shape == (3, 1, 28, 28)
    assert mean.shape == log_variance.shape == (3, 8)
    assert model.decode(torch.randn(5, 8)).shape == (5, 1, 28, 28)


def test_e1_runs():
    from parts.e_generative.e1_vae import main

    assert (
        main(
            [
                "--train-samples",
                "300",
                "--test-samples",
                "150",
                "--epochs",
                "1",
                "--channels",
                "8",
                "--latent",
                "4",
                "--betas",
                "1.0",
                "--sample-count",
                "100",
                "--scorer-epochs",
                "1",
            ]
        )
        == 0
    )


# -- E2: the GAN objectives --------------------------------------------------


def test_minimax_generator_loss_saturates_where_the_generator_is_worst():
    r"""The vanishing gradient the non-saturating form exists to avoid.

    At $s \to -\infty$ the discriminator is confidently rejecting, which is
    exactly when the generator most needs a signal. The minimax gradient
    $-\sigma(s)$ goes to zero there; the non-saturating $1 - \sigma(s)$ goes to
    one.
    """
    from parts.e_generative.e2_dcgan import generator_loss

    for non_saturating, expected in ((False, 0.0), (True, 1.0)):
        scores = torch.full((32,), -12.0, requires_grad=True)
        generator_loss(scores, non_saturating=non_saturating).backward()
        magnitude = scores.grad.abs().mean().item() * len(scores)
        assert magnitude == pytest.approx(expected, abs=1e-4)


def test_both_objectives_agree_about_which_direction_is_better():
    """Same fixed point: both prefer the discriminator being fooled."""
    from parts.e_generative.e2_dcgan import generator_loss

    fooled, rejected = torch.full((16,), 4.0), torch.full((16,), -4.0)
    for non_saturating in (True, False):
        assert generator_loss(fooled, non_saturating=non_saturating) < generator_loss(
            rejected, non_saturating=non_saturating
        )


def test_discriminator_loss_is_lowest_when_it_is_right():
    from parts.e_generative.e2_dcgan import discriminator_loss

    right = discriminator_loss(torch.full((16,), 6.0), torch.full((16,), -6.0))
    wrong = discriminator_loss(torch.full((16,), -6.0), torch.full((16,), 6.0))
    assert right.item() < wrong.item()


def test_label_smoothing_only_touches_the_real_side():
    """Smoothing the fake label too would reward merely-ambiguous samples."""
    from parts.e_generative.e2_dcgan import discriminator_loss

    real, fake = torch.full((8,), 3.0), torch.full((8,), -3.0)
    plain = discriminator_loss(real, fake, label_smoothing=0.0)
    smoothed = discriminator_loss(real, fake, label_smoothing=0.2)
    assert smoothed.item() > plain.item()  # the real term is penalised, the fake term unchanged


def test_class_coverage_detects_a_collapsed_sample_set():
    """A generator stuck on one mode must score 0 entropy, not a middling value."""
    from mlladder.generative import class_coverage, make_feature_extractor

    model = make_feature_extractor(num_classes=10, image_channels=1)()
    with torch.no_grad():  # force every input to class 3
        model.head.bias.fill_(-20.0)
        model.head.bias[3] = 20.0
    stats = class_coverage(model, torch.randn(64, 1, 28, 28), device="cpu", num_classes=10)
    assert stats["coverage"] == pytest.approx(0.1)
    assert stats["entropy"] == pytest.approx(0.0)
    assert stats["max_class_share"] == pytest.approx(1.0)


def test_dcgan_shapes_round_trip():
    from parts.e_generative.e2_dcgan import make_dcgan

    Generator, Discriminator = make_dcgan(latent=16, channels=8, image_size=28, image_channels=1)
    images = Generator()(torch.randn(4, 16))
    assert images.shape == (4, 1, 28, 28)
    assert images.min() >= -1 and images.max() <= 1, "the tanh output must stay in [-1, 1]"
    assert Discriminator()(images).shape == (4,)


def test_e2_runs():
    from parts.e_generative.e2_dcgan import main

    assert (
        main(
            [
                "--train-samples",
                "300",
                "--test-samples",
                "150",
                "--epochs",
                "1",
                "--channels",
                "8",
                "--latent",
                "16",
                "--arms",
                "non_saturating",
                "--sample-count",
                "100",
                "--scorer-epochs",
                "1",
            ]
        )
        == 0
    )
