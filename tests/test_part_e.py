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


# -- E3: the diffusion process ----------------------------------------------


def test_schedules_are_valid_probabilities_and_monotone_in_destruction():
    from parts.e_generative.e3_ddpm import make_schedule

    for kind in ("linear", "cosine"):
        betas = make_schedule(kind, 200)
        assert betas.shape == (200,)
        assert (betas > 0).all() and (betas < 1).all(), f"{kind} produced an invalid beta"
        alpha_bar = np.cumprod(1 - betas)
        assert (np.diff(alpha_bar) <= 0).all(), f"{kind}'s signal must never increase"


def test_cosine_keeps_more_signal_than_linear():
    """Nichol & Dhariwal's whole point: linear destroys the image too early."""
    from parts.e_generative.e3_ddpm import make_schedule

    midpoint = 100
    linear = np.cumprod(1 - make_schedule("linear", 200))[midpoint]
    cosine = np.cumprod(1 - make_schedule("cosine", 200))[midpoint]
    assert cosine > linear


def test_unknown_schedule_is_rejected():
    from parts.e_generative.e3_ddpm import make_schedule

    with pytest.raises(ValueError, match="unknown schedule"):
        make_schedule("quadratic", 10)


def test_add_noise_matches_the_closed_form():
    r"""$x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1-\bar\alpha_t}\epsilon$, exactly."""
    from parts.e_generative.e3_ddpm import Diffusion, make_schedule

    diffusion = Diffusion(make_schedule("linear", 50), device="cpu")
    images = torch.ones(4, 1, 8, 8)
    noise = torch.zeros(4, 1, 8, 8)
    t = torch.full((4,), 10, dtype=torch.long)
    expected = float(torch.sqrt(diffusion.alpha_bar[10]))
    assert diffusion.add_noise(images, t, noise).mean().item() == pytest.approx(expected, rel=1e-5)


def test_noise_overwhelms_the_signal_by_the_final_step():
    from parts.e_generative.e3_ddpm import Diffusion, make_schedule

    diffusion = Diffusion(make_schedule("cosine", 200), device="cpu")
    assert diffusion.signal_to_noise[0] > diffusion.signal_to_noise[-1]
    assert diffusion.signal_to_noise[-1] < 0.1, "the terminal distribution must be nearly pure noise"


def test_sampling_honours_the_requested_step_count():
    """Striding must actually reduce the number of network calls."""
    from parts.e_generative.e3_ddpm import Diffusion, make_schedule

    diffusion = Diffusion(make_schedule("cosine", 40), device="cpu")
    calls = {"n": 0}

    class Counter(torch.nn.Module):
        def forward(self, x, t):
            calls["n"] += 1
            return torch.zeros_like(x)

    diffusion.sample(Counter(), shape=(2, 1, 8, 8), device="cpu", steps=5)
    assert calls["n"] == 5


def test_denoiser_accepts_a_timestep_and_keeps_the_image_shape():
    from parts.e_generative.e3_ddpm import make_denoiser

    model = make_denoiser(channels=16, image_channels=1)()
    output = model(torch.randn(3, 1, 28, 28), torch.randint(0, 100, (3,)))
    assert output.shape == (3, 1, 28, 28)


def test_denoiser_output_depends_on_the_timestep():
    """If it did not, one set of weights could not serve every noise level."""
    from parts.e_generative.e3_ddpm import make_denoiser

    torch.manual_seed(0)
    model = make_denoiser(channels=16, image_channels=1)().eval()
    images = torch.randn(2, 1, 28, 28)
    with torch.no_grad():
        early = model(images, torch.zeros(2, dtype=torch.long))
        late = model(images, torch.full((2,), 99, dtype=torch.long))
    assert not torch.allclose(early, late)


def test_e3_runs():
    from parts.e_generative.e3_ddpm import main

    assert (
        main(
            [
                "--train-samples",
                "200",
                "--test-samples",
                "100",
                "--epochs",
                "1",
                "--channels",
                "8",
                "--schedules",
                "cosine",
                "--sample-steps",
                "4",
                "--sample-count",
                "50",
                "--scorer-epochs",
                "1",
                "--train-steps",
                "20",
            ]
        )
        == 0
    )


# -- E4 and E5 ---------------------------------------------------------------


def test_e4_runs():
    from parts.e_generative.e4_generative_comparison import main

    assert (
        main(
            [
                "--train-samples",
                "200",
                "--test-samples",
                "120",
                "--epochs",
                "1",
                "--channels",
                "8",
                "--latent",
                "16",
                "--sample-count",
                "50",
                "--scorer-epochs",
                "1",
                "--train-steps",
                "20",
                "--sample-steps",
                "4",
            ]
        )
        == 0
    )


def test_latent_positions_compose_the_two_compressions():
    from parts.e_generative.e5_flux_and_sd import latent_positions

    # 1024px at an 8x VAE downsample is 128x128 latent positions...
    assert latent_positions(height=1024, width=1024, downsample=8) == 128 * 128
    # ...and a further 2x2 patch quarters that again.
    assert latent_positions(height=1024, width=1024, downsample=8, patch=2) == 64 * 64


def test_flux_budget_lands_near_its_published_size():
    """~12B is public, so the MMDiT arithmetic can be checked rather than trusted."""
    from parts.e_generative.e5_flux_and_sd import ARCHITECTURES, mmdit_parameters

    total = mmdit_parameters(ARCHITECTURES["FLUX.1 MMDiT"])["total"]
    assert abs(total - 12.0e9) / 12.0e9 < 0.15, f"{total / 1e9:.2f}B against a published ~12B"


def test_double_stream_blocks_cost_about_twice_a_single_stream_one():
    """Two sets of weights per block is what makes them double-stream."""
    from parts.e_generative.e5_flux_and_sd import ARCHITECTURES, mmdit_parameters

    config = dict(ARCHITECTURES["FLUX.1 MMDiT"], depth=1, depth_single_blocks=1)
    budget = mmdit_parameters(config)
    assert budget["double_stream"] == pytest.approx(2 * budget["single_stream"], rel=1e-6)


def test_e5_runs():
    from parts.e_generative.e5_flux_and_sd import main

    assert main(["--resolutions", "512", "1024"]) == 0
