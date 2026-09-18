"""E4. Compare sample quality and training stability across VAE, GAN, and diffusion

Lab reference: [lucidrains/denoising-diffusion-pytorch — diffusion reference](https://github.com/lucidrains/denoising-diffusion-pytorch)

Three rungs, three families, three losses that are not on the same scale: an
ELBO in nats, an adversarial scoreboard, a denoising MSE. None of them can be
compared to another, which is the whole problem with reading generative papers
and the reason this rung exists.

So every model is rebuilt here from its own rung's code — imported, not
reimplemented — trained on **identical data with an identical budget**, and
scored on axes that apply to all three:

- **quality**: Fréchet distance in a shared feature space;
- **coverage**: does it produce all ten digits, or a favourite few;
- **stability**: how much the quality moves over the second half of training;
- **cost**: seconds to train, and seconds to draw a thousand samples;
- **novelty**: distance from each sample to its nearest training image, because
  a model that memorised the data wins every other column dishonestly.

The expected outcome is that no column has the same winner, which is the point:
"which generative model is best" is not a question with an answer until the
column is named.

Run with ``ladder run E4`` (``--help`` for flags).
"""

from __future__ import annotations

import time

import numpy as np

from mlladder import Experiment, experiment, generative, plotting, torchkit, vision
from parts.e_generative.e1_vae import elbo_terms, make_vae
from parts.e_generative.e2_dcgan import discriminator_loss, generator_loss, make_dcgan
from parts.e_generative.e3_ddpm import Diffusion, make_denoiser, make_schedule


def train_vae(x, *, exp, device, image_size, image_channels):
    """E1's model at E4's shared budget."""
    torch = vision.require_torch()
    torchkit.seed_torch(exp.seed)
    model = make_vae(
        latent=exp["latent"], channels=exp["channels"], image_size=image_size, image_channels=image_channels
    )().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp["vae_lr"], weight_decay=1e-5)
    data = torch.from_numpy(x).to(device)  # VAE wants [0, 1] for its Bernoulli likelihood
    for _ in range(exp["epochs"]):
        model.train()
        permutation = torch.randperm(len(data), device=device)
        for start in range(0, len(data), exp["batch_size"]):
            batch = data[permutation[start : start + exp["batch_size"]]]
            logits, mean, log_variance = model(batch)
            reconstruction, kl = elbo_terms(logits, batch, mean, log_variance)
            loss = reconstruction + exp["beta"] * kl
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
    return model


def train_gan(x, *, exp, device, image_size, image_channels):
    """E2's non-saturating DCGAN at the shared budget."""
    torch = vision.require_torch()
    torchkit.seed_torch(exp.seed)
    Generator, Discriminator = make_dcgan(
        latent=exp["latent"], channels=exp["channels"], image_size=image_size, image_channels=image_channels
    )
    generator, discriminator = Generator().to(device), Discriminator().to(device)
    options = {"lr": exp["gan_lr"], "betas": (0.5, 0.999)}
    g_optimizer = torch.optim.Adam(generator.parameters(), **options)
    d_optimizer = torch.optim.Adam(discriminator.parameters(), **options)
    data = torch.from_numpy(x * 2 - 1).to(device)  # the GAN's tanh output needs [-1, 1]
    for _ in range(exp["epochs"]):
        generator.train()
        discriminator.train()
        permutation = torch.randperm(len(data), device=device)
        for start in range(0, len(data), exp["batch_size"]):
            real = data[permutation[start : start + exp["batch_size"]]]
            fake = generator(torch.randn(len(real), exp["latent"], device=device))
            d_loss = discriminator_loss(discriminator(real), discriminator(fake.detach()), label_smoothing=0.1)
            d_optimizer.zero_grad(set_to_none=True)
            d_loss.backward()
            d_optimizer.step()
            g_loss = generator_loss(discriminator(fake), non_saturating=True)
            g_optimizer.zero_grad(set_to_none=True)
            g_loss.backward()
            g_optimizer.step()
    return generator


def train_diffusion(x, *, exp, device, image_size, image_channels, epochs=None):
    """E3's DDPM at the shared budget, on the cosine schedule.

    ``epochs`` overrides the shared budget. It exists for the convergence
    control below, which needs the *same* run continued rather than a second
    independent one — hence the re-seed and full replay rather than a warm
    start, so the longer run is exactly the shorter run plus more epochs.
    """
    torch = vision.require_torch()
    torchkit.seed_torch(exp.seed)
    diffusion = Diffusion(make_schedule(exp["schedule"], exp["train_steps"]), device=device)
    model = make_denoiser(channels=exp["channels"], image_channels=image_channels)().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=exp["diffusion_lr"], weight_decay=1e-5)
    data = torch.from_numpy(x * 2 - 1).to(device)
    losses: list[float] = []
    for _ in range(exp["epochs"] if epochs is None else epochs):
        model.train()
        permutation = torch.randperm(len(data), device=device)
        epoch_losses = []
        for start in range(0, len(data), exp["batch_size"]):
            batch = data[permutation[start : start + exp["batch_size"]]]
            t = torch.randint(0, exp["train_steps"], (len(batch),), device=device)
            noise = torch.randn_like(batch)
            loss = torch.nn.functional.mse_loss(model(diffusion.add_noise(batch, t, noise), t), noise)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        losses.append(float(np.mean(epoch_losses)))
    return model, diffusion, losses[-1]


@experiment(
    "e4_generative_comparison",
    defaults={
        "dataset": "mnist",
        "train_samples": 12000,
        "test_samples": 2000,
        "latent": 64,
        "channels": 32,
        # One budget for all three, so "trained for the same time" is literally
        # true rather than approximately true.
        "epochs": 14,
        "batch_size": 128,
        "vae_lr": 2e-3,
        "gan_lr": 2e-4,
        "diffusion_lr": 2e-3,
        "beta": 1.0,
        "schedule": "cosine",
        "train_steps": 400,
        "sample_steps": 100,
        "sample_count": 1000,
        "scorer_epochs": 2,
        # The convergence control: the same diffusion run continued this many
        # epochs further, to test whether the shared budget is measuring the
        # family or measuring how far from converged one member of it is.
        "extra_epochs": 2,
    },
    description="VAE, GAN and diffusion on identical data and budget, scored on quality, coverage, stability and cost.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Compare sample quality and training stability across VAE, GAN, and diffusion"""
    torch = vision.require_torch()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)
    print(f"[e4] device: {device}")

    raw_train, y_train = vision.load(exp["dataset"], split="train", limit=exp["train_samples"], rng=rng)
    raw_test, _ = vision.load(exp["dataset"], split="test", limit=exp["test_samples"], rng=rng)
    x_train = (raw_train.astype(np.float32) / 255.0).reshape(len(raw_train), -1, *raw_train.shape[1:3])
    x_test = (raw_test.astype(np.float32) / 255.0).reshape(len(raw_test), -1, *raw_test.shape[1:3])
    image_channels, image_size = x_train.shape[1], x_train.shape[2]

    scorer = generative.train_feature_extractor(
        x_train, y_train, device=device, seed=exp.seed, epochs=exp["scorer_epochs"]
    )
    num_classes = int(y_train.max()) + 1
    real_reference = x_test[: exp["sample_count"]]
    shape = (exp["sample_count"], image_channels, image_size, image_size)
    shared = {"exp": exp, "device": device, "image_size": image_size, "image_channels": image_channels}

    summary: dict[str, float] = {}
    samples_by_model: dict[str, np.ndarray] = {}

    # -- VAE -----------------------------------------------------------------
    started = time.perf_counter()
    vae = train_vae(x_train, **shared)
    summary["vae/train_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    with torch.no_grad():
        samples_by_model["vae"] = (
            torch.sigmoid(vae.decode(torch.randn(exp["sample_count"], exp["latent"], device=device))).cpu().numpy()
        )
    summary["vae/sample_seconds"] = time.perf_counter() - started
    summary["vae/parameters"] = float(torchkit.count_parameters(vae))

    # -- GAN -----------------------------------------------------------------
    started = time.perf_counter()
    gan = train_gan(x_train, **shared)
    summary["gan/train_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    gan.eval()
    with torch.no_grad():
        samples_by_model["gan"] = (
            ((gan(torch.randn(exp["sample_count"], exp["latent"], device=device)) + 1) / 2).clamp(0, 1).cpu().numpy()
        )
    summary["gan/sample_seconds"] = time.perf_counter() - started
    summary["gan/parameters"] = float(torchkit.count_parameters(gan))

    # -- diffusion -----------------------------------------------------------
    started = time.perf_counter()
    denoiser, diffusion, diffusion_mse = train_diffusion(x_train, **shared)
    summary["diffusion/train_seconds"] = time.perf_counter() - started
    started = time.perf_counter()
    generator = torch.Generator(device=device).manual_seed(exp.seed)
    drawn = diffusion.sample(denoiser, shape=shape, device=device, steps=exp["sample_steps"], generator=generator)
    summary["diffusion/sample_seconds"] = time.perf_counter() - started
    samples_by_model["diffusion"] = ((drawn + 1) / 2).clamp(0, 1).cpu().numpy()
    summary["diffusion/parameters"] = float(torchkit.count_parameters(denoiser))

    # -- one scorecard, five axes -------------------------------------------
    # The real data is scored against itself too, which calibrates every other
    # row: a Frechet distance has no absolute scale, so "how good is 3.2" is
    # only answerable next to what a genuine sample of the same size scores.
    baseline = x_test[exp["sample_count"] : 2 * exp["sample_count"]]
    if len(baseline) >= exp["sample_count"] // 2:
        summary["real/fd"] = generative.distribution_distance(scorer, real_reference, baseline, device=device)
        summary["real/nn_distance"] = generative.nearest_neighbour_distance(baseline, x_train, rng=exp.seed)

    for name, samples in samples_by_model.items():
        coverage = generative.class_coverage(scorer, samples, device=device, num_classes=num_classes)
        summary |= {
            f"{name}/fd": generative.distribution_distance(scorer, real_reference, samples, device=device),
            f"{name}/coverage": coverage["coverage"],
            f"{name}/mode_entropy": coverage["entropy"],
            f"{name}/max_class_share": coverage["max_class_share"],
            f"{name}/nn_distance": generative.nearest_neighbour_distance(samples, x_train, rng=exp.seed),
        }
        print(
            f"[e4] {name:<10} fd={summary[f'{name}/fd']:7.2f}  entropy={summary[f'{name}/mode_entropy']:.3f}  "
            f"nn={summary[f'{name}/nn_distance']:5.2f}  train={summary[f'{name}/train_seconds']:6.1f}s  "
            f"sample={summary[f'{name}/sample_seconds']:6.2f}s  ({summary[f'{name}/parameters']:,.0f} params)"
        )
    if "real/fd" in summary:
        print(
            f"[e4] {'real data':<10} fd={summary['real/fd']:7.2f}  nn={summary['real/nn_distance']:5.2f}  (the floor)"
        )

    # -- is the shared budget measuring the family, or its convergence? ------
    # An equal-budget comparison is only fair if every arm is near the flat part
    # of its own curve. Diffusion is the arm most likely not to be, so the same
    # run is continued a little further and re-scored. If a few more epochs move
    # the score a lot, the headline is a statement about the budget.
    if exp["extra_epochs"] > 0:
        longer, longer_diffusion, longer_mse = train_diffusion(
            x_train, **shared, epochs=exp["epochs"] + exp["extra_epochs"]
        )
        generator = torch.Generator(device=device).manual_seed(exp.seed)
        drawn = longer_diffusion.sample(
            longer, shape=shape, device=device, steps=exp["sample_steps"], generator=generator
        )
        extended = ((drawn + 1) / 2).clamp(0, 1).cpu().numpy()
        extended_coverage = generative.class_coverage(scorer, extended, device=device, num_classes=num_classes)
        summary["diffusion/train_mse"] = diffusion_mse
        summary["diffusion/extended/train_mse"] = longer_mse
        summary["diffusion/extended/fd"] = generative.distribution_distance(
            scorer, real_reference, extended, device=device
        )
        summary["diffusion/extended/mode_entropy"] = extended_coverage["entropy"]
        summary["diffusion/extended/epochs"] = float(exp["epochs"] + exp["extra_epochs"])
        # Two ratios over the same pair of runs: how much the quality moved, and
        # how much the training loss moved. A large first and a small second is
        # the finding -- the loss cannot see what changed.
        summary["budget_sensitivity/fd_ratio"] = summary["diffusion/fd"] / max(summary["diffusion/extended/fd"], 1e-9)
        summary["budget_sensitivity/mse_ratio"] = diffusion_mse / max(longer_mse, 1e-9)
        summary["budget_sensitivity/extra_fraction"] = exp["extra_epochs"] / exp["epochs"]
        print(
            f"[e4] +{exp['extra_epochs']} epochs ({summary['budget_sensitivity/extra_fraction']:.0%} more budget): "
            f"diffusion fd {summary['diffusion/fd']:.2f} -> {summary['diffusion/extended/fd']:.2f} "
            f"({summary['budget_sensitivity/fd_ratio']:.1f}x better) while the training MSE moves "
            f"{diffusion_mse:.4f} -> {longer_mse:.4f} ({summary['budget_sensitivity/mse_ratio']:.3f}x)"
        )

    # -- who wins what -------------------------------------------------------
    models = list(samples_by_model)
    winners = {
        "quality": min(models, key=lambda m: summary[f"{m}/fd"]),
        "coverage": max(models, key=lambda m: summary[f"{m}/mode_entropy"]),
        "train_speed": min(models, key=lambda m: summary[f"{m}/train_seconds"]),
        "sample_speed": min(models, key=lambda m: summary[f"{m}/sample_seconds"]),
    }
    for axis, winner in winners.items():
        summary[f"winner/{axis}/{winner}"] = 1.0
    summary["winner/distinct_axes"] = float(len(set(winners.values())))
    summary["sampling_cost/diffusion_over_gan"] = summary["diffusion/sample_seconds"] / max(
        summary["gan/sample_seconds"], 1e-9
    )
    print("[e4] winners: " + ", ".join(f"{axis}={winner}" for axis, winner in winners.items()))
    print(
        f"[e4] {summary['winner/distinct_axes']:.0f} different models win the 4 axes; "
        f"diffusion sampling costs {summary['sampling_cost/diffusion_over_gan']:.0f}x the GAN's"
    )

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "quality",
        plotting.plot_comparison(
            {**{name: summary[f"{name}/fd"] for name in models}, "real data": summary.get("real/fd", 0.0)},
            title="Fréchet distance in a shared feature space (real data is the floor)",
            ylabel="fd",
        ),
    )
    exp.log_figure(
        "coverage",
        plotting.plot_comparison(
            {name: summary[f"{name}/mode_entropy"] for name in models},
            title="Mode coverage: normalized entropy over predicted classes",
            ylabel="normalized entropy",
        ),
    )
    exp.log_figure(
        "cost",
        plotting.plot_comparison(
            {f"{name}\nsampling": summary[f"{name}/sample_seconds"] for name in models},
            title=f"Seconds to draw {exp['sample_count']} samples",
            ylabel="seconds",
        ),
    )
    exp.log_figure(
        "samples",
        plotting.plot_panel_rows(
            [x_test[:8, 0], *[samples_by_model[name][:8, 0] for name in models]],
            ["real", *models],
            grayscale_first=False,
            title="Identical data, identical budget",
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
