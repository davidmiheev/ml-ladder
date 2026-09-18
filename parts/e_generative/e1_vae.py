"""E1. VAE on MNIST/FashionMNIST

Lab reference: [lucidrains/denoising-diffusion-pytorch — diffusion reference](https://github.com/lucidrains/denoising-diffusion-pytorch)

Every model so far was trained to be *right*. A generative model is trained to
be *plausible*, which is not a quantity any single number captures — and the
VAE is the cleanest place to see why, because its loss is visibly two terms
pulling in opposite directions.

The ELBO is a reconstruction term and a KL term. The reconstruction wants the
latent code to carry as much about this image as possible; the KL wants it to
carry as little as possible, staying close to a prior the model can later sample
from. That is a **rate-distortion trade-off** with $\\beta$ as the exchange rate,
and it has a failure mode at each end: too little KL and the prior is useless
for sampling, too much and the latent stops carrying anything at all.

So this rung sweeps $\\beta$ and measures both terms separately, plus how many
latent dimensions survive — because "the loss went down" is compatible with the
model having quietly stopped using its latent space.

Run with ``ladder run E1`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np

from mlladder import Experiment, experiment, generative, plotting, torchkit, vision


def make_vae(*, latent: int, channels: int, image_size: int, image_channels: int):
    r"""A convolutional encoder to $(\mu, \log\sigma^2)$ and a mirrored decoder.

    The encoder emits **log-variance** rather than variance or standard
    deviation: it is unconstrained in sign, so no clamping or softplus is needed
    to keep the distribution valid, and the KL below is cheapest in those terms.
    """
    vision.require_torch()
    from torch import nn

    reduced = image_size // 4

    class VAE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.encoder = nn.Sequential(
                nn.Conv2d(image_channels, channels, 3, stride=2, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels, channels * 2, 3, stride=2, padding=1),
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.Flatten(),
            )
            self.to_moments = nn.Linear(channels * 2 * reduced * reduced, latent * 2)
            self.from_latent = nn.Linear(latent, channels * 2 * reduced * reduced)
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channels, image_channels, 4, stride=2, padding=1),
            )
            self.latent, self.channels, self.reduced = latent, channels, reduced

        def encode(self, images):
            moments = self.to_moments(self.encoder(images))
            return moments[:, : self.latent], moments[:, self.latent :]

        def decode(self, code):
            hidden = self.from_latent(code).view(-1, self.channels * 2, self.reduced, self.reduced)
            return self.decoder(hidden)  # logits; the loss applies the sigmoid

        def forward(self, images):
            mean, log_variance = self.encode(images)
            return self.decode(reparameterize(mean, log_variance)), mean, log_variance

    return VAE


def reparameterize(mean, log_variance):
    r"""Sample $z = \mu + \sigma \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$.

    The whole trick of the VAE. Sampling $z \sim q(z \mid x)$ directly has no
    gradient with respect to $\mu$ and $\sigma$ — the randomness sits between
    them and the loss. Writing the sample as a *deterministic* function of the
    parameters plus parameter-free noise moves the randomness out of the path,
    so the gradient flows through $\mu$ and $\sigma$ as through anything else.
    """
    torch = vision.require_torch()
    standard_deviation = torch.exp(0.5 * log_variance)
    return mean + standard_deviation * torch.randn_like(standard_deviation)


def kl_to_standard_normal(mean, log_variance, *, per_dimension: bool = False):
    r"""$D_{KL}(\mathcal{N}(\mu, \sigma^2) \,\|\, \mathcal{N}(0, I))$, in closed form.

    $$D_{KL} = -\tfrac{1}{2}\sum_j \left(1 + \log\sigma_j^2 - \mu_j^2 - \sigma_j^2\right)$$

    Closed form because both distributions are Gaussian, which is most of why
    that prior is chosen — no sampling is needed for this term, so the only
    Monte-Carlo noise in the ELBO comes from the reconstruction.

    ``per_dimension`` keeps the sum unreduced over $j$, which is how posterior
    collapse is detected: a dimension whose KL is ~0 is one the encoder has
    given up on and set to the prior.
    """
    terms = -0.5 * (1 + log_variance - mean.pow(2) - log_variance.exp())
    return terms.mean(dim=0) if per_dimension else terms.sum(dim=1).mean()


def elbo_terms(logits, images, mean, log_variance):
    """Reconstruction (in nats per image) and KL, kept separate.

    Summed over pixels rather than averaged, because the KL is a sum over latent
    dimensions and mixing a per-pixel mean with a per-dimension sum silently
    rescales $\\beta$ by the pixel count — which is the most common way this
    loss is got wrong.
    """
    torch = vision.require_torch()
    reconstruction = (
        torch.nn.functional.binary_cross_entropy_with_logits(logits, images, reduction="none")
        .flatten(1)
        .sum(dim=1)
        .mean()
    )
    return reconstruction, kl_to_standard_normal(mean, log_variance)


@experiment(
    "e1_vae",
    defaults={
        "dataset": "mnist",
        "train_samples": 12000,
        "test_samples": 2000,
        "latent": 16,
        "channels": 32,
        "epochs": 12,
        "batch_size": 128,
        "lr": 2e-3,
        "betas": [0.25, 1.0, 4.0, 16.0],
        "reference_beta": 1.0,
        "active_threshold": 0.01,
        "sample_count": 1000,
        "scorer_epochs": 2,
    },
    description="A convolutional VAE swept over beta, measuring the rate-distortion trade-off and posterior collapse.",
)
def run(exp: Experiment) -> dict[str, float]:
    """VAE on MNIST/FashionMNIST"""
    torch = vision.require_torch()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)
    print(f"[e1] device: {device}")

    raw_train, y_train = vision.load(exp["dataset"], split="train", limit=exp["train_samples"], rng=rng)
    raw_test, _ = vision.load(exp["dataset"], split="test", limit=exp["test_samples"], rng=rng)
    # Scaled to [0, 1] rather than standardized: the decoder emits Bernoulli
    # logits, so the target has to be a probability.
    x_train = (raw_train.astype(np.float32) / 255.0).reshape(len(raw_train), -1, *raw_train.shape[1:3])
    x_test = (raw_test.astype(np.float32) / 255.0).reshape(len(raw_test), -1, *raw_test.shape[1:3])
    if x_train.shape[1] not in (1, 3):
        x_train = x_train.transpose(0, 3, 1, 2)
        x_test = x_test.transpose(0, 3, 1, 2)
    image_channels, image_size = x_train.shape[1], x_train.shape[2]
    print(f"[e1] {len(x_train)} train / {len(x_test)} test, {image_channels}x{image_size}x{image_size}")

    train_tensor = torch.from_numpy(x_train).to(device)
    test_tensor = torch.from_numpy(x_test).to(device)

    # The scorer is trained once on real data and shared by every arm, so the
    # Frechet distances below are comparable with each other.
    scorer = generative.train_feature_extractor(
        x_train, y_train, device=device, seed=exp.seed, epochs=exp["scorer_epochs"]
    )

    summary: dict[str, float] = {}
    curves: dict[str, list[float]] = {}
    samples_by_beta: dict[float, np.ndarray] = {}
    reconstructions_by_beta: dict[float, np.ndarray] = {}

    for beta in exp["betas"]:
        torchkit.seed_torch(exp.seed)
        model = make_vae(
            latent=exp["latent"], channels=exp["channels"], image_size=image_size, image_channels=image_channels
        )().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=1e-5)

        losses: list[float] = []
        for epoch in range(1, exp["epochs"] + 1):
            model.train()
            permutation = torch.randperm(len(train_tensor), device=device)
            epoch_losses = []
            for start in range(0, len(train_tensor), exp["batch_size"]):
                batch = train_tensor[permutation[start : start + exp["batch_size"]]]
                logits, mean, log_variance = model(batch)
                reconstruction, kl = elbo_terms(logits, batch, mean, log_variance)
                loss = reconstruction + beta * kl
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            losses.append(float(np.mean(epoch_losses)))
            if epoch % max(exp["epochs"] // 3, 1) == 0:
                print(f"[e1] beta={beta:<5} epoch {epoch:>3}  loss={losses[-1]:.2f}")
        curves[f"beta={beta}"] = losses

        # -- evaluate: the two ELBO terms, separately ------------------------
        model.eval()
        with torch.no_grad():
            logits, mean, log_variance = model(test_tensor)
            reconstruction, kl = elbo_terms(logits, test_tensor, mean, log_variance)
            per_dimension = kl_to_standard_normal(mean, log_variance, per_dimension=True).cpu().numpy()
            reconstructed = torch.sigmoid(logits).cpu().numpy()
            # Prior samples: the thing a VAE is *for*, and the thing the KL term
            # is paying for. A model with a low reconstruction error and an
            # unusable prior has optimized the wrong half of its own objective.
            code = torch.randn(exp["sample_count"], exp["latent"], device=device)
            samples = torch.sigmoid(model.decode(code)).cpu().numpy()

        active = int((per_dimension > exp["active_threshold"]).sum())
        # -(reconstruction + KL) is the ELBO at beta = 1 regardless of the beta
        # trained at, so arms remain comparable on one scale.
        summary |= {
            f"beta/{beta}/reconstruction": float(reconstruction),
            f"beta/{beta}/kl": float(kl),
            f"beta/{beta}/elbo": float(-(reconstruction + kl)),
            f"beta/{beta}/active_units": float(active),
            f"beta/{beta}/train_loss": losses[-1],
        }
        real_subset = x_test[: exp["sample_count"]]
        summary[f"beta/{beta}/fd_samples"] = generative.distribution_distance(
            scorer, real_subset, samples, device=device
        )
        summary[f"beta/{beta}/fd_reconstructions"] = generative.distribution_distance(
            scorer, real_subset, reconstructed[: exp["sample_count"]], device=device
        )
        samples_by_beta[beta] = samples
        reconstructions_by_beta[beta] = reconstructed
        print(
            f"[e1] beta={beta:<5} recon={reconstruction:7.2f}  KL={kl:6.2f}  "
            f"active={active:>2}/{exp['latent']}  fd(samples)={summary[f'beta/{beta}/fd_samples']:6.2f}  "
            f"fd(recon)={summary[f'beta/{beta}/fd_reconstructions']:6.2f}"
        )

    # -- what the sweep is for -----------------------------------------------
    betas = exp["betas"]
    best_elbo = max(betas, key=lambda b: summary[f"beta/{b}/elbo"])
    best_samples = min(betas, key=lambda b: summary[f"beta/{b}/fd_samples"])
    summary["best/elbo_beta"] = float(best_elbo)
    summary["best/sample_beta"] = float(best_samples)
    summary["best/agree"] = float(best_elbo == best_samples)
    summary["collapse/units_lost"] = (
        summary[f"beta/{max(betas)}/active_units"] - summary[f"beta/{min(betas)}/active_units"]
    )
    summary["rate_range"] = summary[f"beta/{min(betas)}/kl"] - summary[f"beta/{max(betas)}/kl"]
    summary["distortion_range"] = (
        summary[f"beta/{max(betas)}/reconstruction"] - summary[f"beta/{min(betas)}/reconstruction"]
    )
    print(
        f"[e1] best ELBO at beta={best_elbo}, best samples at beta={best_samples} "
        f"({'agree' if best_elbo == best_samples else 'DISAGREE'})"
    )
    print(
        f"[e1] across the sweep the rate falls {summary['rate_range']:.1f} nats while distortion rises "
        f"{summary['distortion_range']:.1f}, and {abs(summary['collapse/units_lost']):.0f} latent units collapse"
    )

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "training",
        plotting.plot_curves(
            curves,
            x=list(range(1, exp["epochs"] + 1)),
            title="Training loss (different objectives: comparable in trend, not in value)",
            xlabel="epoch",
            ylabel="reconstruction + beta*KL",
            logy=True,
        ),
    )
    exp.log_figure(
        "rate_distortion",
        plotting.plot_curves(
            {"test set": [summary[f"beta/{b}/reconstruction"] for b in betas]},
            x=[summary[f"beta/{b}/kl"] for b in betas],
            title="Rate-distortion: each point is one beta, from high KL (left) to collapsed (right)",
            xlabel="rate — KL to the prior (nats)",
            ylabel="distortion — reconstruction (nats)",
            marker="o",
        ),
    )
    exp.log_figure(
        "active_units",
        plotting.plot_curves(
            {"active latent dimensions": [summary[f"beta/{b}/active_units"] for b in betas]},
            x=betas,
            title=f"Posterior collapse: dimensions with KL > {exp['active_threshold']}",
            xlabel="beta",
            ylabel=f"active of {exp['latent']}",
            marker="o",
            logx=True,
        ),
    )
    exp.log_figure(
        "samples",
        plotting.plot_panel_rows(
            [samples_by_beta[b][:8, 0] for b in betas],
            [f"beta={b}" for b in betas],
            grayscale_first=False,
            title="Samples from the prior",
        ),
    )
    exp.log_figure(
        "reconstructions",
        plotting.plot_panel_rows(
            [x_test[:8, 0], *[reconstructions_by_beta[b][:8, 0] for b in betas]],
            ["input", *[f"beta={b}" for b in betas]],
            grayscale_first=False,
            title="Reconstructions",
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
