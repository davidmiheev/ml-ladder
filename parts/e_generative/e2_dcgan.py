"""E2. DCGAN on a small image dataset

Lab reference: [lucidrains/denoising-diffusion-pytorch — diffusion reference](https://github.com/lucidrains/denoising-diffusion-pytorch)

E1's VAE optimizes a bound on the likelihood, so its loss going down means
something. A GAN has no such guarantee: two networks play a game, and the loss
is a scoreboard in that game rather than a measure of quality. A generator whose
loss is falling may be winning against a discriminator that has stopped
learning, which is a different thing from producing good images.

That makes GANs the place where **what you measure** matters most, and this rung
measures three things the loss curve cannot show:

- **mode collapse** — the classic failure, where the generator finds a few
  outputs the discriminator cannot reject and stops exploring. A Fréchet
  distance can miss it; classifying the samples and looking at the histogram
  cannot.
- **the non-saturating trick** — the original minimax objective has a vanishing
  gradient exactly when the generator is worst, and everyone quietly uses a
  different loss. This rung runs both.
- **stability** — the variance of the game over training, which is the property
  practitioners actually complain about.

Run with ``ladder run E2`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np

from mlladder import Experiment, experiment, generative, plotting, torchkit, vision


def make_dcgan(*, latent: int, channels: int, image_size: int, image_channels: int):
    """A DCGAN generator and discriminator, at the paper's structural rules.

    Strided convolutions rather than pooling, batch norm everywhere except the
    generator's output and the discriminator's input, ReLU in the generator and
    leaky ReLU in the discriminator. These are not arbitrary: the paper found
    them by trying to make an unstable game trainable at all, and departing
    from them is the usual reason a from-scratch GAN never converges.
    """
    vision.require_torch()
    from torch import nn

    seed_size = image_size // 4

    class Generator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.project = nn.Linear(latent, channels * 2 * seed_size * seed_size)
            self.body = nn.Sequential(
                nn.BatchNorm2d(channels * 2),
                nn.ReLU(inplace=True),
                nn.ConvTranspose2d(channels * 2, channels, 4, stride=2, padding=1),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
                # No norm on the output layer: it would fight the tanh for
                # control of the output scale.
                nn.ConvTranspose2d(channels, image_channels, 4, stride=2, padding=1),
                nn.Tanh(),
            )
            self.channels, self.seed_size = channels, seed_size

        def forward(self, code):
            hidden = self.project(code).view(-1, self.channels * 2, self.seed_size, self.seed_size)
            return self.body(hidden)

    class Discriminator(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                # No norm on the input layer, for the same reason in reverse.
                nn.Conv2d(image_channels, channels, 4, stride=2, padding=1),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Conv2d(channels, channels * 2, 4, stride=2, padding=1),
                nn.BatchNorm2d(channels * 2),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Flatten(),
                nn.Linear(channels * 2 * seed_size * seed_size, 1),
            )

        def forward(self, images):
            return self.body(images).squeeze(1)

    return Generator, Discriminator


def generator_loss(fake_scores, *, non_saturating: bool):
    r"""The two objectives, and why nobody uses the one in the theorem.

    The minimax game asks the generator to *minimize* $\log(1 - D(G(z)))$. Early
    in training $D(G(z)) \approx 0$, and

    $$\frac{\partial}{\partial s}\log(1 - \sigma(s)) = -\sigma(s) \to 0
    \quad\text{as } s \to -\infty,$$

    so the gradient vanishes precisely when the generator is worst and needs it
    most. The non-saturating form instead *maximizes* $\log D(G(z))$, whose
    gradient is $1 - \sigma(s) \to 1$ in the same limit. Same fixed point,
    opposite gradient behaviour where it matters.
    """
    torch = vision.require_torch()
    import torch.nn.functional as F

    if non_saturating:
        # maximize log D(G(z)) == minimize BCE against the "real" label
        return F.binary_cross_entropy_with_logits(fake_scores, torch.ones_like(fake_scores))
    # minimize log(1 - D(G(z))) directly, as the theorem states it
    return -F.binary_cross_entropy_with_logits(fake_scores, torch.zeros_like(fake_scores))


def discriminator_loss(real_scores, fake_scores, *, label_smoothing: float = 0.0):
    """Binary cross-entropy on both halves, with optional one-sided smoothing.

    Smoothing only the *real* label is deliberate: smoothing the fake label too
    would give the generator an incentive to produce samples the discriminator
    is merely unsure about, which is the opposite of the intended effect.
    """
    torch = vision.require_torch()
    import torch.nn.functional as F

    real_target = torch.full_like(real_scores, 1.0 - label_smoothing)
    return F.binary_cross_entropy_with_logits(real_scores, real_target) + F.binary_cross_entropy_with_logits(
        fake_scores, torch.zeros_like(fake_scores)
    )


@experiment(
    "e2_dcgan",
    defaults={
        "dataset": "mnist",
        "train_samples": 12000,
        "test_samples": 2000,
        "latent": 64,
        "channels": 32,
        "epochs": 14,
        "batch_size": 128,
        "lr": 2e-4,
        "beta1": 0.5,  # DCGAN's momentum fix; 0.9 is famously unstable here
        "label_smoothing": 0.1,
        "arms": ["non_saturating", "minimax"],
        "sample_count": 1000,
        "scorer_epochs": 2,
        "checkpoint_every": 2,
    },
    description="A DCGAN with the minimax and non-saturating generator losses, scored for mode collapse and stability.",
)
def run(exp: Experiment) -> dict[str, float]:
    """DCGAN on a small image dataset"""
    torch = vision.require_torch()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)
    print(f"[e2] device: {device}")

    raw_train, y_train = vision.load(exp["dataset"], split="train", limit=exp["train_samples"], rng=rng)
    raw_test, _ = vision.load(exp["dataset"], split="test", limit=exp["test_samples"], rng=rng)
    scaled = raw_train.astype(np.float32) / 255.0
    x_train = scaled.reshape(len(scaled), -1, *raw_train.shape[1:3])
    x_test = (raw_test.astype(np.float32) / 255.0).reshape(len(raw_test), -1, *raw_test.shape[1:3])
    image_channels, image_size = x_train.shape[1], x_train.shape[2]

    # The generator ends in tanh, so its outputs live in [-1, 1]; the data has
    # to be moved there too or the discriminator separates the two trivially by
    # range alone and the game never starts.
    train_tensor = torch.from_numpy(x_train * 2 - 1).to(device)
    print(f"[e2] {len(x_train)} train images, {image_channels}x{image_size}x{image_size}, scaled to [-1, 1]")

    scorer = generative.train_feature_extractor(
        x_train, y_train, device=device, seed=exp.seed, epochs=exp["scorer_epochs"]
    )
    num_classes = int(y_train.max()) + 1
    real_reference = x_test[: exp["sample_count"]]

    summary: dict[str, float] = {}
    curves: dict[str, list[float]] = {}
    trajectories: dict[str, list[float]] = {}
    samples_by_arm: dict[str, np.ndarray] = {}

    for arm in exp["arms"]:
        torchkit.seed_torch(exp.seed)
        Generator, Discriminator = make_dcgan(
            latent=exp["latent"], channels=exp["channels"], image_size=image_size, image_channels=image_channels
        )
        generator, discriminator = Generator().to(device), Discriminator().to(device)
        # betas=(0.5, 0.999) is DCGAN's other structural fix: Adam's default
        # first moment makes the game oscillate.
        options = {"lr": exp["lr"], "betas": (exp["beta1"], 0.999)}
        g_optimizer = torch.optim.Adam(generator.parameters(), **options)
        d_optimizer = torch.optim.Adam(discriminator.parameters(), **options)

        g_losses, d_losses, fd_trace = [], [], []
        for epoch in range(1, exp["epochs"] + 1):
            generator.train()
            discriminator.train()
            permutation = torch.randperm(len(train_tensor), device=device)
            epoch_g, epoch_d = [], []
            for start in range(0, len(train_tensor), exp["batch_size"]):
                real = train_tensor[permutation[start : start + exp["batch_size"]]]
                code = torch.randn(len(real), exp["latent"], device=device)
                fake = generator(code)

                # -- discriminator: detach so the generator is not updated here
                d_loss = discriminator_loss(
                    discriminator(real), discriminator(fake.detach()), label_smoothing=exp["label_smoothing"]
                )
                d_optimizer.zero_grad(set_to_none=True)
                d_loss.backward()
                d_optimizer.step()

                # -- generator: re-score the same fakes through the updated D
                g_loss = generator_loss(discriminator(fake), non_saturating=arm == "non_saturating")
                g_optimizer.zero_grad(set_to_none=True)
                g_loss.backward()
                g_optimizer.step()

                epoch_g.append(g_loss.item())
                epoch_d.append(d_loss.item())

            g_losses.append(float(np.mean(epoch_g)))
            d_losses.append(float(np.mean(epoch_d)))

            if epoch % exp["checkpoint_every"] == 0 or epoch == exp["epochs"]:
                # Scoring during training, not only at the end: a GAN's quality
                # is not monotone, and the final model is not always the best
                # one it passed through.
                generator.eval()
                with torch.no_grad():
                    code = torch.randn(exp["sample_count"], exp["latent"], device=device)
                    drawn = ((generator(code) + 1) / 2).clamp(0, 1).cpu().numpy()
                fd_trace.append(generative.distribution_distance(scorer, real_reference, drawn, device=device))
                print(
                    f"[e2] {arm:<15} epoch {epoch:>3}  g={g_losses[-1]:6.3f}  d={d_losses[-1]:6.3f}  fd={fd_trace[-1]:6.2f}"
                )

        curves[f"{arm}/generator"] = g_losses
        curves[f"{arm}/discriminator"] = d_losses
        trajectories[arm] = fd_trace

        generator.eval()
        with torch.no_grad():
            code = torch.randn(exp["sample_count"], exp["latent"], device=device)
            samples = ((generator(code) + 1) / 2).clamp(0, 1).cpu().numpy()
        samples_by_arm[arm] = samples

        coverage = generative.class_coverage(scorer, samples, device=device, num_classes=num_classes)
        summary |= {
            f"{arm}/fd": generative.distribution_distance(scorer, real_reference, samples, device=device),
            f"{arm}/fd_best": float(min(fd_trace)),
            f"{arm}/fd_final_over_best": fd_trace[-1] / max(min(fd_trace), 1e-9),
            f"{arm}/coverage": coverage["coverage"],
            f"{arm}/mode_entropy": coverage["entropy"],
            f"{arm}/max_class_share": coverage["max_class_share"],
            f"{arm}/generator_loss": g_losses[-1],
            f"{arm}/discriminator_loss": d_losses[-1],
            # Stability, as the swing of the scoreboard over the second half of
            # training — a converged game should have stopped moving.
            f"{arm}/d_loss_std_late": float(np.std(d_losses[len(d_losses) // 2 :])),
            f"{arm}/fd_std_late": float(np.std(fd_trace[len(fd_trace) // 2 :])),
            f"{arm}/nn_distance": generative.nearest_neighbour_distance(samples, x_train, rng=exp.seed),
        }
        print(
            f"[e2] {arm:<15} final fd={summary[f'{arm}/fd']:.2f} (best {summary[f'{arm}/fd_best']:.2f})  "
            f"coverage={coverage['coverage']:.2f}  entropy={coverage['entropy']:.3f}  "
            f"largest class={coverage['max_class_share']:.2f}"
        )

    # -- what the arms say about each other ----------------------------------
    if "non_saturating" in exp["arms"] and "minimax" in exp["arms"]:
        summary["non_saturating_gain/fd"] = summary["minimax/fd"] - summary["non_saturating/fd"]
        summary["non_saturating_gain/entropy"] = (
            summary["non_saturating/mode_entropy"] - summary["minimax/mode_entropy"]
        )
        print(
            f"[e2] non-saturating over minimax: fd {summary['non_saturating_gain/fd']:+.2f}, "
            f"mode entropy {summary['non_saturating_gain/entropy']:+.3f}"
        )

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "training",
        plotting.plot_curves(
            curves,
            x=list(range(1, exp["epochs"] + 1)),
            title="The scoreboard, not the quality: neither curve says whether samples improved",
            xlabel="epoch",
            ylabel="loss",
        ),
    )
    checkpoints = [
        epoch for epoch in range(1, exp["epochs"] + 1) if epoch % exp["checkpoint_every"] == 0 or epoch == exp["epochs"]
    ]
    exp.log_figure(
        "quality_trace",
        plotting.plot_curves(
            trajectories,
            x=checkpoints,
            title="Sample quality during training — not monotone, unlike the losses",
            xlabel="epoch",
            ylabel="fd (lower is better)",
            marker="o",
        ),
    )
    exp.log_figure(
        "mode_coverage",
        plotting.plot_comparison(
            {arm: summary[f"{arm}/mode_entropy"] for arm in exp["arms"]},
            title="Mode coverage: normalized entropy over predicted classes (1.0 = even)",
            ylabel="normalized entropy",
        ),
    )
    exp.log_figure(
        "samples",
        plotting.plot_panel_rows(
            [samples_by_arm[arm][:8, 0] for arm in exp["arms"]],
            list(exp["arms"]),
            grayscale_first=False,
            title="Samples",
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
