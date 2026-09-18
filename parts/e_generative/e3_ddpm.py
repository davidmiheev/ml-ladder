"""E3. DDPM diffusion model from scratch

Lab reference: [lucidrains/denoising-diffusion-pytorch — diffusion reference](https://github.com/lucidrains/denoising-diffusion-pytorch)

E1 trained one network to compress and reconstruct; E2 trained two networks to
outwit each other. Diffusion does something stranger and much more stable: it
defines a **fixed** process that destroys an image over many steps, and trains a
single network to undo one step of it. There is no adversary, no bound to be
loose, no posterior to collapse — the loss is a plain regression onto noise.

What it buys in stability it pays for in sampling. A VAE or GAN produces an
image in one forward pass. A diffusion model runs its network once per denoising
step, so sampling is hundreds of times more expensive, and **the number of steps
is a dial trading quality against compute**. That dial is what this rung
measures, alongside the noise schedule, which decides how the destruction is
spread over those steps.

Run with ``ladder run E3`` (``--help`` for flags).
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, experiment, generative, plotting, torchkit, vision


def make_schedule(kind: str, steps: int) -> NDArray[np.float64]:
    r"""Betas for the forward process, linear or cosine.

    The forward process is fixed, not learned:

    $$q(x_t \mid x_{t-1}) = \mathcal{N}\!\left(x_t; \sqrt{1-\beta_t}\,x_{t-1}, \beta_t I\right),$$

    and because Gaussians compose, $t$ steps of it have a closed form that
    skips the intermediates entirely:

    $$q(x_t \mid x_0) = \mathcal{N}\!\left(x_t; \sqrt{\bar\alpha_t}\,x_0, (1-\bar\alpha_t) I\right),
    \qquad \bar\alpha_t = \prod_{s \le t}(1 - \beta_s).$$

    That closed form is why training is cheap: a step samples one random $t$ and
    jumps straight there, never simulating the chain.

    The **linear** schedule destroys information fast and spends most of its
    steps on nearly-pure noise, where there is little left to learn. The
    **cosine** schedule (Nichol & Dhariwal) sets $\bar\alpha_t$ to follow a
    squared cosine instead, keeping signal alive longer and distributing the
    difficulty more evenly across $t$.
    """
    if kind == "linear":
        # DDPM's 1e-4..0.02 range is specified *for T = 1000*. Used unscaled at
        # a shorter T it does not finish the job: at T = 400 it leaves
        # alpha_bar = 0.017, so the terminal distribution is not a standard
        # normal — while sampling starts from one. That mismatch is invisible
        # in the training loss and shows up only as bad samples, and it would
        # have made this rung's schedule comparison a measurement of the bug.
        # Scaling the range by 1000/T keeps the total destruction fixed.
        scale = 1000 / steps
        return np.linspace(1e-4 * scale, 0.02 * scale, steps, dtype=np.float64)
    if kind == "cosine":
        t = np.linspace(0, steps, steps + 1, dtype=np.float64) / steps
        offset = 0.008
        alpha_bar = np.cos((t + offset) / (1 + offset) * np.pi / 2) ** 2
        alpha_bar = alpha_bar / alpha_bar[0]
        # Clipped at 0.999: a beta of 1 annihilates the signal and makes the
        # posterior variance below degenerate.
        return np.clip(1 - alpha_bar[1:] / alpha_bar[:-1], 0, 0.999)
    raise ValueError(f"unknown schedule {kind!r}")


class Diffusion:
    """The fixed forward process and the sampler that reverses it."""

    def __init__(self, betas: NDArray[np.float64], *, device):
        torch = vision.require_torch()
        self.betas = torch.tensor(betas, dtype=torch.float32, device=device)
        self.alphas = 1.0 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.steps = len(betas)

    def add_noise(self, images, t, noise):
        r"""Jump to step $t$ in one shot, using the closed form above."""
        torch = vision.require_torch()
        alpha_bar = self.alpha_bar[t].view(-1, 1, 1, 1)
        return torch.sqrt(alpha_bar) * images + torch.sqrt(1 - alpha_bar) * noise

    @property
    def signal_to_noise(self) -> NDArray[np.float64]:
        r"""$\bar\alpha_t / (1 - \bar\alpha_t)$ — how much signal survives to step $t$."""
        alpha_bar = self.alpha_bar.cpu().numpy()
        return alpha_bar / np.maximum(1 - alpha_bar, 1e-12)

    def sample(self, model, *, shape, device, steps: int | None = None, generator=None):
        r"""Ancestral sampling, optionally strided to use fewer steps.

        Each reverse step is

        $$x_{t-1} = \frac{1}{\sqrt{\alpha_t}}\left(x_t - \frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\,\epsilon_\theta(x_t, t)\right) + \sigma_t z,$$

        which is the posterior mean with the network's noise prediction
        substituted for the unknown $\epsilon$. Striding takes a subsequence of
        timesteps and recomputes the coefficients from the *retained*
        $\bar\alpha$ values, which is what makes fewer steps a valid sampler
        rather than a broken one.
        """
        torch = vision.require_torch()
        chosen = (
            torch.arange(self.steps - 1, -1, -1, device=device)
            if steps is None or steps >= self.steps
            else torch.linspace(self.steps - 1, 0, steps, device=device).long()
        )
        x = torch.randn(shape, device=device, generator=generator)
        model.eval()
        with torch.no_grad():
            for index, t in enumerate(chosen):
                batch_t = torch.full((shape[0],), int(t), device=device, dtype=torch.long)
                predicted = model(x, batch_t)
                alpha_bar_t = self.alpha_bar[t]
                previous = chosen[index + 1] if index + 1 < len(chosen) else None
                alpha_bar_prev = self.alpha_bar[previous] if previous is not None else torch.tensor(1.0, device=device)
                # Effective alpha for this (possibly strided) transition.
                alpha = alpha_bar_t / alpha_bar_prev
                beta = 1 - alpha
                mean = (x - beta / torch.sqrt(1 - alpha_bar_t) * predicted) / torch.sqrt(alpha)
                if previous is not None:
                    variance = beta * (1 - alpha_bar_prev) / (1 - alpha_bar_t)
                    mean = mean + torch.sqrt(variance.clamp(min=0)) * torch.randn(
                        shape, device=device, generator=generator
                    )
                x = mean
        return x


def make_denoiser(*, channels: int, image_channels: int, time_dim: int = 64):
    """A small U-Net conditioned on the timestep.

    The timestep has to reach the network somehow, because one set of weights
    handles every noise level and needs to know which it is looking at. The
    standard answer, used here, is a sinusoidal embedding added to the feature
    maps — the same construction as C3's positional encoding, applied to a
    diffusion step instead of a sequence position.
    """
    torch = vision.require_torch()
    from torch import nn

    class TimeEmbedding(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.project = nn.Sequential(nn.Linear(time_dim, time_dim), nn.SiLU(), nn.Linear(time_dim, time_dim))

        def forward(self, t):
            half = time_dim // 2
            frequencies = torch.exp(
                -np.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / (half - 1)
            )
            angles = t.float()[:, None] * frequencies[None]
            return self.project(torch.cat([angles.sin(), angles.cos()], dim=1))

    class Block(nn.Module):
        def __init__(self, inputs: int, outputs: int) -> None:
            super().__init__()
            self.conv1 = nn.Conv2d(inputs, outputs, 3, padding=1)
            self.conv2 = nn.Conv2d(outputs, outputs, 3, padding=1)
            self.norm1 = nn.GroupNorm(8, outputs)
            self.norm2 = nn.GroupNorm(8, outputs)
            self.time = nn.Linear(time_dim, outputs)
            self.skip = nn.Conv2d(inputs, outputs, 1) if inputs != outputs else nn.Identity()

        def forward(self, x, t_embedding):
            hidden = torch.nn.functional.silu(self.norm1(self.conv1(x)))
            hidden = hidden + self.time(t_embedding)[:, :, None, None]
            hidden = torch.nn.functional.silu(self.norm2(self.conv2(hidden)))
            return hidden + self.skip(x)

    class UNet(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.embedding = TimeEmbedding()
            self.down1 = Block(image_channels, channels)
            self.down2 = Block(channels, channels * 2)
            self.middle = Block(channels * 2, channels * 2)
            self.up2 = Block(channels * 4, channels)  # concatenated skip
            self.up1 = Block(channels * 2, channels)
            self.out = nn.Conv2d(channels, image_channels, 1)
            self.pool = nn.AvgPool2d(2)

        def forward(self, x, t):
            embedding = self.embedding(t)
            first = self.down1(x, embedding)
            second = self.down2(self.pool(first), embedding)
            middle = self.middle(second, embedding)
            up = torch.nn.functional.interpolate(
                self.up2(torch.cat([middle, second], dim=1), embedding), scale_factor=2, mode="nearest"
            )
            return self.out(self.up1(torch.cat([up, first], dim=1), embedding))

    return UNet


@experiment(
    "e3_ddpm",
    defaults={
        "dataset": "mnist",
        "train_samples": 12000,
        "test_samples": 2000,
        "channels": 32,
        "train_steps": 400,
        "epochs": 16,
        "batch_size": 128,
        "lr": 2e-3,
        "schedules": ["linear", "cosine"],
        "reference_schedule": "cosine",
        "sample_steps": [10, 25, 50, 100, 400],
        "sample_count": 500,
        "scorer_epochs": 2,
    },
    description="A DDPM from scratch, sweeping the noise schedule and the quality-compute trade of sampling steps.",
)
def run(exp: Experiment) -> dict[str, float]:
    """DDPM diffusion model from scratch"""
    torch = vision.require_torch()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)
    print(f"[e3] device: {device}")

    raw_train, y_train = vision.load(exp["dataset"], split="train", limit=exp["train_samples"], rng=rng)
    raw_test, _ = vision.load(exp["dataset"], split="test", limit=exp["test_samples"], rng=rng)
    x_train = (raw_train.astype(np.float32) / 255.0).reshape(len(raw_train), -1, *raw_train.shape[1:3])
    x_test = (raw_test.astype(np.float32) / 255.0).reshape(len(raw_test), -1, *raw_test.shape[1:3])
    image_channels, image_size = x_train.shape[1], x_train.shape[2]
    # Diffusion assumes data centred on zero with roughly unit scale, because
    # the terminal distribution it denoises from is a standard normal.
    train_tensor = torch.from_numpy(x_train * 2 - 1).to(device)
    print(f"[e3] {len(x_train)} train images, {image_channels}x{image_size}x{image_size}, scaled to [-1, 1]")

    scorer = generative.train_feature_extractor(
        x_train, y_train, device=device, seed=exp.seed, epochs=exp["scorer_epochs"]
    )
    num_classes = int(y_train.max()) + 1
    real_reference = x_test[: exp["sample_count"]]

    summary: dict[str, float] = {}
    curves: dict[str, list[float]] = {}
    schedules_snr: dict[str, list[float]] = {}
    step_curves: dict[str, list[float]] = {}
    samples_by_schedule: dict[str, np.ndarray] = {}

    for schedule in exp["schedules"]:
        betas = make_schedule(schedule, exp["train_steps"])
        diffusion = Diffusion(betas, device=device)
        schedules_snr[schedule] = list(diffusion.signal_to_noise)

        torchkit.seed_torch(exp.seed)
        model = make_denoiser(channels=exp["channels"], image_channels=image_channels)().to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=1e-5)

        losses: list[float] = []
        for epoch in range(1, exp["epochs"] + 1):
            model.train()
            permutation = torch.randperm(len(train_tensor), device=device)
            epoch_losses = []
            for start in range(0, len(train_tensor), exp["batch_size"]):
                batch = train_tensor[permutation[start : start + exp["batch_size"]]]
                t = torch.randint(0, exp["train_steps"], (len(batch),), device=device)
                noise = torch.randn_like(batch)
                # The entire training objective: predict the noise that was
                # added. No adversary, no bound — a regression.
                loss = torch.nn.functional.mse_loss(model(diffusion.add_noise(batch, t, noise), t), noise)
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(loss.item())
            losses.append(float(np.mean(epoch_losses)))
            if epoch % max(exp["epochs"] // 4, 1) == 0:
                print(f"[e3] {schedule:<7} epoch {epoch:>3}  mse={losses[-1]:.4f}")
        curves[schedule] = losses
        summary[f"{schedule}/train_mse"] = losses[-1]
        summary[f"{schedule}/parameters"] = float(torchkit.count_parameters(model))

        # -- the dial: quality against sampling compute ----------------------
        shape = (exp["sample_count"], image_channels, image_size, image_size)
        for steps in exp["sample_steps"]:
            if steps > exp["train_steps"]:
                continue
            generator = torch.Generator(device=device).manual_seed(exp.seed)
            started = time.perf_counter()
            drawn = diffusion.sample(model, shape=shape, device=device, steps=steps, generator=generator)
            elapsed = time.perf_counter() - started
            samples = ((drawn + 1) / 2).clamp(0, 1).cpu().numpy()
            fd = generative.distribution_distance(scorer, real_reference, samples, device=device)
            summary[f"{schedule}/steps/{steps}/fd"] = fd
            summary[f"{schedule}/steps/{steps}/seconds"] = elapsed
            if schedule == exp["reference_schedule"]:
                step_curves.setdefault("fd", []).append(fd)
            if steps == max(exp["sample_steps"]):
                samples_by_schedule[schedule] = samples
                coverage = generative.class_coverage(scorer, samples, device=device, num_classes=num_classes)
                summary |= {
                    f"{schedule}/fd": fd,
                    f"{schedule}/coverage": coverage["coverage"],
                    f"{schedule}/mode_entropy": coverage["entropy"],
                    f"{schedule}/nn_distance": generative.nearest_neighbour_distance(samples, x_train, rng=exp.seed),
                }
            print(f"[e3] {schedule:<7} {steps:>4} steps  fd={fd:7.2f}  ({elapsed:5.1f}s)")

    # -- derived comparisons, guarded on what was swept ----------------------
    swept = [s for s in exp["sample_steps"] if s <= exp["train_steps"]]
    reference = exp["reference_schedule"]
    if reference in exp["schedules"] and len(swept) > 1:
        cheapest, fullest = min(swept), max(swept)
        summary["steps/quality_cost"] = (
            summary[f"{reference}/steps/{cheapest}/fd"] - summary[f"{reference}/steps/{fullest}/fd"]
        )
        summary["steps/speedup"] = (
            summary[f"{reference}/steps/{fullest}/seconds"] / summary[f"{reference}/steps/{cheapest}/seconds"]
        )
        # The knee: fewest steps still within 10% of the full-step score.
        target = summary[f"{reference}/steps/{fullest}/fd"] * 1.1
        affordable = [s for s in swept if summary[f"{reference}/steps/{s}/fd"] <= target]
        summary["steps/knee"] = float(min(affordable)) if affordable else float(fullest)
        print(
            f"[e3] {cheapest} steps is {summary['steps/speedup']:.0f}x faster than {fullest} and costs "
            f"{summary['steps/quality_cost']:+.2f} fd; within 10% of full quality from "
            f"{summary['steps/knee']:.0f} steps"
        )
    if len(exp["schedules"]) > 1:
        linear, cosine = summary.get("linear/fd"), summary.get("cosine/fd")
        if linear is not None and cosine is not None:
            summary["cosine_gain/fd"] = linear - cosine
            print(f"[e3] cosine over linear: {summary['cosine_gain/fd']:+.2f} fd")

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "training",
        plotting.plot_curves(
            curves,
            x=list(range(1, exp["epochs"] + 1)),
            title="Denoising MSE — a regression, and it just goes down",
            xlabel="epoch",
            ylabel="MSE",
            logy=True,
        ),
    )
    exp.log_figure(
        "schedules",
        plotting.plot_curves(
            schedules_snr,
            x=list(range(exp["train_steps"])),
            title="Signal-to-noise along the forward process",
            xlabel="diffusion step",
            ylabel="alpha_bar / (1 - alpha_bar)",
            logy=True,
        ),
    )
    if step_curves:
        exp.log_figure(
            "steps_vs_quality",
            plotting.plot_curves(
                step_curves,
                x=swept,
                title=f"Sampling steps against quality ({reference} schedule)",
                xlabel="denoising steps",
                ylabel="fd (lower is better)",
                marker="o",
                logx=True,
            ),
        )
    exp.log_figure(
        "samples",
        plotting.plot_panel_rows(
            [samples_by_schedule[s][:8, 0] for s in samples_by_schedule],
            list(samples_by_schedule),
            grayscale_first=False,
            title=f"Samples at {max(swept)} steps",
        ),
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
