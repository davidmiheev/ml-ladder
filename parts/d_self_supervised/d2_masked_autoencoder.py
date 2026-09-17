"""D2. Masked autoencoder (MAE) on a small image dataset

Lab reference: [lightly-ai — contrastive-learning reference implementations](https://github.com/lightly-ai/lightly)

D1's contrastive objective needs a carefully designed augmentation pipeline to
say what "same image" means. MAE needs none of that: hide most of the patches
and reconstruct them. The supervision comes from the image's own redundancy, so
the design question moves from *which augmentations* to *how much to hide* —
and the answer (a lot, ~75%) is the paper's headline, so this rung sweeps it.

The other half of MAE's idea is an efficiency trick with a conceptual payoff:
the encoder sees **only the visible patches**, so masking 75% makes pretraining
roughly 4x cheaper rather than 4x more wasteful.

Run with ``ladder run D2`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np

from mlladder import Experiment, experiment, plotting, torchkit, vision
from parts.c_sequence_attention.c3_transformer_from_scratch import make_transformer
from parts.d_self_supervised.d1_simclr_pretraining import linear_probe


def patchify(images, patch: int):
    """``(N, C, H, W)`` to ``(N, num_patches, C*patch*patch)``, row-major."""
    vision.require_torch()
    batch, channels, height, width = images.shape
    if height % patch or width % patch:
        raise ValueError(f"patch {patch} does not tile a {height}x{width} image")
    x = images.reshape(batch, channels, height // patch, patch, width // patch, patch)
    x = x.permute(0, 2, 4, 1, 3, 5).reshape(batch, (height // patch) * (width // patch), channels * patch * patch)
    return x


def unpatchify(patches, *, patch: int, channels: int, height: int, width: int):
    """The inverse of :func:`patchify`, for drawing reconstructions."""
    batch = len(patches)
    rows, columns = height // patch, width // patch
    x = patches.reshape(batch, rows, columns, channels, patch, patch)
    x = x.permute(0, 3, 1, 4, 2, 5).reshape(batch, channels, height, width)
    return x


def make_mae(
    *, num_patches: int, patch_dim: int, width: int = 128, depth: int = 4, heads: int = 4, decoder_depth: int = 2
):
    """A small MAE: ViT encoder over visible patches, light decoder over all of them.

    Reuses C3's transformer block, so this is the same attention implementation
    the language model was built from — only the tokens changed.
    """
    torch = vision.require_torch()
    from torch import nn

    _, sinusoidal_encoding = make_transformer()
    # C3's Block lives inside make_transformer's closure; rebuild one here by
    # asking for a tiny model and borrowing its block class.
    CharTransformer, _ = make_transformer()
    block_class = type(CharTransformer(2, width=width, heads=heads, depth=1, block_size=8).blocks[0])

    class MaskedAutoencoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed = nn.Linear(patch_dim, width)
            self.register_buffer("positions", sinusoidal_encoding(num_patches, width), persistent=False)
            self.encoder = nn.ModuleList([block_class(width, heads) for _ in range(depth)])
            self.encoder_norm = nn.LayerNorm(width)

            self.mask_token = nn.Parameter(torch.zeros(1, 1, width))
            nn.init.normal_(self.mask_token, std=0.02)
            self.decoder = nn.ModuleList([block_class(width, heads) for _ in range(decoder_depth)])
            self.decoder_norm = nn.LayerNorm(width)
            self.decoder_head = nn.Linear(width, patch_dim)

        def encode_visible(self, patches, keep_index):
            """Embed and run the encoder on the kept patches only."""
            tokens = self.patch_embed(patches) + self.positions
            visible = torch.gather(tokens, 1, keep_index[..., None].expand(-1, -1, tokens.shape[-1]))
            for block in self.encoder:
                # Full (non-causal) attention: an image has no arrow of time.
                visible, _ = block(visible)
            return self.encoder_norm(visible)

        def forward(self, patches, keep_index, mask):
            batch, total, _ = patches.shape
            encoded = self.encode_visible(patches, keep_index)

            # Scatter the encoded patches back into place and fill the holes
            # with a learned mask token, then decode every position.
            full = self.mask_token.expand(batch, total, -1).clone()
            full = full.scatter(1, keep_index[..., None].expand(-1, -1, full.shape[-1]), encoded)
            full = full + self.positions
            for block in self.decoder:
                full, _ = block(full)
            prediction = self.decoder_head(self.decoder_norm(full))

            # Loss on masked patches only — reconstructing what was visible is
            # copying, not learning.
            error = (prediction - patches) ** 2
            loss = (error.mean(dim=-1) * mask).sum() / mask.sum().clamp(min=1)
            return loss, prediction

        def represent(self, images, *, patch: int):
            """Mean-pooled encoder features over *all* patches, for probing."""
            patches = patchify(images, patch)
            index = torch.arange(patches.shape[1], device=images.device).expand(len(images), -1)
            return self.encode_visible(patches, index).mean(dim=1)

    return MaskedAutoencoder()


def random_mask(batch: int, num_patches: int, mask_ratio: float, *, device, generator=None):
    """Per-sample random masking; returns ``(keep_index, mask)``.

    ``mask`` is 1 for hidden patches — the ones the loss is computed on.
    """
    torch = vision.require_torch()
    keep = max(1, round(num_patches * (1 - mask_ratio)))
    noise = torch.rand(batch, num_patches, device=device, generator=generator)
    order = noise.argsort(dim=1)
    keep_index = order[:, :keep]

    mask = torch.ones(batch, num_patches, device=device)
    mask.scatter_(1, keep_index, 0.0)
    return keep_index, mask


@experiment(
    "d2_masked_autoencoder",
    defaults={
        "pretrain_samples": 8000,
        "probe_samples": 4000,
        "test_samples": 2000,
        "patch": 4,
        "width": 128,
        "depth": 4,
        "heads": 4,
        "decoder_depth": 2,
        "steps": 600,
        "batch_size": 128,
        "lr": 1e-3,
        "mask_ratio": 0.75,
        "mask_ratios": [0.25, 0.5, 0.75, 0.9],
        "probe_epochs": 12,
    },
    description="A masked autoencoder on CIFAR-10, swept over mask ratio and scored by linear probe.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Masked autoencoder (MAE) on a small image dataset."""
    torch = vision.require_torch()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)
    print(f"[d2] device: {device}")

    raw_pretrain, _ = vision.load("cifar10", split="train", limit=exp["pretrain_samples"], rng=rng)
    raw_probe, y_probe = vision.load("cifar10", split="train", limit=exp["probe_samples"], rng=rng)
    raw_test, y_test = vision.load("cifar10", split="test", limit=exp["test_samples"], rng=rng)
    x_pretrain = vision.normalize(raw_pretrain, "cifar10")
    x_probe = vision.normalize(raw_probe, "cifar10")
    x_test = vision.normalize(raw_test, "cifar10")

    _, channels, height, width = x_pretrain.shape
    patch = exp["patch"]
    num_patches = (height // patch) * (width // patch)
    patch_dim = channels * patch * patch
    print(
        f"[d2] {len(x_pretrain)} unlabelled images, {num_patches} patches of {patch}x{patch} ({patch_dim} values each)"
    )

    summary: dict[str, float] = {}
    curves: dict[str, list[float]] = {}
    probe_scores: list[float] = []
    best_model, best_accuracy, best_ratio = None, 0.0, 0.0

    for mask_ratio in exp["mask_ratios"]:
        torchkit.seed_torch(exp.seed)
        model = make_mae(
            num_patches=num_patches,
            patch_dim=patch_dim,
            width=exp["width"],
            depth=exp["depth"],
            heads=exp["heads"],
            decoder_depth=exp["decoder_depth"],
        ).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=exp["lr"], weight_decay=0.05)
        batch_rng = np.random.default_rng(exp.seed)
        tensor = torch.from_numpy(x_pretrain)

        losses = []
        for step in range(1, exp["steps"] + 1):
            index = batch_rng.integers(0, len(tensor), size=exp["batch_size"])
            images = tensor[index].to(device)
            patches = patchify(images, patch)
            keep_index, mask = random_mask(len(images), num_patches, mask_ratio, device=device)

            model.train()
            loss, _ = model(patches, keep_index, mask)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if step % max(1, exp["steps"] // 3) == 0:
                print(
                    f"[d2] mask={mask_ratio:<5} step {step:>4}/{exp['steps']}  reconstruction MSE={np.mean(losses[-50:]):.4f}"
                )

        class EncoderView(torch.nn.Module):
            """Adapter so the D1 probe can read MAE features."""

            def __init__(self, mae) -> None:
                super().__init__()
                self.mae = mae

            def forward(self, images):
                return self.mae.represent(images, patch=patch)

        accuracy = linear_probe(
            EncoderView(model).to(device), x_probe, y_probe, x_test, y_test, device=device, epochs=exp["probe_epochs"]
        )
        probe_scores.append(accuracy)
        curves[f"mask={mask_ratio}"] = losses
        summary |= {
            f"mask/{mask_ratio}/probe_accuracy": accuracy,
            f"mask/{mask_ratio}/final_mse": float(np.mean(losses[-50:])),
        }
        print(f"[d2] mask={mask_ratio:<5} probe accuracy={accuracy:.4f}  final MSE={np.mean(losses[-50:]):.4f}")
        if accuracy > best_accuracy:
            best_model, best_accuracy, best_ratio = model, accuracy, mask_ratio

    # -- what did it actually reconstruct? -----------------------------------
    if best_model is not None:
        best_model.eval()
        images = torch.from_numpy(x_test[:8]).to(device)
        patches = patchify(images, patch)
        keep_index, mask = random_mask(len(images), num_patches, best_ratio, device=device)
        with torch.no_grad():
            _, prediction = best_model(patches, keep_index, mask)

        visible = patches * (1 - mask)[..., None]
        filled = patches * (1 - mask)[..., None] + prediction * mask[..., None]
        grid = torch.cat(
            [
                unpatchify(patches, patch=patch, channels=channels, height=height, width=width),
                unpatchify(visible, patch=patch, channels=channels, height=height, width=width),
                unpatchify(filled, patch=patch, channels=channels, height=height, width=width),
            ]
        )
        # Undo normalization for display.
        spec = vision.SPECS["cifar10"]
        mean = torch.tensor(spec.mean, device=device)[None, :, None, None]
        std = torch.tensor(spec.std, device=device)[None, :, None, None]
        grid = (grid * std + mean).clamp(0, 1).cpu().numpy()
        exp.log_figure(
            "reconstructions",
            plotting.plot_image_grid(grid, nrow=8, title=f"original / masked ({best_ratio:.0%}) / reconstructed"),
        )

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "reconstruction_loss",
        plotting.plot_curves(
            curves, title="Masked reconstruction MSE", xlabel="step", ylabel="MSE on masked patches", logy=True
        ),
    )
    exp.log_figure(
        "mask_ratio_ablation",
        plotting.plot_curves(
            {"linear probe": probe_scores},
            x=exp["mask_ratios"],
            title="Probe accuracy vs. mask ratio",
            xlabel="fraction of patches hidden",
            ylabel="test accuracy",
            marker="o",
        ),
    )

    summary["best/probe_accuracy"] = best_accuracy
    summary["best/mask_ratio"] = float(best_ratio)
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
