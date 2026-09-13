"""The ladder specification: the single source of truth for this repo.

Every part and item below mirrors the "Broader ML — hands-on ladder" document.
The folder layout under ``parts/`` is generated from (and checked against) this
spec, so adding a rung means adding it here first.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class Lab:
    """A practical-lab reference attached to a part."""

    text: str
    url: str


@dataclass(frozen=True)
class Item:
    """One rung of the ladder."""

    id: str
    title: str
    module: str | None = None
    note: str | None = None

    @property
    def part_id(self) -> str:
        return self.id[0]

    @property
    def is_code(self) -> bool:
        """Setup rungs are repo infrastructure; every other rung is a module."""
        return self.module is not None


@dataclass(frozen=True)
class Part:
    """A group of rungs, one folder under ``parts/``."""

    id: str
    title: str
    items: tuple[Item, ...]
    package: str | None = None
    note: str | None = None
    lab: Lab | None = None
    tags: tuple[str, ...] = field(default_factory=tuple)


LADDER: tuple[Part, ...] = (
    Part(
        id="0",
        title="Setup",
        note="one shared repo, mirrors the RL ladder scaffolding",
        lab=Lab(
            "lightning-hydra-template — structural reference",
            "https://github.com/ashleve/lightning-hydra-template",
        ),
        items=(
            Item("01", "Repo scaffolding — uv-managed Python env, folder structure mirroring this ladder 1:1"),
            Item("02", "Shared eval/plotting harness package used across all parts"),
            Item("03", "W&B (or equivalent) run tracking wired in from part A onward"),
            Item("04", "Cargo workspace stub reserved for later Rust/Candle ports, tied to [[ignite-catalog]]"),
        ),
    ),
    Part(
        id="A",
        title="Classical ML from scratch",
        package="a_classical",
        note="numpy only — no sklearn",
        lab=Lab("Stanford CS229 problem sets", "https://cs229.stanford.edu/"),
        items=(
            Item(
                "A1",
                "Linear & logistic regression via batch/mini-batch gradient descent",
                "a1_linear_logistic_regression",
            ),
            Item(
                "A2",
                "Decision tree + random forest from scratch (Gini/entropy splits, bagging)",
                "a2_trees_and_forests",
            ),
            Item(
                "A3",
                "k-means and PCA from scratch, compared against each other on the same toy dataset",
                "a3_kmeans_and_pca",
            ),
            Item("A4", "Minimal gradient-boosting implementation, benchmarked against XGBoost", "a4_gradient_boosting"),
        ),
    ),
    Part(
        id="B",
        title="Deep learning foundations",
        package="b_deep_learning",
        lab=Lab("karpathy/micrograd — autodiff reference implementation", "https://github.com/karpathy/micrograd"),
        items=(
            Item("B1", "Scalar autodiff engine (micrograd-style) plus a tiny MLP trained on it", "b1_autodiff_engine"),
            Item(
                "B2",
                "MLP on MNIST — manual backprop, then repeat with PyTorch autograd and diff the gradients",
                "b2_mlp_mnist",
            ),
            Item("B3", "CNN on CIFAR-10 with a batchnorm/dropout ablation", "b3_cnn_cifar10"),
            Item("B4", "ResNet block implementation, trained on CIFAR-10/100", "b4_resnet_blocks"),
            Item(
                "B5", "Optimizer bake-off — SGD vs Adam vs AdamW, with LR-schedule comparison", "b5_optimizer_bakeoff"
            ),
        ),
    ),
    Part(
        id="C",
        title="Sequence models & attention",
        package="c_sequence_attention",
        lab=Lab("Annotated Transformer / nanoGPT", "https://nlp.seas.harvard.edu/annotated-transformer/"),
        items=(
            Item("C1", "Char-level RNN/LSTM language model", "c1_char_rnn"),
            Item("C2", "Seq2seq with attention on a toy translation task", "c2_seq2seq_attention"),
            Item(
                "C3",
                "Transformer from scratch (attention, positional encoding, causal masking)",
                "c3_transformer_from_scratch",
            ),
            Item("C4", "Fine-tune a small ViT on a custom image-classification task", "c4_vit_finetune"),
        ),
    ),
    Part(
        id="D",
        title="Self-supervised & representation learning",
        package="d_self_supervised",
        lab=Lab(
            "lightly-ai — contrastive-learning reference implementations",
            "https://github.com/lightly-ai/lightly",
        ),
        items=(
            Item("D1", "SimCLR-style contrastive pretraining on CIFAR-10", "d1_simclr_pretraining"),
            Item("D2", "Masked autoencoder (MAE) on a small image dataset", "d2_masked_autoencoder"),
            Item(
                "D3", "Linear probing of the learned representations against a supervised baseline", "d3_linear_probing"
            ),
        ),
    ),
    Part(
        id="E",
        title="Generative models",
        package="e_generative",
        lab=Lab(
            "lucidrains/denoising-diffusion-pytorch — diffusion reference",
            "https://github.com/lucidrains/denoising-diffusion-pytorch",
        ),
        items=(
            Item("E1", "VAE on MNIST/FashionMNIST", "e1_vae"),
            Item("E2", "DCGAN on a small image dataset", "e2_dcgan"),
            Item("E3", "DDPM diffusion model from scratch", "e3_ddpm"),
            Item(
                "E4",
                "Compare sample quality and training stability across VAE, GAN, and diffusion",
                "e4_generative_comparison",
            ),
        ),
    ),
    Part(
        id="F",
        title="Geometric & equivariant nets",
        package="f_geometric",
        note="ties directly to your algebraic topology / equivariant cohomology background",
        lab=Lab(
            "escnn — equivariant CNN library, and PyTorch Geometric tutorials",
            "https://github.com/QUVA-Lab/escnn",
        ),
        items=(
            Item(
                "F1",
                "Rotation-equivariant CNN on a rotated-MNIST task, compared against a standard CNN",
                "f1_equivariant_cnn",
            ),
            Item("F2", "GCN and GraphSAGE on a node-classification benchmark (Cora/PubMed)", "f2_gcn_graphsage"),
        ),
    ),
    Part(
        id="G",
        title="Scaling & systems",
        package="g_scaling_systems",
        note="bridges into ignite-catalog",
        lab=Lab(
            "PyTorch mixed-precision & profiler docs",
            "https://pytorch.org/docs/stable/notes/amp_examples.html",
        ),
        items=(
            Item(
                "G1",
                "Fit your own scaling-law curve (loss vs. params/data/compute) on a small model family",
                "g1_scaling_law_fit",
            ),
            Item(
                "G2",
                "Mixed-precision + gradient-accumulation training, profiled for throughput and memory",
                "g2_mixed_precision_profiling",
            ),
            Item(
                "G3",
                "Rust/Candle port of one transformer-block forward pass, as a comprehension exercise for [[ignite-catalog]]",
                "g3_candle_transformer_port",
                note="Python driver lives here; the Rust crate lives in rust/crates/ignite-port.",
            ),
        ),
    ),
)

PARTS_DIR_NAME = "parts"


def parts() -> tuple[Part, ...]:
    return LADDER


def iter_items() -> list[Item]:
    return [item for part in LADDER for item in part.items]


def get_part(part_id: str) -> Part:
    key = part_id.upper()
    for part in LADDER:
        if part.id == key:
            return part
    raise KeyError(f"no such part: {part_id!r}")


def get_item(item_id: str) -> Item:
    key = item_id.upper()
    for item in iter_items():
        if item.id == key:
            return item
    raise KeyError(f"no such ladder item: {item_id!r}")


def module_path(item: Item) -> str:
    """Importable module path for a code rung, e.g. ``parts.a_classical.a1_...``."""
    if item.module is None:
        raise ValueError(f"{item.id} is a setup rung and has no module")
    part = get_part(item.part_id)
    assert part.package is not None
    return f"{PARTS_DIR_NAME}.{part.package}.{item.module}"
