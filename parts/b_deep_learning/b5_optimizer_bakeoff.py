"""B5. Optimizer bake-off — SGD vs Adam vs AdamW, with LR-schedule comparison

Lab reference: [karpathy/micrograd — autodiff reference implementation](https://github.com/karpathy/micrograd)

Three questions, in order of how much they matter in practice:

1. Which optimizer wins at *its own* best learning rate?
2. How wide is each one's usable learning-rate band? (This is the real reason
   Adam is a default: it is hard to misconfigure, not that it converges lower.)
3. Does decoupling weight decay — the single line separating Adam from AdamW —
   actually change anything measurable?

FashionMNIST with an MLP, so the whole grid fits in a couple of minutes on CPU
and the comparison is about optimizers rather than architecture.

Run with ``ladder run B5`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np

from mlladder import Experiment, experiment, plotting, torchkit, vision


def build_mlp(input_size: int, hidden: list[int], classes: int = 10):
    vision.require_torch()
    from torch import nn

    sizes = [input_size, *hidden]
    layers: list[nn.Module] = []
    for index in range(len(sizes) - 1):
        layers += [nn.Linear(sizes[index], sizes[index + 1]), nn.ReLU(inplace=True)]
    layers.append(nn.Linear(sizes[-1], classes))
    return nn.Sequential(*layers)


def make_optimizer(name: str, parameters, lr: float, weight_decay: float):
    """The four arms. Note where the decay goes in each case."""
    torch = vision.require_torch()
    if name == "sgd":
        return torch.optim.SGD(parameters, lr=lr)
    if name == "sgd+momentum":
        return torch.optim.SGD(parameters, lr=lr, momentum=0.9)
    if name == "adam":
        # weight_decay here is classic L2: added to the gradient, and therefore
        # divided by sqrt(v) along with everything else.
        return torch.optim.Adam(parameters, lr=lr, weight_decay=weight_decay)
    if name == "adamw":
        # Decoupled: applied straight to the parameter, untouched by v.
        return torch.optim.AdamW(parameters, lr=lr, weight_decay=weight_decay)
    raise ValueError(f"unknown optimizer {name!r}")


OPTIMIZERS = ("sgd", "sgd+momentum", "adam", "adamw")
# Defaults that are each near their own optimum, so the bake-off is not rigged.
BEST_LR = {"sgd": 0.1, "sgd+momentum": 0.05, "adam": 1e-3, "adamw": 1e-3}


def weight_norm(model) -> float:
    torch = vision.require_torch()
    with torch.no_grad():
        return float(torch.sqrt(sum((parameter**2).sum() for parameter in model.parameters())))


@experiment(
    "b5_optimizer_bakeoff",
    defaults={
        "train_samples": 15000,
        "test_samples": 5000,
        "hidden": [256, 128],
        "epochs": 10,
        "batch_size": 128,
        "weight_decay": 0.05,
        "sweep_lrs": [1e-4, 1e-3, 1e-2, 1e-1, 0.5],
        "sweep_epochs": 3,
    },
    description="SGD/momentum/Adam/AdamW on FashionMNIST: best-LR bake-off, LR sensitivity, and decoupled decay.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Optimizer bake-off — SGD vs Adam vs AdamW, with LR-schedule comparison."""
    torch = vision.require_torch()
    rng = np.random.default_rng(exp.seed)

    raw_train, y_train = vision.load("fashion_mnist", split="train", limit=exp["train_samples"], rng=rng)
    raw_test, y_test = vision.load("fashion_mnist", split="test", limit=exp["test_samples"], rng=rng)
    x_train = vision.flatten(vision.normalize(raw_train, "fashion_mnist"))
    x_test = vision.flatten(vision.normalize(raw_test, "fashion_mnist"))
    print(f"[b5] FashionMNIST {len(x_train)} train / {len(x_test)} test")

    def train_arm(optimizer_name: str, lr: float, *, epochs: int, cosine: bool) -> tuple[dict[str, list[float]], float]:
        torchkit.seed_torch(exp.seed)
        model = build_mlp(x_train.shape[1], exp["hidden"])
        train_loader, test_loader = torchkit.loaders(
            x_train, y_train, x_test, y_test, batch_size=exp["batch_size"], seed=exp.seed
        )
        optimizer = make_optimizer(optimizer_name, model.parameters(), lr, exp["weight_decay"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs) if cosine else None
        history = torchkit.fit(model, train_loader, test_loader, optimizer, epochs=epochs, scheduler=scheduler)
        return history, weight_norm(model)

    # -- 1. the bake-off, each optimizer at its own best LR, two schedules ---
    curves: dict[str, list[float]] = {}
    summary: dict[str, float] = {}
    norms: dict[str, float] = {}
    for name in OPTIMIZERS:
        for cosine in (False, True):
            arm = f"{name} ({'cosine' if cosine else 'constant'})"
            history, norm = train_arm(name, BEST_LR[name], epochs=exp["epochs"], cosine=cosine)
            curves[arm] = history["test_accuracy"]
            norms[arm] = norm
            summary[f"{arm}/test_accuracy"] = history["test_accuracy"][-1]
            summary[f"{arm}/best_test_accuracy"] = max(history["test_accuracy"])
            print(
                f"[b5] {arm:<24} lr={BEST_LR[name]:<6} final={history['test_accuracy'][-1]:.4f}  "
                f"best={max(history['test_accuracy']):.4f}  |w|={norm:.1f}"
            )
            exp.log({f"{arm}/test_accuracy": history["test_accuracy"][-1]})

    # -- 2. how forgiving is each optimizer about the learning rate? ---------
    sensitivity: dict[str, list[float]] = {}
    for name in OPTIMIZERS:
        scores = []
        for lr in exp["sweep_lrs"]:
            history, _ = train_arm(name, lr, epochs=exp["sweep_epochs"], cosine=False)
            scores.append(history["test_accuracy"][-1])
        sensitivity[name] = scores
        usable = sum(1 for score in scores if score > 0.8 * max(scores))
        summary[f"sweep/{name}/best"] = max(scores)
        summary[f"sweep/{name}/usable_lrs"] = float(usable)
        print(
            f"[b5] {name:<14} lr sweep "
            + "  ".join(f"{lr:g}:{score:.3f}" for lr, score in zip(exp["sweep_lrs"], scores, strict=True))
        )

    # -- 3. decoupled decay: same lambda, different arithmetic ---------------
    adam_arm, adamw_arm = "adam (cosine)", "adamw (cosine)"
    decay_gap = summary[f"{adamw_arm}/test_accuracy"] - summary[f"{adam_arm}/test_accuracy"]
    norm_ratio = norms[adam_arm] / max(norms[adamw_arm], 1e-12)
    print(f"[b5] decoupled decay: AdamW - Adam = {decay_gap:+.4f} accuracy, |w| ratio Adam/AdamW = {norm_ratio:.2f}")

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "bakeoff",
        plotting.plot_curves(
            curves,
            x=list(range(1, exp["epochs"] + 1)),
            title="FashionMNIST MLP — test accuracy, each optimizer at its best LR",
            xlabel="epoch",
            ylabel="test accuracy",
        ),
    )
    exp.log_figure(
        "lr_sensitivity",
        plotting.plot_curves(
            sensitivity,
            x=exp["sweep_lrs"],
            title=f"Learning-rate sensitivity ({exp['sweep_epochs']} epochs each)",
            xlabel="learning rate",
            ylabel="test accuracy",
            logx=True,
            marker="o",
        ),
    )
    exp.log_figure(
        "final_accuracy",
        plotting.plot_comparison(
            {arm: summary[f"{arm}/test_accuracy"] for arm in curves},
            title="Final test accuracy by optimizer and schedule",
            ylabel="test accuracy",
        ),
    )
    exp.log_figure(
        "weight_norms",
        plotting.plot_comparison(norms, title=f"Final ||w|| (weight decay = {exp['weight_decay']})", ylabel="L2 norm"),
    )

    best_arm = max(curves, key=lambda arm: summary[f"{arm}/test_accuracy"])
    summary["best_arm_accuracy"] = summary[f"{best_arm}/test_accuracy"]
    summary["cosine_minus_constant/adamw"] = (
        summary["adamw (cosine)/test_accuracy"] - summary["adamw (constant)/test_accuracy"]
    )
    summary["cosine_minus_constant/sgd+momentum"] = (
        summary["sgd+momentum (cosine)/test_accuracy"] - summary["sgd+momentum (constant)/test_accuracy"]
    )
    summary["adamw_minus_adam"] = decay_gap
    summary["weight_norm_ratio_adam_over_adamw"] = norm_ratio
    print(f"[b5] best arm: {best_arm} at {summary['best_arm_accuracy']:.4f}")
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
