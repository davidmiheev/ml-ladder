"""End-to-end harness demo — the shape every rung under ``parts/`` follows.

Fits a closed-form linear regression on synthetic data, logs a metric per
iteration, records headline numbers and saves two figures. No ML content worth
reading here; it exists so the harness can be exercised without touching a rung.

    uv run python examples/harness_demo.py --noise 0.3 --iters 40
"""

from __future__ import annotations

import numpy as np

from mlladder import Experiment, experiment, metrics, plotting


@experiment(
    "harness_demo",
    defaults={"samples": 256, "features": 3, "noise": 0.2, "iters": 30, "lr": 0.1},
    description="Synthetic linear regression that exercises the shared harness.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Gradient descent on a known linear model, so the numbers are checkable."""
    rng = np.random.default_rng(exp.seed)
    true_weights = rng.normal(size=exp["features"])
    features = rng.normal(size=(exp["samples"], exp["features"]))
    targets = features @ true_weights + rng.normal(scale=exp["noise"], size=exp["samples"])

    weights = np.zeros(exp["features"])
    losses = []
    for step in range(exp["iters"]):
        predictions = features @ weights
        gradient = 2 * features.T @ (predictions - targets) / len(targets)
        weights -= exp["lr"] * gradient
        loss = metrics.mse(targets, predictions)
        losses.append(loss)
        exp.log({"train/mse": loss, "weight_error": float(np.linalg.norm(weights - true_weights))}, step=step)

    predictions = features @ weights
    exp.log_figure(
        "loss_curve", plotting.plot_curves({"train/mse": losses}, title="Harness demo", ylabel="MSE", logy=True)
    )
    exp.log_figure(
        "fit",
        plotting.plot_comparison(
            {"rmse": metrics.rmse(targets, predictions), "noise": exp["noise"]}, title="Fit vs. noise floor"
        ),
    )

    return {
        "rmse": metrics.rmse(targets, predictions),
        "r2": metrics.r2_score(targets, predictions),
        "weight_error": float(np.linalg.norm(weights - true_weights)),
    }


main = run

if __name__ == "__main__":
    raise SystemExit(main())
