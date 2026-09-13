"""A1. Linear & logistic regression via batch/mini-batch gradient descent

Lab reference: [Stanford CS229 problem sets](https://cs229.stanford.edu/)

Both models are the same three lines — a linear score, a loss, and the gradient
of that loss — so they share one optimizer. The rung's question is how the
update *schedule* (full batch vs. mini-batch vs. a single sample) changes the
path taken to the same optimum, and how close each lands to the closed-form
solution that linear regression happens to have.

Run with ``ladder run A1`` (``--help`` for flags).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, datasets, experiment, metrics, plotting

Array = NDArray[np.float64]


def batches(n_samples: int, batch_size: int, rng: np.random.Generator) -> list[Array]:
    """Index batches for one epoch. ``batch_size <= 0`` means one full batch."""
    if batch_size <= 0 or batch_size >= n_samples:
        return [np.arange(n_samples)]
    order = rng.permutation(n_samples)
    return [order[start : start + batch_size] for start in range(0, n_samples, batch_size)]


@dataclass
class LinearRegression:
    """Least squares by gradient descent, with the normal equation for reference.

    The bias is a separate parameter rather than a column of ones: it keeps the
    L2 penalty off the intercept, which is what you almost always want.
    """

    lr: float = 0.1
    epochs: int = 100
    batch_size: int = 0  # 0 = full batch
    l2: float = 0.0
    lr_decay: float = 0.0  # step size at epoch t is lr / (1 + lr_decay * t)
    weights: Array = field(default_factory=lambda: np.zeros(0))
    bias: float = 0.0
    history: list[float] = field(default_factory=list)

    def predict(self, features: Array) -> Array:
        return features @ self.weights + self.bias

    def loss(self, features: Array, targets: Array) -> float:
        residual = self.predict(features) - targets
        penalty = self.l2 * float(self.weights @ self.weights)
        return float(np.mean(residual**2) + penalty)

    def fit(self, features: Array, targets: Array, *, rng: np.random.Generator | None = None) -> LinearRegression:
        generator = datasets.as_rng(rng)
        self.weights = np.zeros(features.shape[1])
        self.bias = 0.0
        self.history = []

        for epoch in range(self.epochs):
            step = self.lr / (1 + self.lr_decay * epoch)
            for index in batches(len(features), self.batch_size, generator):
                batch_features, batch_targets = features[index], targets[index]
                residual = batch_features @ self.weights + self.bias - batch_targets
                grad_weights = 2 * batch_features.T @ residual / len(index) + 2 * self.l2 * self.weights
                grad_bias = 2 * float(residual.mean())
                self.weights -= step * grad_weights
                self.bias -= step * grad_bias
            self.history.append(self.loss(features, targets))
        return self

    def fit_normal_equation(self, features: Array, targets: Array) -> LinearRegression:
        """Closed form: ``w = (XᵀX + λI)⁻¹ Xᵀy`` on centred data."""
        mean_features = features.mean(axis=0)
        mean_target = float(targets.mean())
        centred = features - mean_features
        gram = centred.T @ centred + self.l2 * len(features) * np.eye(features.shape[1])
        self.weights = np.linalg.solve(gram, centred.T @ (targets - mean_target))
        self.bias = mean_target - float(mean_features @ self.weights)
        self.history = [self.loss(features, targets)]
        return self


@dataclass
class LogisticRegression:
    """Binary logistic regression by gradient descent on the log-loss.

    Same optimizer as above; only the link function and loss change.
    """

    lr: float = 0.5
    epochs: int = 100
    batch_size: int = 0
    l2: float = 0.0
    lr_decay: float = 0.0
    weights: Array = field(default_factory=lambda: np.zeros(0))
    bias: float = 0.0
    history: list[float] = field(default_factory=list)

    @staticmethod
    def _sigmoid(scores: Array) -> Array:
        """Stable logistic: exp never sees a large positive argument."""
        positive = scores >= 0
        result = np.empty_like(scores, dtype=np.float64)
        result[positive] = 1.0 / (1.0 + np.exp(-scores[positive]))
        exponent = np.exp(scores[~positive])
        result[~positive] = exponent / (1.0 + exponent)
        return result

    def predict_proba(self, features: Array) -> Array:
        return self._sigmoid(features @ self.weights + self.bias)

    def predict(self, features: Array) -> NDArray[np.int64]:
        return (self.predict_proba(features) >= 0.5).astype(np.int64)

    def loss(self, features: Array, targets: Array) -> float:
        probabilities = np.clip(self.predict_proba(features), 1e-12, 1 - 1e-12)
        log_loss = -np.mean(targets * np.log(probabilities) + (1 - targets) * np.log(1 - probabilities))
        return float(log_loss + self.l2 * float(self.weights @ self.weights))

    def fit(self, features: Array, targets: Array, *, rng: np.random.Generator | None = None) -> LogisticRegression:
        generator = datasets.as_rng(rng)
        self.weights = np.zeros(features.shape[1])
        self.bias = 0.0
        self.history = []

        for epoch in range(self.epochs):
            step = self.lr / (1 + self.lr_decay * epoch)
            for index in batches(len(features), self.batch_size, generator):
                batch_features, batch_targets = features[index], targets[index]
                error = self._sigmoid(batch_features @ self.weights + self.bias) - batch_targets
                grad_weights = batch_features.T @ error / len(index) + 2 * self.l2 * self.weights
                grad_bias = float(error.mean())
                self.weights -= step * grad_weights
                self.bias -= step * grad_bias
            self.history.append(self.loss(features, targets))
        return self


# Schedules compared in both halves of the rung: (label, batch_size, lr_decay).
# A full-batch gradient is exact, so its step size can stay constant; the noisy
# schedules need a decaying step or they rattle around the optimum forever.
SCHEDULES: tuple[tuple[str, int, float], ...] = (
    ("batch", 0, 0.0),
    ("mini-batch", 32, 0.05),
    ("sgd", 1, 0.5),
)


@experiment(
    "a1_linear_logistic_regression",
    defaults={
        "samples": 600,
        "features": 5,
        "noise": 0.3,
        "epochs": 60,
        "lr": 0.1,
        "logistic_lr": 0.5,
        "l2": 0.0,
        "decay": True,
    },
    description="Linear and logistic regression trained with three batch schedules, against the closed form.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Linear & logistic regression via batch/mini-batch gradient descent."""
    rng = np.random.default_rng(exp.seed)

    # -- linear -----------------------------------------------------------
    features, targets, true_weights = datasets.make_regression(
        exp["samples"], exp["features"], noise=exp["noise"], bias=1.5, rng=rng
    )
    x_train, x_test, y_train, y_test = datasets.train_test_split(features, targets, rng=rng)

    closed_form = LinearRegression(l2=exp["l2"]).fit_normal_equation(x_train, y_train)
    optimum = closed_form.loss(x_train, y_train)

    linear_curves: dict[str, list[float]] = {}
    linear_results: dict[str, dict[str, float]] = {}
    for label, batch_size, decay in SCHEDULES:
        model = LinearRegression(
            lr=exp["lr"],
            epochs=exp["epochs"],
            batch_size=batch_size,
            l2=exp["l2"],
            lr_decay=decay if exp["decay"] else 0.0,
        )
        model.fit(x_train, y_train, rng=rng)
        linear_curves[label] = model.history
        linear_results[label] = {
            "test_rmse": metrics.rmse(y_test, model.predict(x_test)),
            "test_r2": metrics.r2_score(y_test, model.predict(x_test)),
            "weight_error": float(np.linalg.norm(model.weights - true_weights)),
            "excess_train_loss": model.history[-1] - optimum,
        }
        exp.log({f"linear/{label}/{key}": value for key, value in linear_results[label].items()})
        print(
            f"[linear:{label:<10}] rmse={linear_results[label]['test_rmse']:.4f}  |w-w*|={linear_results[label]['weight_error']:.4f}"
        )

    print(
        f"[linear:{'normal-eq':<10}] rmse={metrics.rmse(y_test, closed_form.predict(x_test)):.4f}  "
        f"|w-w*|={np.linalg.norm(closed_form.weights - true_weights):.4f}  (reference)"
    )

    # -- logistic ---------------------------------------------------------
    points, labels, _ = datasets.make_blobs(exp["samples"], centers=2, cluster_std=2.2, center_box=(-3.0, 3.0), rng=rng)
    p_train, p_test, l_train, l_test = datasets.train_test_split(points, labels, rng=rng)
    p_train, p_test = datasets.standardize(p_train, p_test)

    logistic_curves: dict[str, list[float]] = {}
    logistic_results: dict[str, dict[str, float]] = {}
    best_model: LogisticRegression | None = None
    best_accuracy = -1.0
    for label, batch_size, decay in SCHEDULES:
        model = LogisticRegression(
            lr=exp["logistic_lr"],
            epochs=exp["epochs"],
            batch_size=batch_size,
            l2=exp["l2"],
            lr_decay=decay if exp["decay"] else 0.0,
        )
        model.fit(p_train, l_train.astype(np.float64), rng=rng)
        logistic_curves[label] = model.history
        logistic_results[label] = {
            "test_accuracy": metrics.accuracy(l_test, model.predict(p_test)),
            "test_log_loss": model.loss(p_test, l_test.astype(np.float64)),
        }
        exp.log({f"logistic/{label}/{key}": value for key, value in logistic_results[label].items()})
        print(
            f"[logistic:{label:<8}] accuracy={logistic_results[label]['test_accuracy']:.4f}  log-loss={logistic_results[label]['test_log_loss']:.4f}"
        )
        if logistic_results[label]["test_accuracy"] > best_accuracy:
            best_accuracy, best_model = logistic_results[label]["test_accuracy"], model
    assert best_model is not None

    # -- figures ----------------------------------------------------------
    exp.log_figure(
        "linear_convergence",
        plotting.plot_curves(
            linear_curves, title="Linear regression — train MSE by schedule", xlabel="epoch", ylabel="MSE", logy=True
        ),
    )
    exp.log_figure(
        "logistic_convergence",
        plotting.plot_curves(
            logistic_curves, title="Logistic regression — train log-loss by schedule", xlabel="epoch", ylabel="log-loss"
        ),
    )
    exp.log_figure(
        "logistic_boundary",
        plotting.plot_decision_boundary(best_model.predict, p_test, l_test, title="Logistic regression — test set"),
    )
    exp.log_figure(
        "schedule_comparison",
        plotting.plot_comparison(
            {label: result["test_rmse"] for label, result in linear_results.items()},
            title="Linear regression — test RMSE by schedule",
            ylabel="RMSE",
        ),
    )

    summary = {
        f"linear/{label}/{key}": value for label, result in linear_results.items() for key, value in result.items()
    }
    summary |= {
        f"logistic/{label}/{key}": value for label, result in logistic_results.items() for key, value in result.items()
    }
    summary["linear/normal_eq/test_rmse"] = metrics.rmse(y_test, closed_form.predict(x_test))
    summary["linear/normal_eq/weight_error"] = float(np.linalg.norm(closed_form.weights - true_weights))
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
