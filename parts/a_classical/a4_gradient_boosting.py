"""A4. Minimal gradient-boosting implementation, benchmarked against XGBoost

Lab reference: [Stanford CS229 problem sets](https://cs229.stanford.edu/)

Gradient boosting in one sentence: fit the next tree to the negative gradient of
the loss at the current prediction, and take a small step along it. For squared
loss that gradient is just the residual, which is why this rung fits in a page
once A2's regression tree exists.

The benchmark against XGBoost is the point of the rung — same depth, same
learning rate, same number of rounds — so the gap is the one the library's
engineering buys: histogram binning, second-order (Newton) leaf values,
regularization, and C++ speed. XGBoost is optional; without it the rung still
runs and reports the from-scratch numbers against the A2 baselines.

Run with ``ladder run A4`` (``--help`` for flags).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, datasets, experiment, metrics, plotting
from parts.a_classical.a2_trees_and_forests import DecisionTree, RandomForest

Array = NDArray[np.float64]


@dataclass
class GradientBoostingRegressor:
    """Squared-loss gradient boosting over A2's regression trees.

    ``subsample < 1`` makes it stochastic gradient boosting: each round fits its
    tree on a random subset, which decorrelates the trees the way bagging does.
    """

    n_estimators: int = 200
    learning_rate: float = 0.1
    max_depth: int = 3
    min_samples_leaf: int = 5
    subsample: float = 1.0
    rng: np.random.Generator | int | None = None
    base_prediction_: float = 0.0
    trees: list[DecisionTree] = field(default_factory=list)
    train_loss_: list[float] = field(default_factory=list)

    def fit(self, features: Array, targets: Array) -> GradientBoostingRegressor:
        generator = datasets.as_rng(self.rng)
        samples = np.asarray(features, dtype=np.float64)
        values = np.asarray(targets, dtype=np.float64)

        # The constant that minimises squared loss is the mean.
        self.base_prediction_ = float(values.mean())
        predictions = np.full(len(values), self.base_prediction_)
        self.trees = []
        self.train_loss_ = []

        for _ in range(self.n_estimators):
            residuals = values - predictions  # -1/2 · ∂(squared loss)/∂prediction
            if self.subsample < 1.0:
                size = max(1, round(self.subsample * len(values)))
                index = generator.choice(len(values), size=size, replace=False)
            else:
                index = np.arange(len(values))

            tree = DecisionTree(
                criterion="mse",
                max_depth=self.max_depth,
                min_samples_leaf=self.min_samples_leaf,
                rng=generator,
            ).fit(samples[index], residuals[index])
            predictions += self.learning_rate * tree.predict(samples)
            self.trees.append(tree)
            self.train_loss_.append(metrics.mse(values, predictions))
        return self

    def predict(self, features: Array, n_estimators: int | None = None) -> Array:
        samples = np.asarray(features, dtype=np.float64)
        predictions = np.full(len(samples), self.base_prediction_)
        rounds = len(self.trees) if n_estimators is None else n_estimators
        for tree in self.trees[:rounds]:
            predictions += self.learning_rate * tree.predict(samples)
        return predictions

    def staged_rmse(self, features: Array, targets: Array, *, every: int = 1) -> tuple[list[int], list[float]]:
        """Test error after each round — the curve that shows when to stop."""
        samples = np.asarray(features, dtype=np.float64)
        values = np.asarray(targets, dtype=np.float64)
        predictions = np.full(len(samples), self.base_prediction_)
        rounds, errors = [], []
        for index, tree in enumerate(self.trees, start=1):
            predictions += self.learning_rate * tree.predict(samples)
            if index % every == 0 or index == len(self.trees):
                rounds.append(index)
                errors.append(metrics.rmse(values, predictions))
        return rounds, errors


def _xgboost_baseline(
    x_train: Array, y_train: Array, x_test: Array, y_test: Array, *, rounds: int, lr: float, depth: int, seed: int
) -> dict[str, float] | None:
    """Same hyper-parameters as ours, or ``None`` when xgboost is not installed.

    The native Booster API rather than ``XGBRegressor``: the sklearn wrapper
    imports sklearn, and part A is meant to stay numpy-only.
    """
    try:
        import xgboost as xgb
    except ImportError:
        return None

    parameters = {
        "objective": "reg:squarederror",
        "eta": lr,
        "max_depth": depth,
        "lambda": 0.0,  # match our unregularized leaves as closely as possible
        "tree_method": "exact",
        "nthread": 1,
        "seed": seed,
        "verbosity": 0,
    }
    train_matrix = xgb.DMatrix(x_train, label=y_train)
    test_matrix = xgb.DMatrix(x_test)

    started = time.perf_counter()
    model = xgb.train(parameters, train_matrix, num_boost_round=rounds)
    elapsed = time.perf_counter() - started

    predictions = model.predict(test_matrix)
    return {
        "test_rmse": metrics.rmse(y_test, predictions),
        "test_r2": metrics.r2_score(y_test, predictions),
        "fit_seconds": elapsed,
        "version": xgb.__version__,
    }


@experiment(
    "a4_gradient_boosting",
    defaults={
        "samples": 800,
        "features": 10,
        "noise": 1.0,
        "rounds": 150,
        "lr": 0.1,
        "max_depth": 3,
        "min_samples_leaf": 5,
        "learning_rates": [0.03, 0.1, 0.3, 1.0],
        "subsample": 0.8,
        "eval_every": 5,
    },
    description="From-scratch squared-loss gradient boosting on Friedman #1, benchmarked against XGBoost.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Minimal gradient-boosting implementation, benchmarked against XGBoost."""
    rng = np.random.default_rng(exp.seed)
    features, targets = datasets.make_friedman1(exp["samples"], n_features=exp["features"], noise=exp["noise"], rng=rng)
    x_train, x_test, y_train, y_test = datasets.train_test_split(features, targets, rng=rng)

    # -- ours ---------------------------------------------------------------
    started = time.perf_counter()
    booster = GradientBoostingRegressor(
        n_estimators=exp["rounds"],
        learning_rate=exp["lr"],
        max_depth=exp["max_depth"],
        min_samples_leaf=exp["min_samples_leaf"],
        rng=rng,
    ).fit(x_train, y_train)
    ours_seconds = time.perf_counter() - started
    ours_rmse = metrics.rmse(y_test, booster.predict(x_test))
    print(
        f"[ours      ] test RMSE={ours_rmse:.4f}  R²={metrics.r2_score(y_test, booster.predict(x_test)):.4f}  fit={ours_seconds:.2f}s"
    )

    # -- stochastic variant -------------------------------------------------
    stochastic = GradientBoostingRegressor(
        n_estimators=exp["rounds"],
        learning_rate=exp["lr"],
        max_depth=exp["max_depth"],
        min_samples_leaf=exp["min_samples_leaf"],
        subsample=exp["subsample"],
        rng=rng,
    ).fit(x_train, y_train)
    stochastic_rmse = metrics.rmse(y_test, stochastic.predict(x_test))
    print(f"[ours+sub  ] test RMSE={stochastic_rmse:.4f}  (subsample={exp['subsample']})")

    # -- A2 baselines -------------------------------------------------------
    single_tree = DecisionTree(
        criterion="mse", max_depth=exp["max_depth"], min_samples_leaf=exp["min_samples_leaf"], rng=rng
    ).fit(x_train, y_train)
    deep_tree = DecisionTree(criterion="mse", max_depth=10, min_samples_leaf=exp["min_samples_leaf"], rng=rng).fit(
        x_train, y_train
    )
    forest = RandomForest(
        n_estimators=60,
        criterion="mse",
        max_depth=10,
        min_samples_leaf=exp["min_samples_leaf"],
        max_features="sqrt",
        rng=rng,
    ).fit(x_train, y_train)
    baselines = {
        f"tree (depth {exp['max_depth']})": metrics.rmse(y_test, single_tree.predict(x_test)),
        "tree (depth 10)": metrics.rmse(y_test, deep_tree.predict(x_test)),
        "random forest": metrics.rmse(y_test, forest.predict(x_test)),
        "boosting (ours)": ours_rmse,
    }
    for name, score in baselines.items():
        print(f"[baseline  ] {name:<18} test RMSE={score:.4f}")

    # -- XGBoost ------------------------------------------------------------
    reference = _xgboost_baseline(
        x_train, y_train, x_test, y_test, rounds=exp["rounds"], lr=exp["lr"], depth=exp["max_depth"], seed=exp.seed
    )
    if reference is None:
        print("[xgboost   ] not installed — `uv sync --extra boost` to run the benchmark")
    else:
        baselines["xgboost"] = reference["test_rmse"]
        print(
            f"[xgboost   ] test RMSE={reference['test_rmse']:.4f}  R²={reference['test_r2']:.4f}  "
            f"fit={reference['fit_seconds']:.2f}s  (v{reference['version']})  speedup={ours_seconds / reference['fit_seconds']:.0f}x"
        )

    # -- learning-rate sweep: shrinkage vs. rounds ---------------------------
    staged: dict[str, list[float]] = {}
    rounds_axis: list[int] = []
    lr_results: dict[float, float] = {}
    for rate in exp["learning_rates"]:
        model = GradientBoostingRegressor(
            n_estimators=exp["rounds"],
            learning_rate=rate,
            max_depth=exp["max_depth"],
            min_samples_leaf=exp["min_samples_leaf"],
            rng=rng,
        ).fit(x_train, y_train)
        rounds_axis, errors = model.staged_rmse(x_test, y_test, every=exp["eval_every"])
        staged[f"lr={rate}"] = errors
        lr_results[rate] = min(errors)
        best_round = rounds_axis[int(np.argmin(errors))]
        exp.log(
            {
                "lr": rate,
                "boosting/best_test_rmse": min(errors),
                "boosting/best_round": best_round,
                "boosting/final_test_rmse": errors[-1],
            }
        )
        print(f"[lr={rate:<5}] best test RMSE={min(errors):.4f} at round {best_round:>4}  final={errors[-1]:.4f}")

    # -- figures ------------------------------------------------------------
    exp.log_figure(
        "staged_test_error",
        plotting.plot_curves(
            staged, x=rounds_axis, title="Boosting — test RMSE by round", xlabel="boosting round", ylabel="test RMSE"
        ),
    )
    exp.log_figure(
        "train_loss",
        plotting.plot_curves(
            {"ours": booster.train_loss_, f"ours (subsample={exp['subsample']})": stochastic.train_loss_},
            title="Boosting — train MSE by round",
            xlabel="boosting round",
            ylabel="train MSE",
            logy=True,
        ),
    )
    exp.log_figure(
        "model_comparison",
        plotting.plot_comparison(baselines, title="Test RMSE — from-scratch models vs. XGBoost", ylabel="test RMSE"),
    )

    pooled = np.mean([tree.feature_importances() for tree in booster.trees], axis=0)
    exp.log_figure(
        "feature_importance",
        plotting.plot_comparison(
            {f"x{index}": float(value) for index, value in enumerate(pooled)},
            title="Pooled split-gain importance (x0-x4 are the real signal)",
            ylabel="share of gain",
        ),
    )

    summary = {
        "ours/test_rmse": ours_rmse,
        "ours/test_r2": metrics.r2_score(y_test, booster.predict(x_test)),
        "ours/fit_seconds": ours_seconds,
        "ours/stochastic_test_rmse": stochastic_rmse,
        "baseline/tree_test_rmse": baselines[f"tree (depth {exp['max_depth']})"],
        "baseline/forest_test_rmse": baselines["random forest"],
        "boosting/best_lr": float(min(lr_results, key=lambda rate: lr_results[rate])),
        "signal_share_x0_x4": float(pooled[:5].sum()),
    }
    if reference is not None:
        summary |= {
            "xgboost/test_rmse": reference["test_rmse"],
            "xgboost/fit_seconds": reference["fit_seconds"],
            "gap_vs_xgboost_rmse": ours_rmse - reference["test_rmse"],
            "slowdown_vs_xgboost": ours_seconds / reference["fit_seconds"],
        }
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
