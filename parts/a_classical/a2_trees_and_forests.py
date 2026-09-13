"""A2. Decision tree + random forest from scratch (Gini/entropy splits, bagging)

Lab reference: [Stanford CS229 problem sets](https://cs229.stanford.edu/)

One tree class covers both tasks — classification (Gini or entropy) and
regression (variance reduction) — because the split search is identical once
you can score a candidate partition. A4 reuses the regression side as its weak
learner, so the tree is written to be embedded, not just run.

Splits are found by a vectorized scan: sort each candidate feature once, take
cumulative statistics, and score every threshold at the same time.

Run with ``ladder run A2`` (``--help`` for flags).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, datasets, experiment, metrics, plotting

Array = NDArray[np.float64]
Labels = NDArray[np.int64]

CLASSIFICATION_CRITERIA = ("gini", "entropy")
REGRESSION_CRITERIA = ("mse",)


@dataclass
class Node:
    """Either a leaf (``value`` set) or an internal split."""

    value: Array | float | None = None
    feature: int | None = None
    threshold: float = 0.0
    left: Node | None = None
    right: Node | None = None
    n_samples: int = 0
    impurity: float = 0.0

    @property
    def is_leaf(self) -> bool:
        return self.feature is None


def _gini(counts: Array) -> Array:
    """Gini impurity for rows of class counts."""
    totals = counts.sum(axis=-1, keepdims=True)
    probabilities = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    return 1.0 - np.sum(probabilities**2, axis=-1)


def _entropy(counts: Array) -> Array:
    """Shannon entropy (bits) for rows of class counts."""
    totals = counts.sum(axis=-1, keepdims=True)
    probabilities = np.divide(counts, totals, out=np.zeros_like(counts), where=totals > 0)
    logs = np.log2(probabilities, out=np.zeros_like(probabilities), where=probabilities > 0)
    return -np.sum(probabilities * logs, axis=-1)


@dataclass
class DecisionTree:
    """CART, grown greedily and pruned only by the stopping rules.

    ``max_features`` samples a subset of features at each split — the second
    source of decorrelation a random forest needs on top of bagging.
    """

    criterion: str = "gini"
    max_depth: int = 8
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: int | str | None = None  # None/"all", "sqrt", "log2", or a count
    rng: np.random.Generator | int | None = None
    root: Node | None = None
    classes: Labels = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    n_features_: int = 0

    @property
    def is_regressor(self) -> bool:
        return self.criterion in REGRESSION_CRITERIA

    # -- fitting ----------------------------------------------------------
    def fit(self, features: Array, targets: Array) -> DecisionTree:
        if self.criterion not in CLASSIFICATION_CRITERIA + REGRESSION_CRITERIA:
            raise ValueError(f"unknown criterion {self.criterion!r}")
        self._generator = datasets.as_rng(self.rng)
        self.n_features_ = features.shape[1]
        if self.is_regressor:
            encoded = np.asarray(targets, dtype=np.float64)
        else:
            self.classes = np.unique(np.asarray(targets, dtype=np.int64))
            lookup = {label: index for index, label in enumerate(self.classes)}
            encoded = np.array([lookup[int(label)] for label in targets], dtype=np.int64)
        self.root = self._grow(np.asarray(features, dtype=np.float64), encoded, depth=0)
        return self

    def _leaf(self, targets: Array) -> Node:
        if self.is_regressor:
            return Node(value=float(targets.mean()), n_samples=len(targets), impurity=float(targets.var()))
        counts = np.bincount(targets.astype(np.int64), minlength=len(self.classes)).astype(np.float64)
        distribution = counts / counts.sum()
        impurity = float(_gini(counts) if self.criterion == "gini" else _entropy(counts))
        return Node(value=distribution, n_samples=len(targets), impurity=impurity)

    def _feature_subset(self) -> NDArray[np.int64]:
        total = self.n_features_
        if self.max_features in (None, "all"):
            return np.arange(total)
        if self.max_features == "sqrt":
            count = max(1, int(np.sqrt(total)))
        elif self.max_features == "log2":
            count = max(1, int(np.log2(total)))
        else:
            count = max(1, min(total, int(self.max_features)))
        return self._generator.choice(total, size=count, replace=False)

    def _grow(self, features: Array, targets: Array, depth: int) -> Node:
        leaf = self._leaf(targets)
        if depth >= self.max_depth or len(targets) < self.min_samples_split or len(np.unique(targets)) == 1:
            return leaf

        split = self._best_split(features, targets)
        if split is None:
            return leaf

        feature, threshold, gain = split
        mask = features[:, feature] <= threshold
        node = Node(
            feature=feature,
            threshold=threshold,
            n_samples=len(targets),
            impurity=leaf.impurity,
            left=self._grow(features[mask], targets[mask], depth + 1),
            right=self._grow(features[~mask], targets[~mask], depth + 1),
        )
        self._gains.append(gain * len(targets))
        self._gain_features.append(feature)
        return node

    def _best_split(self, features: Array, targets: Array) -> tuple[int, float, float] | None:
        """Return ``(feature, threshold, impurity_gain)`` for the best split."""
        n_samples = len(targets)
        if n_samples < 2 * self.min_samples_leaf:
            return None

        parent = self._impurity_of(targets)
        best: tuple[int, float, float] | None = None

        for feature in self._feature_subset():
            values = features[:, feature]
            order = np.argsort(values, kind="mergesort")
            sorted_values = values[order]
            # Split only between distinct values, honouring min_samples_leaf.
            candidates = np.flatnonzero(sorted_values[:-1] < sorted_values[1:])
            low, high = self.min_samples_leaf - 1, n_samples - self.min_samples_leaf - 1
            candidates = candidates[(candidates >= low) & (candidates <= high)]
            if candidates.size == 0:
                continue

            left_impurity, right_impurity = self._child_impurities(targets[order], candidates)
            left_counts = (candidates + 1).astype(np.float64)
            right_counts = n_samples - left_counts
            weighted = (left_counts * left_impurity + right_counts * right_impurity) / n_samples
            index = int(np.argmin(weighted))
            gain = parent - float(weighted[index])
            if gain > 0 and (best is None or gain > best[2]):
                position = int(candidates[index])
                threshold = float((sorted_values[position] + sorted_values[position + 1]) / 2)
                best = (int(feature), threshold, gain)
        return best

    def _impurity_of(self, targets: Array) -> float:
        if self.is_regressor:
            return float(targets.var())
        counts = np.bincount(targets.astype(np.int64), minlength=len(self.classes)).astype(np.float64)
        return float(_gini(counts) if self.criterion == "gini" else _entropy(counts))

    def _child_impurities(self, sorted_targets: Array, candidates: NDArray[np.int64]) -> tuple[Array, Array]:
        """Impurity of both children at every candidate split, vectorized."""
        n_samples = len(sorted_targets)
        if self.is_regressor:
            cumulative = np.cumsum(sorted_targets)
            cumulative_squares = np.cumsum(sorted_targets**2)
            left_counts = (candidates + 1).astype(np.float64)
            right_counts = n_samples - left_counts
            left_sum = cumulative[candidates]
            left_square = cumulative_squares[candidates]
            right_sum = cumulative[-1] - left_sum
            right_square = cumulative_squares[-1] - left_square
            left = left_square / left_counts - (left_sum / left_counts) ** 2
            right = right_square / right_counts - (right_sum / right_counts) ** 2
            return np.maximum(left, 0.0), np.maximum(right, 0.0)

        one_hot = np.zeros((n_samples, len(self.classes)), dtype=np.float64)
        one_hot[np.arange(n_samples), sorted_targets.astype(np.int64)] = 1.0
        cumulative = np.cumsum(one_hot, axis=0)
        left_counts = cumulative[candidates]
        right_counts = cumulative[-1] - left_counts
        impurity = _gini if self.criterion == "gini" else _entropy
        return impurity(left_counts), impurity(right_counts)

    # -- prediction -------------------------------------------------------
    def _traverse(self, sample: Array) -> Node:
        node = self.root
        assert node is not None, "tree is not fitted"
        while not node.is_leaf:
            assert node.feature is not None and node.left is not None and node.right is not None
            node = node.left if sample[node.feature] <= node.threshold else node.right
        return node

    def predict_proba(self, features: Array) -> Array:
        if self.is_regressor:
            raise ValueError("predict_proba is classification-only")
        return np.vstack([self._traverse(sample).value for sample in np.asarray(features, dtype=np.float64)])

    def predict(self, features: Array) -> Array:
        samples = np.asarray(features, dtype=np.float64)
        if self.is_regressor:
            return np.array([float(self._traverse(sample).value) for sample in samples])
        return self.classes[np.argmax(self.predict_proba(samples), axis=1)]

    def depth(self) -> int:
        def walk(node: Node | None) -> int:
            if node is None or node.is_leaf:
                return 0
            return 1 + max(walk(node.left), walk(node.right))

        return walk(self.root)

    def n_leaves(self) -> int:
        def walk(node: Node | None) -> int:
            if node is None:
                return 0
            return 1 if node.is_leaf else walk(node.left) + walk(node.right)

        return walk(self.root)

    def feature_importances(self) -> Array:
        """Impurity gain per feature, weighted by samples and normalized."""
        importances = np.zeros(self.n_features_)
        for gain, feature in zip(self._gains, self._gain_features, strict=True):
            importances[feature] += gain
        total = importances.sum()
        return importances / total if total > 0 else importances

    # Accumulated during _grow; declared here so an unfitted tree still works.
    _gains: list[float] = field(default_factory=list)
    _gain_features: list[int] = field(default_factory=list)
    _generator: np.random.Generator = field(default_factory=np.random.default_rng)


@dataclass
class RandomForest:
    """Bagging over decision trees, with out-of-bag scoring.

    Each tree sees a bootstrap resample and a random feature subset per split;
    the samples a tree did not see give a free validation estimate.
    """

    n_estimators: int = 50
    criterion: str = "gini"
    max_depth: int = 12
    min_samples_split: int = 2
    min_samples_leaf: int = 1
    max_features: int | str | None = "sqrt"
    rng: np.random.Generator | int | None = None
    trees: list[DecisionTree] = field(default_factory=list)
    classes: Labels = field(default_factory=lambda: np.zeros(0, dtype=np.int64))
    oob_score_: float | None = None

    @property
    def is_regressor(self) -> bool:
        return self.criterion in REGRESSION_CRITERIA

    def fit(self, features: Array, targets: Array) -> RandomForest:
        generator = datasets.as_rng(self.rng)
        samples = np.asarray(features, dtype=np.float64)
        n_samples = len(samples)
        if not self.is_regressor:
            self.classes = np.unique(np.asarray(targets, dtype=np.int64))

        self.trees = []
        oob_predictions: list[tuple[NDArray[np.int64], Array]] = []
        for _ in range(self.n_estimators):
            index = generator.integers(0, n_samples, size=n_samples)
            tree = DecisionTree(
                criterion=self.criterion,
                max_depth=self.max_depth,
                min_samples_split=self.min_samples_split,
                min_samples_leaf=self.min_samples_leaf,
                max_features=self.max_features,
                rng=generator,
            ).fit(samples[index], targets[index])
            self.trees.append(tree)

            out_of_bag = np.setdiff1d(np.arange(n_samples), index, assume_unique=False)
            if out_of_bag.size:
                predictions = (
                    tree.predict_proba(samples[out_of_bag])
                    if not self.is_regressor
                    else tree.predict(samples[out_of_bag])
                )
                oob_predictions.append((out_of_bag, predictions))

        self.oob_score_ = self._out_of_bag_score(oob_predictions, targets, n_samples)
        return self

    def _out_of_bag_score(
        self, collected: list[tuple[NDArray[np.int64], Array]], targets: Array, n_samples: int
    ) -> float | None:
        """Accuracy (classification) or R² (regression) on unseen-by-tree samples."""
        if not collected:
            return None
        width = 1 if self.is_regressor else len(self.classes)
        totals = np.zeros((n_samples, width))
        counts = np.zeros(n_samples)
        for index, predictions in collected:
            totals[index] += predictions.reshape(len(index), width)
            counts[index] += 1

        seen = counts > 0
        if not seen.any():
            return None
        averaged = totals[seen] / counts[seen, None]
        if self.is_regressor:
            return metrics.r2_score(np.asarray(targets)[seen], averaged.ravel())
        return metrics.accuracy(np.asarray(targets)[seen], self.classes[np.argmax(averaged, axis=1)])

    def predict_proba(self, features: Array) -> Array:
        if self.is_regressor:
            raise ValueError("predict_proba is classification-only")
        return np.mean([tree.predict_proba(features) for tree in self.trees], axis=0)

    def predict(self, features: Array) -> Array:
        if self.is_regressor:
            return np.mean([tree.predict(features) for tree in self.trees], axis=0)
        return self.classes[np.argmax(self.predict_proba(features), axis=1)]

    def feature_importances(self) -> Array:
        return np.mean([tree.feature_importances() for tree in self.trees], axis=0)


@experiment(
    "a2_trees_and_forests",
    defaults={
        "samples": 600,
        "noise": 0.7,
        "classes": 3,
        "max_depth": 12,
        "depths": [1, 2, 3, 5, 8, 12, 20],
        "forest_sizes": [1, 5, 15, 40, 80],
        "min_samples_leaf": 2,
    },
    description="A single CART tree (Gini vs entropy, depth sweep) against a bagged random forest.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Decision tree + random forest from scratch (Gini/entropy splits, bagging)."""
    rng = np.random.default_rng(exp.seed)
    features, labels = datasets.make_spirals(exp["samples"], classes=exp["classes"], noise=exp["noise"], rng=rng)
    x_train, x_test, y_train, y_test = datasets.train_test_split(features, labels, rng=rng)

    # -- Gini vs entropy --------------------------------------------------
    criterion_scores: dict[str, float] = {}
    trees: dict[str, DecisionTree] = {}
    for criterion in CLASSIFICATION_CRITERIA:
        tree = DecisionTree(
            criterion=criterion, max_depth=exp["max_depth"], min_samples_leaf=exp["min_samples_leaf"], rng=rng
        ).fit(x_train, y_train)
        score = metrics.accuracy(y_test, tree.predict(x_test))
        criterion_scores[criterion] = score
        trees[criterion] = tree
        exp.log({f"tree/{criterion}/test_accuracy": score, f"tree/{criterion}/leaves": tree.n_leaves()})
        print(f"[tree:{criterion:<8}] test accuracy={score:.4f}  depth={tree.depth()}  leaves={tree.n_leaves()}")

    # -- depth sweep: where a single tree starts memorising ----------------
    train_curve, test_curve = [], []
    for depth in exp["depths"]:
        tree = DecisionTree(criterion="gini", max_depth=depth, min_samples_leaf=exp["min_samples_leaf"], rng=rng).fit(
            x_train, y_train
        )
        train_score = metrics.accuracy(y_train, tree.predict(x_train))
        test_score = metrics.accuracy(y_test, tree.predict(x_test))
        train_curve.append(train_score)
        test_curve.append(test_score)
        exp.log({"depth": depth, "tree/train_accuracy": train_score, "tree/test_accuracy": test_score})
        print(f"[depth {depth:>3}] train={train_score:.4f}  test={test_score:.4f}  gap={train_score - test_score:+.4f}")

    # -- bagging: forest size vs. accuracy, and OOB as a free validation set
    forest_curve, oob_curve = [], []
    forest: RandomForest | None = None
    for size in exp["forest_sizes"]:
        forest = RandomForest(
            n_estimators=size,
            criterion="gini",
            max_depth=exp["max_depth"],
            min_samples_leaf=exp["min_samples_leaf"],
            max_features="sqrt",
            rng=rng,
        ).fit(x_train, y_train)
        test_score = metrics.accuracy(y_test, forest.predict(x_test))
        forest_curve.append(test_score)
        oob_curve.append(forest.oob_score_ or float("nan"))
        exp.log({"forest_size": size, "forest/test_accuracy": test_score, "forest/oob_accuracy": forest.oob_score_})
        print(f"[forest {size:>3}] test={test_score:.4f}  oob={forest.oob_score_:.4f}")
    assert forest is not None

    best_tree = max(criterion_scores, key=lambda key: criterion_scores[key])
    tree_score = criterion_scores[best_tree]
    forest_score = forest_curve[-1]

    # -- figures ----------------------------------------------------------
    exp.log_figure(
        "depth_sweep",
        plotting.plot_curves(
            {"train": train_curve, "test": test_curve},
            x=exp["depths"],
            title="Single tree — accuracy vs. max depth",
            xlabel="max depth",
            ylabel="accuracy",
            marker="o",
        ),
    )
    exp.log_figure(
        "forest_size",
        plotting.plot_curves(
            {"test": forest_curve, "out-of-bag": oob_curve},
            x=exp["forest_sizes"],
            title="Random forest — accuracy vs. number of trees",
            xlabel="trees",
            ylabel="accuracy",
            marker="o",
        ),
    )
    exp.log_figure(
        "tree_boundary",
        plotting.plot_decision_boundary(
            trees[best_tree].predict, x_test, y_test, title=f"Single tree ({best_tree}) — test set"
        ),
    )
    exp.log_figure(
        "forest_boundary",
        plotting.plot_decision_boundary(
            forest.predict, x_test, y_test, title=f"Random forest ({exp['forest_sizes'][-1]} trees) — test set"
        ),
    )
    exp.log_figure(
        "confusion",
        plotting.plot_confusion_matrix(
            metrics.confusion_matrix(y_test, forest.predict(x_test)),
            title="Random forest — test confusion",
            normalize=True,
        ),
    )

    return {
        "tree/gini/test_accuracy": criterion_scores["gini"],
        "tree/entropy/test_accuracy": criterion_scores["entropy"],
        "tree/best_depth": float(exp["depths"][int(np.argmax(test_curve))]),
        "forest/test_accuracy": forest_score,
        "forest/oob_accuracy": float(forest.oob_score_ or float("nan")),
        "forest_minus_tree": forest_score - tree_score,
    }


main = run

if __name__ == "__main__":
    raise SystemExit(main())
