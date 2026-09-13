"""Part A — classical ML from scratch.

Each implementation is checked against a property it must have (recovering
known weights, matching a closed form, separating separable data), not against
a golden number, so the tests stay meaningful if the internals change.
"""

from __future__ import annotations

from itertools import pairwise

import numpy as np
import pytest
from parts.a_classical.a1_linear_logistic_regression import LinearRegression, LogisticRegression, batches
from parts.a_classical.a2_trees_and_forests import DecisionTree, RandomForest
from parts.a_classical.a3_kmeans_and_pca import PCA, KMeans
from parts.a_classical.a4_gradient_boosting import GradientBoostingRegressor, _xgboost_baseline

from mlladder import datasets, metrics


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("MLLADDER_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MLLADDER_TRACKER", "none")


# -- A1 --------------------------------------------------------------------


def test_batches_partition_every_sample():
    assert [index.tolist() for index in batches(5, 0, np.random.default_rng(0))] == [[0, 1, 2, 3, 4]]
    chunks = batches(10, 3, np.random.default_rng(0))
    assert sorted(np.concatenate(chunks).tolist()) == list(range(10))
    assert [len(chunk) for chunk in chunks] == [3, 3, 3, 1]


def test_linear_regression_matches_the_normal_equation():
    features, targets, true_weights = datasets.make_regression(400, 4, noise=0.05, bias=2.0, rng=0)
    closed = LinearRegression().fit_normal_equation(features, targets)
    descent = LinearRegression(lr=0.1, epochs=400).fit(features, targets, rng=0)

    assert np.allclose(closed.weights, true_weights, atol=0.05)
    assert closed.bias == pytest.approx(2.0, abs=0.05)
    assert np.allclose(descent.weights, closed.weights, atol=0.02)
    assert descent.history[-1] < descent.history[0]


def test_linear_regression_schedules_all_converge():
    features, targets, true_weights = datasets.make_regression(300, 3, noise=0.05, rng=1)
    for batch_size, decay in ((0, 0.0), (32, 0.05), (1, 0.5)):
        model = LinearRegression(lr=0.1, epochs=60, batch_size=batch_size, lr_decay=decay).fit(features, targets, rng=1)
        assert np.linalg.norm(model.weights - true_weights) < 0.1, f"batch_size={batch_size} did not converge"


def test_l2_shrinks_weights_but_not_the_intercept():
    features, targets, _ = datasets.make_regression(200, 4, noise=0.1, bias=5.0, rng=2)
    plain = LinearRegression().fit_normal_equation(features, targets)
    ridge = LinearRegression(l2=1.0).fit_normal_equation(features, targets)
    assert np.linalg.norm(ridge.weights) < 0.6 * np.linalg.norm(plain.weights)
    # The intercept is not in the penalty, so it stays at the data's level
    # instead of being pulled toward zero with the weights.
    assert ridge.bias == pytest.approx(5.0, abs=0.25)


def test_sigmoid_is_stable_at_the_extremes():
    extreme = np.array([-800.0, 0.0, 800.0])
    probabilities = LogisticRegression._sigmoid(extreme)
    assert np.all(np.isfinite(probabilities))
    assert probabilities.tolist() == pytest.approx([0.0, 0.5, 1.0])


def test_logistic_regression_separates_separable_blobs():
    points, labels, _ = datasets.make_blobs(300, centers=2, cluster_std=0.6, rng=3)
    model = LogisticRegression(lr=0.5, epochs=200).fit(points, labels.astype(float), rng=3)
    assert metrics.accuracy(labels, model.predict(points)) > 0.97
    assert model.history[-1] < model.history[0]


# -- A2 --------------------------------------------------------------------


def test_tree_fits_separable_data_and_respects_max_depth():
    points, labels, _ = datasets.make_blobs(200, centers=3, cluster_std=0.4, rng=4)
    tree = DecisionTree(max_depth=6, min_samples_leaf=1, rng=4).fit(points, labels)
    assert metrics.accuracy(labels, tree.predict(points)) > 0.98
    assert tree.depth() <= 6

    stump = DecisionTree(max_depth=1, rng=4).fit(points, labels)
    assert stump.depth() == 1
    assert stump.n_leaves() == 2


@pytest.mark.parametrize("criterion", ["gini", "entropy"])
def test_both_classification_criteria_work(criterion):
    points, labels = datasets.make_moons(200, noise=0.1, rng=5)
    tree = DecisionTree(criterion=criterion, max_depth=8, rng=5).fit(points, labels)
    assert metrics.accuracy(labels, tree.predict(points)) > 0.95
    probabilities = tree.predict_proba(points)
    assert np.allclose(probabilities.sum(axis=1), 1.0)


def test_unknown_criterion_is_rejected():
    with pytest.raises(ValueError, match="unknown criterion"):
        DecisionTree(criterion="nonsense").fit(np.zeros((4, 2)), np.zeros(4, dtype=int))


def test_regression_tree_approximates_a_step_function():
    features = np.linspace(-3, 3, 300).reshape(-1, 1)
    targets = np.where(features[:, 0] > 0, 4.0, -1.0)
    tree = DecisionTree(criterion="mse", max_depth=3, rng=6).fit(features, targets)
    assert metrics.rmse(targets, tree.predict(features)) < 0.2
    with pytest.raises(ValueError, match="classification-only"):
        tree.predict_proba(features)


def test_min_samples_leaf_is_honoured():
    features = np.linspace(0, 1, 60).reshape(-1, 1)
    targets = (features[:, 0] > 0.5).astype(int)
    tree = DecisionTree(max_depth=10, min_samples_leaf=10, rng=7).fit(features, targets)

    def leaf_sizes(node):
        if node.is_leaf:
            return [node.n_samples]
        return leaf_sizes(node.left) + leaf_sizes(node.right)

    assert min(leaf_sizes(tree.root)) >= 10


def test_feature_importances_favour_the_informative_feature():
    rng = np.random.default_rng(8)
    features = rng.normal(size=(300, 4))
    labels = (features[:, 1] > 0).astype(int)  # only x1 carries signal
    tree = DecisionTree(max_depth=4, rng=8).fit(features, labels)
    importances = tree.feature_importances()
    assert importances.sum() == pytest.approx(1.0)
    assert int(np.argmax(importances)) == 1


def test_forest_beats_a_deep_tree_on_noisy_data_and_scores_oob():
    features, labels = datasets.make_spirals(500, classes=3, noise=0.7, rng=9)
    x_train, x_test, y_train, y_test = datasets.train_test_split(features, labels, rng=9)

    tree = DecisionTree(max_depth=12, min_samples_leaf=2, rng=9).fit(x_train, y_train)
    forest = RandomForest(n_estimators=30, max_depth=12, min_samples_leaf=2, rng=9).fit(x_train, y_train)

    assert metrics.accuracy(y_test, forest.predict(x_test)) > metrics.accuracy(y_test, tree.predict(x_test))
    assert forest.oob_score_ is not None
    # Out-of-bag accuracy is a validation estimate, so it should track the test score.
    assert abs(forest.oob_score_ - metrics.accuracy(y_test, forest.predict(x_test))) < 0.15
    assert np.allclose(forest.predict_proba(x_test).sum(axis=1), 1.0)


def test_regression_forest_reports_oob_r2():
    features, targets = datasets.make_friedman1(300, noise=1.0, rng=10)
    forest = RandomForest(n_estimators=20, criterion="mse", max_depth=8, max_features="sqrt", rng=10).fit(
        features, targets
    )
    assert forest.is_regressor
    assert 0.0 < (forest.oob_score_ or 0.0) < 1.0
    with pytest.raises(ValueError, match="classification-only"):
        forest.predict_proba(features)


# -- A3 --------------------------------------------------------------------


def test_kmeans_recovers_well_separated_blobs():
    points, labels, _ = datasets.make_blobs(300, centers=4, cluster_std=0.35, rng=11)
    model = KMeans(n_clusters=4, n_init=5, rng=11).fit(points)
    assert metrics.adjusted_rand_index(labels, model.labels_) == pytest.approx(1.0)
    assert np.array_equal(model.predict(points), model.labels_)
    assert model.n_iter_ >= 1


def test_kmeans_inertia_decreases_with_k():
    points, _, _ = datasets.make_blobs(200, centers=3, rng=12)
    inertias = [KMeans(n_clusters=k, n_init=3, rng=12).fit(points).inertia_ for k in (2, 3, 4, 5)]
    assert all(later <= earlier + 1e-9 for earlier, later in pairwise(inertias))


def test_kmeans_survives_duplicate_points():
    points = np.repeat(np.array([[0.0, 0.0], [5.0, 5.0]]), 20, axis=0)
    model = KMeans(n_clusters=4, n_init=3, rng=13).fit(points)
    assert np.isfinite(model.inertia_)
    assert len(np.unique(model.labels_)) <= 4


def test_pca_reconstructs_exactly_with_all_components():
    rng = np.random.default_rng(14)
    features = rng.normal(size=(200, 5))
    pca = PCA(n_components=5).fit(features)
    assert pca.explained_variance_ratio_.sum() == pytest.approx(1.0)
    assert pca.reconstruction_error(features) == pytest.approx(0.0, abs=1e-20)
    assert np.allclose(pca.components_ @ pca.components_.T, np.eye(5), atol=1e-9)


def test_pca_finds_the_dominant_direction():
    rng = np.random.default_rng(15)
    latent = rng.normal(size=(300, 1))
    features = np.hstack([latent * 10, latent * 0.01 + rng.normal(scale=0.01, size=(300, 1))])
    pca = PCA(n_components=2).fit(features)
    assert pca.explained_variance_ratio_[0] > 0.99
    assert abs(pca.components_[0, 0]) > abs(pca.components_[0, 1])
    assert pca.transform(features).shape == (300, 2)


def test_pca_error_shrinks_as_components_are_added():
    features, _, _ = datasets.make_blobs(200, centers=3, n_features=6, rng=16)
    pca = PCA(n_components=6).fit(features)
    errors = [pca.reconstruction_error(features, n_components=k) for k in range(1, 7)]
    assert all(later <= earlier + 1e-9 for earlier, later in pairwise(errors))


# -- A4 --------------------------------------------------------------------


def test_boosting_reduces_train_loss_and_beats_its_own_weak_learner():
    features, targets = datasets.make_friedman1(400, noise=1.0, rng=17)
    x_train, x_test, y_train, y_test = datasets.train_test_split(features, targets, rng=17)

    booster = GradientBoostingRegressor(n_estimators=40, learning_rate=0.1, max_depth=3, rng=17).fit(x_train, y_train)
    stump = DecisionTree(criterion="mse", max_depth=3, rng=17).fit(x_train, y_train)

    assert booster.train_loss_[-1] < booster.train_loss_[0]
    assert metrics.rmse(y_test, booster.predict(x_test)) < metrics.rmse(y_test, stump.predict(x_test))
    assert len(booster.trees) == 40


def test_boosting_prediction_is_incremental():
    features, targets = datasets.make_friedman1(200, noise=0.5, rng=18)
    booster = GradientBoostingRegressor(n_estimators=10, max_depth=2, rng=18).fit(features, targets)
    assert booster.predict(features, n_estimators=0) == pytest.approx(np.full(len(features), booster.base_prediction_))

    rounds, errors = booster.staged_rmse(features, targets, every=2)
    assert rounds == [2, 4, 6, 8, 10]
    assert errors[-1] == pytest.approx(metrics.rmse(targets, booster.predict(features)))


def test_stochastic_boosting_runs_on_a_subsample():
    features, targets = datasets.make_friedman1(200, noise=1.0, rng=19)
    booster = GradientBoostingRegressor(n_estimators=15, subsample=0.5, max_depth=2, rng=19).fit(features, targets)
    assert len(booster.trees) == 15
    assert booster.train_loss_[-1] < booster.train_loss_[0]


def test_xgboost_baseline_is_optional():
    features, targets = datasets.make_friedman1(150, noise=1.0, rng=20)
    x_train, x_test, y_train, y_test = datasets.train_test_split(features, targets, rng=20)
    result = _xgboost_baseline(x_train, y_train, x_test, y_test, rounds=10, lr=0.3, depth=3, seed=0)
    if result is None:
        pytest.skip("xgboost not installed (uv sync --extra boost)")
    assert result["test_rmse"] > 0
    assert result["fit_seconds"] > 0


# -- the rungs end to end ---------------------------------------------------


def test_a1_runs():
    from parts.a_classical.a1_linear_logistic_regression import main

    assert main(["--samples", "120", "--epochs", "5", "--features", "3"]) == 0


def test_a2_runs():
    from parts.a_classical.a2_trees_and_forests import main

    assert main(["--samples", "150", "--depths", "2", "4", "--forest-sizes", "3", "--max-depth", "4"]) == 0


def test_a3_runs():
    from parts.a_classical.a3_kmeans_and_pca import main

    assert main(["--samples", "120", "--features", "4", "--k-values", "2", "3", "--n-init", "2"]) == 0


def test_a4_runs():
    from parts.a_classical.a4_gradient_boosting import main

    assert main(["--samples", "150", "--rounds", "8", "--learning-rates", "0.1", "--eval-every", "4"]) == 0
