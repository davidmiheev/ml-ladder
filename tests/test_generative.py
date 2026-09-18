"""Part E's scoring machinery, checked where the answer is known analytically.

A Fréchet distance is the one number every rung in part E is compared on, and a
wrong constant in it would reorder the whole part's conclusions without ever
failing. The Gaussian cases below have closed-form answers.
"""

from __future__ import annotations

import numpy as np
import pytest

from mlladder.generative import frechet_distance, nearest_neighbour_distance


def gaussian(mean, std, *, n=4000, dim=4, seed=0):
    rng = np.random.default_rng(seed)
    return rng.normal(mean, std, size=(n, dim))


def test_distance_to_itself_is_zero():
    sample = gaussian(0.0, 1.0)
    assert frechet_distance(sample, sample) == pytest.approx(0.0, abs=1e-6)


def test_mean_shift_contributes_its_squared_norm():
    r"""With equal covariances the distance is exactly $\lVert \mu_a - \mu_b \rVert^2$."""
    a, b = gaussian(0.0, 1.0, seed=1), gaussian(0.0, 1.0, seed=2) + 3.0
    # Four dimensions each shifted by 3: 4 * 9 = 36, plus sampling noise.
    assert frechet_distance(a, b) == pytest.approx(36.0, rel=0.05)


def test_scale_difference_is_detected_without_any_mean_shift():
    r"""Same mean, different spread: $(\sigma_a - \sigma_b)^2$ per dimension."""
    a, b = gaussian(0.0, 1.0, seed=3), gaussian(0.0, 3.0, seed=4)
    assert frechet_distance(a, b) == pytest.approx(4 * (3.0 - 1.0) ** 2, rel=0.1)


def test_distance_is_symmetric():
    a, b = gaussian(0.0, 1.0, seed=5), gaussian(1.0, 2.0, seed=6)
    assert frechet_distance(a, b) == pytest.approx(frechet_distance(b, a), rel=1e-6)


def test_distance_grows_with_the_gap():
    reference = gaussian(0.0, 1.0, seed=7)
    near = gaussian(0.5, 1.0, seed=8)
    far = gaussian(4.0, 1.0, seed=9)
    assert frechet_distance(reference, near) < frechet_distance(reference, far)


def test_collapsed_features_do_not_make_it_explode():
    """A mode-collapsed generator emits identical features; the ridge must hold."""
    real = gaussian(0.0, 1.0, seed=10)
    collapsed = np.zeros((500, 4))
    value = frechet_distance(real, collapsed)
    assert np.isfinite(value) and value > 0


def test_nearest_neighbour_distance_is_zero_for_copies():
    """The memorisation check: images lifted from the training set score 0."""
    real = np.random.default_rng(0).random((64, 1, 8, 8)).astype(np.float32)
    assert nearest_neighbour_distance(real[:16], real, rng=0) == pytest.approx(0.0, abs=1e-5)


def test_nearest_neighbour_distance_is_positive_for_novel_images():
    rng = np.random.default_rng(1)
    real = rng.random((64, 1, 8, 8)).astype(np.float32)
    novel = rng.random((16, 1, 8, 8)).astype(np.float32)
    assert nearest_neighbour_distance(novel, real, rng=0) > 0.5
