"""Part D — self-supervised & representation learning.

The load-bearing tests are the two objectives: NT-Xent must actually reward the
positive pair and exclude a view's similarity to itself, and the MAE's loss must
be computed on hidden patches only. Both are easy to write in a way that trains
and teaches nothing.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="uv sync --extra torch")


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("MLLADDER_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MLLADDER_TRACKER", "none")


# -- D1: augmentation and NT-Xent -------------------------------------------


def test_zero_strength_augmentation_is_the_identity():
    from parts.d_self_supervised.d1_simclr_pretraining import augment

    images = torch.randn(4, 3, 32, 32)
    assert torch.equal(augment(images, strength=0.0), images)


def test_augmentation_changes_images_but_not_their_shape():
    from parts.d_self_supervised.d1_simclr_pretraining import augment

    torch.manual_seed(0)
    images = torch.randn(8, 3, 32, 32)
    view = augment(images, strength=1.0)
    assert view.shape == images.shape
    assert not torch.allclose(view, images)
    # Two independent calls must differ, or both "views" are the same view.
    assert not torch.allclose(augment(images, strength=1.0), augment(images, strength=1.0))


def test_nt_xent_rewards_the_positive_pair():
    from parts.d_self_supervised.d1_simclr_pretraining import nt_xent

    # Views stacked [a; b]: row i's positive is row i + N.
    a = torch.tensor([1.0, 0.0, 0.0])
    b = torch.tensor([0.0, 1.0, 0.0])
    aligned = torch.stack([a, b, a, b])
    assert nt_xent(aligned, temperature=0.1).item() < 1e-3

    # Swap so each row's "positive" is an orthogonal vector: loss must jump.
    misaligned = torch.stack([a, b, b, a])
    assert nt_xent(misaligned, temperature=0.1).item() > 5.0


def test_nt_xent_excludes_self_similarity():
    """Without the -inf diagonal, every view's nearest neighbour is itself."""
    from parts.d_self_supervised.d1_simclr_pretraining import nt_xent

    torch.manual_seed(0)
    features = torch.randn(6, 8)
    loss = nt_xent(features, temperature=0.5)
    assert torch.isfinite(loss)
    # With 2N=6 views and one positive among 5 candidates, a random
    # representation should sit near log(5).
    assert 0.5 < loss.item() < 4.0


def test_encoder_and_head_shapes():
    from parts.d_self_supervised.d1_simclr_pretraining import make_encoder, make_projection_head

    encoder, dim = make_encoder(depth=2, width=16)
    features = encoder(torch.randn(3, 3, 32, 32))
    assert features.shape == (3, dim)
    assert make_projection_head(dim, 32)(features).shape == (3, 32)


def test_linear_probe_separates_separable_features():
    from parts.d_self_supervised.d1_simclr_pretraining import linear_probe

    rng = np.random.default_rng(0)
    # An "encoder" that already puts the classes far apart.
    centres = rng.normal(size=(3, 6)).astype(np.float32) * 5
    labels = rng.integers(0, 3, size=200)
    images = (centres[labels] + rng.normal(scale=0.1, size=(200, 6))).astype(np.float32)

    accuracy = linear_probe(torch.nn.Identity(), images, labels, images, labels, device=torch.device("cpu"), epochs=30)
    assert accuracy > 0.95


# -- D2: patches and masking -------------------------------------------------


def test_patchify_roundtrips_exactly():
    from parts.d_self_supervised.d2_masked_autoencoder import patchify, unpatchify

    images = torch.randn(2, 3, 32, 32)
    patches = patchify(images, 4)
    assert patches.shape == (2, 64, 48)
    rebuilt = unpatchify(patches, patch=4, channels=3, height=32, width=32)
    assert torch.equal(rebuilt, images)

    with pytest.raises(ValueError, match="does not tile"):
        patchify(images, 5)


def test_random_mask_hides_the_requested_fraction():
    from parts.d_self_supervised.d2_masked_autoencoder import random_mask

    keep_index, mask = random_mask(4, 64, 0.75, device=torch.device("cpu"))
    assert keep_index.shape == (4, 16)
    assert mask.sum(dim=1).tolist() == [48.0] * 4
    # Kept positions are exactly the unmasked ones.
    for row in range(4):
        assert mask[row, keep_index[row]].sum() == 0


def test_mae_loss_only_sees_masked_patches():
    from parts.d_self_supervised.d2_masked_autoencoder import make_mae, patchify, random_mask

    torch.manual_seed(0)
    model = make_mae(num_patches=16, patch_dim=12, width=32, depth=1, heads=4, decoder_depth=1)
    patches = patchify(torch.randn(2, 3, 8, 8), 2)
    keep_index, mask = random_mask(2, 16, 0.5, device=torch.device("cpu"))

    loss, prediction = model(patches, keep_index, mask)
    assert prediction.shape == patches.shape and torch.isfinite(loss)

    # Perturbing only visible patches' *targets* must not move the loss.
    altered = patches.clone()
    altered[mask == 0] += 10.0
    moved, _ = model(altered, keep_index, mask)
    assert abs(moved.item() - loss.item()) > 0  # visible inputs do feed the encoder

    # But zeroing the mask leaves nothing to score.
    empty, _ = model(patches, keep_index, torch.zeros_like(mask))
    assert empty.item() == 0.0


def test_mae_representation_shape():
    from parts.d_self_supervised.d2_masked_autoencoder import make_mae

    model = make_mae(num_patches=16, patch_dim=12, width=32, depth=1, heads=4, decoder_depth=1)
    assert model.represent(torch.randn(3, 3, 8, 8), patch=2).shape == (3, 32)


# -- D3: the evaluation harness ---------------------------------------------


def test_stratified_subset_is_balanced():
    from parts.d_self_supervised.d3_linear_probing import stratified_subset

    labels = np.repeat(np.arange(5), 40)
    index = stratified_subset(labels, 7, np.random.default_rng(0))
    counts = np.bincount(labels[index], minlength=5)
    assert counts.tolist() == [7] * 5


def test_knn_on_separable_features_is_perfect():
    from parts.d_self_supervised.d3_linear_probing import knn_accuracy

    train = torch.tensor([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])
    labels = np.array([0, 0, 1, 1])
    test = torch.tensor([[0.95, 0.05], [0.05, 0.95]])
    assert knn_accuracy(train, labels, test, np.array([0, 1]), k=1) == pytest.approx(1.0)


# -- the rungs end to end ---------------------------------------------------


def test_d1_runs_and_summarizes_every_arm():
    """The smoke test, plus the two summary metrics that were wrong at first.

    ``best/probe_accuracy`` originally tracked only the temperature loop, so it
    reported the best of half the experiment; ``augmentation_gain`` was defined
    as strongest-minus-weakest, which measures nothing when the strength curve
    has its optimum in the middle — which, on the real budget, it does.
    """
    import json
    import os
    from pathlib import Path

    from parts.d_self_supervised.d1_simclr_pretraining import main

    assert (
        main(
            [
                "--pretrain-samples",
                "200",
                "--probe-samples",
                "200",
                "--test-samples",
                "200",
                "--steps",
                "4",
                "--batch-size",
                "16",
                "--width",
                "16",
                "--depth",
                "2",
                "--temperatures",
                "0.5",
                "--strengths",
                "0.0",
                "0.5",
                "1.0",
                "--probe-epochs",
                "2",
            ]
        )
        == 0
    )

    runs = sorted((Path(os.environ["MLLADDER_ARTIFACTS_DIR"]) / "d1_simclr_pretraining").iterdir())
    metrics = json.loads((runs[-1] / "summary.json").read_text())["metrics"]

    arms = {key: value for key, value in metrics.items() if key.endswith("/probe_accuracy") and "/" in key}
    strengths = {key: value for key, value in arms.items() if key.startswith("strength/")}
    assert len(strengths) == 3, "every swept strength should be reported"

    # Best means best over *every* arm, not just the temperature loop.
    assert metrics["best/probe_accuracy"] == max(
        value for key, value in arms.items() if key != "random_encoder/probe_accuracy"
    )
    # And the worth of augmentation is measured against the no-augmentation
    # control, not against the other end of a curve that need not be monotone.
    assert metrics["augmentation_gain"] == pytest.approx(
        max(strengths.values()) - metrics["strength/0.0/probe_accuracy"]
    )
    assert metrics["best/strength"] in (0.0, 0.5, 1.0)


def test_d2_runs():
    from parts.d_self_supervised.d2_masked_autoencoder import main

    assert (
        main(
            [
                "--pretrain-samples",
                "200",
                "--probe-samples",
                "200",
                "--test-samples",
                "200",
                "--steps",
                "4",
                "--batch-size",
                "16",
                "--width",
                "32",
                "--depth",
                "1",
                "--decoder-depth",
                "1",
                "--mask-ratios",
                "0.75",
                "--probe-epochs",
                "2",
            ]
        )
        == 0
    )


def test_d3_runs():
    from parts.d_self_supervised.d3_linear_probing import main

    assert (
        main(
            [
                "--pretrain-samples",
                "200",
                "--label-pool",
                "200",
                "--test-samples",
                "200",
                "--steps",
                "4",
                "--batch-size",
                "16",
                "--width",
                "16",
                "--depth",
                "2",
                "--mae-width",
                "32",
                "--mae-depth",
                "1",
                "--labels-per-class",
                "2",
                "--probe-epochs",
                "2",
            ]
        )
        == 0
    )
