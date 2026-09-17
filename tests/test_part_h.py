"""Part H — computer vision & OCR.

H1 writes its own metric, so the metric is what needs testing hardest: a wrong
IoU or a wrong precision envelope produces a plausible number rather than a
crash, and every conclusion in the part would inherit it. The cases below are
ones whose answer can be computed by hand.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch", reason="uv sync --extra torch")


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("MLLADDER_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MLLADDER_TRACKER", "none")


def box(*values: float) -> np.ndarray:
    return np.array(values, dtype=np.float32)


# -- H1: IoU -----------------------------------------------------------------


def test_iou_of_known_pairs():
    from parts.h_computer_vision.h1_detection_from_scratch import iou_matrix

    unit = box(0, 0, 10, 10)[None]
    assert iou_matrix(unit, unit)[0, 0] == pytest.approx(1.0)
    assert iou_matrix(unit, box(20, 20, 30, 30)[None])[0, 0] == pytest.approx(0.0)
    # Half-overlap in x: intersection 50, union 150.
    assert iou_matrix(unit, box(5, 0, 15, 10)[None])[0, 0] == pytest.approx(1 / 3, abs=1e-6)
    # Nested quarter: intersection 25, union 100.
    assert iou_matrix(unit, box(0, 0, 5, 5)[None])[0, 0] == pytest.approx(0.25)


def test_iou_handles_empty_inputs():
    """An image with no detections must not raise on the way to a zero score."""
    from parts.h_computer_vision.h1_detection_from_scratch import iou_matrix

    empty = np.zeros((0, 4), dtype=np.float32)
    assert iou_matrix(empty, box(0, 0, 1, 1)[None]).shape == (0, 1)
    assert iou_matrix(box(0, 0, 1, 1)[None], empty).shape == (1, 0)


def test_iou_matrix_is_symmetric():
    from parts.h_computer_vision.h1_detection_from_scratch import iou_matrix

    a = np.stack([box(0, 0, 10, 10), box(4, 4, 14, 14)])
    b = np.stack([box(2, 2, 12, 12), box(0, 0, 4, 4), box(30, 30, 40, 40)])
    assert np.allclose(iou_matrix(a, b), iou_matrix(b, a).T)


# -- H1: non-max suppression -------------------------------------------------


def test_nms_keeps_the_best_of_a_duplicate_cluster():
    from parts.h_computer_vision.h1_detection_from_scratch import nms

    boxes = np.stack([box(0, 0, 10, 10), box(1, 1, 11, 11), box(0, 0, 9, 9), box(50, 50, 60, 60)])
    scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [0, 3]


def test_nms_thresholds_bracket_the_behaviour():
    """At 1.0 nothing is suppressed; at 0.0 anything touching the winner is."""
    from parts.h_computer_vision.h1_detection_from_scratch import nms

    boxes = np.stack([box(0, 0, 10, 10), box(1, 1, 11, 11), box(0, 0, 9, 9), box(50, 50, 60, 60)])
    scores = np.array([0.9, 0.8, 0.7, 0.6], dtype=np.float32)
    assert sorted(nms(boxes, scores, 1.0).tolist()) == [0, 1, 2, 3]
    assert nms(boxes, scores, 0.0).tolist() == [0, 3]


def test_nms_returns_highest_scoring_first():
    from parts.h_computer_vision.h1_detection_from_scratch import nms

    boxes = np.stack([box(0, 0, 10, 10), box(50, 50, 60, 60)])
    scores = np.array([0.2, 0.95], dtype=np.float32)
    assert nms(boxes, scores, 0.5).tolist() == [1, 0]


# -- H1: average precision ---------------------------------------------------


def test_ap_of_perfect_predictions_is_one():
    from parts.h_computer_vision.h1_detection_from_scratch import average_precision

    truths = [(0, box(0, 0, 10, 10)), (1, box(0, 0, 10, 10))]
    predictions = [(0, 0.9, box(0, 0, 10, 10)), (1, 0.8, box(0, 0, 10, 10))]
    assert average_precision(predictions, truths, iou_threshold=0.5) == pytest.approx(1.0)


def test_ap_caps_at_the_achieved_recall():
    """One of two objects found — recall cannot exceed a half, so nor can AP."""
    from parts.h_computer_vision.h1_detection_from_scratch import average_precision

    truths = [(0, box(0, 0, 10, 10)), (1, box(0, 0, 10, 10))]
    assert average_precision([(0, 0.9, box(0, 0, 10, 10))], truths, iou_threshold=0.5) == pytest.approx(0.5)


def test_ap_punishes_a_top_ranked_false_positive():
    r"""A junk box ranked above both hits costs exactly a third.

    Walking the ranking gives precision 0, 1/2, 2/3 at recalls 0, 1/2, 1. The
    envelope flattens all three to 2/3, so $\mathrm{AP} = 2/3$ — worth pinning,
    because an implementation that forgot the envelope would return 7/12.
    """
    from parts.h_computer_vision.h1_detection_from_scratch import average_precision

    truths = [(0, box(0, 0, 10, 10)), (1, box(0, 0, 10, 10))]
    predictions = [
        (0, 0.99, box(40, 40, 50, 50)),
        (0, 0.9, box(0, 0, 10, 10)),
        (1, 0.8, box(0, 0, 10, 10)),
    ]
    assert average_precision(predictions, truths, iou_threshold=0.5) == pytest.approx(2 / 3, abs=1e-6)


def test_duplicate_detections_cannot_claim_the_same_truth_twice():
    """The second box on one object is a false positive, not a second hit."""
    from parts.h_computer_vision.h1_detection_from_scratch import average_precision

    truths = [(0, box(0, 0, 10, 10)), (1, box(0, 0, 10, 10))]
    duplicated = [
        (0, 0.9, box(0, 0, 10, 10)),
        (0, 0.85, box(0, 0, 10, 10)),  # same object again
        (1, 0.5, box(0, 0, 10, 10)),
    ]
    # Recall still reaches 1, but precision along the way is dented.
    assert average_precision(duplicated, truths, iou_threshold=0.5) < 1.0


def test_ap_is_undefined_rather_than_zero_for_an_absent_class():
    from parts.h_computer_vision.h1_detection_from_scratch import average_precision

    assert np.isnan(average_precision([(0, 0.9, box(0, 0, 1, 1))], [], iou_threshold=0.5))


def test_map_skips_absent_classes():
    from parts.h_computer_vision.h1_detection_from_scratch import mean_average_precision

    score, per_class = mean_average_precision(
        [(0, 0, 0.9, box(0, 0, 10, 10))],
        [(0, 0, box(0, 0, 10, 10))],
        num_classes=3,
        iou_threshold=0.5,
    )
    assert score == pytest.approx(1.0)
    assert set(per_class) == {0}, "classes with no ground truth must not be averaged in"


def test_stricter_iou_threshold_cannot_raise_ap():
    """A loose box counts at 0.5 and not at 0.9; AP must be monotone in it."""
    from parts.h_computer_vision.h1_detection_from_scratch import average_precision

    truths = [(0, box(0, 0, 10, 10))]
    loose = [(0, 0.9, box(1, 1, 12, 12))]
    lenient = average_precision(loose, truths, iou_threshold=0.5)
    strict = average_precision(loose, truths, iou_threshold=0.9)
    assert lenient == pytest.approx(1.0)
    assert strict == pytest.approx(0.0)


# -- H1: target encoding and decoding ----------------------------------------


def test_encode_decode_round_trips_a_box():
    """Feeding the encoder's own targets through decode must return the boxes.

    This is the test that catches an off-by-one or a half-cell shift in the
    grid arithmetic, which would otherwise show up only as a mysteriously low
    mAP@0.75.
    """
    from parts.h_computer_vision.h1_detection_from_scratch import decode, encode_targets, iou_matrix

    from mlladder import synth

    scenes = synth.shapes(6, size=64, rng=11)
    objectness, boxes, labels, _ = encode_targets(scenes, grid=8, size=64, num_classes=3)

    # Rebuild the raw tensor the model would have to emit to be perfect.
    raw = torch.zeros(len(scenes), 8, 8, 5 + 3)
    raw[..., 0] = torch.where(objectness > 0.5, 20.0, -20.0)  # saturated logits
    raw[..., 1:3] = torch.logit(boxes[..., 0:2].clamp(1e-4, 1 - 1e-4))
    raw[..., 3:5] = boxes[..., 2:4]
    raw[..., 5:] = torch.nn.functional.one_hot(labels, 3).float() * 20.0

    for (decoded, _scores, classes), scene in zip(decode(raw, grid=8, size=64, score_threshold=0.5), scenes):
        assert len(decoded) == len(scene.boxes), "a box was lost or invented"
        overlaps = iou_matrix(decoded.astype(np.float32), scene.boxes)
        assert overlaps.max(axis=1).min() > 0.99, "grid arithmetic does not round-trip"
        assert sorted(classes.tolist()) == sorted(scene.labels.tolist())


def test_encode_targets_counts_cell_collisions():
    """Two centres in one cell is a ceiling on recall, so it must be reported."""
    from parts.h_computer_vision.h1_detection_from_scratch import encode_targets

    from mlladder import synth

    scene = synth.ShapeScene(
        image=np.zeros((64, 64), dtype=np.uint8),
        boxes=np.array([[0, 0, 6, 6], [1, 1, 7, 7]], dtype=np.float32),  # both centres in cell (0, 0)
        labels=np.array([0, 1]),
        mask=np.zeros((64, 64), dtype=np.int64),
    )
    _, _, _, collisions = encode_targets([scene], grid=8, size=64, num_classes=3)
    assert collisions == 1


def test_detection_loss_ignores_boxes_in_empty_cells():
    """Changing the box prediction where there is no object must not change the loss."""
    from parts.h_computer_vision.h1_detection_from_scratch import detection_loss

    torch.manual_seed(0)
    raw = torch.zeros(2, 4, 4, 8, requires_grad=False)
    objectness = torch.zeros(2, 4, 4)
    objectness[0, 1, 1] = 1.0
    boxes = torch.zeros(2, 4, 4, 4)
    labels = torch.zeros(2, 4, 4, dtype=torch.long)

    baseline, *_ = detection_loss(raw, objectness, boxes, labels, no_object_weight=0.2, box_weight=5.0)
    meddled = raw.clone()
    meddled[0, 3, 3, 1:5] = 7.0  # an empty cell
    changed, *_ = detection_loss(meddled, objectness, boxes, labels, no_object_weight=0.2, box_weight=5.0)
    assert changed.item() == pytest.approx(baseline.item())


def test_h1_runs():
    from parts.h_computer_vision.h1_detection_from_scratch import main

    assert (
        main(
            [
                "--train-images",
                "120",
                "--test-images",
                "60",
                "--epochs",
                "2",
                "--width",
                "8",
                "--nms-thresholds",
                "0.3",
                "0.5",
                "--iou-thresholds",
                "0.5",
            ]
        )
        == 0
    )
