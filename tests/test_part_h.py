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

    for (decoded, _scores, classes), scene in zip(
        decode(raw, grid=8, size=64, score_threshold=0.5), scenes, strict=True
    ):
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


# -- H2: segmentation metrics ------------------------------------------------


def test_confusion_matrix_counts_pixels():
    from parts.h_computer_vision.h2_unet_segmentation import confusion

    truth = np.array([[0, 0], [1, 1]])
    predicted = np.array([[0, 1], [1, 1]])
    matrix = confusion(predicted, truth, num_classes=2)
    # Row = truth, column = prediction: one 0->0, one 0->1, two 1->1.
    assert matrix.tolist() == [[1, 1], [0, 2]]


def test_confusion_matrix_honours_a_pixel_subset():
    from parts.h_computer_vision.h2_unet_segmentation import confusion

    truth = np.array([[[0, 0], [1, 1]]])
    predicted = np.array([[[0, 1], [1, 1]]])
    where = np.array([[[True, False], [False, False]]])
    assert confusion(predicted, truth, num_classes=2, where=where).sum() == 1


def test_iou_per_class_on_a_hand_computed_matrix():
    from parts.h_computer_vision.h2_unet_segmentation import iou_per_class

    # Class 0: 8 correct, 2 leaked to class 1. Class 1: 5 correct, 0 leaked.
    matrix = np.array([[8, 2], [0, 5]])
    ious = iou_per_class(matrix)
    assert ious[0] == pytest.approx(8 / (10 + 8 - 8))  # row 10, column 8
    assert ious[1] == pytest.approx(5 / (5 + 7 - 5))  # row 5, column 7


def test_absent_class_iou_is_nan_not_zero():
    from parts.h_computer_vision.h2_unet_segmentation import iou_per_class

    matrix = np.array([[5, 0, 0], [0, 3, 0], [0, 0, 0]])  # class 2 never appears
    ious = iou_per_class(matrix)
    assert np.isnan(ious[2])
    assert np.nanmean(ious) == pytest.approx(1.0), "a missing class must not drag the mean down"


def test_dice_and_iou_agree_through_their_identity():
    r"""$D = 2I/(1+I)$ — if the two disagree, one of them is wrong."""
    from parts.h_computer_vision.h2_unet_segmentation import dice_per_class, iou_per_class

    matrix = np.array([[40, 5, 1], [3, 20, 2], [0, 4, 15]])
    iou, dice = iou_per_class(matrix), dice_per_class(matrix)
    assert np.allclose(dice, 2 * iou / (1 + iou))


def test_perfect_prediction_has_no_dice_loss():
    from parts.h_computer_vision.h2_unet_segmentation import dice_loss

    targets = torch.tensor([[[0, 1], [2, 2]]])
    logits = torch.nn.functional.one_hot(targets, 3).permute(0, 3, 1, 2).float() * 30.0
    assert dice_loss(logits, targets, num_classes=3).item() == pytest.approx(0.0, abs=1e-3)


def test_dice_loss_prefers_the_better_prediction_and_has_a_gradient():
    from parts.h_computer_vision.h2_unet_segmentation import dice_loss

    targets = torch.tensor([[[0, 0], [1, 1]]])
    good = (torch.nn.functional.one_hot(targets, 2).permute(0, 3, 1, 2).float() * 4.0).requires_grad_(True)
    bad = torch.zeros_like(good).requires_grad_(True)
    better, worse = dice_loss(good, targets, num_classes=2), dice_loss(bad, targets, num_classes=2)
    assert better.item() < worse.item()
    better.backward()
    assert good.grad is not None and good.grad.abs().sum() > 0, "soft Dice must be differentiable"


def test_unknown_loss_is_rejected():
    from parts.h_computer_vision.h2_unet_segmentation import segmentation_loss

    targets = torch.zeros(1, 2, 2, dtype=torch.long)
    with pytest.raises(ValueError, match="unknown loss"):
        segmentation_loss(torch.zeros(1, 2, 2, 2), targets, kind="focal", num_classes=2)


# -- H2: the boundary band ---------------------------------------------------


def test_boundary_band_finds_the_perimeter_not_the_interior():
    from parts.h_computer_vision.h2_unet_segmentation import boundary_band

    mask = np.zeros((1, 11, 11), dtype=np.int64)
    mask[0, 3:8, 3:8] = 1  # a 5x5 square
    band = boundary_band(mask, band=1)
    assert band[0, 3, 3], "a corner of the square is on the boundary"
    assert band[0, 2, 3], "the pixel just outside is on the boundary"
    assert not band[0, 5, 5], "the centre of the square is not a boundary"
    assert not band[0, 0, 0], "far background is not a boundary"


def test_wider_band_covers_more_pixels():
    from parts.h_computer_vision.h2_unet_segmentation import boundary_band

    mask = np.zeros((1, 21, 21), dtype=np.int64)
    mask[0, 6:15, 6:15] = 1
    counts = [boundary_band(mask, band=width).sum() for width in (1, 2, 3)]
    assert counts[0] < counts[1] < counts[2]


def test_uniform_mask_has_no_boundary():
    from parts.h_computer_vision.h2_unet_segmentation import boundary_band

    assert not boundary_band(np.ones((2, 8, 8), dtype=np.int64), band=2).any()


def test_unet_skips_change_capacity_but_not_output_shape():
    from parts.h_computer_vision.h2_unet_segmentation import make_unet

    from mlladder import torchkit

    images = torch.randn(2, 1, 32, 32)
    with_skips = make_unet(num_classes=4, width=8, skips=True)()
    without = make_unet(num_classes=4, width=8, skips=False)()
    assert with_skips(images).shape == without(images).shape == (2, 4, 32, 32)
    # The no-skip arm keeps its decoder; only the concatenated channels go.
    assert torchkit.count_parameters(without) < torchkit.count_parameters(with_skips)
    assert torchkit.count_parameters(without) > 0.8 * torchkit.count_parameters(with_skips)


def test_h2_runs():
    from parts.h_computer_vision.h2_unet_segmentation import main

    assert (
        main(
            [
                "--train-images",
                "60",
                "--test-images",
                "40",
                "--epochs",
                "1",
                "--width",
                "4",
                "--losses",
                "cross_entropy",
                "dice",
            ]
        )
        == 0
    )


# -- H3: OCR metrics ---------------------------------------------------------


def test_edit_distance_on_known_pairs():
    from parts.h_computer_vision.h3_crnn_ctc import edit_distance

    assert edit_distance("", "") == 0
    assert edit_distance("ABC", "ABC") == 0
    assert edit_distance("ABC", "ABD") == 1  # substitution
    assert edit_distance("ABC", "AC") == 1  # deletion
    assert edit_distance("AC", "ABC") == 1  # insertion
    assert edit_distance("", "ABC") == 3
    assert edit_distance("KITTEN", "SITTING") == 3  # the textbook case


def test_edit_distance_is_symmetric():
    from parts.h_computer_vision.h3_crnn_ctc import edit_distance

    for a, b in (("ABC", "AXCY"), ("", "Z"), ("LONGER", "LNGER")):
        assert edit_distance(a, b) == edit_distance(b, a)


def test_one_deletion_costs_one_not_everything_after_it():
    """The reason CER is an edit distance and not a per-position mismatch."""
    from parts.h_computer_vision.h3_crnn_ctc import character_error_rate, edit_distance

    truth, dropped = "ABCDEFGH", "ABDEFGH"
    assert edit_distance(dropped, truth) == 1
    assert character_error_rate([dropped], [truth]) == pytest.approx(1 / 8)


def test_cer_pools_over_the_corpus():
    """Pooled, not averaged per line: the denominator is total characters."""
    from parts.h_computer_vision.h3_crnn_ctc import character_error_rate

    # One error in a 2-char line and none in an 8-char line: 1/10, not 1/4.
    assert character_error_rate(["AX", "ABCDEFGH"], ["AB", "ABCDEFGH"]) == pytest.approx(0.1)


def test_cer_can_exceed_one_when_the_model_hallucinates():
    from parts.h_computer_vision.h3_crnn_ctc import character_error_rate

    assert character_error_rate(["ABCDEFGHIJ"], ["A"]) > 1.0


def test_wer_counts_words_not_characters():
    from parts.h_computer_vision.h3_crnn_ctc import character_error_rate, word_error_rate

    truth, prediction = "ALPHA BETA GAMMA", "ALPHA BXTA GAMMA"
    assert word_error_rate([prediction], [truth]) == pytest.approx(1 / 3)
    # One wrong character ruins one word, so WER is much the worse metric.
    assert word_error_rate([prediction], [truth]) > character_error_rate([prediction], [truth])


def test_wer_is_zero_for_a_perfect_transcription():
    from parts.h_computer_vision.h3_crnn_ctc import word_error_rate

    assert word_error_rate(["ALPHA BETA"], ["ALPHA BETA"]) == 0.0


def test_wer_handles_a_dropped_word():
    from parts.h_computer_vision.h3_crnn_ctc import word_error_rate

    assert word_error_rate(["ALPHA GAMMA"], ["ALPHA BETA GAMMA"]) == pytest.approx(1 / 3)


# -- H3: CTC decoding --------------------------------------------------------


def one_hot_columns(symbols: list[int], *, num_symbols: int) -> np.ndarray:
    """Saturated log-probabilities that put all mass on the given symbol path."""
    columns = np.full((len(symbols), num_symbols), -30.0, dtype=np.float32)
    for row, symbol in enumerate(symbols):
        columns[row, symbol] = 0.0
    return columns


def test_greedy_decode_collapses_repeats_and_drops_blanks():
    from parts.h_computer_vision.h3_crnn_ctc import BLANK, encode_text, greedy_decode

    a, b = encode_text("A")[0], encode_text("B")[0]
    # A A blank A B -> "AAB": the blank separates the repeated A.
    path = [a, a, BLANK, a, b]
    assert greedy_decode(one_hot_columns(path, num_symbols=40)) == "AAB"


def test_greedy_decode_merges_repeats_without_a_blank():
    """The defining CTC behaviour: 'AA' with no blank between collapses to 'A'."""
    from parts.h_computer_vision.h3_crnn_ctc import encode_text, greedy_decode

    a = encode_text("A")[0]
    assert greedy_decode(one_hot_columns([a, a, a], num_symbols=40)) == "A"


def test_all_blank_decodes_to_empty():
    from parts.h_computer_vision.h3_crnn_ctc import BLANK, greedy_decode

    assert greedy_decode(one_hot_columns([BLANK] * 6, num_symbols=40)) == ""


def test_beam_and_greedy_agree_when_the_path_is_certain():
    """With saturated posteriors there is one alignment, so they cannot differ."""
    from parts.h_computer_vision.h3_crnn_ctc import BLANK, beam_decode, encode_text, greedy_decode

    ids = encode_text("CODE")
    path = [ids[0], BLANK, ids[1], BLANK, ids[2], BLANK, ids[3]]
    columns = one_hot_columns(path, num_symbols=40)
    assert greedy_decode(columns) == "CODE"
    assert beam_decode(columns, beam_width=5) == "CODE"


def test_beam_search_can_beat_greedy_on_split_mass():
    r"""The case beam search exists for.

    Greedy takes the best *alignment*; the beam sums over alignments of the
    same string. Here two columns each split mass between blank and 'A', so
    the most likely single alignment is blank-blank (empty), while the string
    "A" collects more total probability. Greedy returns "", the beam "A".
    """
    from parts.h_computer_vision.h3_crnn_ctc import BLANK, beam_decode, encode_text, greedy_decode

    a = encode_text("A")[0]
    columns = np.full((2, 40), -30.0, dtype=np.float32)
    for row in range(2):
        columns[row, BLANK] = np.log(0.55)
        columns[row, a] = np.log(0.45)
    assert greedy_decode(columns) == ""
    assert beam_decode(columns, beam_width=8) == "A"


def test_encode_text_round_trips_through_decode_ids():
    from parts.h_computer_vision.h3_crnn_ctc import decode_ids, encode_text

    for text in ("CODE", "ALPHA BETA", "Z0", "W1DTH 42"):
        assert decode_ids(encode_text(text)) == text


def test_encode_text_never_emits_the_blank():
    """A target containing the blank id would make the CTC loss meaningless."""
    from parts.h_computer_vision.h3_crnn_ctc import BLANK, encode_text

    from mlladder import synth

    assert BLANK not in encode_text(synth.ALPHABET + " ")


# -- H4: the attention decoder ----------------------------------------------


def test_attention_decoder_reuses_h3s_encoder_shape():
    from parts.h_computer_vision.h4_attention_ocr import EOS, make_attention_recognizer

    model = make_attention_recognizer(num_symbols=EOS + 1, channels=8, hidden=16)()
    memory = model.encode(torch.randn(3, 1, 32, 128))
    # 128 px of width, pooled twice, is 32 columns of 2*hidden features.
    assert memory.shape == (3, 32, 32)


def test_teacher_forced_pass_emits_one_logit_per_target_position():
    from parts.h_computer_vision.h4_attention_ocr import EOS, make_attention_recognizer

    model = make_attention_recognizer(num_symbols=EOS + 1, channels=8, hidden=16)()
    logits = model(torch.randn(2, 1, 32, 128), torch.zeros(2, 7, dtype=torch.long))
    assert logits.shape == (2, 7, EOS + 1)


def test_transcribe_stops_and_stays_stopped_after_eos():
    """A finished row must not resume emitting while the batch decodes on."""
    from parts.h_computer_vision.h4_attention_ocr import EOS, make_attention_recognizer

    torch.manual_seed(0)
    model = make_attention_recognizer(num_symbols=EOS + 1, channels=8, hidden=16)()
    # Force EOS to win at every step by loading the head's bias.
    with torch.no_grad():
        model.head.bias.fill_(-20.0)
        model.head.bias[EOS] = 20.0
    emitted, maps = model.transcribe(torch.randn(2, 1, 32, 128), max_length=6)
    assert (emitted[:, 0] == EOS).all()
    assert emitted.shape[1] == 1, "decoding should stop as soon as every row is finished"
    assert maps.shape[0] == 2


def test_attention_weights_are_a_distribution_over_columns():
    from parts.h_computer_vision.h4_attention_ocr import EOS, make_attention_recognizer

    model = make_attention_recognizer(num_symbols=EOS + 1, channels=8, hidden=16)()
    _emitted, maps = model.transcribe(torch.randn(2, 1, 32, 128), max_length=4)
    assert torch.allclose(maps.sum(dim=-1), torch.ones_like(maps.sum(dim=-1)), atol=1e-5)
    assert (maps >= 0).all()


def test_decode_symbols_drops_the_control_tokens():
    from parts.h_computer_vision.h3_crnn_ctc import encode_text
    from parts.h_computer_vision.h4_attention_ocr import EOS, SOS, decode_symbols

    ids = [SOS, *encode_text("CODE"), EOS]
    assert decode_symbols(ids) == "CODE"


def test_error_breakdown_separates_too_long_from_too_short():
    from parts.h_computer_vision.h4_attention_ocr import error_breakdown

    truths = ["ABCD", "ABCD"]
    looping = error_breakdown(["ABCDABCDABCD", "ABCDABCD"], truths)
    truncating = error_breakdown(["A", ""], truths)
    assert looping["length_bias"] > 0 and looping["overlong_rate"] == 1.0
    assert truncating["length_bias"] < 0 and truncating["empty_rate"] == 0.5


def test_h4_runs_without_the_pretrained_arm():
    """The trained arms must work when the optional `ocr` extra is absent."""
    from parts.h_computer_vision.h4_attention_ocr import main

    assert (
        main(
            [
                "--train-lines",
                "80",
                "--test-lines",
                "30",
                "--length-test-lines",
                "20",
                "--epochs",
                "1",
                "--channels",
                "4",
                "--hidden",
                "16",
                "--with-trocr",
                "0",
            ]
        )
        == 0
    )
