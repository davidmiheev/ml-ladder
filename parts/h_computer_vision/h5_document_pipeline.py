"""H5. Two-stage document pipeline — detect text regions, then recognize, with end-to-end error attributed to each stage

Lab reference: [open-mmlab/mmocr — detection and recognition reference implementations](https://github.com/open-mmlab/mmocr)

H1 found text-shaped things. H3 read them. Bolting the two together is how every
classical OCR system works, and it introduces a failure mode that neither rung
can exhibit alone: **the second stage consumes the first stage's mistakes.**

That makes the interesting question not "how good is the pipeline" but "whose
fault is the error", and a single end-to-end number cannot answer it. So this
rung measures the same recognizer three ways:

1. on **ground-truth crops** — the recognizer alone, detection assumed perfect;
2. on **detected crops** — the whole pipeline, as deployed;
3. on **deliberately jittered ground-truth crops** — detection error simulated
   at a known magnitude, with the detector taken out of the loop entirely.

The third is what turns the gap between the first two from an observation into
an explanation. H1 measured its boxes at a mean IoU of 0.79; arm 3 answers
whether a 0.79-IoU crop is still legible, which is the quantity that decides
whether this pipeline's detector is good enough.

Run with ``ladder run H5`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, experiment, plotting, synth, torchkit, vision
from parts.h_computer_vision.h1_detection_from_scratch import (
    decode,
    detection_loss,
    encode_targets,
    iou_matrix,
    make_detector,
    mean_average_precision,
    nms,
)
from parts.h_computer_vision.h3_crnn_ctc import (
    BLANK,
    character_error_rate,
    encode_text,
    greedy_decode,
    make_crnn,
    word_error_rate,
)


def crop_line(page: NDArray[np.uint8], box, *, height: int, width: int) -> NDArray[np.uint8]:
    """Cut a box out of a page and resize it to the recognizer's input size.

    Boxes are clamped to the page and given a minimum extent, because a
    detector early in training emits degenerate boxes and a zero-width crop
    would crash the resize rather than simply score badly.
    """
    from PIL import Image

    page_height, page_width = page.shape
    x1 = int(np.clip(round(float(box[0])), 0, page_width - 1))
    y1 = int(np.clip(round(float(box[1])), 0, page_height - 1))
    x2 = int(np.clip(round(float(box[2])), x1 + 1, page_width))
    y2 = int(np.clip(round(float(box[3])), y1 + 1, page_height))
    patch = Image.fromarray(page[y1:y2, x1:x2])
    return np.asarray(patch.resize((width, height), Image.BILINEAR), dtype=np.uint8)


def jitter_box(box, *, pixels: float, rng: np.random.Generator) -> NDArray[np.float32]:
    """Perturb each edge independently by up to ``pixels``.

    Independent per edge on purpose: a uniform shift moves the crop but keeps
    the text inside it, while independent edges change the *framing* — cutting
    off a first letter or padding in a neighbour's descender, which is what a
    real detector gets wrong.
    """
    offsets = rng.uniform(-pixels, pixels, size=4)
    return np.asarray(box, dtype=np.float32) + offsets.astype(np.float32)


def read_crops(model, crops: list[NDArray[np.uint8]], *, device, batch_size: int = 128) -> list[str]:
    """Run the CTC recognizer over a list of crops and greedily decode each."""
    torch = vision.require_torch()
    if not crops:
        return []
    stacked = np.stack(crops).astype(np.float32) / 255.0
    tensor = torch.from_numpy(1.0 - stacked).unsqueeze(1).to(device)
    model.eval()
    outputs: list[str] = []
    with torch.no_grad():
        for start in range(0, len(tensor), batch_size):
            logits = model(tensor[start : start + batch_size])
            log_probabilities = torch.log_softmax(logits, dim=-1).cpu().numpy()
            outputs.extend(greedy_decode(row) for row in log_probabilities)
    return outputs


def reading_order(boxes: NDArray[np.float32]) -> NDArray[np.int64]:
    """Top-to-bottom, then left-to-right — the order a page is read in.

    The page-level metric concatenates lines, so it depends on this ordering.
    Sorting by the box centre rather than its top edge keeps two lines at
    slightly different heights in a stable order.
    """
    if len(boxes) == 0:
        return np.zeros(0, dtype=np.int64)
    centres = np.stack([(boxes[:, 1] + boxes[:, 3]) / 2, (boxes[:, 0] + boxes[:, 2]) / 2], axis=1)
    return np.lexsort((centres[:, 1], centres[:, 0]))


def page_transcript(texts: list[str], boxes: NDArray[np.float32]) -> str:
    """One string per page, lines joined in reading order.

    This is the end-to-end metric and it is deliberately unforgiving: a missed
    line becomes a run of deletions, a spurious line a run of insertions, and
    lines read in the wrong order cost the edit distance between them. A
    per-line average would hide all three by only scoring lines that were
    found.
    """
    order = reading_order(boxes)
    return " ".join(texts[index] for index in order)


@experiment(
    "h5_document_pipeline",
    defaults={
        "train_pages": 2500,
        "test_pages": 500,
        "page_height": 160,
        "page_width": 256,
        "line_height": 32,
        "crop_width": 128,
        "detector_width": 16,
        "detector_epochs": 14,
        "detector_lr": 2e-3,
        "recognizer_channels": 16,
        "recognizer_hidden": 96,
        "recognizer_epochs": 12,
        "batch_size": 32,
        "recognizer_lr": 2e-3,
        "no_object_weight": 0.2,
        "box_weight": 5.0,
        "score_threshold": 0.3,
        "nms_threshold": 0.3,
        "match_iou": 0.5,
        "jitter_pixels": [0.0, 2.0, 4.0, 8.0, 16.0],
    },
    description="Detect text lines then recognize them, attributing page-level error to each stage.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Two-stage document pipeline — detect text regions, then recognize, with end-to-end error attributed to each stage"""
    torch = vision.require_torch()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)
    print(f"[h5] device: {device}")

    page_shape = (exp["page_height"], exp["page_width"])
    grid = (exp["page_height"] // 8, exp["page_width"] // 8)  # the detector pools three times
    geometry = {
        "page_height": exp["page_height"],
        "page_width": exp["page_width"],
        "line_height": exp["line_height"],
    }
    train_pages = synth.documents(exp["train_pages"], fonts=synth.TRAIN_FONTS, rng=rng, **geometry)
    test_pages = synth.documents(exp["test_pages"], fonts=synth.TRAIN_FONTS, rng=rng, **geometry)
    print(
        f"[h5] {len(train_pages)} train pages / {len(test_pages)} test, "
        f"{sum(len(page.boxes) for page in train_pages)} train lines; grid {grid[0]}x{grid[1]}"
    )

    def pages_to_tensor(pages):
        images = np.stack([page.image for page in pages]).astype(np.float32) / 255.0
        return torch.from_numpy(images).unsqueeze(1)

    x_train_pages = pages_to_tensor(train_pages).to(device)
    x_test_pages = pages_to_tensor(test_pages).to(device)

    summary: dict[str, float] = {}
    curves: dict[str, list[float]] = {}

    # -- stage 1: the text-line detector -------------------------------------
    objectness, boxes, labels, collisions = encode_targets(train_pages, grid=grid, size=page_shape, num_classes=1)
    objectness, boxes, labels = objectness.to(device), boxes.to(device), labels.to(device)
    total_lines = sum(len(page.boxes) for page in train_pages)
    summary["cell_collision_rate"] = collisions / max(total_lines, 1)
    print(f"[h5] {collisions} of {total_lines} lines lost to cell collisions ({summary['cell_collision_rate']:.2%})")

    torchkit.seed_torch(exp.seed)
    detector = make_detector(grid=grid, width=exp["detector_width"], num_classes=1)().to(device)
    optimizer = torch.optim.AdamW(detector.parameters(), lr=exp["detector_lr"], weight_decay=1e-4)
    detector_losses: list[float] = []
    for epoch in range(1, exp["detector_epochs"] + 1):
        detector.train()
        permutation = torch.randperm(len(x_train_pages), device=device)
        epoch_losses = []
        for start in range(0, len(x_train_pages), exp["batch_size"]):
            index = permutation[start : start + exp["batch_size"]]
            loss, *_ = detection_loss(
                detector(x_train_pages[index]),
                objectness[index],
                boxes[index],
                labels[index],
                no_object_weight=exp["no_object_weight"],
                box_weight=exp["box_weight"],
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        detector_losses.append(float(np.mean(epoch_losses)))
        if epoch % max(exp["detector_epochs"] // 3, 1) == 0:
            print(f"[h5] detector epoch {epoch:>3}  loss={detector_losses[-1]:.4f}")
    curves["detector"] = detector_losses
    summary["detector/parameters"] = float(torchkit.count_parameters(detector))

    # -- stage 2: the recognizer, trained on ground-truth crops --------------
    # Training on ground-truth crops and testing on detected ones is the whole
    # point: the distribution shift between them *is* the detector's error.
    train_crops, train_texts = [], []
    for page in train_pages:
        for box, text in zip(page.boxes, page.texts, strict=True):
            train_crops.append(crop_line(page.image, box, height=exp["line_height"], width=exp["crop_width"]))
            train_texts.append(text)
    print(f"[h5] recognizer trains on {len(train_crops)} ground-truth crops")

    num_symbols = len(synth.ALPHABET) + 2
    crop_stack = np.stack(train_crops).astype(np.float32) / 255.0
    x_crops = torch.from_numpy(1.0 - crop_stack).unsqueeze(1).to(device)
    targets = [encode_text(text) for text in train_texts]
    target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long, device=device)
    flat_targets = torch.tensor([symbol for target in targets for symbol in target], dtype=torch.long, device=device)
    offsets = torch.cumsum(torch.cat([torch.zeros(1, dtype=torch.long, device=device), target_lengths]), dim=0)

    torchkit.seed_torch(exp.seed)
    recognizer = make_crnn(
        num_symbols=num_symbols, channels=exp["recognizer_channels"], hidden=exp["recognizer_hidden"]
    )().to(device)
    optimizer = torch.optim.AdamW(recognizer.parameters(), lr=exp["recognizer_lr"], weight_decay=1e-4)
    criterion = torch.nn.CTCLoss(blank=BLANK, zero_infinity=True)
    recognizer_losses: list[float] = []
    for epoch in range(1, exp["recognizer_epochs"] + 1):
        recognizer.train()
        permutation = torch.randperm(len(x_crops), device=device)
        epoch_losses = []
        for start in range(0, len(x_crops), 64):
            index = permutation[start : start + 64]
            logits = recognizer(x_crops[index])
            log_probabilities = torch.log_softmax(logits, dim=-1).permute(1, 0, 2)
            batch_targets = torch.cat([flat_targets[offsets[i] : offsets[i + 1]] for i in index])
            input_lengths = torch.full((len(index),), logits.shape[1], dtype=torch.long, device=device)
            loss = criterion(log_probabilities, batch_targets, input_lengths, target_lengths[index])
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(recognizer.parameters(), 5.0)
            optimizer.step()
            epoch_losses.append(loss.item())
        recognizer_losses.append(float(np.mean(epoch_losses)))
        if epoch % max(exp["recognizer_epochs"] // 3, 1) == 0:
            print(f"[h5] recognizer epoch {epoch:>3}  ctc loss={recognizer_losses[-1]:.4f}")
    curves["recognizer"] = recognizer_losses
    summary["recognizer/parameters"] = float(torchkit.count_parameters(recognizer))

    # -- arm 1: ground-truth crops (detection assumed perfect) ---------------
    truth_pages = [page_transcript(list(page.texts), page.boxes) for page in test_pages]
    gt_crops, gt_index = [], []
    for page_id, page in enumerate(test_pages):
        for box in page.boxes:
            gt_crops.append(crop_line(page.image, box, height=exp["line_height"], width=exp["crop_width"]))
            gt_index.append(page_id)
    gt_predictions = read_crops(recognizer, gt_crops, device=device)
    gt_lines = [text for page in test_pages for text in page.texts]
    summary["oracle/line_cer"] = character_error_rate(gt_predictions, gt_lines)
    summary["oracle/line_wer"] = word_error_rate(gt_predictions, gt_lines)

    per_page: dict[int, list[str]] = {}
    for page_id, prediction in zip(gt_index, gt_predictions, strict=True):
        per_page.setdefault(page_id, []).append(prediction)
    oracle_pages = [
        page_transcript(per_page.get(page_id, []), test_pages[page_id].boxes) for page_id in range(len(test_pages))
    ]
    summary["oracle/page_cer"] = character_error_rate(oracle_pages, truth_pages)
    summary["oracle/page_wer"] = word_error_rate(oracle_pages, truth_pages)
    print(
        f"[h5] arm 1 (ground-truth crops):  line CER={summary['oracle/line_cer']:.4f}  "
        f"page CER={summary['oracle/page_cer']:.4f}"
    )

    # -- arm 2: the real pipeline --------------------------------------------
    detector.eval()
    with torch.no_grad():
        raw = torch.cat([detector(chunk) for chunk in x_test_pages.split(64)])
    detections = decode(raw, grid=grid, size=page_shape, score_threshold=exp["score_threshold"])

    pipeline_pages: list[str] = []
    matched_ious: list[float] = []
    matched_predictions: list[str] = []
    matched_truths: list[str] = []
    kept_per_page: list[tuple[NDArray[np.float32], NDArray[np.float32]]] = []
    found = missed = spurious = 0
    example_pages: list[tuple[int, NDArray[np.float32], list[str]]] = []

    for page_id, (page, (candidate_boxes, scores, _classes)) in enumerate(zip(test_pages, detections, strict=True)):
        if len(candidate_boxes):
            keep = nms(candidate_boxes, scores, exp["nms_threshold"])
            kept_boxes, kept_scores = candidate_boxes[keep], scores[keep]
        else:
            kept_boxes = np.zeros((0, 4), dtype=np.float32)
            kept_scores = np.zeros(0, dtype=np.float32)
        kept_per_page.append((kept_boxes, kept_scores))

        crops = [crop_line(page.image, box, height=exp["line_height"], width=exp["crop_width"]) for box in kept_boxes]
        texts = read_crops(recognizer, crops, device=device)
        pipeline_pages.append(page_transcript(texts, kept_boxes))
        if len(example_pages) < 4:
            example_pages.append((page_id, kept_boxes, texts))

        # Per-line accounting, which separates "never found" from "found badly".
        # A ground-truth line is found when some surviving box clears
        # ``match_iou``; the detection that claims it also supplies the text
        # that line's own CER is computed from.
        overlaps = iou_matrix(kept_boxes, page.boxes)
        claimed: set[int] = set()
        for column in range(len(page.boxes)):
            best_iou = float(overlaps[:, column].max()) if len(kept_boxes) else 0.0
            matched_ious.append(best_iou)
            if best_iou >= exp["match_iou"]:
                found += 1
                row = int(np.argmax(overlaps[:, column]))
                claimed.add(row)
                matched_predictions.append(texts[row])
                matched_truths.append(page.texts[column])
            else:
                missed += 1
        spurious += len(kept_boxes) - len(claimed)

    summary["pipeline/page_cer"] = character_error_rate(pipeline_pages, truth_pages)
    summary["pipeline/page_wer"] = word_error_rate(pipeline_pages, truth_pages)
    summary["detector/recall@match"] = found / max(found + missed, 1)
    summary["detector/spurious_per_page"] = spurious / len(test_pages)
    summary["detector/matched_iou_mean"] = float(np.mean(matched_ious)) if matched_ious else 0.0
    if matched_truths:
        # Recognition quality on the lines detection actually found — the
        # pipeline's error minus everything caused by missing a line outright.
        summary["pipeline/matched_line_cer"] = character_error_rate(matched_predictions, matched_truths)
    map_predictions = [
        (page_id, 0, float(score), box)
        for page_id, (kept, kept_scores) in enumerate(kept_per_page)
        for box, score in zip(kept, kept_scores, strict=True)
    ]
    map_truths = [(page_id, 0, box) for page_id, page in enumerate(test_pages) for box in page.boxes]
    summary["detector/mAP@0.5"], _ = mean_average_precision(
        map_predictions, map_truths, num_classes=1, iou_threshold=0.5
    )
    print(
        f"[h5] arm 2 (detected crops):      page CER={summary['pipeline/page_cer']:.4f}  "
        f"recall={summary['detector/recall@match']:.4f}  mAP@0.5={summary['detector/mAP@0.5']:.4f}  "
        f"matched IoU={summary['detector/matched_iou_mean']:.3f}  spurious/page={summary['detector/spurious_per_page']:.2f}"
    )

    # -- the attribution -----------------------------------------------------
    summary["detection_cost/page_cer"] = summary["pipeline/page_cer"] - summary["oracle/page_cer"]
    if summary["pipeline/page_cer"] > 0:
        summary["detection_share_of_error"] = summary["detection_cost/page_cer"] / summary["pipeline/page_cer"]
    print(
        f"[h5] attribution: recognizer alone {summary['oracle/page_cer']:.4f} CER, "
        f"pipeline {summary['pipeline/page_cer']:.4f}, so detection costs "
        f"{summary['detection_cost/page_cer']:+.4f}"
        + (
            f" ({summary['detection_share_of_error']:.0%} of the total)"
            if "detection_share_of_error" in summary
            else ""
        )
    )

    # -- arm 3: jittered ground-truth crops, the detector removed ------------
    jitter_curve: list[float] = []
    for pixels in exp["jitter_pixels"]:
        jitter_rng = np.random.default_rng(exp.seed)
        crops = [
            crop_line(
                test_pages[page_id].image,
                jitter_box(box, pixels=pixels, rng=jitter_rng),
                height=exp["line_height"],
                width=exp["crop_width"],
            )
            for page_id, page in enumerate(test_pages)
            for box in page.boxes
        ]
        predictions = read_crops(recognizer, crops, device=device)
        error = character_error_rate(predictions, gt_lines)
        summary[f"jitter/{pixels}/line_cer"] = error
        jitter_curve.append(error)
        print(f"[h5] arm 3 jitter +-{pixels:>4.1f}px  line CER={error:.4f}")

    # How much crop sloppiness the recognizer tolerates before it breaks:
    # the smallest jitter that doubles the oracle error.
    baseline = summary["jitter/0.0/line_cer"] if "jitter/0.0/line_cer" in summary else summary["oracle/line_cer"]
    tolerated = [
        pixels for pixels, error in zip(exp["jitter_pixels"], jitter_curve, strict=True) if error <= 2 * baseline + 0.01
    ]
    summary["jitter/tolerated_px"] = float(max(tolerated)) if tolerated else 0.0
    print(f"[h5] recognizer tolerates up to +-{summary['jitter/tolerated_px']:.1f}px of crop error")

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "training",
        plotting.plot_curves(
            curves,
            x=list(range(1, max(len(values) for values in curves.values()) + 1)),
            title="Stage losses (different objectives; only the trend is comparable)",
            xlabel="epoch",
            ylabel="loss",
            logy=True,
        ),
    )
    exp.log_figure(
        "attribution",
        plotting.plot_comparison(
            {
                "recognizer alone\n(GT crops)": summary["oracle/page_cer"],
                "full pipeline\n(detected crops)": summary["pipeline/page_cer"],
            },
            title="Page-level CER: what detection costs",
            ylabel="page CER",
        ),
    )
    exp.log_figure(
        "jitter",
        plotting.plot_curves(
            {"line CER": jitter_curve},
            x=exp["jitter_pixels"],
            title="Recognition against simulated detection error (detector not involved)",
            xlabel="per-edge box jitter (px)",
            ylabel="line CER",
            marker="o",
        ),
    )
    exp.log_figure(
        "matched_iou",
        plotting.plot_histogram(
            matched_ious,
            bins=np.linspace(0, 1, 26),
            markers={f"match at {exp['match_iou']}": exp["match_iou"]},
            title="IoU of the best detection per ground-truth line",
            xlabel="IoU",
            ylabel="lines",
        ),
    )
    exp.log_figure(
        "pages",
        plotting.plot_boxes(
            np.stack([test_pages[page_id].image for page_id, _, _ in example_pages]),
            [kept for _, kept, _ in example_pages],
            truths=[test_pages[page_id].boxes for page_id, _, _ in example_pages],
            nrow=2,
            title="Detected text lines (dashed grey = ground truth)",
        ),
    )

    print("[h5] example pages (pipeline | truth):")
    for page_id, _, _ in example_pages[:3]:
        print(f"  {pipeline_pages[page_id]!r}")
        print(f"  {truth_pages[page_id]!r}")
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
