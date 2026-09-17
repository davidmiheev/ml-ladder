"""Part H's synthetic data — the ground truth has to actually describe the image.

Every rung in part H is scored against these labels, so a generator bug becomes
a wrong headline number rather than a crash. The load-bearing tests are the ones
that check agreement *between* an image and its annotation: ink inside the box,
mask inside the shape, glyphs at the aspect ratio the font drew them at.
"""

from __future__ import annotations

import numpy as np

from mlladder import synth


def ink_width(strip: np.ndarray, *, threshold: int = 200) -> int:
    """Rightmost dark column + 1, i.e. how much of the strip the text uses."""
    columns = np.flatnonzero((strip < threshold).any(axis=0))
    return int(columns[-1]) + 1 if len(columns) else 0


# -- shapes ------------------------------------------------------------------


def test_shape_boxes_lie_inside_the_image():
    for scene in synth.shapes(12, size=64, rng=0):
        assert scene.boxes.min(initial=0.0) >= 0
        assert scene.boxes[:, 2].max(initial=0.0) <= 64
        assert scene.boxes[:, 3].max(initial=0.0) <= 64
        assert (scene.boxes[:, 2] > scene.boxes[:, 0]).all()
        assert (scene.boxes[:, 3] > scene.boxes[:, 1]).all()


def test_shape_mask_labels_match_the_box_labels():
    """A class present in the mask must be a class some box claims.

    Shapes paint over each other, so a label can be *missing* from the mask —
    but the mask must never invent one.
    """
    for scene in synth.shapes(12, rng=1):
        present = {int(value) - 1 for value in np.unique(scene.mask) if value != 0}
        assert present <= set(scene.labels.tolist())


def test_shape_mask_is_confined_to_the_boxes():
    for scene in synth.shapes(8, size=64, rng=2):
        covered = np.zeros_like(scene.mask, dtype=bool)
        for x1, y1, x2, y2 in scene.boxes.astype(int):
            covered[y1:y2, x1:x2] = True
        assert not (scene.mask != 0)[~covered].any(), "mask paints outside every box"


def test_shape_overlap_is_capped():
    scenes = synth.shapes(30, max_overlap=0.3, rng=3)
    worst = 0.0
    for scene in scenes:
        for index, box in enumerate(scene.boxes):
            others = np.delete(scene.boxes, index, axis=0)
            if len(others):
                worst = max(worst, float(synth._iou_numpy(box, others).max()))
    assert worst <= 0.3 + 1e-6, f"scenes are more cluttered than requested: {worst}"


def test_stack_masks_shapes_agree():
    images, masks = synth.stack_masks(synth.shapes(5, size=48, rng=4))
    assert images.shape == (5, 48, 48)
    assert masks.shape == (5, 48, 48)


# -- text lines --------------------------------------------------------------


def test_rendered_text_keeps_the_font_aspect_ratio():
    """Regression test: the line must not be squeezed to fit the scratch canvas.

    ``render_text`` draws on a canvas several times the target width and then
    trims to the ink. The first version trimmed with ``Image.getbbox``, which
    returns non-*zero* pixels — and on a white background that is the whole
    canvas, so every line was silently resized by the canvas-to-target ratio.
    The ink width therefore has to match what the font itself reports.
    """
    from PIL import Image, ImageDraw

    text, height = "MODE", 32
    face = synth.load_font("DejaVuSans.ttf", height - 10)
    draw = ImageDraw.Draw(Image.new("L", (8, 8)))
    expected = draw.textbbox((2, height // 2), text, font=face, anchor="lm")[2]

    strip = synth.render_text(text, font="DejaVuSans.ttf", height=height, width=128, noise=0.0)
    assert abs(ink_width(strip) - expected) <= 4, f"ink {ink_width(strip)} vs font {expected}"


def test_longer_text_uses_more_of_the_strip():
    short = synth.render_text("AB", font="DejaVuSans.ttf", width=128, noise=0.0)
    long = synth.render_text("ABCDEF", font="DejaVuSans.ttf", width=128, noise=0.0)
    assert ink_width(short) < ink_width(long)


def test_text_never_overflows_the_strip():
    """Overlong text is scaled, not cropped — a cropped line mislabels itself."""
    strip = synth.render_text("COLUMN DEPTH LAYER", font="DejaVuSerif.ttf", height=32, width=128, noise=0.0)
    assert strip.shape == (32, 128)
    assert ink_width(strip) <= 128


def test_text_lines_use_only_the_requested_fonts():
    lines = synth.text_lines(20, fonts=synth.HELD_OUT_FONTS, rng=5)
    assert {line.font for line in lines} <= set(synth.HELD_OUT_FONTS)
    assert all(set(line.text) <= set(synth.ALPHABET + " ") for line in lines)


def test_train_and_held_out_fonts_are_disjoint():
    assert not set(synth.TRAIN_FONTS) & set(synth.HELD_OUT_FONTS)


# -- documents ---------------------------------------------------------------


def test_document_boxes_describe_their_texts():
    pages = synth.documents(12, rng=6)
    assert any(len(page.boxes) > 1 for page in pages), "every page has one line; slots are not being used"
    for page in pages:
        assert len(page.boxes) == len(page.texts)
        assert page.image.shape == (160, 256)
        assert (page.boxes[:, 3] - page.boxes[:, 1] == 32).all(), "crop height must match text_lines' height"


def test_document_lines_do_not_overlap():
    """H5 attributes error per line, which needs the lines to be separable."""
    for page in synth.documents(20, rng=7):
        for i in range(len(page.boxes)):
            for j in range(i + 1, len(page.boxes)):
                assert float(synth._iou_numpy(page.boxes[i], page.boxes[j][None])[0]) < 0.1


def test_document_boxes_contain_ink():
    """An empty box would make the detector chase something invisible."""
    for page in synth.documents(10, rng=8):
        for x1, y1, x2, y2 in page.boxes.astype(int):
            crop = page.image[y1:y2, x1:x2]
            assert (crop < 160).mean() > 0.01, "box holds no text"


# -- reproducibility across environments -------------------------------------

# Rendered-pixel digests, verified identical on two machines with different
# library versions: locally (matplotlib 3.11.2, Pillow 12.3.0) and on a Colab
# T4 (matplotlib 3.10.0, Pillow 11.3.0). Every number in part H is computed
# from these bytes, so if a dependency bump changes them the committed metrics
# stop being reproducible and the part needs re-sweeping — which is exactly
# what a failure here means. Re-bless deliberately, never reflexively.
RENDER_DIGESTS = {
    "DejaVuSans.ttf": "686abf10572506da",
    "cmr10.ttf": "471df2226a487735",
}
SCENE_DIGEST = "99f78e4f52286ea7"
PAGE_DIGEST = "4d6dcb9a3e01d4bd"


def digest(array: np.ndarray) -> str:
    import hashlib

    return hashlib.sha256(array.tobytes()).hexdigest()[:16]


def test_rendered_text_is_byte_identical_across_environments():
    for font, expected in RENDER_DIGESTS.items():
        strip = synth.render_text("MODEL BATCH", font=font, height=32, width=128, noise=0.0)
        assert digest(strip) == expected, (
            f"{font} renders differently here; part H's committed numbers were produced from the "
            f"old bytes and would need a re-sweep"
        )


def test_generated_scenes_and_pages_are_byte_identical_across_environments():
    assert digest(synth.shapes(3, rng=0)[0].image) == SCENE_DIGEST
    assert digest(synth.documents(2, rng=0)[0].image) == PAGE_DIGEST


def test_every_declared_font_is_actually_present():
    """A missing face would silently change which letterforms a split contains."""
    for font in synth.TRAIN_FONTS + synth.HELD_OUT_FONTS:
        assert (synth.font_dir() / font).exists(), f"{font} is missing from {synth.font_dir()}"
