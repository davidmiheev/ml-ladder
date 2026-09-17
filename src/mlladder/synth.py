"""Synthetic images with exact ground truth, for part H.

Part H measures *pipelines* — detect then recognize — and a pipeline's error has
to be attributed to a stage. That is only possible when the ground truth is
known exactly: which box, which character, which pixel. Real OCR datasets come
with annotation noise that is itself a research topic, so this module renders
its own data instead.

Three generators, all seeded and all offline:

- :func:`shapes` — grayscale scenes with boxes, class ids and a semantic mask
  (H1's detector and H2's U-Net read the same scenes, different labels);
- :func:`text_lines` — rendered words with their transcription (H3, H4);
- :func:`documents` — pages of text lines at known boxes (H5).

**Why the fonts come from matplotlib.** Rendering depends on the font file, so
using the system's fonts would make every number here a property of the machine
that produced it — this container has 59 TTFs, a Colab VM has a different set.
matplotlib ships DejaVu and Computer Modern, is already a pinned dependency, and
is therefore byte-identical everywhere the lockfile is honoured. Nothing is
downloaded and nothing is vendored.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mlladder.datasets import as_rng

# -- fonts -------------------------------------------------------------------

# Held out deliberately: the italic monospace and Computer Modern faces are the
# two most unlike the training set, so the generalization gap H3 measures is a
# gap across letterforms rather than across noise.
TRAIN_FONTS = (
    "DejaVuSans.ttf",
    "DejaVuSans-Bold.ttf",
    "DejaVuSerif.ttf",
    "DejaVuSansMono.ttf",
    "DejaVuSans-Oblique.ttf",
    "DejaVuSerif-Bold.ttf",
)
HELD_OUT_FONTS = (
    "DejaVuSansMono-Oblique.ttf",
    "DejaVuSerif-Italic.ttf",
    "cmr10.ttf",
)

# 36 symbols: upper case plus digits. Small enough that CTC's label space is
# learnable in a few hundred steps, large enough that CER is not trivial.
ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# A fixed list so that word error rate means something and no download is
# needed. Short words keep the rendered lines narrow at 32 px height.
WORDS = [
    "TIME",
    "PART",
    "FORM",
    "CASE",
    "MARK",
    "LINE",
    "TEXT",
    "WORD",
    "PAGE",
    "BOOK",
    "NOTE",
    "DATA",
    "CODE",
    "TYPE",
    "HAND",
    "HEAD",
    "FACE",
    "EDGE",
    "SIDE",
    "STEP",
    "RATE",
    "COST",
    "PLAN",
    "VIEW",
    "AREA",
    "ITEM",
    "UNIT",
    "RULE",
    "TEST",
    "FILE",
    "NAME",
    "SIZE",
    "ZONE",
    "MODE",
    "GRID",
    "CELL",
    "ROW",
    "COLUMN",
    "SCALE",
    "LABEL",
    "INPUT",
    "MODEL",
    "TRAIN",
    "SCORE",
    "ERROR",
    "VALUE",
    "INDEX",
    "TOKEN",
    "BATCH",
    "LAYER",
    "DEPTH",
    "WIDTH",
    "SHAPE",
    "IMAGE",
    "PIXEL",
    "FRAME",
    "BOUND",
    "CLASS",
    "LOGIT",
    "EPOCH",
    "GRAPH",
    "NODE",
    "PATH",
    "LOSS",
    "RANK",
    "ALPHA",
    "BETA",
    "GAMMA",
    "DELTA",
    "THETA",
    "SIGMA",
    "LAMBDA",
    "KAPPA",
    "ZETA",
    "OMEGA",
]


def font_dir() -> Path:
    """matplotlib's bundled TTF directory — pinned by the lockfile."""
    import matplotlib

    return Path(matplotlib.get_data_path()) / "fonts" / "ttf"


@lru_cache(maxsize=64)
def load_font(name: str, size: int):
    """A PIL font, cached because rendering reloads the same faces constantly."""
    from PIL import ImageFont

    path = font_dir() / name
    if not path.exists():  # pragma: no cover - a broken matplotlib install
        raise SystemExit(f"font {name} is missing from {font_dir()}; is matplotlib installed?")
    return ImageFont.truetype(str(path), size)


# -- shapes: detection and segmentation --------------------------------------

SHAPE_CLASSES = ("rectangle", "ellipse", "triangle")


@dataclass(frozen=True)
class ShapeScene:
    """One scene, with every label a part-H rung needs from it.

    ``boxes`` are ``(x1, y1, x2, y2)`` in pixels, half-open on the far corner
    so that width is ``x2 - x1``. ``mask`` holds ``label + 1`` inside each
    shape and 0 on the background, which is what a segmentation cross-entropy
    wants; later shapes paint over earlier ones, so the mask is consistent with
    what the image shows.
    """

    image: NDArray[np.uint8]  # (size, size)
    boxes: NDArray[np.float32]  # (n, 4)
    labels: NDArray[np.int64]  # (n,)
    mask: NDArray[np.int64]  # (size, size)


def _iou_numpy(box: NDArray[np.float32], others: NDArray[np.float32]) -> NDArray[np.float32]:
    """IoU of one box against many — used here only to cap scene clutter."""
    if len(others) == 0:
        return np.zeros(0, dtype=np.float32)
    x1 = np.maximum(box[0], others[:, 0])
    y1 = np.maximum(box[1], others[:, 1])
    x2 = np.minimum(box[2], others[:, 2])
    y2 = np.minimum(box[3], others[:, 3])
    overlap = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = (box[2] - box[0]) * (box[3] - box[1])
    areas = (others[:, 2] - others[:, 0]) * (others[:, 3] - others[:, 1])
    return (overlap / np.maximum(area + areas - overlap, 1e-9)).astype(np.float32)


def shapes(
    n_images: int,
    *,
    size: int = 64,
    max_objects: int = 3,
    min_side: int = 12,
    max_side: int = 28,
    max_overlap: float = 0.3,
    noise: float = 0.05,
    rng=None,
) -> list[ShapeScene]:
    """Scenes of 1..``max_objects`` shapes on a noisy background.

    Overlap is allowed up to ``max_overlap`` IoU: without any, non-max
    suppression has nothing to suppress and H1's NMS threshold would be
    unmeasurable. With too much, the ground truth stops being well defined
    because one shape hides another.

    Everything is grayscale on purpose. Colour would let a detector separate
    the three classes without ever learning their geometry, which is the one
    thing the rung is trying to make it learn.
    """
    from PIL import Image, ImageDraw

    generator = as_rng(rng)
    scenes: list[ShapeScene] = []
    for _ in range(n_images):
        background = int(generator.integers(90, 150))
        image = Image.new("L", (size, size), background)
        draw = ImageDraw.Draw(image)
        mask_image = Image.new("I", (size, size), 0)
        mask_draw = ImageDraw.Draw(mask_image)

        boxes: list[list[float]] = []
        labels: list[int] = []
        for _attempt in range(int(generator.integers(1, max_objects + 1))):
            for _retry in range(20):
                side_x = int(generator.integers(min_side, max_side + 1))
                side_y = int(generator.integers(min_side, max_side + 1))
                x1 = int(generator.integers(0, max(size - side_x, 1)))
                y1 = int(generator.integers(0, max(size - side_y, 1)))
                candidate = np.array([x1, y1, x1 + side_x, y1 + side_y], dtype=np.float32)
                if len(boxes) == 0 or _iou_numpy(candidate, np.array(boxes, dtype=np.float32)).max() <= max_overlap:
                    break
            else:
                continue

            label = int(generator.integers(0, len(SHAPE_CLASSES)))
            # Keep the fill clear of the background level, or the shape is
            # invisible and the label is a lie.
            fill = int(generator.integers(200, 256)) if background < 120 else int(generator.integers(0, 60))
            box = (float(candidate[0]), float(candidate[1]), float(candidate[2] - 1), float(candidate[3] - 1))
            if SHAPE_CLASSES[label] == "rectangle":
                draw.rectangle(box, fill=fill)
                mask_draw.rectangle(box, fill=label + 1)
            elif SHAPE_CLASSES[label] == "ellipse":
                draw.ellipse(box, fill=fill)
                mask_draw.ellipse(box, fill=label + 1)
            else:
                apex = ((box[0] + box[2]) / 2, box[1])
                triangle = [apex, (box[0], box[3]), (box[2], box[3])]
                draw.polygon(triangle, fill=fill)
                mask_draw.polygon(triangle, fill=label + 1)
            boxes.append(candidate.tolist())
            labels.append(label)

        pixels = np.asarray(image, dtype=np.float32)
        if noise > 0:
            pixels = pixels + generator.normal(0, noise * 255, pixels.shape)
        scenes.append(
            ShapeScene(
                image=np.clip(pixels, 0, 255).astype(np.uint8),
                boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
                labels=np.array(labels, dtype=np.int64),
                mask=np.asarray(mask_image, dtype=np.int64),
            )
        )
    return scenes


def stack_masks(scenes: list[ShapeScene]) -> tuple[NDArray[np.uint8], NDArray[np.int64]]:
    """(N, H, W) images and masks — the batched form H2's U-Net trains on."""
    return (
        np.stack([scene.image for scene in scenes]),
        np.stack([scene.mask for scene in scenes]),
    )


# -- text lines: recognition -------------------------------------------------


@dataclass(frozen=True)
class TextLine:
    """A rendered line and the string it renders, plus the face it used."""

    image: NDArray[np.uint8]  # (height, width)
    text: str
    font: str


def render_text(
    text: str,
    *,
    font: str,
    height: int = 32,
    width: int = 128,
    noise: float = 0.04,
    rng=None,
) -> NDArray[np.uint8]:
    """Render one string to a fixed-size grayscale strip.

    The glyphs are drawn at a size chosen to fill the height, then the whole
    strip is scaled to ``width`` if it overflows. Scaling rather than cropping
    matters: a cropped line would have characters missing from the image but
    present in the label, which teaches the recognizer to hallucinate.
    """
    from PIL import Image, ImageDraw

    generator = as_rng(rng)
    face = load_font(font, max(height - 10, 8))
    canvas = Image.new("L", (width * 4, height), 255)
    draw = ImageDraw.Draw(canvas)
    draw.text((2, height // 2), text, font=face, fill=0, anchor="lm")

    # Trim to the ink, then pad back to the target so the line starts near the
    # left edge regardless of the face's own side bearings. This must use
    # ``textbbox`` and not ``Image.getbbox``: the latter finds non-*zero*
    # pixels, and on a white (255) background that is the entire canvas — which
    # silently squeezed every line by the canvas-to-target ratio.
    bbox = draw.textbbox((2, height // 2), text, font=face, anchor="lm")
    ink = canvas.crop((0, 0, min(bbox[2] + 3, canvas.width), height))
    if ink.width > width:
        ink = ink.resize((width, height), Image.BILINEAR)
    strip = Image.new("L", (width, height), 255)
    strip.paste(ink, (0, 0))

    pixels = np.asarray(strip, dtype=np.float32)
    if noise > 0:
        pixels = pixels + generator.normal(0, noise * 255, pixels.shape)
    return np.clip(pixels, 0, 255).astype(np.uint8)


def text_lines(
    n_lines: int,
    *,
    fonts: tuple[str, ...] = TRAIN_FONTS,
    height: int = 32,
    width: int = 128,
    max_words: int = 2,
    noise: float = 0.04,
    rng=None,
) -> list[TextLine]:
    """Rendered word sequences drawn from :data:`WORDS`.

    Real words rather than random character strings, so that word error rate is
    a meaningful second metric alongside CER — and so the task has the kind of
    redundancy an attention decoder can exploit and CTC cannot.
    """
    generator = as_rng(rng)
    lines: list[TextLine] = []
    for _ in range(n_lines):
        count = int(generator.integers(1, max_words + 1))
        text = " ".join(str(generator.choice(WORDS)) for _ in range(count))
        font = str(generator.choice(fonts))
        image = render_text(text, font=font, height=height, width=width, noise=noise, rng=generator)
        lines.append(TextLine(image=image, text=text, font=font))
    return lines


# -- documents: the two-stage pipeline ---------------------------------------


@dataclass(frozen=True)
class Document:
    """A page, the boxes of its text lines, and their transcriptions.

    ``boxes[i]`` and ``texts[i]`` describe the same line, which is what lets
    H5 score recognition on ground-truth crops and on detected crops and
    attribute the difference to the detector.
    """

    image: NDArray[np.uint8]  # (page_height, page_width)
    boxes: NDArray[np.float32]  # (n, 4)
    texts: list[str]


def documents(
    n_pages: int,
    *,
    page_height: int = 160,
    page_width: int = 256,
    # Matches :func:`text_lines`' default height, so a crop taken off a page is
    # at the scale H3's recognizer was trained on. Mismatched scales would show
    # up in H5 as recognition error and be attributed to the detector.
    line_height: int = 32,
    max_lines: int = 4,
    fonts: tuple[str, ...] = TRAIN_FONTS,
    noise: float = 0.03,
    rng=None,
) -> list[Document]:
    """Pages with a few text lines at known, non-overlapping positions.

    Lines are placed in random vertical slots and given a random left margin,
    so a detector cannot cheat by predicting fixed positions — which a fixed
    layout would let it do, making H5's detection stage look free.
    """
    generator = as_rng(rng)
    pages: list[Document] = []
    slots = max(page_height // line_height, 1)
    for _ in range(n_pages):
        page = np.full((page_height, page_width), 245, dtype=np.float32)
        count = int(generator.integers(1, min(max_lines, slots) + 1))
        chosen = generator.permutation(slots)[:count]
        boxes: list[list[float]] = []
        texts: list[str] = []
        for slot in sorted(int(value) for value in chosen):
            text = " ".join(str(generator.choice(WORDS)) for _ in range(int(generator.integers(1, 3))))
            strip_width = int(generator.integers(page_width // 3, page_width - 8))
            strip = render_text(
                text,
                font=str(generator.choice(fonts)),
                height=line_height,
                width=strip_width,
                noise=0.0,
                rng=generator,
            )
            # The rendered strip is padded to strip_width; the box should cover
            # the ink, not the padding, or the detector is asked to predict a
            # boundary that nothing in the image marks.
            columns = np.flatnonzero((strip < 200).any(axis=0))
            if len(columns) == 0:
                continue
            ink_width = int(columns[-1]) + 1
            left = int(generator.integers(2, max(page_width - ink_width - 2, 3)))
            top = slot * line_height + int(generator.integers(0, max(line_height - strip.shape[0], 1) + 1))
            top = min(top, page_height - strip.shape[0])
            region = page[top : top + strip.shape[0], left : left + ink_width]
            region[:] = np.minimum(region, strip[:, :ink_width])
            boxes.append([left, top, left + ink_width, top + strip.shape[0]])
            texts.append(text)

        if noise > 0:
            page = page + generator.normal(0, noise * 255, page.shape)
        pages.append(
            Document(
                image=np.clip(page, 0, 255).astype(np.uint8),
                boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
                texts=texts,
            )
        )
    return pages
