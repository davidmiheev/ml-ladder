"""Write part H's test pages out as PNGs for a real OCR model to read.

The rng stream matches ``h5_document_pipeline``: the train pages are drawn first
so page k here is page k of H5's test split, and the ground truth is the ground
truth H5's two-stage pipeline was scored against.
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
from PIL import Image

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
from mlladder import synth

GEOMETRY = {"page_height": 160, "page_width": 256, "line_height": 32}
TRAIN_PAGES = 2500


def main(out_dir: str = "vlm_pages", count: int | str = 24, seed: int | str = 0) -> int:
    count, seed = int(count), int(seed)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    synth.documents(TRAIN_PAGES, fonts=synth.TRAIN_FONTS, rng=rng, **GEOMETRY)
    test = synth.documents(500, fonts=synth.TRAIN_FONTS, rng=rng, **GEOMETRY)

    records = []
    for index, page in enumerate(test[:count]):
        image = Image.fromarray(page.image)
        image.save(out / f"page{index:03d}_native.png")
        # Upscaling adds no information, so the ground truth is unchanged; it
        # only removes the excuse that the page is smaller than the model's
        # patch grid.
        image.resize((page.image.shape[1] * 4, page.image.shape[0] * 4), Image.LANCZOS).save(
            out / f"page{index:03d}_4x.png"
        )
        records.append({"page": index, "lines": list(page.texts), "boxes": [list(map(float, b)) for b in page.boxes]})
    (out / "truth.json").write_text(json.dumps(records, indent=1))
    print(f"wrote {len(records)} pages and {sum(len(r['lines']) for r in records)} lines to {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:]))
