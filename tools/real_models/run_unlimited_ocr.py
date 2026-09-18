"""Read the exact H5 test pages with baidu/Unlimited-OCR.

H5 measured a two-stage detect-then-recognise pipeline on these pages and found
99.98% of its error in the interface between the stages; H6 priced deleting the
interface at 904M parameters. This runs a model that deleted it, on the same
pages, against the same ground truth, so the comparison is a measurement rather
than an argument.
"""

from __future__ import annotations

import contextlib
import inspect
import io
import json
import os
import pathlib
import re
import sys
import time

# The checkpoint ships vendored DeepSeek-V2 code written against transformers
# ~4.4x. Two things moved since, and both are shimmed rather than worked around
# by pinning, because the other models in this sweep need a recent transformers.
import transformers.utils.import_utils as _iu

if not hasattr(_iu, "is_torch_fx_available"):
    _iu.is_torch_fx_available = lambda: False  # we never trace with fx here
import transformers.utils as _u

if not hasattr(_u, "is_torch_fx_available"):
    _u.is_torch_fx_available = _iu.is_torch_fx_available

import torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from parts.h_computer_vision.h3_crnn_ctc import character_error_rate, word_error_rate

MODEL = sys.argv[1] if len(sys.argv) > 1 else "baidu/Unlimited-OCR"
SCALE = sys.argv[2] if len(sys.argv) > 2 else "4x"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 24

pages_dir = pathlib.Path(os.environ.get("MLLADDER_VLM_PAGES", "vlm_pages"))
truth = json.loads((pages_dir / "truth.json").read_text())[:LIMIT]

config = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
# Newer transformers stopped handing every config a default pad_token_id; the
# vendored code still reads one off both the outer and the language config.
for holder in (config, getattr(config, "language_config", None)):
    if holder is None:
        continue
    if isinstance(holder, dict):
        holder.setdefault("pad_token_id", None)
    elif not hasattr(holder, "pad_token_id"):
        holder.pad_token_id = None

print(f"loading {MODEL}", flush=True)
started = time.perf_counter()
model = AutoModel.from_pretrained(MODEL, config=config, trust_remote_code=True, torch_dtype=torch.bfloat16)
model = model.eval().cuda()
load_seconds = time.perf_counter() - started
parameters = sum(p.numel() for p in model.parameters())
print(f"loaded in {load_seconds:.1f}s, {parameters / 1e9:.3f}B parameters", flush=True)

tokenizer = AutoTokenizer.from_pretrained(MODEL, trust_remote_code=True)
entrypoints = [name for name in ("infer", "chat", "generate") if hasattr(model, name)]
print("entrypoints:", entrypoints, flush=True)
for name in entrypoints:
    with contextlib.suppress(TypeError, ValueError):
        print(f"  {name}{inspect.signature(getattr(model, name))}", flush=True)

PROMPT = "<image>\nFree OCR. Transcribe every line of text, one line per line."
records = []
for record in truth:
    path = pages_dir / f"page{record['page']:03d}_{SCALE}.png"
    started = time.perf_counter()
    # ``infer`` streams its transcript to stdout and returns None, so the
    # transcript has to be captured rather than assigned. Reading the return
    # value scored the model at CER 3.85 on a first pass -- entirely my
    # measurement, not the model, which had transcribed every page correctly.
    buffer = io.StringIO()
    with torch.no_grad(), contextlib.redirect_stdout(buffer):
        returned = model.infer(
            tokenizer,
            prompt=PROMPT,
            image_file=str(path),
            output_path="/content/ocr_out",
            base_size=1024,
            image_size=640,
            crop_mode=True,
            save_results=False,
            test_compress=False,
        )
    elapsed = time.perf_counter() - started
    text = returned if isinstance(returned, str) else buffer.getvalue()
    # The model emits grounding as <|det|>label [x1, y1, x2, y2]<|/det|> before
    # each line's text; the transcript is what is left once those are removed.
    text = re.sub(r"<\|det\|>.*?<\|/det\|>", "", text, flags=re.S)
    text = text.replace("<|ref|>", "").replace("<|/ref|>", "")
    text = "\n".join(
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith(("Free OCR", "Do not attempt", "<image>"))
    )
    records.append({"page": record["page"], "truth": record["lines"], "prediction": text, "seconds": elapsed})
    print(f"[page {record['page']:>3}] {elapsed:5.1f}s  truth={record['lines']}  pred={text[:160]!r}", flush=True)

# Scored the way H5 scores its pipeline: one transcript per page, lines joined
# in reading order, so the comparison is like for like.
references = ["\n".join(r["truth"]) for r in records]
hypotheses = []
for r in records:
    lines = [line.strip() for line in r["prediction"].splitlines() if line.strip()]
    hypotheses.append("\n".join(lines))
cer = character_error_rate(references, hypotheses)
wer = word_error_rate(references, hypotheses)
summary = {
    "model": MODEL,
    "scale": SCALE,
    "pages": len(records),
    "parameters": parameters,
    "load_seconds": load_seconds,
    "seconds_per_page": sum(r["seconds"] for r in records) / len(records),
    "cer": cer,
    "wer": wer,
}
print("SUMMARY " + json.dumps(summary), flush=True)
out = pathlib.Path(f"/content/ocr_{MODEL.replace('/', '_')}_{SCALE}.json")
out.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
print("wrote", out, flush=True)
