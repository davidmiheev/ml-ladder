"""Read the exact H5 test pages with PaddlePaddle/PaddleOCR-VL.

H5 measured a two-stage detect-then-recognise pipeline on these pages and found
99.98% of its error in the interface between the stages; H6 priced deleting that
interface at 904M parameters and 182 tokens per 384px page. This runs the model
H6 costed, on the same pages, against the same ground truth -- so the claim that
deleting the interface is worth it stops being an argument and becomes a number.

CPU only, which is why the page count is small: the point is the error rate, and
seconds-per-page is recorded as a fact about this machine, not about the model.
"""

from __future__ import annotations

import json
import os
import pathlib
import sys
import time

import torch

# The checkpoint's remote modeling code calls transformers' create_causal_mask
# with ``inputs_embeds=``; the parameter is spelled ``input_embeds`` in the
# transformers this runs against. Accepting both keeps the published weights
# usable without editing the cached module or hunting for the one release where
# the spelling matched.
import transformers.masking_utils as _mu
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor

_create_causal_mask = _mu.create_causal_mask


def _create_causal_mask_compat(*args, **kwargs):
    if "inputs_embeds" in kwargs:
        kwargs["input_embeds"] = kwargs.pop("inputs_embeds")
    return _create_causal_mask(*args, **kwargs)


_mu.create_causal_mask = _create_causal_mask_compat

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
from parts.h_computer_vision.h3_crnn_ctc import character_error_rate, word_error_rate  # noqa: E402

MODEL = sys.argv[1] if len(sys.argv) > 1 else "PaddlePaddle/PaddleOCR-VL"
SCALE = sys.argv[2] if len(sys.argv) > 2 else "4x"
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 8

pages_dir = pathlib.Path(os.environ.get("MLLADDER_VLM_PAGES", "vlm_pages"))
truth = json.loads((pages_dir / "truth.json").read_text())[:LIMIT]

print(f"loading {MODEL} on cpu", flush=True)
started = time.perf_counter()
processor = AutoProcessor.from_pretrained(MODEL, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(MODEL, trust_remote_code=True, torch_dtype=torch.float32)
model = model.eval()
load_seconds = time.perf_counter() - started
parameters = sum(p.numel() for p in model.parameters())
print(f"loaded in {load_seconds:.1f}s, {parameters / 1e9:.3f}B parameters", flush=True)

PROMPT = "OCR:"
records = []
for record in truth:
    path = pages_dir / f"page{record['page']:03d}_{SCALE}.png"
    image = Image.open(path).convert("RGB")
    messages = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": PROMPT}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image], return_tensors="pt").to(model.device)
    # H6 priced this model at 182 tokens for a 384px page from its patch size
    # and compression ratio. Counting the image tokens the processor actually
    # emits is the check on that arithmetic.
    image_token_id = getattr(model.config, "image_token_id", None)
    image_tokens = int((inputs["input_ids"] == image_token_id).sum()) if image_token_id is not None else -1
    prompt_tokens = int(inputs["input_ids"].shape[1])
    started = time.perf_counter()
    with torch.no_grad():
        generated = model.generate(**inputs, max_new_tokens=128, do_sample=False)
    elapsed = time.perf_counter() - started
    trimmed = generated[:, inputs["input_ids"].shape[1] :]
    prediction = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
    records.append(
        {
            "page": record["page"],
            "truth": record["lines"],
            "prediction": prediction,
            "seconds": elapsed,
            "image_tokens": image_tokens,
            "prompt_tokens": prompt_tokens,
            "image_size": list(image.size),
        }
    )
    print(
        f"[page {record['page']:>3}] {elapsed:6.1f}s {image_tokens:>5} image tokens "
        f"({prompt_tokens} prompt) truth={record['lines']} pred={prediction[:120]!r}",
        flush=True,
    )

references = ["\n".join(r["truth"]) for r in records]
hypotheses = ["\n".join(line.strip() for line in r["prediction"].splitlines() if line.strip()) for r in records]
summary = {
    "model": MODEL,
    "scale": SCALE,
    "device": "cpu",
    "pages": len(records),
    "parameters": parameters,
    "load_seconds": load_seconds,
    "seconds_per_page": sum(r["seconds"] for r in records) / len(records),
    "image_tokens_median": sorted(r["image_tokens"] for r in records)[len(records) // 2],
    "cer": character_error_rate(references, hypotheses),
    "wer": word_error_rate(references, hypotheses),
}
print("SUMMARY " + json.dumps(summary), flush=True)
out = pathlib.Path(f"/tmp/claude-0/vlm/ocr_{MODEL.replace('/', '_')}_{SCALE}.json")
out.write_text(json.dumps({"summary": summary, "records": records}, indent=1))
print("wrote", out, flush=True)
