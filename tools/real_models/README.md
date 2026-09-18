# Running the real models

H6 prices PaddleOCR-VL and Unlimited-OCR from their published configs; C5 does
the same for four decoders. Arithmetic over a config can be wrong in ways no
amount of internal consistency reveals, so these scripts load the actual weights
and run them on part H's own test pages.

They live here rather than in `parts/` because they are not rungs: they need
network access, several gigabytes of weights and, for anything past a billion
parameters, a GPU — none of which the ladder's tests may assume. What they
produce is recorded in [`../../docs/writeups/data/h-real-models.json`](../../docs/writeups/data/h-real-models.json)
with its provenance, and what it corrected is in H6 (`page_tokens`) and in the
part H writeup.

```bash
# the pages, written out of the same generator H5 uses
uv run python -c "..."   # see the writeup section "The models, actually run"

# OCR, scored with H3's CER/WER against H5's ground truth
python tools/real_models/run_paddleocr_vl.py PaddlePaddle/PaddleOCR-VL 4x 10
python tools/real_models/run_unlimited_ocr.py baidu/Unlimited-OCR 4x 10

# a decoder, quantised to fit
python tools/real_models/run_llm.py deepseek-ai/DeepSeek-V2-Lite 4bit 48
```

## The version pins, and why each one exists

Published checkpoints carry vendored modelling code written against the
transformers of their release, and transformers moves. Every one of these
scripts hit that:

- **PaddleOCR-VL** calls `create_causal_mask(inputs_embeds=…)`; the parameter is
  `input_embeds` in the transformers it runs against. Shimmed in the script, so
  the cached module is never edited.
- **Unlimited-OCR** ships DeepSeek-V2 code needing `is_torch_fx_available` and a
  default `pad_token_id`, both gone from current transformers. Its config names
  4.46.3; installing that into a directory and putting it first on `PYTHONPATH`
  is cheaper and more honest than shimming each API that moved.
- **DeepSeek-V2-Lite** reads `DynamicCache.seen_tokens`, removed after 4.4x. It
  loads fine on a recent transformers and fails at the first generated token —
  so it needs the pin too, and loading is not evidence that generating works.
