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

## Through the candle fork

The accompanying candle fork implements PaddleOCR-VL, Unlimited-OCR, Gemma-4 and
DeepSeek-V2 directly. Running the same weights through a second implementation
is the strongest check available on any of these numbers: PaddleOCR-VL's CER
comes out identical to transformers' to sixteen digits, which makes it a
property of the model rather than of a runtime.

```bash
# One architecture's kernels, not every architecture candle knows about.
# T4 is sm_75; use your own card's capability.
CUDA_COMPUTE_CAP=75 cargo build --release --features cuda --example paddleocr-vl

./target/release/examples/paddleocr-vl --batch vlm_pages/page00*_4x.png \
    --task ocr --max-length 256
```

Measured on a Colab T4: **18m 44s** for that first build, then **37s** to add
the `gemma4` and `deepseekv2` examples. The cost is the first build, not the
model.

Three things to know before trying it:

- **CUDA 12.8 breaks the kernel build.** `candle-kernels/src/compatibility.cuh`
  defines `__hmax_nan`/`__hmin_nan` under `#if __CUDA_ARCH__ < 800`, but 12.8
  defines them itself for arch >= 530, so `nvcc` fails on `src/affine.cu`.
  Adding `&& CUDART_VERSION < 12000` to that guard fixes it.
- **`--bf16` needs sm_80.** On an older card it fails with
  `DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found")`, which names a
  missing symbol rather than the missing capability. Omit the flag.
- **A CPU-only build does not work on `ignite-perf`.** `qwen3_5`, `gemma4` and
  `unlimited_ocr` call `candle_nn::fused::static_decode`, which is behind
  `#[cfg(feature = "cuda")]`, from call sites that are not: 51 errors in
  `candle-transformers` without `--features cuda`.

What each model needed is recorded in
[`../../docs/writeups/data/h-real-models.json`](../../docs/writeups/data/h-real-models.json)
(OCR) and [`../../docs/writeups/data/c-real-models.json`](../../docs/writeups/data/c-real-models.json)
(decoders), including the two patches Gemma-4 needed and why its output is still
wrong on an sm_75 card.

## Unlimited-OCR

`candle-patches/unlimited-ocr-example.rs` goes at
`candle-examples/examples/unlimited-ocr/main.rs` in the fork. It is the
`ignite-catalog` executor's Unlimited-OCR path with the ignite SDK removed, so
it runs from the hub with no platform dependencies:

```bash
CUDA_COMPUTE_CAP=75 cargo build --release --features cuda --example unlimited-ocr
./target/release/examples/unlimited-ocr --dtype f16 --no-crops \
    --image vlm_pages/page000_4x.png
```

`--no-crops` is the global-1024-view path, which is the reference default under
640px and the only path the catalog executor ships. The tiled path needs the two
batch fixes in `ignite-perf-fixes.patch` even to run, and then reads worse than
the global view — the grid search is not reference-equivalent yet.
