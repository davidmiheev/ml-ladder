# Fixes needed to build and run the candle fork

`ignite-perf-fixes.patch` applies to `davidmiheev/candle` at `ignite-perf`
(9aa02f1). It is three independent changes, all found by trying to run the
models part C and part H price:

1. **`candle-kernels/src/compatibility.cuh`** — the `__hmax_nan`/`__hmin_nan`
   shim is guarded on `__CUDA_ARCH__ < 800`, but CUDA 12.0+ defines both itself
   for arch >= 530. On CUDA 12.8 `nvcc` fails on `src/affine.cu` with
   "function has already been defined", so the CUDA build does not complete at
   all. Guarded on `CUDART_VERSION < 12000` as well.

2. **`candle-transformers/src/models/gemma4/text.rs`** — `TextModel` loads its
   decoder from `model.*`. A multimodal checkpoint (gemma-4-E2B-it) keeps the
   identical decoder at `model.language_model.*` beside its vision and audio
   towers, so text-only loading fails at `model.embed_tokens.weight`.
   `new_with_quant_at` takes the decoder's own prefix and `new_nested` supplies
   it; `new_with_quant` is unchanged in behaviour.

3. **`candle-examples/examples/gemma4/main.rs`** — picks the nested constructor
   when the config has a `text_config`, and adds `--dtype` because the example
   hardcodes bf16 on GPU and bf16 kernels need sm_80. On a T4 the old default
   fails with `DriverError(CUDA_ERROR_NOT_FOUND, "named symbol not found")`.

**These are not sufficient to make Gemma-4 work.** With all three applied,
gemma-4-E2B-it loads and generates at 25.7 tok/s on a T4 and its f16 output is
degenerate. Transformers at f16 on the same card with the same prompt answers
correctly, so it is not precision — see
[`../../../docs/writeups/data/c-real-models.json`](../../../docs/writeups/data/c-real-models.json)
for the control.

**Not fixed here:** `qwen3_5`, `gemma4` and `unlimited_ocr` call
`candle_nn::fused::static_decode` (which is `#[cfg(feature = "cuda")]`) from
call sites that are not gated, so a build without `--features cuda` fails with
51 errors. Gating the call sites the same way would fix it; that is a larger
change than belongs in a patch found by an outsider running the code.
