# Fixes and additions for the candle fork

`ignite-perf-fixes.patch` applies to `davidmiheev/candle` at `ignite-perf`
(9aa02f1); `unlimited-ocr-example.rs` is a new example that belongs at
`candle-examples/examples/unlimited-ocr/main.rs`. All of it was found by
trying to run the models part C and part H price.

## Build fixes

1. **`candle-kernels/src/compatibility.cuh`** — the `__hmax_nan`/`__hmin_nan`
   shim is guarded on `__CUDA_ARCH__ < 800`, but CUDA 12.0+ defines both itself
   for arch >= 530. On CUDA 12.8 `nvcc` fails on `src/affine.cu` with "function
   has already been defined", so **the CUDA build does not complete at all**.
   Guarded on `CUDART_VERSION < 12000` as well.

Measured build times on a Colab T4 with `CUDA_COMPUTE_CAP=75` (one
architecture's PTX instead of all of them):

| what | time |
| --- | --- |
| first build, three examples | 22m 42s |
| adding two more examples, warm | 37s |
| after a `candle-transformers` change | 3m 03s |

## Batch bugs in the vision path

Both only bite when the encoder gets more than one crop, which is why the
catalog executor's global-only V1 scope never hit them.

2. **`segment_anything/image_encoder.rs`** — `(xs + pos_embed)` where `xs` is
   `[n, h, w, c]` and the learned grid is `[1, h, w, c]`. candle's `+` needs
   equal shapes, so Unlimited-OCR's tiled path dies with
   `shape mismatch in add, lhs: [8, 40, 40, 768], rhs: [1, 40, 40, 768]`.
   `broadcast_add` is identical when `n == 1`.

3. **`unlimited_ocr/vision.rs`** — in `forward_crops`, `sam_seq` is a transpose
   and the `cat` of it keeps non-standard strides, which the batched matmul in
   the projector rejects: `matmul is only supported for contiguous tensors`.
   One `.contiguous()`.

With both applied the tiled path runs. It is **not** yet reference-equivalent:
for a 1024x640 page `pick_crop_ratio` chooses a 5x3 grid (1803 image tokens
against the global view's 273) and reads *worse* than the global view — see the
numbers in [`../../../docs/writeups/data/h-real-models.json`](../../../docs/writeups/data/h-real-models.json).
Whatever bounds the reference puts on the grid, this port does not have them.

## DeepSeek-V2 quantization

4. **`deepseek2.rs` + `examples/deepseekv2/main.rs`** — DeepSeek-V2-Lite is
   31.4 GB in f16 against a 15.4 GB card, and the example had no way to ask for
   less. Added the same load-time quantization `gemma4` and `qwen3_5` already
   use: a `Proj` that quantizes on the CPU (so a full f32 copy never reaches the
   GPU) and keeps a `QMatMul`, threaded through attention, the MLP, the MoE
   experts and `lm_head`, plus `DeepSeekV2::new_with_quant` and a `--quant`
   flag taking q8_0/q6k/q5k/q4k/q4_0/q3k/q2k. `--dtype` is there too, because
   the example hardcoded bf16 and bf16 needs sm_80.

   `quant_dtype_for` handles the case k-quants cannot: the dense MLP's
   `down_proj` reduces over 10944 and each expert's over 1408, neither a
   multiple of 256. Both are multiples of 32, so those fall back to Q8_0.

   **Status: compiles, and the flags are live on the binary — but not yet run
   end to end.** The GPU session was reclaimed during the 31.4 GB download,
   twice. The wiring is unverified against generated tokens.

5. **`examples/gemma4/main.rs`** — a nested-prefix constructor
   (`TextModel::new_nested`), because a multimodal checkpoint keeps its decoder
   at `model.language_model.*` and text-only loading died at
   `model.embed_tokens.weight`; a `--dtype` flag; and the BOS prepend your
   `candle-llm-safetensors` executor documents. See the note below — the BOS
   change did not visibly fix the output in the one configuration I tried.

## The new example

`unlimited-ocr-example.rs` is the catalog executor's Unlimited-OCR path with
the ignite SDK taken out: hub loading instead of `AssetLoader`, the same
1024-square mean-gray pad, the same InternVL-style grid search, the same
sliding-window n-gram suppression, the ordinary KV-cache decode rather than
static R-SWA, and `--no-crops` for the global-only path. It reads part H's test
pages at **CER 0.229** (8 of 10 exactly) through the global view.

## The missing qwen3_5 binary

`qwen3_5-example.rs` goes at `candle-examples/examples/qwen3_5/main.rs`.
`candle-transformers` has carried `qwen3_5.rs` — the Gated-DeltaNet hybrid stack
behind Qwen3.6 and Qwen3.8 — with no example to invoke it. This is that example:
`--quant q8_0..q2k` (a 27B is 55.6 GB in bf16, so it is not optional on anything
smaller than an A100), `--dtype` for pre-sm_80 cards, and Qwen's chat template.

Worth noting while patching `gemma4`: `qwen3_5.rs` **already** detects the
`model.language_model` prefix itself and **already** has load-time quantization
with a GGUF disk cache. Both are things `gemma4` had to be patched for. The two
siblings have drifted, and `gemma4` is the one behind.

## Gemma-4: the BOS fix, verified

The third `gemma4` patch is the one that mattered. `gemma-4-E2B-it` ships a
`tokenizer.json` with no BOS post-processor, and gemma-family models fall into
token copy-loops without one — which your `candle-llm-safetensors` executor
already documents. Same binary, same T4, same f16, same chat template:

- **without** the BOS prepend: the word "thought" repeated to the token limit
- **with** it: a correct two-sentence answer, 5.89 tok/s

A one-token difference, which is exactly what it takes to distinguish a missing
BOS from a broken kernel — and I had guessed the kernel.

## FLUX: two blockers before it can run at all

6. **`examples/flux/main.rs`** — the repo is hardcoded to
   `black-forest-labs/FLUX.1-schnell`, which is **gated**: without an accepted
   licence the example dies with a 401 before loading anything. `--bf-repo`
   points the transformer and the autoencoder at an ungated mirror;
   `unsloth/FLUX.1-schnell` carries the identical layout. This is needed even on
   the `--quantized` path, which takes its transformer from `lmz/candle-flux`
   but still fetches `ae.safetensors` from the gated repo.

   Also `--dtype`, because `device.bf16_default_to_f32()` picks bf16 on any CUDA
   device and bf16 kernels need sm_80. That is now the **third** example in this
   fork with the same sm_75 trap (`gemma4`, `deepseekv2`, `flux`) — it may be
   worth a shared helper that consults the compute capability rather than just
   `is_cuda()`.

## Verified runs

| model | how | result |
| --- | --- | --- |
| PaddleOCR-VL | candle, f16 | CER 0.000, identical to transformers to 16 digits |
| Unlimited-OCR | candle, f16, new example | CER 0.229, 8/10 pages exact |
| gemma-4-E2B-it | candle, f16, BOS fix | correct answer, 5.89 tok/s |
| Qwen3.5-4B | candle, q4k, new example | correct answer, 26.79 tok/s |
| DeepSeek-V2-Lite | transformers, bnb NF4 | coherent, 4.49 tok/s, 9.62 GB |

`Qwen3.6-8B-A1.6B` (a community REAP-pruned derivative) produces token soup
through the same binary and flags that Qwen3.5-4B answers correctly with — the
derivative, not the example.

### FLUX still has not produced an image

Both blockers above are fixed and the example builds, but the run has not
completed in two attempts: a from-scratch build (35m50s when it shares I/O with
the download) plus ~56 GB of weights does not fit a ~50-minute session.

Most of that 56 GB is one file. `google/t5-v1_1-xxl` at `refs/pr/2` ships
`model.safetensors` in **float32** — an 11B encoder at four bytes a parameter is
~44 GB — while the quantized FLUX transformer is 12.64 GB and the autoencoder
0.34 GB. The example converts to the run dtype at load, so those f32 bytes are
downloaded only to be halved in memory. A `--t5-repo` flag pointing at an f16
mirror would roughly halve the download and probably bring the run inside one
session. Not implemented.
