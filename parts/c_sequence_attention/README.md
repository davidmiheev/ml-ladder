# C. Sequence models & attention

**Lab:** [Annotated Transformer / nanoGPT](https://nlp.seas.harvard.edu/annotated-transformer/)

| Rung | Goal | Module |
| ---- | ---- | ------ |
| C1 | Char-level RNN/LSTM language model | `c1_char_rnn.py` |
| C2 | Seq2seq with attention on a toy translation task | `c2_seq2seq_attention.py` |
| C3 | Transformer from scratch (attention, positional encoding, causal masking) | `c3_transformer_from_scratch.py` |
| C4 | Fine-tune a small ViT on a custom image-classification task | `c4_vit_finetune.py` |
| C5 | Modern decoder architectures read from source — Qwen3.5, Gemma 4 and DeepSeek-V2 against C3 | `c5_modern_decoders.py` |

Run any rung with `ladder run C1`; `--help` lists its flags. All four need
`uv sync --extra torch`. Text comes from `mlladder.text` (tiny Shakespeare, a
character tokenizer, and the synthetic date/recall generators); images and torch
plumbing come from `mlladder.vision` and `mlladder.torchkit` as in part B.

**The long version is [docs/writeups/c-sequence-attention.md](../../docs/writeups/c-sequence-attention.md)**
— derivations, per-seed numbers, surprises and open threads.

## What the default runs show

Medians over seeds 0–2. C1 and C3 share a corpus, split, eval windows, step
count, batch size and context length, so their numbers are directly comparable.

**C1 — the LSTM's memory is real; at this context it doesn't pay.** Gradient
horizon (the distance at which the gradient is still a millionth of its
final-step value) is **63 steps for the LSTM against 27 for either vanilla RNN**,
and at 63 steps back the LSTM retains about $10^{-2}$ of gradient against the
RNN's $10^{-11}$ — a factor of about $10^{9}$. But on Shakespeare at a 64-character context the
width-matched vanilla RNN *wins* on bits/char (2.263 vs the LSTM's 2.334), because
character prediction here is dominated by short-range structure. The recall probe
isolates the difference: with the label sitting at position 0 and nothing else
carrying signal, the LSTM solves 15-step recall perfectly where the RNN is
already at chance, and both fail at 30 within this training budget.

**C2 — the bottleneck's cost is enormous once the content is long.** Exact-match
accuracy with filler words around the date:

| source length | bottleneck | attention |
| --- | --- | --- |
| 18 chars | 0.997 | 1.000 |
| 101 chars | 0.326 | 1.000 |
| 193 chars | **0.005** | 1.000 |

The fixed-size context vector degrades from perfect to unusable while attention
does not move. Character accuracy degrades far more gently than exact match
(0.999 → 0.899 → 0.710) — the bottleneck model keeps getting *most* characters
right, which is exactly the failure mode that makes a date parser useless.

**C3 — position is most of the signal.** Removing the positional encoding costs
**0.304 bits/char** (2.641 vs 2.349), which is the largest single ablation in the
part: without it the architecture is permutation-equivariant and can only model a
bag of characters. Learned encodings beat sinusoidal by 0.040. The causal mask
leaks exactly **0.0** — changing the last token moves no earlier logit at all.

**C4 — pretraining is the whole story.** Same ViT-Tiny architecture, three ways:
from scratch 0.420, linear probe 0.866, full fine-tune 0.955. The probe trains
**772 parameters** against 5.5M and gets within 9 points of the full fine-tune;
training from scratch on 1,500 images gets barely above two-class chance,
because a transformer has none of a CNN's locality or translation priors.

## The uncomfortable cross-rung result

At matched budgets on this corpus, **recurrence beats the transformer**:

| model | parameters | bits/char |
| --- | --- | --- |
| C1 vanilla RNN (wide) | 343k | **2.263** |
| C3 transformer, learned PE | 1.82M | 2.309 |
| C1 LSTM | 320k | 2.334 |
| C3 transformer at C1's budget | 242k | 2.508 |

500 steps, 64-character context and 250k characters of Shakespeare is a regime
transformers are not built for, and the transformer here gets no warmup (part
B5's lesson about Adam's schedule applies directly). The writeup takes this
apart rather than explaining it away.

## C5 — what changed since C3

Decode state in MiB (fp16), computed from published configs rather than trained:

| decoder | 1k | 64k | 256k |
| --- | --- | --- | --- |
| C3 (this repo, 4 layers MHA) | 3 | 192 | 768 |
| Qwen3-8B (GQA 32:8) | 144 | 9,216 | 36,864 |
| Mistral-7B (sliding 4096) | 128 | **512** | **512** |
| DeepSeek-V2-Lite (MLA r=512) | 30 | 1,944 | 7,776 |
| Qwen3-Next (36 linear + 12 full) | 62 | 1,574 | 6,182 |
| Qwen3.6-27B / Qwen3.8-27B (48 linear + 16 full) | 139 | 4,171 | 16,459 |

**Two generations, one architecture.** Qwen3.6-27B and Qwen3.8-27B differ in
**1 of 61 config keys** — the transformers version that wrote the file — and ship
the same 55.56 GB of weights, so C5 carries them as one entry. Both were run at
3-bit GGUF on a T4, along with DeepSeek-V2-Lite and Gemma-4 at 4-bit NF4; see
[the writeup](../../docs/writeups/c-sequence-attention.md#the-decoders-actually-run)
and `tools/real_models/`.

Each mechanism ablated against its own config without it, at 64k: GQA **4.0×**,
sliding window **16.0×**, MLA **10.7×**, delta rule **3.9×**.

Qwen3-8B's cache at 64k tokens is 9.2 GiB — larger than the model's own weights
in fp16. The delta-rule state is constant in context (38 MiB before reading
anything) and pays for itself from 537 tokens onward, which is C1's bargain with
a bigger budget. The accounting is validated against DeepSeek's published
15.7B/2.4B figures to 0.6%/3.2%.
