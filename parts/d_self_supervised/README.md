# D. Self-supervised & representation learning

**Lab:** [lightly-ai — contrastive-learning reference implementations](https://github.com/lightly-ai/lightly)

| Rung | Goal | Module |
| ---- | ---- | ------ |
| D1 | SimCLR-style contrastive pretraining on CIFAR-10 | `d1_simclr_pretraining.py` |
| D2 | Masked autoencoder (MAE) on a small image dataset | `d2_masked_autoencoder.py` |
| D3 | Linear probing of the learned representations against a supervised baseline | `d3_linear_probing.py` |

Run any rung with `ladder run D1`; `--help` lists its flags. All three need
`uv sync --extra torch`, and the sweep behind the numbers below ran on a Colab
T4 — see the writeup for the CPU-sized flags.

**The long version is [docs/writeups/d-self-supervised.md](../../docs/writeups/d-self-supervised.md)**
— derivations, per-seed numbers, surprises and open threads.

## What this part reuses

Nothing here is a fresh stack. D1's encoder is **B4's residual network** with
its classifier removed; D2's transformer block is **C3's**, applied to image
patches instead of characters; D1's feature-space figure is **A3's PCA**; and
D3's probe is the one D1 introduced. The ladder's parts are supposed to compose,
and this is where that starts paying.

## The shape of the argument

Self-supervised learning is not sold on beating supervised training when labels
are plentiful — it is sold on what happens when they are not. So each rung ends
in the same measurement, a linear probe on frozen features, and D3 sweeps the
one variable that decides whether any of it was worth doing: **how many labels
you have**.

Two readings of every representation, because they answer different questions:

- **linear probe** — what a trained classifier can extract from the features;
- **k-NN** — what the feature *geometry* already encodes, with nothing trained.

A representation that probes well but k-NNs badly is linearly separable without
being well organized, which is worth knowing before building anything on it.

## What the sweep found

Medians over seeds 0–2 on a Colab T4. The full argument, with derivations, is in
the writeup.

**D1 — SimCLR.** Probe accuracy against an untrained encoder's 0.256:

| arm | probe accuracy |
| --- | --- |
| $\tau$ = 0.1 / 0.5 / 1.0 | 0.359 / 0.339 / 0.318 |
| strength 0.0 / 0.5 / 1.0 | 0.316 / **0.430** / 0.339 |

Temperature is monotone; **augmentation strength is not** — the middle arm wins
by nine points over full strength on every seed, because the positive pair
defines the invariance being learned and full-strength crops often share no
object at all. The no-augmentation arm still reaches 0.316, which is 72% of what
the full-strength arm gains over an untrained encoder — most of the apparent
"contrastive learning" at strength 1.0 is not contrastive learning.

**D2 — MAE.** Masked MSE rises by 2.5× across the mask sweep while probe
accuracy moves by 0.7 points (0.349 → 0.351 → 0.356 → 0.345 for 0.25 → 0.9). The
median winner is the published 75%, but `best/mask_ratio` is 0.75, 0.75, 0.5
across seeds — not stable, and a single-seed run would have "confirmed" the
paper. **The pretext loss is not the metric you care about.**

**D3 — probing.** Linear probe / k-NN at 400 labels per class:

| encoder | probe | k-NN |
| --- | --- | --- |
| untrained | 0.271 | 0.305 |
| SimCLR | 0.350 | **0.340** |
| MAE | **0.368** | 0.293 |

MAE wins the probe and loses to an *untrained* network on k-NN — linearly
separable without being metrically organized, exactly as the objectives predict
(NT-Xent optimizes cosine similarity; MAE's pixel MSE never mentions distances
between images). At 5 labels per class both pretrained encoders beat the
untrained one by 0.07–0.09.

The supervised arm is **not** a fair baseline: it trains on the full labelled
pool and is then probed with $n$ labels, so its column is flat by construction.
The writeup says what it would take to fix.
