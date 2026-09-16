# Part C — Sequence models & attention

> **Status:** complete (C1–C4) · **Constraint:** subsets and short budgets; C1/C3 share every setting so they can be compared
> **Numbers:** seeds 0–2, reference seed 0 · raw values in [`data/c-metrics.json`](data/c-metrics.json)
> **Reproduce:** `ladder run C1` … `C4` (needs `uv sync --extra torch`), or `uv run python tools/writeup_figures.py C`
> **Hardware:** this part's sweep ran on a Colab T4 — C4 is ~25 min/seed on four CPU cores against ~35 s on the GPU

Part B's models all see their whole input at once. Part C's have to decide what
to *carry*: a recurrent cell squeezes the past into a fixed state, attention
keeps everything and learns where to look, and a ViT applies the second answer
to images. Four rungs, one question —

**where does the information have to travel, and what does the architecture
charge for the trip?** A vanilla RNN charges a multiplicative factor per step,
which is why its gradient dies. An LSTM buys an additive path, the same trick as
B4's residual connection, one year earlier and in time rather than depth.
Attention refuses to compress at all and pays quadratic cost instead. Every
result below is a price for distance.

Notation: $T$ is sequence length, $h_t$ the hidden state at step $t$,
$Q, K, V \in \mathbb{R}^{T\times d}$ the attention projections, and $d_k$ the
per-head width.

---

## C1 — Char-level RNN and LSTM

**Question.** The LSTM is supposed to remember further than a vanilla RNN. How
much further, and does the extra memory show up as a better language model?

### The math

**Why the vanilla gradient dies.** With $h_t = \tanh(W_x x_t + W_h h_{t-1} + b)$,
backpropagation through time multiplies one Jacobian per step:

$$\frac{\partial h_T}{\partial h_t} = \prod_{i=t+1}^{T} \frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=t+1}^{T} \operatorname{diag}\!\left(1 - h_i^2\right) W_h^{\top}.$$

Taking norms, $\lVert\partial h_T/\partial h_t\rVert \le (\gamma\lVert W_h\rVert)^{T-t}$
where $\gamma = \max_i \lVert\operatorname{diag}(1-h_i^2)\rVert \le 1$ for $\tanh$.
The bound is **geometric in the distance**: if $\gamma\lVert W_h\rVert < 1$ the
gradient vanishes exponentially, and if it exceeds 1 the gradient explodes
instead. There is no setting of $W_h$ that keeps a long-range signal alive
without risking the other failure — which is the whole problem, and why the
rungs clip gradients.

**What the LSTM changes.** The cell state's recurrence is additive:

$$c_t = f_t \odot c_{t-1} + i_t \odot g_t \quad\Longrightarrow\quad \frac{\partial c_t}{\partial c_{t-1}} = \operatorname{diag}(f_t).$$

The product over steps becomes $\prod_i \operatorname{diag}(f_i)$ — no weight
matrix, no saturating derivative, just the forget gates. A gate near 1 passes
gradient through essentially unchanged, and the network *learns* which of them to
open. That is why the forget-gate bias is initialized to 1: start by remembering.

**Measuring it.** The rung differentiates a scalar that depends only on the final
state and reports $\lVert\partial L_T/\partial x_t\rVert$ against distance, then
defines the **effective horizon** as the largest distance at which that norm is
still above $10^{-6}$ of its value at the final position.

**Results** (tiny Shakespeare, 250k characters, 64-character context, 500 steps;
medians over seeds 0–2):

| model | parameters | bits/char | horizon | gradient at 63 steps back |
| --- | --- | --- | --- | --- |
| vanilla RNN | 98k | 2.366 | 27 | 1.4e-11 |
| vanilla RNN (width-matched) | 343k | **2.263** | 27 | 6.1e-11 |
| LSTM | 320k | 2.334 | **63** | 9.4e-03 |

The horizon gap is the cleanest measurement in the part: 63 against 27 on every
seed, and about **$10^{9}$ times more gradient** at 63 steps back.

**What surprised me.** *The LSTM loses the language-modelling comparison.* At a
matched parameter budget the width-matched vanilla RNN is 0.07 bits/char better,
consistently across seeds (2.261–2.274 against 2.319–2.336, non-overlapping). The
memory is real and does not pay, because next-character prediction in Shakespeare
at a 64-character context is dominated by short-range structure — spelling,
morphology, the line break after a speaker's name. Nothing in that needs to
reach 63 steps back, and the LSTM spends four gates' worth of parameters per unit
of state to buy reach the task does not use.

So the rung grew a second experiment: a **recall probe** where reach is the only
thing that matters. The label is the sequence's first symbol, every later
position is filler, and chance is 1/8:

| sequence length | vanilla RNN | LSTM |
| --- | --- | --- |
| 5 | 1.000 | 1.000 |
| 15 | 0.127 (chance) | **1.000** |
| 30 | 0.127 | 0.115 (chance) |
| 60 | 0.127 | 0.115 |

There it is, and also its limit: the LSTM solves 15-step recall where the RNN has
already collapsed to chance — about **3× the reach** — but at 30 steps it fails
too, at this budget (120 steps, hidden 64). The horizon measurement says gradient
*arrives* at 63 steps; the recall probe says arriving is not the same as learning
to use it. I would rather report that than quietly pick the length where the
LSTM still wins.

![Validation loss](figures/c/c1_validation_loss.png)
![Gradient reach](figures/c/c1_gradient_reach.png)
![Recall accuracy](figures/c/c1_recall_accuracy.png)
![Bits per character](figures/c/c1_bits_per_char.png)

**Open threads.** GRUs, to see how much of the LSTM's benefit survives with one
gate fewer. Truncated BPTT over a longer context, which is how these models were
actually trained. And the recall probe with a proper budget sweep, to find where
the LSTM's 63-step gradient becomes 63-step learning.

**Reading:** [Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf)
(Hochreiter & Schmidhuber); Pascanu et al., *On the difficulty of training
recurrent neural networks*; Karpathy, *The Unreasonable Effectiveness of RNNs*.

---

## C2 — Seq2seq with and without attention

**Question.** How much does forcing a whole source sentence through one
fixed-size vector actually cost?

### The math

**The bottleneck.** A classic encoder-decoder compresses the source to a single
vector $c = h_{T_s}$ and generates from it:

$$p(y_1,\dots,y_{T_t} \mid x) = \prod_{j} p\!\left(y_j \mid y_{<j},\, c\right).$$

Everything the decoder can ever know about the source must fit in $c$, whose size
is fixed while $T_s$ grows. With $d$-dimensional states the channel carries
$O(d)$ numbers regardless of input length — so the cost of the bottleneck should
grow with the amount of *information* in the source, not its padded length. (The
first version of this rung missed exactly that, swept padding, and found
nothing.)

**Attention removes the constraint.** Keep every encoder state and build a fresh
context per output step:

$$e_{ji} = \operatorname{score}(s_j, h_i), \qquad \alpha_{ji} = \frac{\exp e_{ji}}{\sum_{i'} \exp e_{ji'}}, \qquad c_j = \sum_i \alpha_{ji} h_i .$$

This rung uses Luong's dot product, $\operatorname{score}(s, h) = s^\top h$
(Bahdanau's additive $v^\top\tanh(W[s;h])$ is the other standard choice). The
context is now a *convex combination of all source states*, so no fixed budget
limits what reaches the decoder; the price is $O(T_s T_t)$ scores instead of
$O(T_t)$, and it is the same price the transformer pays in C3.

Padding must be masked **before** the softmax, by setting those logits to
$-\infty$: masking afterwards would leave the remaining weights un-normalized.

**Why this task.** Every output character of an ISO date is a copy of some input
span, so the correct alignment is known in advance and the attention map can be
*scored* — the fraction of output characters whose attention peak sits on a
matching input character — rather than admired.

**Results** (date normalization, filler words around the date, medians over
seeds 0–2):

| filler | source length | bottleneck exact | attention exact | bottleneck char-level |
| --- | --- | --- | --- | --- |
| none | 18 | 0.997 | 1.000 | 0.999 |
| 10 words | 101 | 0.326 | 1.000 | 0.899 |
| 24 words | 193 | **0.005** | **1.000** | 0.710 |

The attention model is perfect at every length; the bottleneck model falls off a
cliff. Alignment score 0.729 median — high, and *not* 1.0 by construction, since
the month digits must come from a month *name* and count as misses.

**What surprised me.** The gap between the two accuracy measures. At 193
characters the bottleneck model still gets **71% of characters** right while
getting **0.5% of dates** right. It has learned the *shape* of the answer — four
digits, dash, two digits, dash, two digits, roughly the right ranges — and lost
the content. That is the most honest picture of what a too-small channel does:
not noise, but confident, well-formed, wrong output. Character accuracy would
have made this look like a mild degradation; exact match shows it is total.

![Accuracy by source length](figures/c/c2_accuracy_by_length.png)
![Attention map](figures/c/c2_attention_map.png)
![Training loss](figures/c/c2_training.png)

**Open threads.** Bahdanau's additive score, which has an extra parameter matrix
and is claimed to matter for longer sources. Coverage penalties, the standard fix
for attention that revisits one position and drops another. And beam search
instead of greedy decoding, which should matter most exactly where the bottleneck
model is uncertain.

**Reading:** [Neural Machine Translation by Jointly Learning to Align and
Translate](https://arxiv.org/abs/1409.0473) (Bahdanau et al.);
[Effective Approaches to Attention-based NMT](https://arxiv.org/abs/1508.04025) (Luong et al.);
[Sequence to Sequence Learning](https://arxiv.org/abs/1409.3215) (Sutskever et al.).

---

## C3 — A transformer from scratch

**Question.** Does the implementation actually respect causality — and how much
of a transformer's ability is the positional encoding?

### The math

**Scaled dot-product attention**, and why the scale is there:

$$\operatorname{Attention}(Q, K, V) = \operatorname{softmax}\!\left(\frac{QK^\top}{\sqrt{d_k}} + M\right)V .$$

If the entries of $q$ and $k$ are independent with zero mean and unit variance,
then $q^\top k = \sum_{i=1}^{d_k} q_i k_i$ has variance $d_k$. Without the
$1/\sqrt{d_k}$, logits grow like $\sqrt{d_k}$, the softmax saturates toward a
one-hot distribution, and its Jacobian $\operatorname{diag}(p) - pp^\top$ goes to
zero — attention stops learning precisely when the model is made wider. Dividing
by $\sqrt{d_k}$ holds the logit variance at 1 regardless of head width.

**The causal mask** is $M_{ij} = -\infty$ for $j > i$ and 0 otherwise, applied
*before* the softmax so the forbidden positions get exactly zero weight rather
than a small one. This is trivially easy to get subtly wrong — off by one, or
masked after the softmax — and a model that leaks a single future position still
trains beautifully while being worthless. The rung therefore checks it: perturb
the last token, and no earlier logit may move.

**Positional encoding, and what happens without it.** Self-attention is
*permutation-equivariant*: for a permutation matrix $P$,

$$\operatorname{Attention}(PQ, PK, PV) = P\operatorname{Attention}(Q, K, V),$$

so a transformer with no positional information cannot distinguish "abc" from
"cba" — with a causal mask it degenerates to a function of the *multiset* of
preceding characters. The sinusoidal encoding

$$PE_{(t, 2i)} = \sin\!\left(\frac{t}{10000^{2i/d}}\right), \qquad PE_{(t,2i+1)} = \cos\!\left(\frac{t}{10000^{2i/d}}\right)$$

fixes this with a useful extra property: for a fixed offset $k$, the pair
$(\sin\omega t, \cos\omega t)$ maps to $(\sin\omega(t+k), \cos\omega(t+k))$ by a
rotation that depends on $k$ but **not** on $t$, so "$k$ steps back" is a linear
operation the attention can express uniformly across the sequence.

**Pre-LN blocks.** The rung normalizes inside the residual branch,
$x \mapsto x + f(\mathrm{LN}(x))$, rather than the 2017 paper's
$x \mapsto \mathrm{LN}(x + f(x))$. Post-LN puts a normalization on the skip path
itself and needs learning-rate warmup to train at all; pre-LN keeps the identity
path clean — B4's argument again.

**Results** (same corpus, split, eval windows, steps, batch and context as C1;
medians over seeds 0–2):

| variant | parameters | bits/char |
| --- | --- | --- |
| learned positions | 1.82M | **2.309** |
| sinusoidal positions | 1.80M | 2.349 |
| **no positional encoding** | 1.80M | **2.641** |
| learned positions, at C1's budget | 242k | 2.508 |

Removing position costs **0.304 bits/char** (range 0.265–0.348) — the largest
ablation in the part, and exactly what the equivariance argument predicts.
Learned beats sinusoidal by 0.040 at this scale. The causal check returns
**exactly 0.0** on every seed, for both the logit perturbation and the attention
mass above the diagonal.

**What surprised me — and the result I cannot explain away.** Line up C1 and C3,
which share every training setting:

| model | parameters | bits/char |
| --- | --- | --- |
| vanilla RNN (wide) | 343k | **2.263** |
| transformer, learned PE | 1.82M | 2.309 |
| LSTM | 320k | 2.334 |
| transformer at C1's budget | 242k | 2.508 |

**The wide vanilla RNN beats every transformer arm, using 5× fewer parameters
than the best of them**, and the parameter-matched transformer is the worst model
in the comparison. That is not the story anyone expects in 2026, and it is not a
claim that transformers are worse — it is a claim about this regime: 500 steps,
64 characters of context, 250k characters of training text, and *no warmup* for
the transformer, which B5 showed is the one thing Adam-family optimizers are
picky about at the start of training. Transformers are built to exploit long
context and large data, and this rung deliberately gives them neither. The honest
summary is that I measured a regime where the inductive bias of recurrence is
worth more than the flexibility of attention, and that regime is small.

![Positional ablation](figures/c/c3_positional_ablation.png)
![Attention map](figures/c/c3_attention_map.png)
![Positional similarity](figures/c/c3_positional_similarity.png)
![Bits per character](figures/c/c3_bits_per_char.png)

**Open threads.** Warmup plus a cosine schedule, which would make the comparison
above fair to the transformer — the obvious next run. Longer context, where the
recurrent models must lose. RoPE instead of additive encodings. And KV caching,
which is the difference between a sampler that is $O(T)$ per token and one that
is $O(T^2)$.

**Reading:** [Attention Is All You Need](https://arxiv.org/abs/1706.03762);
[The Annotated Transformer](https://nlp.seas.harvard.edu/annotated-transformer/);
Xiong et al., *On Layer Normalization in the Transformer Architecture* (pre-LN vs post-LN).

---

## C4 — Fine-tuning a small ViT

**Question.** A ViT is C3's block applied to image patches. Without a CNN's
built-in priors, how much does it depend on pretraining?

### The math

**Patchification is the only image-specific step.** An image
$x \in \mathbb{R}^{H\times W\times C}$ is cut into $N = HW/P^2$ patches of side
$P$, each flattened and projected:

$$z_0 = \left[x_{\text{cls}};\, x^1_p E;\, x^2_p E;\, \dots;\, x^N_p E\right] + E_{\text{pos}}, \qquad E \in \mathbb{R}^{P^2C \times d}.$$

After that it is C3's encoder verbatim. At $224^2$ with $P=16$ that is $N=196$
tokens, and attention's $O(N^2)$ cost is why input resolution is expensive here
in a way it is not for a CNN.

**What the architecture does not assume.** A convolution hard-codes locality
(a unit sees a small neighbourhood) and translation equivariance
($f(\text{shift}(x)) = \text{shift}(f(x))$). Self-attention assumes neither:
every patch can attend to every other from layer 1, and position enters only
through $E_{\text{pos}}$, which is *learned*. That is more expressive and
strictly less constrained — so with little data the ViT must learn from examples
what a CNN gets for free, which is the standard explanation for ViTs being
data-hungry and the thing this rung measures.

**Probe versus fine-tune** ask different questions. A linear probe on frozen
features asks *are the pretrained representations already linearly separable for
this task?* — it optimizes $\min_W \sum_i \ell(W\phi(x_i), y_i)$ with $\phi$
fixed, a convex problem in $W$. Fine-tuning asks *can the representation itself
be improved?*, at 7000× the trainable parameters and the risk of destroying
pretrained features with too large a step (hence the much smaller learning rate).

**Results** (ViT-Tiny, 4-class CIFAR subset — plane/car/ship/truck — 1500 train
images, 2 epochs, medians over seeds 0–2):

| arm | test accuracy | trainable parameters | seconds (T4) |
| --- | --- | --- | --- |
| from scratch | 0.420 | 5.53M | 12.8 |
| linear probe | 0.866 | **772** | 5.5 |
| full fine-tune | **0.955** | 5.53M | 12.8 |

Pretraining is worth **+0.452** (probe over scratch, range 0.421–0.458);
fine-tuning adds a further **+0.090** over the probe (range 0.073–0.100).

**What surprised me.** Not the ordering — the *ratio*. The linear probe trains
**772 parameters**, four classes' worth of weights on a 192-dimensional feature,
and lands within 9 points of a full fine-tune of 5.5 million. Essentially all of
the task-relevant work was already done by ImageNet pretraining; the fine-tune is
polishing. And the from-scratch arm at 0.420 is barely above the 0.25 chance of
four classes — 1500 images is simply not enough to teach a transformer what an
edge is.

![Arms](figures/c/c4_arms.png)
![Final accuracy](figures/c/c4_final_accuracy.png)
![Trainable parameters](figures/c/c4_trainable_parameters.png)
![Confusion](figures/c/c4_confusion.png)

**Open threads.** Freezing progressively more blocks, to find where "probe" turns
into "fine-tune". Data augmentation, which is how the ViT paper's own
from-scratch results were rescued. DINO or MAE features instead of supervised
ImageNet ones — that comparison is part D's business.

**Reading:** [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929) (ViT);
Steiner et al., *How to train your ViT*; [timm](https://github.com/huggingface/pytorch-image-models).

---

## What part C settled

1. **Every architecture here is a different answer to "how does information
   travel?"** Recurrence carries it in a state and pays multiplicatively per
   step; gating buys an additive path; attention refuses to compress and pays
   $O(T^2)$. The measurements line up with the algebra: the RNN's geometric
   Jacobian product shows up as a 27-step horizon, the LSTM's product of forget
   gates as 63, and attention's convex combination as a bottleneck-vs-attention
   gap that grows from 0.003 to 0.98 as the source gets longer.
2. **Reach is not the same as use.** The LSTM's gradient arrives 63 steps back
   and it still cannot solve 30-step recall at this budget, and its extra memory
   *loses* to a wide vanilla RNN on the actual language-modelling task. An
   architecture's capability and what a training run extracts from it are
   different quantities.
3. **The famous architecture lost, and the regime is why.** A 343k-parameter
   vanilla RNN beat a 1.8M-parameter transformer on 250k characters at 64 tokens
   of context with no warmup. Nothing about that contradicts the literature; it
   locates the boundary of it, which is more useful than a fourth reproduction of
   a result everyone already believes.
4. **Two metrics on the same run told opposite stories.** C2's bottleneck model
   keeps 71% character accuracy while getting 0.5% of dates right. Choosing the
   metric that matches the task's actual failure mode did more work here than any
   modelling decision.
5. **A seed does not pin a number, and which numbers it fails to pin is
   informative.** Across three full sweeps — one on CPU, two on separate T4
   sessions — C1, C3 and C4 reproduced their medians *exactly* (bits/char,
   horizons, accuracies, the causal-leak zero). C2's bottleneck exact-match did
   not: 0.363, 0.349 and 0.326 at 101 characters, 0.012, 0.017 and 0.005 at 193.
   Seeding does not make cuDNN's backward deterministic, and the bottleneck arm
   sits exactly at the edge of failing, where an all-or-nothing metric over ten
   characters amplifies any tiny difference in weights. Its character accuracy,
   measuring the same runs, moved by only ~0.015. So the metric that made the
   failure legible is also the least stable one — both facts follow from it being
   a threshold measure near a threshold. The conclusion (the bottleneck collapses
   while attention does not) is unaffected on every run; the third decimal place
   is fiction. Figures and metrics here therefore come from one sweep on one
   device, and `--check` should be run on that device.

**Cost of the part:** ~1,800 lines including tests, 21 test cases, and a full
three-seed sweep in about 8 minutes on a T4 (C1 67s, C2 25s, C3 33s, C4 35s per
seed) — the first part where the CPU-only path stopped being comfortable.

---

*Figures regenerate with `uv run python tools/writeup_figures.py C`; the numbers
they are quoted against live in [`data/c-metrics.json`](data/c-metrics.json).
The [part README](../../parts/c_sequence_attention/README.md) is the short
version of this document.*
