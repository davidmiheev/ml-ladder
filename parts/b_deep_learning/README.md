# B. Deep learning foundations

**Lab:** [karpathy/micrograd — autodiff reference implementation](https://github.com/karpathy/micrograd)

| Rung | Goal | Module |
| ---- | ---- | ------ |
| B1 | Scalar autodiff engine (micrograd-style) plus a tiny MLP trained on it | `b1_autodiff_engine.py` |
| B2 | MLP on MNIST — manual backprop, then repeat with PyTorch autograd and diff the gradients | `b2_mlp_mnist.py` |
| B3 | CNN on CIFAR-10 with a batchnorm/dropout ablation | `b3_cnn_cifar10.py` |
| B4 | ResNet block implementation, trained on CIFAR-10/100 | `b4_resnet_blocks.py` |
| B5 | Optimizer bake-off — SGD vs Adam vs AdamW, with LR-schedule comparison | `b5_optimizer_bakeoff.py` |

Run any rung with `ladder run B1`; `--help` lists its flags. B2–B5 need
`uv sync --extra torch`. Datasets download on first use into `data/`
(git-ignored) through `mlladder.vision`; the torch plumbing every rung shares
lives in `mlladder.torchkit`.

**The long version is [docs/writeups/b-deep-learning.md](../../docs/writeups/b-deep-learning.md)**
— derivations, per-seed numbers, surprises and open threads. What follows is
the summary.

## Sized for CPU

Every rung here runs on four CPU cores in minutes, on subsets of the real
datasets. The comparisons are what matter — all arms share data, schedule and
seed — and none of the absolute numbers are competitive with a GPU run. Scale
any of them up with flags (`--train-samples`, `--epochs`).

## What the default runs show

**B1 — the chain rule, one scalar at a time.** Gradients match central
differences to ~2e-8 (median over seeds; 9e-9 to 1.4e-7), about as close as a
finite difference over a 90-term sum gets before its own accumulation error
dominates. The bill for that clarity: 120 samples through a 105-parameter MLP
builds an 18,380-node graph, and 60 epochs take ~18s against ~5ms for the same
network in numpy — a factor of ~3,550. Vectorization is not an optimization
here; it is the difference between a toy and a tool.

**B2 — hand-derived backprop is autograd.** Same initial weights, same batch:
the largest gradient disagreement across all six parameter tensors is ~3e-7
against a gradient scale of 0.17, i.e. float32 rounding. Losses agree to six
decimals and the two training runs stay locked together epoch by epoch. The
matrix form is also *faster* than torch at this size — the framework's
per-op overhead dominates when the ops are this small.

**B3 — batchnorm is an optimizer, dropout is a regularizer.** BatchNorm adds
+0.137 test accuracy at the median (+0.097 to +0.159) while *doubling* the
train-test gap (0.066 → 0.130); dropout adds ~2.5 points and barely moves the
gap (0.062). The labels people attach to these two are backwards as often as
not. The learning-rate probe is sharper but not free of noise: at lr=0.5 the
plain network collapses to chance on all three seeds (0.094–0.098) while the
batchnorm one still learns on two of three.

**B4 — degradation, not overfitting.** Stacking plain blocks makes the network
worse *on the training set*: median train accuracy goes 0.478 → 0.540 → 0.453 at
depths 2/4/8, a loss of 8.7 points for the last doubling, while the residual
version of the identical architecture loses 2.0. At depth 8 the skips are worth
+0.072 test accuracy at the median, though the range (−0.016 to +0.091) includes
one seed where plain won — the training column is the robust evidence, not the
test column. The usual explanation, vanishing gradients, is not what is
happening: with BatchNorm the plain net's early-layer gradients are healthy.
What the identity paths change is the *spread* of gradient magnitude across
depth (first/last norm ratio 5.0–9.0 plain vs 2.9–3.9 residual, non-overlapping
across seeds).

**B5 — Adam's advantage is robustness, not the optimum.** At each optimizer's
own best learning rate, SGD with momentum wins (0.874 with cosine) and AdamW is
a whisker behind (0.867 median). The difference shows up in the learning-rate
sweep: AdamW clears 0.72 at four of five rates spanning three orders of
magnitude and peaks at 0.841, while plain SGD never exceeds 0.737 anywhere.
And decoupling weight decay is worth +0.069 accuracy at the same λ (the
tightest effect in the part, +0.064 to +0.069) — Adam's coupled L2 divides the
decay by the second-moment estimate, over-shrinking the weights to 24% of
AdamW's final norm.

## Reuse between rungs

B2's `ManualMLP` is the vectorized form of B1's scalar engine; B4's residual
block is B3's conv stack with a skip; B5 reuses `mlladder.torchkit.fit` like
everything else in the part. A1's variance floor makes a second appearance in
B5: SGD at a constant step ends at 0.813 having peaked at 0.843 — bouncing
around the optimum, exactly as in part A.
