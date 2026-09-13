# A. Classical ML from scratch

*numpy only — no sklearn*

**Lab:** [Stanford CS229 problem sets](https://cs229.stanford.edu/)

| Rung | Goal | Module |
| ---- | ---- | ------ |
| A1 | Linear & logistic regression via batch/mini-batch gradient descent | `a1_linear_logistic_regression.py` |
| A2 | Decision tree + random forest from scratch (Gini/entropy splits, bagging) | `a2_trees_and_forests.py` |
| A3 | k-means and PCA from scratch, compared against each other on the same toy dataset | `a3_kmeans_and_pca.py` |
| A4 | Minimal gradient-boosting implementation, benchmarked against XGBoost | `a4_gradient_boosting.py` |

Run any rung with `ladder run A1`; `--help` lists its flags. Outputs land in
`artifacts/<rung>/<timestamp>/`. Data comes from `mlladder.datasets`, which is
numpy-only for the same reason this part is.

**The long version is [docs/writeups/a-classical.md](../../docs/writeups/a-classical.md)**
— design decisions, per-seed numbers, surprises and open threads. What follows
is the summary.

## What the default runs show

Numbers from `--seed 0` with default flags, with medians across seeds 0-2 where
a single seed would mislead; every rung prints these as it goes.

**A1 — the schedule changes the path, not the destination.** All three
schedules land on the closed-form solution (test RMSE 0.303 vs. 0.303 for the
normal equation, `|w − w*|` ≈ 0.02–0.05) *once the noisy ones decay their step
size*. With a constant step, mini-batch and SGD rattle around the optimum
instead of settling into it — the variance floor, visible as a flat, noisy tail
in `linear_convergence.png`. Run `--no-decay` to see the failure directly.

**A2 — depth is variance, bagging buys it back.** On spirals with `noise=0.7`, a
single tree peaks at depth 5 (test 0.847) and then memorises: by depth 8 it is
at train 0.938 / test 0.767, and the gap stops growing only because the tree
runs out of splits. An 80-tree forest over the same depth-12 trees gets 0.887,
beating the tuned single tree by +0.073 median across seeds. Its out-of-bag
estimate (0.836) is consistently pessimistic — each OOB prediction averages only
the ~37% of trees that never saw the sample — but tracks the test score closely
enough for model selection without a validation split. Gini and entropy are
within noise of each other; the split *criterion* matters far less than depth.

**A3 — two answers to one least-squares question.** k-means picks out the true
cluster count exactly: the adjusted Rand index peaks at k=4 (ARI 1.00) while
purity keeps climbing with k, which is precisely why purity alone is a bad
score. PCA needs 2 components for 59% of the variance in 8-D. Per unit of
budget, the continuous compression wins — at k=4, PCA's reconstruction error is
0.93 against k-means' 2.41 — because a rank-k subspace is a strictly richer
approximation than k points. Clustering in the 2-D PCA subspace costs a little
accuracy (ARI 0.96 vs. 1.00 on raw features) in exchange for 4× fewer
dimensions.

**A4 — from-scratch boosting matches XGBoost's accuracy and is ~10× slower.**
On Friedman #1 at matched depth, learning rate and rounds: ours 1.482 test RMSE
against XGBoost's 1.452 on seed 0 — but across seeds 0-2 the gap is -0.008
median (range -0.031 to +0.030), i.e. no sign. What the library buys at this
scale is speed (9-15× faster fits), not accuracy; histogram binning and Newton
leaf values start to matter on data much larger than 800×10. The sweep shows
the shrinkage trade-off: `lr=0.03` is still improving at round 150, `lr=0.3`
bottoms out around round 55 and then degrades, and `lr=1.0` never gets there at
all. Boosting beats every A2 baseline on the same data (forest 2.33, depth-10
tree 3.14).

The XGBoost benchmark is optional — `uv sync --extra boost` to enable it; the
rung reports the from-scratch numbers alone when it is missing. It uses
XGBoost's native Booster API rather than `XGBRegressor`, whose sklearn wrapper
would drag sklearn into a part that bans it.

## Reuse between rungs

A4 imports A2's `DecisionTree` with `criterion="mse"` as its weak learner, so
the boosting rung is genuinely built on the tree rung rather than on a second
implementation of one.
