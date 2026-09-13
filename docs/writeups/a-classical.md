# Part A — Classical ML from scratch

> **Status:** complete (A1–A4) · **Constraint:** numpy only, no sklearn
> **Numbers:** seeds 0–2, reference seed 0 · raw values in [`data/a-metrics.json`](data/a-metrics.json)
> **Reproduce:** `ladder run A1` … `A4`, or `uv run python tools/writeup_figures.py A`

Four algorithms, written from the loss function up, on synthetic data generated
by [`mlladder.datasets`](../../src/mlladder/datasets.py) — also numpy-only, for
the same reason the rungs are.

The through-line: **every one of these methods is a squared-error or log-loss
objective plus a search strategy, and what separates them is how they spend
variance.** Gradient descent spends it on step noise, a deep tree on splits, a
forest averages it away, boosting trades it against bias one shrunken step at a
time. That is the frame the later parts inherit — a transformer is a different
function class, not a different bargain.

Notation throughout: $X \in \mathbb{R}^{n\times d}$ is the design matrix,
$y \in \mathbb{R}^n$ the targets, $X_c = X - \mathbf{1}\bar{x}^\top$ the centred
design, and $\lVert\cdot\rVert$ the Euclidean norm.

---

## A1 — Linear & logistic regression by gradient descent

**Question.** Linear regression has a closed form. So what does gradient
descent actually cost you, and how does the *batch schedule* — full batch,
mini-batch, single sample — change the answer?

### The math

**Least squares.** With $f(x) = w^\top x + b$ and an L2 penalty on the weights
only,

$$\mathcal{L}(w, b) = \tfrac{1}{n}\lVert Xw + b\mathbf{1} - y\rVert^2 + \lambda\lVert w\rVert^2 ,$$

$$\nabla_w\mathcal{L} = \tfrac{2}{n}X^\top(Xw + b\mathbf{1} - y) + 2\lambda w, \qquad \partial_b\mathcal{L} = \tfrac{2}{n}\mathbf{1}^\top(Xw + b\mathbf{1} - y).$$

Setting both to zero and eliminating $b$ gives the closed form on centred data,

$$w^\star = \left(X_c^\top X_c + \lambda n I\right)^{-1} X_c^\top (y - \bar{y}\mathbf{1}), \qquad b^\star = \bar{y} - \bar{x}^\top w^\star .$$

The intercept equation $\partial_b\mathcal{L}=0$ says exactly *the residuals have
mean zero*, which is why $b$ absorbs the level of $y$. Penalizing it instead
would give $b^\star = (\bar y - \bar x^\top w^\star)/(1+\lambda')$ — an estimator
that shrinks predictions toward the origin and is not equivariant under shifting
$y$, which is why it is excluded from the penalty here.

**Why the schedule matters.** The Hessian is $H = \tfrac{2}{n}X^\top X + 2\lambda I$,
with eigenvalues in $[\mu, L]$. Full-batch gradient descent is the linear
recurrence

$$w_{t+1} - w^\star = (I - \eta H)(w_t - w^\star),$$

so the error contracts by $\rho(\eta) = \max\{|1-\eta\mu|,\,|1-\eta L|\}$ — convergent
iff $0 < \eta < 2/L$, and at best $\rho = \frac{\kappa-1}{\kappa+1}$ with
$\kappa = L/\mu$. Deterministic, geometric, boring in the good way.

A stochastic gradient over a batch $B$ is unbiased with covariance
$\operatorname{Cov}(\hat{g}) = \Sigma/|B|$, and the recurrence picks up a noise
term that does **not** vanish as $w_t \to w^\star$. For a constant step the
iterate converges to a stationary distribution rather than a point, with

$$\limsup_{t\to\infty}\ \mathbb{E}\lVert w_t - w^\star\rVert^2 \;=\; O\!\left(\frac{\eta\,\operatorname{tr}\Sigma}{\mu\,|B|}\right) \quad\Longrightarrow\quad \lVert w_t - w^\star\rVert \sim \sqrt{\eta / |B|}.$$

That is the *variance floor*: a ball around the optimum whose radius shrinks
only if $\eta$ shrinks. The Robbins–Monro conditions
$\sum_t \eta_t = \infty$, $\sum_t \eta_t^2 < \infty$ are what collapse the ball
to a point, and the schedule implemented here, $\eta_t = \eta_0/(1+\gamma t)$,
satisfies both (harmonic series diverges, its squares converge).

**Logistic regression.** Bernoulli likelihood with $\sigma(z) = (1+e^{-z})^{-1}$
gives the average negative log-likelihood

$$\mathcal{L}(w,b) = -\tfrac{1}{n}\sum_i \Big[y_i\log\sigma(z_i) + (1-y_i)\log\big(1-\sigma(z_i)\big)\Big], \qquad z_i = w^\top x_i + b,$$

and the identity $\sigma' = \sigma(1-\sigma)$ collapses the chain rule to

$$\nabla_w \mathcal{L} = \tfrac{1}{n}X^\top\big(\sigma(z) - y\big),$$

the same shape as least squares with $\sigma(z)$ in place of the linear
prediction — both are canonical-link GLMs, which is precisely why one optimizer
serves both. The Hessian $\tfrac{1}{n}X^\top S X$ with
$S = \operatorname{diag}\big(\sigma_i(1-\sigma_i)\big) \succeq 0$ makes the problem
convex, and strictly so when $X$ has full rank and no $\sigma_i \in \{0,1\}$.
(On perfectly separable data no minimizer exists — $\lVert w\rVert \to \infty$
drives the loss to zero — which is what the L2 term quietly prevents.)

**What I built.** One optimizer, two losses. Three details worth keeping:

- **The bias is a separate parameter**, not a column of ones, for the reason
  derived above. The test asserts it: ridge halves $\lVert w\rVert$ while the
  intercept stays at 5.0.
- **The sigmoid is branch-split by sign** — $1/(1+e^{-z})$ for $z \ge 0$ and
  $e^{z}/(1+e^{z})$ otherwise — so `exp` never sees a large positive argument.
  At $\pm 800$ the naive form overflows; this one returns 0 and 1.
- **The closed form is solved, not inverted**: `np.linalg.solve` on the centred
  normal equations.

**Results** (test RMSE and distance to the true weights, seeds 0–2):

| schedule | test RMSE (seed 0) | ‖w − w\*‖ median | with constant step |
| --- | --- | --- | --- |
| full batch | 0.3031 | 0.023 | 0.023 |
| mini-batch (32) | 0.3109 | 0.051 | **0.214** |
| SGD (1) | 0.3070 | 0.037 | **0.499** |
| normal equation | 0.3031 | 0.023 | — |

Full-batch descent reproduces the closed-form solution to four decimals
(`weight_error` 0.01794 against the normal equation's 0.01794 on seed 0 — the
same number, not a coincidence: same objective, one is just slower about it).

**What surprised me.** The noisy schedules *do not converge at all* with a
constant step size — and the theory above says exactly how badly. Measuring
$\lVert w - w^\star\rVert$ against batch size at fixed $\eta$, over
$|B| \in \{1, 2, \dots, 128\}$ and three seeds, the fitted slope is

$$\frac{d \log \lVert w - w^\star\rVert}{d \log |B|} = -0.54 \qquad (\text{theory: } -\tfrac12),$$

so the floor really does scale as $|B|^{-1/2}$. I had expected mini-batch to
simply converge *faster* per epoch, since it takes 15× more updates per epoch;
instead the comparison is meaningless until the step schedule is part of the
setup. Decay it, and all three schedules land on the closed form together.

![Linear convergence by schedule](figures/a/a1_linear_convergence.png)
![Logistic convergence by schedule](figures/a/a1_logistic_convergence.png)
![Logistic decision boundary](figures/a/a1_logistic_boundary.png)

**Open threads.** Momentum and adaptive steps are exactly the B5 bake-off.
Logistic regression has a Newton step — IRLS, $w \leftarrow w - H^{-1}\nabla$
with the $X^\top S X$ above — that converges in a handful of iterations; worth
implementing to watch second-order information pay off before it reappears as
XGBoost's Newton leaf values in A4. Multiclass softmax is a small extension of
the same loop.

**Reading:** ESL ch. 3–4; CS229 notes 1 (supervised learning, GLMs).

---

## A2 — Decision tree and random forest

**Question.** Trees are the one model here with no continuous parameters at
all — how much does the split *criterion* matter, and how much is really about
depth? And does bagging buy back what depth costs?

### The math

**Impurity and gain.** For a node with class distribution $p$,

$$G(p) = 1 - \sum_k p_k^2 = \sum_k p_k(1-p_k), \qquad H(p) = -\sum_k p_k \log_2 p_k .$$

Gini is the variance of the one-hot class indicator (equivalently, the
probability that two independent draws from the node disagree); entropy is
expected surprisal, and its gain is the mutual information
$I\big(Y;\mathbf{1}[x_j \le \tau]\big)$. A split $(j,\tau)$ is scored by

$$\Delta = I(p) - \frac{n_L}{n}I(p_L) - \frac{n_R}{n}I(p_R).$$

Because $p = \frac{n_L}{n}p_L + \frac{n_R}{n}p_R$ and both $G$ and $H$ are
strictly concave on the simplex, Jensen gives $\Delta \ge 0$ always, with
equality iff $p_L = p_R$. **Every split weakly helps** — which is why the tree
needs external stopping rules rather than an internal one, and why depth, not
the criterion, is the real hyper-parameter.

**Regression splits, and the identity the implementation runs on.** With
$I = \operatorname{Var}$, the leaf constant is the mean
($\arg\min_c \sum (y_i - c)^2 = \bar{y}$), and the weighted reduction telescopes:

$$n\operatorname{Var}(y) - n_L\operatorname{Var}(y_L) - n_R\operatorname{Var}(y_R) \;=\; \frac{\left(\sum_L y\right)^2}{n_L} + \frac{\left(\sum_R y\right)^2}{n_R} - \frac{\left(\sum y\right)^2}{n}.$$

Everything on the right is a *cumulative sum*. Sort the feature once, take
`cumsum`, and every threshold is scored in one vectorized pass:
$O(dn\log n)$ per node instead of the naive $O(dn^2)$. This identity returns
verbatim as XGBoost's gain formula in A4.

**Why bagging works.** For $B$ identically distributed trees with variance
$\sigma^2$ and pairwise correlation $\rho$,

$$\operatorname{Var}\left(\frac{1}{B}\sum_{b=1}^{B} T_b\right) = \rho\sigma^2 + \frac{1-\rho}{B}\sigma^2 .$$

Averaging only kills the second term; the floor $\rho\sigma^2$ is why
`max_features = √d` exists — decorrelating the trees attacks the term that
$B \to \infty$ cannot. Bias is essentially unchanged (each deep tree is close to
unbiased), so the entire gain is variance reduction.

**Why out-of-bag is pessimistic, not just noisy.** A bootstrap sample misses a
given point with probability

$$\left(1 - \tfrac{1}{n}\right)^{n} \xrightarrow[n\to\infty]{} e^{-1} \approx 0.368,$$

so each OOB prediction averages $\approx 0.368B$ trees. By the variance formula
above, an ensemble of $0.368B$ trees is strictly worse than one of $B$ — the OOB
score measures a forest roughly a third the size of the one you ship. The bias
is downward and consistent, which is what makes it usable for model selection.

**What I built.** One `DecisionTree` covering both tasks — Gini and entropy for
classification, variance reduction for regression — because the search is
identical once a partition can be scored. That is what lets A4 embed this class
as its weak learner instead of reimplementing one. Candidates are restricted to
positions where the sorted feature value changes and both sides clear
`min_samples_leaf`. The forest adds bootstrap resampling plus a $\sqrt{d}$
feature subset per split, and scores itself out-of-bag.

**Results** (spirals, 3 classes, `noise=0.7` — deliberately hard):

| model | test accuracy (seed 0) | median (seeds 0–2) |
| --- | --- | --- |
| tree, depth 12, Gini | 0.760 | 0.760 |
| tree, depth 12, entropy | 0.780 | 0.787 |
| tree at its best depth | 0.847 (depth 5) | depth 3–8 |
| forest, 80 trees | 0.887 | 0.860 |
| forest out-of-bag estimate | 0.836 | 0.851 |

`forest − tree` is **+0.073 median (+0.007 to +0.107)**. Bagging reliably helps;
*how much* is itself seed-dependent, which is the honest version of the claim.

**What surprised me.** Two things. First, Gini versus entropy is a coin flip
(0.76 vs 0.78 on seed 0, and they swap on other seeds) — consistent with the
concavity argument, which says both are valid concave impurities and neither has
a claim to the *right* partition; depth swings accuracy by 8 points and the
criterion by under 2. Second, the OOB estimate is measurably pessimistic (0.836
vs a 0.887 test score), in the direction the $e^{-1}$ argument predicts; an
80-tree forest is OOB-scored as roughly a 29-tree one. The gap is somewhat
larger than tree count alone accounts for, so the counting argument is a
mechanism, not a complete model.

The depth sweep is the clean bias-variance picture: train accuracy climbs to
0.938 and stops (the tree runs out of splits), while test peaks at depth 5 and
falls back, opening a 17-point gap.

![Depth sweep](figures/a/a2_depth_sweep.png)
![Forest size vs OOB](figures/a/a2_forest_size.png)
![Single tree boundary](figures/a/a2_tree_boundary.png)
![Forest boundary](figures/a/a2_forest_boundary.png)

The two boundary plots are the whole argument in one pair of images: the single
tree carves the plane into confident rectangles that chase individual noisy
points; the forest's averaged boundary is visibly smoother and follows the
spiral arms.

**Open threads.** Cost-complexity pruning — minimizing
$\sum_{\text{leaves}} n_m I_m + \alpha|T|$ — as the principled alternative to a
depth cap. Extremely randomized trees (random thresholds rather than best ones)
to test how much of the forest's gain is the $\rho$ term rather than the fit.
And the known bias of impurity-gain importances toward high-cardinality
features, which A4's importance plot quietly depends on.

**Reading:** [Random Forests](https://link.springer.com/article/10.1023/A:1010933404324)
(Breiman 2001); ESL ch. 9, 15.

---

## A3 — k-means and PCA on one dataset

**Question.** Both are unsupervised, both minimize squared reconstruction
error, and both are usually taught apart. Run them on the same cloud: what does
each actually recover?

### The math

**k-means is alternating minimization.** The objective over assignments
$c: \{1..n\} \to \{1..k\}$ and centres $\mu$,

$$\phi(c, \mu) = \sum_{i=1}^{n} \lVert x_i - \mu_{c(i)}\rVert^2 ,$$

is minimized by Lloyd's algorithm as exact coordinate descent: with $\mu$ fixed,
$c(i) = \arg\min_k \lVert x_i - \mu_k\rVert^2$; with $c$ fixed, stationarity
$\sum_{i \in C_k}(x_i - \mu_k) = 0$ gives $\mu_k = \operatorname{mean}(C_k)$.
Both steps are non-increasing and there are finitely many partitions, so the
iteration terminates — at a *local* minimum, which is why restarts and seeding
carry the whole burden. k-means++ draws each new centre with probability
proportional to $D(x)^2$, the squared distance to the nearest chosen centre,
and buys the guarantee $\mathbb{E}[\phi] \le 8(\ln k + 2)\,\phi_{\mathrm{OPT}}$.

**PCA is the Eckart–Young theorem.** Minimizing reconstruction error over
rank-$k$ orthogonal projections,

$$\min_{V^\top V = I_k}\ \frac{1}{n}\sum_i \big\lVert (x_i - \bar{x}) - VV^\top(x_i - \bar{x}) \big\rVert^2 ,$$

is equivalent (by Pythagoras, since total variance is fixed) to *maximizing*
projected variance, and both are solved by the top-$k$ eigenvectors of
$S = \frac{1}{n-1}X_c^\top X_c$. Taking the SVD $X_c = U\Sigma V^\top$ gives them
without forming $S$ — better conditioned, since squaring the matrix squares its
condition number — with explained variance $\sigma_i^2/(n-1)$ and residual

$$\frac{1}{n}\sum_{i>k}\sigma_i^2 .$$

Eckart–Young–Mirsky says no other rank-$k$ linear compression does better in
Frobenius norm.

**The pairing.** Both are the same constrained approximation problem
$\min \lVert X - \hat{X}\rVert_F^2$ over different feasible sets: PCA over
rank-$k$ affine approximations, k-means over $\hat{X} = ZM$ with
$Z \in \{0,1\}^{n\times k}$ one-hot — also rank $\le k$, but *discrete*. The $k$
centroids span an affine subspace of dimension $\le k-1$ through the data mean,
and k-means lands each point on one of $k$ points of that subspace rather than
projecting orthogonally onto the best one, which forces the sandwich

$$E_{\text{k-means}}(k) \;\ge\; E_{\text{PCA}}(k-1) \;\ge\; E_{\text{PCA}}(k).$$

Measured on this dataset, it holds at every $k$ tested (e.g. $k=4$: 2.406 ≥
1.754 ≥ 0.929). Ding & He (2004) sharpen the connection: the continuous
relaxation of the k-means indicator matrix is spanned by the principal
components — PCA *is* relaxed k-means.

**Why purity misleads and ARI does not.** For a contingency table $n_{ij}$
between true classes and clusters, with row/column sums $a_i, b_j$,

$$\mathrm{ARI} = \frac{\sum_{ij}\binom{n_{ij}}{2} - \dfrac{\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}}{\binom{n}{2}}}{\dfrac{1}{2}\left[\sum_i\binom{a_i}{2} + \sum_j\binom{b_j}{2}\right] - \dfrac{\sum_i\binom{a_i}{2}\sum_j\binom{b_j}{2}}{\binom{n}{2}}} .$$

The subtracted term is the expected agreement under a hypergeometric null, so a
random partition scores 0 in expectation and splitting a correct cluster is
*penalized*. Purity, $\frac{1}{n}\sum_j \max_i n_{ij}$, has no such correction
and is non-decreasing under refinement — it is maximized by the singleton
clustering.

**What I built.** k-means with k-means++ seeding, Lloyd iterations to a relative
tolerance, and `n_init` restarts keeping the lowest inertia; an emptied cluster
is re-seeded at the worst-fit point. PCA via SVD with a sign convention so
components are reproducible across runs.

**Results** (4 gaussian blobs in 8-D, standardized):

| k | k-means reconstruction | PCA(k) reconstruction | PCA(k−1) | ARI vs true labels |
| --- | --- | --- | --- | --- |
| 2 | 5.692 | 3.293 | 5.152 | +0.495 |
| 3 | 4.000 | 1.754 | 3.293 | +0.713 |
| 4 | 2.406 | 0.929 | 1.754 | **+1.000** |
| 6 | 2.115 | 0.269 | 0.505 | +0.827 |

ARI peaks exactly at the true k on every seed, and reaches 1.000 — a perfect
partition. Clustering the 2-D PCA projection instead of the raw 8-D features
costs almost nothing (ARI 0.960 on seed 0, 0.991 median) for a 4× dimension
reduction.

**What surprised me.** Purity climbs monotonically with k — 1.000 at k=4, still
1.000 at k=8 — while ARI falls away from the true k. The formula above says it
must; watching it confidently reward the wrong answer is what makes it stick,
and the unit test now pins the pathological case (purity 1.0, ARI 0.0 for
one-point-per-cluster). The reconstruction gap also came out wider than I
expected — PCA wins by roughly 2.5× at every budget — but the sandwich
inequality says the direction was never in doubt; only the size was.

![Elbow](figures/a/a3_kmeans_elbow.png)
![Agreement with true labels](figures/a/a3_kmeans_agreement.png)
![Explained variance](figures/a/a3_pca_variance.png)
![Reconstruction error, same budget](figures/a/a3_reconstruction_error.png)
![PCA projection, true labels](figures/a/a3_pca_true_labels.png)
![PCA projection, k-means labels](figures/a/a3_pca_kmeans_labels.png)

**Open threads.** Gaussian mixtures by EM — the soft-assignment version, where a
responsibility $\gamma_{ik}$ replaces the hard indicator and Lloyd becomes a
special case at zero covariance. Kernel PCA, which is the bridge to part D's
representation learning. And choosing k honestly (silhouette, gap statistic)
rather than by peeking at the labels as I do here.

**Reading:** ESL ch. 14; MML book ch. 10 (PCA derived three ways).

---

## A4 — Gradient boosting, benchmarked against XGBoost

**Question.** How close does a from-scratch booster get to the library everyone
actually uses, given the same hyper-parameters — and where exactly does the
difference come from?

### The math

**Boosting is gradient descent in function space.** Forward stagewise additive
modelling builds $F_M = F_0 + \nu\sum_{m=1}^{M} h_m$ where each step would ideally
solve $h_m = \arg\min_h \sum_i L\big(y_i, F_{m-1}(x_i) + h(x_i)\big)$. Gradient
boosting replaces that with a first-order surrogate: compute the functional
gradient at the current predictions,

$$g_i = \left.\frac{\partial L(y_i, F)}{\partial F}\right|_{F = F_{m-1}(x_i)}, \qquad h_m \approx \arg\min_h \sum_i \big(-g_i - h(x_i)\big)^2 ,$$

and take a shrunken step $F_m = F_{m-1} + \nu h_m$. For squared loss
$L = \tfrac{1}{2}(y-F)^2$ the negative gradient is just the residual,
$-g_i = y_i - F_{m-1}(x_i)$, and the initial constant is
$F_0 = \arg\min_c \sum_i L(y_i, c) = \bar{y}$. That is the whole algorithm: fit a
tree to the residuals, add a fraction of it, repeat.

**The second-order version — and why our first-order one matches it here.**
XGBoost expands the loss to second order with
$h_i = \partial_F^2 L$, and adds leaf regularization:

$$\tilde{\mathcal{L}}^{(m)} = \sum_i \left[g_i f(x_i) + \tfrac{1}{2}h_i f(x_i)^2\right] + \gamma T + \tfrac{\lambda}{2}\sum_{j=1}^{T} w_j^2 ,$$

which is a quadratic in each leaf weight and therefore solvable in closed form:

$$w_j^\star = -\frac{\sum_{i \in I_j} g_i}{\sum_{i \in I_j} h_i + \lambda}, \qquad \text{Gain} = \frac{1}{2}\left[\frac{G_L^2}{H_L + \lambda} + \frac{G_R^2}{H_R + \lambda} - \frac{G^2}{H + \lambda}\right] - \gamma .$$

Now substitute squared loss, for which $h_i \equiv 1$, and set $\lambda = \gamma = 0$:

$$w_j^\star = \frac{\sum_{i\in I_j} r_i}{n_j} = \overline{r}_{I_j}, \qquad \text{Gain} = \frac{1}{2}\left[\frac{(\sum_L r)^2}{n_L} + \frac{(\sum_R r)^2}{n_R} - \frac{(\sum r)^2}{n}\right].$$

The leaf weight is the mean residual — exactly what a regression tree puts in
its leaves — and the gain is one half of **the variance-reduction identity from
A2**, so it selects the same split. Under squared loss with no regularization,
the two implementations are not similar algorithms; they are *the same
algorithm*, and any difference must come from defaults, arithmetic, or
engineering.

That is a testable claim, so it is now a test. Matching the defaults that
differ — `min_child_weight = min_samples_leaf = 5`, `base_score = ȳ`, exact
splits, $\lambda = 0$ — the two implementations agree on held-out predictions to

$$\max_i \lvert \hat{y}^{\text{ours}}_i - \hat{y}^{\text{xgb}}_i \rvert \approx 6\times10^{-6}$$

on targets with standard deviation 4.7 — float32 versus float64 rounding, and
nothing else.

**Shrinkage.** $\nu$ scales each step, so the effective complexity of $F_M$ grows
with $\nu M$: halving $\nu$ roughly doubles the rounds needed to reach the same
fit, while lowering the variance of the path taken there. Since each round can
only decrease training loss, test error is U-shaped in $M$ and the location of
its minimum is a function of $\nu$ — which is what the sweep below shows.

**The task.** Friedman #1 gives signal to five of ten features:

$$y = 10\sin(\pi x_0 x_1) + 20\left(x_2 - \tfrac{1}{2}\right)^2 + 10x_3 + 5x_4 + \varepsilon, \qquad x \sim \mathcal{U}[0,1]^{10} .$$

**What I built.** Squared-loss gradient boosting over A2's regression tree, with
`subsample < 1` for the stochastic variant, and `staged_rmse` to replay the
ensemble round by round. The benchmark goes through XGBoost's **native Booster
API**, not `XGBRegressor` — the sklearn wrapper imports sklearn, and this part
bans it. (That was not a guess; `XGBRegressor` raised
`ImportError: sklearn needs to be installed`.)

**Results** (Friedman #1, 800 samples, 10 features of which 5 carry signal):

| model | test RMSE (seed 0) | median (seeds 0–2) |
| --- | --- | --- |
| tree, depth 3 | 3.457 | 3.327 |
| tree, depth 10 | 3.138 | — |
| random forest, 60 trees | 2.333 | 2.333 |
| **ours, 150 rounds** | **1.482** | **1.588** |
| ours + subsample 0.8 | 1.461 | 1.492 |
| **XGBoost, 150 rounds** | **1.452** | **1.595** |

**What surprised me — and a claim I had to retract.** On seed 0, XGBoost wins by
2% and it is tempting to write "we land within 2% of the library". Across seeds
0–2 the gap is **−0.008 median, range −0.031 to +0.030**: our implementation is
ahead on two of three seeds. There is no measurable accuracy difference — and
once the derivation above is in hand, there *cannot* be one, because at
$\lambda=0$ with squared loss the algorithms coincide. What the library buys is
**9–15× faster fits** (0.75s vs 0.07s): histogram binning, threading and C++.
My first writeup of this rung stated a seed-0 artifact as a finding; the
multi-seed metrics file exists because of it, and the equivalence test exists so
the claim cannot silently rot.

The learning-rate sweep is the clearest picture of shrinkage in the part:
`lr=0.03` is still improving at round 150 (under-trained), `lr=0.1` converges
cleanly, `lr=0.3` bottoms out around round 55 and then *degrades*, and `lr=1.0`
never gets there at all — the U-shape the $\nu M$ argument predicts, with its
minimum sliding left as $\nu$ grows.

Pooled split-gain importance puts 74–79% of the gain on $x_0$–$x_4$ — exactly
the five features Friedman #1 gives signal to — an end-to-end check that the
tree, the boosting loop and the importance accounting all agree.

![Staged test error by learning rate](figures/a/a4_staged_test_error.png)
![Train loss](figures/a/a4_train_loss.png)
![Model comparison](figures/a/a4_model_comparison.png)
![Feature importance](figures/a/a4_feature_importance.png)

**Open threads.** Implement the Newton step properly — $w^\star = -\sum g/(\sum h + \lambda)$
for a loss where $h \not\equiv 1$, e.g. log-loss where $h_i = p_i(1-p_i)$ — since
that is the point at which the second-order version stops being equivalent and
starts being better. Early stopping on a validation fold instead of a fixed round
count. And histogram binning, to find out where the 10× actually goes.

**Reading:** [XGBoost](https://arxiv.org/abs/1603.02754) (Chen & Guestrin);
ESL ch. 10 (boosting as forward stagewise additive modelling).

---

## What part A settled

1. **One objective, many search strategies.** Least squares appears four times:
   A1's normal equation, A2's variance-reduction splits, A3's inertia and
   reconstruction error, A4's residual fitting. Twice the *same identity* shows
   up — the cumulative-sum form of variance reduction is A2's split scan and
   A4's boosting gain — which is why A4 could reuse A2's tree unmodified.
2. **The math predicts the measurements, and the measurements audit the math.**
   The variance floor scales as $|B|^{-0.54}$ against a predicted $|B|^{-1/2}$;
   the k-means/PCA sandwich holds at every $k$; the two boosters agree to
   $6\times10^{-6}$ once their defaults are aligned. Each of those was a
   derivation first and a number second.
3. **Single-seed numbers lie.** The XGBoost gap changed sign across three seeds.
   Every headline number now carries a range, regenerated by
   `tools/writeup_figures.py`.
4. **Variance is the whole game at this scale.** Depth, learning rate and batch
   size are all variance dials, and the two cures — averaging (bagging,
   restarts) and shrinkage (small steps, step decay, small $\nu$) — recur in
   every rung with formulas that rhyme: $\rho\sigma^2 + \frac{1-\rho}{B}\sigma^2$
   for bagging, $O(\eta\operatorname{tr}\Sigma/\mu|B|)$ for SGD.

**Cost of the part:** ~1800 lines including tests and the dataset generators,
30 tests, and a full run of all four rungs in about 21 seconds — small enough to
rerun on every change, which is the property that made the multi-seed check
cheap enough to adopt.

---

*Figures regenerate with `uv run python tools/writeup_figures.py A`; the numbers
they are quoted against live in [`data/a-metrics.json`](data/a-metrics.json).
The [part README](../../parts/a_classical/README.md) is the short version of
this document.*
