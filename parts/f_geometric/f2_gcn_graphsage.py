"""F2. GCN and GraphSAGE on a node-classification benchmark (Cora/PubMed)

Lab reference: [escnn — equivariant CNN library, and PyTorch Geometric tutorials](https://github.com/QUVA-Lab/escnn)

F1 built equivariance to a group that acts on the *pixel grid*. A graph has no
grid, and its symmetry group is bigger and cruder: relabel the nodes however you
like and it is the same graph. A layer that respects that must treat a node's
neighbours as a **set**, which is why every graph network in use is a permutation-
equivariant aggregate — sum, mean, max — followed by a shared transform. The
architecture is not a design choice so much as the shape that survives the
constraint.

What the constraint does *not* supply is a reason to believe the neighbours are
worth aggregating. Message passing is a bet that a node's neighbours carry
information about its label, and the bet has a name: homophily. This rung makes
that the measured quantity rather than an assumption.

Three things get measured:

1. **Cora**, where the bet pays, against a graph-blind MLP on identical features
   and against label propagation — a control with *no features and no
   parameters* that asks how much of the benchmark is the graph rather than the
   learning.
2. **Depth**, because stacking these layers does not behave like stacking
   convolutions. Repeated averaging is a diffusion, and diffusions converge.
3. **Homophily**, swept on a synthetic graph whose edges can be dialled from
   informative to worthless, to locate the point where using the graph stops
   helping — and to see whether GCN and GraphSAGE reach it together.

PubMed is the other benchmark the spec names; it is skipped here because its
19,717 nodes make the dense adjacency this module uses a 1.5 GB tensor. So Cora
carries the real-data half and the SBM carries the controlled half.

The adjacency is dense on purpose, and not only for simplicity. Cora's operator
has 13,264 non-zeros in a 7,333,264-entry matrix — 0.18% — which sounds like an
obvious case for `torch.sparse`, and measured on this CPU the sparse multiply
takes **46.8 ms against the dense 31.1 ms**. At 2,708 nodes the dense path is a
single BLAS call on a matrix that fits in cache, and beating it needs a bigger
graph than this. The sparsity argument is real; it just does not start here.

Run with ``ladder run F2`` (``--help`` for flags).
"""

from __future__ import annotations

import numpy as np

from mlladder import Experiment, experiment, graphs, plotting, torchkit, vision


def mean_adjacency(graph: graphs.Graph):
    r"""Row-normalized neighbour mean $D^{-1}A$, **without** a self-loop.

    This is GraphSAGE's aggregator, and it differs from the GCN's
    $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ in two ways that both matter at
    low homophily. The normalization is a plain mean instead of a
    degree-symmetric one, and — the load-bearing difference — the node itself is
    *excluded*, so its own features reach the next layer through a separate
    weight matrix rather than being averaged in with its neighbours'.

    A GCN layer cannot tell a node's own features from a neighbour's: they enter
    the same sum through the same $W$. GraphSAGE can. That is the entire extra
    freedom, and the homophily sweep below is where it is supposed to show up,
    because "ignore my neighbours" is a function GraphSAGE can represent and a
    GCN cannot.
    """
    torch = vision.require_torch()
    n = graph.num_nodes
    adjacency = torch.zeros((n, n), dtype=torch.float32)
    if graph.edges.shape[1]:
        adjacency[torch.from_numpy(graph.edges[0]), torch.from_numpy(graph.edges[1])] = 1.0
    degree = adjacency.sum(dim=1, keepdim=True).clamp_min(1.0)
    return adjacency / degree


def make_mlp(in_features: int, hidden: int, num_classes: int, *, layers: int = 2, dropout: float = 0.5):
    """The graph-blind control: the same features, the same budget, no edges.

    Without this arm, "the GCN gets 0.80 on Cora" is unreadable, because Cora's
    bag-of-words features are informative on their own. The MLP is what the
    graph has to beat.
    """
    torch = vision.require_torch()

    class MLP(torch.nn.Module):
        def __init__(self):
            super().__init__()
            widths = [in_features] + [hidden] * (layers - 1) + [num_classes]
            self.linears = torch.nn.ModuleList(
                [torch.nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)]
            )
            self.dropout = torch.nn.Dropout(dropout)

        def embed(self, x, adjacency):  # the adjacency is accepted and ignored: that is the point
            h = x
            for linear in self.linears[:-1]:
                h = torch.relu(linear(self.dropout(h)))
            return h

        def forward(self, x, adjacency):
            return self.linears[-1](self.dropout(self.embed(x, adjacency)))

    return MLP


def make_gcn(in_features: int, hidden: int, num_classes: int, *, layers: int = 2, dropout: float = 0.5):
    r"""Kipf & Welling's graph convolution.

    Each layer is

    $$H^{(\ell+1)} = \sigma\!\left(\hat{A}\,H^{(\ell)}W^{(\ell)}\right),
    \qquad \hat{A} = \tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2},$$

    with $\tilde{A} = A + I$. Two facts about $\hat A$ explain most of this
    rung's results. It is permutation-equivariant, which is why the layer is
    well defined on a graph at all; and it is a *averaging* operator with
    spectral radius $1$ whose top eigenvector is $\tilde{D}^{1/2}\mathbf{1}$,
    which is why $\hat{A}^k$ converges and deep stacks collapse.
    """
    torch = vision.require_torch()

    class GCN(torch.nn.Module):
        def __init__(self):
            super().__init__()
            widths = [in_features] + [hidden] * (layers - 1) + [num_classes]
            self.linears = torch.nn.ModuleList(
                [torch.nn.Linear(widths[i], widths[i + 1], bias=True) for i in range(len(widths) - 1)]
            )
            self.dropout = torch.nn.Dropout(dropout)

        def embed(self, x, adjacency):
            h = x
            for linear in self.linears[:-1]:
                h = torch.relu(adjacency @ linear(self.dropout(h)))
            return h

        def forward(self, x, adjacency):
            return adjacency @ self.linears[-1](self.dropout(self.embed(x, adjacency)))

    return GCN


def make_graphsage(in_features: int, hidden: int, num_classes: int, *, layers: int = 2, dropout: float = 0.5):
    r"""GraphSAGE with the mean aggregator.

    $$H^{(\ell+1)} = \sigma\!\left(H^{(\ell)}W^{(\ell)}_{\text{self}}
    + \bar{A}H^{(\ell)}W^{(\ell)}_{\text{neigh}}\right),
    \qquad \bar{A} = D^{-1}A .$$

    Twice the parameters per layer, spent entirely on being able to weigh "me"
    against "my neighbourhood". Hamilton et al. introduced this for *inductive*
    use — sampling neighbourhoods so the model generalizes to nodes unseen at
    training time — but the sampling is an optimization for large graphs, not
    the modelling content. On a 2,708-node graph the full neighbourhood fits,
    so what is compared here is the architecture: the separate self-weight.
    """
    torch = vision.require_torch()

    class GraphSAGE(torch.nn.Module):
        def __init__(self):
            super().__init__()
            widths = [in_features] + [hidden] * (layers - 1) + [num_classes]
            self.self_linears = torch.nn.ModuleList(
                [torch.nn.Linear(widths[i], widths[i + 1]) for i in range(len(widths) - 1)]
            )
            self.neighbour_linears = torch.nn.ModuleList(
                [torch.nn.Linear(widths[i], widths[i + 1], bias=False) for i in range(len(widths) - 1)]
            )
            self.dropout = torch.nn.Dropout(dropout)

        def _layer(self, index, h, adjacency):
            dropped = self.dropout(h)
            return self.self_linears[index](dropped) + self.neighbour_linears[index](adjacency @ dropped)

        def embed(self, x, adjacency):
            h = x
            for index in range(len(self.self_linears) - 1):
                h = torch.relu(self._layer(index, h, adjacency))
            return h

        def forward(self, x, adjacency):
            return self._layer(len(self.self_linears) - 1, self.embed(x, adjacency), adjacency)

    return GraphSAGE


ARCHITECTURES = {"mlp": make_mlp, "gcn": make_gcn, "sage": make_graphsage}
#: Which normalized operator each arm is fed. The MLP gets one too and ignores
#: it, so the training loop has no per-arm branches to get wrong.
AGGREGATORS = {"mlp": "gcn", "gcn": "gcn", "sage": "sage"}


def label_propagation(graph: graphs.Graph, train_index, *, steps: int = 50, alpha: float = 0.9):
    r"""The control with no features and no parameters.

    Put a one-hot on each labelled node, average repeatedly over neighbours with
    a restart, $Y^{(t+1)} = \alpha \hat{A} Y^{(t)} + (1-\alpha)Y^{(0)}$, and read
    off the argmax. It is the graph's own answer to the task, and on a homophilous
    citation network it is a startlingly strong one.

    Its purpose here is to bound the claim: whatever a GCN scores, the honest
    question is how much of that is the graph structure that this needs no
    training at all to exploit.
    """
    torch = vision.require_torch()
    adjacency = graphs.normalized_adjacency(graph)
    seed = torch.zeros((graph.num_nodes, graph.num_classes), dtype=torch.float32)
    seed[torch.from_numpy(train_index), torch.from_numpy(graph.labels[train_index])] = 1.0
    scores = seed.clone()
    for _ in range(steps):
        scores = alpha * (adjacency @ scores) + (1 - alpha) * seed
    return scores.argmax(dim=1).numpy()


def smoothness(embeddings) -> float:
    r"""Mean pairwise cosine similarity of the node representations.

    The direct measurement of oversmoothing: $\hat{A}^k$ contracts toward its
    leading eigenvector, so as depth grows every node's representation points
    the same way and this number climbs to $1$. Accuracy falling with depth is
    consistent with several stories — vanishing gradients, overfitting, bad
    initialization — and this one distinguishes them, because only the diffusion
    story predicts that the *representations themselves* collapse together.

    **It has a floor, and the floor is not zero.** These embeddings come out of
    a ReLU, so they live in the non-negative orthant where every pairwise cosine
    is $\ge 0$ and a random pair already scores well above it. The absolute
    number is therefore uninterpretable on its own: measured on Cora, the
    *graph-blind* MLP scored 0.71 against the GCN's 0.63, which read naively
    would say the model with no edges oversmoothed harder. It did not; it has
    the same nonlinearity. That is why the depth sweep runs an MLP arm at every
    depth -- the quantity that means something is the **difference** between a
    graph model and a graph-blind one of identical shape.
    """
    torch = vision.require_torch()
    with torch.no_grad():
        normalized = torch.nn.functional.normalize(embeddings, dim=1)
        gram = normalized @ normalized.T
        n = gram.shape[0]
        off_diagonal = (gram.sum() - gram.diagonal().sum()) / (n * (n - 1))
    return float(off_diagonal)


def train_node_classifier(
    model, features, labels, adjacency, split, *, exp: Experiment, epochs: int, lr: float, weight_decay: float
) -> tuple[list[float], list[float], float, int]:
    """Full-batch training with model selection on the validation split.

    Full batch because the graph does not decompose: a node's representation
    depends on its neighbours', so a minibatch of nodes still needs their
    neighbourhoods. With 140 labels and 2,708 nodes, early stopping on the
    500-node validation split is doing real work — these models overfit the
    training nodes within a few dozen epochs and keep improving on the rest.
    """
    torch = vision.require_torch()
    train_index, val_index, _ = split
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses: list[float] = []
    validations: list[float] = []
    best_validation, best_state, best_epoch = -1.0, None, 0
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        logits = model(features, adjacency)
        loss = torch.nn.functional.cross_entropy(logits[train_index], labels[train_index])
        loss.backward()
        optimizer.step()
        losses.append(float(loss.item()))

        model.eval()
        with torch.no_grad():
            validation = float(
                (model(features, adjacency)[val_index].argmax(dim=1) == labels[val_index]).float().mean()
            )
        validations.append(validation)
        if validation > best_validation:
            best_validation = validation
            best_epoch = epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is not None:
        model.load_state_dict(best_state)
    # ``best_epoch`` is reported so the budget can be checked rather than
    # assumed. A deep stack trains *slower* as well as worse, and a fixed epoch
    # budget that binds it would manufacture the oversmoothing result out of
    # undertraining: measured at 200 epochs, a two-layer GCN peaked at epoch 63
    # and an eight-layer one at epoch 186 -- still climbing when the budget ran
    # out. Hence ``depth_epochs``.
    return losses, validations, best_validation, best_epoch


def test_accuracy(model, features, labels, adjacency, test_index) -> float:
    torch = vision.require_torch()
    model.eval()
    with torch.no_grad():
        return float((model(features, adjacency)[test_index].argmax(dim=1) == labels[test_index]).float().mean())


def prepare(graph: graphs.Graph, *, device):
    """Tensors and both normalized operators for one graph, built once."""
    torch = vision.require_torch()
    features = torch.from_numpy(graph.features).to(device)
    labels = torch.from_numpy(graph.labels).to(device)
    operators = {
        "gcn": graphs.normalized_adjacency(graph).to(device),
        "sage": mean_adjacency(graph).to(device),
    }
    return features, labels, operators


def fit_arm(
    arm: str,
    graph: graphs.Graph,
    prepared,
    split,
    *,
    exp: Experiment,
    layers: int,
    hidden: int,
    device,
    epochs: int | None = None,
) -> tuple[float, float, list[float], float, int]:
    """Seed, build, train and score one architecture.

    Returns test accuracy, best validation accuracy, the loss curve, the
    representation smoothness, and the epoch the selected model came from.
    """
    torchkit.seed_torch(exp.seed)
    features, labels, operators = prepared
    model = ARCHITECTURES[arm](
        graph.features.shape[1], hidden, graph.num_classes, layers=layers, dropout=exp["dropout"]
    )().to(device)
    adjacency = operators[AGGREGATORS[arm]]
    losses, _, best_validation, best_epoch = train_node_classifier(
        model,
        features,
        labels,
        adjacency,
        split,
        exp=exp,
        epochs=exp["epochs"] if epochs is None else epochs,
        lr=exp["lr"],
        weight_decay=exp["weight_decay"],
    )
    accuracy = test_accuracy(model, features, labels, adjacency, split[2])
    torch = vision.require_torch()
    model.eval()
    with torch.no_grad():
        collapse = smoothness(model.embed(features, adjacency))
    return accuracy, best_validation, losses, collapse, best_epoch


def predicted_crossover(num_classes: int, average_degree: float) -> float:
    r"""Where the neighbour aggregate should stop being worth more than the node.

    Derived rather than fitted, so the sweep has something to disagree with.
    Take balanced classes with centroids $\mu_c$ summing to zero and features
    $x_i = \mu_{c_i} + \varepsilon_i$. The mean over $d$ neighbours of a node in
    class $c$ has expectation

    $$h\,\mu_c + (1-h)\,\frac{-\mu_c}{C-1} = \frac{hC-1}{C-1}\,\mu_c,$$

    so the class signal is scaled by $s(h) = (hC-1)/(C-1)$ — exactly zero at
    $h = 1/C$ and **negative** below it, where the neighbourhood carries the
    signal *inverted*. Averaging $d$ independent noise terms meanwhile shrinks
    the noise by $\sqrt{d}$, so the aggregate's signal-to-noise relative to the
    node's own features is $\sqrt{d}\,|s(h)|$. Setting that to 1:

    $$h^{*} = \frac{1}{C}\left(1 + \frac{C-1}{\sqrt{d}}\right).$$

    This is the crossover for a model using *only* the aggregate. Both graph
    arms also see the node itself — the GCN through its self-loop, GraphSAGE
    through a separate weight — so both should cross lower than $h^{*}$, and
    GraphSAGE lower still. The sweep measures whether they do.
    """
    return float((1.0 + (num_classes - 1) / np.sqrt(average_degree)) / num_classes)


def crossover(homophilies: list[float], gaps: list[float]) -> float:
    """Where the graph stops paying: the homophily at which a gap curve crosses zero.

    Linear interpolation between the bracketing sweep points, reported as NaN
    when the curve never changes sign — because "the crossover is off the grid"
    is a different statement from "the crossover is at the last point I tried".
    """
    for i in range(len(gaps) - 1):
        if gaps[i] <= 0.0 <= gaps[i + 1] or gaps[i] >= 0.0 >= gaps[i + 1]:
            span = gaps[i + 1] - gaps[i]
            if abs(span) < 1e-12:
                return float(homophilies[i])
            t = -gaps[i] / span
            return float(homophilies[i] + t * (homophilies[i + 1] - homophilies[i]))
    return float("nan")


@experiment(
    "f2_gcn_graphsage",
    defaults={
        # Kipf & Welling's width, kept so the Cora numbers sit beside the
        # published ones rather than beside a wider variant of them.
        "hidden": 16,
        "layers": 2,
        "epochs": 200,
        # The depth sweep gets its own, larger budget: see train_node_classifier.
        # 300 was not enough -- three to four of the twelve arms were still
        # improving when it ran out, so the sweep was measuring the budget.
        "depth_epochs": 600,
        "lr": 1e-2,
        "weight_decay": 5e-4,
        "dropout": 0.5,
        "train_per_class": 20,
        "arms": ["mlp", "gcn", "sage"],
        # Oversmoothing: accuracy and representation collapse against depth.
        "depths": [1, 2, 3, 4, 6, 8],
        # The homophily dial. 0.25 is 1/C for the four-class SBM -- an
        # Erdos-Renyi graph whose edges carry nothing.
        "homophilies": [0.05, 0.15, 0.25, 0.4, 0.55, 0.7, 0.85],
        "sbm_nodes": 1200,
        "sbm_classes": 4,
        "sbm_degree": 8.0,
        # Weak node features, so the sweep is about the edges. With strong
        # features every arm saturates and the crossover is invisible.
        "sbm_feature_noise": 2.5,
    },
    description="GCN and GraphSAGE against a graph-blind MLP, swept over depth and over homophily.",
)
def run(exp: Experiment) -> dict[str, float]:
    """GCN and GraphSAGE on a node-classification benchmark (Cora/PubMed)"""
    torch = vision.require_torch()
    device = torchkit.device()
    rng = np.random.default_rng(exp.seed)
    print(f"[f2] device: {device}")

    summary: dict[str, float] = {}

    # -- Cora ----------------------------------------------------------------
    cora = graphs.load_cora()
    summary["cora/nodes"] = float(cora.num_nodes)
    summary["cora/edges"] = float(cora.edges.shape[1] // 2)
    summary["cora/homophily"] = cora.homophily()
    summary["cora/mean_degree"] = float(cora.degree().mean())
    print(
        f"[f2] cora: {cora.num_nodes} nodes, {cora.edges.shape[1] // 2} undirected edges, "
        f"{cora.num_classes} classes, homophily {summary['cora/homophily']:.3f} "
        f"(random floor ~{1 / cora.num_classes:.3f})"
    )

    split = graphs.split_nodes(cora, train_per_class=exp["train_per_class"], rng=rng)
    summary["cora/labels_used"] = float(len(split[0]))
    prepared = prepare(cora, device=device)
    index = tuple(torch.from_numpy(part).to(device) for part in split)

    curves: dict[str, list[float]] = {}
    cora_scores: dict[str, float] = {}
    for arm in exp["arms"]:
        accuracy, validation, losses, collapse, best_epoch = fit_arm(
            arm, cora, prepared, index, exp=exp, layers=exp["layers"], hidden=exp["hidden"], device=device
        )
        curves[arm] = losses
        cora_scores[arm] = accuracy
        summary[f"cora/{arm}/test_accuracy"] = accuracy
        summary[f"cora/{arm}/val_accuracy"] = validation
        summary[f"cora/{arm}/smoothness"] = collapse
        summary[f"cora/{arm}/best_epoch"] = float(best_epoch)
        print(f"[f2] cora {arm:<5} test {accuracy:.3f}  val {validation:.3f}")

    # The control that needs no features and no training at all.
    predicted = label_propagation(cora, split[0])
    propagation = float((predicted[split[2]] == cora.labels[split[2]]).mean())
    cora_scores["label_prop"] = propagation
    summary["cora/label_propagation/test_accuracy"] = propagation
    summary["cora/graph_gain"] = cora_scores["gcn"] - cora_scores["mlp"]
    # The same derivation applied to Cora. Its assumptions (balanced Gaussian
    # classes, independent noise) are wrong for a sparse bag of words, so this
    # is a sanity check rather than a prediction: it should put Cora's 0.81
    # comfortably on the paying side, and if it did not, the model behind the
    # SBM sweep's prediction would be the thing to distrust.
    summary["cora/predicted_crossover"] = predicted_crossover(cora.num_classes, float(cora.degree().mean()))
    summary["cora/learning_gain_over_propagation"] = cora_scores["gcn"] - propagation
    print(
        f"[f2] cora label-propagation (no features, no parameters) {propagation:.3f}  "
        f"-- the GCN's {cora_scores['gcn']:.3f} is only {summary['cora/learning_gain_over_propagation']:+.3f} beyond it"
    )

    # -- depth: oversmoothing -------------------------------------------------
    # The MLP runs at every depth too. Its representations pass through the same
    # ReLUs and the same widths and touch no edges, so it is the control that
    # turns an uninterpretable cosine similarity into a statement about the
    # graph: see ``smoothness``.
    depth_accuracy: dict[str, list[float]] = {arm: [] for arm in ("mlp", "gcn", "sage")}
    depth_smoothness: dict[str, list[float]] = {arm: [] for arm in ("mlp", "gcn", "sage")}
    budget_bound = 0
    hidden_start = next((i for i, d in enumerate(exp["depths"]) if d >= 2), 0)
    for arm in depth_accuracy:
        for depth in exp["depths"]:
            accuracy, _, _, collapse, best_epoch = fit_arm(
                arm,
                cora,
                prepared,
                index,
                exp=exp,
                layers=depth,
                hidden=exp["hidden"],
                device=device,
                epochs=exp["depth_epochs"],
            )
            depth_accuracy[arm].append(accuracy)
            depth_smoothness[arm].append(collapse)
            summary[f"depth/{arm}/{depth}/test_accuracy"] = accuracy
            summary[f"depth/{arm}/{depth}/smoothness"] = collapse
            summary[f"depth/{arm}/{depth}/best_epoch"] = float(best_epoch)
            # A model still improving when the budget ended is undertrained, and
            # its accuracy is not evidence about depth.
            budget_bound += int(best_epoch > 0.9 * exp["depth_epochs"])
        best = int(np.argmax(depth_accuracy[arm]))
        summary[f"depth/{arm}/best_depth"] = float(exp["depths"][best])
        summary[f"depth/{arm}/deepest_drop"] = depth_accuracy[arm][best] - depth_accuracy[arm][-1]
        # A "one-layer" model is a single linear map, so ``embed`` hands back the
        # raw features and its smoothness describes Cora's bag of words, not any
        # representation. The rise is therefore measured from the shallowest
        # depth that actually has a hidden layer.
        summary[f"depth/{arm}/smoothness_rise"] = depth_smoothness[arm][-1] - depth_smoothness[arm][hidden_start]
        print(
            f"[f2] depth {arm:<5} best {exp['depths'][best]} layers ({depth_accuracy[arm][best]:.3f}); "
            f"at {exp['depths'][-1]} layers {depth_accuracy[arm][-1]:.3f}, "
            f"cosine similarity {depth_smoothness[arm][hidden_start]:.3f} -> {depth_smoothness[arm][-1]:.3f}"
        )

    # What the graph adds to the collapse, at each depth, over a graph-blind
    # model of identical shape. This -- not the raw cosine similarity -- is the
    # oversmoothing signal.
    graph_excess = {
        arm: [depth_smoothness[arm][i] - depth_smoothness["mlp"][i] for i in range(len(exp["depths"]))]
        for arm in ("gcn", "sage")
    }
    for arm, excess in graph_excess.items():
        for depth, value in zip(exp["depths"], excess, strict=True):
            summary[f"depth/{arm}/{depth}/smoothness_over_mlp"] = value
        summary[f"depth/{arm}/excess_rise"] = excess[-1] - excess[hidden_start]
        print(
            f"[f2] depth {arm:<5} collapse beyond the graph-blind MLP: "
            f"{excess[hidden_start]:+.3f} at {exp['depths'][hidden_start]} layers -> "
            f"{excess[-1]:+.3f} at {exp['depths'][-1]}"
        )
    if exp["depths"][0] == 1:
        # Recorded once under its true name, so nobody reads it as a model's.
        summary["depth/input_features_smoothness"] = depth_smoothness["gcn"][0]

    summary["depth/arms_bound_by_budget"] = float(budget_bound)
    if budget_bound:
        print(
            f"[f2] WARNING: {budget_bound} depth arm(s) were still improving at the end of "
            f"{exp['depth_epochs']} epochs -- their accuracies measure the budget, not the depth"
        )

    # -- homophily: where the graph stops paying ------------------------------
    sweep: dict[str, list[float]] = {arm: [] for arm in exp["arms"]}
    measured_homophily: list[float] = []
    for target in exp["homophilies"]:
        graph = graphs.make_sbm(
            num_nodes=exp["sbm_nodes"],
            num_classes=exp["sbm_classes"],
            homophily=target,
            average_degree=exp["sbm_degree"],
            feature_noise=exp["sbm_feature_noise"],
            rng=np.random.default_rng(exp.seed + int(target * 1000)),
        )
        measured_homophily.append(graph.homophily())
        summary[f"homophily/{target}/measured"] = graph.homophily()
        sbm_split = graphs.split_nodes(
            graph, train_per_class=exp["train_per_class"], val_size=200, test_size=500, rng=rng
        )
        sbm_prepared = prepare(graph, device=device)
        sbm_index = tuple(torch.from_numpy(part).to(device) for part in sbm_split)
        for arm in exp["arms"]:
            accuracy, _, _, _, _ = fit_arm(
                arm, graph, sbm_prepared, sbm_index, exp=exp, layers=exp["layers"], hidden=exp["hidden"], device=device
            )
            sweep[arm].append(accuracy)
            summary[f"homophily/{target}/{arm}/test_accuracy"] = accuracy
        print(
            f"[f2] sbm h={target:.2f} (measured {graph.homophily():.3f})  "
            + "  ".join(f"{arm} {sweep[arm][-1]:.3f}" for arm in exp["arms"])
        )

    gaps = {arm: [sweep[arm][i] - sweep["mlp"][i] for i in range(len(exp["homophilies"]))] for arm in ("gcn", "sage")}
    for arm, gap in gaps.items():
        # Interpolated over the *measured* homophily of each sampled graph, not
        # the value that was asked for. The dial is calibrated to ~0.001 so the
        # two barely differ now -- but the crossover is the headline number of
        # this sweep, and it should be a statement about the graphs that were
        # actually trained on.
        summary[f"homophily/{arm}/crossover"] = crossover(measured_homophily, gap)
        summary[f"homophily/{arm}/gap_at_lowest"] = gap[0]
        summary[f"homophily/{arm}/gap_at_highest"] = gap[-1]
        print(
            f"[f2] {arm} beats the graph-blind MLP above homophily "
            f"{summary[f'homophily/{arm}/crossover']:.3f}; at h={measured_homophily[0]:.2f} it is {gap[0]:+.3f}"
        )
    summary["homophily/sage_over_gcn_at_lowest"] = sweep["sage"][0] - sweep["gcn"][0]
    summary["homophily/predicted_crossover"] = predicted_crossover(exp["sbm_classes"], exp["sbm_degree"])
    summary["homophily/random_floor"] = 1.0 / exp["sbm_classes"]
    print(
        f"[f2] the neighbour aggregate alone should stop paying at h="
        f"{summary['homophily/predicted_crossover']:.3f} (random floor {summary['homophily/random_floor']:.2f}); "
        f"both arms also see the node itself, so both should cross lower"
    )

    # -- figures --------------------------------------------------------------
    exp.log_figure(
        "architectures",
        plotting.plot_architecture(
            {
                "MLP (graph-blind)": [
                    ("features X", f"{cora.features.shape[1]}-d bag of words"),
                    ("dropout"),
                    ("XW", "no edges used"),
                    ("ReLU"),
                    ("XW -> logits", f"{cora.num_classes} classes"),
                ],
                "GCN": [
                    ("features X", "same features"),
                    ("dropout"),
                    ("A-hat (XW)", "sym. norm, self-loop in the sum"),
                    ("ReLU"),
                    ("A-hat (HW) -> logits", "one W for self and neighbours"),
                ],
                "GraphSAGE (mean)": [
                    ("features X", "same features"),
                    ("dropout"),
                    ("XWs + A-bar XWn", "separate self weight"),
                    ("ReLU"),
                    ("HWs + A-bar HWn -> logits", "can learn to ignore edges"),
                ],
            },
            highlight={"GCN": [2, 4], "GraphSAGE (mean)": [2, 4]},
            title="Two layers of each arm: the only difference is how a node's own features enter",
        ),
    )
    exp.log_figure(
        "cora",
        plotting.plot_comparison(cora_scores, title="Cora test accuracy (20 labels per class)", ylabel="accuracy"),
    )
    exp.log_figure(
        "depth_accuracy",
        plotting.plot_curves(
            depth_accuracy,
            x=exp["depths"],
            title="Cora accuracy against depth -- deeper is worse, and the MLP is not",
            xlabel="message-passing layers",
            ylabel="test accuracy",
            marker="o",
        ),
    )
    exp.log_figure(
        "depth_smoothness",
        plotting.plot_curves(
            depth_smoothness,
            x=exp["depths"],
            title="Mean pairwise cosine similarity -- note the graph-blind MLP's floor",
            xlabel="layers",
            ylabel="cosine similarity",
            marker="o",
        ),
    )
    exp.log_figure(
        "depth_smoothness_excess",
        plotting.plot_curves(
            graph_excess,
            x=exp["depths"],
            title="Collapse the graph is responsible for (same shape, minus the MLP)",
            xlabel="message-passing layers",
            ylabel="cosine similarity over the MLP",
            marker="o",
        ),
    )
    exp.log_figure(
        "homophily",
        plotting.plot_curves(
            sweep,
            x=measured_homophily,
            title="SBM accuracy against measured homophily (0.25 is the four-class random floor)",
            xlabel="homophily",
            ylabel="test accuracy",
            marker="o",
        ),
    )
    exp.log_figure(
        "homophily_gap",
        plotting.plot_curves(
            gaps,
            x=measured_homophily,
            title="Accuracy gained over the graph-blind MLP -- negative means the edges hurt",
            xlabel="homophily",
            ylabel="accuracy difference",
            marker="o",
        ),
    )
    exp.log_figure(
        "training", plotting.plot_curves(curves, title="Cora training loss", xlabel="epoch", ylabel="cross-entropy")
    )
    return summary


main = run

if __name__ == "__main__":
    raise SystemExit(main())
