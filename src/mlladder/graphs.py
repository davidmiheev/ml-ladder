"""Graph data for part F: a real citation network and a controllable synthetic one.

Node classification is the setting where "use the graph" stops being obviously
right. A graph helps a classifier exactly insofar as **a node's neighbours carry
information about its label** — homophily — and message passing is a bet that
they do. On a citation network that bet pays; on a graph whose edges are
uninformative it is worse than ignoring the edges, because it mixes noise into
every representation.

So this module provides both: Cora, where the bet pays, and a stochastic block
model whose homophily is a dial. The dial is what turns "graph networks are
good" into a claim with a crossover point that can be located.

Downloads land in :func:`mlladder.paths.data_dir` and are cached.
"""

from __future__ import annotations

import io
import tarfile
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from numpy.typing import NDArray

from mlladder import paths
from mlladder.datasets import as_rng

CORA_URL = "https://linqs-data.soe.ucsc.edu/public/lbc/cora.tgz"


@dataclass
class Graph:
    """A node-classification problem.

    ``edges`` is an undirected edge list with both directions present and no
    self-loops; the adjacency is built from it on demand so the normalization
    is a property of the model, not baked into the data.
    """

    features: NDArray[np.float32]  # [n, d]
    labels: NDArray[np.int64]  # [n]
    edges: NDArray[np.int64]  # [2, m]
    name: str = "graph"

    @property
    def num_nodes(self) -> int:
        return len(self.labels)

    @property
    def num_classes(self) -> int:
        return int(self.labels.max()) + 1

    def homophily(self) -> float:
        """Fraction of edges joining nodes of the same class.

        The single number that predicts whether message passing will help.
        Its floor is not zero but roughly $1/C$ for $C$ balanced classes, since
        even random edges agree that often -- a detail worth keeping, because
        "homophily 0.3" sounds low until you notice there are seven classes.
        """
        if self.edges.shape[1] == 0:
            return float("nan")
        return float((self.labels[self.edges[0]] == self.labels[self.edges[1]]).mean())

    def degree(self) -> NDArray[np.int64]:
        counts = np.bincount(self.edges[0], minlength=self.num_nodes)
        return counts.astype(np.int64)


def _download(url: str, destination: Path) -> bytes:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        return destination.read_bytes()
    with urllib.request.urlopen(url, timeout=120) as response:
        payload = response.read()
    destination.write_bytes(payload)
    return payload


def load_cora() -> Graph:
    """The Cora citation network: 2708 papers, 1433 binary features, 7 classes.

    Parsed from the original LINQS archive rather than a framework's cached
    tensor, so the node ordering and the feature construction are visible here
    instead of inherited. Paper ids in ``cora.cites`` are arbitrary integers,
    so they are mapped through the order they appear in ``cora.content``.
    """
    payload = _download(CORA_URL, paths.data_dir() / "cora.tgz")
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        content = archive.extractfile("cora/cora.content").read().decode()  # type: ignore[union-attr]
        cites = archive.extractfile("cora/cora.cites").read().decode()  # type: ignore[union-attr]

    ids: list[str] = []
    rows: list[list[float]] = []
    names: list[str] = []
    for line in content.strip().splitlines():
        parts = line.split("\t")
        ids.append(parts[0])
        rows.append([float(v) for v in parts[1:-1]])
        names.append(parts[-1])
    index = {paper: i for i, paper in enumerate(ids)}
    classes = {name: i for i, name in enumerate(sorted(set(names)))}

    features = np.asarray(rows, dtype=np.float32)
    labels = np.asarray([classes[name] for name in names], dtype=np.int64)

    pairs: list[tuple[int, int]] = []
    for line in cites.strip().splitlines():
        cited, citing = line.split("\t")
        if cited in index and citing in index:
            a, b = index[cited], index[citing]
            if a != b:
                pairs.append((a, b))
                pairs.append((b, a))  # undirected
    edges = np.asarray(sorted(set(pairs)), dtype=np.int64).T
    return Graph(features=features, labels=labels, edges=edges, name="cora")


def make_sbm(
    *,
    num_nodes: int = 1200,
    num_classes: int = 4,
    feature_dim: int = 32,
    homophily: float = 0.8,
    average_degree: float = 8.0,
    feature_noise: float = 1.0,
    rng=None,
) -> Graph:
    r"""A stochastic block model with a homophily dial.

    Each node gets a class, a feature vector drawn around that class's centroid,
    and edges sampled so that a fraction ``homophily`` of them join same-class
    nodes. Writing $p_{\text{in}}$ and $p_{\text{out}}$ for the within- and
    between-class edge probabilities, and $C$ for the number of balanced
    classes, the expected same-class edge fraction is

    $$h = \frac{p_{\text{in}}}{p_{\text{in}} + (C - 1)\,p_{\text{out}}},$$

    so asking for $h$ at a fixed mean degree fixes both probabilities. Setting
    $h = 1/C$ gives an Erdos-Renyi graph whose edges carry nothing, which is
    the control the sweep needs at one end.

    ``feature_noise`` is the other half of the experiment: message passing has
    the most to offer when a node's *own* features are weak, so the honest
    comparison against a graph-blind model has to say how weak.
    """
    generator = as_rng(rng)
    labels = generator.integers(0, num_classes, size=num_nodes).astype(np.int64)

    centroids = generator.normal(0.0, 1.0, size=(num_classes, feature_dim))
    features = centroids[labels] + generator.normal(0.0, feature_noise, size=(num_nodes, feature_dim))

    # Solve for p_in / p_out at the requested homophily and mean degree.
    counts = np.bincount(labels, minlength=num_classes)
    same_pairs = float((counts * (counts - 1) / 2).sum())
    total_pairs = num_nodes * (num_nodes - 1) / 2
    cross_pairs = total_pairs - same_pairs
    target_edges = average_degree * num_nodes / 2
    same_edges = homophily * target_edges
    cross_edges = target_edges - same_edges
    p_in = min(same_edges / max(same_pairs, 1.0), 1.0)
    p_out = min(cross_edges / max(cross_pairs, 1.0), 1.0)

    # Sample per-pair Bernoulli without materializing n^2 for large n: draw a
    # Poisson number of candidate pairs per block instead.
    pairs: set[tuple[int, int]] = set()
    expected_same = int(p_in * same_pairs)
    expected_cross = int(p_out * cross_pairs)
    by_class = [np.flatnonzero(labels == c) for c in range(num_classes)]
    for _ in range(expected_same):
        c = generator.integers(0, num_classes)
        members = by_class[c]
        if len(members) < 2:
            continue
        a, b = generator.choice(members, size=2, replace=False)
        if a != b:
            pairs.add((int(min(a, b)), int(max(a, b))))
    for _ in range(expected_cross):
        # Draw the two *classes* first, then a member of each. The obvious
        # alternative -- draw two nodes and discard the pair when they happen to
        # share a class -- silently throws away a $1/C$ share of the cross
        # edges, which inflates the realized homophily: asking a four-class
        # graph for 0.25 delivered 0.31. The dial has to be calibrated, because
        # it is the x-axis of the sweep that uses it.
        c, d = generator.choice(num_classes, size=2, replace=False)
        if len(by_class[c]) == 0 or len(by_class[d]) == 0:
            continue
        a = int(generator.choice(by_class[c]))
        b = int(generator.choice(by_class[d]))
        pairs.add((min(a, b), max(a, b)))

    both = [(a, b) for a, b in pairs] + [(b, a) for a, b in pairs]
    edges = np.asarray(sorted(both), dtype=np.int64).T if both else np.zeros((2, 0), dtype=np.int64)
    return Graph(
        features=features.astype(np.float32),
        labels=labels,
        edges=edges,
        name=f"sbm(h={homophily:.2f})",
    )


def split_nodes(
    graph: Graph, *, train_per_class: int = 20, val_size: int = 500, test_size: int = 1000, rng=None
) -> tuple[NDArray[np.int64], NDArray[np.int64], NDArray[np.int64]]:
    """Planetoid-style split: a few labelled nodes per class, the rest held out.

    Twenty labels per class is the standard Cora protocol and it is the reason
    this benchmark is about *semi-supervised* learning: with 140 labels for 2708
    nodes, whatever the graph can propagate is most of what the model has.
    """
    generator = as_rng(rng)
    train: list[int] = []
    for c in range(graph.num_classes):
        members = np.flatnonzero(graph.labels == c)
        generator.shuffle(members)
        train.extend(members[:train_per_class].tolist())
    train_index = np.asarray(sorted(train), dtype=np.int64)
    remaining = np.setdiff1d(np.arange(graph.num_nodes, dtype=np.int64), train_index)
    generator.shuffle(remaining)
    val_index = np.sort(remaining[:val_size])
    test_index = np.sort(remaining[val_size : val_size + test_size])
    return train_index, val_index, test_index


def normalized_adjacency(graph: Graph, *, self_loops: bool = True):
    r"""$\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ as a dense tensor.

    The symmetric normalization Kipf and Welling derive from a first-order
    approximation to a spectral filter. The self-loop is not cosmetic: without
    it a node's own features never reach its representation, so the layer
    computes a function of the neighbourhood *excluding* the node, and a
    one-layer GCN cannot express the identity.

    Dense because these graphs are small; a sparse version is the obvious
    change for anything larger and would not alter a single number here.
    """
    from mlladder import vision

    torch = vision.require_torch()
    n = graph.num_nodes
    adjacency = torch.zeros((n, n), dtype=torch.float32)
    if graph.edges.shape[1]:
        rows = torch.from_numpy(graph.edges[0])
        cols = torch.from_numpy(graph.edges[1])
        adjacency[rows, cols] = 1.0
    if self_loops:
        adjacency += torch.eye(n)
    degree = adjacency.sum(dim=1).clamp_min(1.0)
    inverse_sqrt = degree.pow(-0.5)
    return inverse_sqrt.unsqueeze(1) * adjacency * inverse_sqrt.unsqueeze(0)
