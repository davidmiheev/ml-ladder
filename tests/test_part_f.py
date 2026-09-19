"""Part F: the equivariance claims, checked rather than asserted in prose."""

from __future__ import annotations

import numpy as np
import pytest

from mlladder import vision

torch = pytest.importorskip("torch")


def _probe(n=4, size=16, seed=0):
    """Structured, *asymmetric* probe images.

    Not iid noise: uniform noise is statistically rotation-symmetric, so any
    network ending in a global average pool scores near-zero invariance error
    on it whether or not it is invariant. An untrained plain CNN measures
    0.005 on noise and 0.5 on this -- the probe, not the network, was doing the
    talking. The rung itself probes with real digits, which have the same
    asymmetry for the same reason.
    """
    rng = np.random.default_rng(seed)
    images = rng.random((n, 1, size, size)).astype(np.float32) * 0.1
    # An off-centre bright quadrant plus a bar: nothing here survives a
    # quarter turn.
    images[:, :, : size // 2, : size // 3] += 1.0
    images[:, :, size // 3 : size // 3 + 2, :] += 0.6
    return torch.from_numpy(images)


def test_rotate_batch_is_the_identity_at_zero():
    from parts.f_geometric.f1_equivariant_cnn import rotate_batch

    x = _probe()
    assert torch.allclose(rotate_batch(x, 0.0), x)
    assert torch.allclose(rotate_batch(x, 360.0), x)


def test_rotate_batch_four_quarter_turns_return_home():
    """Resampling is lossy, so this is a tolerance, not an identity.

    Recording how lossy is the point: the invariance numbers this rung reports
    cannot be tighter than the rotation operator used to measure them.
    """
    from parts.f_geometric.f1_equivariant_cnn import rotate_batch

    x = _probe()
    y = x
    for _ in range(4):
        y = rotate_batch(y, 90.0)
    assert (y - x).norm() / x.norm() < 0.05


def test_lifting_layer_is_equivariant_to_quarter_turns():
    r"""Rotating the input must permute the orientation axis and rotate space.

    This is the defining property of the lifting layer, and it holds for a
    *random* filter -- no training involved, which is what "by construction"
    means.
    """
    from parts.f_geometric.f1_equivariant_cnn import make_lifting_conv

    torch.manual_seed(0)
    layer = make_lifting_conv(1, 3, 3)()
    x = _probe(n=2, size=15)  # odd size: the rotation centre is a pixel
    with torch.no_grad():
        base = layer(x)
        rotated = layer(torch.rot90(x, 1, dims=(2, 3)))
    # Rotating the input by one quarter turn rotates each response spatially
    # and shifts the orientation axis by one.
    expected = torch.roll(torch.rot90(base, 1, dims=(3, 4)), shifts=1, dims=2)
    assert (rotated - expected).norm() / expected.norm() < 1e-5


def test_group_conv_is_equivariant_to_quarter_turns():
    from parts.f_geometric.f1_equivariant_cnn import make_group_conv

    torch.manual_seed(0)
    layer = make_group_conv(2, 3, 3)()
    rng = np.random.default_rng(1)
    x = torch.from_numpy(rng.random((2, 2, 4, 15, 15)).astype(np.float32))
    acted = torch.roll(torch.rot90(x, 1, dims=(3, 4)), shifts=1, dims=2)
    with torch.no_grad():
        base, rotated = layer(x), layer(acted)
    expected = torch.roll(torch.rot90(base, 1, dims=(3, 4)), shifts=1, dims=2)
    assert (rotated - expected).norm() / expected.norm() < 1e-5


def test_the_orientation_roll_is_load_bearing():
    """Rotating the filter spatially but not cycling its group axis fails.

    Group convolution on $p4$ needs both halves of $g^{-1}h = (r^{-1}s,\\;
    r^{-1}(y-x))$: the spatial ``rot90`` *and* the cyclic shift of the
    orientation axis. The version without the roll still runs, still trains and
    still looks like a group convolution, which is exactly why this is pinned
    rather than argued.
    """
    from parts.f_geometric.f1_equivariant_cnn import ROTATIONS

    torch.manual_seed(0)
    weight = torch.randn(3, 2, ROTATIONS, 3, 3)
    rng = np.random.default_rng(2)
    x = torch.from_numpy(rng.random((2, 2, ROTATIONS, 15, 15)).astype(np.float32))

    def act(tensor):
        """The $p4$ action on a group-valued feature map: turn it and shift its fibre."""
        return torch.roll(torch.rot90(tensor, 1, dims=(3, 4)), shifts=1, dims=2)

    def measure(use_roll: bool) -> float:
        with torch.no_grad():
            filters = []
            for r in range(ROTATIONS):
                rotated = torch.rot90(weight, r, dims=(3, 4))
                filters.append(torch.roll(rotated, shifts=r, dims=2) if use_roll else rotated)
            stacked = torch.cat(filters, dim=0).reshape(3 * ROTATIONS, 2 * ROTATIONS, 3, 3)

            def forward(inp):
                out = torch.nn.functional.conv2d(inp.reshape(2, 2 * ROTATIONS, 15, 15), stacked, padding=1)
                return out.view(2, ROTATIONS, 3, 15, 15).permute(0, 2, 1, 3, 4)

            expected = act(forward(x))
            return float((forward(act(x)) - expected).norm() / expected.norm())

    assert measure(True) < 1e-5
    assert measure(False) > 0.1


def test_p4_network_is_invariant_on_the_group_and_not_off_it():
    r"""The rung's headline property, on an untrained network.

    Untrained is deliberate: equivariance is a property of the architecture, so
    it must hold before any data is seen. The off-group angle is the control --
    a number that is zero everywhere would mean the measurement is broken, not
    that the network is miraculous.
    """
    from parts.f_geometric.f1_equivariant_cnn import invariance_error, make_p4_cnn

    torch.manual_seed(0)
    model = make_p4_cnn(width=4)()
    x = _probe(n=8, size=28)
    on_group = invariance_error(model, x, 90.0)
    off_group = invariance_error(model, x, 45.0)
    assert on_group < 0.05
    assert off_group > on_group


def test_global_pooling_flatters_the_plain_cnn():
    """Why this rung measures feature maps, not the pooled vector.

    A global average over spatial positions is blind to a permutation of those
    positions, so a plain CNN ending in one looks nearly invariant. The same
    network measured on its feature maps -- where the claim actually lives --
    is an order of magnitude worse. If the pooled number were the headline, the
    rung would be reporting a property of ``mean``.
    """
    from parts.f_geometric.f1_equivariant_cnn import equivariance_error, invariance_error, make_plain_cnn

    torch.manual_seed(0)
    model = make_plain_cnn(width=4)()
    x = _probe(n=8, size=28)
    pooled = invariance_error(model, x, 90.0)
    maps = equivariance_error(model, x, 90.0)
    assert pooled < 0.1, "the pooled vector barely moves -- that is the confound"
    assert maps > 5 * pooled, (pooled, maps)


def test_p4_beats_the_plain_cnn_on_feature_map_equivariance():
    from parts.f_geometric.f1_equivariant_cnn import equivariance_error, make_p4_cnn, make_plain_cnn

    torch.manual_seed(0)
    x = _probe(n=8, size=28)
    p4 = equivariance_error(make_p4_cnn(width=4)(), x, 90.0)
    plain = equivariance_error(make_plain_cnn(width=4)(), x, 90.0)
    assert p4 < plain


def test_f1_runs():
    from parts.f_geometric.f1_equivariant_cnn import main

    vision.require_torch()
    assert (
        main(
            [
                "--train-samples",
                "256",
                "--test-samples",
                "128",
                "--epochs",
                "1",
                "--width",
                "4",
                "--eval-batch",
                "32",
                "--angles",
                "0",
                "90",
                "45",
                "--arms",
                "p4",
                "plain",
            ]
        )
        == 0
    )


# -- F2 -----------------------------------------------------------------------


def _toy_graph(nodes=40, classes=3, seed=0):
    from mlladder import graphs

    return graphs.make_sbm(
        num_nodes=nodes,
        num_classes=classes,
        feature_dim=8,
        homophily=0.8,
        average_degree=6.0,
        rng=np.random.default_rng(seed),
    )


def test_mean_aggregator_excludes_the_node_and_averages_its_neighbours():
    from parts.f_geometric.f2_gcn_graphsage import mean_adjacency

    graph = _toy_graph()
    bar = mean_adjacency(graph)
    # No self-loop: the node's own features do not enter the aggregate at all,
    # which is exactly what the separate self-weight is for.
    assert torch.allclose(bar.diagonal(), torch.zeros(graph.num_nodes))
    connected = torch.from_numpy(graph.degree()) > 0
    assert torch.allclose(bar.sum(dim=1)[connected], torch.ones(int(connected.sum())), atol=1e-6)


def test_gcn_normalization_keeps_the_node_and_stays_contractive():
    from mlladder import graphs

    graph = _toy_graph()
    hat = graphs.normalized_adjacency(graph)
    assert torch.allclose(hat, hat.T, atol=1e-6)  # symmetric
    assert (hat.diagonal() > 0).all()  # the self-loop is present
    # Spectral radius 1 is why the diffusion converges rather than diverging.
    assert float(torch.linalg.eigvalsh(hat).abs().max()) <= 1.0 + 1e-5


def test_message_passing_is_permutation_equivariant():
    """The symmetry this part is about, checked the way F1 checks rotations.

    Relabel the nodes and the answers come back relabelled, exactly -- no
    training, no tolerance beyond float32. This is the graph analogue of the
    quarter-turn test above, and it holds for both graph arms by construction.
    """
    from parts.f_geometric.f2_gcn_graphsage import ARCHITECTURES, mean_adjacency

    from mlladder import graphs

    graph = _toy_graph()
    features = torch.from_numpy(graph.features)
    operators = {"gcn": graphs.normalized_adjacency(graph), "sage": mean_adjacency(graph)}
    permutation = torch.from_numpy(np.random.default_rng(1).permutation(graph.num_nodes))

    for arm, adjacency in (("gcn", operators["gcn"]), ("sage", operators["sage"])):
        torch.manual_seed(0)
        model = ARCHITECTURES[arm](graph.features.shape[1], 16, graph.num_classes, dropout=0.0)().eval()
        with torch.no_grad():
            direct = model(features, adjacency)[permutation]
            relabelled = model(features[permutation], adjacency[permutation][:, permutation])
        assert torch.allclose(direct, relabelled, atol=1e-5), arm


def test_graphsage_contains_the_graph_blind_model_and_the_gcn_does_not():
    """Zero GraphSAGE's neighbour weights and it *is* the MLP, to floating point.

    That is the architectural claim the homophily sweep tests empirically: at
    low homophily the best available function is "ignore the edges", and
    GraphSAGE can represent it exactly while a GCN -- whose single $W$ is
    applied to self and neighbours alike -- can only approach it by making the
    whole layer small.
    """
    from parts.f_geometric.f2_gcn_graphsage import ARCHITECTURES, mean_adjacency

    graph = _toy_graph()
    features = torch.from_numpy(graph.features)
    adjacency = mean_adjacency(graph)

    torch.manual_seed(0)
    sage = ARCHITECTURES["sage"](graph.features.shape[1], 16, graph.num_classes, dropout=0.0)().eval()
    torch.manual_seed(0)
    mlp = ARCHITECTURES["mlp"](graph.features.shape[1], 16, graph.num_classes, dropout=0.0)().eval()
    with torch.no_grad():
        for linear in sage.neighbour_linears:
            linear.weight.zero_()
        for theirs, ours in zip(mlp.linears, sage.self_linears, strict=True):
            ours.weight.copy_(theirs.weight)
            ours.bias.copy_(theirs.bias)
        assert torch.allclose(sage(features, adjacency), mlp(features, adjacency), atol=1e-5)


def test_repeated_diffusion_collapses_the_representations():
    """Oversmoothing without a single trained parameter.

    $\\hat{A}^k$ converges to a projector onto its top eigenspace, so any
    features pushed through enough layers end up collinear. This is the
    mechanism the depth sweep measures; separating it from training pathologies
    is why the rung reports cosine similarity alongside accuracy.

    The limit is rank one only on a *connected* graph. A graph with $k$
    components has $k$ eigenvalues equal to 1 and collapses to a $k$-dimensional
    span instead, so the similarity plateaus short of 1 -- measured at 0.988 for
    a two-component graph, and it stays there from 32 iterations to 256. The
    convergence is fast; what is left is structure, not slowness.
    """
    from parts.f_geometric.f2_gcn_graphsage import smoothness

    from mlladder import graphs

    def collapse(average_degree):
        graph = _toy_graph_with_degree(average_degree)
        hat = graphs.normalized_adjacency(graph)
        components = int((torch.linalg.eigvalsh(hat) > 1 - 1e-5).sum())
        h = torch.from_numpy(np.random.default_rng(0).normal(size=(graph.num_nodes, 16)).astype(np.float32))
        start = smoothness(h)
        for _ in range(64):
            h = hat @ h
        return components, start, smoothness(h)

    components, start, end = collapse(12.0)
    assert components == 1
    assert start < 0.25  # random features point nowhere in particular
    assert end > 0.999  # ... and after twelve dozen averagings, all the same way

    components, _, end = collapse(6.0)
    assert components > 1
    assert 0.95 < end < 1.0  # collapsed per component, not to a single vector


def _toy_graph_with_degree(average_degree):
    from mlladder import graphs

    return graphs.make_sbm(
        num_nodes=120,
        num_classes=3,
        feature_dim=8,
        homophily=0.8,
        average_degree=average_degree,
        rng=np.random.default_rng(0),
    )


def test_the_homophily_dial_hits_what_it_is_asked_for():
    from mlladder import graphs

    for target in (0.3, 0.6, 0.9):
        graph = graphs.make_sbm(num_nodes=1500, num_classes=4, homophily=target, rng=np.random.default_rng(0))
        assert abs(graph.homophily() - target) < 0.02, (target, graph.homophily())


def test_the_random_floor_is_one_over_c_and_the_dial_goes_below_it():
    """Two facts that keep the homophily axis readable.

    First, *measured* homophily on edges drawn without regard to class is not 0
    but $1/C$ -- they agree that often by accident -- so the four-class sweep's
    "random" control sits at 0.25, and a sweep that labelled that point 0 would
    look like a bug.

    Second, the generator is not sampling a random graph and reporting what it
    got: it places same- and cross-class edges to hit the target, so it can go
    *below* the random floor into deliberate heterophily, where the neighbours
    are evidence against a node's label rather than merely uninformative. That
    is a different regime, and the sweep visits it on purpose.
    """
    from mlladder import graphs

    random_floor = graphs.make_sbm(num_nodes=1500, num_classes=4, homophily=0.25, rng=np.random.default_rng(0))
    assert abs(random_floor.homophily() - 0.25) < 0.02

    heterophilous = graphs.make_sbm(num_nodes=1500, num_classes=4, homophily=0.05, rng=np.random.default_rng(0))
    assert heterophilous.homophily() < 0.1


def test_crossover_interpolates_and_admits_when_there_is_none():
    from parts.f_geometric.f2_gcn_graphsage import crossover

    assert crossover([0.0, 1.0], [-1.0, 1.0]) == pytest.approx(0.5)
    assert np.isnan(crossover([0.0, 1.0], [1.0, 2.0]))


def test_split_is_planetoid_sized_and_disjoint():
    from mlladder import graphs

    graph = _toy_graph(nodes=400, classes=4)
    train, val, test = graphs.split_nodes(graph, train_per_class=20, val_size=50, test_size=100)
    assert len(train) == 80
    assert len(set(train.tolist()) & set(val.tolist())) == 0
    assert len(set(train.tolist()) & set(test.tolist())) == 0
    assert len(set(val.tolist()) & set(test.tolist())) == 0
    counts = np.bincount(graph.labels[train], minlength=graph.num_classes)
    assert (counts == 20).all()


def test_f2_runs():
    from parts.f_geometric.f2_gcn_graphsage import main

    vision.require_torch()
    assert (
        main(
            [
                "--epochs",
                "3",
                "--depth-epochs",
                "4",
                "--hidden",
                "8",
                "--depths",
                "1",
                "2",
                "--homophilies",
                "0.3",
                "0.9",
                "--sbm-nodes",
                "200",
                "--train-per-class",
                "5",
            ]
        )
        == 0
    )
