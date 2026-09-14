"""Part B — deep learning foundations.

The load-bearing tests are the two agreement checks: the scalar engine against
central differences, and hand-derived matrix backprop against autograd. The
rest are shape and behaviour checks that keep the rungs runnable.
"""

from __future__ import annotations

import numpy as np
import pytest
from parts.b_deep_learning.b1_autodiff_engine import MLP, NumpyMLP, Value, binary_cross_entropy, numerical_gradient
from parts.b_deep_learning.b2_mlp_mnist import ManualMLP, softmax_cross_entropy, torch_model_from

from mlladder import datasets, vision

torch = pytest.importorskip("torch", reason="uv sync --extra torch")


@pytest.fixture(autouse=True)
def isolated_runs(tmp_path, monkeypatch):
    monkeypatch.setenv("MLLADDER_ARTIFACTS_DIR", str(tmp_path / "artifacts"))
    monkeypatch.setenv("MLLADDER_TRACKER", "none")


# -- B1: the scalar engine -------------------------------------------------


def test_chain_rule_on_a_hand_checkable_graph():
    a, b, c = Value(2.0), Value(-3.0), Value(10.0)
    out = a * b + c  # d/da = b, d/db = a, d/dc = 1
    out.backward()
    assert out.data == pytest.approx(4.0)
    assert (a.grad, b.grad, c.grad) == pytest.approx((-3.0, 2.0, 1.0))


def test_gradients_accumulate_when_a_node_is_reused():
    x = Value(3.0)
    out = x * x  # d/dx = 2x, only correct if both paths accumulate
    out.backward()
    assert x.grad == pytest.approx(6.0)


@pytest.mark.parametrize(
    ("method", "point", "derivative"),
    [
        ("tanh", 0.5, 1 - np.tanh(0.5) ** 2),
        ("relu", 1.5, 1.0),
        ("relu", -1.5, 0.0),
        ("exp", 0.3, float(np.exp(0.3))),
        ("log", 2.0, 0.5),
        ("sigmoid", 0.7, float(1 / (1 + np.exp(-0.7)) * (1 - 1 / (1 + np.exp(-0.7))))),
    ],
)
def test_elementwise_derivatives(method, point, derivative):
    x = Value(point)
    getattr(x, method)().backward()
    assert x.grad == pytest.approx(derivative, rel=1e-9)


def test_backward_survives_a_graph_deeper_than_the_recursion_limit():
    x = Value(1.0)
    node = x
    for _ in range(5000):  # a recursive implementation dies here
        node = node + 1.0
    node.backward()
    assert x.grad == pytest.approx(1.0)


def test_scalar_mlp_gradients_match_central_differences():
    import random

    rng = np.random.default_rng(0)
    points, labels = datasets.make_moons(24, noise=0.1, rng=rng)
    inputs = [[Value(feature) for feature in sample] for sample in points]
    model = MLP([2, 4, 1], rng=random.Random(0))

    def total_loss() -> float:
        return float(sum(binary_cross_entropy(model(x)[0], y).data for x, y in zip(inputs, labels, strict=True)))

    model.zero_grad()
    loss = sum((binary_cross_entropy(model(x)[0], y) for x, y in zip(inputs, labels, strict=True)), Value(0.0))
    loss.backward()

    for parameter in model.parameters()[:8]:
        analytic = parameter.grad
        numeric = numerical_gradient(total_loss, parameter)
        assert analytic == pytest.approx(numeric, rel=1e-4, abs=1e-6)


def test_numpy_mlp_trains():
    rng = np.random.default_rng(1)
    points, labels = datasets.make_moons(200, noise=0.1, rng=rng)
    model = NumpyMLP([2, 8, 1], rng)
    first = model.step(points, labels.astype(float), 0.5)
    for _ in range(50):
        last = model.step(points, labels.astype(float), 0.5)
    assert last < first


# -- B2: matrix backprop vs autograd ---------------------------------------


def test_softmax_cross_entropy_matches_torch():
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(16, 5)).astype(np.float32)
    labels = rng.integers(0, 5, size=16)

    loss, gradient = softmax_cross_entropy(logits, labels)

    tensor = torch.from_numpy(logits).requires_grad_(True)
    torch_loss = torch.nn.functional.cross_entropy(tensor, torch.from_numpy(labels))
    torch_loss.backward()

    assert loss == pytest.approx(torch_loss.item(), rel=1e-6)
    assert np.allclose(gradient, tensor.grad.numpy(), atol=1e-6)


def test_manual_backprop_matches_autograd_layer_by_layer():
    rng = np.random.default_rng(3)
    features = rng.normal(size=(32, 12)).astype(np.float32)
    labels = rng.integers(0, 4, size=32)

    manual = ManualMLP([12, 16, 8, 4]).initialize(rng)
    model, linear_layers = torch_model_from(manual)

    loss, grad_weights, grad_biases = manual.gradients(features, labels)
    torch_loss = torch.nn.functional.cross_entropy(model(torch.from_numpy(features)), torch.from_numpy(labels))
    model.zero_grad()
    torch_loss.backward()

    assert loss == pytest.approx(torch_loss.item(), rel=1e-5)
    for index, layer in enumerate(linear_layers):
        assert np.allclose(layer.weight.grad.numpy().T, grad_weights[index], atol=1e-5)
        assert np.allclose(layer.bias.grad.numpy(), grad_biases[index], atol=1e-5)


def test_copied_weights_give_identical_forward_passes():
    rng = np.random.default_rng(4)
    manual = ManualMLP([6, 5, 3]).initialize(rng)
    model, _ = torch_model_from(manual)
    features = rng.normal(size=(7, 6)).astype(np.float32)
    with torch.no_grad():
        assert np.allclose(manual.forward(features)[0], model(torch.from_numpy(features)).numpy(), atol=1e-5)


# -- B3/B4/B5: architectures -----------------------------------------------


def test_cnn_shapes_and_bias_convention():
    from parts.b_deep_learning.b3_cnn_cifar10 import build_cnn

    plain = build_cnn(use_batchnorm=False, dropout=0.0)
    with_norm = build_cnn(use_batchnorm=True, dropout=0.25)
    batch = torch.randn(4, 3, 32, 32)
    assert plain(batch).shape == (4, 10)
    assert with_norm(batch).shape == (4, 10)

    convolutions = [m for m in with_norm.modules() if isinstance(m, torch.nn.Conv2d)]
    assert all(conv.bias is None for conv in convolutions), "BN makes the conv bias redundant"
    assert any(isinstance(m, torch.nn.Dropout) for m in with_norm.modules())


def test_residual_block_shortcut_behaviour():
    from parts.b_deep_learning.b4_resnet_blocks import make_block_class

    ResidualBlock = make_block_class()
    batch = torch.randn(2, 8, 16, 16)

    same_shape = ResidualBlock(8, 8, residual=True)
    assert isinstance(same_shape.shortcut, torch.nn.Identity), "identity skip where shapes allow"
    assert same_shape(batch).shape == (2, 8, 16, 16)

    projected = ResidualBlock(8, 16, stride=2, residual=True)
    assert not isinstance(projected.shortcut, torch.nn.Identity)
    assert projected(batch).shape == (2, 16, 8, 8)

    torch.manual_seed(0)
    with_skip = ResidualBlock(8, 8, residual=True)
    torch.manual_seed(0)
    without_skip = ResidualBlock(8, 8, residual=False)
    with_skip.eval(), without_skip.eval()
    with torch.no_grad():
        assert not torch.allclose(with_skip(batch), without_skip(batch))


def test_residual_networks_have_a_flatter_gradient_profile():
    """The mechanism, as measured rather than as usually told.

    With BatchNorm everywhere, the plain stack does *not* suffer vanishing
    gradients — its early-layer gradients are larger, if anything. What the
    identity paths change is the spread of gradient magnitude across depth:
    the first/last ratio is consistently smaller for the residual net.
    """
    from parts.b_deep_learning.b4_resnet_blocks import build_network, first_layer_gradient_norms

    from mlladder import torchkit

    features = np.random.default_rng(5).normal(size=(32, 3, 32, 32)).astype(np.float32)
    labels = np.random.default_rng(6).integers(0, 10, size=32)
    loader = torchkit.eval_loader(features, labels, batch_size=32)

    torchkit.seed_torch(0)
    plain = first_layer_gradient_norms(build_network(8, residual=False, width=8), loader)
    torchkit.seed_torch(0)
    residual = first_layer_gradient_norms(build_network(8, residual=True, width=8), loader)

    assert all(np.isfinite(plain)) and all(np.isfinite(residual))
    assert min(residual) > 0, "no tensor should be cut off from the loss"
    plain_spread = plain[0] / plain[-1]
    residual_spread = residual[0] / residual[-1]
    assert residual_spread < plain_spread


def test_optimizer_factory():
    from parts.b_deep_learning.b5_optimizer_bakeoff import build_mlp, make_optimizer

    model = build_mlp(20, [8])
    assert isinstance(make_optimizer("sgd", model.parameters(), 0.1, 0.0), torch.optim.SGD)
    assert make_optimizer("sgd+momentum", model.parameters(), 0.1, 0.0).param_groups[0]["momentum"] == 0.9
    assert isinstance(make_optimizer("adamw", model.parameters(), 1e-3, 0.01), torch.optim.AdamW)
    with pytest.raises(ValueError, match="unknown optimizer"):
        make_optimizer("nope", model.parameters(), 0.1, 0.0)


# -- shared plumbing --------------------------------------------------------


def test_vision_loader_shapes_and_normalization():
    images, labels = vision.load("mnist", split="test", limit=64, rng=0)
    assert images.shape == (64, 28, 28, 1) and images.dtype == np.uint8
    assert labels.shape == (64,)

    normalized = vision.normalize(images, "mnist")
    assert normalized.shape == (64, 1, 28, 28) and normalized.dtype == np.float32
    assert vision.flatten(normalized).shape == (64, 784)

    with pytest.raises(ValueError, match="unknown dataset"):
        vision.load("not-a-dataset")


def test_torchkit_fit_reduces_loss():
    from mlladder import torchkit

    rng = np.random.default_rng(7)
    features = rng.normal(size=(256, 10)).astype(np.float32)
    labels = (features[:, 0] > 0).astype(np.int64)
    model = torch.nn.Sequential(torch.nn.Linear(10, 16), torch.nn.ReLU(), torch.nn.Linear(16, 2))
    train_loader, test_loader = torchkit.loaders(features, labels, features, labels, batch_size=32)

    history = torchkit.fit(model, train_loader, test_loader, torch.optim.Adam(model.parameters(), lr=0.05), epochs=5)
    assert history["train_loss"][-1] < history["train_loss"][0]
    assert history["test_accuracy"][-1] > 0.8
    assert len(history["seconds"]) == 5

    predicted, actual = torchkit.predictions(model, test_loader)
    assert predicted.shape == actual.shape == (256,)


# -- the rungs end to end ---------------------------------------------------


def test_b1_runs():
    from parts.b_deep_learning.b1_autodiff_engine import main

    assert main(["--samples", "40", "--epochs", "3", "--hidden", "4", "--gradient-checks", "5"]) == 0


def test_b2_runs():
    from parts.b_deep_learning.b2_mlp_mnist import main

    assert main(["--train-samples", "2000", "--test-samples", "500", "--epochs", "1", "--hidden", "32"]) == 0


def test_b3_runs():
    from parts.b_deep_learning.b3_cnn_cifar10 import main

    assert main(["--train-samples", "256", "--test-samples", "256", "--epochs", "1", "--probe-epochs", "1"]) == 0


def test_b4_runs():
    from parts.b_deep_learning.b4_resnet_blocks import main

    assert (
        main(
            [
                "--train-samples",
                "256",
                "--test-samples",
                "256",
                "--epochs",
                "1",
                "--depths",
                "2",
                "--width",
                "8",
                "--cifar100-depth",
                "2",
                "--cifar100-epochs",
                "1",
            ]
        )
        == 0
    )


def test_b5_runs():
    from parts.b_deep_learning.b5_optimizer_bakeoff import main

    assert (
        main(
            [
                "--train-samples",
                "512",
                "--test-samples",
                "256",
                "--epochs",
                "1",
                "--hidden",
                "32",
                "--sweep-lrs",
                "0.01",
                "--sweep-epochs",
                "1",
            ]
        )
        == 0
    )
