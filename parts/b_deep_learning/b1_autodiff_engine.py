"""B1. Scalar autodiff engine (micrograd-style) plus a tiny MLP trained on it

Lab reference: [karpathy/micrograd — autodiff reference implementation](https://github.com/karpathy/micrograd)

Reverse-mode automatic differentiation is the chain rule applied to a DAG in
reverse topological order. Every operation records how to push a gradient from
its output to its inputs; ``backward()`` seeds the output with 1 and walks the
graph once. One backward pass costs about one forward pass and yields *every*
parameter gradient, which is the whole reason deep learning is affordable.

The engine here is scalar — one node per number — which makes the mechanism
completely visible and the arithmetic completely impractical. The rung measures
both: gradients against central differences, and the price of scalars against
the same network written in numpy.

Run with ``ladder run B1`` (``--help`` for flags).
"""

from __future__ import annotations

import math
import random
import time
from collections.abc import Iterator

import numpy as np

from mlladder import Experiment, datasets, experiment, metrics, plotting


class Value:
    """A scalar node in the computation graph.

    ``data`` is the forward value, ``grad`` the accumulated $\\partial L/\\partial v$,
    and ``_backward`` the local rule that pushes this node's gradient to its
    inputs. Gradients accumulate (``+=``) because a node used twice receives a
    contribution along each path — the multivariable chain rule.
    """

    __slots__ = ("_backward", "_op", "_prev", "data", "grad")

    def __init__(self, data: float, children: tuple[Value, ...] = (), op: str = "") -> None:
        self.data = float(data)
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = children
        self._op = op

    # -- arithmetic -------------------------------------------------------
    def __add__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), "+")

        def _backward() -> None:
            self.grad += out.grad
            other.grad += out.grad

        out._backward = _backward
        return out

    def __mul__(self, other: Value | float) -> Value:
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), "*")

        def _backward() -> None:
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad

        out._backward = _backward
        return out

    def __pow__(self, exponent: float) -> Value:
        out = Value(self.data**exponent, (self,), f"**{exponent}")

        def _backward() -> None:
            self.grad += exponent * self.data ** (exponent - 1) * out.grad

        out._backward = _backward
        return out

    def exp(self) -> Value:
        out = Value(math.exp(self.data), (self,), "exp")

        def _backward() -> None:
            self.grad += out.data * out.grad  # d/dx e^x = e^x

        out._backward = _backward
        return out

    def log(self) -> Value:
        out = Value(math.log(self.data), (self,), "log")

        def _backward() -> None:
            self.grad += out.grad / self.data

        out._backward = _backward
        return out

    def tanh(self) -> Value:
        t = math.tanh(self.data)
        out = Value(t, (self,), "tanh")

        def _backward() -> None:
            self.grad += (1 - t * t) * out.grad

        out._backward = _backward
        return out

    def relu(self) -> Value:
        out = Value(self.data if self.data > 0 else 0.0, (self,), "relu")

        def _backward() -> None:
            self.grad += (out.data > 0) * out.grad

        out._backward = _backward
        return out

    def sigmoid(self) -> Value:
        s = 1.0 / (1.0 + math.exp(-self.data)) if self.data >= 0 else math.exp(self.data) / (1.0 + math.exp(self.data))
        out = Value(s, (self,), "sigmoid")

        def _backward() -> None:
            self.grad += s * (1 - s) * out.grad

        out._backward = _backward
        return out

    def __neg__(self) -> Value:
        return self * -1

    def __radd__(self, other: float) -> Value:
        return self + other

    def __sub__(self, other: Value | float) -> Value:
        return self + (-other if isinstance(other, Value) else Value(-other))

    def __rsub__(self, other: float) -> Value:
        return Value(other) + (-self)

    def __rmul__(self, other: float) -> Value:
        return self * other

    def __truediv__(self, other: Value | float) -> Value:
        return self * (other**-1 if isinstance(other, Value) else Value(other) ** -1)

    def __repr__(self) -> str:
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"

    # -- the engine -------------------------------------------------------
    def backward(self) -> None:
        """Seed this node with 1 and push gradients through the graph once."""
        order: list[Value] = []
        visited: set[int] = set()

        # Iterative DFS: a deep graph would blow the Python recursion limit.
        stack: list[tuple[Value, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for child in node._prev:
                if id(child) not in visited:
                    stack.append((child, False))

        self.grad = 1.0
        for node in reversed(order):
            node._backward()

    def graph_size(self) -> int:
        seen: set[int] = set()
        stack = [self]
        while stack:
            node = stack.pop()
            if id(node) in seen:
                continue
            seen.add(id(node))
            stack.extend(node._prev)
        return len(seen)


class Neuron:
    """One unit: $\\sigma(w^\\top x + b)$, with every weight its own ``Value``."""

    def __init__(self, n_inputs: int, activation: str = "tanh", rng: random.Random | None = None) -> None:
        generator = rng or random.Random(0)
        # Xavier-ish scaling keeps early activations off the saturating tails.
        scale = 1.0 / math.sqrt(n_inputs)
        self.w = [Value(generator.uniform(-scale, scale)) for _ in range(n_inputs)]
        self.b = Value(0.0)
        self.activation = activation

    def __call__(self, x: list[Value]) -> Value:
        total = self.b
        for weight, value in zip(self.w, x, strict=True):
            total = total + weight * value
        if self.activation == "tanh":
            return total.tanh()
        if self.activation == "relu":
            return total.relu()
        return total

    def parameters(self) -> Iterator[Value]:
        yield from self.w
        yield self.b


class MLP:
    """A stack of fully connected layers; the output layer stays linear."""

    def __init__(self, sizes: list[int], activation: str = "tanh", rng: random.Random | None = None) -> None:
        generator = rng or random.Random(0)
        self.layers = [
            [
                Neuron(sizes[i], activation="linear" if i == len(sizes) - 2 else activation, rng=generator)
                for _ in range(sizes[i + 1])
            ]
            for i in range(len(sizes) - 1)
        ]

    def __call__(self, x: list[Value]) -> list[Value]:
        for layer in self.layers:
            x = [neuron(x) for neuron in layer]
        return x

    def parameters(self) -> list[Value]:
        return [parameter for layer in self.layers for neuron in layer for parameter in neuron.parameters()]

    def zero_grad(self) -> None:
        for parameter in self.parameters():
            parameter.grad = 0.0


def binary_cross_entropy(logit: Value, target: float) -> Value:
    """Numerically sane BCE on a logit: $\\log(1+e^{-z})$ for $y=1$."""
    probability = logit.sigmoid()
    epsilon = 1e-9
    if target > 0.5:
        return -((probability + epsilon).log())
    return -((1.0 - probability + epsilon).log())


def numerical_gradient(loss_fn, parameter: Value, epsilon: float = 1e-5) -> float:
    """Central difference: $\\frac{f(x+h) - f(x-h)}{2h}$, error $O(h^2)$."""
    original = parameter.data
    parameter.data = original + epsilon
    plus = loss_fn()
    parameter.data = original - epsilon
    minus = loss_fn()
    parameter.data = original
    return (plus - minus) / (2 * epsilon)


class NumpyMLP:
    """The same network, vectorized — the control for the cost measurement."""

    def __init__(self, sizes: list[int], rng: np.random.Generator) -> None:
        self.weights = [
            rng.uniform(-1, 1, size=(sizes[i], sizes[i + 1])) / np.sqrt(sizes[i]) for i in range(len(sizes) - 1)
        ]
        self.biases = [np.zeros(sizes[i + 1]) for i in range(len(sizes) - 1)]

    def forward(self, features: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        activations = [features]
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            pre = activations[-1] @ weight + bias
            activations.append(pre if index == len(self.weights) - 1 else np.tanh(pre))
        return activations[-1].ravel(), activations

    def step(self, features: np.ndarray, targets: np.ndarray, lr: float) -> float:
        """One full-batch step of manual backprop through the same architecture."""
        logits, activations = self.forward(features)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        loss = float(
            -np.mean(targets * np.log(probabilities + 1e-9) + (1 - targets) * np.log(1 - probabilities + 1e-9))
        )

        delta = ((probabilities - targets) / len(targets)).reshape(-1, 1)
        for index in reversed(range(len(self.weights))):
            grad_weight = activations[index].T @ delta
            grad_bias = delta.sum(axis=0)
            if index > 0:
                delta = (delta @ self.weights[index].T) * (1 - activations[index] ** 2)
            self.weights[index] -= lr * grad_weight
            self.biases[index] -= lr * grad_bias
        return loss


@experiment(
    "b1_autodiff_engine",
    defaults={
        "samples": 120,
        "hidden": [8, 8],
        "epochs": 60,
        "lr": 0.3,
        "noise": 0.18,
        "gradient_checks": 25,
    },
    description="A scalar reverse-mode autodiff engine, gradient-checked and priced against numpy.",
)
def run(exp: Experiment) -> dict[str, float]:
    """Scalar autodiff engine (micrograd-style) plus a tiny MLP trained on it."""
    rng = random.Random(exp.seed)
    numpy_rng = np.random.default_rng(exp.seed)

    points, labels = datasets.make_moons(exp["samples"], noise=exp["noise"], rng=numpy_rng)
    x_train, x_test, y_train, y_test = datasets.train_test_split(points, labels, rng=numpy_rng)
    inputs = [[Value(feature) for feature in sample] for sample in x_train]

    sizes = [x_train.shape[1], *exp["hidden"], 1]
    model = MLP(sizes, rng=rng)
    print(f"[b1] architecture {sizes} — {len(model.parameters())} parameters")

    def total_loss() -> float:
        """Forward-only loss, for the finite-difference check."""
        return float(
            sum(binary_cross_entropy(model(x)[0], target).data for x, target in zip(inputs, y_train, strict=True))
        )

    # -- gradient check against central differences ------------------------
    model.zero_grad()
    loss = sum(
        (binary_cross_entropy(model(x)[0], target) for x, target in zip(inputs, y_train, strict=True)), Value(0.0)
    )
    graph_nodes = loss.graph_size()
    loss.backward()

    parameters = model.parameters()
    checked = numpy_rng.choice(len(parameters), size=min(exp["gradient_checks"], len(parameters)), replace=False)
    analytic = np.array([parameters[index].grad for index in checked])
    numerical = np.array([numerical_gradient(total_loss, parameters[index]) for index in checked])
    scale = np.maximum(np.abs(analytic) + np.abs(numerical), 1e-12)
    max_relative_error = float(np.max(np.abs(analytic - numerical) / scale))
    print(f"[b1] graph: {graph_nodes} nodes for {len(inputs)} samples ({graph_nodes / len(inputs):.0f} per sample)")
    print(f"[b1] gradient check on {len(checked)} parameters: max relative error {max_relative_error:.2e}")

    # -- train on the scalar engine ----------------------------------------
    started = time.perf_counter()
    losses, accuracies = [], []
    for epoch in range(exp["epochs"]):
        model.zero_grad()
        batch_loss = sum(
            (binary_cross_entropy(model(x)[0], target) for x, target in zip(inputs, y_train, strict=True)), Value(0.0)
        ) * (1.0 / len(inputs))
        batch_loss.backward()
        for parameter in model.parameters():
            parameter.data -= exp["lr"] * parameter.grad

        predictions = np.array([1.0 if model([Value(f) for f in sample])[0].data > 0 else 0.0 for sample in x_test])
        accuracy = metrics.accuracy(y_test, predictions)
        losses.append(batch_loss.data)
        accuracies.append(accuracy)
        if epoch % 10 == 0 or epoch == exp["epochs"] - 1:
            print(f"[b1] epoch {epoch:>3}  loss={batch_loss.data:.4f}  test accuracy={accuracy:.4f}")
        exp.log({"train/loss": batch_loss.data, "test/accuracy": accuracy}, step=epoch)
    scalar_seconds = time.perf_counter() - started

    # -- the same network, vectorized --------------------------------------
    numpy_model = NumpyMLP(sizes, np.random.default_rng(exp.seed))
    started = time.perf_counter()
    numpy_losses = [numpy_model.step(x_train, y_train.astype(float), exp["lr"]) for _ in range(exp["epochs"])]
    numpy_seconds = time.perf_counter() - started
    numpy_predictions = (numpy_model.forward(x_test)[0] > 0).astype(float)
    numpy_accuracy = metrics.accuracy(y_test, numpy_predictions)

    print(
        f"[b1] scalar engine: {scalar_seconds:.2f}s   numpy: {numpy_seconds:.3f}s   ratio {scalar_seconds / numpy_seconds:.0f}x"
    )
    print(f"[b1] final accuracy — scalar {accuracies[-1]:.4f}, numpy {numpy_accuracy:.4f}")

    # -- figures -----------------------------------------------------------
    exp.log_figure(
        "training",
        plotting.plot_curves(
            {"scalar engine": losses, "numpy (same net)": numpy_losses},
            title="Tiny MLP — train BCE",
            xlabel="epoch",
            ylabel="loss",
        ),
    )
    exp.log_figure(
        "test_accuracy",
        plotting.plot_curves(
            {"test accuracy": accuracies}, title="Tiny MLP on moons", xlabel="epoch", ylabel="accuracy"
        ),
    )

    def predict(grid: np.ndarray) -> np.ndarray:
        return np.array([1 if model([Value(f) for f in sample])[0].data > 0 else 0 for sample in grid])

    exp.log_figure(
        "decision_boundary",
        plotting.plot_decision_boundary(predict, x_test, y_test, title="Scalar-engine MLP — test set", resolution=60),
    )
    exp.log_figure(
        "gradient_check",
        plotting.plot_comparison(
            {"analytic": float(np.abs(analytic).mean()), "central difference": float(np.abs(numerical).mean())},
            title=f"Mean |gradient| over {len(checked)} parameters (max rel. error {max_relative_error:.1e})",
            ylabel="mean |gradient|",
        ),
    )

    return {
        "parameters": float(len(parameters)),
        "graph_nodes": float(graph_nodes),
        "nodes_per_sample": graph_nodes / len(inputs),
        "gradient/max_relative_error": max_relative_error,
        "scalar/test_accuracy": accuracies[-1],
        "scalar/seconds": scalar_seconds,
        "numpy/test_accuracy": numpy_accuracy,
        "numpy/seconds": numpy_seconds,
        "slowdown_vs_numpy": scalar_seconds / numpy_seconds,
    }


main = run

if __name__ == "__main__":
    raise SystemExit(main())
