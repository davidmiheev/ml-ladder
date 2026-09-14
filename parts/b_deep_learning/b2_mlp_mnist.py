"""B2. MLP on MNIST — manual backprop, then repeat with PyTorch autograd and diff the gradients

Lab reference: [karpathy/micrograd — autodiff reference implementation](https://github.com/karpathy/micrograd)

B1 built the chain rule one scalar at a time; here it is written once per
*layer*, in matrix form, which is what makes it fast. The rung's claim is that
the hand-derived gradients are not merely close to autograd's but equal to it
within floating-point error — checked layer by layer on a shared batch with
identical initial weights, before either network is trained.

Run with ``ladder run B2`` (``--help`` for flags).
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from numpy.typing import NDArray

from mlladder import Experiment, experiment, metrics, plotting, vision

Array = NDArray[np.float32]


def softmax_cross_entropy(logits: Array, labels: NDArray[np.int64]) -> tuple[float, Array]:
    """Mean loss and its gradient w.r.t. the logits: $(p - y)/n$.

    The softmax and the log cancel analytically, so the gradient never forms
    $\\partial p/\\partial z$ — the reason this pair is always fused.
    """
    shifted = logits - logits.max(axis=1, keepdims=True)  # stability: max logit becomes 0
    exponentials = np.exp(shifted)
    probabilities = exponentials / exponentials.sum(axis=1, keepdims=True)

    n = len(labels)
    loss = float(-np.mean(np.log(probabilities[np.arange(n), labels] + 1e-12)))
    gradient = probabilities.copy()
    gradient[np.arange(n), labels] -= 1.0
    return loss, gradient / n


@dataclass
class ManualMLP:
    """A ReLU MLP with backprop written out by hand.

    Layer $l$ computes $z^{(l)} = a^{(l-1)}W^{(l)} + b^{(l)}$, and the backward
    recursion is $\\delta^{(l)} = (\\delta^{(l+1)}W^{(l+1)\\top}) \\odot \\mathbf{1}[z^{(l)} > 0]$
    with $\\partial L/\\partial W^{(l)} = a^{(l-1)\\top}\\delta^{(l)}$.
    """

    sizes: list[int]
    weights: list[Array] = field(default_factory=list)
    biases: list[Array] = field(default_factory=list)

    def initialize(self, rng: np.random.Generator) -> ManualMLP:
        # He initialization: variance 2/fan_in keeps ReLU activations from
        # collapsing or exploding as depth grows.
        self.weights = [
            (rng.normal(size=(self.sizes[i], self.sizes[i + 1])) * np.sqrt(2.0 / self.sizes[i])).astype(np.float32)
            for i in range(len(self.sizes) - 1)
        ]
        self.biases = [np.zeros(self.sizes[i + 1], dtype=np.float32) for i in range(len(self.sizes) - 1)]
        return self

    def forward(self, features: Array) -> tuple[Array, list[Array], list[Array]]:
        activations = [features]
        pre_activations: list[Array] = []
        for index, (weight, bias) in enumerate(zip(self.weights, self.biases, strict=True)):
            pre = activations[-1] @ weight + bias
            pre_activations.append(pre)
            activations.append(pre if index == len(self.weights) - 1 else np.maximum(pre, 0.0))
        return activations[-1], activations, pre_activations

    def backward(
        self, activations: list[Array], pre_activations: list[Array], gradient: Array
    ) -> tuple[list[Array], list[Array]]:
        grad_weights: list[Array] = [np.zeros(0, dtype=np.float32)] * len(self.weights)
        grad_biases: list[Array] = [np.zeros(0, dtype=np.float32)] * len(self.biases)
        delta = gradient
        for index in reversed(range(len(self.weights))):
            grad_weights[index] = activations[index].T @ delta
            grad_biases[index] = delta.sum(axis=0)
            if index > 0:
                delta = (delta @ self.weights[index].T) * (pre_activations[index - 1] > 0)
        return grad_weights, grad_biases

    def gradients(self, features: Array, labels: NDArray[np.int64]) -> tuple[float, list[Array], list[Array]]:
        logits, activations, pre_activations = self.forward(features)
        loss, gradient = softmax_cross_entropy(logits, labels)
        grad_weights, grad_biases = self.backward(activations, pre_activations, gradient)
        return loss, grad_weights, grad_biases

    def step(self, features: Array, labels: NDArray[np.int64], lr: float) -> float:
        loss, grad_weights, grad_biases = self.gradients(features, labels)
        for index in range(len(self.weights)):
            self.weights[index] -= lr * grad_weights[index]
            self.biases[index] -= lr * grad_biases[index]
        return loss

    def predict(self, features: Array) -> NDArray[np.int64]:
        return np.argmax(self.forward(features)[0], axis=1)


def torch_model_from(manual: ManualMLP):
    """The same architecture in torch, seeded with the same weights.

    Identical initial parameters are what make the gradient comparison a
    statement about the *math* rather than about initialization.
    """
    torch = vision.require_torch()
    from torch import nn

    layers: list[nn.Module] = []
    for index in range(len(manual.sizes) - 1):
        layers.append(nn.Linear(manual.sizes[index], manual.sizes[index + 1]))
        if index < len(manual.sizes) - 2:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)

    linear_layers = [layer for layer in model if isinstance(layer, nn.Linear)]
    with torch.no_grad():
        for layer, weight, bias in zip(linear_layers, manual.weights, manual.biases, strict=True):
            # torch stores Linear weights transposed relative to the manual form.
            layer.weight.copy_(torch.from_numpy(weight.T))
            layer.bias.copy_(torch.from_numpy(bias))
    return model, linear_layers


@experiment(
    "b2_mlp_mnist",
    defaults={
        "train_samples": 20000,
        "test_samples": 5000,
        "hidden": [256, 64],
        "epochs": 8,
        "lr": 0.1,
        "batch_size": 128,
        "check_batch": 512,
    },
    description="A hand-derived MLP on MNIST, gradient-diffed against PyTorch autograd, then both trained.",
)
def run(exp: Experiment) -> dict[str, float]:
    """MLP on MNIST — manual backprop, then repeat with PyTorch autograd and diff the gradients."""
    torch = vision.require_torch()
    rng = np.random.default_rng(exp.seed)

    raw_train, y_train = vision.load("mnist", split="train", limit=exp["train_samples"], rng=rng)
    raw_test, y_test = vision.load("mnist", split="test", limit=exp["test_samples"], rng=rng)
    x_train = vision.flatten(vision.normalize(raw_train, "mnist"))
    x_test = vision.flatten(vision.normalize(raw_test, "mnist"))
    print(f"[b2] {len(x_train)} train / {len(x_test)} test, {x_train.shape[1]} features")

    sizes = [x_train.shape[1], *exp["hidden"], 10]
    manual = ManualMLP(sizes).initialize(rng)
    model, linear_layers = torch_model_from(manual)

    # -- the diff: same weights, same batch, both gradients -----------------
    batch = slice(0, exp["check_batch"])
    manual_loss, grad_weights, grad_biases = manual.gradients(x_train[batch], y_train[batch])

    features = torch.from_numpy(x_train[batch])
    labels = torch.from_numpy(y_train[batch])
    torch_loss = torch.nn.functional.cross_entropy(model(features), labels)
    model.zero_grad()
    torch_loss.backward()

    differences: dict[str, float] = {}
    for index, layer in enumerate(linear_layers):
        weight_difference = float(np.max(np.abs(layer.weight.grad.numpy().T - grad_weights[index])))
        bias_difference = float(np.max(np.abs(layer.bias.grad.numpy() - grad_biases[index])))
        differences[f"layer{index}/weight"] = weight_difference
        differences[f"layer{index}/bias"] = bias_difference
        print(f"[b2] layer {index}: max |dW diff|={weight_difference:.3e}  max |db diff|={bias_difference:.3e}")

    loss_difference = abs(manual_loss - torch_loss.item())
    worst = max(differences.values())
    gradient_scale = float(np.max(np.abs(grad_weights[0])))
    print(f"[b2] loss: manual {manual_loss:.6f} vs torch {torch_loss.item():.6f} (diff {loss_difference:.2e})")
    print(f"[b2] worst gradient difference {worst:.3e} against a gradient scale of {gradient_scale:.3e}")

    # -- train both, from the same weights ----------------------------------
    optimizer = torch.optim.SGD(model.parameters(), lr=exp["lr"])
    manual_losses, torch_losses, manual_accuracy, torch_accuracy = [], [], [], []
    manual_seconds = torch_seconds = 0.0

    for epoch in range(exp["epochs"]):
        epoch_rng = np.random.default_rng(exp.seed * 1000 + epoch)  # identical batch order for both
        started = time.perf_counter()
        losses = [
            manual.step(x_train[index], y_train[index], exp["lr"])
            for index in vision.batches(len(x_train), exp["batch_size"], epoch_rng)
        ]
        manual_seconds += time.perf_counter() - started
        manual_losses.append(float(np.mean(losses)))

        epoch_rng = np.random.default_rng(exp.seed * 1000 + epoch)
        started = time.perf_counter()
        batch_losses = []
        for index in vision.batches(len(x_train), exp["batch_size"], epoch_rng):
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(
                model(torch.from_numpy(x_train[index])), torch.from_numpy(y_train[index])
            )
            loss.backward()
            optimizer.step()
            batch_losses.append(loss.item())
        torch_seconds += time.perf_counter() - started
        torch_losses.append(float(np.mean(batch_losses)))

        manual_accuracy.append(metrics.accuracy(y_test, manual.predict(x_test)))
        with torch.no_grad():
            torch_accuracy.append(metrics.accuracy(y_test, model(torch.from_numpy(x_test)).argmax(dim=1).numpy()))
        print(
            f"[b2] epoch {epoch}: manual loss={manual_losses[-1]:.4f} acc={manual_accuracy[-1]:.4f} | "
            f"torch loss={torch_losses[-1]:.4f} acc={torch_accuracy[-1]:.4f}"
        )
        exp.log(
            {
                "manual/train_loss": manual_losses[-1],
                "manual/test_accuracy": manual_accuracy[-1],
                "torch/train_loss": torch_losses[-1],
                "torch/test_accuracy": torch_accuracy[-1],
            },
            step=epoch,
        )

    print(
        f"[b2] time: manual {manual_seconds:.1f}s, torch {torch_seconds:.1f}s ({torch_seconds / manual_seconds:.2f}x)"
    )

    # -- figures -------------------------------------------------------------
    exp.log_figure(
        "loss_curves",
        plotting.plot_curves(
            {"manual backprop": manual_losses, "torch autograd": torch_losses},
            title="MNIST MLP — train cross-entropy",
            xlabel="epoch",
            ylabel="loss",
        ),
    )
    exp.log_figure(
        "accuracy_curves",
        plotting.plot_curves(
            {"manual backprop": manual_accuracy, "torch autograd": torch_accuracy},
            title="MNIST MLP — test accuracy",
            xlabel="epoch",
            ylabel="accuracy",
        ),
    )
    exp.log_figure(
        "gradient_diff",
        plotting.plot_comparison(
            differences, title="Max |manual - autograd| gradient, per parameter tensor", ylabel="absolute difference"
        ),
    )
    exp.log_figure(
        "confusion",
        plotting.plot_confusion_matrix(
            metrics.confusion_matrix(y_test, manual.predict(x_test), num_classes=10),
            title="Manual MLP — test confusion",
            normalize=True,
        ),
    )
    first_layer = manual.weights[0].T.reshape(-1, 28, 28)
    exp.log_figure(
        "first_layer_filters",
        plotting.plot_image_grid(first_layer[:32], nrow=8, title="First-layer weights as images", cmap="RdBu_r"),
    )

    return {
        "gradient/worst_abs_difference": worst,
        "gradient/scale": gradient_scale,
        "gradient/relative": worst / gradient_scale,
        "loss/abs_difference": loss_difference,
        "manual/test_accuracy": manual_accuracy[-1],
        "torch/test_accuracy": torch_accuracy[-1],
        "accuracy_gap": abs(manual_accuracy[-1] - torch_accuracy[-1]),
        "manual/seconds": manual_seconds,
        "torch/seconds": torch_seconds,
    }


main = run

if __name__ == "__main__":
    raise SystemExit(main())
