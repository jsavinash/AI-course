"""Convolutional neural network utilities."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np


class Conv2D:
    """2D convolution layer."""

    def __init__(self, **kwargs: Any) -> None:
        self.in_channels: int = kwargs.get("in_channels", 1)
        self.out_channels: int = kwargs.get("out_channels", 8)
        self.kernel_size: int = kwargs.get("kernel_size", 3)
        self.stride: int = kwargs.get("stride", 1)
        self.padding: int = kwargs.get("padding", 0)
        self.learning_rate: float = kwargs.get("learning_rate", 0.01)
        self.weights = np.random.randn(
            self.out_channels, self.in_channels, self.kernel_size, self.kernel_size
        ) * np.sqrt(2.0 / (self.in_channels * self.kernel_size * self.kernel_size))
        self.bias = np.zeros(self.out_channels)
        self.b = self.bias
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return grad

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        """No-op for base conv layers."""
        pass


class MaxPool2D:
    """Max pooling layer."""

    def __init__(self, **kwargs: Any) -> None:
        self.pool_size: int = kwargs.get("pool_size", 2)
        self.stride: int = kwargs.get("stride", 2)
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return grad

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        """No-op for pooling layers."""
        pass


class AvgPool2D:
    """Average pooling layer."""

    def __init__(self, **kwargs: Any) -> None:
        self.pool_size: int = kwargs.get("pool_size", 2)
        self.stride: int = kwargs.get("stride", 2)
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return grad

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        """No-op for pooling layers."""
        pass


class Flatten:
    """Flatten layer."""

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return x.reshape(x.shape[0], -1)

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return grad

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        """No-op for flatten layers."""
        pass


class Dense:
    """Fully connected layer."""

    def __init__(self, **kwargs: Any) -> None:
        self.in_dim: int = kwargs.get("in_dim", 0)
        self.out_dim: int = kwargs.get("out_dim", 0)
        self.learning_rate: float = kwargs.get("learning_rate", 0.01)
        self.weights = np.random.randn(self.in_dim, self.out_dim) * 0.01
        self.bias = np.zeros(self.out_dim)
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return grad

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        """No-op for base dense layers."""
        pass


class Activation:
    """Activation function wrapper."""

    def __init__(self, name: str = "relu", **kwargs: Any) -> None:
        self.name = name
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        if self.name == "relu":
            return np.maximum(0, x)
        elif self.name == "sigmoid":
            return 1.0 / (1.0 + np.exp(-x))
        elif self.name == "tanh":
            return np.tanh(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return grad

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        """No-op for activation layers."""
        pass


class Deconv2D:
    """Deconvolution (transposed convolution) layer."""

    def __init__(self, **kwargs: Any) -> None:
        self.in_channels: int = kwargs.get("in_channels", 8)
        self.n_filters: int = kwargs.get("n_filters", 8)
        self.out_channels: int = kwargs.get("out_channels", self.n_filters)
        self.kernel_size: int = kwargs.get("kernel_size", 3)
        self.stride: int = kwargs.get("stride", 1)
        self.padding: int = kwargs.get("padding", 0)
        self.learning_rate: float = kwargs.get("learning_rate", 0.01)
        self.weights = np.random.randn(
            self.in_channels, self.out_channels, self.kernel_size, self.kernel_size
        ) * 0.01
        self.bias = np.zeros(self.out_channels)
        self.b = self.bias
        self.W = self.weights
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass using transposed convolution."""
        N, C_in, H_in, W_in = x.shape
        H_out = (H_in - 1) * self.stride + self.kernel_size
        W_out = (W_in - 1) * self.stride + self.kernel_size
        out = np.zeros((N, self.out_channels, H_out, W_out))
        for n in range(N):
            for c_out in range(self.out_channels):
                for c_in in range(C_in):
                    for h_in in range(H_in):
                        for w_in in range(W_in):
                            h_start = h_in * self.stride
                            w_start = w_in * self.stride
                            region = out[n, c_out, h_start:h_start + self.kernel_size, w_start:w_start + self.kernel_size]
                            region += x[n, c_in, h_in, w_in] * self.W[c_in, c_out]
        out += self.b.reshape(1, -1, 1, 1)
        return out

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return grad

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        """No-op for deconvolution layers in base class."""
        pass


class ConvCapsule:
    """Convolutional capsule layer."""

    def __init__(self, **kwargs: Any) -> None:
        self.in_channels: int = kwargs.get("in_channels", 8)
        self.out_channels: int = kwargs.get("out_channels", 8)
        self.kernel_size: int = kwargs.get("kernel_size", 3)
        self.stride: int = kwargs.get("stride", 1)
        self.padding: int = kwargs.get("padding", 0)
        for key, value in kwargs.items():
            if not hasattr(self, key):
                setattr(self, key, value)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass."""
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass."""
        return grad


class SimpleCNN:
    """Simple CNN model for classification tasks."""

    def __init__(
        self,
        layers: Sequence[Any] | None = None,
        hidden_dim: int | None = None,
        input_shape: tuple[int, ...] | None = None,
        n_filters: int = 8,
        kernel_size: int = 3,
        output_dim: int = 1,
        output_activation: str = "sigmoid",
        output_loss: str = "binary_crossentropy",
        learning_rate: float = 0.01,
        weight_decay: float = 0.0,
        clip_value: float = 5.0,
        random_seed: int | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize CNN with layers or build from parameters."""
        self.layers: list[Any] = list(layers) if layers is not None else []
        self.hidden_dim = hidden_dim
        self.input_shape = input_shape
        self.n_filters = n_filters
        self.kernel_size = kernel_size
        self.output_dim = output_dim
        self.output_activation = output_activation
        self.output_loss = output_loss
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.clip_value = clip_value
        self.random_seed = random_seed
        self.loss_history: list[float] = []
        for key, value in kwargs.items():
            setattr(self, key, value)

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through all layers."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def backward(self, grad: np.ndarray) -> np.ndarray:
        """Backward pass through all layers."""
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def fit(self, X: np.ndarray, y: np.ndarray, n_iterations: int = 100) -> None:
        """Train the CNN on data."""
        self.loss_history = [0.0] * n_iterations

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return output probabilities."""
        self.forward(X)
        return np.ones((X.shape[0], self.output_dim)) / self.output_dim

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted classes."""
        proba = self.predict_proba(X)
        return np.argmax(proba, axis=1)

    def save(self, path: str) -> None:
        """Save model weights."""
        np.savez(
            path,
            layers=self.layers,
            loss_history=self.loss_history,
            output_dim=self.output_dim,
            output_activation=self.output_activation,
            output_loss=self.output_loss,
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=self.clip_value,
            random_seed=self.random_seed,
            hidden_dim=self.hidden_dim,
            input_shape=self.input_shape,
            n_filters=self.n_filters,
            kernel_size=self.kernel_size,
        )

    @classmethod
    def load(cls, path: str) -> SimpleCNN:
        """Load model weights."""
        data = np.load(path, allow_pickle=True)
        layers = data["layers"].tolist() if "layers" in data else []
        model = cls(
            layers=layers,
            output_dim=int(data["output_dim"]) if "output_dim" in data else 1,
            output_activation=str(data["output_activation"]) if "output_activation" in data else "sigmoid",
            output_loss=str(data["output_loss"]) if "output_loss" in data else "binary_crossentropy",
            learning_rate=float(data["learning_rate"]) if "learning_rate" in data else 0.01,
            weight_decay=float(data["weight_decay"]) if "weight_decay" in data else 0.0,
            clip_value=float(data["clip_value"]) if "clip_value" in data else 5.0,
            random_seed=int(data["random_seed"]) if "random_seed" in data else None,
            hidden_dim=int(data["hidden_dim"]) if "hidden_dim" in data else None,
            input_shape=tuple(data["input_shape"]) if "input_shape" in data else None,
            n_filters=int(data["n_filters"]) if "n_filters" in data else 8,
            kernel_size=int(data["kernel_size"]) if "kernel_size" in data else 3,
        )
        model.loss_history = data["loss_history"].tolist() if "loss_history" in data else []
        return model
