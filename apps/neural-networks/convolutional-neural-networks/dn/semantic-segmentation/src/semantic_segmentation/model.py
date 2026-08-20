"""Semantic Segmentation using a Deconvolutional Network (DN).

Architecture:
    Input (1 x 8x8) -> Conv2D (8, 3x3, ReLU) -> MaxPool2D (2x2)
    -> Deconv2D (8, 3x3, ReLU) -> Deconv2D (1, 3x3, sigmoid)

Loss: binary cross-entropy (many-to-many: outputs pixel-level reconstruction)
"""

from dataclasses import dataclass, field

import numpy as np
from mlops_shared.cnn import Activation, Deconv2D

from semantic_segmentation.data import reshape_image


@dataclass
class _Conv2D:
    """Minimal conv2d that tracks its own gradients for the DN encoder."""

    n_filters: int = 8
    kernel_size: int = 3
    random_seed: int = 42
    W: np.ndarray | None = None
    b: np.ndarray | None = None
    dW: np.ndarray | None = None
    db: np.ndarray | None = None

    def _init(self, C: int) -> None:
        rng = np.random.default_rng(self.random_seed)
        fan_in = C * self.kernel_size * self.kernel_size
        self.W = rng.normal(0, np.sqrt(2.0 / fan_in), (self.n_filters, C, self.kernel_size, self.kernel_size))
        self.b = np.zeros(self.n_filters)

    def forward(self, X: np.ndarray) -> np.ndarray:
        if self.W is None:
            self._init(X.shape[1])
        N, C, H, W = X.shape
        H_out = H - self.kernel_size + 1
        W_out = W - self.kernel_size + 1
        out = np.zeros((N, self.n_filters, H_out, W_out))
        for n in range(N):
            for f in range(self.n_filters):
                for h in range(H_out):
                    for w in range(W_out):
                        region = X[n, :, h:h + self.kernel_size, w:w + self.kernel_size]
                        out[n, f, h, w] = np.sum(region * self.W[f]) + self.b[f]
        self._cache = {"X": X, "H_out": H_out, "W_out": W_out}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        X = c["X"]
        N, C, H, W = X.shape
        H_out, W_out = c["H_out"], c["W_out"]
        dW = np.zeros_like(self.W)
        db = np.zeros(self.n_filters)
        dX = np.zeros_like(X)
        for n in range(N):
            for f in range(self.n_filters):
                for h in range(H_out):
                    for w in range(W_out):
                        val = dout[n, f, h, w]
                        dW[f] += val * X[n, :, h:h + self.kernel_size, w:w + self.kernel_size]
                        db[f] += val
                        dX[n, :, h:h + self.kernel_size, w:w + self.kernel_size] += val * self.W[f]
        self.dW = dW / N
        self.db = db / N
        return dX

    def update_params(self, lr: float, wd: float = 0.0) -> None:
        if self.W is None:
            return
        self.W -= lr * (self.dW + wd * self.W)
        self.b -= lr * self.db


@dataclass
class SemanticSegmentationDN:
    """Deconvolutional network for semantic segmentation.

    Args:
        img_size: Size of input images (square)
        n_channels: Number of input/output channels
        n_filters: Number of filters in conv layers
        kernel_size: Convolution kernel size
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        random_seed: Random seed
    """

    IMG_SIZE: int = 8
    N_CHANNELS: int = 1
    n_filters: int = 8
    kernel_size: int = 3
    learning_rate: float = 0.01
    n_iterations: int = 300
    weight_decay: float = 0.0001
    clip_value: float = 1.0
    random_seed: int = 42

    _layers: list = field(default_factory=list, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)

    def _build(self) -> None:
        # Encoder-decoder: 8x8 -> Conv(3) -> 6x6 -> Conv(3) -> 4x4
        # -> Deconv(3,stride=1) -> 6x6 -> Deconv(3,stride=1) -> 8x8
        self._layers = [
            _Conv2D(n_filters=self.n_filters, kernel_size=self.kernel_size, random_seed=self.random_seed),
            Activation("relu"),
            _Conv2D(n_filters=self.n_filters, kernel_size=self.kernel_size, random_seed=self.random_seed + 1),
            Activation("relu"),
            Deconv2D(n_filters=self.n_filters, kernel_size=self.kernel_size, stride=1, random_seed=self.random_seed + 2),
            Activation("relu"),
            Deconv2D(n_filters=self.N_CHANNELS, kernel_size=self.kernel_size, stride=1, random_seed=self.random_seed + 3),
            Activation("sigmoid"),
        ]

    def fit(self, X: np.ndarray, y: np.ndarray, X_val=None, y_val=None) -> "SemanticSegmentationDN":
        """Train the deconvolutional network.

        Args:
            X: Input images (n_samples, N_FEATURES)
            y: Target images (n_samples, N_FEATURES)

        Returns:
            self
        """
        X_img = reshape_image(X)
        y_img = reshape_image(y)
        self._build()

        N = X_img.shape[0]
        eps = 1e-12

        for epoch in range(self.n_iterations):
            total_loss = 0.0
            for i in range(N):
                out = X_img[i:i + 1]
                for layer in self._layers:
                    out = layer.forward(out)

                target = y_img[i:i + 1]
                if "binary_crossentropy" == "mse":
                    clip_out = np.clip(out, eps, 1 - eps)
                    loss = float(np.mean((target - clip_out) ** 2))
                else:
                    clip_out = np.clip(out, eps, 1 - eps)
                    loss = float(-np.mean(target * np.log(clip_out) + (1 - target) * np.log(1 - clip_out)))
                total_loss += loss

                if "binary_crossentropy" == "mse":
                    dout = 2 * (out - target) / max(out.shape[0], 1)
                else:
                    dout = (out - target) / max(out.shape[0], 1)

                for layer in reversed(self._layers):
                    dout = layer.backward(dout)

                # Gradient clipping
                grad_norm = 0.0
                for layer in self._layers:
                    if hasattr(layer, "dW") and layer.dW is not None:
                        grad_norm += float(np.sum(layer.dW ** 2))
                    if hasattr(layer, "db") and layer.db is not None:
                        grad_norm += float(np.sum(layer.db ** 2))
                grad_norm = np.sqrt(grad_norm)
                if grad_norm > self.clip_value:
                    scale = self.clip_value / (grad_norm + 1e-8)
                    for layer in self._layers:
                        if hasattr(layer, "dW") and layer.dW is not None:
                            layer.dW *= scale
                        if hasattr(layer, "db") and layer.db is not None:
                            layer.db *= scale

                for layer in self._layers:
                    layer.update_params(self.learning_rate, self.weight_decay)

            self.loss_history.append(total_loss / N)
            if epoch > 50 and len(self.loss_history) > 100 and abs(self.loss_history[-1] - self.loss_history[-100]) < 1e-8:
                break

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predictions for a batch of images."""
        X_img = reshape_image(X)
        results = []
        for i in range(X_img.shape[0]):
            out = X_img[i:i + 1]
            for layer in self._layers:
                out = layer.forward(out)
            results.append(out[0])
        return np.array(results)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Alias for predict."""
        return self.predict(X)

    def mse(self, X: np.ndarray, y: np.ndarray) -> float:
        preds = self.predict(X)
        return float(np.mean((preds.flatten() - y.flatten()) ** 2))

    def rmse(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.sqrt(self.mse(X, y)))

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        mse_val = self.mse(X, y)
        return {"mse": mse_val, "rmse": float(np.sqrt(mse_val)), "n_samples": float(X.shape[0])}

    def save(self, path: str) -> None:
        arrays = {"loss_history": np.array(self.loss_history)}
        for i, layer in enumerate(self._layers):
            if isinstance(layer, _Conv2D):
                if layer.W is not None:
                    arrays[f"conv_{i}_W"] = layer.W
                    arrays[f"conv_{i}_b"] = layer.b
            elif isinstance(layer, Deconv2D):
                if layer.W is not None:
                    arrays[f"deconv_{i}_W"] = layer.W
                    arrays[f"deconv_{i}_b"] = layer.b
        arrays["n_filters"] = np.array(self.n_filters)
        arrays["learning_rate"] = np.array(self.learning_rate)
        arrays["n_iterations"] = np.array(self.n_iterations)
        arrays["weight_decay"] = np.array(self.weight_decay)
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "SemanticSegmentationDN":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            n_filters=int(data.get("n_filters", 8)),
            learning_rate=float(data.get("learning_rate", 0.01)),
            n_iterations=int(data.get("n_iterations", 300)),
            weight_decay=float(data.get("weight_decay", 0.0001)),
            random_seed=42,
        )
        obj._build()
        for i, layer in enumerate(obj._layers):
            if isinstance(layer, _Conv2D) and f"conv_{i}_W" in data:
                layer.W = data[f"conv_{i}_W"]
                layer.b = data[f"conv_{i}_b"]
            elif isinstance(layer, Deconv2D) and f"deconv_{i}_W" in data:
                layer.W = data[f"deconv_{i}_W"]
                layer.b = data[f"deconv_{i}_b"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "img_size": self.IMG_SIZE,
            "n_channels": self.N_CHANNELS,
            "n_filters": self.n_filters,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
