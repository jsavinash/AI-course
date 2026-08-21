"""Autoencoder for dimensionality reduction and denoising.

Architecture:
    Encoder: Input (n_features,) -> Dense (hidden_dim, ReLU) -> Dense (latent_dim, ReLU)
    Decoder: Input (latent_dim) -> Dense (hidden_dim, ReLU) -> Dense (n_features, sigmoid)

Loss: Mean Squared Error (reconstruction loss)
"""

from dataclasses import dataclass, field

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(sig: np.ndarray) -> np.ndarray:
    return sig * (1.0 - sig)


@dataclass
class Autoencoder:
    """Autoencoder for dimensionality reduction and denoising.

    Compresses input data into a lower-dimensional latent code and reconstructs it.
    Used for denoising, anomaly detection, and dimensionality reduction.

    Args:
        n_features: Number of input features
        latent_dim: Dimension of the compressed latent representation
        hidden_dim: Hidden units in encoder/decoder layers
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization
        clip_value: Gradient clipping threshold
        noise_rate: Fraction of input to zero out for denoising (0 = vanilla AE)
        random_seed: Random seed
    """

    n_features: int = 32
    latent_dim: int = 8
    hidden_dim: int = 16
    learning_rate: float = 0.01
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    noise_rate: float = 0.0
    random_seed: int = 42

    _W1: np.ndarray | None = None
    _b1: np.ndarray | None = None
    _W2: np.ndarray | None = None
    _b2: np.ndarray | None = None
    _W3: np.ndarray | None = None
    _b3: np.ndarray | None = None
    _W4: np.ndarray | None = None
    _b4: np.ndarray | None = None

    training_mode: str = "unsupervised"
    loss_history: list[float] = field(default_factory=list)

    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self._W1 = rng.normal(0, np.sqrt(1.0 / self.n_features), (self.n_features, self.hidden_dim))
        self._b1 = np.zeros(self.hidden_dim)
        self._W2 = rng.normal(0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, self.latent_dim))
        self._b2 = np.zeros(self.latent_dim)
        self._W3 = rng.normal(0, np.sqrt(1.0 / self.latent_dim), (self.latent_dim, self.hidden_dim))
        self._b3 = np.zeros(self.hidden_dim)
        self._W4 = rng.normal(0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, self.n_features))
        self._b4 = np.zeros(self.n_features)

    def _forward(self, X: np.ndarray, dropout: bool = True) -> tuple[np.ndarray, dict]:
        h1 = relu(X @ self._W1 + self._b1)
        h2 = relu(h1 @ self._W2 + self._b2)
        h3 = relu(h2 @ self._W3 + self._b3)
        out = sigmoid(h3 @ self._W4 + self._b4)
        cache = {"X": X, "h1": h1, "h2": h2, "h3": h3, "out": out}
        return out, cache

    def fit(
        self,
        X: np.ndarray,
        n_iterations: int | None = None,
    ) -> "Autoencoder":
        """Train the autoencoder on input data.

        Args:
            X: Input data (n_samples, n_features)
        """
        if self._W1 is None:
            self._init_weights()

        if n_iterations is None:
            n_iterations = self.n_iterations

        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_seed)

        for _epoch in range(n_iterations):
            X_shuffled = X[rng.permutation(n_samples)]
            epoch_loss = 0.0

            for i in range(n_samples):
                x_i = X_shuffled[i:i + 1]

                if self.noise_rate > 0:
                    noise_mask = rng.random(x_i.shape) > self.noise_rate
                    x_noisy = x_i * noise_mask
                else:
                    x_noisy = x_i

                out, cache = self._forward(x_noisy)

                loss = np.mean((x_i - out) ** 2)
                epoch_loss += loss

                dout = -2 * (x_i - out) / x_i.size
                dout = dout * sigmoid_derivative(cache["out"])

                dh3 = dout @ self._W4.T * relu_derivative(cache["h3"])
                dW4 = cache["h3"].T @ dout
                db4 = np.sum(dout, axis=0)

                dh2 = dh3 @ self._W3.T * relu_derivative(cache["h2"])
                dW3 = cache["h2"].T @ dh3
                db3 = np.sum(dh3, axis=0)

                dh1 = dh2 @ self._W2.T * relu_derivative(cache["h1"])
                dW2 = cache["h1"].T @ dh2
                db2 = np.sum(dh2, axis=0)

                dW1 = cache["X"].T @ dh1
                db1 = np.sum(dh1, axis=0)

                grads = {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2,
                         "dW3": dW3, "db3": db3, "dW4": dW4, "db4": db4}

                grad_norm = np.sqrt(sum(np.sum(g**2) for g in grads.values()))
                if grad_norm > self.clip_value:
                    scale = self.clip_value / (grad_norm + 1e-8)
                    for k in grads:
                        grads[k] *= scale

                lr = self.learning_rate
                wd = self.weight_decay
                self._W1 -= lr * (grads["dW1"] + wd * self._W1)
                self._b1 -= lr * grads["db1"]
                self._W2 -= lr * (grads["dW2"] + wd * self._W2)
                self._b2 -= lr * grads["db2"]
                self._W3 -= lr * (grads["dW3"] + wd * self._W3)
                self._b3 -= lr * grads["db3"]
                self._W4 -= lr * (grads["dW4"] + wd * self._W4)
                self._b4 -= lr * grads["db4"]

            self.loss_history.append(epoch_loss / n_samples)

        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode input to latent representation."""
        h1 = relu(X @ self._W1 + self._b1)
        return relu(h1 @ self._W2 + self._b2)

    def decode(self, Z: np.ndarray) -> np.ndarray:
        """Decode latent representation to reconstruction."""
        h3 = relu(Z @ self._W3 + self._b3)
        return sigmoid(h3 @ self._W4 + self._b4)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return reconstruction error as anomaly score."""
        out, _ = self._forward(X)
        return np.mean((X - out) ** 2, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return reconstructed data."""
        out, _ = self._forward(X)
        return out

    def evaluate(self, X: np.ndarray) -> dict[str, float]:
        out, _ = self._forward(X)
        mse = float(np.mean((X - out) ** 2))
        return {"reconstruction_error": mse, "n_samples": float(len(X))}

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "W1": self._W1, "b1": self._b1,
            "W2": self._W2, "b2": self._b2,
            "W3": self._W3, "b3": self._b3,
            "W4": self._W4, "b4": self._b4,
            "n_features": np.array([self.n_features]),
            "latent_dim": np.array([self.latent_dim]),
            "hidden_dim": np.array([self.hidden_dim]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "weight_decay": np.array([self.weight_decay]),
            "noise_rate": np.array([self.noise_rate]),
        }
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "Autoencoder":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            n_features=int(data["n_features"].item()),
            latent_dim=int(data["latent_dim"].item()),
            hidden_dim=int(data["hidden_dim"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            weight_decay=float(data["weight_decay"].item()),
            noise_rate=float(data.get("noise_rate", [0.0]).item()),
            random_seed=42,
        )
        obj._init_weights()
        obj._W1 = data["W1"]
        obj._b1 = data["b1"]
        obj._W2 = data["W2"]
        obj._b2 = data["b2"]
        obj._W3 = data["W3"]
        obj._b3 = data["b3"]
        obj._W4 = data["W4"]
        obj._b4 = data["b4"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "latent_dim": self.latent_dim,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "noise_rate": self.noise_rate,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
