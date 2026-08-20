"""Deep Belief Network for unsupervised pre-training.

Architecture:
    Stack of Restricted Boltzmann Machines (RBMs), each pre-trained greedily
    using Contrastive Divergence. The stack forms a deep generative model.

    RBM_i: visible=n_features (or hidden_dim from layer above) -> hidden=hidden_dim

Loss: Reconstruction error (cross-entropy for binary data)
"""

from dataclasses import dataclass, field

import numpy as np


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def _bernoulli_sample(probs: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(probs.shape) < probs).astype(float)


@dataclass
class RBM:
    """Restricted Boltzmann Machine with binary visible and hidden units.

    Args:
        n_visible: Number of visible units
        n_hidden: Number of hidden units
        learning_rate: Learning rate for weight updates
        n_cd_steps: Number of Contrastive Divergence steps (CD-k)
        weight_decay: L2 regularization
        random_seed: Random seed
    """

    n_visible: int = 32
    n_hidden: int = 16
    learning_rate: float = 0.01
    n_cd_steps: int = 1
    weight_decay: float = 0.001
    random_seed: int = 42

    W: np.ndarray | None = None
    b: np.ndarray | None = None
    c: np.ndarray | None = None
    dW: np.ndarray | None = None
    db: np.ndarray | None = None
    dc: np.ndarray | None = None

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.W = rng.normal(0, 0.01, (self.n_visible, self.n_hidden))
        self.b = np.zeros(self.n_visible)
        self.c = np.zeros(self.n_hidden)

    def _sample_h(self, v: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        probs = _sigmoid(v @ self.W + self.c)
        samples = _bernoulli_sample(probs, rng)
        return probs, samples

    def _sample_v(self, h: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        probs = _sigmoid(h @ self.W.T + self.b)
        samples = _bernoulli_sample(probs, rng)
        return probs, samples

    def fit(self, X: np.ndarray, n_epochs: int = 50) -> "RBM":
        """Train the RBM using Contrastive Divergence.

        Args:
            X: Binary input data (n_samples, n_visible) in [0, 1]
        """
        if self.W is None:
            self.init_weights()

        rng = np.random.default_rng(self.random_seed)
        n_samples = X.shape[0]

        for epoch in range(n_epochs):
            X_shuffled = X[rng.permutation(n_samples)]
            for i in range(n_samples):
                v = X_shuffled[i:i + 1]

                h_prob, h_sample = self._sample_h(v, rng)

                for _ in range(self.n_cd_steps - 1):
                    v_prob, v_sample = self._sample_v(h_sample, rng)
                    h_prob, h_sample = self._sample_h(v_prob, rng)

                v_k_prob, v_k_sample = self._sample_v(h_sample, rng)
                h_k_prob, _ = self._sample_h(v_k_sample, rng)

                self.dW = np.outer(v[0], h_prob[0]) - np.outer(v_k_prob[0], h_k_prob[0])
                self.db = v[0] - v_k_prob[0]
                self.dc = h_prob[0] - h_k_prob[0]

                self.W -= self.learning_rate * (self.dW + self.weight_decay * self.W)
                self.b -= self.learning_rate * self.db
                self.c -= self.learning_rate * self.dc

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode input data to hidden representation."""
        return _sigmoid(X @ self.W + self.c)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input through the RBM."""
        h = _sigmoid(X @ self.W + self.c)
        return _sigmoid(h @ self.W.T + self.b)

    def reconstruction_error(self, X: np.ndarray) -> float:
        recon = self.reconstruct(X)
        return float(np.mean((X - recon) ** 2))

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self.reconstruct(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self.transform(X)

    def evaluate(self, X: np.ndarray) -> dict[str, float]:
        return {"reconstruction_error": self.reconstruction_error(X), "n_samples": float(len(X))}

    def save(self, path: str) -> None:
        arrays = {
            "W": self.W, "b": self.b, "c": self.c,
            "n_visible": np.array([self.n_visible]),
            "n_hidden": np.array([self.n_hidden]),
        }
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "RBM":
        data = np.load(path)
        obj = cls(
            n_visible=int(data["n_visible"].item()),
            n_hidden=int(data["n_hidden"].item()),
        )
        obj.W = data["W"]
        obj.b = data["b"]
        obj.c = data["c"]
        return obj


@dataclass
class DeepBeliefNetwork:
    """Deep Belief Network - stack of RBMs for unsupervised pre-training.

    Each RBM layer is greedily pre-trained, then the network can be used for
    feature extraction, dimensionality reduction, or fine-tuned with backprop.

    Args:
        n_features: Number of input features
        hidden_dims: List of hidden dimensions for each RBM layer
        learning_rate: Learning rate for RBM training
        n_cd_steps: Number of CD steps per RBM
        n_epochs: Epochs per RBM layer
        weight_decay: L2 regularization
        random_seed: Random seed
    """

    n_features: int = 32
    hidden_dims: list[int] = field(default_factory=lambda: [16, 8])
    learning_rate: float = 0.01
    n_cd_steps: int = 1
    n_epochs: int = 50
    weight_decay: float = 0.001
    random_seed: int = 42

    layers: list = field(default_factory=list, repr=False)
    loss_history: list[float] = field(default_factory=list)
    training_mode: str = "unsupervised"

    def fit(self, X: np.ndarray, n_epochs: int | None = None) -> "DeepBeliefNetwork":
        """Greedy layer-wise pre-training of stacked RBMs.

        Args:
            X: Input data (n_samples, n_features)
        """
        if n_epochs is None:
            n_epochs = self.n_epochs

        data = X.copy()
        self.layers = []

        for layer_idx, hidden_dim in enumerate(self.hidden_dims):
            rbm = RBM(
                n_visible=data.shape[1],
                n_hidden=hidden_dim,
                learning_rate=self.learning_rate,
                n_cd_steps=self.n_cd_steps,
                weight_decay=self.weight_decay,
                random_seed=self.random_seed + layer_idx * 100,
            )
            rbm.fit(data, n_epochs=n_epochs)
            self.layers.append(rbm)

            error = rbm.reconstruction_error(data)
            self.loss_history.append(error)
            data = rbm.transform(data)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode input through all RBM layers to get latent representation."""
        data = X.copy()
        for rbm in self.layers:
            data = rbm.transform(data)
        return data

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input through the deep network."""
        encoded = self.transform(X)
        data = encoded
        for rbm in reversed(self.layers):
            data = rbm.reconstruct(data.T).T if data.ndim == 1 else rbm.reconstruct(data)
        return data

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return reconstruction error as anomaly score."""
        recon = self.reconstruct(X)
        return np.mean((X - recon) ** 2, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return latent representation."""
        return self.transform(X)

    def evaluate(self, X: np.ndarray) -> dict[str, float]:
        recon = self.reconstruct(X)
        mse = float(np.mean((X - recon) ** 2))
        return {"reconstruction_error": mse, "n_samples": float(len(X))}

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "n_features": np.array([self.n_features]),
            "n_layers": np.array([len(self.layers)]),
            "hidden_dims": np.array(self.hidden_dims),
            "learning_rate": np.array([self.learning_rate]),
            "n_epochs": np.array([self.n_epochs]),
            "weight_decay": np.array([self.weight_decay]),
            "random_seed": np.array([self.random_seed]),
        }
        for i, rbm in enumerate(self.layers):
            arrays[f"rbm_{i}_W"] = rbm.W
            arrays[f"rbm_{i}_b"] = rbm.b
            arrays[f"rbm_{i}_c"] = rbm.c
            arrays[f"rbm_{i}_n_visible"] = np.array([rbm.n_visible])
            arrays[f"rbm_{i}_n_hidden"] = np.array([rbm.n_hidden])
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "DeepBeliefNetwork":
        data = np.load(path)
        hidden_dims = list(data["hidden_dims"])
        obj = cls(
            n_features=int(data["n_features"].item()),
            hidden_dims=hidden_dims,
            learning_rate=float(data["learning_rate"].item()),
            n_epochs=int(data["n_epochs"].item()),
            weight_decay=float(data["weight_decay"].item()),
            random_seed=42,
        )
        n_layers = int(data["n_layers"].item())
        obj.layers = []
        for i in range(n_layers):
            rbm = RBM(
                n_visible=int(data[f"rbm_{i}_n_visible"].item()),
                n_hidden=int(data[f"rbm_{i}_n_hidden"].item()),
                random_seed=42,
            )
            rbm.W = data[f"rbm_{i}_W"]
            rbm.b = data[f"rbm_{i}_b"]
            rbm.c = data[f"rbm_{i}_c"]
            obj.layers.append(rbm)
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "hidden_dims": self.hidden_dims,
            "learning_rate": self.learning_rate,
            "n_epochs": self.n_epochs,
            "weight_decay": self.weight_decay,
            "training_mode": self.training_mode,
            "n_layers": len(self.layers),
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
