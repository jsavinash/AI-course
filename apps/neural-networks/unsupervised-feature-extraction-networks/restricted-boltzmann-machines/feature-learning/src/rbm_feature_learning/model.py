"""Restricted Boltzmann Machine for feature learning.

Architecture:
    Binary visible units (n_features) <-> Hidden units (n_hidden)
    Fully connected undirected bipartite graph

Training: Contrastive Divergence (CD-k)
Loss: Reconstruction cross-entropy
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
    """Restricted Boltzmann Machine for unsupervised feature learning.

    Learns a probability distribution over binary inputs and extracts
    hierarchical features through its hidden representation.

    Args:
        n_features: Number of visible units (input features)
        n_hidden: Number of hidden units
        learning_rate: Learning rate
        n_cd_steps: Contrastive Divergence steps (CD-k)
        n_epochs: Number of training epochs
        weight_decay: L2 regularization
        random_seed: Random seed
    """

    n_features: int = 32
    n_hidden: int = 16
    learning_rate: float = 0.05
    n_cd_steps: int = 1
    n_epochs: int = 100
    weight_decay: float = 0.001
    random_seed: int = 42

    W: np.ndarray | None = None
    b: np.ndarray | None = None
    c: np.ndarray | None = None
    loss_history: list[float] = field(default_factory=list)
    training_mode: str = "unsupervised"

    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.W = rng.normal(0, 0.01, (self.n_features, self.n_hidden))
        self.b = np.zeros(self.n_features)
        self.c = np.zeros(self.n_hidden)

    def _sample_h(self, v: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        probs = _sigmoid(v @ self.W + self.c)
        samples = _bernoulli_sample(probs, rng)
        return probs, samples

    def _sample_v(self, h: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
        probs = _sigmoid(h @ self.W.T + self.b)
        samples = _bernoulli_sample(probs, rng)
        return probs, samples

    def fit(
        self,
        X: np.ndarray,
        n_epochs: int | None = None,
    ) -> "RBM":
        """Train the RBM using Contrastive Divergence.

        Args:
            X: Binary input data (n_samples, n_features) in [0, 1]
        """
        if self.W is None:
            self._init_weights()

        if n_epochs is None:
            n_epochs = self.n_epochs

        rng = np.random.default_rng(self.random_seed)
        n_samples = X.shape[0]

        for epoch in range(n_epochs):
            epoch_loss = 0.0
            X_shuffled = X[rng.permutation(n_samples)]

            for i in range(n_samples):
                v = X_shuffled[i:i + 1]

                h_prob, h_sample = self._sample_h(v, rng)

                for _ in range(self.n_cd_steps - 1):
                    v_prob, v_sample = self._sample_v(h_sample, rng)
                    h_prob, h_sample = self._sample_h(v_prob, rng)

                v_k_prob, v_k_sample = self._sample_v(h_sample, rng)
                h_k_prob, _ = self._sample_h(v_k_sample, rng)

                dW = np.outer(v[0], h_prob[0]) - np.outer(v_k_prob[0], h_k_prob[0])
                db = v[0] - v_k_prob[0]
                dc = h_prob[0] - h_k_prob[0]

                self.W -= self.learning_rate * (dW + self.weight_decay * self.W)
                self.b -= self.learning_rate * db
                self.c -= self.learning_rate * dc

                epoch_loss += np.mean((v[0] - v_k_prob[0]) ** 2)

            self.loss_history.append(epoch_loss / n_samples)

        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Encode input data to hidden representation."""
        return _sigmoid(X @ self.W + self.c)

    def reconstruct(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input through the RBM."""
        h = _sigmoid(X @ self.W + self.c)
        return _sigmoid(h @ self.W.T + self.b)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return hidden feature activations."""
        return self.transform(X)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return hidden representations (binary samples)."""
        h_probs = _sigmoid(X @ self.W + self.c)
        return (h_probs > 0.5).astype(float)

    def evaluate(self, X: np.ndarray) -> dict[str, float]:
        recon = self.reconstruct(X)
        mse = float(np.mean((X - recon) ** 2))
        return {"reconstruction_error": mse, "n_samples": float(len(X))}

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "W": self.W, "b": self.b, "c": self.c,
            "n_features": np.array([self.n_features]),
            "n_hidden": np.array([self.n_hidden]),
            "learning_rate": np.array([self.learning_rate]),
            "n_cd_steps": np.array([self.n_cd_steps]),
            "n_epochs": np.array([self.n_epochs]),
            "weight_decay": np.array([self.weight_decay]),
            "random_seed": np.array([self.random_seed]),
        }
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "RBM":
        data = np.load(path)
        obj = cls(
            n_features=int(data["n_features"].item()),
            n_hidden=int(data["n_hidden"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_cd_steps=int(data["n_cd_steps"].item()),
            n_epochs=int(data["n_epochs"].item()),
            weight_decay=float(data["weight_decay"].item()),
            random_seed=int(data["random_seed"].item()),
        )
        obj.W = data["W"]
        obj.b = data["b"]
        obj.c = data["c"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "n_hidden": self.n_hidden,
            "learning_rate": self.learning_rate,
            "n_cd_steps": self.n_cd_steps,
            "n_epochs": self.n_epochs,
            "weight_decay": self.weight_decay,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
