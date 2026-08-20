"""Self-Organizing Map for unsupervised clustering.

Architecture:
    Input (n_features,) -> Competitive layer (grid_height x grid_width neurons)
    Each neuron has a weight vector of dimension n_features.

    Training: For each input, find Best Matching Unit (BMU) and update
    weights of BMU and its neighbors (Gaussian neighborhood function).

Loss: Average quantization error (distance to BMU)
"""

from dataclasses import dataclass, field

import numpy as np


@dataclass
class SelfOrganizingMap:
    """Self-Organizing Map for unsupervised clustering and visualization.

    Produces a low-dimensional (2D grid) discretized representation of the input space.

    Args:
        n_features: Number of input features
        grid_height: Height of the 2D neuron grid
        grid_width: Width of the 2D neuron grid
        learning_rate: Initial learning rate
        n_iterations: Number of training iterations
        sigma: Initial neighborhood radius
        random_seed: Random seed
    """

    n_features: int = 32
    grid_height: int = 5
    grid_width: int = 5
    learning_rate: float = 0.5
    n_iterations: int = 300
    sigma: float = 2.0
    random_seed: int = 42

    weights: np.ndarray | None = None
    loss_history: list[float] = field(default_factory=list)
    training_mode: str = "unsupervised"

    @property
    def n_neurons(self) -> int:
        return self.grid_height * self.grid_width

    def _init_weights(self, X: np.ndarray, rng: np.random.Generator) -> None:
        """Initialize weights by sampling from the data distribution."""
        n = self.n_neurons
        indices = rng.choice(len(X), size=min(n, len(X)), replace=len(X) < n)
        self.weights = X[indices].copy()
        if len(self.weights) < n:
            extra = X[rng.choice(len(X), n - len(self.weights))]
            self.weights = np.vstack([self.weights, extra])

    def _find_bmu(self, x: np.ndarray) -> tuple[int, int]:
        """Find Best Matching Unit (neuron closest to input)."""
        distances = np.sqrt(np.sum((self.weights - x) ** 2, axis=1))
        bmu_idx = np.argmin(distances)
        row = bmu_idx // self.grid_width
        col = bmu_idx % self.grid_width
        return int(row), int(col)

    def _neighborhood(self, bmu_row: int, bmu_col: int, sigma: float) -> np.ndarray:
        """Compute Gaussian neighborhood function for all neurons."""
        neighborhood = np.zeros((self.grid_height, self.grid_width))
        for r in range(self.grid_height):
            for c in range(self.grid_width):
                dist = np.sqrt((r - bmu_row) ** 2 + (c - bmu_col) ** 2)
                neighborhood[r, c] = np.exp(-dist ** 2 / (2 * sigma ** 2 + 1e-8))
        return neighborhood.flatten()

    def fit(
        self,
        X: np.ndarray,
        n_iterations: int | None = None,
    ) -> "SelfOrganizingMap":
        """Train the SOM on input data using batch learning.

        Args:
            X: Input data (n_samples, n_features)
        """
        if self.weights is None:
            rng = np.random.default_rng(self.random_seed)
            self._init_weights(X, rng)

        if n_iterations is None:
            n_iterations = self.n_iterations

        rng = np.random.default_rng(self.random_seed)
        n_samples = X.shape[0]

        for iteration in range(n_iterations):
            t = iteration / n_iterations
            lr = self.learning_rate * (1 - t)
            sigma = self.sigma * (1 - t)

            epoch_loss = 0.0
            for i in range(n_samples):
                x = X[rng.integers(0, n_samples)]
                bmu_row, bmu_col = self._find_bmu(x)
                bmu_idx = bmu_row * self.grid_width + bmu_col

                neighborhood = self._neighborhood(bmu_row, bmu_col, sigma)

                for n in range(self.n_neurons):
                    delta = lr * neighborhood[n] * (x - self.weights[n])
                    self.weights[n] += delta

                epoch_loss += np.sum((x - self.weights[bmu_idx]) ** 2)

            self.loss_history.append(epoch_loss / n_samples)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return quantization error (distance to BMU) for each sample."""
        errors = []
        for x in X:
            _, bmu_idx = self._find_bmu(x)
            errors.append(np.sqrt(np.sum((x - self.weights[bmu_idx]) ** 2)))
        return np.array(errors)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return BMU coordinates (row, col) for each input sample."""
        results = []
        for x in X:
            bmu_row, bmu_col = self._find_bmu(x)
            results.append((bmu_row, bmu_col))
        return np.array(results)

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Return BMU neuron indices for each input sample."""
        indices = []
        for x in X:
            bmu_row, bmu_col = self._find_bmu(x)
            indices.append(bmu_row * self.grid_width + bmu_col)
        return np.array(indices)

    def evaluate(self, X: np.ndarray) -> dict[str, float]:
        errors = self.predict_proba(X)
        return {
            "quantization_error": float(np.mean(errors)),
            "n_samples": float(len(X)),
            "unique_neurons_used": float(len(set(self.transform(X).tolist()))),
            "grid_size": float(self.n_neurons),
        }

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "weights": self.weights,
            "n_features": np.array([self.n_features]),
            "grid_height": np.array([self.grid_height]),
            "grid_width": np.array([self.grid_width]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "sigma": np.array([self.sigma]),
            "random_seed": np.array([self.random_seed]),
        }
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "SelfOrganizingMap":
        data = np.load(path)
        obj = cls(
            n_features=int(data["n_features"].item()),
            grid_height=int(data["grid_height"].item()),
            grid_width=int(data["grid_width"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            sigma=float(data["sigma"].item()),
            random_seed=int(data["random_seed"].item()),
        )
        obj.weights = data["weights"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "grid_height": self.grid_height,
            "grid_width": self.grid_width,
            "n_neurons": self.n_neurons,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "sigma": self.sigma,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
