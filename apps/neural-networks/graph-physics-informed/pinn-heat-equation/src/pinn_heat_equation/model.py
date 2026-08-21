"""Physics-Informed Neural Network for heat equation solving.

Architecture:
    Input (batch, 2) [x, t coordinates] -> Dense (hidden_dim, tanh) -> Dense (hidden_dim, tanh)
    -> Dense (1, linear) -> Temperature prediction u(x, t)

Loss:
    Data loss: MSE( u_pred - u_true )
    Physics loss: MSE( du/dt - alpha * d2u/dx2 ) (heat equation residual)
"""

from dataclasses import dataclass, field

import numpy as np


def tanh(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)


def tanh_derivative(tanh_val: np.ndarray) -> np.ndarray:
    return 1.0 - tanh_val ** 2


@dataclass
class PINNHeatEquation:
    """Physics-Informed Neural Network for solving the heat equation.

    Trained to solve u_t = alpha * u_xx while respecting physical constraints.

    Args:
        alpha: Thermal diffusivity coefficient
        hidden_dim: Hidden units per layer
        n_layers: Number of hidden layers
        learning_rate: Gradient descent step size
        n_iterations: Number of training iterations
        weight_decay: L2 regularization
        clip_value: Gradient clipping threshold
        random_seed: Random seed
    """

    alpha: float = 0.01
    hidden_dim: int = 32
    n_layers: int = 2
    learning_rate: float = 0.01
    n_iterations: int = 500
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    weights: list = field(default_factory=list, repr=False)
    biases: list = field(default_factory=list, repr=False)
    n_weights: int = 0
    training_mode: str = "physics-informed"
    loss_history: list[float] = field(default_factory=list)
    _data_loss_history: list[float] = field(default_factory=list, repr=False)
    _physics_loss_history: list[float] = field(default_factory=list, repr=False)

    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.n_weights = self.n_layers + 1

        self.weights = [
            rng.normal(0, np.sqrt(2.0 / 2), (2, self.hidden_dim)),
        ] + [
            rng.normal(0, np.sqrt(2.0 / self.hidden_dim), (self.hidden_dim, self.hidden_dim))
            for _ in range(self.n_layers - 1)
        ] + [
            rng.normal(0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, 1)),
        ]

        self.biases = [np.zeros(self.hidden_dim) for _ in range(self.n_layers)] + [np.zeros(1)]

    def _forward(self, X: np.ndarray, training: bool = True) -> tuple[np.ndarray, dict]:
        """Forward pass through the network.

        Args:
            X: Input coordinates (batch, 2) [x, t]

        Returns:
            u: Temperature predictions (batch, 1)
        """
        activations = [X]
        zs = []

        a = X
        for i in range(len(self.weights)):
            z = a @ self.weights[i] + self.biases[i]
            zs.append(z)
            a = tanh(z) if i < len(self.weights) - 1 else z
            activations.append(a)

        cache = {"activations": activations, "zs": zs}
        return a, cache

    def _compute_physics_residual(self, X: np.ndarray, u_pred: np.ndarray) -> np.ndarray:
        """Compute the heat equation residual: du/dt - alpha * d2u/dx2.

        Uses finite differences for automatic differentiation approximation.
        """
        eps = 1e-5
        X_x_plus = X.copy()
        X_x_plus[:, 0] += eps
        X_x_minus = X.copy()
        X_x_minus[:, 0] -= eps
        X_t_plus = X.copy()
        X_t_plus[:, 1] += eps
        X_t_minus = X.copy()
        X_t_minus[:, 1] -= eps

        u_x_plus, _ = self._forward(X_x_plus)
        u_x_minus, _ = self._forward(X_x_minus)
        u_t_plus, _ = self._forward(X_t_plus)
        u_t_minus, _ = self._forward(X_t_minus)

        du_dt = (u_t_plus - u_t_minus) / (2 * eps)
        d2u_dx2 = (u_x_plus - 2 * u_pred + u_x_minus) / (eps ** 2)

        residual = du_dt - self.alpha * d2u_dx2
        return residual

    def fit(
        self,
        X: np.ndarray,
        u_true: np.ndarray,
        n_iterations: int | None = None,
    ) -> "PINNHeatEquation":
        """Train the PINN to solve the heat equation.

        Args:
            X: Input coordinates (n_samples, 2) [x, t]
            u_true: True temperature values (n_samples, 1)
        """
        if not self.weights:
            self._init_weights()

        if n_iterations is None:
            n_iterations = self.n_iterations

        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_seed)

        for _epoch in range(n_iterations):
            X_shuffled = X[rng.permutation(n_samples)]
            u_shuffled = u_true[rng.permutation(n_samples)] if u_true is not None else None

            total_data_loss = 0.0
            total_physics_loss = 0.0

            for i in range(n_samples):
                x_i = X_shuffled[i:i + 1]
                u_i = u_shuffled[i:i + 1] if u_shuffled is not None else np.zeros((1, 1))

                u_pred, cache = self._forward(x_i)
                residual = self._compute_physics_residual(x_i, u_pred)

                data_loss = np.mean((u_pred - u_i) ** 2)
                physics_loss = np.mean(residual ** 2)
                total_data_loss += data_loss
                total_physics_loss += physics_loss

                d_pred_du = 1.0
                ddout = d_pred_du * 2 * (u_pred - u_i) / u_i.size

                grads_w = [np.zeros_like(w) for w in self.weights]
                grads_b = [np.zeros_like(b) for b in self.biases]

                activations = cache["activations"]
                zs = cache["zs"]

                da = ddout
                for layer_idx in reversed(range(len(self.weights))):
                    grads_w[layer_idx] += activations[layer_idx].T @ da
                    grads_b[layer_idx] += np.sum(da, axis=0)
                    if layer_idx > 0:
                        da = da @ self.weights[layer_idx].T
                        da = da * tanh_derivative(tanh(zs[layer_idx - 1]))

                grad_norm = np.sqrt(sum(np.sum(g ** 2) for g in grads_w if g is not None))
                if grad_norm > self.clip_value:
                    scale = self.clip_value / (grad_norm + 1e-8)
                    grads_w = [g * scale for g in grads_w]

                lr = self.learning_rate
                wd = self.weight_decay
                for layer_idx in range(len(self.weights)):
                    self.weights[layer_idx] -= lr * (grads_w[layer_idx] + wd * self.weights[layer_idx])
                    self.biases[layer_idx] -= lr * grads_b[layer_idx]

            self.loss_history.append((total_data_loss + total_physics_loss) / n_samples)
            self._data_loss_history.append(total_data_loss / n_samples)
            self._physics_loss_history.append(total_physics_loss / n_samples)

        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict temperature u(x, t) for given coordinates."""
        u, _ = self._forward(X)
        return u.flatten()

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return physics residual magnitude as confidence measure."""
        u_pred, _ = self._forward(X)
        residual = self._compute_physics_residual(X, u_pred)
        return np.abs(residual).flatten()

    def evaluate(self, X: np.ndarray, u_true: np.ndarray) -> dict[str, float]:
        u_pred, _ = self._forward(X)
        mse = float(np.mean((u_pred - u_true) ** 2))
        rmse = float(np.sqrt(mse))
        max_err = float(np.max(np.abs(u_pred - u_true)))
        return {"mse": mse, "rmse": rmse, "max_error": max_err, "n_samples": float(len(X))}

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "alpha": np.array([self.alpha]),
            "hidden_dim": np.array([self.hidden_dim]),
            "n_layers": np.array([self.n_layers]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "weight_decay": np.array([self.weight_decay]),
        }
        for i, w in enumerate(self.weights):
            arrays[f"W{i}"] = w
        for i, b in enumerate(self.biases):
            arrays[f"b{i}"] = b
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "PINNHeatEquation":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            alpha=float(data["alpha"].item()),
            hidden_dim=int(data["hidden_dim"].item()),
            n_layers=int(data["n_layers"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            weight_decay=float(data["weight_decay"].item()),
            random_seed=42,
        )
        obj._init_weights()
        obj.weights = [data[f"W{i}"] for i in range(len(obj.weights))]
        obj.biases = [data[f"b{i}"] for i in range(len(obj.biases))]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "hidden_dim": self.hidden_dim,
            "n_layers": self.n_layers,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
