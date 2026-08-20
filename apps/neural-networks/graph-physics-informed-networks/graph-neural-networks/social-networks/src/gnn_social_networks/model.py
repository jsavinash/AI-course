"""Graph Neural Network for social network analysis.

Architecture:
    Graph Convolution: H^{(l+1)} = sigma(A_norm * H^{(l)} * W^{(l)})
    Where A_norm is the normalized adjacency matrix and W is learnable weights.

    Input: node features (n_nodes, n_features) + adjacency matrix (n_nodes, n_nodes)
    -> GCN layer (n_features -> hidden_dim, ReLU)
    -> GCN layer (hidden_dim -> hidden_dim, ReLU)
    -> Dense (hidden_dim -> n_classes, softmax)

Loss: cross-entropy (node classification) or MSE (reconstruction)
"""

from dataclasses import dataclass, field

import numpy as np


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def softmax(z: np.ndarray) -> np.ndarray:
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def normalize_adjacency(A: np.ndarray) -> np.ndarray:
    """Compute normalized adjacency matrix: A_norm = (A + I) * D^{-1/2}"""
    A_eye = A + np.eye(A.shape[0])
    D = np.diag(np.sum(A_eye, axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(np.diag(D) + 1e-8))
    return D_inv_sqrt @ A_eye @ D_inv_sqrt


@dataclass
class GCNLayer:
    """Graph Convolutional Network layer.

    Args:
        input_dim: Input feature dimension
        output_dim: Output feature dimension
        random_seed: Random seed
    """

    input_dim: int = 32
    output_dim: int = 16
    random_seed: int = 42
    W: np.ndarray | None = None
    dW: np.ndarray | None = None
    _cache: dict = field(default_factory=dict, repr=False)

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.W = rng.normal(0, np.sqrt(2.0 / self.input_dim), (self.input_dim, self.output_dim))

    def forward(self, H: np.ndarray, A_norm: np.ndarray) -> np.ndarray:
        """Forward pass.

        Args:
            H: Node features (n_nodes, input_dim)
            A_norm: Normalized adjacency matrix (n_nodes, n_nodes)

        Returns:
            Output features (n_nodes, output_dim)
        """
        if self.W is None:
            self.init_weights()

        out = A_norm @ H @ self.W
        self._cache = {"H": H, "A_norm": A_norm}
        return out

    def backward(self, dout: np.ndarray) -> np.ndarray:
        c = self._cache
        H = c["H"]
        A_norm = c["A_norm"]

        self.dW = H.T @ (A_norm.T @ dout)
        dH = A_norm @ dout @ self.W.T
        return dH

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W is None:
            return
        self.W -= lr * (self.dW + weight_decay * self.W)


@dataclass
class GNNSocialNetworks:
    """Graph Neural Network for social network analysis.

    Uses Graph Convolution to process node features and adjacency relationships.
    Can perform node classification or graph-level prediction.

    Args:
        n_features: Number of input features per node
        n_classes: Number of output classes (for node classification)
        hidden_dim: Hidden dimension for GCN layers
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization
        clip_value: Gradient clipping threshold
        random_seed: Random seed
    """

    n_features: int = 32
    n_classes: int = 2
    hidden_dim: int = 16
    learning_rate: float = 0.05
    n_iterations: int = 200
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    layers: list = field(default_factory=list, repr=False)
    W_out: np.ndarray | None = None
    b_out: np.ndarray | None = None
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)

    def _build(self) -> None:
        rng = np.random.default_rng(self.random_seed + 100)
        self.layers = [
            GCNLayer(input_dim=self.n_features, output_dim=self.hidden_dim, random_seed=self.random_seed),
            "relu",
            GCNLayer(input_dim=self.hidden_dim, output_dim=self.hidden_dim, random_seed=self.random_seed + 1),
            "relu",
        ]
        self.W_out = rng.normal(0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, self.n_classes))
        self.b_out = np.zeros(self.n_classes)

    def _forward(self, X: np.ndarray, A_norm: np.ndarray) -> tuple[np.ndarray, dict]:
        """Forward pass.

        Args:
            X: Node features (n_nodes, n_features)
            A_norm: Normalized adjacency matrix (n_nodes, n_nodes)

        Returns:
            logits: Class logits per node (n_nodes, n_classes)
            cache: intermediate values
        """
        gcn1: GCNLayer = self.layers[0]
        h1 = gcn1.forward(X, A_norm)
        h1 = relu(h1)

        gcn2: GCNLayer = self.layers[2]
        h2 = gcn2.forward(h1, A_norm)
        h2 = relu(h2)

        logits = h2 @ self.W_out + self.b_out

        cache = {"X": X, "h1": h1, "h2": h2, "gcn1_cache": gcn1._cache, "gcn2_cache": gcn2._cache}
        return logits, cache

    def fit(
        self,
        X: np.ndarray,
        A: np.ndarray,
        y: np.ndarray,
        n_iterations: int | None = None,
    ) -> "GNNSocialNetworks":
        """Train the GNN on graph-structured data.

        Args:
            X: Node features (n_nodes, n_features)
            A: Adjacency matrix (n_nodes, n_nodes)
            y: Node labels (n_nodes,)
        """
        if not self.layers:
            self._build()

        if n_iterations is None:
            n_iterations = self.n_iterations

        A_norm = normalize_adjacency(A)
        n_nodes = X.shape[0]
        eps = 1e-12

        y_onehot = np.zeros((n_nodes, self.n_classes))
        y_onehot[np.arange(n_nodes), y.astype(int)] = 1.0

        for _epoch in range(n_iterations):
            logits, cache = self._forward(X, A_norm)

            probs = softmax(logits)
            loss = -np.sum(y_onehot * np.log(np.clip(probs, eps, 1))) / n_nodes
            self.loss_history.append(loss)

            dlogits = (probs - y_onehot) / n_nodes

            dW_out = cache["h2"].T @ dlogits
            db_out = np.sum(dlogits, axis=0)
            dh2 = dlogits @ self.W_out.T * relu_derivative(cache["h2"])

            gcn2: GCNLayer = self.layers[2]
            dh2_pre = gcn2.backward(dh2)
            dh1 = dh2_pre @ gcn2.W.T * relu_derivative(cache["h1"])

            gcn1: GCNLayer = self.layers[0]
            _ = gcn1.backward(dh1)

            grad_norm = np.sqrt(
                np.sum(gcn1.dW**2) + np.sum(gcn2.dW**2) + np.sum(dW_out**2)
            )
            if grad_norm > self.clip_value:
                scale = self.clip_value / (grad_norm + 1e-8)
                gcn1.dW *= scale
                gcn2.dW *= scale
                dW_out *= scale

            lr = self.learning_rate
            wd = self.weight_decay
            gcn1.W -= lr * (gcn1.dW + wd * gcn1.W)
            gcn2.W -= lr * (gcn2.dW + wd * gcn2.W)
            self.W_out -= lr * (dW_out + wd * self.W_out)
            self.b_out -= lr * db_out

        return self

    def predict_proba(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        A_norm = normalize_adjacency(A)
        logits, _ = self._forward(X, A_norm)
        return softmax(logits)

    def predict(self, X: np.ndarray, A: np.ndarray) -> np.ndarray:
        probs = self.predict_proba(X, A)
        return np.argmax(probs, axis=-1)

    def evaluate(self, X: np.ndarray, A: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds = self.predict(X, A)
        accuracy = float(np.mean(preds == y))
        return {"accuracy": accuracy, "n_nodes": float(len(y))}

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "W_out": self.W_out, "b_out": self.b_out,
            "gcn1_W": self.layers[0].W,
            "gcn2_W": self.layers[2].W,
            "n_features": np.array([self.n_features]),
            "n_classes": np.array([self.n_classes]),
            "hidden_dim": np.array([self.hidden_dim]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "weight_decay": np.array([self.weight_decay]),
        }
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "GNNSocialNetworks":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            n_features=int(data["n_features"].item()),
            n_classes=int(data["n_classes"].item()),
            hidden_dim=int(data["hidden_dim"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            weight_decay=float(data["weight_decay"].item()),
            random_seed=42,
        )
        obj._build()
        obj.W_out = data["W_out"]
        obj.b_out = data["b_out"]
        obj.layers[0].W = data["gcn1_W"]
        obj.layers[2].W = data["gcn2_W"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
