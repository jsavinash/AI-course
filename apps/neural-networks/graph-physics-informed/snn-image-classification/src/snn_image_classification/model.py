"""SNN model for image classification using spiking neurons.

Architecture:
    Input (batch, n_features) -> Linear (input_dim -> hidden_dim) -> LIF Neuron
    -> Linear (hidden_dim -> hidden_dim) -> LIF Neuron
    -> Linear (hidden_dim -> n_classes) -> Output

    Uses Leaky Integrate-and-Fire (LIF) neurons that communicate via discrete spikes.
    Neurons accumulate membrane potential; when threshold is reached, they fire (spike).

    Input encoding: rate coding (pixel intensity -> spike probability)
    Training: surrogate gradient descent through spike generation
"""

from dataclasses import dataclass, field

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -20, 20)))


def sigmoid_derivative(sig_val: np.ndarray) -> np.ndarray:
    return sig_val * (1.0 - sig_val)


def softmax(z: np.ndarray) -> np.ndarray:
    z_shifted = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


@dataclass
class LIFNeuron:
    """Leaky Integrate-and-Fire neuron layer.

    Membrane dynamics:
        tau_m * dv/dt = -(v - v_rest) + R * I
        If v >= v_threshold: fire spike, v <- v_reset

    Args:
        n_neurons: number of neurons
        n_inputs: input dimension
        threshold: spike threshold
        reset_voltage: voltage after spike
        leak_rate: leak coefficient (tau_m inverse)
        v_rest: resting potential
        random_seed: random seed
    """

    n_neurons: int = 64
    n_inputs: int = 64
    threshold: float = 1.0
    reset_voltage: float = 0.0
    leak_rate: float = 0.9
    v_rest: float = 0.0
    random_seed: int = 42

    W: np.ndarray | None = None
    b: np.ndarray | None = None
    dW: np.ndarray | None = None
    db: np.ndarray | None = None

    _cache: dict = field(default_factory=dict, repr=False)

    def init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.W = rng.normal(0, np.sqrt(2.0 / self.n_inputs), (self.n_inputs, self.n_neurons))
        self.b = np.zeros(self.n_neurons)

    def forward(self, x: np.ndarray, n_timesteps: int = 10) -> np.ndarray:
        """Forward pass over multiple timesteps (temporal coding).

        Args:
            x: Input spike trains (batch, n_inputs) encoded as rates
            n_timesteps: number of simulation steps

        Returns:
            spikes: spike trains (batch, n_neurons, n_timesteps)
        """
        if self.W is None:
            self.init_weights()

        batch_size = x.shape[0]
        spike_trains = np.zeros((batch_size, self.n_neurons, n_timesteps))

        membrane = np.full((batch_size, self.n_neurons), self.v_rest)

        for t in range(n_timesteps):
            I_in = x @ self.W + self.b
            membrane = self.leak_rate * membrane + (1 - self.leak_rate) * (self.v_rest + I_in)

            new_spikes = (membrane >= self.threshold).astype(np.float32)
            membrane = np.where(new_spikes > 0, self.reset_voltage, membrane)

            spike_trains[:, :, t] = new_spikes

        self._cache = {"x": x, "spike_trains": spike_trains, "membrane_trace": membrane}
        return spike_trains

    def backward(self, dout: np.ndarray) -> np.ndarray:
        """Backward pass using surrogate gradient.

        Args:
            dout: gradient from next layer (batch, n_neurons, n_timesteps)

        Returns:
            gradient w.r.t. input x (batch, n_inputs)
        """
        c = self._cache
        x = c["x"]
        spike_trains = c["spike_trains"]

        batch_size = x.shape[0]
        n_timesteps = spike_trains.shape[2]

        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)
        grad_membrane = np.zeros((batch_size, self.n_neurons))

        for t in range(n_timesteps):
            spikes_t = spike_trains[:, :, t]
            surrogate = sigmoid_derivative(spikes_t)
            grad_out_t = dout[:, :, t]
            grad_membrane += grad_out_t

            grad_spikes = grad_out_t * surrogate
            self.dW += x.T @ grad_spikes
            self.db += np.sum(grad_spikes, axis=0)

        self.dW /= n_timesteps
        self.db /= n_timesteps

        dx = grad_membrane @ self.W.T
        return dx

    def update_params(self, lr: float, weight_decay: float = 0.0) -> None:
        if self.W is None:
            return
        self.W -= lr * (self.dW + weight_decay * self.W)
        self.b -= lr * self.db


@dataclass
class SNNImageClassification:
    """Spiking Neural Network for image classification.

    Uses Leaky Integrate-and-Fire (LIF) neurons that communicate via discrete spikes,
    closely mimicking biological brain activity.

    Args:
        n_features: Number of input features (e.g., flattened 8x8 image = 64)
        n_classes: Number of output classes
        hidden_dim: Hidden dimension for LIF layers
        learning_rate: Gradient descent step size
        n_iterations: Number of training iterations
        n_timesteps: Number of temporal simulation steps per forward pass
        weight_decay: L2 regularization
        threshold: Spike threshold for LIF neurons
        leak_rate: Decay rate for membrane potential
        clip_value: Gradient clipping threshold
        random_seed: Random seed
    """

    n_features: int = 64
    n_classes: int = 10
    hidden_dim: int = 128
    learning_rate: float = 0.01
    n_iterations: int = 200
    n_timesteps: int = 10
    weight_decay: float = 0.0001
    threshold: float = 1.0
    leak_rate: float = 0.9
    clip_value: float = 5.0
    random_seed: int = 42

    layers: list = field(default_factory=list, repr=False)
    W_out: np.ndarray | None = None
    b_out: np.ndarray | None = None
    training_mode: str = "spiking"
    loss_history: list[float] = field(default_factory=list)

    def _build(self) -> None:
        rng = np.random.default_rng(self.random_seed + 200)
        self.layers = [
            LIFNeuron(
                n_neurons=self.hidden_dim,
                n_inputs=self.n_features,
                threshold=self.threshold,
                leak_rate=self.leak_rate,
                random_seed=self.random_seed,
            ),
            LIFNeuron(
                n_neurons=self.hidden_dim,
                n_inputs=self.hidden_dim,
                threshold=self.threshold,
                leak_rate=self.leak_rate,
                random_seed=self.random_seed + 1,
            ),
        ]
        self.W_out = rng.normal(0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, self.n_classes))
        self.b_out = np.zeros(self.n_classes)

    def _forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Forward pass through SNN.

        Args:
            X: Input features (batch, n_features)

        Returns:
            logits: output logits (batch, n_classes)
            cache: intermediate values
        """
        x = X

        lif1: LIFNeuron = self.layers[0]
        spikes1 = lif1.forward(x, n_timesteps=self.n_timesteps)
        pooled1 = np.mean(spikes1, axis=2)

        lif2: LIFNeuron = self.layers[1]
        spikes2 = lif2.forward(pooled1, n_timesteps=self.n_timesteps)
        pooled2 = np.mean(spikes2, axis=2)

        logits = pooled2 @ self.W_out + self.b_out
        cache = {"x": x, "pooled1": pooled1, "pooled2": pooled2, "spikes1": spikes1, "spikes2": spikes2}
        return logits, cache

    def fit(self, X: np.ndarray, y: np.ndarray, n_iterations: int | None = None) -> "SNNImageClassification":
        """Train the SNN using surrogate gradient descent.

        Args:
            X: Input features (n_samples, n_features)
            y: Labels (n_samples,)
        """
        if not self.layers:
            self._build()

        if n_iterations is None:
            n_iterations = self.n_iterations

        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_seed)
        eps = 1e-12

        y_onehot = np.zeros((n_samples, self.n_classes))
        y_onehot[np.arange(n_samples), y.astype(int)] = 1.0

        for _epoch in range(n_iterations):
            perm = rng.permutation(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y_onehot[perm]

            epoch_loss = 0.0
            for i in range(n_samples):
                x_i = X_shuffled[i:i + 1]
                y_i = y_shuffled[i:i + 1]

                logits, cache = self._forward(x_i)
                probs = softmax(logits)
                loss = -np.sum(y_i * np.log(np.clip(probs, eps, 1)))
                epoch_loss += loss

                dlogits = (probs - y_i) / 1.0
                dW_out = cache["pooled2"].T @ dlogits
                db_out = np.sum(dlogits, axis=0)

                dpooled2 = dlogits @ self.W_out.T
                dout2 = np.zeros_like(cache["spikes2"])
                dout2[:, :, :] = dpooled2[:, :, np.newaxis] / self.n_timesteps

                dh1 = self.layers[1].backward(dout2)
                dph1 = np.mean(dh1, axis=2) if dh1.ndim > 2 else dh1
                dout1 = np.zeros_like(cache["spikes1"])
                dout1[:, :, :] = dph1[:, :, np.newaxis] / self.n_timesteps

                _ = self.layers[0].backward(dout1)

                grad_norm = np.sqrt(
                    np.sum(self.layers[0].dW ** 2) + np.sum(self.layers[1].dW ** 2) + np.sum(dW_out ** 2)
                )
                if grad_norm > self.clip_value:
                    scale = self.clip_value / (grad_norm + 1e-8)
                    self.layers[0].dW *= scale
                    self.layers[1].dW *= scale
                    dW_out *= scale

                lr = self.learning_rate
                wd = self.weight_decay
                lif1: LIFNeuron = self.layers[0]
                lif2: LIFNeuron = self.layers[1]
                lif1.W -= lr * (lif1.dW + wd * lif1.W)
                lif1.b -= lr * lif1.db
                lif2.W -= lr * (lif2.dW + wd * lif2.W)
                lif2.b -= lr * lif2.db
                self.W_out -= lr * (dW_out + wd * self.W_out)
                self.b_out -= lr * db_out

            self.loss_history.append(epoch_loss / n_samples)

        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        logits, _ = self._forward(X)
        return softmax(logits)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=-1)

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> dict[str, float]:
        preds = self.predict(X)
        accuracy = float(np.mean(preds == y))
        return {"accuracy": accuracy, "n_samples": float(len(y))}

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "n_features": np.array([self.n_features]),
            "n_classes": np.array([self.n_classes]),
            "hidden_dim": np.array([self.hidden_dim]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "n_timesteps": np.array([self.n_timesteps]),
            "weight_decay": np.array([self.weight_decay]),
            "threshold": np.array([self.threshold]),
            "leak_rate": np.array([self.leak_rate]),
            "lif1_W": self.layers[0].W,
            "lif1_b": self.layers[0].b,
            "lif2_W": self.layers[1].W,
            "lif2_b": self.layers[1].b,
            "W_out": self.W_out,
            "b_out": self.b_out,
        }
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "SNNImageClassification":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            n_features=int(data["n_features"].item()),
            n_classes=int(data["n_classes"].item()),
            hidden_dim=int(data["hidden_dim"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            n_timesteps=int(data["n_timesteps"].item()),
            weight_decay=float(data["weight_decay"].item()),
            threshold=float(data["threshold"].item()),
            leak_rate=float(data["leak_rate"].item()),
            random_seed=42,
        )
        obj._build()
        obj.layers[0].W = data["lif1_W"]
        obj.layers[0].b = data["lif1_b"]
        obj.layers[1].W = data["lif2_W"]
        obj.layers[1].b = data["lif2_b"]
        obj.W_out = data["W_out"]
        obj.b_out = data["b_out"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "n_features": self.n_features,
            "n_classes": self.n_classes,
            "hidden_dim": self.hidden_dim,
            "training_mode": self.training_mode,
            "n_timesteps": self.n_timesteps,
            "threshold": self.threshold,
            "leak_rate": self.leak_rate,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
