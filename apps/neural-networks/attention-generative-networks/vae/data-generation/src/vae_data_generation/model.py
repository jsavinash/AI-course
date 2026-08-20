"""Variational Autoencoder (VAE) for data generation.

Architecture:
    Encoder: Input (n_features,) -> Dense (hidden_dim) -> Dense (2*latent_dim)
    -> Split into mean (mu) and log-variance (log_var)
    Reparameterization: z = mu + exp(0.5 * log_var) * epsilon
    Decoder: Input (latent_dim,) -> Dense (hidden_dim) -> Dense (n_features, sigmoid)

Loss: Reconstruction (MSE) + KL divergence (regularization on latent space)
"""

from dataclasses import dataclass, field

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


def sigmoid_derivative(sig: np.ndarray) -> np.ndarray:
    return sig * (1.0 - sig)


@dataclass
class VAE:
    """Variational Autoencoder for data generation.

    Learns a probabilistic latent space and can generate new data variations.

    Args:
        n_features: Number of input features
        latent_dim: Dimension of the latent space
        hidden_dim: Hidden units in encoder/decoder
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization
        clip_value: Gradient clipping threshold
        random_seed: Random seed
    """

    n_features: int = 32
    latent_dim: int = 16
    hidden_dim: int = 64
    learning_rate: float = 0.01
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    W_enc: np.ndarray | None = None
    b_enc: np.ndarray | None = None
    W_mu: np.ndarray | None = None
    b_mu: np.ndarray | None = None
    W_logvar: np.ndarray | None = None
    b_logvar: np.ndarray | None = None
    W_dec: np.ndarray | None = None
    b_dec: np.ndarray | None = None
    W_out: np.ndarray | None = None
    b_out: np.ndarray | None = None

    dW_enc: np.ndarray | None = None
    db_enc: np.ndarray | None = None
    dW_mu: np.ndarray | None = None
    db_mu: np.ndarray | None = None
    dW_logvar: np.ndarray | None = None
    db_logvar: np.ndarray | None = None
    dW_dec: np.ndarray | None = None
    db_dec: np.ndarray | None = None
    dW_out: np.ndarray | None = None
    db_out: np.ndarray | None = None

    training_mode: str = "unsupervised"
    loss_history: list[float] = field(default_factory=list)
    _recon_loss_history: list[float] = field(default_factory=list, repr=False)
    _kl_loss_history: list[float] = field(default_factory=list, repr=False)

    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)
        self.W_enc = rng.normal(0, np.sqrt(1.0 / self.n_features), (self.n_features, self.hidden_dim))
        self.b_enc = np.zeros(self.hidden_dim)
        self.W_mu = rng.normal(0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, self.latent_dim))
        self.b_mu = np.zeros(self.latent_dim)
        self.W_logvar = rng.normal(0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, self.latent_dim))
        self.b_logvar = np.zeros(self.latent_dim)
        self.W_dec = rng.normal(0, np.sqrt(1.0 / self.latent_dim), (self.latent_dim, self.hidden_dim))
        self.b_dec = np.zeros(self.hidden_dim)
        self.W_out = rng.normal(0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, self.n_features))
        self.b_out = np.zeros(self.n_features)

    def _encode(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
        """Forward pass through encoder.

        Returns: mu, log_var, z, z_sample (for output), cache
        """
        h = relu(X @ self.W_enc + self.b_enc)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_logvar + self.b_logvar

        eps = np.random.default_rng(self.random_seed).normal(0, 1, size=mu.shape)
        std = np.exp(0.5 * log_var)
        z = mu + std * eps

        cache = {"X": X, "h": h, "mu": mu, "log_var": log_var, "std": std, "eps": eps, "z": z}
        return mu, log_var, z, cache

    def _decode(self, z: np.ndarray, cache: dict, training: bool = True) -> tuple[np.ndarray, dict]:
        """Forward pass through decoder."""
        h = relu(z @ self.W_dec + self.b_dec)
        x_recon = sigmoid(h @ self.W_out + self.b_out)
        if training:
            cache["dec_h"] = h
            cache["x_recon"] = x_recon
        return x_recon, cache

    def fit(
        self,
        X: np.ndarray,
        n_iterations: int | None = None,
    ) -> "VAE":
        """Train the VAE on input data.

        Args:
            X: Input data (n_samples, n_features)
        """
        if self.W_enc is None:
            self._init_weights()

        if n_iterations is None:
            n_iterations = self.n_iterations

        n_samples = X.shape[0]
        rng = np.random.default_rng(self.random_seed)

        for epoch in range(n_iterations):
            X_shuffled = X[rng.permutation(n_samples)]
            epoch_loss = 0.0

            for i in range(n_samples):
                x_i = X_shuffled[i:i + 1]

                mu, log_var, z, cache = self._encode(x_i)
                x_recon, cache = self._decode(z, cache)

                eps = 1e-8
                recon_loss = np.mean((x_i - x_recon) ** 2)
                kl_loss = -0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))
                total_loss = recon_loss + kl_loss
                epoch_loss += total_loss

                # Backward pass
                dx_recon = -2 * (x_i - x_recon) / x_recon.shape[-1]

                dh_dec = dx_recon * sigmoid_derivative(x_recon)
                self.dW_out = cache["dec_h"].T @ dh_dec
                self.db_out = np.sum(dh_dec, axis=0)
                dh_dec2 = dh_dec @ self.W_out.T * relu_derivative(cache["dec_h"])

                dz = dh_dec2 @ self.W_dec.T

                dmu = dz * cache["std"] * (-1) * cache["eps"] * (-0.5)
                dlog_var = dz * cache["eps"] * 0.5 * cache["std"] * 0.5 * np.exp(0.5 * cache["log_var"])

                self.dW_dec = z.T @ dh_dec2
                self.db_dec = np.sum(dh_dec2, axis=0)

                dh_enc = dmu @ self.W_mu.T + dlog_var @ self.W_logvar.T
                self.dW_mu = cache["h"].T @ dmu
                self.db_mu = np.sum(dmu, axis=0)
                self.dW_logvar = cache["h"].T @ dlog_var
                self.db_logvar = np.sum(dlog_var, axis=0)
                dh_enc2 = dh_enc * relu_derivative(cache["h"])
                self.dW_enc = cache["X"].T @ dh_enc2
                self.db_enc = np.sum(dh_enc2, axis=0)

                # Gradient clipping
                grad_norm = np.sqrt(
                    np.sum(self.dW_enc**2) + np.sum(self.dW_mu**2) + np.sum(self.dW_logvar**2)
                    + np.sum(self.dW_dec**2) + np.sum(self.dW_out**2)
                )
                if grad_norm > self.clip_value:
                    scale = self.clip_value / (grad_norm + 1e-8)
                    self.dW_enc *= scale
                    self.dW_mu *= scale
                    self.dW_logvar *= scale
                    self.dW_dec *= scale
                    self.dW_out *= scale

                # Update
                self.W_enc -= self.learning_rate * (self.dW_enc + self.weight_decay * self.W_enc)
                self.b_enc -= self.learning_rate * self.db_enc
                self.W_mu -= self.learning_rate * (self.dW_mu + self.weight_decay * self.W_mu)
                self.b_mu -= self.learning_rate * self.db_mu
                self.W_logvar -= self.learning_rate * (self.dW_logvar + self.weight_decay * self.W_logvar)
                self.b_logvar -= self.learning_rate * self.db_logvar
                self.W_dec -= self.learning_rate * (self.dW_dec + self.weight_decay * self.W_dec)
                self.b_dec -= self.learning_rate * self.db_dec
                self.W_out -= self.learning_rate * (self.dW_out + self.weight_decay * self.W_out)
                self.b_out -= self.learning_rate * self.db_out

            self.loss_history.append(epoch_loss / n_samples)
            self._recon_loss_history.append(float(recon_loss))
            self._kl_loss_history.append(float(np.sum(-0.5 * (1 + log_var - mu**2 - np.exp(log_var)))))

        return self

    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode input data to latent mean vector."""
        h = relu(X @ self.W_enc + self.b_enc)
        return h @ self.W_mu + self.b_mu

    def decode(self, z: np.ndarray) -> np.ndarray:
        """Decode latent vectors to reconstructed data."""
        h = relu(z @ self.W_dec + self.b_dec)
        return sigmoid(h @ self.W_out + self.b_out)

    def generate(self, n_samples: int, random_seed: int | None = None) -> np.ndarray:
        """Generate new data samples from random latent vectors.

        Returns:
            (n_samples, n_features) generated data
        """
        rng = np.random.default_rng(random_seed or self.random_seed)
        z = rng.normal(0, 1, size=(n_samples, self.latent_dim))
        return self.decode(z)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return reconstruction error as anomaly score."""
        h = relu(X @ self.W_enc + self.b_enc)
        z = h @ self.W_mu + self.b_mu * 0.5
        h_dec = relu(z @ self.W_dec + self.b_dec)
        x_recon = sigmoid(h_dec @ self.W_out + self.b_out)
        return np.mean((X - x_recon) ** 2, axis=1)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Reconstruct input data."""
        h = relu(X @ self.W_enc + self.b_enc)
        mu = h @ self.W_mu + self.b_mu
        h_dec = relu(mu @ self.W_dec + self.b_dec)
        return sigmoid(h_dec @ self.W_out + self.b_out)

    def evaluate(self, X: np.ndarray) -> dict[str, float]:
        h = relu(X @ self.W_enc + self.b_enc)
        mu = h @ self.W_mu + self.b_mu
        log_var = h @ self.W_logvar + self.b_logvar
        z = mu + np.exp(0.5 * log_var) * 0.5
        h_dec = relu(z @ self.W_dec + self.b_dec)
        x_recon = sigmoid(h_dec @ self.W_out + self.b_out)
        recon_loss = float(np.mean((X - x_recon) ** 2))
        kl_loss = float(np.mean(-0.5 * np.sum(1 + log_var - mu**2 - np.exp(log_var))))
        return {
            "reconstruction_loss": recon_loss,
            "kl_divergence": kl_loss,
            "total_loss": recon_loss + kl_loss,
            "n_samples": float(len(X)),
        }

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "W_enc": self.W_enc, "b_enc": self.b_enc,
            "W_mu": self.W_mu, "b_mu": self.b_mu,
            "W_logvar": self.W_logvar, "b_logvar": self.b_logvar,
            "W_dec": self.W_dec, "b_dec": self.b_dec,
            "W_out": self.W_out, "b_out": self.b_out,
            "n_features": np.array([self.n_features]),
            "latent_dim": np.array([self.latent_dim]),
            "hidden_dim": np.array([self.hidden_dim]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "weight_decay": np.array([self.weight_decay]),
        }
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "VAE":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            n_features=int(data["n_features"].item()),
            latent_dim=int(data["latent_dim"].item()),
            hidden_dim=int(data["hidden_dim"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            weight_decay=float(data["weight_decay"].item()),
            random_seed=42,
        )
        obj._init_weights()
        obj.W_enc = data["W_enc"]
        obj.b_enc = data["b_enc"]
        obj.W_mu = data["W_mu"]
        obj.b_mu = data["b_mu"]
        obj.W_logvar = data["W_logvar"]
        obj.b_logvar = data["b_logvar"]
        obj.W_dec = data["W_dec"]
        obj.b_dec = data["b_dec"]
        obj.W_out = data["W_out"]
        obj.b_out = data["b_out"]
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
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
