"""Generative Adversarial Network (GAN) for image generation.

Architecture:
    Generator:
        Input (latent_dim,) -> Dense (hidden, ReLU) -> Dense (n_pixels, sigmoid) -> Output (1 x 8x8)
    Discriminator:
        Input (n_pixels) -> Dense (hidden, ReLU) -> Dense (1, sigmoid)

Loss: Binary cross-entropy for both generator and discriminator
"""

from dataclasses import dataclass, field

import numpy as np


def sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def sigmoid_derivative(sig: np.ndarray) -> np.ndarray:
    return sig * (1.0 - sig)


def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0, z)


def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(z.dtype)


@dataclass
class GAN:
    """Generative Adversarial Network for image generation.

    Two networks compete: a generator creates synthetic images from random noise,
    and a discriminator learns to distinguish real from fake images.

    Args:
        latent_dim: Dimension of the random noise input to the generator
        img_size: Size of images (square)
        n_channels: Number of image channels
        hidden_dim: Hidden units in generator/discriminator
        learning_rate: Gradient descent step size for both networks
        n_iterations: Number of training iterations
        weight_decay: L2 regularization
        clip_value: Gradient clipping threshold
        random_seed: Random seed
    """

    latent_dim: int = 16
    img_size: int = 8
    n_channels: int = 1
    hidden_dim: int = 32
    learning_rate: float = 0.01
    n_iterations: int = 200
    weight_decay: float = 0.0001
    clip_value: float = 1.0
    random_seed: int = 42

    n_features: int = field(init=False, repr=False)

    # Generator weights
    _gen_W1: np.ndarray | None = None
    _gen_b1: np.ndarray | None = None
    _gen_W2: np.ndarray | None = None
    _gen_b2: np.ndarray | None = None

    # Discriminator weights
    _disc_W1: np.ndarray | None = None
    _disc_b1: np.ndarray | None = None
    _disc_W2: np.ndarray | None = None
    _disc_b2: np.ndarray | None = None

    training_mode: str = "gan"
    loss_history: list[float] = field(default_factory=list)
    _gen_loss_history: list[float] = field(default_factory=list, repr=False)
    _disc_loss_history: list[float] = field(default_factory=list, repr=False)

    def __post_init__(self) -> None:
        self.n_features = self.img_size * self.img_size * self.n_channels

    def _init_weights(self) -> None:
        rng = np.random.default_rng(self.random_seed)

        self._gen_W1 = rng.normal(0, np.sqrt(1.0 / self.latent_dim), (self.latent_dim, self.hidden_dim))
        self._gen_b1 = np.zeros(self.hidden_dim)
        self._gen_W2 = rng.normal(0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, self.n_features))
        self._gen_b2 = np.zeros(self.n_features)

        self._disc_W1 = rng.normal(0, np.sqrt(1.0 / self.n_features), (self.n_features, self.hidden_dim))
        self._disc_b1 = np.zeros(self.hidden_dim)
        self._disc_W2 = rng.normal(0, np.sqrt(1.0 / self.hidden_dim), (self.hidden_dim, 1))
        self._disc_b2 = np.zeros(1)

    def _generator_forward(self, z: np.ndarray) -> tuple[np.ndarray, dict]:
        """Forward pass for generator. Returns generated image and cache."""
        N = z.shape[0]
        h1 = relu(z @ self._gen_W1 + self._gen_b1)
        h1 = np.clip(h1, -10, 10)
        logits = h1 @ self._gen_W2 + self._gen_b2
        img = sigmoid(logits)

        cache = {"z": z, "h1": h1, "logits": logits, "img": img, "N": N}
        return img, cache

    def _generator_backward(self, d_logits: np.ndarray, cache: dict) -> dict:
        """Backward pass for generator. Returns weight gradients."""
        z = cache["z"]
        h1 = cache["h1"]
        N = cache["N"]

        d_h1 = (d_logits @ self._gen_W2.T).reshape(N, -1) * relu_derivative(h1)

        dW_gen2 = h1.T @ d_logits / N
        db_gen2 = np.sum(d_logits, axis=0) / N
        dW_gen1 = z.T @ d_h1 / N
        db_gen1 = np.sum(d_h1, axis=0) / N

        return {"dW_gen1": dW_gen1, "db_gen1": db_gen1, "dW_gen2": dW_gen2, "db_gen2": db_gen2}

    def _discriminator_forward(self, X: np.ndarray) -> tuple[np.ndarray, dict]:
        """Forward pass for discriminator. Returns probability and cache.

        Args:
            X: (N, n_features) input data
        """
        h1 = relu(X @ self._disc_W1 + self._disc_b1)
        h1 = np.clip(h1, -10, 10)
        logits = h1 @ self._disc_W2 + self._disc_b2
        prob = sigmoid(logits)

        cache = {"X": X, "h1": h1, "logits": logits, "prob": prob}
        return prob, cache

    def _discriminator_backward(self, d_logits: np.ndarray, cache: dict) -> dict:
        """Backward pass for discriminator. Returns weight gradients and input gradient."""
        X = cache["X"]
        h1 = cache["h1"]
        N = X.shape[0]

        d_h1 = d_logits @ self._disc_W2.T * relu_derivative(h1)

        dW_disc1 = X.T @ d_h1 / N
        db_disc1 = np.sum(d_h1, axis=0) / N
        dW_disc2 = h1.T @ d_logits / N
        db_disc2 = np.sum(d_logits, axis=0) / N

        dX = d_h1 @ self._disc_W1.T

        return {
            "dW_disc1": dW_disc1, "db_disc1": db_disc1,
            "dW_disc2": dW_disc2, "db_disc2": db_disc2,
            "dX": dX,
        }

    def fit(
        self,
        X_real: np.ndarray,
        n_iterations: int | None = None,
    ) -> "GAN":
        """Train the GAN on real images.

        Args:
            X_real: Real images (n_samples, N_FEATURES) flattened, values in [0, 1]
        """
        if self._gen_W1 is None:
            self._init_weights()

        if n_iterations is None:
            n_iterations = self.n_iterations

        n_samples = X_real.shape[0]
        rng = np.random.default_rng(self.random_seed)
        eps_bce = 1e-12

        for _epoch in range(n_iterations):
            z = rng.normal(0, 1, size=(n_samples, self.latent_dim))

            gen_img, gen_cache = self._generator_forward(z)

            real_prob, real_cache = self._discriminator_forward(X_real)
            fake_prob, fake_cache = self._discriminator_forward(gen_img)

            # Discriminator loss: BCE(real -> 1, fake -> 0)
            disc_loss = -np.mean(
                np.log(real_prob + eps_bce)
                + np.log(1 - fake_prob + eps_bce)
            )

            # Discriminator gradients
            d_real_logits = (real_prob - 1) / n_samples
            d_fake_logits = fake_prob / n_samples

            grad_real = self._discriminator_backward(d_real_logits, real_cache)
            grad_fake = self._discriminator_backward(d_fake_logits, fake_cache)

            # Update discriminator
            self._disc_W1 -= self.learning_rate * (grad_real["dW_disc1"] + grad_fake["dW_disc1"] + self.weight_decay * self._disc_W1)
            self._disc_b1 -= self.learning_rate * (grad_real["db_disc1"] + grad_fake["db_disc1"])
            self._disc_W2 -= self.learning_rate * (grad_real["dW_disc2"] + grad_fake["dW_disc2"] + self.weight_decay * self._disc_W2)
            self._disc_b2 -= self.learning_rate * (grad_real["db_disc2"] + grad_fake["db_disc2"])

            # Generator loss: fool discriminator (fake -> 1)
            gen_loss = -np.mean(np.log(fake_prob + eps_bce))

            # Generator gradients (backprop through discriminator)
            d_fake_logits_g = -(1 - fake_prob) / n_samples
            grad_fake_g = self._discriminator_backward(d_fake_logits_g, fake_cache)
            d_gen_img = grad_fake_g["dX"]
            d_gen_logits = d_gen_img * sigmoid_derivative(sigmoid(gen_cache["logits"]))

            gen_grads = self._generator_backward(d_gen_logits, gen_cache)

            # Gradient clipping for generator
            gen_grad_norm = np.sqrt(
                np.sum(gen_grads["dW_gen1"]**2) + np.sum(gen_grads["dW_gen2"]**2)
            )
            if gen_grad_norm > self.clip_value:
                scale = self.clip_value / (gen_grad_norm + 1e-8)
                for k in gen_grads:
                    gen_grads[k] *= scale

            self._gen_W2 -= self.learning_rate * (gen_grads["dW_gen2"] + self.weight_decay * self._gen_W2)
            self._gen_b2 -= self.learning_rate * gen_grads["db_gen2"]
            self._gen_W1 -= self.learning_rate * (gen_grads["dW_gen1"] + self.weight_decay * self._gen_W1)
            self._gen_b1 -= self.learning_rate * gen_grads["db_gen1"]

            self._gen_loss_history.append(float(gen_loss))
            self._disc_loss_history.append(float(disc_loss))
            self.loss_history.append(float(gen_loss + disc_loss))

        return self

    def generate(self, n_samples: int, random_seed: int | None = None) -> np.ndarray:
        """Generate n_samples synthetic images from random noise.

        Returns:
            (n_samples, N_FEATURES) generated images in [0, 1]
        """
        if self._gen_W1 is None:
            raise ValueError("Model not trained")
        rng = np.random.default_rng(random_seed or self.random_seed)
        z = rng.normal(0, 1, size=(n_samples, self.latent_dim))
        img, _ = self._generator_forward(z)
        return img

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return discriminator probability that input is real."""
        if self._disc_W1 is None:
            raise ValueError("Model not trained")
        prob, _ = self._discriminator_forward(X)
        return prob.flatten()

    def predict(self, z: np.ndarray) -> np.ndarray:
        """Generate images from latent vectors."""
        img, _ = self._generator_forward(z)
        return img

    def evaluate(self, X: np.ndarray) -> dict[str, float]:
        real_prob, _ = self._discriminator_forward(X)
        gen = self.generate(n_samples=len(X))
        fake_prob, _ = self._discriminator_forward(gen)

        return {
            "generator_loss": float(self._gen_loss_history[-1]) if self._gen_loss_history else 0.0,
            "discriminator_loss": float(self._disc_loss_history[-1]) if self._disc_loss_history else 0.0,
            "real_accuracy": float(np.mean(real_prob > 0.5)),
            "fake_accuracy": float(np.mean(fake_prob < 0.5)),
            "n_samples": float(len(X)),
        }

    def save(self, path: str) -> None:
        arrays = {
            "loss_history": np.array(self.loss_history),
            "gen_W1": self._gen_W1, "gen_b1": self._gen_b1,
            "gen_W2": self._gen_W2, "gen_b2": self._gen_b2,
            "disc_W1": self._disc_W1, "disc_b1": self._disc_b1,
            "disc_W2": self._disc_W2, "disc_b2": self._disc_b2,
            "latent_dim": np.array([self.latent_dim]),
            "img_size": np.array([self.img_size]),
            "n_channels": np.array([self.n_channels]),
            "hidden_dim": np.array([self.hidden_dim]),
            "learning_rate": np.array([self.learning_rate]),
            "n_iterations": np.array([self.n_iterations]),
            "weight_decay": np.array([self.weight_decay]),
        }
        np.savez(path, **arrays)

    @classmethod
    def load(cls, path: str) -> "GAN":
        data = np.load(path, allow_pickle=True)
        obj = cls(
            latent_dim=int(data["latent_dim"].item()),
            img_size=int(data["img_size"].item()),
            n_channels=int(data["n_channels"].item()),
            hidden_dim=int(data["hidden_dim"].item()),
            learning_rate=float(data["learning_rate"].item()),
            n_iterations=int(data["n_iterations"].item()),
            weight_decay=float(data["weight_decay"].item()),
            random_seed=42,
        )
        obj.n_features = obj.img_size * obj.img_size * obj.n_channels
        obj._init_weights()
        obj._gen_W1 = data["gen_W1"]
        obj._gen_b1 = data["gen_b1"]
        obj._gen_W2 = data["gen_W2"]
        obj._gen_b2 = data["gen_b2"]
        obj._disc_W1 = data["disc_W1"]
        obj._disc_b1 = data["disc_b1"]
        obj._disc_W2 = data["disc_W2"]
        obj._disc_b2 = data["disc_b2"]
        obj.loss_history = list(data.get("loss_history", [0.0]))
        return obj

    def to_dict(self) -> dict:
        return {
            "latent_dim": self.latent_dim,
            "img_size": self.img_size,
            "n_channels": self.n_channels,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
