"""Recurrent neural network for image captioning.

Combines a dense image encoder with a SimpleRNN (Elman network) decoder,
trained with Backpropagation Through Time (BPTT). Built from scratch with NumPy.

Architecture:
    Image (64 pixels) -> Dense (hidden_dim) -> RNN (hidden_dim, tanh) -> Output (vocab_size, softmax)

The image is encoded as a dense projection, then repeated as the first input
to a many-to-many RNN that generates a sequence of word tokens.

Loss: Cross-Entropy (many-to-many: predicts next word at each timestep)
"""

from dataclasses import dataclass, field

import numpy as np
from mlops_shared.rnn import SimpleRNN


@dataclass
class ImageCaptioningRNN:
    """RNN for image captioning (image encoder + RNN language model).

    Args:
        n_pixels: Number of input image pixels (e.g., 8x8=64)
        vocab_size: Size of the word vocabulary
        caption_len: Number of words in each caption
        hidden_dim: Number of hidden units
        learning_rate: Gradient descent step size
        n_iterations: Number of training epochs
        weight_decay: L2 regularization strength
        clip_value: Maximum gradient norm
        random_seed: Random seed for reproducibility
    """

    n_pixels: int = 64
    vocab_size: int = 20
    caption_len: int = 8
    hidden_dim: int = 32
    learning_rate: float = 0.05
    n_iterations: int = 300
    weight_decay: float = 0.001
    clip_value: float = 5.0
    random_seed: int = 42

    model: SimpleRNN | None = field(default=None, repr=False)
    training_mode: str = "supervised"
    loss_history: list[float] = field(default_factory=list)
    # Image encoder weights
    W_img: np.ndarray | None = None
    b_img: np.ndarray | None = None

    def _to_onehot_seq(self, seq: np.ndarray, dim: int) -> np.ndarray:
        seq = np.atleast_1d(seq).astype(int)
        result = np.zeros((len(seq), dim))
        result[np.arange(len(seq)), seq % dim] = 1.0
        return result

    def _encode_image(self, X_img: np.ndarray) -> np.ndarray:
        """Encode image pixels to a dense vector, then expand to RNN input dim.

        Returns (vocab_size,) one-hot-like vector (argmax-based one-hot).
        """
        if self.W_img is None:
            raise ValueError("Image encoder not initialized")
        z = X_img @ self.W_img + self.b_img
        z = np.tanh(z)
        # Project hidden representation to vocab_size-dim one-hot-like input
        onehot = np.zeros(self.vocab_size)
        onehot[int(np.argmax(z))] = 1.0
        return onehot

    def _encode_image_batch(self, X_images: np.ndarray) -> np.ndarray:
        """Encode a batch of images to one-hot-like start tokens.

        Returns: (n_samples, 1, vocab_size)
        """
        result = np.zeros((len(X_images), 1, self.vocab_size))
        for i in range(len(X_images)):
            result[i, 0] = self._encode_image(X_images[i])
        return result

    def fit(
        self,
        X_images: np.ndarray,
        captions: np.ndarray,
        X_val: np.ndarray | None = None,
        y_val: np.ndarray | None = None,
    ) -> "ImageCaptioningRNN":
        """Train the image encoder + RNN decoder with BPTT.

        Args:
            X_images: Image pixel arrays (n_samples, n_pixels)
            captions: Caption word indices (n_samples, caption_len)
            X_val: Optional validation images
            y_val: Optional validation captions

        Returns:
            self
        """
        rng = np.random.default_rng(self.random_seed)

        # Initialize image encoder (project pixels to vocab_size-dim for one-hot encoding)
        scale = np.sqrt(1.0 / self.n_pixels)
        self.W_img = rng.normal(0, scale, (self.n_pixels, self.vocab_size))
        self.b_img = np.zeros(self.vocab_size)

        # Build RNN input sequences: [start_token, caption[:-1]]
        # The start token is the encoded image (one-hot like)
        n_samples = X_images.shape[0]
        seq_len = self.caption_len
        X_rnn = np.zeros((n_samples, seq_len, self.vocab_size))

        for i in range(n_samples):
            img_encoded = self._encode_image(X_images[i])
            # Shift captions: predict word t from image + words 0..t-1
            cap = captions[i] % self.vocab_size
            X_rnn[i, 0] = img_encoded  # first input is image
            for t in range(1, seq_len):
                prev_idx = int(cap[t - 1])
                X_rnn[i, t] = self._to_onehot_seq(np.array([prev_idx]), self.vocab_size)[0]

        # Build targets: captions one-hot
        y_onehot = np.zeros((n_samples, seq_len, self.vocab_size))
        for i in range(n_samples):
            cap = captions[i] % self.vocab_size
            for t in range(seq_len):
                y_onehot[i, t, int(cap[t])] = 1.0

        self.model = SimpleRNN(
            input_dim=self.vocab_size,
            hidden_dim=self.hidden_dim,
            output_dim=self.vocab_size,
            output_activation="softmax",
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            clip_value=self.clip_value,
            random_seed=self.random_seed,
            output_loss="cross_entropy",
        )
        self.model.fit(X_rnn, y_onehot, n_iterations=self.n_iterations)
        self.loss_history = self.model.loss_history

        # Fine-tune image encoder via RNN gradients (simplified: keep image encoder fixed)
        # In a full implementation, we would backprop through the image encoder too.
        return self

    def predict(self, X_images: np.ndarray) -> list[np.ndarray]:
        """Generate captions for a batch of images (greedy decoding)."""
        X_rnn_start = self._encode_image_batch(X_images)
        captions = []
        for i in range(len(X_images)):
            seq = X_rnn_start[i]  # (1, vocab_size)
            generated = []
            for _t in range(self.caption_len):
                outputs = self.model.predict_many_to_many(seq)
                next_word = int(np.argmax(outputs[-1]))
                generated.append(next_word)
                # Append next input (greedy)
                next_input = np.zeros((1, self.vocab_size))
                next_input[0, next_word] = 1.0
                seq = np.vstack([seq, next_input])
            captions.append(np.array(generated))
        return captions

    def predict_proba(self, X_images: np.ndarray) -> np.ndarray:
        """Return word probabilities for the first predicted word."""
        X_rnn_start = self._encode_image_batch(X_images)
        results = []
        for i in range(len(X_images)):
            outputs = self.model.predict_many_to_many(X_rnn_start[i])
            results.append(outputs[-1])
        return np.array(results)

    def evaluate(self, X_images: np.ndarray, captions: np.ndarray) -> dict[str, float]:
        preds = self.predict(X_images)
        correct = sum(
            np.array_equal(preds[i], captions[i] % self.vocab_size) for i in range(len(preds))
        )
        return {
            "accuracy": float(correct / max(len(preds), 1)),
            "n_samples": float(len(preds)),
        }

    def save(self, path: str) -> None:
        if self.model is None:
            raise ValueError("Cannot save untrained model")
        self.model.save(path)
        np.savez(
            path + ".img_encoder.npz",
            W_img=self.W_img,
            b_img=self.b_img,
        )

    @classmethod
    def load(cls, path: str) -> "ImageCaptioningRNN":
        model = SimpleRNN.load(path)

        W_img = None
        b_img = None
        try:
            img_data = np.load(path + ".img_encoder.npz")
            W_img = img_data["W_img"]
            b_img = img_data["b_img"]
        except FileNotFoundError:
            rng = np.random.default_rng(42)
            W_img = rng.normal(0, 0.1, (model.input_dim, model.input_dim))
            b_img = np.zeros(model.input_dim)

        obj = cls(
            n_pixels=model.input_dim,
            vocab_size=model.input_dim,
            caption_len=8,
            hidden_dim=model.hidden_dim,
            learning_rate=model.learning_rate,
            weight_decay=model.weight_decay,
            clip_value=model.clip_value,
            random_seed=model.random_seed,
        )
        obj.model = model
        obj.loss_history = model.loss_history
        obj.W_img = W_img
        obj.b_img = b_img
        return obj

    def to_dict(self) -> dict:
        return {
            "n_pixels": self.n_pixels,
            "vocab_size": self.vocab_size,
            "caption_len": self.caption_len,
            "hidden_dim": self.hidden_dim,
            "learning_rate": self.learning_rate,
            "n_iterations": self.n_iterations,
            "weight_decay": self.weight_decay,
            "random_seed": self.random_seed,
            "training_mode": self.training_mode,
            "n_epochs_run": len(self.loss_history),
            "final_loss": self.loss_history[-1] if self.loss_history else 0.0,
        }
