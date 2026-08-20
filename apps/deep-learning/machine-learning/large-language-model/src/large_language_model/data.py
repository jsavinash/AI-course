"""Data loading and preprocessing for LLM."""

from pathlib import Path

import numpy as np

VOCAB_SIZE = 100
MAX_SEQ_LEN = 32

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    vocab_size: int = VOCAB_SIZE,
    seq_len: int = MAX_SEQ_LEN,
    random_seed: int = 42,
) -> np.ndarray:
    """Generate synthetic token sequences for LLM training.

    Creates sequences where certain patterns are repeated, allowing the
    model to learn next-token prediction.

    Returns:
        X: (n_samples, seq_len) token indices
    """
    rng = np.random.default_rng(random_seed)
    X = rng.integers(0, vocab_size, size=(n_samples, seq_len))

    for i in range(n_samples):
        if i % 3 == 0:
            X[i, -1] = X[i, 0]

    return X


def build_vocab(text: str) -> dict[str, int]:
    """Build a vocabulary mapping from a text string."""
    chars = sorted(set(text))
    return {ch: i for i, ch in enumerate(chars)}


def encode_text(text: str, vocab: dict[str, int], max_len: int = MAX_SEQ_LEN) -> np.ndarray:
    """Encode text into token indices."""
    tokens = [vocab.get(ch, 0) for ch in text[:max_len]]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return np.array(tokens)


def train_test_split(
    X: np.ndarray, test_size: float = 0.2, random_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray]:
    n = len(X)
    n_test = max(1, int(n * test_size))
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(n)
    else:
        indices = np.random.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx]


def save_training_data(X: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X)


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    vocab_size: int = VOCAB_SIZE,
    random_seed: int = 42,
) -> np.ndarray:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"]
    return generate_synthetic_data(n_samples=n_samples, vocab_size=vocab_size, random_seed=random_seed)
