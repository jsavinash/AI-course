"""Data loading and preprocessing for Transformer-based language modeling.

Generates synthetic token sequences and their next-token-shifted targets.
"""

from pathlib import Path

import numpy as np

VOCAB_SIZE = 100
SEQ_LEN = 16
N_FEATURES = SEQ_LEN

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic token sequences and shifted next-token targets.

    Returns:
        X: (n_samples, SEQ_LEN) input token IDs
        y: (n_samples, SEQ_LEN) target token IDs (shifted by one)
    """
    rng = np.random.default_rng(random_seed)
    X = rng.integers(0, VOCAB_SIZE, size=(n_samples, SEQ_LEN))
    # Target is shifted by one position (last token predicts a random token)
    y = np.roll(X, -1, axis=1)
    y[:, -1] = rng.integers(0, VOCAB_SIZE, size=n_samples)
    return X, y


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["y"]
    return generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n = len(X)
    n_test = max(1, int(n * test_size))
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(n)
    else:
        indices = np.random.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return X[train_idx], X[test_idx], y[train_idx], y[test_idx]


def save_training_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y)
