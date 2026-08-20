"""Data loading and preprocessing for Attention Mechanism."""

from pathlib import Path

import numpy as np

INPUT_DIM = 32
HIDDEN_DIM = 64
OUTPUT_DIM = 32
SEQ_LEN = 16

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    input_dim: int = INPUT_DIM,
    output_dim: int = OUTPUT_DIM,
    seq_len: int = SEQ_LEN,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic sequence data for attention-based modeling.

    Creates input and output sequences for encoder-decoder tasks.

    Returns:
        X: (n_samples, seq_len, input_dim) input sequence
        y: (n_samples, seq_len, output_dim) target sequence
    """
    rng = np.random.default_rng(random_seed)
    X = rng.normal(0, 1, (n_samples, seq_len, input_dim)).astype(np.float32)
    y = rng.normal(0, 1, (n_samples, seq_len, output_dim)).astype(np.float32)

    return X, y


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    input_dim: int = INPUT_DIM,
    output_dim: int = OUTPUT_DIM,
    seq_len: int = SEQ_LEN,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["y"]
    return generate_synthetic_data(
        n_samples=n_samples, input_dim=input_dim, output_dim=output_dim, seq_len=seq_len, random_seed=random_seed
    )


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
