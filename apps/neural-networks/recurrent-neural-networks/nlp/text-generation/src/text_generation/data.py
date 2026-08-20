"""Data loading and preprocessing for text generation (RNN language model).

Generates synthetic character-level sequences for language modeling.
Characters are indexed as integers: a=0, b=1, ..., z=25.
"""

from pathlib import Path

import numpy as np

VOCAB_SIZE = 26
SEQ_LEN = 20

DEFAULT_N_SAMPLES = 500


def _int_to_char(idx: int) -> str:
    """Map integer index to lowercase letter (0->'a', 1->'b', ...)."""
    return chr(ord("a") + idx % 26)


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> np.ndarray:
    """Generate synthetic character-index sequences for language modeling.

    Sequences are generated from simple probabilistic patterns so the RNN
    can learn next-character prediction.

    Returns:
        X: (n_samples, SEQ_LEN) character indices
    """
    rng = np.random.default_rng(random_seed)

    X = np.zeros((n_samples, SEQ_LEN), dtype=int)

    for i in range(n_samples):
        seq = np.zeros(SEQ_LEN, dtype=int)
        seq[0] = rng.integers(0, VOCAB_SIZE)
        for t in range(1, SEQ_LEN):
            # Bias toward repeating or incrementing (patterns the RNN can learn)
            if rng.random() < 0.6:
                seq[t] = (seq[t - 1] + rng.integers(-1, 2)) % VOCAB_SIZE
            else:
                seq[t] = rng.integers(0, VOCAB_SIZE)
        X[i] = seq

    return X


def save_training_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y)


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> np.ndarray:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"]
    return generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)


def train_test_split(
    X: np.ndarray,
    y: np.ndarray | None = None,
    test_size: float = 0.2,
    random_seed: int | None = None,
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

    return (
        X[train_idx],
        X[test_idx],
        (X[train_idx] if y is None else y[train_idx]),
        (X[test_idx] if y is None else y[test_idx]),
    )
