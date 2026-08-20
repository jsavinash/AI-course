"""Data loading and preprocessing for sentiment analysis (RNN).

Generates synthetic word-index sequences labeled with positive/negative sentiment.
"""

from pathlib import Path

import numpy as np

VOCAB_SIZE = 50
SEQ_LEN = 10

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic word-index sequences with sentiment labels.

    Positive sequences tend to use 'positive' words (low indices),
    while negative sequences tend to use 'negative' words (high indices).

    Returns:
        X: (n_samples, SEQ_LEN) word indices
        y: (n_samples,) labels — 1=positive, 0=negative
    """
    rng = np.random.default_rng(random_seed)

    n_positive = n_samples // 2
    n_negative = n_samples - n_positive

    X = np.zeros((n_samples, SEQ_LEN), dtype=int)
    y = np.zeros(n_samples, dtype=int)

    # Positive sequences: more words from index range [0, 20)
    # Negative sequences: more words from index range [30, 50)
    p_pos = np.concatenate([np.full(20, 3.0), np.full(30, 2.0)])
    p_pos = p_pos / p_pos.sum()
    p_neg = np.concatenate([np.full(30, 2.0), np.full(20, 3.0)])
    p_neg = p_neg / p_neg.sum()

    for i in range(n_positive):
        seq = rng.choice(50, size=SEQ_LEN, p=p_pos)
        X[i] = seq
        y[i] = 1

    for i in range(n_negative):
        seq = rng.choice(50, size=SEQ_LEN, p=p_neg)
        X[n_positive + i] = seq
        y[n_positive + i] = 0

    # Shuffle
    perm = rng.permutation(n_samples)
    return X[perm], y[perm]


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
