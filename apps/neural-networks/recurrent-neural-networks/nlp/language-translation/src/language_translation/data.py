"""Data loading and preprocessing for language translation (RNN).

Generates synthetic word-index sequences for a simple translation task.
Two pseudo-languages are used: 'Source' (indices 0-19) and 'Target' (indices 20-39).
The model learns to map source sequences to target translations.
"""

from pathlib import Path

import numpy as np

VOCAB_SIZE = 40
SEQ_LEN = 8

WORD_NAMES = [
    "alpha",
    "beta",
    "gamma",
    "delta",
    "epsilon",
    "zeta",
    "eta",
    "theta",
    "iota",
    "kappa",
    "lambda",
    "mu",
    "nu",
    "xi",
    "omicron",
    "pi",
    "rho",
    "sigma",
    "tau",
    "upsilon",
    "trans1",
    "trans2",
    "trans3",
    "trans4",
    "trans5",
    "trans6",
    "trans7",
    "trans8",
    "trans9",
    "trans10",
    "trans11",
    "trans12",
    "trans13",
    "trans14",
    "trans15",
    "trans16",
    "trans17",
    "trans18",
    "trans19",
    "trans20",
]

DEFAULT_N_SAMPLES = 500


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic word-index sequences and their translations.

    Source words are indices 0-19; target words are indices 20-39.
    Translation rule: source index i maps to target index 20+i
    (with some noise and varying sequence patterns).

    Returns:
        X: (n_samples, SEQ_LEN) source word indices
        y: (n_samples,) target word index (the 'translation' of the sequence)
    """
    rng = np.random.default_rng(random_seed)

    X = np.zeros((n_samples, SEQ_LEN), dtype=int)
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        # Random source sequence (indices 0-19)
        seq = rng.integers(0, 20, size=SEQ_LEN)
        X[i] = seq

        # Translation: use the most common word in the sequence, translated
        counts = np.bincount(seq, minlength=20)
        most_common = int(np.argmax(counts))
        y[i] = 20 + most_common  # translate to target vocabulary

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
