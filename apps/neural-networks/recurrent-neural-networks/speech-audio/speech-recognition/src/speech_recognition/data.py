"""Data loading and preprocessing for speech recognition (RNN).

Generates synthetic audio-like feature sequences (e.g., MFCC-like 16-dim vectors)
labeled with spoken word classes.
"""

from pathlib import Path

import numpy as np

N_FEATURES = 16
SEQ_LEN = 20
N_CLASSES = 10

DEFAULT_N_SAMPLES = 500

WORD_NAMES = [
    "hello",
    "world",
    "yes",
    "no",
    "good",
    "bad",
    "up",
    "down",
    "left",
    "right",
]


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic audio feature sequences and their word labels.

    Each word class has a characteristic feature pattern (mean vector).
    The RNN must learn to classify the sequence of acoustic frames.

    Returns:
        X: (n_samples, SEQ_LEN, N_FEATURES) audio feature sequences
        y: (n_samples,) word class indices
    """
    rng = np.random.default_rng(random_seed)

    # Each class has a characteristic mean feature vector
    class_means = rng.normal(0, 1, size=(N_CLASSES, N_FEATURES))

    X = np.zeros((n_samples, SEQ_LEN, N_FEATURES))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        label = rng.integers(0, N_CLASSES)
        y[i] = label

        # Generate a sequence that gradually reveals the class
        base = class_means[label]
        for t in range(SEQ_LEN):
            # Early frames are noisy, later frames are clearer (as word is pronounced)
            noise_scale = 1.0 - (t / SEQ_LEN) * 0.4
            X[i, t] = base + rng.normal(0, noise_scale, size=N_FEATURES)

        # Normalize each sequence
        X[i] = (X[i] - X[i].mean()) / (X[i].std() + 1e-8)

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
