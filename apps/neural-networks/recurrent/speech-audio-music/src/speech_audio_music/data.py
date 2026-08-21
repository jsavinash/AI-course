"""Data loading and preprocessing for music generation (RNN).

Generates synthetic musical note sequences for language-model-style generation.
Notes are represented as MIDI-style integer indices (0-39).
"""

from pathlib import Path

import numpy as np

VOCAB_SIZE = 40
SEQ_LEN = 20

DEFAULT_N_SAMPLES = 500

NOTE_NAMES = [
    "C4",
    "C#4",
    "D4",
    "D#4",
    "E4",
    "F4",
    "F#4",
    "G4",
    "G#4",
    "A4",
    "A#4",
    "B4",
    "C5",
    "C#5",
    "D5",
    "D#5",
    "E5",
    "F5",
    "F#5",
    "G5",
    "rest",
    "C3",
    "D3",
    "E3",
    "F3",
    "G3",
    "A3",
    "B3",
    "C4b",
    "pause",
    "D5b",
    "E5b",
    "F5b",
    "G5b",
    "A5b",
    "B5b",
    "high_C",
    "low_C",
    "chord",
    "end",
]


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> np.ndarray:
    """Generate synthetic note-index sequences with musical patterns.

    Sequences follow simple probabilistic patterns (e.g., stepwise motion,
    repeated notes, chord progressions) so the RNN can learn next-note prediction.

    Returns:
        X: (n_samples, SEQ_LEN) note indices
    """
    rng = np.random.default_rng(random_seed)

    X = np.zeros((n_samples, SEQ_LEN), dtype=int)

    for i in range(n_samples):
        seq = np.zeros(SEQ_LEN, dtype=int)
        seq[0] = rng.integers(0, 20)  # start with a note
        for t in range(1, SEQ_LEN):
            r = rng.random()
            if r < 0.5:
                # Stepwise motion (step up/down)
                seq[t] = (seq[t - 1] + rng.choice([-2, -1, 1, 2])) % VOCAB_SIZE
            elif r < 0.7:
                # Repeat the same note
                seq[t] = seq[t - 1]
            else:
                # Random jump
                seq[t] = rng.integers(0, VOCAB_SIZE)
        X[i] = seq

    return X


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    random_seed: int = 42,
) -> np.ndarray:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"]
    return generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)


def save_training_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y)


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
