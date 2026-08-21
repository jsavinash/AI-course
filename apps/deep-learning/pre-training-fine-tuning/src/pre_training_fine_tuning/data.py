"""Data loading and preprocessing for pre-training and fine-tuning."""

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
    phase: str = "pretrain",
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic data for pre-training or fine-tuning.

    For pre-training:
        - MLM objective: returns (masked_sequences, original_sequences, mask_positions)
        - NTP objective: returns (input_sequences, target_sequences)

    For fine-tuning:
        - Returns (sequences, labels) for classification/regression

    Returns:
        X: input data
        y: target data (format depends on phase and objective)
    """
    rng = np.random.default_rng(random_seed)

    if phase == "pretrain":
        X = rng.integers(1, vocab_size, size=(n_samples, seq_len))
        y = np.zeros_like(X)
        y[:, :-1] = X[:, 1:]
        y[:, -1] = rng.integers(1, vocab_size)
        return X, y

    X = rng.integers(1, vocab_size, size=(n_samples, seq_len))
    y = rng.integers(0, 10, size=n_samples)
    return X, y


def generate_mlm_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    vocab_size: int = VOCAB_SIZE,
    seq_len: int = MAX_SEQ_LEN,
    mask_prob: float = 0.15,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate data for Masked Language Modeling (MLM) pre-training.

    Args:
        mask_prob: probability of masking a token (default 15% like BERT)

    Returns:
        masked_X: input sequences with masked tokens (replaced with [MASK] token id=0)
        original_y: original token sequences (targets)
        mask_positions: boolean array indicating masked positions
    """
    rng = np.random.default_rng(random_seed)
    X = rng.integers(1, vocab_size, size=(n_samples, seq_len))
    mask_positions = rng.random((n_samples, seq_len)) < mask_prob
    masked_X = X.copy()
    masked_X[mask_positions] = 0
    return masked_X, X, mask_positions


def generate_ntp_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    vocab_size: int = VOCAB_SIZE,
    seq_len: int = MAX_SEQ_LEN,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate data for Next-Token Prediction (NTP) pre-training.

    Returns:
        X: input sequences (all tokens except last)
        y: target next tokens (shifted by one position)
    """
    rng = np.random.default_rng(random_seed)
    X = rng.integers(1, vocab_size, size=(n_samples, seq_len))
    y = np.zeros_like(X)
    y[:, :-1] = X[:, 1:]
    y[:, -1] = rng.integers(1, vocab_size)
    return X, y


def generate_finetune_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    vocab_size: int = VOCAB_SIZE,
    seq_len: int = MAX_SEQ_LEN,
    n_classes: int = 10,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate data for fine-tuning (classification task).

    Returns:
        X: input sequences
        y: class labels
    """
    rng = np.random.default_rng(random_seed)
    X = rng.integers(1, vocab_size, size=(n_samples, seq_len))
    y = rng.integers(0, n_classes, size=n_samples)
    return X, y


def train_test_split(
    X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_seed: int | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data into train and test sets."""
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
    """Save training data to npz file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y)


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    vocab_size: int = VOCAB_SIZE,
    random_seed: int = 42,
    phase: str = "pretrain",
) -> tuple[np.ndarray, np.ndarray]:
    """Load training data from file or generate synthetic data."""
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["y"]
    return generate_synthetic_data(
        n_samples=n_samples, vocab_size=vocab_size, random_seed=random_seed, phase=phase
    )
