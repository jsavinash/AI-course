"""Data loading and preprocessing for Code Generation."""

from pathlib import Path

import numpy as np

VOCAB_SIZE = 1000
MAX_SEQ_LEN = 128
DEFAULT_N_SAMPLES = 500


def generate_synthetic_code_data(n_samples: int = DEFAULT_N_SAMPLES, vocab_size: int = VOCAB_SIZE, seq_len: int = MAX_SEQ_LEN, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    X = rng.integers(0, vocab_size, size=(n_samples, seq_len))
    y = np.zeros_like(X)
    y[:, :-1] = X[:, 1:]
    y[:, -1] = rng.integers(0, vocab_size)
    return X, y


def generate_code_completion_data(n_samples: int = DEFAULT_N_SAMPLES, vocab_size: int = VOCAB_SIZE, seq_len: int = MAX_SEQ_LEN, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    X = rng.integers(0, vocab_size, size=(n_samples, seq_len // 2))
    y = rng.integers(0, vocab_size, size=(n_samples, seq_len // 2))
    return X, y


def generate_text_to_code_data(n_samples: int = DEFAULT_N_SAMPLES, vocab_size: int = VOCAB_SIZE, seq_len: int = MAX_SEQ_LEN, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    X = rng.integers(0, vocab_size, size=(n_samples, seq_len // 4))
    y = rng.integers(0, vocab_size, size=(n_samples, seq_len))
    return X, y


def load_code_dataset(data_path: Path | None = None, n_samples: int = DEFAULT_N_SAMPLES, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["y"]
    return generate_synthetic_code_data(n_samples=n_samples, random_seed=random_seed)


def tokenize_code(code: str, tokenizer: "CodeTokenizer | None" = None) -> list[int]:
    if tokenizer is None:
        tokenizer = CodeTokenizer()
    return tokenizer.encode(code)


def detokenize_code(token_ids: list[int], tokenizer: "CodeTokenizer | None" = None) -> str:
    if tokenizer is None:
        tokenizer = CodeTokenizer()
    return tokenizer.decode(token_ids)


def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_seed: int | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
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


def save_dataset(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, X=X, y=y)
