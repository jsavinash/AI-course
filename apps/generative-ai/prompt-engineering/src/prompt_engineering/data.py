"""Data loading and preprocessing for Prompt Engineering."""

from pathlib import Path

import numpy as np

DEFAULT_N_SAMPLES = 500
DEFAULT_VOCAB_SIZE = 1000


def generate_synthetic_prompts(n_samples: int = DEFAULT_N_SAMPLES, vocab_size: int = DEFAULT_VOCAB_SIZE, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    prompt_lengths = rng.integers(5, 50, size=n_samples)
    max_len = prompt_lengths.max()
    X = np.zeros((n_samples, max_len), dtype=int)
    for i, length in enumerate(prompt_lengths):
        X[i, :length] = rng.integers(0, vocab_size, size=length)
    y = rng.integers(0, 10, size=n_samples)
    return X, y


def load_prompt_dataset(data_path: Path | None = None, n_samples: int = DEFAULT_N_SAMPLES, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["y"]
    return generate_synthetic_prompts(n_samples=n_samples, random_seed=random_seed)


def build_vocab(texts: list[str], max_vocab_size: int = DEFAULT_VOCAB_SIZE) -> dict[str, int]:
    from collections import Counter
    word_counts = Counter()
    for text in texts:
        word_counts.update(text.lower().split())
    most_common = word_counts.most_common(max_vocab_size - 1)
    vocab = {word: idx + 1 for idx, (word, _) in enumerate(most_common)}
    vocab["<PAD>"] = 0
    vocab["<UNK>"] = max_vocab_size - 1
    return vocab


def encode_text(text: str, vocab: dict[str, int], max_len: int = 128) -> np.ndarray:
    tokens = [vocab.get(word, vocab.get("<UNK>", 0)) for word in text.lower().split()]
    if len(tokens) < max_len:
        tokens += [0] * (max_len - len(tokens))
    return np.array(tokens[:max_len])


def decode_tokens(tokens: np.ndarray, vocab: dict[str, int]) -> str:
    inv_vocab = {v: k for k, v in vocab.items()}
    words = [inv_vocab.get(int(t), "<UNK>") for t in tokens if int(t) != 0]
    return " ".join(words)


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
