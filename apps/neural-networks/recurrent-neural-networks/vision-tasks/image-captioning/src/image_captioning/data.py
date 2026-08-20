"""Data loading and preprocessing for image captioning (RNN).

Generates synthetic image pixel arrays (8x8=64) and corresponding caption word sequences.
"""

from pathlib import Path

import numpy as np

N_PIXELS = 64
VOCAB_SIZE = 20
CAPTION_LEN = 8

DEFAULT_N_SAMPLES = 500

# Simple vocabulary: objects + descriptors
VOCAB_TOKENS = [
    "start",
    "a",
    "the",
    "object",
    "bright",
    "dark",
    "round",
    "square",
    "small",
    "large",
    "circle",
    "box",
    "shape",
    "is",
    "this",
    "red",
    "blue",
    "green",
    "pattern",
    "end",
]


def _create_image_template(
    pattern_type: int, noise_level: float = 0.1, rng: np.random.Generator | None = None
) -> np.ndarray:
    """Generate an 8x8 image with a specific pattern.

    pattern_type determines the pattern (0=circle-like, 1=corner-like, etc.)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    img = np.zeros(N_PIXELS)

    if pattern_type == 0:
        # Circle in center
        img[27:29] = 0.9
        img[28:31] += 0.8
        img[27:29] += 0.4
    elif pattern_type == 1:
        # Corner
        img[0:3] = 0.9
        img[8:11] = 0.8
    elif pattern_type == 2:
        # Horizontal bar
        img[28:36] = 0.9
    elif pattern_type == 3:
        # Vertical bar
        img[::8] = 0.9
    elif pattern_type == 4:
        # Diagonal
        img[::9] = 0.9
    else:
        img.flat[rng.integers(0, N_PIXELS, size=10)] = 0.9

    img = np.clip(img + rng.normal(0, noise_level, N_PIXELS), 0, 1)
    return img


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.1,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic images and their caption sequences.

    Each image has one of 5 basic patterns, and the caption describes it.

    Returns:
        X_images: (n_samples, N_PIXELS) image pixel arrays
        captions: (n_samples, CAPTION_LEN) word token indices
    """
    rng = np.random.default_rng(random_seed)

    # Pattern-to-caption mapping
    pattern_captions = {
        0: [0, 1, 3, 10, 2, 4, 5, 19],  # start a object circle the bright blue end
        1: [0, 1, 3, 11, 2, 6, 12, 19],  # start a object box the round shape end
        2: [0, 1, 3, 2, 7, 13, 14, 19],  # start a object the square is this end
        3: [0, 1, 3, 11, 2, 8, 15, 19],  # start a object box the small red end
        4: [0, 1, 3, 10, 2, 9, 16, 19],  # start a object circle the large blue end
    }

    X_images = np.zeros((n_samples, N_PIXELS))
    captions = np.zeros((n_samples, CAPTION_LEN), dtype=int)

    for i in range(n_samples):
        pattern = rng.integers(0, 5)
        X_images[i] = _create_image_template(pattern, noise_level, rng)
        captions[i] = pattern_captions[pattern]

    # Shuffle
    perm = rng.permutation(n_samples)
    return X_images[perm], captions[perm]


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.1,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["X"], data["y"]
    return generate_synthetic_data(
        n_samples=n_samples, noise_level=noise_level, random_seed=random_seed
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
