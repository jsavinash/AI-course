"""Data generation and preprocessing for handwritten digit recognition.

Generates synthetic digit feature vectors (8x8 = 64 features) representing
simplified handwritten digits 0-9. Each digit is represented as a flattened
8x8 grayscale image with pixel values normalized to [0, 1].
"""

from pathlib import Path

import numpy as np
import pandas as pd

GRID_SIZE = 8
N_FEATURES = GRID_SIZE * GRID_SIZE
N_CLASSES = 10

FEATURE_NAMES = [f"pixel_{i}" for i in range(N_FEATURES)]

DEFAULT_N_SAMPLES = 1000


def _create_digit_template(digit: int) -> np.ndarray:
    """Create a template 8x8 pattern for a given digit (0-9)."""
    grid = np.zeros((GRID_SIZE, GRID_SIZE), dtype=float)

    if digit == 0:
        grid[1:7, 1:7] = 1.0
        grid[2:6, 2:6] = 0.1
        grid[3:5, 3:5] = 0.0

    elif digit == 1:
        grid[1:7, 3:5] = 1.0

    elif digit == 2:
        grid[1, 1:7] = 1.0
        grid[2:4, 6] = 1.0
        grid[4, 1:7] = 1.0
        grid[5:6, 1] = 1.0
        grid[6, 1:7] = 1.0

    elif digit == 3:
        grid[1:3, 5:7] = 1.0
        grid[3, 1:6] = 1.0
        grid[4:6, 5:7] = 1.0
        grid[6, 1:6] = 1.0

    elif digit == 4:
        grid[1:5, 1] = 1.0
        grid[4, 1:5] = 1.0
        grid[1:5, 4] = 1.0
        grid[1, 4] = 1.0

    elif digit == 5:
        grid[1, 1:7] = 1.0
        grid[2:4, 1] = 1.0
        grid[1:3, 5:7] = 1.0
        grid[3, 4:6] = 1.0
        grid[4:7, 5] = 1.0
        grid[6, 1:6] = 1.0

    elif digit == 6:
        grid[1, 4:7] = 1.0
        grid[2:4, 3] = 1.0
        grid[2, 5:7] = 1.0
        grid[4, 3:6] = 1.0
        grid[5:7, 6] = 1.0
        grid[6, 3:5] = 1.0

    elif digit == 7:
        grid[1, 1:7] = 1.0
        grid[2:7, 5:6] = 1.0
        grid[3:7, 3:4] = 1.0

    elif digit == 8:
        grid[1:7, 1:3] = 1.0
        grid[1:2, 4:7] = 1.0
        grid[3:4, 4:7] = 1.0
        grid[5:6, 4:7] = 1.0
        grid[6:7, 1:3] = 1.0
        grid[1, 3:4] = 1.0
        grid[7, 3:4] = 1.0

    elif digit == 9:
        grid[1, 1:4] = 1.0
        grid[2:4, 4] = 1.0
        grid[4:5, 1:5] = 1.0
        grid[5:7, 5:6] = 1.0
        grid[6, 1:5] = 1.0

    return grid.flatten()


def generate_synthetic_data(
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.3,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Generate synthetic handwritten digit data.

    Each sample is an 8x8=64 pixel image of a digit (0-9) with added noise.

    Returns:
        Tuple of (X, y) where X is (n_samples, 64) and y is digit labels (0-9).
    """
    rng = np.random.default_rng(random_seed)

    templates = {d: _create_digit_template(d) for d in range(N_CLASSES)}

    X = np.zeros((n_samples, N_FEATURES))
    y = np.zeros(n_samples, dtype=int)

    for i in range(n_samples):
        digit = rng.integers(0, N_CLASSES)
        template = templates[digit].copy()

        noise = rng.normal(0, noise_level, N_FEATURES)
        noisy = template + noise
        noisy = np.clip(noisy, 0, 1)

        X[i] = noisy
        y[i] = digit

    indices = rng.permutation(n_samples)
    return X[indices], y[indices]


def load_training_data(
    data_path: Path | None = None,
    n_samples: int = DEFAULT_N_SAMPLES,
    noise_level: float = 0.3,
    random_seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Load or generate digit data for training."""
    if data_path and Path(data_path).exists():
        df = pd.read_csv(data_path)
        feature_cols = [c for c in df.columns if c.startswith("pixel_")]
        X = df[feature_cols].values.astype(float)
        y = df["label"].values.astype(int)
        return X, y

    return generate_synthetic_data(
        n_samples=n_samples, noise_level=noise_level, random_seed=random_seed
    )


def save_training_data(X: np.ndarray, y: np.ndarray, path: Path) -> None:
    """Save training data to CSV."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(X, columns=FEATURE_NAMES)
    df["label"] = y
    df.to_csv(path, index=False)


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


def one_hot_encode(y: np.ndarray, n_classes: int = N_CLASSES) -> np.ndarray:
    """Convert integer labels to one-hot encoded vectors."""
    one_hot = np.zeros((len(y), n_classes))
    one_hot[np.arange(len(y)), y] = 1.0
    return one_hot


def image_to_string(X: np.ndarray, threshold: float = 0.5) -> str:
    """Convert a flattened 8x8 image to an ASCII string for debug display."""
    grid = X.reshape(GRID_SIZE, GRID_SIZE)
    chars = []
    for row in grid:
        line = ""
        for val in row:
            line += "\u2588" if val >= threshold else " "
        chars.append(line)
    return "\n".join(chars)
