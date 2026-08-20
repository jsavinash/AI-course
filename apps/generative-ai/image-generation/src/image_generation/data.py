"""Data loading and preprocessing for Image Generation."""

from pathlib import Path

import numpy as np

DEFAULT_N_SAMPLES = 500
DEFAULT_IMG_SIZE = 32
DEFAULT_LATENT_DIM = 64


def generate_synthetic_images(n_samples: int = DEFAULT_N_SAMPLES, img_size: int = DEFAULT_IMG_SIZE, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    images = rng.uniform(0, 1, size=(n_samples, img_size, img_size, 3))
    labels = rng.integers(0, 10, size=n_samples)
    return images, labels


def generate_synthetic_prompts(n_samples: int = DEFAULT_N_SAMPLES, random_seed: int = 42) -> list[str]:
    rng = np.random.default_rng(random_seed)
    templates = [
        "a photo of a {object} in {setting}",
        "a painting of {object} with {style} style",
        "a digital render of {object} in {color} lighting",
        "an abstract image of {object} and {object}",
    ]
    objects = ["cat", "dog", "bird", "car", "city", "forest", "ocean", "mountain", "flower", "sunset"]
    settings = ["a forest", "the ocean", "a city", "a mountain", "space", "a desert", "a garden", "the sky"]
    styles = ["modern", "classic", "futuristic", "vintage", "minimalist", "impressionist", "surreal"]
    colors = ["bright", "dark", "warm", "cool", "neon", "soft", "vivid"]
    prompts = []
    for _ in range(n_samples):
        template = rng.choice(templates)
        prompt = template.format(
            object=rng.choice(objects),
            setting=rng.choice(settings),
            style=rng.choice(styles),
            color=rng.choice(colors),
        )
        prompts.append(prompt)
    return prompts


def load_image_dataset(data_path: Path | None = None, n_samples: int = DEFAULT_N_SAMPLES, random_seed: int = 42) -> tuple[np.ndarray, list[str]]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["images"], data["prompts"].tolist()
    images, _ = generate_synthetic_images(n_samples=n_samples, random_seed=random_seed)
    prompts = generate_synthetic_prompts(n_samples=n_samples, random_seed=random_seed)
    return images, prompts


def train_test_split_images(images: np.ndarray, prompts: list[str], test_size: float = 0.2, random_seed: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    n = len(images)
    n_test = max(1, int(n * test_size))
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(n)
    else:
        indices = np.random.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return images[train_idx], images[test_idx], [prompts[i] for i in train_idx], [prompts[i] for i in test_idx]


def save_dataset(images: np.ndarray, prompts: list[str], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, images=images, prompts=np.array(prompts, dtype=object))
