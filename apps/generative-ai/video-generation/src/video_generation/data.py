"""Data loading and preprocessing for Video Generation."""

from pathlib import Path

import numpy as np

DEFAULT_N_SAMPLES = 200
DEFAULT_IMG_SIZE = 32
DEFAULT_N_FRAMES = 8


def generate_synthetic_videos(n_samples: int = DEFAULT_N_SAMPLES, img_size: int = DEFAULT_IMG_SIZE, n_frames: int = DEFAULT_N_FRAMES, random_seed: int = 42) -> tuple[np.ndarray, list[str]]:
    rng = np.random.default_rng(random_seed)
    videos = rng.uniform(0, 1, size=(n_samples, n_frames, img_size, img_size, 3))
    templates = [
        "a {object} moving across a {setting}",
        "a {object} spinning in {setting}",
        "a {object} walking through {setting}",
        "an animation of {object} in {setting}",
    ]
    objects = ["cat", "dog", "bird", "car", "robot", "person", "fish", "butterfly"]
    settings = ["forest", "ocean", "city", "space", "desert", "garden", "mountain", "sky"]
    prompts = []
    for _ in range(n_samples):
        template = rng.choice(templates)
        prompt = template.format(object=rng.choice(objects), setting=rng.choice(settings))
        prompts.append(prompt)
    return videos, prompts


def load_video_dataset(data_path: Path | None = None, n_samples: int = DEFAULT_N_SAMPLES, random_seed: int = 42) -> tuple[np.ndarray, list[str]]:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return data["videos"], data["prompts"].tolist()
    return generate_synthetic_videos(n_samples=n_samples, random_seed=random_seed)


def train_test_split_videos(videos: np.ndarray, prompts: list[str], test_size: float = 0.2, random_seed: int | None = None) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    n = len(videos)
    n_test = max(1, int(n * test_size))
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(n)
    else:
        indices = np.random.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    return videos[train_idx], videos[test_idx], [prompts[i] for i in train_idx], [prompts[i] for i in test_idx]


def save_dataset(videos: np.ndarray, prompts: list[str], path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, videos=videos, prompts=np.array(prompts, dtype=object))
