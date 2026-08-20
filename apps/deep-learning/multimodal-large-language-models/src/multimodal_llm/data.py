"""Data loading and preprocessing for Multimodal Language Modeling."""

from pathlib import Path

import numpy as np

VOCAB_SIZE = 1000
MAX_SEQ_LEN = 64
DEFAULT_N_SAMPLES = 500

TEXT_DIM = 256
IMAGE_DIM = 768
AUDIO_DIM = 80
IMAGE_PATCH_SIZE = 16
N_PATCHES = 49
AUDIO_TIME_STEPS = 100


def generate_synthetic_text(n_samples: int = DEFAULT_N_SAMPLES, seq_len: int = MAX_SEQ_LEN, vocab_size: int = VOCAB_SIZE, random_seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    X = rng.integers(0, vocab_size, size=(n_samples, seq_len))
    y = np.zeros_like(X)
    y[:, :-1] = X[:, 1:]
    y[:, -1] = rng.integers(0, vocab_size)
    return X, y


def generate_synthetic_image_patches(n_samples: int = DEFAULT_N_SAMPLES, n_patches: int = N_PATCHES, patch_size: int = IMAGE_PATCH_SIZE, random_seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    patch_dim = patch_size * patch_size * 3
    patches = rng.normal(0, 1, size=(n_samples, n_patches, patch_dim))
    return patches


def generate_synthetic_audio(n_samples: int = DEFAULT_N_SAMPLES, n_mels: int = AUDIO_DIM, n_time_steps: int = AUDIO_TIME_STEPS, random_seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(random_seed)
    mel_spec = rng.normal(0, 1, size=(n_samples, n_time_steps, n_mels))
    return mel_spec


def generate_synthetic_multimodal_data(n_samples: int = DEFAULT_N_SAMPLES, vocab_size: int = VOCAB_SIZE, seq_len: int = MAX_SEQ_LEN, random_seed: int = 42, include_image: bool = True, include_audio: bool = True) -> dict:
    text_X, text_y = generate_synthetic_text(n_samples, seq_len, vocab_size, random_seed)
    data = {"text_tokens": text_X, "text_targets": text_y}

    if include_image:
        data["image_patches"] = generate_synthetic_image_patches(n_samples, random_seed=random_seed)

    if include_audio:
        data["mel_spectrogram"] = generate_synthetic_audio(n_samples, random_seed=random_seed)

    return data


def extract_text_features(tokens: np.ndarray, vocab_size: int = VOCAB_SIZE) -> np.ndarray:
    features = np.zeros((tokens.shape[0], vocab_size))
    for i in range(tokens.shape[0]):
        for j in range(tokens.shape[1]):
            features[i, int(tokens[i, j])] += 1
    return features


def extract_image_features(patches: np.ndarray) -> np.ndarray:
    return np.mean(patches, axis=-1)


def extract_audio_features(mel_spec: np.ndarray) -> np.ndarray:
    return np.mean(mel_spec, axis=-1)


def create_joint_representation(text_features: np.ndarray, image_features: np.ndarray | None = None, audio_features: np.ndarray | None = None) -> np.ndarray:
    joint = text_features
    if image_features is not None:
        joint = np.concatenate([joint, image_features], axis=-1)
    if audio_features is not None:
        joint = np.concatenate([joint, audio_features], axis=-1)
    return joint


def load_multimodal_data(data_path: Path | None = None, n_samples: int = DEFAULT_N_SAMPLES, vocab_size: int = VOCAB_SIZE, seq_len: int = MAX_SEQ_LEN, random_seed: int = 42, include_image: bool = True, include_audio: bool = True) -> dict:
    if data_path and Path(data_path).exists():
        data = np.load(data_path, allow_pickle=True)
        return {key: data[key] for key in data.files}
    return generate_synthetic_multimodal_data(n_samples, vocab_size, seq_len, random_seed, include_image, include_audio)


def train_test_split_multimodal(data: dict, test_size: float = 0.2, random_seed: int | None = None) -> tuple[dict, dict]:
    n = data["text_tokens"].shape[0]
    n_test = max(1, int(n * test_size))
    if random_seed is not None:
        rng = np.random.default_rng(random_seed)
        indices = rng.permutation(n)
    else:
        indices = np.random.permutation(n)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]
    train_data = {k: v[train_idx] for k, v in data.items()}
    test_data = {k: v[test_idx] for k, v in data.items()}
    return train_data, test_data


def save_multimodal_data(data: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **data)
