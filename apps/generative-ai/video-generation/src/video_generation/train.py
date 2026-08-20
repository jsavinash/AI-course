"""Training pipeline for Video Generation."""

import argparse
import os
from pathlib import Path

import numpy as np
from mlops_shared.logging import get_logger, setup_logging
from mlops_shared.model_registry import ModelRegistry

from video_generation.data import load_video_dataset, save_dataset, train_test_split_videos
from video_generation.model import VideoGenerationModel

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 200,
    img_size: int = 32,
    n_frames: int = 8,
    latent_dim: int = 64,
    model_id: str = "video-generation-v1",
    n_diffusion_steps: int = 1000,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -> dict:
    logger.info("Loading video dataset", n_samples=n_samples, n_frames=n_frames)
    videos, prompts = load_video_dataset(data_path=data_path, n_samples=n_samples, random_seed=random_seed)

    X_train, X_test, prompts_train, prompts_test = train_test_split_videos(videos, prompts, test_size=0.2, random_seed=random_seed)
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(videos, prompts, model_dir / "training_data.npz")

    model = VideoGenerationModel(
        model_id=model_id,
        img_size=img_size,
        n_frames=n_frames,
        latent_dim=latent_dim,
        n_diffusion_steps=n_diffusion_steps,
        random_seed=random_seed,
    )
    model._init()

    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    metrics = model.fit(X_train_flat, np.zeros(len(X_train_flat)), n_iterations=10)
    logger.info("Training finished", metrics=metrics)

    eval_metrics = model.evaluate(X_test_flat, np.zeros(len(X_test_flat)))
    logger.info("Evaluation metrics", metrics=eval_metrics)

    model_path = model_dir / f"video_generation_v{model_version}.npz"
    model.save(str(model_path))

    combined_metrics = {**metrics, **eval_metrics}
    combined_metrics.update({
        "img_size": float(img_size),
        "n_frames": float(n_frames),
        "latent_dim": float(latent_dim),
        "n_diffusion_steps": float(n_diffusion_steps),
        "n_samples": float(n_samples),
    })

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="video-generation",
        model_version=model_version,
        model_type="generative",
        metrics=combined_metrics,
        parameters={
            "model_id": model_id,
            "img_size": img_size,
            "n_frames": n_frames,
            "latent_dim": latent_dim,
            "n_diffusion_steps": n_diffusion_steps,
            "n_samples": n_samples,
            "random_seed": random_seed,
        },
        artifacts={f"video_generation_v{model_version}.npz": model_path, "training_data.npz": model_dir / "training_data.npz"},
        tags={"framework": "numpy", "task": "video_generation", "model_type": "VideoGeneration"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="video-generation",
            model_version=model_version,
            metrics=combined_metrics,
            params={"model_id": model_id, "img_size": img_size, "n_frames": n_frames, "latent_dim": latent_dim, "n_diffusion_steps": n_diffusion_steps, "n_samples": n_samples},
            artifacts={"model": str(model_path)},
            tags={"model_type": "video_generation", "framework": "numpy"},
        )

    return combined_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Video Generation Model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "200")))
    parser.add_argument("--img-size", type=int, default=int(os.getenv("IMG_SIZE", "32")))
    parser.add_argument("--n-frames", type=int, default=int(os.getenv("N_FRAMES", "8")))
    parser.add_argument("--latent-dim", type=int, default=int(os.getenv("LATENT_DIM", "64")))
    parser.add_argument("--model-id", type=str, default=os.getenv("MODEL_ID", "video-generation-v1"))
    parser.add_argument("--n-diffusion-steps", type=int, default=int(os.getenv("N_DIFFUSION_STEPS", "1000")))
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument("--register-mlflow", action="store_true", default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true")
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_samples=args.n_samples,
        img_size=args.img_size,
        n_frames=args.n_frames,
        latent_dim=args.latent_dim,
        model_id=args.model_id,
        n_diffusion_steps=args.n_diffusion_steps,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )
    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
