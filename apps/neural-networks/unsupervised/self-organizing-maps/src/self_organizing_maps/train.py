"""Training pipeline for Self-Organizing Maps."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_self_organizing_maps_schema

from self_organizing_maps.data import (
    GRID_HEIGHT,
    GRID_WIDTH,
    N_FEATURES,
    load_training_data,
    save_training_data,
    train_test_split,
)
from self_organizing_maps.model import SelfOrganizingMap

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    grid_height: int = GRID_HEIGHT,
    grid_width: int = GRID_WIDTH,
    learning_rate: float = 0.5,
    n_iterations: int = 300,
    sigma: float = 2.0,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    X, y = load_training_data(data_path, n_samples=n_samples, random_seed=random_seed)
    logger.info("Loaded training data", n_samples=len(X), data_path=str(data_path))

    validator = DataValidator(create_self_organizing_maps_schema())
    validation = validator.validate(X.reshape(-1, 1))
    if not validation.valid:
        logger.error("Training data validation failed", errors=validation.errors)
        raise ValueError(f"Training data validation failed: {validation.errors}")

    X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size, random_seed=random_seed)
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / "training_data.npz")

    model = SelfOrganizingMap(
        n_features=N_FEATURES,
        grid_height=grid_height,
        grid_width=grid_width,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        sigma=sigma,
        random_seed=random_seed,
    )
    model.fit(X_train)

    test_metrics = model.evaluate(X_test)
    logger.info("Training complete", training_mode=model.training_mode, final_loss=model.loss_history[-1])

    model_path = model_dir / f"som_model_v{model_version}.npz"
    model.save(str(model_path))

    metrics = {
        **test_metrics,
        "training_mode": "unsupervised",
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "n_train_samples": float(len(X_train)),
        "n_test_samples": float(len(X_test)),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="self-organizing-maps",
        model_version=model_version,
        model_type="clustering",
        metrics=metrics,
        parameters={
            "n_features": N_FEATURES,
            "grid_height": grid_height,
            "grid_width": grid_width,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "sigma": sigma,
            "random_seed": random_seed,
        },
        artifacts={
            f"som_model_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        },
        tags={"framework": "numpy", "task": "self_organizing_maps", "model_type": "SOM"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="self-organizing-maps",
            model_version=model_version,
            metrics=metrics,
            params={"n_features": N_FEATURES, "grid_height": grid_height, "grid_width": grid_width, "learning_rate": learning_rate, "n_iterations": n_iterations},
            artifacts={"model": str(model_path)},
            tags={"model_type": "som", "framework": "numpy"},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Self-Organizing Maps model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument("--grid-height", type=int, default=int(os.getenv("GRID_HEIGHT", "5")))
    parser.add_argument("--grid-width", type=int, default=int(os.getenv("GRID_WIDTH", "5")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.5")))
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "300")))
    parser.add_argument("--sigma", type=float, default=float(os.getenv("SIGMA", "2.0")))
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--test-size", type=float, default=float(os.getenv("TEST_SIZE", "0.2")))
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
        grid_height=args.grid_height,
        grid_width=args.grid_width,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        sigma=args.sigma,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )
    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
