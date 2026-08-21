"""Training pipeline for Autoencoder Dimensionality Reduction."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_autoencoders_dimensionality_reduction_schema

from autoencoders_dimensionality_reduction.data import (
    N_FEATURES,
    load_training_data,
    save_training_data,
    train_test_split,
)
from autoencoders_dimensionality_reduction.model import Autoencoder

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    latent_dim: int = 8,
    hidden_dim: int = 16,
    learning_rate: float = 0.01,
    n_iterations: int = 300,
    weight_decay: float = 0.001,
    noise_rate: float = 0.0,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """Train the autoencoder model and save artifacts."""
    X, y = load_training_data(data_path, n_samples=n_samples, random_seed=random_seed)
    logger.info("Loaded training data", n_samples=len(X), data_path=str(data_path))

    validator = DataValidator(create_autoencoders_dimensionality_reduction_schema())
    validation = validator.validate(X)
    if not validation.valid:
        logger.error("Training data validation failed", errors=validation.errors)
        raise ValueError(f"Training data validation failed: {validation.errors}")

    X_train, X_test, _, _ = train_test_split(X, y, test_size=test_size, random_seed=random_seed)
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / "training_data.npz")

    model = Autoencoder(
        n_features=N_FEATURES,
        latent_dim=latent_dim,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        noise_rate=noise_rate,
        random_seed=random_seed,
    )
    model.fit(X_train)

    model.evaluate(X_train)
    test_metrics = model.evaluate(X_test)

    logger.info("Training complete", training_mode=model.training_mode, final_loss=model.loss_history[-1])

    model_path = model_dir / f"autoencoders_model_v{model_version}.npz"
    model.save(str(model_path))

    _save_chart(model, model_dir, model_version)

    metrics = {
        **test_metrics,
        "training_mode": "unsupervised",
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "n_train_samples": float(len(X_train)),
        "n_test_samples": float(len(X_test)),
        "latent_dim": float(latent_dim),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="autoencoders-dimensionality-reduction",
        model_version=model_version,
        model_type="generative",
        metrics=metrics,
        parameters={
            "n_features": N_FEATURES,
            "latent_dim": latent_dim,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "noise_rate": noise_rate,
            "random_seed": random_seed,
        },
        artifacts={
            f"autoencoders_model_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        },
        tags={"framework": "numpy", "task": "autoencoders_dimensionality_reduction", "model_type": "AE"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="autoencoders-dimensionality-reduction",
            model_version=model_version,
            metrics=metrics,
            params={
                "n_features": N_FEATURES,
                "latent_dim": latent_dim,
                "hidden_dim": hidden_dim,
                "learning_rate": learning_rate,
                "n_iterations": n_iterations,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"autoencoders_v{model_version}.png"),
            },
            tags={"model_type": "autoencoders", "framework": "numpy"},
        )

    return metrics


def _save_chart(model, output_dir: Path, version: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Loss")
    ax.set_title("Autoencoder Training Loss")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    chart_path = output_dir / f"autoencoders_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Train Autoencoder Dimensionality Reduction model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument("--latent-dim", type=int, default=int(os.getenv("LATENT_DIM", "8")))
    parser.add_argument("--hidden-dim", type=int, default=int(os.getenv("HIDDEN_DIM", "16")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.01")))
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "300")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.001")))
    parser.add_argument("--noise-rate", type=float, default=float(os.getenv("NOISE_RATE", "0.0")))
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
        latent_dim=args.latent_dim,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        noise_rate=args.noise_rate,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )
    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
