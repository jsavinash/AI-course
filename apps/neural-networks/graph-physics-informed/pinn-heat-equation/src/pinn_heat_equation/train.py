"""Training pipeline for PINN Heat Equation Solver."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_pinn_heat_equation_schema

from pinn_heat_equation.data import (
    generate_synthetic_data,
    save_training_data,
    train_test_split,
)
from pinn_heat_equation.model import PINNHeatEquation

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 200,
    alpha: float = 0.01,
    hidden_dim: int = 32,
    n_layers: int = 2,
    learning_rate: float = 0.01,
    n_iterations: int = 500,
    weight_decay: float = 0.001,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    X, u_true = generate_synthetic_data(n_samples=n_samples, random_seed=random_seed, alpha=alpha)
    logger.info("Generated PDE training data", n_samples=n_samples, alpha=alpha)

    validator = DataValidator(create_pinn_heat_equation_schema())
    validation = validator.validate(X)
    if not validation.valid:
        logger.error("Training data validation failed", errors=validation.errors)
        raise ValueError(f"Training data validation failed: {validation.errors}")

    X_train, X_test, u_train, u_test = train_test_split(X, u_true, test_size=test_size, random_seed=random_seed)
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, u_true, model_dir / "training_data.npz")

    model = PINNHeatEquation(
        alpha=alpha,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X_train, u_train)

    test_metrics = model.evaluate(X_test, u_test)
    logger.info("Training complete", training_mode=model.training_mode, final_loss=model.loss_history[-1])

    model_path = model_dir / f"pinn_model_v{model_version}.npz"
    model.save(str(model_path))

    metrics = {
        **test_metrics,
        "training_mode": "physics-informed",
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "n_train_samples": float(len(X_train)),
        "n_test_samples": float(len(X_test)),
        "alpha": float(alpha),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="pinn-heat-equation",
        model_version=model_version,
        model_type="regression",
        metrics=metrics,
        parameters={
            "alpha": alpha,
            "hidden_dim": hidden_dim,
            "n_layers": n_layers,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
        },
        artifacts={
            f"pinn_model_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        },
        tags={"framework": "numpy", "task": "pinn_heat_equation", "model_type": "PINN"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="pinn-heat-equation",
            model_version=model_version,
            metrics=metrics,
            params={"alpha": alpha, "hidden_dim": hidden_dim, "n_layers": n_layers, "learning_rate": learning_rate, "n_iterations": n_iterations},
            artifacts={"model": str(model_path)},
            tags={"model_type": "pinn", "framework": "numpy"},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train PINN Heat Equation model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "200")))
    parser.add_argument("--alpha", type=float, default=float(os.getenv("ALPHA", "0.01")))
    parser.add_argument("--hidden-dim", type=int, default=int(os.getenv("HIDDEN_DIM", "32")))
    parser.add_argument("--n-layers", type=int, default=int(os.getenv("N_LAYERS", "2")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.01")))
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "500")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.001")))
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
        alpha=args.alpha,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )
    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
