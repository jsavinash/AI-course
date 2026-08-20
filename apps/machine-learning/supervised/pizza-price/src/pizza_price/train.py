"""Production training pipeline for pizza price prediction."""

import argparse
import os
from pathlib import Path

import numpy as np
from mlops_shared.logging import get_logger, setup_logging
from mlops_shared.model_registry import ModelRegistry
from mlops_shared.validation import DataValidator, create_pizza_schema

from pizza_price.data import load_training_data, save_training_data, train_test_split
from pizza_price.model import LinearRegression

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path,
    learning_rate: float,
    n_iterations: int,
    model_version: str,
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -> dict:
    """Train the pizza price model and save artifacts."""
    # Load training data
    X, y = load_training_data(data_path)
    logger.info("Loaded training data", n_samples=len(X), data_path=str(data_path))

    # Validate training data
    validator = DataValidator(create_pizza_schema())
    validation = validator.validate(X, y)
    if not validation.valid:
        logger.error("Training data validation failed", errors=validation.errors)
        raise ValueError(f"Training data validation failed: {validation.errors}")
    logger.info("Training data validated", stats=validation.stats)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_seed=random_seed
    )
    logger.info(
        "Data split",
        n_train=len(X_train),
        n_test=len(X_test),
        test_size=test_size,
        random_seed=random_seed,
    )

    # Save training data for reproducibility
    save_training_data(X, y, model_dir / "training_data.csv")

    # Train model
    model = LinearRegression(learning_rate=learning_rate, n_iterations=n_iterations)
    model.fit(X_train, y_train)

    # Evaluate on train and test
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(
        "Training complete",
        weight=model.weight,
        bias=model.bias,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        iterations=n_iterations,
    )

    # Model validation - check metrics meet thresholds
    if test_metrics["rmse"] > 5.0:
        logger.warning("Model RMSE above threshold", rmse=test_metrics["rmse"], threshold=5.0)

    # Save model
    model_path = model_dir / f"pizza_model_v{model_version}.npz"
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, X, y, model_dir, model_version)

    # Combined metrics for registry
    metrics = {
        "mse": test_metrics["mse"],
        "rmse": test_metrics["rmse"],
        "mae": test_metrics["mae"],
        "r2": test_metrics["r2"],
        "train_mse": train_metrics["mse"],
        "train_r2": train_metrics["r2"],
        "weight": model.weight,
        "bias": model.bias,
        "n_samples": len(X),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="pizza-price",
        model_version=model_version,
        model_type="regression",
        metrics=metrics,
        parameters={
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "random_seed": random_seed,
        },
        artifacts={
            f"pizza_model_v{model_version}.npz": model_path,
            "training_data.csv": model_dir / "training_data.csv",
        },
        tags={"framework": "numpy", "task": "regression"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="pizza-price",
            model_version=model_version,
            metrics=metrics,
            params={
                "learning_rate": learning_rate,
                "n_iterations": n_iterations,
                "random_seed": random_seed,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"pizza_regression_v{model_version}.png"),
                "training_data": str(model_dir / "training_data.csv"),
            },
            tags={"model_type": "regression", "framework": "numpy"},
        )
        logger.info("Registered model to MLflow", model="pizza-price", version=model_version)

    return metrics


def _save_chart(
    model: LinearRegression, X: np.ndarray, y: np.ndarray, output_dir: Path, version: str
) -> None:
    """Save the regression chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color="blue", s=100, label="Training data")

    line_x = np.linspace(min(X) - 1, max(X) + 1, 100)
    line_y = model.predict(line_x)
    plt.plot(line_x, line_y, color="red", linewidth=2, label="Fitted line")

    plt.xlabel("Pizza Diameter (inches)")
    plt.ylabel("Price (USD)")
    plt.title(f"Pizza Price vs Diameter - Trained Model v{version}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    chart_path = output_dir / f"pizza_regression_v{version}.png"
    plt.tight_layout()
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description="Train pizza price prediction model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument(
        "--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.001"))
    )
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "2000")))
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--test-size", type=float, default=float(os.getenv("TEST_SIZE", "0.2")))
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument(
        "--register-mlflow",
        action="store_true",
        default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true",
    )
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
