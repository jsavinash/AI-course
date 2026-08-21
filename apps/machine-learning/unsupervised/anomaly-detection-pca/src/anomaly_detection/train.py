"""Production training pipeline for PCA-based anomaly detection."""

import argparse
import os
from pathlib import Path

import numpy as np
from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_anomaly_detection_schema

from anomaly_detection.data import load_training_data, save_training_data
from anomaly_detection.model import PCAAnomalyDetector

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path,
    n_components: int | float,
    threshold_method: str,
    threshold_percentile: float,
    threshold_iqr_multiplier: float,
    model_version: str,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -> dict:
    """Train the PCA anomaly detection model and save artifacts.

    Args:
        model_dir: Directory to save model artifacts
        data_path: Optional path to CSV data
        n_components: Number of PCA components or variance ratio to retain
        threshold_method: Method for anomaly threshold ("percentile", "iqr", "fixed")
        threshold_percentile: Percentile for threshold if method="percentile"
        threshold_iqr_multiplier: IQR multiplier if method="iqr"
        model_version: Model version string
        register_to_mlflow: Whether to register to MLflow
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with training metrics
    """
    # Load training data
    X, y = load_training_data(data_path, random_seed=random_seed)
    logger.info("Loaded training data", n_samples=len(X), n_features=X.shape[1])

    # Validate training data
    validator = DataValidator(create_anomaly_detection_schema())
    validation = validator.validate(X)
    if not validation.valid:
        logger.error("Training data validation failed", errors=validation.errors)
        raise ValueError(f"Training data validation failed: {validation.errors}")
    logger.info("Training data validated", stats=validation.stats)

    # Save training data for reproducibility
    save_training_data(X, y, model_dir / "training_data.csv")

    # Use only normal samples for PCA training (unsupervised anomaly detection)
    X_normal = X[y == 0]
    logger.info("Training on normal samples", n_normal=len(X_normal), n_anomaly=int(np.sum(y)))

    # Train model
    model = PCAAnomalyDetector(
        n_components=n_components,
        threshold_method=threshold_method,
        threshold_percentile=threshold_percentile,
        threshold_iqr_multiplier=threshold_iqr_multiplier,
        random_seed=random_seed,
    )
    model.fit(X_normal)

    # Evaluate on all data
    metrics = model.evaluate(X, y)
    logger.info(
        "Training complete",
        n_components=model.n_components_selected,
        explained_variance=metrics["explained_variance_ratio"],
        threshold=model.threshold,
        mean_error=metrics["mean_reconstruction_error"],
        max_error=metrics["max_reconstruction_error"],
    )

    if "accuracy" in metrics:
        logger.info(
            "Evaluation metrics",
            accuracy=metrics["accuracy"],
            precision=metrics["precision"],
            recall=metrics["recall"],
            f1=metrics["f1"],
        )

    # Save model
    model_path = model_dir / f"anomaly_detection_model_v{model_version}.npz"
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, X, y, model_dir, model_version)

    # Combined metrics for registry
    training_metrics = {
        "mean_reconstruction_error": metrics["mean_reconstruction_error"],
        "std_reconstruction_error": metrics["std_reconstruction_error"],
        "max_reconstruction_error": metrics["max_reconstruction_error"],
        "threshold": model.threshold,
        "n_components": float(model.n_components_selected),
        "explained_variance_ratio": metrics["explained_variance_ratio"],
        "n_samples": float(len(X)),
        "n_normal": float(len(X_normal)),
        "n_anomaly": float(int(np.sum(y))),
    }

    if "accuracy" in metrics:
        training_metrics.update(
            {
                "accuracy": metrics["accuracy"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "false_positive_rate": metrics["false_positive_rate"],
                "true_positives": metrics["true_positives"],
                "false_positives": metrics["false_positives"],
                "true_negatives": metrics["true_negatives"],
                "false_negatives": metrics["false_negatives"],
            }
        )

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="anomaly-detection",
        model_version=model_version,
        model_type="anomaly_detection",
        metrics=training_metrics,
        parameters={
            "n_components": n_components,
            "threshold_method": threshold_method,
            "threshold_percentile": threshold_percentile,
            "threshold_iqr_multiplier": threshold_iqr_multiplier,
            "random_seed": random_seed,
        },
        artifacts={
            f"anomaly_detection_model_v{model_version}.npz": model_path,
            "training_data.csv": model_dir / "training_data.csv",
        },
        tags={"framework": "numpy", "task": "anomaly_detection", "method": "pca"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="anomaly-detection",
            model_version=model_version,
            metrics=training_metrics,
            params={
                "n_components": n_components,
                "threshold_method": threshold_method,
                "threshold_percentile": threshold_percentile,
                "threshold_iqr_multiplier": threshold_iqr_multiplier,
                "random_seed": random_seed,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"anomaly_detection_v{model_version}.png"),
                "training_data": str(model_dir / "training_data.csv"),
            },
            tags={"model_type": "anomaly_detection", "framework": "numpy", "method": "pca"},
        )
        logger.info("Registered model to MLflow", model="anomaly-detection", version=model_version)

    return training_metrics


def _save_chart(
    model: PCAAnomalyDetector,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    version: str,
) -> None:
    """Save the anomaly detection visualization chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if model.components is None:
        return

    # Project data to 2D using first 2 principal components
    projected = model.transform(X)
    errors = model.reconstruction_error(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: PCA projection colored by anomaly
    ax1 = axes[0]
    normal_mask = y == 0
    anomaly_mask = y == 1

    ax1.scatter(
        projected[normal_mask, 0],
        projected[normal_mask, 1],
        c="steelblue",
        s=30,
        alpha=0.5,
        label="Normal",
    )
    ax1.scatter(
        projected[anomaly_mask, 0],
        projected[anomaly_mask, 1],
        c="crimson",
        s=50,
        alpha=0.8,
        marker="x",
        label="Anomaly",
    )
    ax1.set_xlabel(f"PC1 ({model.explained_variance_ratio[0]:.1%} variance)")
    ax1.set_ylabel(f"PC2 ({model.explained_variance_ratio[1]:.1%} variance)")
    ax1.set_title(f"PCA Projection - v{version}")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Reconstruction error histogram with threshold
    ax2 = axes[1]
    ax2.hist(
        errors[normal_mask],
        bins=50,
        alpha=0.6,
        label="Normal",
        color="steelblue",
        density=True,
    )
    ax2.hist(
        errors[anomaly_mask],
        bins=50,
        alpha=0.6,
        label="Anomaly",
        color="crimson",
        density=True,
    )
    ax2.axvline(
        model.threshold,
        color="black",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({model.threshold:.2f})",
    )
    ax2.set_xlabel("Reconstruction Error")
    ax2.set_ylabel("Density")
    ax2.set_title(f"Reconstruction Error Distribution - v{version}")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    chart_path = output_dir / f"anomaly_detection_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description="Train PCA anomaly detection model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-components", type=str, default=os.getenv("N_COMPONENTS", "0.95"))
    parser.add_argument(
        "--threshold-method", type=str, default=os.getenv("THRESHOLD_METHOD", "percentile")
    )
    parser.add_argument(
        "--threshold-percentile", type=float, default=float(os.getenv("THRESHOLD_PERCENTILE", "95"))
    )
    parser.add_argument(
        "--threshold-iqr-multiplier",
        type=float,
        default=float(os.getenv("THRESHOLD_IQR_MULTIPLIER", "1.5")),
    )
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument(
        "--register-mlflow",
        action="store_true",
        default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true",
    )
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    # Parse n_components (could be int or float)
    n_components: int | float
    try:
        n_components = int(args.n_components)
    except ValueError:
        n_components = float(args.n_components)

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_components=n_components,
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        threshold_iqr_multiplier=args.threshold_iqr_multiplier,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )

    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
