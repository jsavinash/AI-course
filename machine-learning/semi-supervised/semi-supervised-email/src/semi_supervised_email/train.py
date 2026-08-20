"""Production training pipeline for semi-supervised email classification."""

import argparse
import os
from pathlib import Path

import numpy as np
from mlops_shared.logging import get_logger, setup_logging
from mlops_shared.model_registry import ModelRegistry

from semi_supervised_email.data import (
    load_training_data,
    save_training_data,
)
from semi_supervised_email.model import SelfTrainingClassifier

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path,
    labeled_ratio: float,
    confidence_threshold: float,
    max_iterations: int,
    model_version: str,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -> dict:
    """Train the semi-supervised email classification model and save artifacts.

    Returns:
        Dictionary with training metrics
    """
    # Load semi-supervised training data
    X, y, is_labeled = load_training_data(
        data_path=data_path if data_path and data_path.exists() else None,
        labeled_ratio=labeled_ratio,
        random_seed=random_seed,
    )
    logger.info(
        "Loaded semi-supervised training data",
        n_samples=len(X),
        n_features=X.shape[1],
        n_labeled=int(np.sum(is_labeled)),
        n_unlabeled=int(np.sum(~is_labeled)),
        labeled_ratio=labeled_ratio,
    )

    # Save training data for reproducibility
    save_training_data(X, y, is_labeled, model_dir / "training_data.csv")

    # Train self-training model
    model = SelfTrainingClassifier(
        confidence_threshold=confidence_threshold,
        max_iterations=max_iterations,
        random_seed=random_seed,
    )
    model.fit(X, y)

    training_mode = model.training_mode
    n_iterations = model.n_iterations_used
    n_labeled_final = model.n_labeled_history[-1] if model.n_labeled_history else np.sum(is_labeled)

    logger.info(
        "Self-training complete",
        training_mode=training_mode,
        n_iterations=n_iterations,
        n_labeled_initial=int(np.sum(is_labeled)),
        n_labeled_final=n_labeled_final,
        n_pseudo_labeled=n_labeled_final - int(np.sum(is_labeled)),
    )

    # Evaluate on all labeled data
    X_labeled, y_labeled = _get_labeled_data(X, y)
    metrics = model.evaluate(X_labeled, y_labeled)

    # Add semi-supervised specific metrics
    metrics.update({
        "training_mode": float(training_mode == "semi-supervised"),
        "n_labeled_initial": float(np.sum(is_labeled)),
        "n_labeled_final": float(n_labeled_final),
        "n_pseudo_labeled": float(n_labeled_final - np.sum(is_labeled)),
        "n_unlabeled_initial": float(np.sum(~is_labeled)),
        "n_iterations": float(n_iterations),
        "confidence_threshold": confidence_threshold,
        "labeled_ratio": labeled_ratio,
    })

    # Save model
    model_path = model_dir / f"semi_supervised_email_model_v{model_version}.npz"
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, model_dir, model_version)

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="semi-supervised-email",
        model_version=model_version,
        model_type="semi_supervised_classification",
        metrics=metrics,
        parameters={
            "labeled_ratio": labeled_ratio,
            "confidence_threshold": confidence_threshold,
            "max_iterations": max_iterations,
            "random_seed": random_seed,
        },
        artifacts={
            f"semi_supervised_email_model_v{model_version}.npz": model_path,
            "training_data.csv": model_dir / "training_data.csv",
        },
        tags={"framework": "numpy", "task": "semi_supervised_classification", "base_model": "logistic_regression"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="semi-supervised-email",
            model_version=model_version,
            metrics=metrics,
            params={
                "labeled_ratio": labeled_ratio,
                "confidence_threshold": confidence_threshold,
                "max_iterations": max_iterations,
                "random_seed": random_seed,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"semi_supervised_email_v{model_version}.png"),
                "training_data": str(model_dir / "training_data.csv"),
            },
            tags={"model_type": "semi_supervised_classification", "framework": "numpy"},
        )
        logger.info("Registered model to MLflow", model="semi-supervised-email", version=model_version)

    return metrics


def _get_labeled_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Extract only the labeled subset of the data."""
    mask = y != -1
    return X[mask], y[mask]


def _save_chart(model: SelfTrainingClassifier, output_dir: Path, version: str) -> None:
    """Save the semi-supervised training chart."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not model.n_labeled_history:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Labeled samples over iterations
    iterations = list(range(len(model.n_labeled_history)))
    ax1.plot(iterations, model.n_labeled_history, marker="o", color="steelblue", linewidth=2)
    ax1.set_xlabel("Self-Training Iteration")
    ax1.set_ylabel("Number of Labeled Samples")
    ax1.set_title("Labeled Samples Growth")
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy over iterations (if available)
    if model.accuracy_history:
        ax2.plot(iterations[:len(model.accuracy_history)], model.accuracy_history, marker="s", color="green", linewidth=2)
        ax2.set_xlabel("Self-Training Iteration")
        ax2.set_ylabel("Accuracy")
        ax2.set_title("Model Accuracy During Self-Training")
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, "No accuracy data available", ha="center", va="center", transform=ax2.transAxes)
        ax2.set_title("Model Accuracy During Self-Training")

    plt.tight_layout()

    chart_path = output_dir / f"semi_supervised_email_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description="Train semi-supervised email classification model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--labeled-ratio", type=float, default=float(os.getenv("LABELED_RATIO", "0.1")))
    parser.add_argument("--confidence-threshold", type=float, default=float(os.getenv("CONFIDENCE_THRESHOLD", "0.95")))
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("MAX_ITERATIONS", "10")))
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
        labeled_ratio=args.labeled_ratio,
        confidence_threshold=args.confidence_threshold,
        max_iterations=args.max_iterations,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )

    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
