"""Production training pipeline for market segmentation (unsupervised K-Means)."""

import argparse
import os
from pathlib import Path

import numpy as np
from mlops_shared.logging import get_logger, setup_logging
from mlops_shared.model_registry import ModelRegistry
from mlops_shared.validation import DataValidator, create_market_segmentation_schema

from market_segmentation.data import load_training_data, save_training_data
from market_segmentation.model import KMeans

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path,
    n_clusters: int,
    max_iterations: int,
    n_init: int,
    model_version: str,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -> dict:
    """Train the market segmentation K-Means model and save artifacts.

    Returns:
        Dictionary with training metrics
    """
    # Load training data
    X, y = load_training_data(data_path)
    logger.info("Loaded training data", n_samples=len(X), n_features=X.shape[1])

    # Validate training data
    validator = DataValidator(create_market_segmentation_schema())
    validation = validator.validate(X)
    if not validation.valid:
        logger.error("Training data validation failed", errors=validation.errors)
        raise ValueError(f"Training data validation failed: {validation.errors}")
    logger.info("Training data validated", stats=validation.stats)

    # Save training data for reproducibility
    save_training_data(X, y, model_dir / "training_data.csv")

    # Train model
    model = KMeans(
        n_clusters=n_clusters,
        max_iterations=max_iterations,
        n_init=n_init,
        random_seed=random_seed,
    )
    model.fit(X)

    # Evaluate clustering quality
    metrics = model.evaluate(X)
    logger.info(
        "Training complete",
        n_clusters=model.n_clusters,
        inertia=model.inertia,
        silhouette=metrics["silhouette"],
        n_iterations_used=model.n_iterations_used,
    )

    # Model validation - check silhouette score
    if metrics["silhouette"] < 0.1:
        logger.warning(
            "Model silhouette score below threshold",
            silhouette=metrics["silhouette"],
            threshold=0.1,
        )

    # Save model
    model_path = model_dir / f"market_segmentation_model_v{model_version}.npz"
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, X, model_dir, model_version)

    # Combined metrics for registry
    training_metrics = {
        "inertia": metrics["inertia"],
        "silhouette": metrics["silhouette"],
        "n_clusters": float(n_clusters),
        "n_samples": len(X),
        "n_iterations_used": float(model.n_iterations_used),
    }

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="market-segmentation",
        model_version=model_version,
        model_type="clustering",
        metrics=training_metrics,
        parameters={
            "n_clusters": n_clusters,
            "max_iterations": max_iterations,
            "n_init": n_init,
            "random_seed": random_seed,
        },
        artifacts={
            f"market_segmentation_model_v{model_version}.npz": model_path,
            "training_data.csv": model_dir / "training_data.csv",
        },
        tags={"framework": "numpy", "task": "clustering"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="market-segmentation",
            model_version=model_version,
            metrics=training_metrics,
            params={
                "n_clusters": n_clusters,
                "max_iterations": max_iterations,
                "n_init": n_init,
                "random_seed": random_seed,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"market_segmentation_v{model_version}.png"),
                "training_data": str(model_dir / "training_data.csv"),
            },
            tags={"model_type": "clustering", "framework": "numpy"},
        )
        logger.info(
            "Registered model to MLflow", model="market-segmentation", version=model_version
        )

    return training_metrics


def _save_chart(model: KMeans, X: np.ndarray, output_dir: Path, version: str) -> None:
    """Save the clustering chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if model.centroids is None:
        return

    plt.figure(figsize=(10, 6))

    # Plot data points colored by cluster
    labels = model.predict(X)
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap="viridis",
        s=50,
        alpha=0.6,
        label="Customers",
    )

    # Plot centroids
    # Need to unstandardize centroids for plotting
    if model.feature_mean is not None and model.feature_std is not None:
        centroids_orig = model.centroids * model.feature_std + model.feature_mean
        plt.scatter(
            centroids_orig[:, 0],
            centroids_orig[:, 1],
            c="red",
            marker="X",
            s=200,
            edgecolors="black",
            linewidths=2,
            label="Centroids",
        )

    plt.colorbar(scatter, label="Cluster")
    plt.xlabel("Annual Income (k$)")
    plt.ylabel("Spending Score (0-100)")
    plt.title(f"Market Segmentation Clusters - v{version}")
    plt.grid(True, alpha=0.3)
    plt.legend()

    chart_path = output_dir / f"market_segmentation_v{version}.png"
    plt.tight_layout()
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description="Train market segmentation K-Means model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-clusters", type=int, default=int(os.getenv("N_CLUSTERS", "5")))
    parser.add_argument(
        "--max-iterations", type=int, default=int(os.getenv("MAX_ITERATIONS", "300"))
    )
    parser.add_argument("--n-init", type=int, default=int(os.getenv("N_INIT", "10")))
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
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
        n_clusters=args.n_clusters,
        max_iterations=args.max_iterations,
        n_init=args.n_init,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )

    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
