"""Production training pipeline for self-supervised server monitoring.

Trains a denoising autoencoder to reconstruct normal server metrics from
corrupted inputs. The self-supervised signal comes from the data itself -
no human labels are required for training.
"""

import argparse
import os
from pathlib import Path

import numpy as np
from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from self_supervised_monitoring.data import (
    generate_synthetic_data,
    save_training_data,
)
from self_supervised_monitoring.model import DenoisingAutoencoder

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 2000,
    hidden_dim: int = 16,
    learning_rate: float = 0.01,
    n_iterations: int = 5000,
    noise_rate: float = 0.25,
    threshold_percentile: float = 95.0,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -> dict:
    """Train the self-supervised denoising autoencoder and save artifacts.

    The model is trained on normal server metrics only. Anomalies are
    detected at inference time via high reconstruction error.

    Returns:
        Dictionary with training metrics
    """
    # Generate or load data
    # For self-supervised training, we use only the anomaly-free portion
    X_full, y_full = generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)

    # Separate normal and anomalous data
    X_normal = X_full[y_full == 0]
    X_anomaly = X_full[y_full == 1]

    # Split normal data for train/validation
    rng = np.random.default_rng(random_seed)
    n_val = max(1, int(len(X_normal) * 0.2))
    val_idx = rng.choice(len(X_normal), size=n_val, replace=False)
    val_mask = np.zeros(len(X_normal), dtype=bool)
    val_mask[val_idx] = True

    X_train = X_normal[~val_mask]
    X_val = X_normal[val_mask]

    # Split anomaly data for test evaluation
    n_test_anomaly = max(1, int(len(X_anomaly) * 0.5))
    test_anom_idx = rng.choice(len(X_anomaly), size=n_test_anomaly, replace=False)
    X_test_anomaly = X_anomaly[test_anom_idx]
    y_test_anomaly = np.ones(n_test_anomaly, dtype=int)

    # Use some normal data for test too
    test_norm_idx = rng.choice(len(X_normal), size=n_test_anomaly, replace=False)
    X_test_normal = X_normal[test_norm_idx]
    y_test_normal = np.zeros(n_test_anomaly, dtype=int)

    # Combine test set
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.concatenate([y_test_normal, y_test_anomaly])

    logger.info(
        "Loaded self-supervised training data",
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        n_features=X_train.shape[1],
        training_mode="self-supervised (denoising autoencoder)",
    )

    # Save full dataset for reproducibility
    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X_full, y_full, model_dir / "training_data.csv")

    # Train self-supervised model
    model = DenoisingAutoencoder(
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        noise_rate=noise_rate,
        random_seed=random_seed,
    )
    model.threshold_percentile = threshold_percentile
    model.fit(X_train, X_val=X_val, X_test=X_test, y_test=y_test)

    # Compute metrics
    test_metrics = model.evaluate(X_test, y_test)
    train_errors = model.reconstruction_error(X_train)
    val_errors = model.reconstruction_error(X_val)

    metrics = {
        **test_metrics,
        "training_mode": "self-supervised",
        "n_train_samples": float(len(X_train)),
        "n_val_samples": float(len(X_val)),
        "n_test_samples": float(len(X_test)),
        "n_anomaly_test": float(np.sum(y_test == 1)),
        "n_normal_test": float(np.sum(y_test == 0)),
        "train_mean_recon_error": float(np.mean(train_errors)),
        "train_max_recon_error": float(np.max(train_errors)),
        "val_mean_recon_error": float(np.mean(val_errors)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "n_epochs_run": float(len(model.loss_history)),
        "reconstruction_threshold": float(model.threshold),
        "threshold_percentile": float(model.threshold_percentile),
        "noise_rate": float(noise_rate),
        "hidden_dim": float(hidden_dim),
        "learning_rate": float(learning_rate),
    }

    logger.info(
        "Self-supervised training complete",
        training_mode="self-supervised",
        n_epochs=len(model.loss_history),
        final_loss=model.loss_history[-1] if model.loss_history else 0.0,
        threshold=model.threshold,
        test_accuracy=test_metrics["accuracy"],
    )

    # Save model
    model_path = model_dir / f"self_supervised_monitoring_model_v{model_version}.npz"
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, model_dir, model_version)

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="self-supervised-monitoring",
        model_version=model_version,
        model_type="self_supervised_anomaly_detection",
        metrics=metrics,
        parameters={
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "noise_rate": noise_rate,
            "threshold_percentile": threshold_percentile,
            "random_seed": random_seed,
        },
        artifacts={
            f"self_supervised_monitoring_model_v{model_version}.npz": model_path,
            "training_data.csv": model_dir / "training_data.csv",
        },
        tags={
            "framework": "numpy",
            "task": "self_supervised_anomaly_detection",
            "base_model": "denoising_autoencoder",
        },
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="self-supervised-monitoring",
            model_version=model_version,
            metrics=metrics,
            params={
                "hidden_dim": hidden_dim,
                "learning_rate": learning_rate,
                "n_iterations": n_iterations,
                "noise_rate": noise_rate,
                "threshold_percentile": threshold_percentile,
                "random_seed": random_seed,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"self_supervised_monitoring_v{model_version}.png"),
                "training_data": str(model_dir / "training_data.csv"),
            },
            tags={"model_type": "self_supervised_anomaly_detection", "framework": "numpy"},
        )
        logger.info(
            "Registered model to MLflow", model="self-supervised-monitoring", version=model_version
        )

    return metrics


def _save_chart(model: DenoisingAutoencoder, output_dir: Path, version: str) -> None:
    """Save the training loss chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color="steelblue", linewidth=1.5)
    ax.set_xlabel("Training Iteration")
    ax.set_ylabel("Reconstruction Loss (MSE)")
    ax.set_title("Self-Supervised Denoising Autoencoder Training Loss")
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")

    plt.tight_layout()
    chart_path = output_dir / f"self_supervised_monitoring_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(
        description="Train self-supervised monitoring model (denoising autoencoder)"
    )
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "2000")))
    parser.add_argument("--hidden-dim", type=int, default=int(os.getenv("HIDDEN_DIM", "16")))
    parser.add_argument(
        "--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.01"))
    )
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "5000")))
    parser.add_argument("--noise-rate", type=float, default=float(os.getenv("NOISE_RATE", "0.25")))
    parser.add_argument(
        "--threshold-percentile",
        type=float,
        default=float(os.getenv("THRESHOLD_PERCENTILE", "95.0")),
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

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_samples=args.n_samples,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        noise_rate=args.noise_rate,
        threshold_percentile=args.threshold_percentile,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )

    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
