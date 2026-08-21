"""Training pipeline for GNN Social Network Analysis."""

import argparse
import os
from pathlib import Path

import numpy as np
from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_gnn_social_networks_schema

from gnn_social_networks.data import (
    N_CLASSES,
    N_FEATURES,
    generate_synthetic_data,
    save_training_data,
)
from gnn_social_networks.model import GNNSocialNetworks

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_nodes: int = 20,
    hidden_dim: int = 16,
    learning_rate: float = 0.05,
    n_iterations: int = 200,
    weight_decay: float = 0.001,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -> dict:
    X, A, y = generate_synthetic_data(
        n_samples=n_nodes, n_nodes=n_nodes, n_features=N_FEATURES, random_seed=random_seed
    )
    logger.info("Generated graph data", n_nodes=n_nodes, data_path=str(data_path))

    validator = DataValidator(create_gnn_social_networks_schema())
    validation = validator.validate(X)
    if not validation.valid:
        logger.error("Training data validation failed", errors=validation.errors)
        raise ValueError(f"Training data validation failed: {validation.errors}")

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, A, y, model_dir / "training_data.npz")

    model = GNNSocialNetworks(
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X, A, y)

    metrics = model.evaluate(X, A, y)
    logger.info("Training complete", training_mode=model.training_mode, final_loss=model.loss_history[-1])

    model_path = model_dir / f"gnn_model_v{model_version}.npz"
    model.save(str(model_path))
    np.savez(model_dir / "adjacency_matrix.npz", A=A)

    metrics_summary = {
        **metrics,
        "training_mode": "supervised",
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "n_nodes": float(n_nodes),
        "hidden_dim": float(hidden_dim),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="gnn-social-networks",
        model_version=model_version,
        model_type="classification",
        metrics=metrics_summary,
        parameters={
            "n_features": N_FEATURES,
            "n_classes": N_CLASSES,
            "hidden_dim": hidden_dim,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "n_nodes": n_nodes,
            "random_seed": random_seed,
        },
        artifacts={
            f"gnn_model_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
            "adjacency_matrix.npz": model_dir / "adjacency_matrix.npz",
        },
        tags={"framework": "numpy", "task": "gnn_social_networks", "model_type": "GNN"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="gnn-social-networks",
            model_version=model_version,
            metrics=metrics_summary,
            params={"n_features": N_FEATURES, "n_classes": N_CLASSES, "hidden_dim": hidden_dim, "learning_rate": learning_rate, "n_iterations": n_iterations},
            artifacts={"model": str(model_path)},
            tags={"model_type": "gnn", "framework": "numpy"},
        )

    return metrics_summary


def main():
    parser = argparse.ArgumentParser(description="Train GNN Social Network model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-nodes", type=int, default=int(os.getenv("N_NODES", "20")))
    parser.add_argument("--hidden-dim", type=int, default=int(os.getenv("HIDDEN_DIM", "16")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.05")))
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "200")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.001")))
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
        n_nodes=args.n_nodes,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )
    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
