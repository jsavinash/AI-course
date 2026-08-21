"""Training pipeline for Transfer Learning."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_transfer_learning_schema

from transfer_learning.data import (
    generate_synthetic_data,
    load_dataset,
    save_dataset,
    train_test_split,
)
from transfer_learning.model import TransferLearningModel

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    vocab_size: int = 1000,
    seq_len: int = 32,
    d_model: int = 128,
    n_heads: int = 4,
    n_base_layers: int = 2,
    d_ff: int = 512,
    max_seq_len: int = 32,
    n_classes: int = 10,
    freeze_base: bool = True,
    fine_tune_layers: int = 0,
    learning_rate: float = 0.001,
    fine_tune_lr: float = 0.0001,
    n_iterations: int = 100,
    weight_decay: float = 0.01,
    model_version: str = "1.0.0",
    fine_tune_at: int | None = None,
    random_seed: int = 42,
    register_to_mlflow: bool = False,
) -> dict:
    logger.info("Loading data", n_samples=n_samples)
    if data_path and Path(data_path).exists():
        X, y = load_dataset(data_path)
    else:
        X, y = generate_synthetic_data(n_samples=n_samples, vocab_size=vocab_size, seq_len=seq_len, n_classes=n_classes, random_seed=random_seed)

    validator = DataValidator(create_transfer_learning_schema())
    validation = validator.validate(X.reshape(-1, X.shape[-1]))
    if not validation.valid:
        logger.error("Data validation failed", errors=validation.errors)
        raise ValueError(f"Data validation failed: {validation.errors}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=random_seed)
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(X, y, model_dir / "training_data.npz")

    model = TransferLearningModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_base_layers=n_base_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        n_classes=n_classes,
        freeze_base=freeze_base,
        fine_tune_layers=fine_tune_layers,
        learning_rate=learning_rate,
        fine_tune_lr=fine_tune_lr,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )

    logger.info("Starting transfer learning training", freeze_base=freeze_base, fine_tune_layers=fine_tune_layers)
    model.fit(X_train, y_train, n_iterations=n_iterations, fine_tune_at=fine_tune_at)

    test_metrics = model.evaluate(X_test, y_test)
    logger.info("Training complete", final_loss=model.loss_history[-1], test_accuracy=test_metrics["accuracy"])

    model_path = model_dir / f"transfer_learning_v{model_version}.npz"
    model.save(str(model_path))

    metrics = {
        **test_metrics,
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "n_train_samples": float(len(X_train)),
        "n_test_samples": float(len(X_test)),
        "vocab_size": float(vocab_size),
        "d_model": float(d_model),
        "n_base_layers": float(n_base_layers),
        "d_ff": float(d_ff),
        "n_classes": float(n_classes),
        "freeze_base": 1.0 if freeze_base else 0.0,
        "fine_tune_layers": float(fine_tune_layers),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="transfer-learning",
        model_version=model_version,
        model_type="classification",
        metrics=metrics,
        parameters={
            "vocab_size": vocab_size,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_base_layers": n_base_layers,
            "d_ff": d_ff,
            "max_seq_len": max_seq_len,
            "n_classes": n_classes,
            "freeze_base": freeze_base,
            "fine_tune_layers": fine_tune_layers,
            "learning_rate": learning_rate,
            "fine_tune_lr": fine_tune_lr,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
            "fine_tune_at": fine_tune_at,
        },
        artifacts={
            f"transfer_learning_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        },
        tags={"framework": "numpy", "task": "transfer_learning", "model_type": "TransferLearning"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="transfer-learning",
            model_version=model_version,
            metrics=metrics,
            params={"vocab_size": vocab_size, "d_model": d_model, "freeze_base": freeze_base, "fine_tune_layers": fine_tune_layers},
            artifacts={"model": str(model_path)},
            tags={"model_type": "transfer_learning", "framework": "numpy"},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Transfer Learning Model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument("--vocab-size", type=int, default=int(os.getenv("VOCAB_SIZE", str(1000))))
    parser.add_argument("--seq-len", type=int, default=int(os.getenv("SEQ_LEN", "32")))
    parser.add_argument("--d-model", type=int, default=int(os.getenv("D_MODEL", "128")))
    parser.add_argument("--n-heads", type=int, default=int(os.getenv("N_HEADS", "4")))
    parser.add_argument("--n-base-layers", type=int, default=int(os.getenv("N_BASE_LAYERS", "2")))
    parser.add_argument("--d-ff", type=int, default=int(os.getenv("D_FF", "512")))
    parser.add_argument("--max-seq-len", type=int, default=int(os.getenv("MAX_SEQ_LEN", "32")))
    parser.add_argument("--n-classes", type=int, default=int(os.getenv("N_CLASSES", "10")))
    parser.add_argument("--freeze-base", action="store_true", default=os.getenv("FREEZE_BASE", "true").lower() == "true")
    parser.add_argument("--no-freeze-base", dest="freeze_base", action="store_false")
    parser.add_argument("--fine-tune-layers", type=int, default=int(os.getenv("FINE_TUNE_LAYERS", "0")))
    parser.add_argument("--fine-tune-at", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.001")))
    parser.add_argument("--fine-tune-lr", type=float, default=float(os.getenv("FINE_TUNE_LR", "0.0001")))
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "100")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.01")))
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
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_base_layers=args.n_base_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        n_classes=args.n_classes,
        freeze_base=args.freeze_base,
        fine_tune_layers=args.fine_tune_layers,
        learning_rate=args.learning_rate,
        fine_tune_lr=args.fine_tune_lr,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        fine_tune_at=args.fine_tune_at,
        random_seed=args.random_seed,
        register_to_mlflow=args.register_mlflow,
    )
    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
