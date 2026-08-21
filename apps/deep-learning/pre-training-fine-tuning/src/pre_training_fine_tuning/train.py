"""Training pipeline for Pre-training and Fine-tuning."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from pre_training_fine_tuning.data import (
    VOCAB_SIZE,
    generate_mlm_data,
    generate_ntp_data,
    generate_synthetic_data,
    save_training_data,
    train_test_split,
)
from pre_training_fine_tuning.model import Transformer

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    d_model: int = 128,
    n_heads: int = 4,
    n_encoder_layers: int = 2,
    n_decoder_layers: int = 2,
    d_ff: int = 512,
    max_seq_len: int = 32,
    learning_rate: float = 0.001,
    n_iterations: int = 100,
    weight_decay: float = 0.01,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
    phase: str = "pretrain",
    objective: str = "ntp",
    strategy: str = "full",
) -> dict:
    """Train model for pre-training or fine-tuning.

    Args:
        phase: "pretrain" or "finetune"
        objective: "mlm" or "ntp" (for pre-training)
        strategy: "full", "feature_extraction", "partial", "peft" (for fine-tuning)
    """
    if phase == "pretrain":
        if objective == "mlm":
            X, y, mask_positions = generate_mlm_data(n_samples=n_samples, vocab_size=VOCAB_SIZE, random_seed=random_seed)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_seed=random_seed)
            logger.info("Generated MLM pre-training data", n_samples=n_samples, mask_prob=0.15)
        else:
            X, y = generate_ntp_data(n_samples=n_samples, vocab_size=VOCAB_SIZE, random_seed=random_seed)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_seed=random_seed)
            mask_positions = None
            logger.info("Generated NTP pre-training data", n_samples=n_samples)
    else:
        X, y = generate_synthetic_data(n_samples=n_samples, vocab_size=VOCAB_SIZE, random_seed=random_seed, phase="finetune")
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_seed=random_seed)
        mask_positions = None
        logger.info("Generated fine-tuning data", n_samples=n_samples, strategy=strategy)

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / "training_data.npz")

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )

    model.fit(
        X_train,
        y_train,
        phase=phase,
        objective=objective,
        strategy=strategy,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        mask_positions=mask_positions if phase == "pretrain" and objective == "mlm" else None,
    )

    test_metrics = model.evaluate(X_test, y_test, phase=phase)
    logger.info("Training complete", training_mode=model.training_mode, final_loss=model.loss_history[-1])

    model_path = model_dir / f"model_v{model_version}.npz"
    model.save(str(model_path))

    metrics = {
        **test_metrics,
        "training_mode": phase,
        "objective": objective if phase == "pretrain" else "finetune",
        "strategy": strategy if phase == "finetune" else "none",
        "n_epochs_run": float(len(model.loss_history)),
        "final_loss": model.loss_history[-1] if model.loss_history else 0.0,
        "n_train_samples": float(len(X_train)),
        "n_test_samples": float(len(X_test)),
        "vocab_size": float(VOCAB_SIZE),
        "d_model": float(d_model),
        "n_heads": float(n_heads),
        "n_encoder_layers": float(n_encoder_layers),
        "n_decoder_layers": float(n_decoder_layers),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="pre-training-fine-tuning",
        model_version=model_version,
        model_type="transformer",
        metrics=metrics,
        parameters={
            "vocab_size": VOCAB_SIZE,
            "d_model": d_model,
            "n_heads": n_heads,
            "n_encoder_layers": n_encoder_layers,
            "n_decoder_layers": n_decoder_layers,
            "d_ff": d_ff,
            "max_seq_len": max_seq_len,
            "learning_rate": learning_rate,
            "n_iterations": n_iterations,
            "weight_decay": weight_decay,
            "random_seed": random_seed,
            "phase": phase,
            "objective": objective,
            "strategy": strategy,
        },
        artifacts={
            f"model_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        },
        tags={"framework": "numpy", "task": "pre_training_fine_tuning", "model_type": "Transformer", "phase": phase},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="pre-training-fine-tuning",
            model_version=model_version,
            metrics=metrics,
            params={"vocab_size": VOCAB_SIZE, "d_model": d_model, "n_heads": n_heads, "n_iterations": n_iterations},
            artifacts={"model": str(model_path)},
            tags={"model_type": "transformer", "framework": "numpy", "phase": phase},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Pre-training and Fine-tuning Transformer")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument("--d-model", type=int, default=int(os.getenv("D_MODEL", "128")))
    parser.add_argument("--n-heads", type=int, default=int(os.getenv("N_HEADS", "4")))
    parser.add_argument("--n-encoder-layers", type=int, default=int(os.getenv("N_ENCODER_LAYERS", "2")))
    parser.add_argument("--n-decoder-layers", type=int, default=int(os.getenv("N_DECODER_LAYERS", "2")))
    parser.add_argument("--d-ff", type=int, default=int(os.getenv("D_FF", "512")))
    parser.add_argument("--max-seq-len", type=int, default=int(os.getenv("MAX_SEQ_LEN", "32")))
    parser.add_argument("--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.001")))
    parser.add_argument("--n-iterations", type=int, default=int(os.getenv("N_ITERATIONS", "100")))
    parser.add_argument("--weight-decay", type=float, default=float(os.getenv("WEIGHT_DECAY", "0.01")))
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--test-size", type=float, default=float(os.getenv("TEST_SIZE", "0.2")))
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument("--phase", type=str, default=os.getenv("PHASE", "pretrain"), choices=["pretrain", "finetune"])
    parser.add_argument("--objective", type=str, default=os.getenv("OBJECTIVE", "ntp"), choices=["mlm", "ntp"])
    parser.add_argument("--strategy", type=str, default=os.getenv("STRATEGY", "full"), choices=["full", "feature_extraction", "partial", "peft"])
    parser.add_argument("--register-mlflow", action="store_true", default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true")
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_samples=args.n_samples,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        test_size=args.test_size,
        random_seed=args.random_seed,
        phase=args.phase,
        objective=args.objective,
        strategy=args.strategy,
    )
    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
