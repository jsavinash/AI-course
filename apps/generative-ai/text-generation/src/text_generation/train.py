"""Training pipeline for Text Generation."""

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from text_gen.data import load_text_dataset, save_dataset, train_test_split
from text_gen.model import TextGenerationModel

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    vocab_size: int = 1000,
    model_id: str = "text-generation-v1",
    temperature: float = 0.8,
    top_k: int = 50,
    top_p: float = 0.9,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -> dict:
    logger.info("Loading text dataset", n_samples=n_samples, temperature=temperature)
    X, y = load_text_dataset(data_path=data_path, n_samples=n_samples, random_seed=random_seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=random_seed)
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(X, y, model_dir / "training_data.npz")

    model = TextGenerationModel(
        model_id=model_id,
        vocab_size=vocab_size,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        random_seed=random_seed,
    )
    model._init()

    metrics = model.fit(X_train, y_train)
    logger.info("Training finished", metrics=metrics)

    eval_metrics = model.evaluate(X_test, y_test)
    logger.info("Evaluation metrics", metrics=eval_metrics)

    model_path = model_dir / f"text_generation_v{model_version}.npz"
    model.save(str(model_path))

    combined_metrics = {**metrics, **eval_metrics}
    combined_metrics.update({
        "temperature": temperature,
        "top_k": float(top_k),
        "top_p": top_p,
        "n_samples": float(n_samples),
        "vocab_size": float(vocab_size),
    })

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="text-generation",
        model_version=model_version,
        model_type="generative",
        metrics=combined_metrics,
        parameters={
            "model_id": model_id,
            "vocab_size": vocab_size,
            "temperature": temperature,
            "top_k": top_k,
            "top_p": top_p,
            "n_samples": n_samples,
            "random_seed": random_seed,
        },
        artifacts={f"text_generation_v{model_version}.npz": model_path, "training_data.npz": model_dir / "training_data.npz"},
        tags={"framework": "numpy", "task": "text_generation", "model_type": "TextGeneration"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="text-generation",
            model_version=model_version,
            metrics=combined_metrics,
            params={"model_id": model_id, "temperature": temperature, "top_k": top_k, "top_p": top_p, "n_samples": n_samples},
            artifacts={"model": str(model_path)},
            tags={"model_type": "text_generation", "framework": "numpy"},
        )

    return combined_metrics


def main():
    parser = argparse.ArgumentParser(description="Train Text Generation Model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument("--vocab-size", type=int, default=int(os.getenv("VOCAB_SIZE", "1000")))
    parser.add_argument("--model-id", type=str, default=os.getenv("MODEL_ID", "text-generation-v1"))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("TEMPERATURE", "0.8")))
    parser.add_argument("--top-k", type=int, default=int(os.getenv("TOP_K", "50")))
    parser.add_argument("--top-p", type=float, default=float(os.getenv("TOP_P", "0.9")))
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
        model_id=args.model_id,
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )
    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
