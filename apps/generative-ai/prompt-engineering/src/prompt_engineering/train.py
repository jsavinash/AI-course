"""Training pipeline for Prompt Engineering."""

import argparse
import os
from pathlib import Path

from mlops_shared.logging import get_logger, setup_logging
from mlops_shared.model_registry import ModelRegistry
from mlops_shared.validation import DataValidator

from prompt_engineering.data import load_prompt_dataset, save_dataset, train_test_split
from prompt_engineering.model import PromptEngineeringModel, PromptTemplate, PromptTechnique, PromptEvaluator, PromptOptimizer

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    vocab_size: int = 1000,
    model_id: str = "prompt-engineering-v1",
    base_model_name: str = "default",
    technique: str = "zero-shot",
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -> dict:
    logger.info("Loading prompt dataset", n_samples=n_samples, technique=technique)
    X, y = load_prompt_dataset(data_path=data_path, n_samples=n_samples, random_seed=random_seed)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=random_seed)
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(X, y, model_dir / "training_data.npz")

    model = PromptEngineeringModel(model_id=model_id, base_model_name=base_model_name)
    model._init()

    template = PromptTemplate(
        template_id="default",
        template_text="Analyze the following: {input_text}",
        placeholders=["input_text"],
        technique=technique,
    )
    model.register_template(template)
    model.set_technique(technique)

    prompt = model.generate_prompt("default", input_text="Sample prompt for testing")
    logger.info("Generated prompt", prompt=prompt[:100])

    evaluator = PromptEvaluator()
    test_responses = [
        ("Sample response 1", "Expected output 1"),
        ("Sample response 2", "Expected output 2"),
        ("Sample response 3", "Expected output 3"),
    ]
    for response, expected in test_responses:
        scores = model.evaluate_prompt(prompt, response, expected)
        logger.info("Evaluation scores", scores=scores)

    avg_scores = evaluator.get_average_scores()
    logger.info("Average evaluation scores", scores=avg_scores)

    if len(test_responses) > 0:
        optimized_prompt = model.optimize_prompt(prompt, test_responses)
        logger.info("Optimized prompt", optimized=optimized_prompt[:100])

    model_path = model_dir / f"prompt_engineering_v{model_version}.json"
    model.save(str(model_path))

    metrics = {
        "n_samples": float(len(X)),
        "n_train": float(len(X_train)),
        "n_test": float(len(X_test)),
        "vocab_size": float(vocab_size),
        "technique": technique,
        "n_templates": float(len(model.templates)),
        "n_techniques": float(len(model.techniques)),
        "avg_relevance": avg_scores.get("relevance", 0.0),
        "avg_clarity": avg_scores.get("clarity", 0.0),
        "avg_completeness": avg_scores.get("completeness", 0.0),
        "avg_accuracy": avg_scores.get("accuracy", 0.0),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="prompt-engineering",
        model_version=model_version,
        model_type="nlp",
        metrics=metrics,
        parameters={
            "model_id": model_id,
            "base_model_name": base_model_name,
            "technique": technique,
            "n_samples": n_samples,
            "vocab_size": vocab_size,
            "random_seed": random_seed,
        },
        artifacts={f"prompt_engineering_v{model_version}.json": model_path, "training_data.npz": model_dir / "training_data.npz"},
        tags={"framework": "numpy", "task": "prompt_engineering", "model_type": "PromptEngineering"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="prompt-engineering",
            model_version=model_version,
            metrics=metrics,
            params={"model_id": model_id, "technique": technique, "n_samples": n_samples},
            artifacts={"model": str(model_path)},
            tags={"model_type": "prompt_engineering", "framework": "numpy"},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Prompt Engineering Model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "500")))
    parser.add_argument("--vocab-size", type=int, default=int(os.getenv("VOCAB_SIZE", "1000")))
    parser.add_argument("--model-id", type=str, default=os.getenv("MODEL_ID", "prompt-engineering-v1"))
    parser.add_argument("--base-model-name", type=str, default=os.getenv("BASE_MODEL_NAME", "default"))
    parser.add_argument("--technique", type=str, default=os.getenv("TECHNIQUE", "zero-shot"), choices=["zero-shot", "few-shot", "chain-of-thought", "self-ask", "least-to-most", "meta-prompting", "context-amplification", "iterative"])
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
        base_model_name=args.base_model_name,
        technique=args.technique,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )
    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
