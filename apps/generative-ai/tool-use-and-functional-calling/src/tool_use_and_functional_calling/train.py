"""Training pipeline for Tool Use and Functional Calling.

Demonstrates the complete 5-step function-calling workflow:
1. Register tools (tool definitions)
2. LLM tool decision (select tool + extract arguments)
3. Application-side execution
4. Result concatenation
5. Final generation
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
from mlops_shared.logging import get_logger, setup_logging
from mlops_shared.model_registry import ModelRegistry
from mlops_shared.validation import DataValidator

from tool_use_and_functional_calling.data import (
    DEFAULT_N_SAMPLES,
    DEFAULT_VOCAB_SIZE,
    load_tool_dataset,
    save_dataset,
    train_test_split,
)
from tool_use_and_functional_calling.model import ToolSpec, ToolUseModel

logger = get_logger(__name__)


def train(
    model_dir: Path,
    n_samples: int = DEFAULT_N_SAMPLES,
    vocab_size: int = DEFAULT_VOCAB_SIZE,
    n_tools: int = 5,
    model_id: str = "tool-use-and-functional-calling-v1",
    base_model_name: str = "tool-use-v1",
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -> dict:
    logger.info("Loading tool-call dataset", n_samples=n_samples, n_tools=n_tools)
    X, y = load_tool_dataset(
        n_samples=n_samples,
        vocab_size=vocab_size,
        random_seed=random_seed,
    )

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=random_seed)
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(X, y, model_dir / "tool_use_training_data.npz")

    model = ToolUseModel(model_id=model_id, base_model_name=base_model_name)
    model._init()

    for tool_name in list(model.tools.keys())[:n_tools]:
        spec = model.get_tool(tool_name)
        if spec:
            model.register_tool(spec)

    logger.info("Registered tools", n_tools=len(model.tools), tools=list(model.tools.keys()))

    test_queries = [
        "What is the current TCS stock price?",
        "What is the weather in Mumbai?",
        "Calculate 25 + 37",
        "What is the status of order ORD-12345?",
        "Show me all users who signed up in the last 7 days",
        "What is the weather in London?",
        "Calculate 100 divided by 4",
        "Track my order ORD-54321",
    ]

    logger.info("Running function-calling workflow on test queries")
    workflow_results = []
    for query in test_queries:
        result = model.invoke(query)
        workflow_results.append(result)
        logger.info(
            "Workflow result",
            query=query[:50],
            tool_called=result.get("tool_call", {}).get("tool_name") if result.get("tool_call") else None,
            success=result.get("success"),
            response=result.get("final_response", "")[:60],
        )

    n_successful = sum(1 for r in workflow_results if r.get("success"))
    n_with_tool_call = sum(1 for r in workflow_results if r.get("tool_call") is not None)

    logger.info(
        "Workflow summary",
        total=len(workflow_results),
        successful=n_successful,
        with_tool_call=n_with_tool_call,
    )

    sample_queries = X_test[:5]
    predicted_tools = []
    for query_tokens in sample_queries:
        query_text = " ".join([str(int(t)) for t in query_tokens if int(t) > 0])
        tool_call = model.decide_tool(query_text)
        predicted_tools.append(tool_call.tool_name if tool_call else "none")

    model_path = model_dir / f"tool_use_v{model_version}.json"
    model.save(str(model_path))

    metrics = {
        "n_samples": float(len(X)),
        "n_train": float(len(X_train)),
        "n_test": float(len(X_test)),
        "vocab_size": float(vocab_size),
        "n_tools": float(len(model.tools)),
        "n_workflow_queries": float(len(workflow_results)),
        "n_successful_workflows": float(n_successful),
        "n_tool_calls": float(n_with_tool_call),
        "accuracy": float(n_with_tool_call) / max(len(workflow_results), 1),
        "success_rate": float(n_successful) / max(len(workflow_results), 1),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="tool-use-and-functional-calling",
        model_version=model_version,
        model_type="tool_use",
        metrics=metrics,
        parameters={
            "model_id": model_id,
            "base_model_name": base_model_name,
            "n_tools": n_tools,
            "n_samples": n_samples,
            "vocab_size": vocab_size,
            "random_seed": random_seed,
        },
        artifacts={
            f"tool_use_v{model_version}.json": model_path,
            "tool_use_training_data.npz": model_dir / "tool_use_training_data.npz",
        },
        tags={"framework": "numpy", "task": "tool_use_and_functional_calling", "model_type": "ToolUseModel"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="tool-use-and-functional-calling",
            model_version=model_version,
            metrics=metrics,
            params={"model_id": model_id, "n_tools": n_tools, "n_samples": n_samples},
            artifacts={"model": str(model_path)},
            tags={"model_type": "tool_use", "framework": "numpy"},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train Tool Use and Functional Calling Model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", str(DEFAULT_N_SAMPLES))))
    parser.add_argument("--vocab-size", type=int, default=int(os.getenv("VOCAB_SIZE", str(DEFAULT_VOCAB_SIZE))))
    parser.add_argument("--n-tools", type=int, default=int(os.getenv("N_TOOLS", "5")))
    parser.add_argument("--model-id", type=str, default=os.getenv("MODEL_ID", "tool-use-and-functional-calling-v1"))
    parser.add_argument("--base-model-name", type=str, default=os.getenv("BASE_MODEL_NAME", "tool-use-v1"))
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument("--register-mlflow", action="store_true", default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true")
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        n_samples=args.n_samples,
        vocab_size=args.vocab_size,
        n_tools=args.n_tools,
        model_id=args.model_id,
        base_model_name=args.base_model_name,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )
    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
