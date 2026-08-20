"""Production training pipeline for the recommendation engine (Apriori)."""

import argparse
import os
from pathlib import Path

from mlops_shared.logging import get_logger, setup_logging
from mlops_shared.model_registry import ModelRegistry

from recommendation_engine.data import load_training_data, save_training_data
from recommendation_engine.model import Apriori

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path,
    min_support: float,
    min_confidence: float,
    min_lift: float,
    max_itemset_size: int,
    model_version: str,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -> dict:
    """Train the recommendation engine Apriori model and save artifacts.

    Returns:
        Dictionary with training metrics
    """
    # Load training data
    transactions = load_training_data(data_path, random_seed=random_seed)
    logger.info("Loaded training data", n_transactions=len(transactions))

    # Save training data for reproducibility
    save_training_data(transactions, model_dir / "training_data.csv")

    # Train model
    model = Apriori(
        min_support=min_support,
        min_confidence=min_confidence,
        min_lift=min_lift,
        max_itemset_size=max_itemset_size,
    )
    model.fit(transactions)

    # Evaluate model quality
    metrics = model.evaluate(transactions)
    logger.info(
        "Training complete",
        n_rules=metrics["n_rules"],
        n_frequent_itemsets=metrics["n_frequent_itemsets"],
        coverage=metrics["coverage"],
        avg_confidence=metrics["avg_confidence"],
        avg_lift=metrics["avg_lift"],
    )

    # Model validation - check rule quality
    if metrics["n_rules"] == 0:
        logger.warning(
            "No rules generated. Consider lowering min_support or min_confidence.",
            min_support=min_support,
            min_confidence=min_confidence,
        )

    # Save model
    model_path = model_dir / f"recommendation_model_v{model_version}.npz"
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, transactions, model_dir, model_version)

    # Combined metrics for registry
    training_metrics = {
        "n_rules": metrics["n_rules"],
        "n_frequent_itemsets": metrics["n_frequent_itemsets"],
        "coverage": metrics["coverage"],
        "avg_confidence": metrics["avg_confidence"],
        "avg_lift": metrics["avg_lift"],
        "avg_support": metrics["avg_support"],
        "n_transactions": float(len(transactions)),
        "n_products": float(len(model.products)),
    }

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="recommendation-engine",
        model_version=model_version,
        model_type="association_rules",
        metrics=training_metrics,
        parameters={
            "min_support": min_support,
            "min_confidence": min_confidence,
            "min_lift": min_lift,
            "max_itemset_size": max_itemset_size,
            "random_seed": random_seed,
        },
        artifacts={
            f"recommendation_model_v{model_version}.npz": model_path,
            "training_data.csv": model_dir / "training_data.csv",
        },
        tags={"framework": "numpy", "task": "association_rules"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="recommendation-engine",
            model_version=model_version,
            metrics=training_metrics,
            params={
                "min_support": min_support,
                "min_confidence": min_confidence,
                "min_lift": min_lift,
                "max_itemset_size": max_itemset_size,
                "random_seed": random_seed,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"recommendation_engine_v{model_version}.png"),
                "training_data": str(model_dir / "training_data.csv"),
            },
            tags={"model_type": "association_rules", "framework": "numpy"},
        )
        logger.info(
            "Registered model to MLflow", model="recommendation-engine", version=model_version
        )

    return training_metrics


def _save_chart(
    model: Apriori,
    transactions: list[list[str]],
    output_dir: Path,
    version: str,
) -> None:
    """Save the association rules chart."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    if not model.rules:
        return

    plt.figure(figsize=(12, 6))

    # Plot top rules by lift
    top_rules = model.rules[:15]
    labels = [
        f"{'+'.join(sorted(r.antecedent))} -> {'+'.join(sorted(r.consequent))}" for r in top_rules
    ]
    lifts = [r.lift for r in top_rules]
    confidences = [r.confidence for r in top_rules]

    x = range(len(top_rules))
    bars = plt.bar(x, lifts, color="steelblue", alpha=0.7, label="Lift")

    # Add confidence as text on bars
    for _i, (bar, conf) in enumerate(zip(bars, confidences, strict=False)):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.05,
            f"conf={conf:.2f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    plt.xlabel("Association Rule")
    plt.ylabel("Lift")
    plt.title(f"Top Association Rules by Lift - v{version}")
    plt.xticks(x, labels, rotation=45, ha="right", fontsize=8)
    plt.grid(True, alpha=0.3, axis="y")
    plt.legend()
    plt.tight_layout()

    chart_path = output_dir / f"recommendation_engine_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description="Train recommendation engine Apriori model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument(
        "--min-support", type=float, default=float(os.getenv("MIN_SUPPORT", "0.05"))
    )
    parser.add_argument(
        "--min-confidence", type=float, default=float(os.getenv("MIN_CONFIDENCE", "0.5"))
    )
    parser.add_argument("--min-lift", type=float, default=float(os.getenv("MIN_LIFT", "1.0")))
    parser.add_argument(
        "--max-itemset-size", type=int, default=int(os.getenv("MAX_ITEMSET_SIZE", "4"))
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
        min_support=args.min_support,
        min_confidence=args.min_confidence,
        min_lift=args.min_lift,
        max_itemset_size=args.max_itemset_size,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )

    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
