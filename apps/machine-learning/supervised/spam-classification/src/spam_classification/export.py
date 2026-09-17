"""Export step of the model lifecycle for spam classification.

Lifecycle: train.py -> MLflow -> export.py -> artifacts/ -> FastAPI -> prediction
                                              (signed file)

Reads the trained `spam_model_v<version>.npz` plus the registry's
`model_info.json`, packages them into a single HMAC-SHA256-signed
`model.signed` file under `<model-dir>/spam-classification/<version>/`, and
records the signed artifact back to MLflow so every served artifact is
traceable to a training run.

Usage::

    python -m spam_classification.export \
        --model-dir ./artifacts/models --model-version 1.0.0
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from ai_core.export import export_signed_model, get_git_sha
from ai_core.logging import get_logger, setup_logging

logger = get_logger(__name__)

MODEL_NAME = "spam-classification"


def export(
    model_dir: Path,
    model_version: str,
    signing_key: str | None = None,
    mlflow_run_id: str | None = None,
) -> Path:
    """Package the trained model into a signed artifact.

    Returns:
        Path to the written ``model.signed`` file.
    """
    version_dir = model_dir / MODEL_NAME / model_version
    model_path = model_dir / f"spam_model_v{model_version}.npz"
    if not model_path.exists():
        # Fall back to the registry layout copy produced by train.py
        model_path = version_dir / f"spam_model_v{model_version}.npz"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found for version {model_version}. Run train.py first."
        )

    # Enrich the manifest with registry metadata (metrics/params from train.py)
    info: dict = {}
    info_path = version_dir / "model_info.json"
    if info_path.exists():
        info = json.loads(info_path.read_text())

    manifest = {
        "model_name": MODEL_NAME,
        "model_version": model_version,
        "model_type": info.get("model_type", "classification"),
        "metrics": info.get("metrics", {}),
        "parameters": info.get("parameters", {}),
        "mlflow_run_id": mlflow_run_id or info.get("mlflow_run_id"),
        "git_sha": get_git_sha(),
        "stage": info.get("status", "staging"),
    }

    signed_path = export_signed_model(
        model_path=model_path,
        out_dir=version_dir,
        manifest=manifest,
        signing_key=signing_key,
    )
    logger.info("Export complete", signed_file=str(signed_path))
    return signed_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Export signed spam-classification model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--signing-key", type=str, default=os.getenv("MODEL_SIGNING_KEY"))
    parser.add_argument(
        "--mlflow-run-id", type=str, default=os.getenv("MLFLOW_RUN_ID"),
        help="MLflow run to attach the signed artifact to (defaults to registry metadata)",
    )
    parser.add_argument(
        "--record-mlflow",
        action="store_true",
        default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true",
        help="Log the signed artifact back to MLflow",
    )
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
    args = parser.parse_args()

    setup_logging(args.log_level)
    signed_path = export(
        model_dir=args.model_dir,
        model_version=args.model_version,
        signing_key=args.signing_key,
        mlflow_run_id=args.mlflow_run_id,
    )

    if args.record_mlflow:
        try:
            import mlflow

            with mlflow.start_run(run_name=f"{MODEL_NAME}-export-{args.model_version}") as run:
                mlflow.set_tag("model_version", args.model_version)
                mlflow.set_tag("lifecycle_stage", "export")
                mlflow.log_artifact(str(signed_path), "signed_model")
                mlflow.log_artifact(str(signed_path.parent / "manifest.json"), "signed_model")
                logger.info("Signed artifact recorded to MLflow", run_id=run.info.run_id)
        except Exception as e:
            logger.warning("MLflow recording of signed artifact failed", error=str(e))


if __name__ == "__main__":
    main()
