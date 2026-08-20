"""Production-grade model registry and lifecycle management."""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

MODEL_STAGES = ("staging", "production", "archived", "champion", "candidate")
TRANSITIONS = {
    "staging": ("production", "archived"),
    "production": ("archived", "staging"),
    "archived": ("staging",),
    "champion": ("candidate", "archived"),
    "candidate": ("champion", "archived"),
}


@dataclass
class ModelInfo:
    """Metadata about a trained model."""

    model_name: str
    model_version: str
    model_type: str  # "regression" or "classification"
    created_at: str
    metrics: dict[str, float]
    parameters: dict[str, Any]
    artifact_path: str
    mlflow_run_id: str | None = None
    mlflow_model_uri: str | None = None
    status: str = "staging"
    created_by: str | None = None
    tags: dict[str, str] = field(default_factory=dict)


class ModelRegistry:
    """Manage model artifacts, versioning, and MLflow integration."""

    def __init__(
        self,
        base_dir: Path = Path("/models"),
        mlflow_uri: str | None = None,
        mlflow_experiment: str | None = None,
    ):
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        if mlflow_uri or mlflow_experiment:
            try:
                import mlflow
                if mlflow_uri:
                    mlflow.set_tracking_uri(mlflow_uri)
                if mlflow_experiment:
                    mlflow.set_experiment(mlflow_experiment)
            except Exception as e:
                import logging
                logging.getLogger(__name__).warning(f"Could not configure MLflow: {e}")

    # ---------- Artifact management ----------

    def save_model(
        self,
        model_name: str,
        model_version: str,
        model_type: str,
        metrics: dict[str, float],
        parameters: dict[str, Any],
        artifacts: dict[str, Path],
        status: str = "staging",
        created_by: str | None = None,
        tags: dict[str, str] | None = None,
    ) -> ModelInfo:
        """Save a trained model and its artifacts to the registry."""
        if status not in MODEL_STAGES:
            raise ValueError(f"Invalid model status: {status}. Must be one of {MODEL_STAGES}")

        model_dir = self.base_dir / model_name / model_version
        model_dir.mkdir(parents=True, exist_ok=True)

        # Copy artifacts to the model directory
        for artifact_name, artifact_path in artifacts.items():
            src = Path(artifact_path)
            dest = model_dir / artifact_name
            if src.is_file():
                shutil.copy2(src, dest)
            elif src.is_dir():
                shutil.copytree(src, dest, dirs_exist_ok=True)

        model_info = ModelInfo(
            model_name=model_name,
            model_version=model_version,
            model_type=model_type,
            created_at=datetime.now(UTC).isoformat(),
            metrics=metrics,
            parameters=parameters,
            artifact_path=str(model_dir),
            status=status,
            created_by=created_by,
            tags=tags or {},
        )

        self._write_info(model_name, model_version, model_info)
        self._enforce_max_versions(model_name)
        return model_info

    def _write_info(self, model_name: str, model_version: str, info: ModelInfo) -> None:
        model_dir = self.base_dir / model_name / model_version
        info_path = model_dir / "model_info.json"
        with open(info_path, "w") as f:
            json.dump(asdict(info), f, indent=2)

    def _enforce_max_versions(self, model_name: str, max_versions: int = 5) -> None:
        """Keep only the N most recent versions, archiving the rest."""
        versions = sorted(
            (v for v in (self.base_dir / model_name).iterdir() if v.is_dir()),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if len(versions) > max_versions:
            for stale in versions[max_versions:]:
                info_path = stale / "model_info.json"
                if info_path.exists():
                    with open(info_path) as f:
                        info = json.load(f)
                    if info["status"] == "production":
                        continue  # Never delete production models
                    info["status"] = "archived"
                    with open(info_path, "w") as f:
                        json.dump(info, f, indent=2)

    def load_model(self, model_name: str, model_version: str) -> dict[str, Any]:
        """Load model artifacts from the registry."""
        model_dir = self.base_dir / model_name / model_version
        if not model_dir.exists():
            raise FileNotFoundError(f"Model {model_name}:{model_version} not found")

        with open(model_dir / "model_info.json") as f:
            model_info = json.load(f)

        npz_files = list(model_dir.glob("*.npz")) + list(model_dir.glob("*.pkl")) + list(model_dir.glob("*.joblib"))
        if not npz_files:
            raise FileNotFoundError(f"No model weights found in {model_dir}")

        if npz_files[0].suffix == ".npz":
            data = np.load(npz_files[0])
            weights = {k: data[k] for k in data.files}
        else:
            weights = {"path": str(npz_files[0])}

        return {
            "model_info": model_info,
            "weights": weights,
            "model_dir": model_dir,
        }

    def list_models(self) -> list[dict[str, Any]]:
        """List all models in the registry."""
        models = []
        if not self.base_dir.exists():
            return models
        for model_dir in sorted(self.base_dir.iterdir(), key=lambda p: p.name):
            if model_dir.is_dir():
                for version_dir in sorted(model_dir.iterdir(), key=lambda p: p.name):
                    info_path = version_dir / "model_info.json"
                    if info_path.exists():
                        with open(info_path) as f:
                            models.append(json.load(f))
        return models

    def get_model(self, model_name: str, model_version: str) -> dict[str, Any]:
        """Get model info for a specific version."""
        info_path = self.base_dir / model_name / model_version / "model_info.json"
        if not info_path.exists():
            raise FileNotFoundError(f"Model {model_name}:{model_version} not found")
        with open(info_path) as f:
            return json.load(f)

    def get_latest(self, model_name: str, status: str | None = None) -> dict[str, Any]:
        """Get the latest model version, optionally filtering by status."""
        models = [m for m in self.list_models() if m["model_name"] == model_name]
        if status:
            models = [m for m in models if m["status"] == status]
        if not models:
            raise FileNotFoundError(f"No models found for {model_name} with status={status}")
        models.sort(key=lambda m: m["model_version"], reverse=True)
        return models[0]

    def list_versions(self, model_name: str) -> list[dict[str, Any]]:
        """List all versions of a specific model."""
        return [m for m in self.list_models() if m["model_name"] == model_name]

    # ---------- Lifecycle management ----------

    def transition(self, model_name: str, model_version: str, new_status: str) -> None:
        """Transition a model to a new lifecycle stage."""
        if new_status not in MODEL_STAGES:
            raise ValueError(f"Invalid target status: {new_status}")
        model_info = self.get_model(model_name, model_version)
        current = model_info["status"]
        if new_status not in TRANSITIONS.get(current, ()):
            raise ValueError(
                f"Cannot transition model from '{current}' to '{new_status}'. "
                f"Allowed transitions from {current}: {TRANSITIONS.get(current, ())}"
            )
        model_info["status"] = new_status
        model_info["updated_at"] = datetime.now(UTC).isoformat()
        self.base_dir / model_name / model_version / "model_info.json"
        with open(self.base_dir / model_name / model_version / "model_info.json", "w") as f:
            json.dump(model_info, f, indent=2)

    def promote_to_production(self, model_name: str, model_version: str) -> None:
        """Convenience method: promote model to production."""
        # Archive any existing production model for this name
        for m in self.list_models():
            if m["model_name"] == model_name and m["status"] == "production":
                self.transition(model_name, m["model_version"], "archived")
        self.transition(model_name, model_version, "production")

    # ---------- MLflow integration ----------

    def log_to_mlflow(
        self,
        model_name: str,
        model_version: str,
        metrics: dict[str, float],
        params: dict[str, Any],
        artifacts: dict[str, str],
        tags: dict[str, str] | None = None,
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ) -> str | None:
        """Log model training to MLflow and register the model with retry resilience."""
        import logging
        import time

        log = logging.getLogger(__name__)

        try:
            import mlflow
        except ImportError:
            log.warning("mlflow is not installed. Skipping MLflow logging.")
            return None

        for attempt in range(1, max_retries + 1):
            try:
                with mlflow.start_run(run_name=f"{model_name}-{model_version}") as run:
                    # Tags
                    if tags:
                        mlflow.set_tags(tags)
                    mlflow.set_tag("model_name", model_name)
                    mlflow.set_tag("model_version", model_version)

                    # Parameters & metrics
                    for key, value in params.items():
                        mlflow.log_param(key, value)
                    for key, value in metrics.items():
                        mlflow.log_metric(key, value)

                    # Artifacts
                    for artifact_name, artifact_path in artifacts.items():
                        p = Path(artifact_path)
                        if p.exists():
                            mlflow.log_artifact(str(p), artifact_name)

                    # Register model
                    try:
                        model_uri = f"runs:/{run.info.run_id}"
                        mlflow.register_model(
                            model_uri=model_uri,
                            name=model_name,
                            tags={"version": model_version},
                        )
                    except Exception as reg_err:
                        log.warning(f"MLflow model registration failed: {reg_err}")

                    return run.info.run_id
            except Exception as e:
                log.warning(f"MLflow logging attempt {attempt}/{max_retries} failed: {e}")
                if attempt < max_retries:
                    time.sleep(retry_delay)
                else:
                    log.error(f"Failed to log to MLflow after {max_retries} attempts. Continuing without MLflow.")
                    return None
