"""Production-grade configuration management for MLOps pipelines.

Implements 12-factor config with:
- Environment variable overrides
- YAML config files
- Nested config sections
- Validation and type coercion
- Secrets redaction
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import Any, TypeVar, get_type_hints

import yaml

T = TypeVar("T", bound="BaseConfig")

# Secrets that should never be logged
SENSITIVE_KEYS = {"password", "secret", "token", "api_key", "apikey", "credential"}


def _redact(value: Any, key: str = "") -> Any:
    """Redact sensitive values for logging."""
    if any(s in key.lower() for s in SENSITIVE_KEYS):
        return "***REDACTED***"
    if isinstance(value, dict):
        return {k: _redact(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [_redact(v) for v in value]
    return value


@dataclass
class BaseConfig:
    """Base configuration with env var support."""

    @classmethod
    def from_env(cls: type[T], prefix: str = "") -> T:
        """Create config from environment variables.

        Maps env vars like PREFIX_FIELD_NAME to dataclass fields.
        Supports nested dataclasses via PREFIX_SECTION_FIELD.
        """
        type_hints = get_type_hints(cls)
        kwargs: dict[str, Any] = {}

        for f in fields(cls):
            env_key = f"{prefix}_{f.name.upper()}" if prefix else f.name.upper()
            env_val = os.getenv(env_key)
            if env_val is not None:
                kwargs[f.name] = _coerce_value(env_val, type_hints.get(f.name, f.type))

        return cls(**kwargs)

    @classmethod
    def from_yaml(cls: type[T], path: Path) -> T:
        """Load config from a YAML file."""
        with open(path) as f:
            data = yaml.safe_load(f) or {}
        return cls(**data)

    @classmethod
    def from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Create config from a dictionary."""
        return cls(**data)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary (with secrets redacted)."""
        return _redact(asdict(self))

    def to_yaml(self, path: Path) -> None:
        """Save config to YAML file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False)

    def to_json(self) -> str:
        """Convert to JSON string (with secrets redacted)."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    def validate(self) -> list[str]:
        """Validate config. Returns list of validation errors."""
        errors: list[str] = []
        for f in fields(self):
            val = getattr(self, f.name)
            if (
                val is None
                and f.default is None
                and f.default_factory is field(default_factory=lambda: None)
            ):
                errors.append(f"Field '{f.name}' is required")
        return errors


def _coerce_value(value: str, target_type: Any) -> Any:
    """Coerce a string env var to the target type."""
    if target_type is bool or target_type == "bool":
        return value.lower() in ("true", "1", "yes", "on")
    if target_type is int or target_type == "int":
        return int(value)
    if target_type is float or target_type == "float":
        return float(value)
    if target_type is Path or target_type == "Path":
        return Path(value)
    if target_type is list[str] or target_type == "List[str]":
        return [v.strip() for v in value.split(",") if v.strip()]
    if target_type is dict[str, Any] or target_type == "Dict[str, Any]":
        return json.loads(value)
    return value


@dataclass
class MLflowConfig(BaseConfig):
    """MLflow tracking configuration."""

    tracking_uri: str = "http://mlflow:5000"
    registry_uri: str = "http://mlflow:5000"
    experiment_name: str = "mlops"
    artifact_root: str = "/mlflow/artifacts"
    s3_endpoint_url: str | None = None
    s3_access_key: str | None = None
    s3_secret_key: str | None = None
    s3_bucket: str | None = None


@dataclass
class ModelRegistryConfig(BaseConfig):
    """Model registry configuration."""

    base_dir: Path = Path("/models")
    default_status: str = "staging"
    max_versions: int = 5
    auto_promote: bool = False
    promotion_threshold: float = 0.9


@dataclass
class MonitoringConfig(BaseConfig):
    """Monitoring and observability configuration."""

    metrics_port: int = 8000
    enable_prometheus: bool = True
    drift_threshold: float = 0.2
    drift_check_interval: int = 100
    log_level: str = "INFO"
    enable_request_logging: bool = True


@dataclass
class DataConfig(BaseConfig):
    """Data pipeline configuration."""

    data_path: Path | None = None
    validation_enabled: bool = True
    cache_enabled: bool = True
    cache_dir: Path = Path("/tmp/mlops-cache")
    feature_store_path: Path | None = None


@dataclass
class Config(BaseConfig):
    """Top-level application configuration."""

    model_name: str = "default"
    model_version: str = "1.0.0"
    environment: str = "development"  # development, staging, production
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)
    registry: ModelRegistryConfig = field(default_factory=ModelRegistryConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    data: DataConfig = field(default_factory=DataConfig)

    @classmethod
    def load(cls: type[Config], path: Path | None = None, prefix: str = "MLOPS") -> Config:
        """Load config from YAML file and/or environment variables.

        Priority: env vars > YAML file > defaults
        """
        config = cls()

        # Load from YAML if provided
        if path and path.exists():
            config = cls.from_yaml(path)

        # Override with env vars
        env_config = cls.from_env(prefix)
        for f in fields(cls):
            env_val = getattr(env_config, f.name)
            if env_val is not None and env_val != getattr(cls(), f.name):
                setattr(config, f.name, env_val)

        # Load nested configs from env
        config.mlflow = MLflowConfig.from_env(f"{prefix}_MLFLOW")
        config.registry = ModelRegistryConfig.from_env(f"{prefix}_REGISTRY")
        config.monitoring = MonitoringConfig.from_env(f"{prefix}_MONITORING")
        config.data = DataConfig.from_env(f"{prefix}_DATA")

        return config


def load_config(path: Path | None = None, prefix: str = "MLOPS") -> Config:
    """Load configuration from file or environment."""
    return Config.load(path=path, prefix=prefix)
