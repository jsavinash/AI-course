<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>anomaly-detection-pca - AI App Documentation</title>
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.js"></script>
<link rel="stylesheet" href="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/katex.min.css">
<script defer src="https://cdn.jsdelivr.net/npm/katex@0.16.9/dist/contrib/auto-render.min.js" onload="renderMath()"></script>
<script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
<style>
/* CSS styles here */
</style>
</head>
<body>
<section id="math" class="section math-section">
<h2><span class="section-icon">∫</span> Mathematics &amp; Theory</h2>
<p class="section-subtitle">Anomaly Detection / PCA — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$X_{\text{centered}} = X - \bar{x}$$</div>
<div class="math-block">$$\Sigma = \frac{1}{n} X_{\text{centered}}^T X_{\text{centered}}$$</div>
<div class="math-block">$$\Sigma v = \lambda v$$</div>
<div class="math-block">$$X_{\text{reduced}} = X_{\text{centered}} V_k$$</div>
<div class="math-block">$$\text{recon error} = \|X - X_{\text{reconstructed}}\|^2$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>PCA finds orthogonal directions of maximum variance. By computing the SVD of centered data $X = U\Sigma V^T$, the right singular vectors $V$ are the principal components. Anomalies are detected from large reconstruction error after projection.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive 2D/3D PCA projection; explained variance scree plot; anomaly score distribution.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  PCAAnomalyDetector</pre>
</div>
<div class="mermaid-wrapper">
<h3>Data Flow</h3>
<pre class="mermaid">graph TD
  A[Input Data] --> B[Preprocessing]
  B --> C[Model Training]
  C --> D[Evaluation]
  D --> E[Model Registry]
  E --> F[Serving API]</pre>
</div>
</section>
<section id="api" class="section api-section">
<h2><span class="section-icon">⚡</span> API Reference</h2>
<p class="section-subtitle">FastAPI endpoints and model interfaces</p>
<table class="api-table">
<thead><tr><th>Method</th><th>Endpoint</th></tr></thead>
<tbody><tr><td><code>GET</code></td><td><code>/</code></td></tr>
<tr><td><code>GET</code></td><td><code>/health</code></td></tr>
<tr><td><code>GET</code></td><td><code>/metrics</code></td></tr>
<tr><td><code>POST</code></td><td><code>/reload</code></td></tr></tbody>
</table>
</section>
<section id="usage" class="section usage-section">
<h2><span class="section-icon">▶</span> Usage</h2>
<p class="section-subtitle">Code examples and CLI commands</p>
<h3>Training Script</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-3510345627')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3510345627"><code class="language-python">&quot;&quot;&quot;Production training pipeline for PCA-based anomaly detection.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

import numpy as np
from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_anomaly_detection_schema

from anomaly_detection.data import load_training_data, save_training_data
from anomaly_detection.model import PCAAnomalyDetector

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path,
    n_components: int | float,
    threshold_method: str,
    threshold_percentile: float,
    threshold_iqr_multiplier: float,
    model_version: str,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -&gt; dict:
    &quot;&quot;&quot;Train the PCA anomaly detection model and save artifacts.

    Args:
        model_dir: Directory to save model artifacts
        data_path: Optional path to CSV data
        n_components: Number of PCA components or variance ratio to retain
        threshold_method: Method for anomaly threshold (&quot;percentile&quot;, &quot;iqr&quot;, &quot;fixed&quot;)
        threshold_percentile: Percentile for threshold if method=&quot;percentile&quot;
        threshold_iqr_multiplier: IQR multiplier if method=&quot;iqr&quot;
        model_version: Model version string
        register_to_mlflow: Whether to register to MLflow
        random_seed: Random seed for reproducibility

    Returns:
        Dictionary with training metrics
    &quot;&quot;&quot;
    # Load training data
    X, y = load_training_data(data_path, random_seed=random_seed)
    logger.info(&quot;Loaded training data&quot;, n_samples=len(X), n_features=X.shape[1])

    # Validate training data
    validator = DataValidator(create_anomaly_detection_schema())
    validation = validator.validate(X)
    if not validation.valid:
        logger.error(&quot;Training data validation failed&quot;, errors=validation.errors)
        raise ValueError(f&quot;Training data validation failed: {validation.errors}&quot;)
    logger.info(&quot;Training data validated&quot;, stats=validation.stats)

    # Save training data for reproducibility
    save_training_data(X, y, model_dir / &quot;training_data.csv&quot;)

    # Use only normal samples for PCA training (unsupervised anomaly detection)
    X_normal = X[y == 0]
    logger.info(&quot;Training on normal samples&quot;, n_normal=len(X_normal), n_anomaly=int(np.sum(y)))

    # Train model
    model = PCAAnomalyDetector(
        n_components=n_components,
        threshold_method=threshold_method,
        threshold_percentile=threshold_percentile,
        threshold_iqr_multiplier=threshold_iqr_multiplier,
        random_seed=random_seed,
    )
    model.fit(X_normal)

    # Evaluate on all data
    metrics = model.evaluate(X, y)
    logger.info(
        &quot;Training complete&quot;,
        n_components=model.n_components_selected,
        explained_variance=metrics[&quot;explained_variance_ratio&quot;],
        threshold=model.threshold,
        mean_error=metrics[&quot;mean_reconstruction_error&quot;],
        max_error=metrics[&quot;max_reconstruction_error&quot;],
    )

    if &quot;accuracy&quot; in metrics:
        logger.info(
            &quot;Evaluation metrics&quot;,
            accuracy=metrics[&quot;accuracy&quot;],
            precision=metrics[&quot;precision&quot;],
            recall=metrics[&quot;recall&quot;],
            f1=metrics[&quot;f1&quot;],
        )

    # Save model
    model_path = model_dir / f&quot;anomaly_detection_model_v{model_version}.npz&quot;
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, X, y, model_dir, model_version)

    # Combined metrics for registry
    training_metrics = {
        &quot;mean_reconstruction_error&quot;: metrics[&quot;mean_reconstruction_error&quot;],
        &quot;std_reconstruction_error&quot;: metrics[&quot;std_reconstruction_error&quot;],
        &quot;max_reconstruction_error&quot;: metrics[&quot;max_reconstruction_error&quot;],
        &quot;threshold&quot;: model.threshold,
        &quot;n_components&quot;: float(model.n_components_selected),
        &quot;explained_variance_ratio&quot;: metrics[&quot;explained_variance_ratio&quot;],
        &quot;n_samples&quot;: float(len(X)),
        &quot;n_normal&quot;: float(len(X_normal)),
        &quot;n_anomaly&quot;: float(int(np.sum(y))),
    }

    if &quot;accuracy&quot; in metrics:
        training_metrics.update(
            {
                &quot;accuracy&quot;: metrics[&quot;accuracy&quot;],
                &quot;precision&quot;: metrics[&quot;precision&quot;],
                &quot;recall&quot;: metrics[&quot;recall&quot;],
                &quot;f1&quot;: metrics[&quot;f1&quot;],
                &quot;false_positive_rate&quot;: metrics[&quot;false_positive_rate&quot;],
                &quot;true_positives&quot;: metrics[&quot;true_positives&quot;],
                &quot;false_positives&quot;: metrics[&quot;false_positives&quot;],
                &quot;true_negatives&quot;: metrics[&quot;true_negatives&quot;],
                &quot;false_negatives&quot;: metrics[&quot;false_negatives&quot;],
            }
        )

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;anomaly-detection&quot;,
        model_version=model_version,
        model_type=&quot;anomaly_detection&quot;,
        metrics=training_metrics,
        parameters={
            &quot;n_components&quot;: n_components,
            &quot;threshold_method&quot;: threshold_method,
            &quot;threshold_percentile&quot;: threshold_percentile,
            &quot;threshold_iqr_multiplier&quot;: threshold_iqr_multiplier,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;anomaly_detection_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.csv&quot;: model_dir / &quot;training_data.csv&quot;,
        },
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;anomaly_detection&quot;, &quot;method&quot;: &quot;pca&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;anomaly-detection&quot;,
            model_version=model_version,
            metrics=training_metrics,
            params={
                &quot;n_components&quot;: n_components,
                &quot;threshold_method&quot;: threshold_method,
                &quot;threshold_percentile&quot;: threshold_percentile,
                &quot;threshold_iqr_multiplier&quot;: threshold_iqr_multiplier,
                &quot;random_seed&quot;: random_seed,
            },
            artifacts={
                &quot;model&quot;: str(model_path),
                &quot;chart&quot;: str(model_dir / f&quot;anomaly_detection_v{model_version}.png&quot;),
                &quot;training_data&quot;: str(model_dir / &quot;training_data.csv&quot;),
            },
            tags={&quot;model_type&quot;: &quot;anomaly_detection&quot;, &quot;framework&quot;: &quot;numpy&quot;, &quot;method&quot;: &quot;pca&quot;},
        )
        logger.info(&quot;Registered model to MLflow&quot;, model=&quot;anomaly-detection&quot;, version=model_version)

    return training_metrics


def _save_chart(
    model: PCAAnomalyDetector,
    X: np.ndarray,
    y: np.ndarray,
    output_dir: Path,
    version: str,
) -&gt; None:
    &quot;&quot;&quot;Save the anomaly detection visualization chart.&quot;&quot;&quot;
    import matplotlib

    matplotlib.use(&quot;Agg&quot;)
    import matplotlib.pyplot as plt

    if model.components is None:
        return

    # Project data to 2D using first 2 principal components
    projected = model.transform(X)
    errors = model.reconstruction_error(X)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: PCA projection colored by anomaly
    ax1 = axes[0]
    normal_mask = y == 0
    anomaly_mask = y == 1

    ax1.scatter(
        projected[normal_mask, 0],
        projected[normal_mask, 1],
        c=&quot;steelblue&quot;,
        s=30,
        alpha=0.5,
        label=&quot;Normal&quot;,
    )
    ax1.scatter(
        projected[anomaly_mask, 0],
        projected[anomaly_mask, 1],
        c=&quot;crimson&quot;,
        s=50,
        alpha=0.8,
        marker=&quot;x&quot;,
        label=&quot;Anomaly&quot;,
    )
    ax1.set_xlabel(f&quot;PC1 ({model.explained_variance_ratio[0]:.1%} variance)&quot;)
    ax1.set_ylabel(f&quot;PC2 ({model.explained_variance_ratio[1]:.1%} variance)&quot;)
    ax1.set_title(f&quot;PCA Projection - v{version}&quot;)
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Plot 2: Reconstruction error histogram with threshold
    ax2 = axes[1]
    ax2.hist(
        errors[normal_mask],
        bins=50,
        alpha=0.6,
        label=&quot;Normal&quot;,
        color=&quot;steelblue&quot;,
        density=True,
    )
    ax2.hist(
        errors[anomaly_mask],
        bins=50,
        alpha=0.6,
        label=&quot;Anomaly&quot;,
        color=&quot;crimson&quot;,
        density=True,
    )
    ax2.axvline(
        model.threshold,
        color=&quot;black&quot;,
        linestyle=&quot;--&quot;,
        linewidth=2,
        label=f&quot;Threshold ({model.threshold:.2f})&quot;,
    )
    ax2.set_xlabel(&quot;Reconstruction Error&quot;)
    ax2.set_ylabel(&quot;Density&quot;)
    ax2.set_title(f&quot;Reconstruction Error Distribution - v{version}&quot;)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    chart_path = output_dir / f&quot;anomaly_detection_v{version}.png&quot;
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info(&quot;Chart saved&quot;, path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description=&quot;Train PCA anomaly detection model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-components&quot;, type=str, default=os.getenv(&quot;N_COMPONENTS&quot;, &quot;0.95&quot;))
    parser.add_argument(
        &quot;--threshold-method&quot;, type=str, default=os.getenv(&quot;THRESHOLD_METHOD&quot;, &quot;percentile&quot;)
    )
    parser.add_argument(
        &quot;--threshold-percentile&quot;, type=float, default=float(os.getenv(&quot;THRESHOLD_PERCENTILE&quot;, &quot;95&quot;))
    )
    parser.add_argument(
        &quot;--threshold-iqr-multiplier&quot;,
        type=float,
        default=float(os.getenv(&quot;THRESHOLD_IQR_MULTIPLIER&quot;, &quot;1.5&quot;)),
    )
    parser.add_argument(&quot;--model-version&quot;, type=str, default=os.getenv(&quot;MODEL_VERSION&quot;, &quot;1.0.0&quot;))
    parser.add_argument(&quot;--random-seed&quot;, type=int, default=int(os.getenv(&quot;RANDOM_SEED&quot;, &quot;42&quot;)))
    parser.add_argument(
        &quot;--register-mlflow&quot;,
        action=&quot;store_true&quot;,
        default=os.getenv(&quot;REGISTER_MLFLOW&quot;, &quot;false&quot;).lower() == &quot;true&quot;,
    )
    parser.add_argument(&quot;--log-level&quot;, type=str, default=os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    args = parser.parse_args()

    # Parse n_components (could be int or float)
    n_components: int | float
    try:
        n_components = int(args.n_components)
    except ValueError:
        n_components = float(args.n_components)

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_components=n_components,
        threshold_method=args.threshold_method,
        threshold_percentile=args.threshold_percentile,
        threshold_iqr_multiplier=args.threshold_iqr_multiplier,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )

    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-3043064947')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3043064947"><code class="language-python">&quot;&quot;&quot;Production serving API for PCA-based anomaly detection.&quot;&quot;&quot;

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from ai_core.drift import DriftDetector
from ai_core.fastapi_middleware import add_observability_middleware
from ai_core.logging import get_logger, setup_logging
from ai_core.metrics import MetricsCollector
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_anomaly_detection_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from anomaly_detection.data import FEATURE_NAMES
from anomaly_detection.model import PCAAnomalyDetector

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;METRICS_PORT&quot;, os.getenv(&quot;ANOMALY_METRICS_PORT&quot;, &quot;8005&quot;)))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class MetricsRequest(BaseModel):
    &quot;&quot;&quot;Single metrics observation for anomaly detection.&quot;&quot;&quot;

    request_count: float = Field(..., ge=0, description=&quot;Number of requests&quot;)
    bytes_per_request: float = Field(..., ge=0, description=&quot;Average bytes per request&quot;)
    cpu_usage: float = Field(..., ge=0, le=100, description=&quot;CPU usage percentage&quot;)
    memory_usage: float = Field(..., ge=0, le=100, description=&quot;Memory usage percentage&quot;)
    disk_io: float = Field(..., ge=0, description=&quot;Disk I/O operations per second&quot;)
    network_in: float = Field(..., ge=0, description=&quot;Network inbound MB/s&quot;)
    network_out: float = Field(..., ge=0, description=&quot;Network outbound MB/s&quot;)
    error_rate: float = Field(..., ge=0, le=100, description=&quot;Error rate percentage&quot;)
    connection_count: float = Field(..., ge=0, description=&quot;Active connections&quot;)
    response_time: float = Field(..., ge=0, description=&quot;Average response time in ms&quot;)


class MetricsBulkRequest(BaseModel):
    &quot;&quot;&quot;Bulk metrics request for anomaly detection.&quot;&quot;&quot;

    samples: list[MetricsRequest] = Field(..., min_length=1, max_length=100)


class AnomalyResponse(BaseModel):
    &quot;&quot;&quot;Anomaly detection response for a single observation.&quot;&quot;&quot;

    is_anomaly: bool
    anomaly_score: float
    anomaly_probability: float
    reconstruction_error: float
    anomaly_threshold: float
    model_version: str


class BulkAnomalyResponse(BaseModel):
    &quot;&quot;&quot;Bulk anomaly detection response.&quot;&quot;&quot;

    samples: list[AnomalyResponse]
    n_anomalies: int
    n_samples: int
    model_version: str


class StatsResponse(BaseModel):
    &quot;&quot;&quot;Model statistics response.&quot;&quot;&quot;

    n_features: int
    n_components: int
    explained_variance_ratio: float
    reconstruction_threshold: float
    threshold_method: str
    mean_reconstruction_error: float
    max_reconstruction_error: float
    model_version: str


class ModelInfoResponse(BaseModel):
    &quot;&quot;&quot;Model information response.&quot;&quot;&quot;

    n_components: int
    n_features: int
    feature_names: list[str]
    cumulative_variance_ratio: float
    reconstruction_threshold: float
    model_version: str


class DriftResponse(BaseModel):
    &quot;&quot;&quot;Drift detection response.&quot;&quot;&quot;

    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


# Global model state
_model: PCAAnomalyDetector | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_validator: DataValidator | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    &quot;&quot;&quot;Load model at startup and clean up at shutdown.&quot;&quot;&quot;
    global _model, _model_version, _metrics, _validator, _drift_detector, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;anomaly_detection&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_anomaly_detection_schema())
    _drift_detector = DriftDetector(
        feature_names=FEATURE_NAMES,
        feature_types={f: &quot;float&quot; for f in FEATURE_NAMES},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;anomaly-detection&quot;, model_version=_model_version, model_type=&quot;anomaly_detection&quot;
    )

    # Load reference data for drift detection
    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;anomaly-detection&quot;, version=_model_version)

    yield

    logger.info(&quot;Shutting down anomaly-detection API&quot;)


def _load_model() -&gt; tuple[PCAAnomalyDetector, str]:
    &quot;&quot;&quot;Load the latest model from the registry or model directory with resilient fallback.&quot;&quot;&quot;
    # 1. Try model registry
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            ad_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;anomaly-detection&quot;]
            if ad_models:
                ad_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = ad_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;anomaly_detection_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return PCAAnomalyDetector.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;anomaly-detection&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;anomaly_detection_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return PCAAnomalyDetector.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    # 2. Try direct model in MODEL_DIR
    npz_path = MODEL_DIR / &quot;anomaly_detection_model.npz&quot;
    if npz_path.exists():
        return PCAAnomalyDetector.load(str(npz_path)), &quot;legacy&quot;

    # 3. Try bundled artifacts directory
    candidate_paths = [
        Path(&quot;/app/artifacts/models/anomaly_detection_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3]
        / &quot;artifacts&quot;
        / &quot;models&quot;
        / &quot;anomaly_detection_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return PCAAnomalyDetector.load(str(p)), &quot;1.0.0-bundled&quot;

    # 4. In-memory baseline fallback (never crash cold start)
    logger.warning(&quot;No pre-existing model found on disk. Initializing baseline PCA model.&quot;)
    from anomaly_detection.data import load_training_data

    X_base, y_base = load_training_data(None)
    X_normal = X_base[y_base == 0]
    model = PCAAnomalyDetector(n_components=0.95, threshold_method=&quot;percentile&quot;, random_seed=42)
    model.fit(X_normal)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    &quot;&quot;&quot;Load reference training data for drift detection.&quot;&quot;&quot;
    candidate_csvs = [
        MODEL_DIR / &quot;anomaly-detection&quot; / _model_version / &quot;training_data.csv&quot;,
        MODEL_DIR / &quot;training_data.csv&quot;,
        Path(&quot;/app/artifacts/models/training_data.csv&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;training_data.csv&quot;,
    ]
    for csv_path in candidate_csvs:
        if csv_path.exists():
            try:
                import pandas as pd

                df = pd.read_csv(csv_path)
                if all(f in df.columns for f in FEATURE_NAMES):
                    return df[FEATURE_NAMES].values
            except Exception as e:
                logger.warning(&quot;Could not read reference csv&quot;, path=str(csv_path), error=str(e))

    from anomaly_detection.data import load_training_data

    X_base, _ = load_training_data(None)
    return X_base


# Create FastAPI app
app = FastAPI(
    title=&quot;Anomaly Detection API&quot;,
    description=&quot;PCA-based anomaly detection using dimensionality reduction&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

# Add observability middleware
add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    &quot;&quot;&quot;Service information.&quot;&quot;&quot;
    return {
        &quot;service&quot;: &quot;anomaly-detection-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;features&quot;: FEATURE_NAMES,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;predict&quot;: &quot;POST /predict&quot;,
            &quot;predict_bulk&quot;: &quot;POST /predict/bulk&quot;,
            &quot;stats&quot;: &quot;GET /stats&quot;,
            &quot;model_info&quot;: &quot;GET /model/info&quot;,
            &quot;drift&quot;: &quot;GET /drift&quot;,
            &quot;metrics&quot;: &quot;/metrics&quot;,
        },
    }


@app.get(&quot;/health&quot;)
def health_check():
    &quot;&quot;&quot;Kubernetes liveness/readiness probe.&quot;&quot;&quot;
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    return {
        &quot;status&quot;: &quot;healthy&quot;,
        &quot;model_loaded&quot;: True,
        &quot;model_version&quot;: _model_version,
    }


@app.get(&quot;/metrics&quot;)
def metrics():
    &quot;&quot;&quot;Prometheus metrics endpoint.&quot;&quot;&quot;
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(&quot;/reload&quot;)
def reload_model():
    &quot;&quot;&quot;Dynamically reload the model from disk/registry.&quot;&quot;&quot;
    global _model, _model_version, _reference_data
    try:
        _model, _model_version = _load_model()
        if _metrics:
            _metrics.set_model_version(_model_version)
            _metrics.set_model_info(
                model_name=&quot;anomaly-detection&quot;,
                model_version=_model_version,
                model_type=&quot;anomaly_detection&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(&quot;Model reloaded dynamically&quot;, model=&quot;anomaly-detection&quot;, version=_model_version)
        return {&quot;status&quot;: &quot;reloaded&quot;, &quot;model_version&quot;: _model_version}
    except Exception as e:
        logger.exception(&quot;Model reload failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=f&quot;Reload failed: {e}&quot;) from e


@app.get(&quot;/drift&quot;, response_model=DriftResponse)
def drift_check():
    &quot;&quot;&quot;Check for data drift between reference and recent predictions.&quot;&quot;&quot;
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail=&quot;Drift detection not available&quot;)

    if len(_recent_predictions) &lt; 10:
        return DriftResponse(
            total_features=len(FEATURE_NAMES),
            drifted_features=0,
            drift_ratio=0.0,
            drifted=[],
            all_results=[],
        )

    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)

    if _metrics:
        _metrics.set_drift_ratio(summary[&quot;drift_ratio&quot;])

    return DriftResponse(**summary)


@app.get(&quot;/stats&quot;, response_model=StatsResponse)
def get_stats():
    &quot;&quot;&quot;Return model statistics.&quot;&quot;&quot;
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    evr = _model.explained_variance_ratio

    return StatsResponse(
        n_features=_model.n_features,
        n_components=_model.n_components_selected,
        explained_variance_ratio=float(
            np.sum(evr[: _model.n_components_selected]) if evr is not None else 0.0
        ),
        reconstruction_threshold=round(_model.threshold, 4),
        threshold_method=_model.threshold_method,
        mean_reconstruction_error=float(
            np.mean(_model.reconstruction_error(_reference_data))
            if _reference_data is not None
            else 0.0
        ),
        max_reconstruction_error=float(
            np.max(_model.reconstruction_error(_reference_data))
            if _reference_data is not None
            else 0.0
        ),
        model_version=_model_version,
    )


@app.get(&quot;/model/info&quot;, response_model=ModelInfoResponse)
def get_model_info():
    &quot;&quot;&quot;Return detailed model information.&quot;&quot;&quot;
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    return ModelInfoResponse(
        n_components=_model.n_components_selected,
        n_features=_model.n_features,
        feature_names=FEATURE_NAMES,
        cumulative_variance_ratio=_model.cumulative_variance_ratio,
        reconstruction_threshold=round(_model.threshold, 4),
        model_version=_model_version,
    )


def _compute_anomaly(observation: MetricsRequest) -&gt; AnomalyResponse:
    &quot;&quot;&quot;Core anomaly detection logic shared by all detection endpoints.&quot;&quot;&quot;
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    # Validate input
    X = np.array(
        [
            [
                observation.request_count,
                observation.bytes_per_request,
                observation.cpu_usage,
                observation.memory_usage,
                observation.disk_io,
                observation.network_in,
                observation.network_out,
                observation.error_rate,
                observation.connection_count,
                observation.response_time,
            ]
        ]
    )
    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        recon_error = float(_model.reconstruction_error(X)[0])
        is_anom = bool(_model.is_anomaly(X)[0])
        proba = float(_model.predict_proba(X)[0])
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        # Track for drift detection
        _recent_predictions.append(
            [
                observation.request_count,
                observation.bytes_per_request,
                observation.cpu_usage,
                observation.memory_usage,
                observation.disk_io,
                observation.network_in,
                observation.network_out,
                observation.error_rate,
                observation.connection_count,
                observation.response_time,
            ]
        )
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return AnomalyResponse(
            is_anomaly=is_anom,
            anomaly_score=round(recon_error, 4),
            anomaly_probability=round(proba, 4),
            reconstruction_error=round(recon_error, 4),
            anomaly_threshold=round(_model.threshold, 4),
            model_version=_model_version,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Anomaly detection failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Anomaly detection failed&quot;) from e


@app.post(&quot;/predict&quot;, response_model=AnomalyResponse)
def predict_anomaly(body: MetricsRequest):
    &quot;&quot;&quot;Detect anomaly for a single metrics observation.&quot;&quot;&quot;
    return _compute_anomaly(body)


@app.post(&quot;/predict/bulk&quot;, response_model=BulkAnomalyResponse)
def predict_anomaly_bulk(body: MetricsBulkRequest):
    &quot;&quot;&quot;Detect anomalies for multiple metrics observations.&quot;&quot;&quot;
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    X = np.array(
        [
            [
                obs.request_count,
                obs.bytes_per_request,
                obs.cpu_usage,
                obs.memory_usage,
                obs.disk_io,
                obs.network_in,
                obs.network_out,
                obs.error_rate,
                obs.connection_count,
                obs.response_time,
            ]
            for obs in body.samples
        ]
    )

    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        recon_errors = _model.reconstruction_error(X)
        anomalies = _model.is_anomaly(X)
        probas = _model.predict_proba(X)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        # Track for drift detection
        _recent_predictions.extend(X.tolist())
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions = _recent_predictions[-1000:]

        results = [
            AnomalyResponse(
                is_anomaly=bool(anom),
                anomaly_score=round(float(recon_error), 4),
                anomaly_probability=round(float(proba), 4),
                reconstruction_error=round(float(recon_error), 4),
                anomaly_threshold=round(_model.threshold, 4),
                model_version=_model_version,
            )
            for anom, proba, recon_error in zip(anomalies, probas, recon_errors, strict=False)
        ]

        n_anomalies = int(np.sum(anomalies))
        return BulkAnomalyResponse(
            samples=results,
            n_anomalies=n_anomalies,
            n_samples=len(results),
            model_version=_model_version,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Bulk anomaly detection failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Bulk anomaly detection failed&quot;) from e</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-3583439326')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3583439326"><code class="language-bash">uv run python -m anomaly_detection_pca.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>
<div class="related-links">
<h3>Related Apps</h3>
<ul><li><a href="../anomaly-detection-fraud/README.md">anomaly-detection-fraud</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>anomaly-detection-pca</strong></p>
</footer>
<script>
function copyCode(id) {
  const el = document.getElementById(id);
  navigator.clipboard.writeText(el.innerText);
}
function renderMath() {
  renderMathInElement(document.body, { delimiters: [{left: "$$", right: "$$", display: true}] });
}
</script>
</body>
</html>