<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>self-supervised-monitoring - AI App Documentation</title>
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
<p class="section-subtitle">Self-Supervised Learning — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$\mathcal{L}_{InfoNCE} = -\log \frac{\exp(\text{sim}(z_i, z_j) / \tau)}{\sum_{k=1}^{2N} \mathbb{1}_{[k \neq i]} \exp(\text{sim}(z_i, z_k) / \tau)}$$</div>
<div class="math-block">$$z_i = g_\theta(f_\theta(x_i))$$</div>
<div class="math-block">$$\text{sim}(u, v) = \frac{u^T v}{\|u\| \|v\|}$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>Self-supervised learning creates labels from the data itself via pretext tasks. Contrastive learning (e.g., SimCLR, MoCo) maximizes agreement between augmented views of the same sample. The InfoNCE loss pulls positive pairs together while pushing apart negatives. A temperature parameter $\tau$ controls the sharpness of the distribution.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive augmentation preview; contrastive embedding t-SNE; similarity matrix heatmap.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  DenoisingAutoencoder</pre>
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
<button class="copy-btn" onclick="copyCode('code-3503285913')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3503285913"><code class="language-python">&quot;&quot;&quot;Production training pipeline for self-supervised server monitoring.

Trains a denoising autoencoder to reconstruct normal server metrics from
corrupted inputs. The self-supervised signal comes from the data itself -
no human labels are required for training.
&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

import numpy as np
from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from self_supervised_monitoring.data import (
    generate_synthetic_data,
    save_training_data,
)
from self_supervised_monitoring.model import DenoisingAutoencoder

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 2000,
    hidden_dim: int = 16,
    learning_rate: float = 0.01,
    n_iterations: int = 5000,
    noise_rate: float = 0.25,
    threshold_percentile: float = 95.0,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -&gt; dict:
    &quot;&quot;&quot;Train the self-supervised denoising autoencoder and save artifacts.

    The model is trained on normal server metrics only. Anomalies are
    detected at inference time via high reconstruction error.

    Returns:
        Dictionary with training metrics
    &quot;&quot;&quot;
    # Generate or load data
    # For self-supervised training, we use only the anomaly-free portion
    X_full, y_full = generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)

    # Separate normal and anomalous data
    X_normal = X_full[y_full == 0]
    X_anomaly = X_full[y_full == 1]

    # Split normal data for train/validation
    rng = np.random.default_rng(random_seed)
    n_val = max(1, int(len(X_normal) * 0.2))
    val_idx = rng.choice(len(X_normal), size=n_val, replace=False)
    val_mask = np.zeros(len(X_normal), dtype=bool)
    val_mask[val_idx] = True

    X_train = X_normal[~val_mask]
    X_val = X_normal[val_mask]

    # Split anomaly data for test evaluation
    n_test_anomaly = max(1, int(len(X_anomaly) * 0.5))
    test_anom_idx = rng.choice(len(X_anomaly), size=n_test_anomaly, replace=False)
    X_test_anomaly = X_anomaly[test_anom_idx]
    y_test_anomaly = np.ones(n_test_anomaly, dtype=int)

    # Use some normal data for test too
    test_norm_idx = rng.choice(len(X_normal), size=n_test_anomaly, replace=False)
    X_test_normal = X_normal[test_norm_idx]
    y_test_normal = np.zeros(n_test_anomaly, dtype=int)

    # Combine test set
    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.concatenate([y_test_normal, y_test_anomaly])

    logger.info(
        &quot;Loaded self-supervised training data&quot;,
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        n_features=X_train.shape[1],
        training_mode=&quot;self-supervised (denoising autoencoder)&quot;,
    )

    # Save full dataset for reproducibility
    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X_full, y_full, model_dir / &quot;training_data.csv&quot;)

    # Train self-supervised model
    model = DenoisingAutoencoder(
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        noise_rate=noise_rate,
        random_seed=random_seed,
    )
    model.threshold_percentile = threshold_percentile
    model.fit(X_train, X_val=X_val, X_test=X_test, y_test=y_test)

    # Compute metrics
    test_metrics = model.evaluate(X_test, y_test)
    train_errors = model.reconstruction_error(X_train)
    val_errors = model.reconstruction_error(X_val)

    metrics = {
        **test_metrics,
        &quot;training_mode&quot;: &quot;self-supervised&quot;,
        &quot;n_train_samples&quot;: float(len(X_train)),
        &quot;n_val_samples&quot;: float(len(X_val)),
        &quot;n_test_samples&quot;: float(len(X_test)),
        &quot;n_anomaly_test&quot;: float(np.sum(y_test == 1)),
        &quot;n_normal_test&quot;: float(np.sum(y_test == 0)),
        &quot;train_mean_recon_error&quot;: float(np.mean(train_errors)),
        &quot;train_max_recon_error&quot;: float(np.max(train_errors)),
        &quot;val_mean_recon_error&quot;: float(np.mean(val_errors)),
        &quot;final_loss&quot;: model.loss_history[-1] if model.loss_history else 0.0,
        &quot;n_epochs_run&quot;: float(len(model.loss_history)),
        &quot;reconstruction_threshold&quot;: float(model.threshold),
        &quot;threshold_percentile&quot;: float(model.threshold_percentile),
        &quot;noise_rate&quot;: float(noise_rate),
        &quot;hidden_dim&quot;: float(hidden_dim),
        &quot;learning_rate&quot;: float(learning_rate),
    }

    logger.info(
        &quot;Self-supervised training complete&quot;,
        training_mode=&quot;self-supervised&quot;,
        n_epochs=len(model.loss_history),
        final_loss=model.loss_history[-1] if model.loss_history else 0.0,
        threshold=model.threshold,
        test_accuracy=test_metrics[&quot;accuracy&quot;],
    )

    # Save model
    model_path = model_dir / f&quot;self_supervised_monitoring_model_v{model_version}.npz&quot;
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, model_dir, model_version)

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;self-supervised-monitoring&quot;,
        model_version=model_version,
        model_type=&quot;self_supervised_anomaly_detection&quot;,
        metrics=metrics,
        parameters={
            &quot;hidden_dim&quot;: hidden_dim,
            &quot;learning_rate&quot;: learning_rate,
            &quot;n_iterations&quot;: n_iterations,
            &quot;noise_rate&quot;: noise_rate,
            &quot;threshold_percentile&quot;: threshold_percentile,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;self_supervised_monitoring_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.csv&quot;: model_dir / &quot;training_data.csv&quot;,
        },
        tags={
            &quot;framework&quot;: &quot;numpy&quot;,
            &quot;task&quot;: &quot;self_supervised_anomaly_detection&quot;,
            &quot;base_model&quot;: &quot;denoising_autoencoder&quot;,
        },
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;self-supervised-monitoring&quot;,
            model_version=model_version,
            metrics=metrics,
            params={
                &quot;hidden_dim&quot;: hidden_dim,
                &quot;learning_rate&quot;: learning_rate,
                &quot;n_iterations&quot;: n_iterations,
                &quot;noise_rate&quot;: noise_rate,
                &quot;threshold_percentile&quot;: threshold_percentile,
                &quot;random_seed&quot;: random_seed,
            },
            artifacts={
                &quot;model&quot;: str(model_path),
                &quot;chart&quot;: str(model_dir / f&quot;self_supervised_monitoring_v{model_version}.png&quot;),
                &quot;training_data&quot;: str(model_dir / &quot;training_data.csv&quot;),
            },
            tags={&quot;model_type&quot;: &quot;self_supervised_anomaly_detection&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )
        logger.info(
            &quot;Registered model to MLflow&quot;, model=&quot;self-supervised-monitoring&quot;, version=model_version
        )

    return metrics


def _save_chart(model: DenoisingAutoencoder, output_dir: Path, version: str) -&gt; None:
    &quot;&quot;&quot;Save the training loss chart.&quot;&quot;&quot;
    import matplotlib

    matplotlib.use(&quot;Agg&quot;)
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color=&quot;steelblue&quot;, linewidth=1.5)
    ax.set_xlabel(&quot;Training Iteration&quot;)
    ax.set_ylabel(&quot;Reconstruction Loss (MSE)&quot;)
    ax.set_title(&quot;Self-Supervised Denoising Autoencoder Training Loss&quot;)
    ax.grid(True, alpha=0.3)
    ax.set_yscale(&quot;log&quot;)

    plt.tight_layout()
    chart_path = output_dir / f&quot;self_supervised_monitoring_v{version}.png&quot;
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info(&quot;Chart saved&quot;, path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(
        description=&quot;Train self-supervised monitoring model (denoising autoencoder)&quot;
    )
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;2000&quot;)))
    parser.add_argument(&quot;--hidden-dim&quot;, type=int, default=int(os.getenv(&quot;HIDDEN_DIM&quot;, &quot;16&quot;)))
    parser.add_argument(
        &quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.01&quot;))
    )
    parser.add_argument(&quot;--n-iterations&quot;, type=int, default=int(os.getenv(&quot;N_ITERATIONS&quot;, &quot;5000&quot;)))
    parser.add_argument(&quot;--noise-rate&quot;, type=float, default=float(os.getenv(&quot;NOISE_RATE&quot;, &quot;0.25&quot;)))
    parser.add_argument(
        &quot;--threshold-percentile&quot;,
        type=float,
        default=float(os.getenv(&quot;THRESHOLD_PERCENTILE&quot;, &quot;95.0&quot;)),
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

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_samples=args.n_samples,
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        noise_rate=args.noise_rate,
        threshold_percentile=args.threshold_percentile,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )

    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-4073351800')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-4073351800"><code class="language-python">&quot;&quot;&quot;Production serving API for self-supervised server monitoring anomaly detection.

Uses a denoising autoencoder trained on normal server metrics to detect
anomalies via reconstruction error. The model is trained in a self-supervised
manner - no human labels are required.
&quot;&quot;&quot;

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
from ai_core.validation import DataValidator, create_self_supervised_monitoring_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from self_supervised_monitoring.data import FEATURE_NAMES
from self_supervised_monitoring.model import DenoisingAutoencoder

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(
    os.getenv(&quot;METRICS_PORT&quot;, os.getenv(&quot;SELF_SUPERVISED_MONITORING_METRICS_PORT&quot;, &quot;8007&quot;))
)
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
    training_mode: str


class BulkAnomalyResponse(BaseModel):
    &quot;&quot;&quot;Bulk anomaly detection response.&quot;&quot;&quot;

    samples: list[AnomalyResponse]
    n_anomalies: int
    n_samples: int
    model_version: str


class StatsResponse(BaseModel):
    &quot;&quot;&quot;Model statistics response.&quot;&quot;&quot;

    n_features: int
    hidden_dim: int
    threshold: float
    threshold_percentile: float
    noise_rate: float
    training_mode: str
    n_train_samples: int
    final_loss: float
    n_epochs_run: int
    model_version: str


class ModelInfoResponse(BaseModel):
    &quot;&quot;&quot;Model information response.&quot;&quot;&quot;

    n_features: int
    hidden_dim: int
    threshold: float
    feature_names: list[str]
    training_mode: str
    model_version: str


class DriftResponse(BaseModel):
    &quot;&quot;&quot;Drift detection response.&quot;&quot;&quot;

    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


# Global model state
_model: DenoisingAutoencoder | None = None
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
    _metrics = MetricsCollector(&quot;self_supervised_monitoring&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_self_supervised_monitoring_schema())
    _drift_detector = DriftDetector(
        feature_names=FEATURE_NAMES,
        feature_types={f: &quot;float&quot; for f in FEATURE_NAMES},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;self-supervised-monitoring&quot;,
        model_version=_model_version,
        model_type=&quot;self_supervised_anomaly_detection&quot;,
    )

    # Load reference data for drift detection
    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;self-supervised-monitoring&quot;, version=_model_version)

    yield

    logger.info(&quot;Shutting down self-supervised-monitoring API&quot;)


def _load_model() -&gt; tuple[DenoisingAutoencoder, str]:
    &quot;&quot;&quot;Load the latest model from the registry or model directory with resilient fallback.&quot;&quot;&quot;
    # 1. Try model registry
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            ss_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;self-supervised-monitoring&quot;]
            if ss_models:
                ss_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = ss_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;self_supervised_monitoring_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return DenoisingAutoencoder.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;self-supervised-monitoring&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;self_supervised_monitoring_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return DenoisingAutoencoder.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    # 2. Try direct model in MODEL_DIR
    npz_path = MODEL_DIR / &quot;self_supervised_monitoring_model.npz&quot;
    if npz_path.exists():
        return DenoisingAutoencoder.load(str(npz_path)), &quot;legacy&quot;

    # 3. Try bundled artifacts directory
    candidate_paths = [
        Path(&quot;/app/artifacts/models/self_supervised_monitoring_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3]
        / &quot;artifacts&quot;
        / &quot;models&quot;
        / &quot;self_supervised_monitoring_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return DenoisingAutoencoder.load(str(p)), &quot;1.0.0-bundled&quot;

    # 4. In-memory baseline fallback (never crash cold start)
    logger.warning(
        &quot;No pre-existing model found on disk. Initializing baseline self-supervised model.&quot;
    )
    from self_supervised_monitoring.data import generate_normal_data

    X_base = generate_normal_data(n_samples=2000, random_seed=42)
    model = DenoisingAutoencoder(
        hidden_dim=16,
        learning_rate=0.01,
        n_iterations=1000,
        noise_rate=0.25,
        random_seed=42,
    )
    model.fit(X_base)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    &quot;&quot;&quot;Load reference training data for drift detection.&quot;&quot;&quot;
    candidate_csvs = [
        MODEL_DIR / &quot;self-supervised-monitoring&quot; / _model_version / &quot;training_data.csv&quot;,
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

    from self_supervised_monitoring.data import generate_normal_data

    X_base = generate_normal_data(n_samples=500, random_seed=42)
    return X_base


# Create FastAPI app
app = FastAPI(
    title=&quot;Self-Supervised Monitoring API&quot;,
    description=&quot;Self-supervised anomaly detection using a denoising autoencoder trained on normal server metrics&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

# Add observability middleware
add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    &quot;&quot;&quot;Service information.&quot;&quot;&quot;
    return {
        &quot;service&quot;: &quot;self-supervised-monitoring-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;training_mode&quot;: _model.training_mode if _model else &quot;unknown&quot;,
        &quot;features&quot;: FEATURE_NAMES,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;predict&quot;: &quot;POST /predict&quot;,
            &quot;predict/bulk&quot;: &quot;POST /predict/bulk&quot;,
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
        &quot;training_mode&quot;: _model.training_mode,
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
                model_name=&quot;self-supervised-monitoring&quot;,
                model_version=_model_version,
                model_type=&quot;self_supervised_anomaly_detection&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(
            &quot;Model reloaded dynamically&quot;, model=&quot;self-supervised-monitoring&quot;, version=_model_version
        )
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
    if _model is None or _model.W1 is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    return StatsResponse(
        n_features=_model.input_dim,
        hidden_dim=_model.hidden_dim,
        threshold=round(_model.threshold, 4),
        threshold_percentile=_model.threshold_percentile,
        noise_rate=_model.noise_rate,
        training_mode=_model.training_mode,
        n_train_samples=len(_reference_data) if _reference_data is not None else 0,
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        n_epochs_run=len(_model.loss_history),
        model_version=_model_version,
    )


@app.get(&quot;/model/info&quot;, response_model=ModelInfoResponse)
def get_model_info():
    &quot;&quot;&quot;Return detailed model information.&quot;&quot;&quot;
    if _model is None or _model.W1 is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    return ModelInfoResponse(
        n_features=_model.input_dim,
        hidden_dim=_model.hidden_dim,
        threshold=round(_model.threshold, 4),
        feature_names=FEATURE_NAMES,
        training_mode=_model.training_mode,
        model_version=_model_version,
    )


def _extract_features(observation: MetricsRequest) -&gt; list[float]:
    &quot;&quot;&quot;Extract feature vector from request.&quot;&quot;&quot;
    return [
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


def _compute_anomaly(observation: MetricsRequest) -&gt; AnomalyResponse:
    &quot;&quot;&quot;Core anomaly detection logic shared by all detection endpoints.&quot;&quot;&quot;
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    features = _extract_features(observation)
    X = np.array([features])

    # Validate input
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
        _recent_predictions.append(features)
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return AnomalyResponse(
            is_anomaly=is_anom,
            anomaly_score=round(recon_error, 4),
            anomaly_probability=round(proba, 4),
            reconstruction_error=round(recon_error, 4),
            anomaly_threshold=round(_model.threshold, 4),
            model_version=_model_version,
            training_mode=_model.training_mode if _model else &quot;unknown&quot;,
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

    if len(body.samples) &lt; 1 or len(body.samples) &gt; 100:
        raise HTTPException(status_code=422, detail=&quot;Batch size must be between 1 and 100&quot;)

    X = np.array([_extract_features(s) for s in body.samples])

    # Validate input
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
                training_mode=_model.training_mode if _model else &quot;unknown&quot;,
            )
            for anom, recon_error, proba in zip(anomalies, recon_errors, probas, strict=False)
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
<button class="copy-btn" onclick="copyCode('code-445399025')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-445399025"><code class="language-bash">uv run python -m self_supervised_monitoring.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>
<div class="related-links">
<h3>Related Apps</h3>
<ul><li><a href="../semi-supervised-email/README.md">semi-supervised-email</a></li>
<li><a href="../self-organizing-maps/README.md">self-organizing-maps</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>self-supervised-monitoring</strong></p>
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