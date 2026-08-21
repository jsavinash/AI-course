<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>anomaly-detection-fraud - AI App Documentation</title>
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
<p class="section-subtitle">Anomaly Detection / Autoencoder — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$z = f(x) = \sigma(W_e x + b_e) \quad \text{(encoder)}$$</div>
<div class="math-block">$$\hat{x} = g(z) = \sigma(W_d z + b_d) \quad \text{(decoder)}$$</div>
<div class="math-block">$$\mathcal{L} = \|x - \hat{x}\|^2 + \lambda (\|W_e\|^2 + \|W_d\|^2)$$</div>
<div class="math-block">$$\text{anomaly score} = \|x - \hat{x}\|^2$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>Autoencoders learn compressed representations by minimizing reconstruction error. The encoder maps input $x$ to a latent code $z$. The decoder reconstructs $\hat{x}$ from $z$. L2 regularization and bottleneck architecture prevent trivial identity solutions.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive latent space traversal; reconstruction error vs latent dimension; bottleneck visualization.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  FraudDetectionAutoencoder</pre>
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
<button class="copy-btn" onclick="copyCode('code-1368717151')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-1368717151"><code class="language-python">&quot;&quot;&quot;Training pipeline for credit card fraud detection using a feedforward autoencoder.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

import numpy as np
from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from anomaly_detection_fraud.data import generate_synthetic_data, save_training_data
from anomaly_detection_fraud.model import FraudDetectionAutoencoder

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 2000,
    anomaly_fraction: float = 0.05,
    hidden_dim: int = 8,
    learning_rate: float = 0.001,
    n_iterations: int = 2000,
    threshold_percentile: float = 95.0,
    weight_decay: float = 0.0001,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -&gt; dict:
    &quot;&quot;&quot;Train the fraud detection autoencoder and save artifacts.

    The model is trained on normal transactions only. Fraudulent transactions
    are detected at inference time via high reconstruction error.
    &quot;&quot;&quot;
    X_full, y_full = generate_synthetic_data(
        n_samples=n_samples, anomaly_fraction=anomaly_fraction, random_seed=random_seed
    )

    n_normal = int(np.sum(y_full == 0))
    n_fraud = int(np.sum(y_full == 1))
    logger.info(
        &quot;Loaded training data&quot;,
        n_total=len(X_full),
        n_normal=n_normal,
        n_fraud=n_fraud,
        data_path=str(data_path),
    )

    # Split: training on normal only, test with both
    X_normal = X_full[y_full == 0]
    X_anomaly = X_full[y_full == 1]

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

    test_norm_idx = rng.choice(len(X_normal), size=n_test_anomaly, replace=False)
    X_test_normal = X_normal[test_norm_idx]
    y_test_normal = np.zeros(n_test_anomaly, dtype=int)

    X_test = np.vstack([X_test_normal, X_test_anomaly])
    y_test = np.concatenate([y_test_normal, y_test_anomaly])

    logger.info(
        &quot;Data split for anomaly detection&quot;,
        n_train=len(X_train),
        n_val=len(X_val),
        n_test=len(X_test),
        n_features=X_train.shape[1],
        training_mode=&quot;autoencoder (normal data only)&quot;,
    )

    # Save training data for reproducibility
    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X_full, y_full, model_dir / &quot;training_data.csv&quot;)

    # Train model
    model = FraudDetectionAutoencoder(
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        threshold_percentile=threshold_percentile,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X_train, X_val=X_val, X_test=X_test, y_test=y_test)

    # Evaluate
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(
        &quot;Training complete&quot;,
        training_mode=model.training_mode,
        n_epochs=len(model.loss_history),
        final_loss=model.loss_history[-1] if model.loss_history else 0.0,
        threshold=model.threshold,
        test_metrics=test_metrics,
    )

    # Save model
    model_path = model_dir / f&quot;fraud_detection_model_v{model_version}.npz&quot;
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, model_dir, model_version)

    # Combined metrics for registry
    train_errors = model.reconstruction_error(X_train)
    metrics = {
        **test_metrics,
        &quot;training_mode&quot;: &quot;anomaly_detection&quot;,
        &quot;n_epochs_run&quot;: float(len(model.loss_history)),
        &quot;final_loss&quot;: model.loss_history[-1] if model.loss_history else 0.0,
        &quot;anomaly_threshold&quot;: float(model.threshold),
        &quot;threshold_percentile&quot;: float(threshold_percentile),
        &quot;n_train_samples&quot;: float(len(X_train)),
        &quot;n_val_samples&quot;: float(len(X_val)),
        &quot;n_test_samples&quot;: float(len(X_test)),
        &quot;n_normal_train&quot;: float(len(X_train)),
        &quot;n_fraud_detected&quot;: float(test_metrics[&quot;n_true_positives&quot;]),
        &quot;hidden_dim&quot;: float(hidden_dim),
        &quot;learning_rate&quot;: float(learning_rate),
        &quot;weight_decay&quot;: float(weight_decay),
        &quot;train_mean_recon_error&quot;: float(np.mean(train_errors)),
        &quot;n_features&quot;: float(X_train.shape[1]),
    }

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;credit-card-fraud-detection&quot;,
        model_version=model_version,
        model_type=&quot;anomaly_detection&quot;,
        metrics=metrics,
        parameters={
            &quot;hidden_dim&quot;: hidden_dim,
            &quot;learning_rate&quot;: learning_rate,
            &quot;n_iterations&quot;: n_iterations,
            &quot;threshold_percentile&quot;: threshold_percentile,
            &quot;weight_decay&quot;: weight_decay,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;fraud_detection_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.csv&quot;: model_dir / &quot;training_data.csv&quot;,
        },
        tags={
            &quot;framework&quot;: &quot;numpy&quot;,
            &quot;task&quot;: &quot;anomaly_detection&quot;,
            &quot;model_type&quot;: &quot;feedforward_neural_network&quot;,
        },
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;credit-card-fraud-detection&quot;,
            model_version=model_version,
            metrics=metrics,
            params={
                &quot;hidden_dim&quot;: hidden_dim,
                &quot;learning_rate&quot;: learning_rate,
                &quot;n_iterations&quot;: n_iterations,
                &quot;threshold_percentile&quot;: threshold_percentile,
                &quot;weight_decay&quot;: weight_decay,
                &quot;random_seed&quot;: random_seed,
            },
            artifacts={
                &quot;model&quot;: str(model_path),
                &quot;chart&quot;: str(model_dir / f&quot;fraud_detection_v{model_version}.png&quot;),
                &quot;training_data&quot;: str(model_dir / &quot;training_data.csv&quot;),
            },
            tags={&quot;model_type&quot;: &quot;anomaly_detection&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )
        logger.info(
            &quot;Registered model to MLflow&quot;, model=&quot;credit-card-fraud-detection&quot;, version=model_version
        )

    return metrics


def _save_chart(model: FraudDetectionAutoencoder, output_dir: Path, version: str) -&gt; None:
    &quot;&quot;&quot;Save the training loss chart.&quot;&quot;&quot;
    import matplotlib

    matplotlib.use(&quot;Agg&quot;)
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color=&quot;steelblue&quot;, linewidth=1.5)
    ax.set_xlabel(&quot;Training Iteration&quot;)
    ax.set_ylabel(&quot;Loss (MSE + L2)&quot;)
    ax.set_title(&quot;Credit Card Fraud Detection Autoencoder Training Loss&quot;)
    ax.grid(True, alpha=0.3)
    ax.set_yscale(&quot;log&quot;)

    plt.tight_layout()
    chart_path = output_dir / f&quot;fraud_detection_v{version}.png&quot;
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info(&quot;Chart saved&quot;, path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description=&quot;Train credit card fraud detection neural network&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;2000&quot;)))
    parser.add_argument(&quot;--hidden-dim&quot;, type=int, default=int(os.getenv(&quot;HIDDEN_DIM&quot;, &quot;8&quot;)))
    parser.add_argument(
        &quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.001&quot;))
    )
    parser.add_argument(&quot;--n-iterations&quot;, type=int, default=int(os.getenv(&quot;N_ITERATIONS&quot;, &quot;2000&quot;)))
    parser.add_argument(
        &quot;--threshold-percentile&quot;,
        type=float,
        default=float(os.getenv(&quot;THRESHOLD_PERCENTILE&quot;, &quot;95.0&quot;)),
    )
    parser.add_argument(
        &quot;--weight-decay&quot;, type=float, default=float(os.getenv(&quot;WEIGHT_DECAY&quot;, &quot;0.0001&quot;))
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
        threshold_percentile=args.threshold_percentile,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )

    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-27629925')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-27629925"><code class="language-python">&quot;&quot;&quot;Production serving API for credit card fraud detection via autoencoder reconstruction error.&quot;&quot;&quot;

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
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from anomaly_detection_fraud.data import FEATURE_NAMES
from anomaly_detection_fraud.model import FraudDetectionAutoencoder

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;METRICS_PORT&quot;, os.getenv(&quot;FRAUD_DETECTION_METRICS_PORT&quot;, &quot;8010&quot;)))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class FraudRequest(BaseModel):
    &quot;&quot;&quot;Single credit card transaction for fraud detection.&quot;&quot;&quot;

    time_since_last_transaction: float = Field(
        ..., ge=0, description=&quot;Minutes since last transaction&quot;
    )
    transaction_amount: float = Field(..., ge=0, description=&quot;Transaction amount in USD&quot;)
    merchant_category: float = Field(..., ge=0, le=11, description=&quot;Merchant category code (0-11)&quot;)
    merchant_risk_score: float = Field(..., ge=0, le=1, description=&quot;Merchant risk score (0-1)&quot;)
    cardholder_risk_score: float = Field(..., ge=0, le=1, description=&quot;Cardholder risk score (0-1)&quot;)
    distance_from_home: float = Field(..., ge=0, description=&quot;Distance from home in miles&quot;)
    is_online: float = Field(..., ge=0, le=1, description=&quot;Whether transaction is online (0/1)&quot;)
    is_foreign: float = Field(..., ge=0, le=1, description=&quot;Whether transaction is foreign (0/1)&quot;)
    hour_of_day: float = Field(..., ge=0, le=23, description=&quot;Hour of day (0-23)&quot;)
    day_of_week: float = Field(..., ge=0, le=6, description=&quot;Day of week (0-6)&quot;)
    account_age_days: float = Field(..., ge=0, description=&quot;Account age in days&quot;)
    recent_transaction_count: float = Field(..., ge=0, description=&quot;Recent transaction count&quot;)
    avg_transaction_amount_24h: float = Field(..., ge=0, description=&quot;Avg transaction amount (24h)&quot;)
    device_risk_score: float = Field(..., ge=0, le=1, description=&quot;Device risk score (0-1)&quot;)
    ip_risk_score: float = Field(..., ge=0, le=1, description=&quot;IP risk score (0-1)&quot;)


class FraudBulkRequest(BaseModel):
    &quot;&quot;&quot;Bulk fraud detection request.&quot;&quot;&quot;

    samples: list[FraudRequest] = Field(..., min_length=1, max_length=100)


class FraudResponse(BaseModel):
    &quot;&quot;&quot;Fraud detection response for a single transaction.&quot;&quot;&quot;

    is_fraud: bool
    fraud_probability: float
    reconstruction_error: float
    anomaly_threshold: float
    model_version: str
    training_mode: str


class BulkFraudResponse(BaseModel):
    &quot;&quot;&quot;Bulk fraud detection response.&quot;&quot;&quot;

    samples: list[FraudResponse]
    n_frauds: int
    n_samples: int
    model_version: str


class DriftResponse(BaseModel):
    &quot;&quot;&quot;Drift detection response.&quot;&quot;&quot;

    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


class StatsResponse(BaseModel):
    &quot;&quot;&quot;Model statistics response.&quot;&quot;&quot;

    n_features: int
    hidden_dim: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    threshold: float
    model_version: str


# Global model state
_model: FraudDetectionAutoencoder | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;credit_card_fraud_detection&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _drift_detector = DriftDetector(
        feature_names=FEATURE_NAMES,
        feature_types={f: &quot;float&quot; for f in FEATURE_NAMES},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;credit-card-fraud-detection&quot;,
        model_version=_model_version,
        model_type=&quot;anomaly_detection&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;credit-card-fraud-detection&quot;, version=_model_version)

    yield

    logger.info(&quot;Shutting down credit-card-fraud-detection API&quot;)


def _load_model() -&gt; tuple[FraudDetectionAutoencoder, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            fraud_models = [
                m for m in models if m.get(&quot;model_name&quot;) == &quot;credit-card-fraud-detection&quot;
            ]
            if fraud_models:
                fraud_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = fraud_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;fraud_detection_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return FraudDetectionAutoencoder.load(str(npz_files[0])), latest[
                        &quot;model_version&quot;
                    ]
        else:
            model_dir = MODEL_DIR / &quot;credit-card-fraud-detection&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;fraud_detection_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return FraudDetectionAutoencoder.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    npz_path = MODEL_DIR / &quot;fraud_detection_model.npz&quot;
    if npz_path.exists():
        return FraudDetectionAutoencoder.load(str(npz_path)), &quot;legacy&quot;

    candidate_paths = [
        Path(&quot;/app/artifacts/models/fraud_detection_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3]
        / &quot;artifacts&quot;
        / &quot;models&quot;
        / &quot;fraud_detection_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return FraudDetectionAutoencoder.load(str(p)), &quot;1.0.0-bundled&quot;

    logger.warning(&quot;No pre-existing model found on disk. Initializing baseline model.&quot;)
    from anomaly_detection_fraud.data import generate_normal_data

    X_base = generate_normal_data(n_samples=2000, random_seed=42)
    model = FraudDetectionAutoencoder(
        hidden_dim=8, learning_rate=0.001, n_iterations=500, random_seed=42
    )
    model.fit(X_base)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    candidate_csvs = [
        MODEL_DIR / &quot;credit-card-fraud-detection&quot; / _model_version / &quot;training_data.csv&quot;,
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

    from anomaly_detection_fraud.data import generate_normal_data

    X_base = generate_normal_data(n_samples=500, random_seed=42)
    return X_base


app = FastAPI(
    title=&quot;Credit Card Fraud Detection API&quot;,
    description=&quot;Feedforward autoencoder for detecting fraudulent credit card transactions&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;credit-card-fraud-detection-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;training_mode&quot;: _model.training_mode if _model else &quot;unknown&quot;,
        &quot;features&quot;: FEATURE_NAMES,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;predict&quot;: &quot;POST /predict&quot;,
            &quot;predict/bulk&quot;: &quot;POST /predict/bulk&quot;,
            &quot;stats&quot;: &quot;GET /stats&quot;,
            &quot;drift&quot;: &quot;GET /drift&quot;,
            &quot;metrics&quot;: &quot;/metrics&quot;,
        },
    }


@app.get(&quot;/health&quot;)
def health_check():
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    return {
        &quot;status&quot;: &quot;healthy&quot;,
        &quot;model_loaded&quot;: True,
        &quot;model_version&quot;: _model_version,
        &quot;training_mode&quot;: _model.training_mode if _model else &quot;unknown&quot;,
    }


@app.get(&quot;/metrics&quot;)
def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(&quot;/reload&quot;)
def reload_model():
    global _model, _model_version, _reference_data
    try:
        _model, _model_version = _load_model()
        if _metrics:
            _metrics.set_model_version(_model_version)
            _metrics.set_model_info(
                model_name=&quot;credit-card-fraud-detection&quot;,
                model_version=_model_version,
                model_type=&quot;anomaly_detection&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(
            &quot;Model reloaded dynamically&quot;,
            model=&quot;credit-card-fraud-detection&quot;,
            version=_model_version,
        )
        return {&quot;status&quot;: &quot;reloaded&quot;, &quot;model_version&quot;: _model_version}
    except Exception as e:
        logger.exception(&quot;Model reload failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=f&quot;Reload failed: {e}&quot;) from e


@app.get(&quot;/drift&quot;, response_model=DriftResponse)
def drift_check():
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
    if _model is None or _model.W1 is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    return StatsResponse(
        n_features=_model.input_dim,
        hidden_dim=_model.hidden_dim,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        threshold=_model.threshold,
        model_version=_model_version,
    )


def _extract_features(obs: FraudRequest) -&gt; list[float]:
    return [
        obs.time_since_last_transaction,
        obs.transaction_amount,
        obs.merchant_category,
        obs.merchant_risk_score,
        obs.cardholder_risk_score,
        obs.distance_from_home,
        obs.is_online,
        obs.is_foreign,
        obs.hour_of_day,
        obs.day_of_week,
        obs.account_age_days,
        obs.recent_transaction_count,
        obs.avg_transaction_amount_24h,
        obs.device_risk_score,
        obs.ip_risk_score,
    ]


def _compute_fraud(obs: FraudRequest) -&gt; FraudResponse:
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    features = _extract_features(obs)
    X = np.array([features])

    start = time.time()
    try:
        recon_error = float(_model.reconstruction_error(X)[0])
        is_fraud = bool(_model.is_fraud(X)[0])
        proba = float(_model.predict_proba(X)[0])
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append(features)
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return FraudResponse(
            is_fraud=is_fraud,
            fraud_probability=round(proba, 4),
            reconstruction_error=round(recon_error, 4),
            anomaly_threshold=round(_model.threshold, 4),
            model_version=_model_version,
            training_mode=_model.training_mode if _model else &quot;unknown&quot;,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Fraud detection failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Fraud detection failed&quot;) from e


@app.post(&quot;/predict&quot;, response_model=FraudResponse)
def predict_fraud(body: FraudRequest):
    &quot;&quot;&quot;Detect fraud for a single transaction.&quot;&quot;&quot;
    return _compute_fraud(body)


@app.post(&quot;/predict/bulk&quot;, response_model=BulkFraudResponse)
def predict_fraud_bulk(body: FraudBulkRequest):
    &quot;&quot;&quot;Detect fraud for multiple transactions.&quot;&quot;&quot;
    global _recent_predictions
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    if len(body.samples) &lt; 1 or len(body.samples) &gt; 100:
        raise HTTPException(status_code=422, detail=&quot;Batch size must be between 1 and 100&quot;)

    X = np.array([_extract_features(s) for s in body.samples])

    start = time.time()
    try:
        recon_errors = _model.reconstruction_error(X)
        anomalies = _model.is_fraud(X)
        probas = _model.predict_proba(X)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.extend(X.tolist())
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions = _recent_predictions[-1000:]

        results = [
            FraudResponse(
                is_fraud=bool(anom),
                fraud_probability=round(float(proba), 4),
                reconstruction_error=round(float(recon_error), 4),
                anomaly_threshold=round(_model.threshold, 4),
                model_version=_model_version,
                training_mode=_model.training_mode if _model else &quot;unknown&quot;,
            )
            for anom, recon_error, proba in zip(anomalies, recon_errors, probas, strict=False)
        ]

        return BulkFraudResponse(
            samples=results,
            n_frauds=int(np.sum(anomalies)),
            n_samples=len(results),
            model_version=_model_version,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Bulk fraud detection failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Bulk fraud detection failed&quot;) from e</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-2517555061')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2517555061"><code class="language-bash">uv run python -m anomaly_detection_fraud.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>
<div class="related-links">
<h3>Related Apps</h3>
<ul><li><a href="../anomaly-detection-pca/README.md">anomaly-detection-pca</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>anomaly-detection-fraud</strong></p>
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