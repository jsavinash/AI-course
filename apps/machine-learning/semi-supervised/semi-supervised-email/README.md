<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>semi-supervised-email - AI App Documentation</title>
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
<p class="section-subtitle">Semi-Supervised Learning — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$\mathcal{L} = \mathcal{L}_{sup} + \lambda_t \mathcal{L}_{unsup}$$</div>
<div class="math-block">$$\mathcal{L}_{unsup} = \text{MSE}(f_\theta(x'), f_\theta(x)) \quad \text{(Mean Teacher)}$$</div>
<div class="math-block">$$p_t = \min\left(1, \frac{T}{T_0}\right)$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>Semi-supervised learning leverages unlabeled data by enforcing consistency. Given an input $x$, augmented views $x'$ should produce similar predictions. The total loss combines supervised cross-entropy on labeled data and consistency regularization on all data. A time-dependent weight $\lambda_t$ ramps up the unsupervised loss.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive pseudo-label confidence distribution; labeled vs unlabeled loss curves; decision boundary animation.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  LogisticRegression
  SelfTrainingClassifier</pre>
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
<button class="copy-btn" onclick="copyCode('code-3846615534')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3846615534"><code class="language-python">&quot;&quot;&quot;Production training pipeline for semi-supervised email classification.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

import numpy as np
from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from semi_supervised_email.data import (
    load_training_data,
    save_training_data,
)
from semi_supervised_email.model import SelfTrainingClassifier

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path,
    labeled_ratio: float,
    confidence_threshold: float,
    max_iterations: int,
    model_version: str,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -&gt; dict:
    &quot;&quot;&quot;Train the semi-supervised email classification model and save artifacts.

    Returns:
        Dictionary with training metrics
    &quot;&quot;&quot;
    # Load semi-supervised training data
    X, y, is_labeled = load_training_data(
        data_path=data_path if data_path and data_path.exists() else None,
        labeled_ratio=labeled_ratio,
        random_seed=random_seed,
    )
    logger.info(
        &quot;Loaded semi-supervised training data&quot;,
        n_samples=len(X),
        n_features=X.shape[1],
        n_labeled=int(np.sum(is_labeled)),
        n_unlabeled=int(np.sum(~is_labeled)),
        labeled_ratio=labeled_ratio,
    )

    # Save training data for reproducibility
    save_training_data(X, y, is_labeled, model_dir / &quot;training_data.csv&quot;)

    # Train self-training model
    model = SelfTrainingClassifier(
        confidence_threshold=confidence_threshold,
        max_iterations=max_iterations,
        random_seed=random_seed,
    )
    model.fit(X, y)

    training_mode = model.training_mode
    n_iterations = model.n_iterations_used
    n_labeled_final = model.n_labeled_history[-1] if model.n_labeled_history else np.sum(is_labeled)

    logger.info(
        &quot;Self-training complete&quot;,
        training_mode=training_mode,
        n_iterations=n_iterations,
        n_labeled_initial=int(np.sum(is_labeled)),
        n_labeled_final=n_labeled_final,
        n_pseudo_labeled=n_labeled_final - int(np.sum(is_labeled)),
    )

    # Evaluate on all labeled data
    X_labeled, y_labeled = _get_labeled_data(X, y)
    metrics = model.evaluate(X_labeled, y_labeled)

    # Add semi-supervised specific metrics
    metrics.update(
        {
            &quot;training_mode&quot;: float(training_mode == &quot;semi-supervised&quot;),
            &quot;n_labeled_initial&quot;: float(np.sum(is_labeled)),
            &quot;n_labeled_final&quot;: float(n_labeled_final),
            &quot;n_pseudo_labeled&quot;: float(n_labeled_final - np.sum(is_labeled)),
            &quot;n_unlabeled_initial&quot;: float(np.sum(~is_labeled)),
            &quot;n_iterations&quot;: float(n_iterations),
            &quot;confidence_threshold&quot;: confidence_threshold,
            &quot;labeled_ratio&quot;: labeled_ratio,
        }
    )

    # Save model
    model_path = model_dir / f&quot;semi_supervised_email_model_v{model_version}.npz&quot;
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, model_dir, model_version)

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;semi-supervised-email&quot;,
        model_version=model_version,
        model_type=&quot;semi_supervised_classification&quot;,
        metrics=metrics,
        parameters={
            &quot;labeled_ratio&quot;: labeled_ratio,
            &quot;confidence_threshold&quot;: confidence_threshold,
            &quot;max_iterations&quot;: max_iterations,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;semi_supervised_email_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.csv&quot;: model_dir / &quot;training_data.csv&quot;,
        },
        tags={
            &quot;framework&quot;: &quot;numpy&quot;,
            &quot;task&quot;: &quot;semi_supervised_classification&quot;,
            &quot;base_model&quot;: &quot;logistic_regression&quot;,
        },
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;semi-supervised-email&quot;,
            model_version=model_version,
            metrics=metrics,
            params={
                &quot;labeled_ratio&quot;: labeled_ratio,
                &quot;confidence_threshold&quot;: confidence_threshold,
                &quot;max_iterations&quot;: max_iterations,
                &quot;random_seed&quot;: random_seed,
            },
            artifacts={
                &quot;model&quot;: str(model_path),
                &quot;chart&quot;: str(model_dir / f&quot;semi_supervised_email_v{model_version}.png&quot;),
                &quot;training_data&quot;: str(model_dir / &quot;training_data.csv&quot;),
            },
            tags={&quot;model_type&quot;: &quot;semi_supervised_classification&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )
        logger.info(
            &quot;Registered model to MLflow&quot;, model=&quot;semi-supervised-email&quot;, version=model_version
        )

    return metrics


def _get_labeled_data(X: np.ndarray, y: np.ndarray) -&gt; tuple[np.ndarray, np.ndarray]:
    &quot;&quot;&quot;Extract only the labeled subset of the data.&quot;&quot;&quot;
    mask = y != -1
    return X[mask], y[mask]


def _save_chart(model: SelfTrainingClassifier, output_dir: Path, version: str) -&gt; None:
    &quot;&quot;&quot;Save the semi-supervised training chart.&quot;&quot;&quot;
    import matplotlib

    matplotlib.use(&quot;Agg&quot;)
    import matplotlib.pyplot as plt

    if not model.n_labeled_history:
        return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Labeled samples over iterations
    iterations = list(range(len(model.n_labeled_history)))
    ax1.plot(iterations, model.n_labeled_history, marker=&quot;o&quot;, color=&quot;steelblue&quot;, linewidth=2)
    ax1.set_xlabel(&quot;Self-Training Iteration&quot;)
    ax1.set_ylabel(&quot;Number of Labeled Samples&quot;)
    ax1.set_title(&quot;Labeled Samples Growth&quot;)
    ax1.grid(True, alpha=0.3)

    # Plot 2: Accuracy over iterations (if available)
    if model.accuracy_history:
        ax2.plot(
            iterations[: len(model.accuracy_history)],
            model.accuracy_history,
            marker=&quot;s&quot;,
            color=&quot;green&quot;,
            linewidth=2,
        )
        ax2.set_xlabel(&quot;Self-Training Iteration&quot;)
        ax2.set_ylabel(&quot;Accuracy&quot;)
        ax2.set_title(&quot;Model Accuracy During Self-Training&quot;)
        ax2.set_ylim([0, 1.05])
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(
            0.5,
            0.5,
            &quot;No accuracy data available&quot;,
            ha=&quot;center&quot;,
            va=&quot;center&quot;,
            transform=ax2.transAxes,
        )
        ax2.set_title(&quot;Model Accuracy During Self-Training&quot;)

    plt.tight_layout()

    chart_path = output_dir / f&quot;semi_supervised_email_v{version}.png&quot;
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info(&quot;Chart saved&quot;, path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description=&quot;Train semi-supervised email classification model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(
        &quot;--labeled-ratio&quot;, type=float, default=float(os.getenv(&quot;LABELED_RATIO&quot;, &quot;0.1&quot;))
    )
    parser.add_argument(
        &quot;--confidence-threshold&quot;,
        type=float,
        default=float(os.getenv(&quot;CONFIDENCE_THRESHOLD&quot;, &quot;0.95&quot;)),
    )
    parser.add_argument(
        &quot;--max-iterations&quot;, type=int, default=int(os.getenv(&quot;MAX_ITERATIONS&quot;, &quot;10&quot;))
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
        labeled_ratio=args.labeled_ratio,
        confidence_threshold=args.confidence_threshold,
        max_iterations=args.max_iterations,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )

    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-3409943923')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3409943923"><code class="language-python">&quot;&quot;&quot;Production serving API for semi-supervised email classification.&quot;&quot;&quot;

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
from ai_core.validation import DataValidator, create_semi_supervised_email_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from semi_supervised_email.data import FEATURE_NAMES
from semi_supervised_email.model import SelfTrainingClassifier

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;METRICS_PORT&quot;, os.getenv(&quot;SEMI_SUPERVISED_METRICS_PORT&quot;, &quot;8006&quot;)))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class PredictRequest(BaseModel):
    &quot;&quot;&quot;Single email classification request.&quot;&quot;&quot;

    has_free: int = Field(..., ge=0, le=1, description=&quot;Contains 'free' keyword&quot;)
    has_win: int = Field(..., ge=0, le=1, description=&quot;Contains 'win' keyword&quot;)
    has_link: int = Field(..., ge=0, le=1, description=&quot;Contains a link&quot;)
    has_exclamation: int = Field(..., ge=0, le=1, description=&quot;Contains 3+ exclamation marks&quot;)
    has_meeting: int = Field(..., ge=0, le=1, description=&quot;Contains 'meeting' keyword&quot;)
    length_score: int = Field(..., ge=1, le=10, description=&quot;Email length score (1-10)&quot;)
    has_caps: int = Field(..., ge=0, le=1, description=&quot;Contains excessive caps&quot;)


class PredictResponse(BaseModel):
    &quot;&quot;&quot;Email classification response.&quot;&quot;&quot;

    is_spam: bool
    spam_probability: float
    label: str
    model_version: str
    training_mode: str


class BulkPredictResponse(BaseModel):
    &quot;&quot;&quot;Bulk email classification response.&quot;&quot;&quot;

    predictions: list[PredictResponse]
    model_version: str


class StatsResponse(BaseModel):
    &quot;&quot;&quot;Model statistics response.&quot;&quot;&quot;

    n_features: int
    confidence_threshold: float
    max_iterations: int
    n_iterations_used: int
    training_mode: str
    n_labeled_initial: int
    n_labeled_final: int
    n_pseudo_labeled: int
    accuracy_history: list[float]
    model_version: str


class DriftResponse(BaseModel):
    &quot;&quot;&quot;Drift detection response.&quot;&quot;&quot;

    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


# Global model state
_model: SelfTrainingClassifier | None = None
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
    _metrics = MetricsCollector(&quot;semi_supervised_email&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_semi_supervised_email_schema())
    _drift_detector = DriftDetector(
        feature_names=FEATURE_NAMES,
        feature_types={f: &quot;float&quot; for f in FEATURE_NAMES},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;semi-supervised-email&quot;,
        model_version=_model_version,
        model_type=&quot;semi_supervised_classification&quot;,
    )

    # Load reference data for drift detection
    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;semi-supervised-email&quot;, version=_model_version)

    yield

    logger.info(&quot;Shutting down semi-supervised-email API&quot;)


def _load_model() -&gt; tuple[SelfTrainingClassifier, str]:
    &quot;&quot;&quot;Load the latest model from the registry or model directory with resilient fallback.&quot;&quot;&quot;
    # 1. Try model registry
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            ss_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;semi-supervised-email&quot;]
            if ss_models:
                ss_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = ss_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;semi_supervised_email_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return SelfTrainingClassifier.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;semi-supervised-email&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;semi_supervised_email_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return SelfTrainingClassifier.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    # 2. Try direct model in MODEL_DIR
    npz_path = MODEL_DIR / &quot;semi_supervised_email_model.npz&quot;
    if npz_path.exists():
        return SelfTrainingClassifier.load(str(npz_path)), &quot;legacy&quot;

    # 3. Try bundled artifacts directory
    candidate_paths = [
        Path(&quot;/app/artifacts/models/semi_supervised_email_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3]
        / &quot;artifacts&quot;
        / &quot;models&quot;
        / &quot;semi_supervised_email_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return SelfTrainingClassifier.load(str(p)), &quot;1.0.0-bundled&quot;

    # 4. In-memory baseline fallback (never crash cold start)
    logger.warning(
        &quot;No pre-existing model found on disk. Initializing baseline self-training model.&quot;
    )
    from semi_supervised_email.data import load_training_data

    X_base, y_base, _ = load_training_data(None, labeled_ratio=0.1, random_seed=42)
    model = SelfTrainingClassifier(confidence_threshold=0.95, max_iterations=10, random_seed=42)
    model.fit(X_base, y_base)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    &quot;&quot;&quot;Load reference training data for drift detection.&quot;&quot;&quot;
    candidate_csvs = [
        MODEL_DIR / &quot;semi-supervised-email&quot; / _model_version / &quot;training_data.csv&quot;,
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

    from semi_supervised_email.data import load_training_data

    X_base, _, _ = load_training_data(None, labeled_ratio=0.1, random_seed=42)
    return X_base


# Create FastAPI app
app = FastAPI(
    title=&quot;Semi-Supervised Email Classification API&quot;,
    description=&quot;Self-training semi-supervised learning for email spam classification&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

# Add observability middleware
add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    &quot;&quot;&quot;Service information.&quot;&quot;&quot;
    return {
        &quot;service&quot;: &quot;semi-supervised-email-api&quot;,
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
                model_name=&quot;semi-supervised-email&quot;,
                model_version=_model_version,
                model_type=&quot;semi_supervised_classification&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(
            &quot;Model reloaded dynamically&quot;, model=&quot;semi-supervised-email&quot;, version=_model_version
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
    if _model is None or _model.model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    return StatsResponse(
        n_features=_model.n_features,
        confidence_threshold=_model.confidence_threshold,
        max_iterations=_model.max_iterations,
        n_iterations_used=_model.n_iterations_used,
        training_mode=_model.training_mode,
        n_labeled_initial=_model.n_labeled_history[0] if _model.n_labeled_history else 0,
        n_labeled_final=_model.n_labeled_history[-1] if _model.n_labeled_history else 0,
        n_pseudo_labeled=(_model.n_labeled_history[-1] - _model.n_labeled_history[0])
        if _model.n_labeled_history
        else 0,
        accuracy_history=_model.accuracy_history,
        model_version=_model_version,
    )


def _compute_prediction(features: PredictRequest) -&gt; PredictResponse:
    &quot;&quot;&quot;Core classification logic shared by all prediction endpoints.&quot;&quot;&quot;
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    # Validate input
    X = np.array(
        [
            [
                features.has_free,
                features.has_win,
                features.has_link,
                features.has_exclamation,
                features.has_meeting,
                features.length_score,
                features.has_caps,
            ]
        ]
    )
    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        proba = float(_model.predict_proba(X)[0])
        is_spam = bool(_model.predict(X)[0])
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        # Track for drift detection
        _recent_predictions.append(
            [
                features.has_free,
                features.has_win,
                features.has_link,
                features.has_exclamation,
                features.has_meeting,
                features.length_score,
                features.has_caps,
            ]
        )
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return PredictResponse(
            is_spam=is_spam,
            spam_probability=round(proba, 4),
            label=&quot;SPAM&quot; if is_spam else &quot;NOT spam&quot;,
            model_version=_model_version,
            training_mode=_model.training_mode if _model else &quot;unknown&quot;,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Classification failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Classification failed&quot;) from e


@app.post(&quot;/predict&quot;, response_model=PredictResponse)
def predict_single(body: PredictRequest):
    &quot;&quot;&quot;Classify a single email.&quot;&quot;&quot;
    return _compute_prediction(body)


@app.post(&quot;/predict/bulk&quot;, response_model=BulkPredictResponse)
def predict_bulk(body: list[PredictRequest]):
    &quot;&quot;&quot;Classify multiple emails (1 to 100).&quot;&quot;&quot;
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    if len(body) &lt; 1 or len(body) &gt; 100:
        raise HTTPException(status_code=422, detail=&quot;Batch size must be between 1 and 100&quot;)

    X = np.array(
        [
            [
                r.has_free,
                r.has_win,
                r.has_link,
                r.has_exclamation,
                r.has_meeting,
                r.length_score,
                r.has_caps,
            ]
            for r in body
        ]
    )

    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        probas = _model.predict_proba(X)
        predictions = _model.predict(X)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        # Track for drift detection
        _recent_predictions.extend(X.tolist())
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions = _recent_predictions[-1000:]

        responses = [
            PredictResponse(
                is_spam=bool(pred),
                spam_probability=round(float(prob), 4),
                label=&quot;SPAM&quot; if pred else &quot;NOT spam&quot;,
                model_version=_model_version,
                training_mode=_model.training_mode if _model else &quot;unknown&quot;,
            )
            for pred, prob in zip(predictions, probas, strict=False)
        ]
        return BulkPredictResponse(predictions=responses, model_version=_model_version)
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Bulk classification failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Bulk classification failed&quot;) from e</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-1337169880')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-1337169880"><code class="language-bash">uv run python -m semi_supervised_email.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>
<div class="related-links">
<h3>Related Apps</h3>
<ul><li><a href="../self-supervised-monitoring/README.md">self-supervised-monitoring</a></li>
<li><a href="../classification-email-spam/README.md">classification-email-spam</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>semi-supervised-email</strong></p>
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