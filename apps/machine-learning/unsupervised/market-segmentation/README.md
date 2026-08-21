<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>market-segmentation - AI App Documentation</title>
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
<p class="section-subtitle">Market Segmentation (K-Means) — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$\min_S \sum_{i=1}^{k} \sum_{x \in S_i} \|x - \mu_i\|^2$$</div>
<div class="math-block">$$\mu_i = \frac{1}{|S_i|} \sum_{x \in S_i} x$$</div>
<div class="math-block">$$J = \sum_{i=1}^{n} \|x^{(i)} - \mu_{c^{(i)}}\|^2$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>K-Means partitions data into $k$ clusters by minimizing within-cluster sum of squares. The Expectation-Maximization (EM) algorithm alternates between: (1) assigning each point to the nearest centroid, and (2) recomputing centroids as the mean of assigned points. Convergence is guaranteed but the solution depends on initialization.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive elbow method plot; cluster visualization with centroids; silhouette score explorer.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  KMeans</pre>
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
<button class="copy-btn" onclick="copyCode('code-3752656883')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3752656883"><code class="language-python">&quot;&quot;&quot;Production training pipeline for market segmentation (unsupervised K-Means).&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

import numpy as np
from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_market_segmentation_schema

from market_segmentation.data import load_training_data, save_training_data
from market_segmentation.model import KMeans

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path,
    n_clusters: int,
    max_iterations: int,
    n_init: int,
    model_version: str,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -&gt; dict:
    &quot;&quot;&quot;Train the market segmentation K-Means model and save artifacts.

    Returns:
        Dictionary with training metrics
    &quot;&quot;&quot;
    # Load training data
    X, y = load_training_data(data_path)
    logger.info(&quot;Loaded training data&quot;, n_samples=len(X), n_features=X.shape[1])

    # Validate training data
    validator = DataValidator(create_market_segmentation_schema())
    validation = validator.validate(X)
    if not validation.valid:
        logger.error(&quot;Training data validation failed&quot;, errors=validation.errors)
        raise ValueError(f&quot;Training data validation failed: {validation.errors}&quot;)
    logger.info(&quot;Training data validated&quot;, stats=validation.stats)

    # Save training data for reproducibility
    save_training_data(X, y, model_dir / &quot;training_data.csv&quot;)

    # Train model
    model = KMeans(
        n_clusters=n_clusters,
        max_iterations=max_iterations,
        n_init=n_init,
        random_seed=random_seed,
    )
    model.fit(X)

    # Evaluate clustering quality
    metrics = model.evaluate(X)
    logger.info(
        &quot;Training complete&quot;,
        n_clusters=model.n_clusters,
        inertia=model.inertia,
        silhouette=metrics[&quot;silhouette&quot;],
        n_iterations_used=model.n_iterations_used,
    )

    # Model validation - check silhouette score
    if metrics[&quot;silhouette&quot;] &lt; 0.1:
        logger.warning(
            &quot;Model silhouette score below threshold&quot;,
            silhouette=metrics[&quot;silhouette&quot;],
            threshold=0.1,
        )

    # Save model
    model_path = model_dir / f&quot;market_segmentation_model_v{model_version}.npz&quot;
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, X, model_dir, model_version)

    # Combined metrics for registry
    training_metrics = {
        &quot;inertia&quot;: metrics[&quot;inertia&quot;],
        &quot;silhouette&quot;: metrics[&quot;silhouette&quot;],
        &quot;n_clusters&quot;: float(n_clusters),
        &quot;n_samples&quot;: len(X),
        &quot;n_iterations_used&quot;: float(model.n_iterations_used),
    }

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;market-segmentation&quot;,
        model_version=model_version,
        model_type=&quot;clustering&quot;,
        metrics=training_metrics,
        parameters={
            &quot;n_clusters&quot;: n_clusters,
            &quot;max_iterations&quot;: max_iterations,
            &quot;n_init&quot;: n_init,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;market_segmentation_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.csv&quot;: model_dir / &quot;training_data.csv&quot;,
        },
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;clustering&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;market-segmentation&quot;,
            model_version=model_version,
            metrics=training_metrics,
            params={
                &quot;n_clusters&quot;: n_clusters,
                &quot;max_iterations&quot;: max_iterations,
                &quot;n_init&quot;: n_init,
                &quot;random_seed&quot;: random_seed,
            },
            artifacts={
                &quot;model&quot;: str(model_path),
                &quot;chart&quot;: str(model_dir / f&quot;market_segmentation_v{model_version}.png&quot;),
                &quot;training_data&quot;: str(model_dir / &quot;training_data.csv&quot;),
            },
            tags={&quot;model_type&quot;: &quot;clustering&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )
        logger.info(
            &quot;Registered model to MLflow&quot;, model=&quot;market-segmentation&quot;, version=model_version
        )

    return training_metrics


def _save_chart(model: KMeans, X: np.ndarray, output_dir: Path, version: str) -&gt; None:
    &quot;&quot;&quot;Save the clustering chart.&quot;&quot;&quot;
    import matplotlib

    matplotlib.use(&quot;Agg&quot;)
    import matplotlib.pyplot as plt

    if model.centroids is None:
        return

    plt.figure(figsize=(10, 6))

    # Plot data points colored by cluster
    labels = model.predict(X)
    scatter = plt.scatter(
        X[:, 0],
        X[:, 1],
        c=labels,
        cmap=&quot;viridis&quot;,
        s=50,
        alpha=0.6,
        label=&quot;Customers&quot;,
    )

    # Plot centroids
    # Need to unstandardize centroids for plotting
    if model.feature_mean is not None and model.feature_std is not None:
        centroids_orig = model.centroids * model.feature_std + model.feature_mean
        plt.scatter(
            centroids_orig[:, 0],
            centroids_orig[:, 1],
            c=&quot;red&quot;,
            marker=&quot;X&quot;,
            s=200,
            edgecolors=&quot;black&quot;,
            linewidths=2,
            label=&quot;Centroids&quot;,
        )

    plt.colorbar(scatter, label=&quot;Cluster&quot;)
    plt.xlabel(&quot;Annual Income (k$)&quot;)
    plt.ylabel(&quot;Spending Score (0-100)&quot;)
    plt.title(f&quot;Market Segmentation Clusters - v{version}&quot;)
    plt.grid(True, alpha=0.3)
    plt.legend()

    chart_path = output_dir / f&quot;market_segmentation_v{version}.png&quot;
    plt.tight_layout()
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info(&quot;Chart saved&quot;, path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description=&quot;Train market segmentation K-Means model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-clusters&quot;, type=int, default=int(os.getenv(&quot;N_CLUSTERS&quot;, &quot;5&quot;)))
    parser.add_argument(
        &quot;--max-iterations&quot;, type=int, default=int(os.getenv(&quot;MAX_ITERATIONS&quot;, &quot;300&quot;))
    )
    parser.add_argument(&quot;--n-init&quot;, type=int, default=int(os.getenv(&quot;N_INIT&quot;, &quot;10&quot;)))
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
        n_clusters=args.n_clusters,
        max_iterations=args.max_iterations,
        n_init=args.n_init,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )

    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-3777040552')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3777040552"><code class="language-python">&quot;&quot;&quot;Production serving API for market segmentation (unsupervised K-Means).&quot;&quot;&quot;

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
from ai_core.validation import DataValidator, create_market_segmentation_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from market_segmentation.data import FEATURE_NAMES
from market_segmentation.model import KMeans

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;METRICS_PORT&quot;, os.getenv(&quot;MARKET_METRICS_PORT&quot;, &quot;8003&quot;)))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class SegmentRequest(BaseModel):
    &quot;&quot;&quot;Single customer segmentation request.&quot;&quot;&quot;

    annual_income: float = Field(
        ..., gt=0, le=200, description=&quot;Annual income in thousands of dollars&quot;
    )
    spending_score: float = Field(..., ge=0, le=100, description=&quot;Spending score (0-100)&quot;)


class SegmentBulkRequest(BaseModel):
    &quot;&quot;&quot;Bulk customer segmentation request.&quot;&quot;&quot;

    customers: list[SegmentRequest] = Field(..., min_length=1, max_length=100)


class SegmentResponse(BaseModel):
    &quot;&quot;&quot;Segmentation response for a single customer.&quot;&quot;&quot;

    annual_income: float
    spending_score: float
    segment: int
    segment_name: str
    confidence: float
    model_version: str


class BulkSegmentResponse(BaseModel):
    &quot;&quot;&quot;Bulk segmentation response.&quot;&quot;&quot;

    segments: list[SegmentResponse]
    model_version: str


class ProfilesResponse(BaseModel):
    &quot;&quot;&quot;Cluster profiles for business interpretation.&quot;&quot;&quot;

    n_clusters: int
    profiles: list[dict]
    model_version: str


class DriftResponse(BaseModel):
    &quot;&quot;&quot;Drift detection response.&quot;&quot;&quot;

    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


# Human-readable segment names (assigned by cluster index after training)
SEGMENT_NAMES = [
    &quot;Premium Shoppers&quot;,
    &quot;Cautious High-Earners&quot;,
    &quot;Impulsive Shoppers&quot;,
    &quot;Budget-Conscious&quot;,
    &quot;Average Shoppers&quot;,
]


# Global model state
_model: KMeans | None = None
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
    _metrics = MetricsCollector(&quot;market_segmentation&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_market_segmentation_schema())
    _drift_detector = DriftDetector(
        feature_names=FEATURE_NAMES,
        feature_types={&quot;annual_income&quot;: &quot;float&quot;, &quot;spending_score&quot;: &quot;float&quot;},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;market-segmentation&quot;, model_version=_model_version, model_type=&quot;clustering&quot;
    )

    # Load reference data for drift detection
    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;market-segmentation&quot;, version=_model_version)

    yield

    logger.info(&quot;Shutting down market-segmentation API&quot;)


def _load_model() -&gt; tuple[KMeans, str]:
    &quot;&quot;&quot;Load the latest model from the registry or model directory with resilient fallback.&quot;&quot;&quot;
    # 1. Try model registry
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            seg_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;market-segmentation&quot;]
            if seg_models:
                seg_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = seg_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;market_segmentation_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return KMeans.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;market-segmentation&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;market_segmentation_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return KMeans.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    # 2. Try direct model in MODEL_DIR
    npz_path = MODEL_DIR / &quot;market_segmentation_model.npz&quot;
    if npz_path.exists():
        return KMeans.load(str(npz_path)), &quot;legacy&quot;

    # 3. Try bundled artifacts directory
    candidate_paths = [
        Path(&quot;/app/artifacts/models/market_segmentation_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3]
        / &quot;artifacts&quot;
        / &quot;models&quot;
        / &quot;market_segmentation_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return KMeans.load(str(p)), &quot;1.0.0-bundled&quot;

    # 4. In-memory baseline fallback (never crash cold start)
    logger.warning(&quot;No pre-existing model found on disk. Initializing baseline K-Means model.&quot;)
    from market_segmentation.data import load_training_data

    X_base, _ = load_training_data(None)
    model = KMeans(n_clusters=5, max_iterations=300, n_init=10, random_seed=42)
    model.fit(X_base)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    &quot;&quot;&quot;Load reference training data for drift detection.&quot;&quot;&quot;
    candidate_csvs = [
        MODEL_DIR / &quot;market-segmentation&quot; / _model_version / &quot;training_data.csv&quot;,
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

    from market_segmentation.data import load_training_data

    X_base, _ = load_training_data(None)
    return X_base


def _segment_name(segment: int) -&gt; str:
    &quot;&quot;&quot;Return a human-readable name for a segment index.&quot;&quot;&quot;
    if 0 &lt;= segment &lt; len(SEGMENT_NAMES):
        return SEGMENT_NAMES[segment]
    return f&quot;Segment {segment}&quot;


# Create FastAPI app
app = FastAPI(
    title=&quot;Market Segmentation API&quot;,
    description=&quot;Unsupervised K-Means clustering for customer market segmentation&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

# Add observability middleware
add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    &quot;&quot;&quot;Service information.&quot;&quot;&quot;
    return {
        &quot;service&quot;: &quot;market-segmentation-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;features&quot;: FEATURE_NAMES,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;segment&quot;: &quot;POST /segment&quot;,
            &quot;segment_bulk&quot;: &quot;POST /segment/bulk&quot;,
            &quot;profiles&quot;: &quot;GET /profiles&quot;,
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
                model_name=&quot;market-segmentation&quot;,
                model_version=_model_version,
                model_type=&quot;clustering&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(
            &quot;Model reloaded dynamically&quot;, model=&quot;market-segmentation&quot;, version=_model_version
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


@app.get(&quot;/profiles&quot;, response_model=ProfilesResponse)
def get_profiles():
    &quot;&quot;&quot;Return cluster profiles for business interpretation.&quot;&quot;&quot;
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    # Recompute profiles from reference data
    profiles = _model.cluster_profiles(_reference_data) if _reference_data is not None else []

    return ProfilesResponse(
        n_clusters=_model.n_clusters,
        profiles=profiles,
        model_version=_model_version,
    )


def _compute_segment(customer: SegmentRequest) -&gt; SegmentResponse:
    &quot;&quot;&quot;Core segmentation logic shared by all segment endpoints.&quot;&quot;&quot;
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    # Validate input
    X = np.array([[customer.annual_income, customer.spending_score]])
    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        segment = int(_model.predict(X)[0])
        confidence = float(_model.predict_confidence(X)[0])
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        # Track for drift detection
        _recent_predictions.append([customer.annual_income, customer.spending_score])
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return SegmentResponse(
            annual_income=customer.annual_income,
            spending_score=customer.spending_score,
            segment=segment,
            segment_name=_segment_name(segment),
            confidence=round(confidence, 4),
            model_version=_model_version,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Segmentation failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Segmentation failed&quot;) from e


@app.post(&quot;/segment&quot;, response_model=SegmentResponse)
def segment_customer(body: SegmentRequest):
    &quot;&quot;&quot;Segment a single customer.&quot;&quot;&quot;
    return _compute_segment(body)


@app.post(&quot;/segment/bulk&quot;, response_model=BulkSegmentResponse)
def segment_bulk(body: SegmentBulkRequest):
    &quot;&quot;&quot;Segment multiple customers.&quot;&quot;&quot;
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    X = np.array([[c.annual_income, c.spending_score] for c in body.customers])
    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        segments = _model.predict(X)
        confidences = _model.predict_confidence(X)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        # Track for drift detection
        _recent_predictions.extend(X.tolist())
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions = _recent_predictions[-1000:]

        responses = [
            SegmentResponse(
                annual_income=c.annual_income,
                spending_score=c.spending_score,
                segment=int(seg),
                segment_name=_segment_name(int(seg)),
                confidence=round(float(conf), 4),
                model_version=_model_version,
            )
            for c, seg, conf in zip(body.customers, segments, confidences, strict=False)
        ]
        return BulkSegmentResponse(segments=responses, model_version=_model_version)
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Bulk segmentation failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Bulk segmentation failed&quot;) from e</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-2577752954')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2577752954"><code class="language-bash">uv run python -m market_segmentation.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>
<div class="related-links">
<h3>Related Apps</h3>
<ul><li><a href="../advanced-semantic-segmentation/README.md">advanced-semantic-segmentation</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>market-segmentation</strong></p>
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