<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>regression-house-price - AI App Documentation</title>
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
<p class="section-subtitle">Linear Regression — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$\hat{y} = w \cdot x + b$$</div>
<div class="math-block">$$\mathcal{L}_{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2$$</div>
<div class="math-block">$$\frac{\partial \mathcal{L}}{\partial w} = -\frac{2}{n} \sum_{i=1}^{n} x_i(y_i - \hat{y}_i)$$</div>
<div class="math-block">$$\frac{\partial \mathcal{L}}{\partial b} = -\frac{2}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)$$</div>
<div class="math-block">$$w \leftarrow w - \alpha \cdot \frac{\partial \mathcal{L}}{\partial w}, \quad b \leftarrow b - \alpha \cdot \frac{\partial \mathcal{L}}{\partial b}$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>Starting from the hypothesis $h(x) = wx + b$, we minimize the MSE loss. Taking partial derivatives w.r.t. $w$ and $b$ and applying gradient descent yields the update rules. The learning rate $\alpha$ controls step size; too large causes divergence, too small causes slow convergence.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive scatter plot with regression line, showing loss descent over iterations.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  HousePriceNN</pre>
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
<button class="copy-btn" onclick="copyCode('code-395873256')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-395873256"><code class="language-python">&quot;&quot;&quot;Training pipeline for house price prediction using a feedforward neural network.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_house_price_schema

from regression_house_price.data import load_training_data, save_training_data, train_test_split
from regression_house_price.model import HousePriceNN

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 1000,
    hidden_dim: int = 32,
    learning_rate: float = 0.001,
    n_iterations: int = 2000,
    weight_decay: float = 0.0001,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -&gt; dict:
    &quot;&quot;&quot;Train the house price prediction neural network and save artifacts.&quot;&quot;&quot;
    X, y = load_training_data(data_path, n_samples=n_samples, random_seed=random_seed)
    logger.info(&quot;Loaded training data&quot;, n_samples=len(X), data_path=str(data_path))

    # Validate training data
    validator = DataValidator(create_house_price_schema())
    validation = validator.validate(X, y)
    if not validation.valid:
        logger.error(&quot;Training data validation failed&quot;, errors=validation.errors)
        raise ValueError(f&quot;Training data validation failed: {validation.errors}&quot;)
    logger.info(&quot;Training data validated&quot;, stats=validation.stats)

    # Split train/test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_seed=random_seed
    )
    logger.info(
        &quot;Data split&quot;,
        n_train=len(X_train),
        n_test=len(X_test),
        test_size=test_size,
        random_seed=random_seed,
    )

    # Save training data for reproducibility
    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / &quot;training_data.csv&quot;)

    # Train model
    model = HousePriceNN(
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    # Evaluate
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(
        &quot;Training complete&quot;,
        training_mode=model.training_mode,
        n_epochs=len(model.loss_history),
        final_loss=model.loss_history[-1] if model.loss_history else 0.0,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )

    # Save model
    model_path = model_dir / f&quot;house_price_model_v{model_version}.npz&quot;
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, model_dir, model_version)

    # Combined metrics for registry
    metrics = {
        **test_metrics,
        &quot;training_mode&quot;: &quot;supervised&quot;,
        &quot;n_epochs_run&quot;: float(len(model.loss_history)),
        &quot;final_loss&quot;: model.loss_history[-1] if model.loss_history else 0.0,
        &quot;train_mse&quot;: train_metrics[&quot;mse&quot;],
        &quot;train_r2&quot;: train_metrics[&quot;r2&quot;],
        &quot;n_train_samples&quot;: float(len(X_train)),
        &quot;n_test_samples&quot;: float(len(X_test)),
        &quot;hidden_dim&quot;: float(hidden_dim),
        &quot;learning_rate&quot;: float(learning_rate),
        &quot;weight_decay&quot;: float(weight_decay),
        &quot;n_features&quot;: float(X_train.shape[1]),
    }

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;house-price-prediction&quot;,
        model_version=model_version,
        model_type=&quot;regression&quot;,
        metrics=metrics,
        parameters={
            &quot;hidden_dim&quot;: hidden_dim,
            &quot;learning_rate&quot;: learning_rate,
            &quot;n_iterations&quot;: n_iterations,
            &quot;weight_decay&quot;: weight_decay,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;house_price_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.csv&quot;: model_dir / &quot;training_data.csv&quot;,
        },
        tags={
            &quot;framework&quot;: &quot;numpy&quot;,
            &quot;task&quot;: &quot;regression&quot;,
            &quot;model_type&quot;: &quot;feedforward_neural_network&quot;,
        },
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;house-price-prediction&quot;,
            model_version=model_version,
            metrics=metrics,
            params={
                &quot;hidden_dim&quot;: hidden_dim,
                &quot;learning_rate&quot;: learning_rate,
                &quot;n_iterations&quot;: n_iterations,
                &quot;weight_decay&quot;: weight_decay,
                &quot;random_seed&quot;: random_seed,
            },
            artifacts={
                &quot;model&quot;: str(model_path),
                &quot;chart&quot;: str(model_dir / f&quot;house_price_regression_v{model_version}.png&quot;),
                &quot;training_data&quot;: str(model_dir / &quot;training_data.csv&quot;),
            },
            tags={&quot;model_type&quot;: &quot;regression&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )
        logger.info(
            &quot;Registered model to MLflow&quot;, model=&quot;house-price-prediction&quot;, version=model_version
        )

    return metrics


def _save_chart(model: HousePriceNN, output_dir: Path, version: str) -&gt; None:
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
    ax.set_title(&quot;House Price Prediction NN Training Loss&quot;)
    ax.grid(True, alpha=0.3)
    ax.set_yscale(&quot;log&quot;)

    plt.tight_layout()
    chart_path = output_dir / f&quot;house_price_regression_v{version}.png&quot;
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info(&quot;Chart saved&quot;, path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description=&quot;Train house price prediction neural network&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;1000&quot;)))
    parser.add_argument(&quot;--hidden-dim&quot;, type=int, default=int(os.getenv(&quot;HIDDEN_DIM&quot;, &quot;32&quot;)))
    parser.add_argument(
        &quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.001&quot;))
    )
    parser.add_argument(&quot;--n-iterations&quot;, type=int, default=int(os.getenv(&quot;N_ITERATIONS&quot;, &quot;2000&quot;)))
    parser.add_argument(
        &quot;--weight-decay&quot;, type=float, default=float(os.getenv(&quot;WEIGHT_DECAY&quot;, &quot;0.0001&quot;))
    )
    parser.add_argument(&quot;--model-version&quot;, type=str, default=os.getenv(&quot;MODEL_VERSION&quot;, &quot;1.0.0&quot;))
    parser.add_argument(&quot;--test-size&quot;, type=float, default=float(os.getenv(&quot;TEST_SIZE&quot;, &quot;0.2&quot;)))
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
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-649754494')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-649754494"><code class="language-python">&quot;&quot;&quot;Production serving API for house price prediction via feedforward neural network.&quot;&quot;&quot;

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
from ai_core.validation import DataValidator, create_house_price_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from regression_house_price.data import FEATURE_NAMES
from regression_house_price.model import HousePriceNN

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;METRICS_PORT&quot;, os.getenv(&quot;HOUSE_PRICE_METRICS_PORT&quot;, &quot;8009&quot;)))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class PredictRequest(BaseModel):
    &quot;&quot;&quot;House price prediction request.&quot;&quot;&quot;

    features: list[float] = Field(..., min_length=10, max_length=10)


class PredictBulkRequest(BaseModel):
    &quot;&quot;&quot;Bulk house price prediction request.&quot;&quot;&quot;

    requests: list[list[float]] = Field(..., min_length=1, max_length=100)


class PredictResponse(BaseModel):
    &quot;&quot;&quot;Prediction response.&quot;&quot;&quot;

    predicted_price: float
    model_version: str
    training_mode: str


class BulkPredictResponse(BaseModel):
    &quot;&quot;&quot;Bulk prediction response.&quot;&quot;&quot;

    predictions: list[PredictResponse]
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
    model_version: str


# Global model state
_model: HousePriceNN | None = None
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
    _metrics = MetricsCollector(&quot;house_price_prediction&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_house_price_schema())
    _drift_detector = DriftDetector(
        feature_names=FEATURE_NAMES,
        feature_types={f: &quot;float&quot; for f in FEATURE_NAMES},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;house-price-prediction&quot;,
        model_version=_model_version,
        model_type=&quot;regression&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;house-price-prediction&quot;, version=_model_version)

    yield

    logger.info(&quot;Shutting down house-price-prediction API&quot;)


def _load_model() -&gt; tuple[HousePriceNN, str]:
    &quot;&quot;&quot;Load the latest model from the registry or model directory with resilient fallback.&quot;&quot;&quot;
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            hp_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;house-price-prediction&quot;]
            if hp_models:
                hp_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = hp_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;house_price_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return HousePriceNN.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;house-price-prediction&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;house_price_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return HousePriceNN.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    # Try direct model in MODEL_DIR
    npz_path = MODEL_DIR / &quot;house_price_model.npz&quot;
    if npz_path.exists():
        return HousePriceNN.load(str(npz_path)), &quot;legacy&quot;

    # Try bundled artifacts directory
    candidate_paths = [
        Path(&quot;/app/artifacts/models/house_price_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3]
        / &quot;artifacts&quot;
        / &quot;models&quot;
        / &quot;house_price_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return HousePriceNN.load(str(p)), &quot;1.0.0-bundled&quot;

    # In-memory baseline fallback
    logger.warning(&quot;No pre-existing model found on disk. Initializing baseline NN model.&quot;)
    from regression_house_price.data import generate_synthetic_data

    X_base, y_base = generate_synthetic_data(n_samples=200, random_seed=42)
    model = HousePriceNN(hidden_dim=32, learning_rate=0.001, n_iterations=500, random_seed=42)
    model.fit(X_base, y_base)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    &quot;&quot;&quot;Load reference training data for drift detection.&quot;&quot;&quot;
    candidate_csvs = [
        MODEL_DIR / &quot;house-price-prediction&quot; / _model_version / &quot;training_data.csv&quot;,
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

    from regression_house_price.data import generate_synthetic_data

    X_base, _ = generate_synthetic_data(n_samples=200, random_seed=42)
    return X_base


# Create FastAPI app
app = FastAPI(
    title=&quot;House Price Prediction API&quot;,
    description=&quot;Feedforward neural network for predicting house prices from features&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    &quot;&quot;&quot;Service information.&quot;&quot;&quot;
    return {
        &quot;service&quot;: &quot;house-price-prediction-api&quot;,
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
        &quot;training_mode&quot;: _model.training_mode if _model else &quot;unknown&quot;,
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
                model_name=&quot;house-price-prediction&quot;,
                model_version=_model_version,
                model_type=&quot;regression&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(
            &quot;Model reloaded dynamically&quot;, model=&quot;house-price-prediction&quot;, version=_model_version
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
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )


def _compute_prediction(features: list[float]) -&gt; PredictResponse:
    &quot;&quot;&quot;Core prediction logic shared by all prediction endpoints.&quot;&quot;&quot;
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    X = np.array([features])

    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        price = float(_model.predict(X)[0])
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append(features)
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return PredictResponse(
            predicted_price=round(price, 2),
            model_version=_model_version,
            training_mode=_model.training_mode if _model else &quot;unknown&quot;,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Prediction failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Prediction failed&quot;) from e


@app.post(&quot;/predict&quot;, response_model=PredictResponse)
def predict(body: PredictRequest):
    &quot;&quot;&quot;Predict house price for a single property.&quot;&quot;&quot;
    return _compute_prediction(body.features)


@app.post(&quot;/predict/bulk&quot;, response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    &quot;&quot;&quot;Predict house prices for multiple properties.&quot;&quot;&quot;
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    if len(body.requests) &lt; 1 or len(body.requests) &gt; 100:
        raise HTTPException(status_code=422, detail=&quot;Batch size must be between 1 and 100&quot;)

    X = np.array(body.requests)

    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        predictions = _model.predict(X)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.extend(body.requests)
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions = _recent_predictions[-1000:]

        results = [
            PredictResponse(
                predicted_price=round(float(p), 2),
                model_version=_model_version,
                training_mode=_model.training_mode if _model else &quot;unknown&quot;,
            )
            for p in predictions
        ]
        return BulkPredictResponse(predictions=results, model_version=_model_version)
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Bulk prediction failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Bulk prediction failed&quot;) from e</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-2868036556')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2868036556"><code class="language-bash">uv run python -m regression_house_price.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>
<div class="related-links">
<h3>Related Apps</h3>
<ul><li><a href="../pizza-price/README.md">pizza-price</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>regression-house-price</strong></p>
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