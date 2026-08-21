<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pizza-price - AI App Documentation</title>
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
<pre class="ascii-diagram">  LinearRegression</pre>
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
<button class="copy-btn" onclick="copyCode('code-2563474581')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2563474581"><code class="language-python">&quot;&quot;&quot;Production training pipeline for pizza price prediction.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

import numpy as np
from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_pizza_schema

from pizza_price.data import load_training_data, save_training_data, train_test_split
from pizza_price.model import LinearRegression

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path,
    learning_rate: float,
    n_iterations: int,
    model_version: str,
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -&gt; dict:
    &quot;&quot;&quot;Train the pizza price model and save artifacts.&quot;&quot;&quot;
    # Load training data
    X, y = load_training_data(data_path)
    logger.info(&quot;Loaded training data&quot;, n_samples=len(X), data_path=str(data_path))

    # Validate training data
    validator = DataValidator(create_pizza_schema())
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
    save_training_data(X, y, model_dir / &quot;training_data.csv&quot;)

    # Train model
    model = LinearRegression(learning_rate=learning_rate, n_iterations=n_iterations)
    model.fit(X_train, y_train)

    # Evaluate on train and test
    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(
        &quot;Training complete&quot;,
        weight=model.weight,
        bias=model.bias,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        iterations=n_iterations,
    )

    # Model validation - check metrics meet thresholds
    if test_metrics[&quot;rmse&quot;] &gt; 5.0:
        logger.warning(&quot;Model RMSE above threshold&quot;, rmse=test_metrics[&quot;rmse&quot;], threshold=5.0)

    # Save model
    model_path = model_dir / f&quot;pizza_model_v{model_version}.npz&quot;
    model.save(str(model_path))

    # Save training chart
    _save_chart(model, X, y, model_dir, model_version)

    # Combined metrics for registry
    metrics = {
        &quot;mse&quot;: test_metrics[&quot;mse&quot;],
        &quot;rmse&quot;: test_metrics[&quot;rmse&quot;],
        &quot;mae&quot;: test_metrics[&quot;mae&quot;],
        &quot;r2&quot;: test_metrics[&quot;r2&quot;],
        &quot;train_mse&quot;: train_metrics[&quot;mse&quot;],
        &quot;train_r2&quot;: train_metrics[&quot;r2&quot;],
        &quot;weight&quot;: model.weight,
        &quot;bias&quot;: model.bias,
        &quot;n_samples&quot;: len(X),
        &quot;n_train&quot;: len(X_train),
        &quot;n_test&quot;: len(X_test),
    }

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;pizza-price&quot;,
        model_version=model_version,
        model_type=&quot;regression&quot;,
        metrics=metrics,
        parameters={
            &quot;learning_rate&quot;: learning_rate,
            &quot;n_iterations&quot;: n_iterations,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;pizza_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.csv&quot;: model_dir / &quot;training_data.csv&quot;,
        },
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;regression&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;pizza-price&quot;,
            model_version=model_version,
            metrics=metrics,
            params={
                &quot;learning_rate&quot;: learning_rate,
                &quot;n_iterations&quot;: n_iterations,
                &quot;random_seed&quot;: random_seed,
            },
            artifacts={
                &quot;model&quot;: str(model_path),
                &quot;chart&quot;: str(model_dir / f&quot;pizza_regression_v{model_version}.png&quot;),
                &quot;training_data&quot;: str(model_dir / &quot;training_data.csv&quot;),
            },
            tags={&quot;model_type&quot;: &quot;regression&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )
        logger.info(&quot;Registered model to MLflow&quot;, model=&quot;pizza-price&quot;, version=model_version)

    return metrics


def _save_chart(
    model: LinearRegression, X: np.ndarray, y: np.ndarray, output_dir: Path, version: str
) -&gt; None:
    &quot;&quot;&quot;Save the regression chart.&quot;&quot;&quot;
    import matplotlib

    matplotlib.use(&quot;Agg&quot;)
    import matplotlib.pyplot as plt

    plt.figure(figsize=(10, 6))
    plt.scatter(X, y, color=&quot;blue&quot;, s=100, label=&quot;Training data&quot;)

    line_x = np.linspace(min(X) - 1, max(X) + 1, 100)
    line_y = model.predict(line_x)
    plt.plot(line_x, line_y, color=&quot;red&quot;, linewidth=2, label=&quot;Fitted line&quot;)

    plt.xlabel(&quot;Pizza Diameter (inches)&quot;)
    plt.ylabel(&quot;Price (USD)&quot;)
    plt.title(f&quot;Pizza Price vs Diameter - Trained Model v{version}&quot;)
    plt.grid(True, alpha=0.3)
    plt.legend()

    chart_path = output_dir / f&quot;pizza_regression_v{version}.png&quot;
    plt.tight_layout()
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info(&quot;Chart saved&quot;, path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description=&quot;Train pizza price prediction model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(
        &quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.001&quot;))
    )
    parser.add_argument(&quot;--n-iterations&quot;, type=int, default=int(os.getenv(&quot;N_ITERATIONS&quot;, &quot;2000&quot;)))
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
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
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
<button class="copy-btn" onclick="copyCode('code-4164873965')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-4164873965"><code class="language-python">&quot;&quot;&quot;Production serving API for pizza price prediction.&quot;&quot;&quot;

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
from ai_core.validation import DataValidator, create_pizza_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from pizza_price.model import LinearRegression

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;METRICS_PORT&quot;, os.getenv(&quot;PIZZA_METRICS_PORT&quot;, &quot;8001&quot;)))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class PredictRequest(BaseModel):
    &quot;&quot;&quot;Single pizza price prediction request.&quot;&quot;&quot;

    diameter: float = Field(..., gt=0, le=50, description=&quot;Pizza diameter in inches&quot;)


class PredictBulkRequest(BaseModel):
    &quot;&quot;&quot;Bulk pizza price prediction request.&quot;&quot;&quot;

    diameters: list[float] = Field(..., min_length=1, max_length=100)


class PredictResponse(BaseModel):
    &quot;&quot;&quot;Prediction response.&quot;&quot;&quot;

    diameter: float
    predicted_price: float
    equation: str
    model_version: str


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


# Global model state
_model: LinearRegression | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_validator: DataValidator | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[float] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    &quot;&quot;&quot;Load model at startup and clean up at shutdown.&quot;&quot;&quot;
    global _model, _model_version, _metrics, _validator, _drift_detector, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;pizza_price&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_pizza_schema())
    _drift_detector = DriftDetector(
        feature_names=[&quot;diameter&quot;],
        feature_types={&quot;diameter&quot;: &quot;float&quot;},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;pizza-price&quot;, model_version=_model_version, model_type=&quot;regression&quot;
    )

    # Load reference data for drift detection
    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;pizza-price&quot;, version=_model_version)

    yield

    logger.info(&quot;Shutting down pizza-price API&quot;)


def _load_model() -&gt; tuple[LinearRegression, str]:
    &quot;&quot;&quot;Load the latest model from the registry or model directory with resilient fallback.&quot;&quot;&quot;
    # 1. Try model registry
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            pizza_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;pizza-price&quot;]
            if pizza_models:
                pizza_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = pizza_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;pizza_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return LinearRegression.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;pizza-price&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;pizza_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return LinearRegression.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    # 2. Try direct model in MODEL_DIR
    npz_path = MODEL_DIR / &quot;pizza_model.npz&quot;
    if npz_path.exists():
        return LinearRegression.load(str(npz_path)), &quot;legacy&quot;

    # 3. Try bundled artifacts directory
    candidate_paths = [
        Path(&quot;/app/artifacts/models/pizza_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;pizza_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return LinearRegression.load(str(p)), &quot;1.0.0-bundled&quot;

    # 4. In-memory baseline fallback (never crash cold start)
    logger.warning(&quot;No pre-existing model found on disk. Initializing baseline linear model.&quot;)
    from pizza_price.data import load_training_data

    X_base, y_base = load_training_data(None)
    model = LinearRegression(learning_rate=0.001, n_iterations=2000)
    model.fit(X_base, y_base)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    &quot;&quot;&quot;Load reference training data for drift detection.&quot;&quot;&quot;
    candidate_csvs = [
        MODEL_DIR / &quot;pizza-price&quot; / _model_version / &quot;training_data.csv&quot;,
        MODEL_DIR / &quot;training_data.csv&quot;,
        Path(&quot;/app/artifacts/models/training_data.csv&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;training_data.csv&quot;,
    ]
    for csv_path in candidate_csvs:
        if csv_path.exists():
            try:
                import pandas as pd

                df = pd.read_csv(csv_path)
                if &quot;diameter&quot; in df.columns:
                    return df[[&quot;diameter&quot;]].values
            except Exception as e:
                logger.warning(&quot;Could not read reference csv&quot;, path=str(csv_path), error=str(e))

    from pizza_price.data import load_training_data

    X_base, _ = load_training_data(None)
    return X_base.reshape(-1, 1)


# Create FastAPI app
app = FastAPI(
    title=&quot;Pizza Price Prediction API&quot;,
    description=&quot;Linear Regression model for predicting pizza prices from diameter&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

# Add observability middleware
add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    &quot;&quot;&quot;Service information.&quot;&quot;&quot;
    return {
        &quot;service&quot;: &quot;pizza-price-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;predict&quot;: &quot;POST /predict&quot;,
            &quot;predict_bulk&quot;: &quot;POST /predict/bulk&quot;,
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
                model_name=&quot;pizza-price&quot;, model_version=_model_version, model_type=&quot;regression&quot;
            )
        _reference_data = _load_reference_data()
        logger.info(&quot;Model reloaded dynamically&quot;, model=&quot;pizza-price&quot;, version=_model_version)
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
            total_features=1,
            drifted_features=0,
            drift_ratio=0.0,
            drifted=[],
            all_results=[],
        )

    current = np.array(_recent_predictions[-100:]).reshape(-1, 1)
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)

    return DriftResponse(**summary)


@app.post(&quot;/predict&quot;, response_model=PredictResponse)
def predict(body: PredictRequest):
    &quot;&quot;&quot;Predict pizza price for a single diameter.&quot;&quot;&quot;
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    # Validate input
    validation = _validator.validate(np.array([body.diameter]))
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        price = _model.predict(np.array([body.diameter]))[0]
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        # Track for drift detection
        _recent_predictions.append(body.diameter)
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return PredictResponse(
            diameter=body.diameter,
            predicted_price=round(float(price), 2),
            equation=f&quot;price = {_model.weight:.4f} * diameter + {_model.bias:.4f}&quot;,
            model_version=_model_version,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version)
        logger.exception(&quot;Prediction failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Prediction failed&quot;) from e


@app.post(&quot;/predict/bulk&quot;, response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    &quot;&quot;&quot;Predict pizza prices for multiple diameters.&quot;&quot;&quot;
    global _recent_predictions
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    # Validate input
    validation = _validator.validate(np.array(body.diameters))
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        diameters = np.array(body.diameters)
        prices = _model.predict(diameters)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        # Track for drift detection
        _recent_predictions.extend(body.diameters)
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions = _recent_predictions[-1000:]

        predictions = [
            PredictResponse(
                diameter=float(d),
                predicted_price=round(float(p), 2),
                equation=f&quot;price = {_model.weight:.4f} * diameter + {_model.bias:.4f}&quot;,
                model_version=_model_version,
            )
            for d, p in zip(diameters, prices, strict=False)
        ]
        return BulkPredictResponse(predictions=predictions, model_version=_model_version)
    except Exception as e:
        _metrics.record_error(model_version=_model_version)
        logger.exception(&quot;Bulk prediction failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Bulk prediction failed&quot;) from e</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-3149133432')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3149133432"><code class="language-bash">uv run python -m pizza_price.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>
<div class="related-links">
<h3>Related Apps</h3>
<ul><li><a href="../regression-house-price/README.md">regression-house-price</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>pizza-price</strong></p>
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