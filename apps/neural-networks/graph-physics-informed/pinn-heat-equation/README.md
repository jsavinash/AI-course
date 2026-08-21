<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pinn-heat-equation - AI App Documentation</title>
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
<p class="section-subtitle">Physics-Informed Neural Network (PINN) — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$\mathcal{L}_{total} = \mathcal{L}_{data} + \lambda \mathcal{L}_{pde}$$</div>
<div class="math-block">$$\mathcal{L}_{data} = \frac{1}{N} \sum_{i=1}^{N} |u_\theta(x_i, t_i) - u_i|^2$$</div>
<div class="math-block">$$\mathcal{L}_{pde} = \frac{1}{N_f} \sum_{i=1}^{N_f} \left| \mathcal{F}\left(u_\theta, x_i, t_i; \frac{\partial u_\theta}{\partial x}, \frac{\partial u_\theta}{\partial t}, \ldots \right) \right|^2$$</div>
<div class="math-block">$$u_t + u u_x = \nu u_{xx} \quad \text{(Burgers' equation)}$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>PINNs embed physical laws as soft constraints via automatic differentiation. The total loss combines data fitting $\mathcal{L}_{data}$ and PDE residual $\mathcal{L}_{pde}$. Gradients of $u_\theta$ w.r.t. inputs are computed symbolically via autograd. This enables solving PDEs without labeled data in the domain interior.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive PDE solution comparison: PINN vs finite difference; residual heatmap; loss decomposition pie chart.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  PINNHeatEquation</pre>
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
<button class="copy-btn" onclick="copyCode('code-3035966504')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3035966504"><code class="language-python">&quot;&quot;&quot;Training pipeline for PINN Heat Equation Solver.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_pinn_heat_equation_schema

from pinn_heat_equation.data import (
    generate_synthetic_data,
    save_training_data,
    train_test_split,
)
from pinn_heat_equation.model import PINNHeatEquation

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 200,
    alpha: float = 0.01,
    hidden_dim: int = 32,
    n_layers: int = 2,
    learning_rate: float = 0.01,
    n_iterations: int = 500,
    weight_decay: float = 0.001,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -&gt; dict:
    X, u_true = generate_synthetic_data(n_samples=n_samples, random_seed=random_seed, alpha=alpha)
    logger.info(&quot;Generated PDE training data&quot;, n_samples=n_samples, alpha=alpha)

    validator = DataValidator(create_pinn_heat_equation_schema())
    validation = validator.validate(X.reshape(-1, 1))
    if not validation.valid:
        logger.error(&quot;Training data validation failed&quot;, errors=validation.errors)
        raise ValueError(f&quot;Training data validation failed: {validation.errors}&quot;)

    X_train, X_test, u_train, u_test = train_test_split(X, u_true, test_size=test_size, random_seed=random_seed)
    logger.info(&quot;Data split&quot;, n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, u_true, model_dir / &quot;training_data.npz&quot;)

    model = PINNHeatEquation(
        alpha=alpha,
        hidden_dim=hidden_dim,
        n_layers=n_layers,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X_train, u_train)

    test_metrics = model.evaluate(X_test, u_test)
    logger.info(&quot;Training complete&quot;, training_mode=model.training_mode, final_loss=model.loss_history[-1])

    model_path = model_dir / f&quot;pinn_model_v{model_version}.npz&quot;
    model.save(str(model_path))

    metrics = {
        **test_metrics,
        &quot;training_mode&quot;: &quot;physics-informed&quot;,
        &quot;n_epochs_run&quot;: float(len(model.loss_history)),
        &quot;final_loss&quot;: model.loss_history[-1] if model.loss_history else 0.0,
        &quot;n_train_samples&quot;: float(len(X_train)),
        &quot;n_test_samples&quot;: float(len(X_test)),
        &quot;alpha&quot;: float(alpha),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;pinn-heat-equation&quot;,
        model_version=model_version,
        model_type=&quot;regression&quot;,
        metrics=metrics,
        parameters={
            &quot;alpha&quot;: alpha,
            &quot;hidden_dim&quot;: hidden_dim,
            &quot;n_layers&quot;: n_layers,
            &quot;learning_rate&quot;: learning_rate,
            &quot;n_iterations&quot;: n_iterations,
            &quot;weight_decay&quot;: weight_decay,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;pinn_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.npz&quot;: model_dir / &quot;training_data.npz&quot;,
        },
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;pinn_heat_equation&quot;, &quot;model_type&quot;: &quot;PINN&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;pinn-heat-equation&quot;,
            model_version=model_version,
            metrics=metrics,
            params={&quot;alpha&quot;: alpha, &quot;hidden_dim&quot;: hidden_dim, &quot;n_layers&quot;: n_layers, &quot;learning_rate&quot;: learning_rate, &quot;n_iterations&quot;: n_iterations},
            artifacts={&quot;model&quot;: str(model_path)},
            tags={&quot;model_type&quot;: &quot;pinn&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description=&quot;Train PINN Heat Equation model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;200&quot;)))
    parser.add_argument(&quot;--alpha&quot;, type=float, default=float(os.getenv(&quot;ALPHA&quot;, &quot;0.01&quot;)))
    parser.add_argument(&quot;--hidden-dim&quot;, type=int, default=int(os.getenv(&quot;HIDDEN_DIM&quot;, &quot;32&quot;)))
    parser.add_argument(&quot;--n-layers&quot;, type=int, default=int(os.getenv(&quot;N_LAYERS&quot;, &quot;2&quot;)))
    parser.add_argument(&quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.01&quot;)))
    parser.add_argument(&quot;--n-iterations&quot;, type=int, default=int(os.getenv(&quot;N_ITERATIONS&quot;, &quot;500&quot;)))
    parser.add_argument(&quot;--weight-decay&quot;, type=float, default=float(os.getenv(&quot;WEIGHT_DECAY&quot;, &quot;0.001&quot;)))
    parser.add_argument(&quot;--model-version&quot;, type=str, default=os.getenv(&quot;MODEL_VERSION&quot;, &quot;1.0.0&quot;))
    parser.add_argument(&quot;--test-size&quot;, type=float, default=float(os.getenv(&quot;TEST_SIZE&quot;, &quot;0.2&quot;)))
    parser.add_argument(&quot;--random-seed&quot;, type=int, default=int(os.getenv(&quot;RANDOM_SEED&quot;, &quot;42&quot;)))
    parser.add_argument(&quot;--register-mlflow&quot;, action=&quot;store_true&quot;, default=os.getenv(&quot;REGISTER_MLFLOW&quot;, &quot;false&quot;).lower() == &quot;true&quot;)
    parser.add_argument(&quot;--log-level&quot;, type=str, default=os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_samples=args.n_samples,
        alpha=args.alpha,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
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
<button class="copy-btn" onclick="copyCode('code-1695321246')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-1695321246"><code class="language-python">&quot;&quot;&quot;Serving API for PINN Heat Equation Solver.&quot;&quot;&quot;

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
from ai_core.validation import DataValidator, create_pinn_heat_equation_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from pinn_heat_equation.data import N_FEATURES, generate_synthetic_data
from pinn_heat_equation.model import PINNHeatEquation

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;PINN_METRICS_PORT&quot;, &quot;8030&quot;))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class PredictRequest(BaseModel):
    x: float = Field(..., ge=0.0, le=1.0)
    t: float = Field(..., ge=0.0, le=0.5)


class PredictBulkRequest(BaseModel):
    requests: list[dict] = Field(..., min_length=1, max_length=50)


class PredictResponse(BaseModel):
    temperature: float
    physics_residual: float
    model_version: str
    training_mode: str


class BulkPredictResponse(BaseModel):
    predictions: list[PredictResponse]
    model_version: str


class DriftResponse(BaseModel):
    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


class StatsResponse(BaseModel):
    alpha: float
    hidden_dim: int
    n_layers: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: PINNHeatEquation | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_validator: DataValidator | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _validator, _drift_detector, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;pinn_heat_equation&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_pinn_heat_equation_schema())
    feature_names = [&quot;x&quot;, &quot;t&quot;]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={&quot;x&quot;: &quot;float&quot;, &quot;t&quot;: &quot;float&quot;},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;pinn-heat-equation&quot;,
        model_version=_model_version,
        model_type=&quot;regression&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;pinn-heat-equation&quot;, version=_model_version)

    yield
    logger.info(&quot;Shutting down pinn-heat-equation API&quot;)


def _load_model() -&gt; tuple[PINNHeatEquation, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            nn_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;pinn-heat-equation&quot;]
            if nn_models:
                nn_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;pinn_model_*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return PINNHeatEquation.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;pinn-heat-equation&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;pinn_model_*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return PINNHeatEquation.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    npz_path = MODEL_DIR / &quot;pinn_model.npz&quot;
    if npz_path.exists():
        return PINNHeatEquation.load(str(npz_path)), &quot;legacy&quot;

    candidate_paths = [
        Path(&quot;/app/artifacts/models/pinn_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;pinn_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return PINNHeatEquation.load(str(p)), &quot;1.0.0-bundled&quot;

    logger.warning(&quot;No pre-existing model found. Initializing baseline model.&quot;)
    X_base, u_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = PINNHeatEquation(
        alpha=0.01,
        hidden_dim=16,
        n_layers=2,
        learning_rate=0.01,
        n_iterations=50,
        random_seed=42,
    )
    model.fit(X_base, u_base)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    return X_base


app = FastAPI(
    title=&quot;PINN Heat Equation Solver API&quot;,
    description=&quot;Solves supervised learning tasks while respecting physical laws described by differential equations&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;pinn_heat_equation-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;training_mode&quot;: _model.training_mode if _model else &quot;unknown&quot;,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;predict&quot;: &quot;POST /predict&quot;,
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
                model_name=&quot;pinn-heat-equation&quot;,
                model_version=_model_version,
                model_type=&quot;regression&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(&quot;Model reloaded&quot;, model=&quot;pinn-heat-equation&quot;, version=_model_version)
        return {&quot;status&quot;: &quot;reloaded&quot;, &quot;model_version&quot;: _model_version}
    except Exception as e:
        logger.exception(&quot;Model reload failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=f&quot;Reload failed: {e}&quot;) from e


@app.get(&quot;/drift&quot;, response_model=DriftResponse)
def drift_check():
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail=&quot;Drift detection not available&quot;)
    if len(_recent_predictions) &lt; 10:
        return {&quot;total_features&quot;: N_FEATURES, &quot;drifted_features&quot;: 0, &quot;drift_ratio&quot;: 0.0, &quot;drifted&quot;: [], &quot;all_results&quot;: []}
    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)
    if _metrics:
        _metrics.set_drift_ratio(summary[&quot;drift_ratio&quot;])
    return summary


@app.get(&quot;/stats&quot;, response_model=StatsResponse)
def get_stats():
    if _model is None or not _model.weights:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    return StatsResponse(
        alpha=_model.alpha,
        hidden_dim=_model.hidden_dim,
        n_layers=_model.n_layers,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )


@app.post(&quot;/predict&quot;, response_model=PredictResponse)
def predict(body: PredictRequest):
    &quot;&quot;&quot;Predict temperature u(x, t) using physics-informed network.&quot;&quot;&quot;
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    X = np.array([[body.x, body.t]])
    validation = _validator.validate(X.reshape(-1, 1))
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        u_pred = _model.predict(X)[0]
        residual = _model.predict_proba(X)[0]
        response = PredictResponse(
            temperature=round(float(u_pred), 6),
            physics_residual=round(float(residual), 6),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append([body.x, body.t])
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Prediction failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Prediction failed&quot;) from e


@app.post(&quot;/predict/bulk&quot;, response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    &quot;&quot;&quot;Make multiple predictions.&quot;&quot;&quot;
    global _recent_predictions
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    if len(body.requests) &lt; 1 or len(body.requests) &gt; 50:
        raise HTTPException(status_code=422, detail=&quot;Batch size must be between 1 and 50&quot;)

    predictions = []
    for req in body.requests:
        x = float(req.get(&quot;x&quot;, 0.5))
        t = float(req.get(&quot;t&quot;, 0.1))
        X = np.array([[x, t]])
        u_pred = _model.predict(X)[0]
        residual = _model.predict_proba(X)[0]
        predictions.append(PredictResponse(
            temperature=round(float(u_pred), 6),
            physics_residual=round(float(residual), 6),
            model_version=_model_version,
            training_mode=_model.training_mode,
        ))
        _recent_predictions.append([x, t])
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

    return BulkPredictResponse(predictions=predictions, model_version=_model_version)</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-947146450')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-947146450"><code class="language-bash">uv run python -m pinn_heat_equation.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>

</main>
<footer class="app-footer">
<p>Generated documentation for <strong>pinn-heat-equation</strong></p>
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