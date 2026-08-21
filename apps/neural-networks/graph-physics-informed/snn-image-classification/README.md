<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>snn-image-classification - AI App Documentation</title>
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
<p class="section-subtitle">Spiking Neural Network (SNN) — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$\tau_m \frac{dV_m}{dt} = -(V_m - V_{rest}) + R_m I(t)$$</div>
<div class="math-block">$$\text{if } V_m \geq V_{th}: \text{ emit spike}, V_m \leftarrow V_{reset}$$</div>
<div class="math-block">$$S(t) = \sum_{i} \delta(t - t_i)$$</div>
<div class="math-block">$$\tau_s \frac{dS}{dt} = -S$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>SNNs compute with discrete spike events rather than continuous activations. The membrane potential integrates input current and leaks over time. When the potential exceeds a threshold, a spike is emitted and the membrane is reset. This event-driven computation is energy-efficient and biologically plausible.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive membrane potential trace; spike raster plot; synaptic current decomposition.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  LIFNeuron
  SNNImageClassification</pre>
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
<button class="copy-btn" onclick="copyCode('code-465951722')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-465951722"><code class="language-python">&quot;&quot;&quot;Training pipeline for SNN Image Classification.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_snn_image_classification_schema

from snn_image_classification.data import (
    N_CLASSES,
    N_FEATURES,
    generate_synthetic_data,
    save_training_data,
    train_test_split,
)
from snn_image_classification.model import SNNImageClassification

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    hidden_dim: int = 128,
    learning_rate: float = 0.01,
    n_iterations: int = 200,
    n_timesteps: int = 10,
    weight_decay: float = 0.0001,
    threshold: float = 1.0,
    leak_rate: float = 0.9,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -&gt; dict:
    X, y = generate_synthetic_data(n_samples=n_samples, random_seed=random_seed)
    logger.info(&quot;Generated image data&quot;, n_samples=n_samples)

    validator = DataValidator(create_snn_image_classification_schema())
    validation = validator.validate(X.reshape(-1, 1))
    if not validation.valid:
        logger.error(&quot;Training data validation failed&quot;, errors=validation.errors)
        raise ValueError(f&quot;Training data validation failed: {validation.errors}&quot;)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_seed=random_seed)
    logger.info(&quot;Data split&quot;, n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / &quot;training_data.npz&quot;)

    model = SNNImageClassification(
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        n_timesteps=n_timesteps,
        weight_decay=weight_decay,
        threshold=threshold,
        leak_rate=leak_rate,
        random_seed=random_seed,
    )
    model.fit(X_train, y_train)

    test_metrics = model.evaluate(X_test, y_test)
    logger.info(&quot;Training complete&quot;, training_mode=model.training_mode, final_loss=model.loss_history[-1])

    model_path = model_dir / f&quot;snn_model_v{model_version}.npz&quot;
    model.save(str(model_path))

    metrics = {
        **test_metrics,
        &quot;training_mode&quot;: &quot;spiking&quot;,
        &quot;n_epochs_run&quot;: float(len(model.loss_history)),
        &quot;final_loss&quot;: model.loss_history[-1] if model.loss_history else 0.0,
        &quot;n_train_samples&quot;: float(len(X_train)),
        &quot;n_test_samples&quot;: float(len(X_test)),
        &quot;n_timesteps&quot;: float(n_timesteps),
        &quot;threshold&quot;: float(threshold),
        &quot;leak_rate&quot;: float(leak_rate),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;snn-image-classification&quot;,
        model_version=model_version,
        model_type=&quot;classification&quot;,
        metrics=metrics,
        parameters={
            &quot;n_features&quot;: N_FEATURES,
            &quot;n_classes&quot;: N_CLASSES,
            &quot;hidden_dim&quot;: hidden_dim,
            &quot;learning_rate&quot;: learning_rate,
            &quot;n_iterations&quot;: n_iterations,
            &quot;n_timesteps&quot;: n_timesteps,
            &quot;weight_decay&quot;: weight_decay,
            &quot;threshold&quot;: threshold,
            &quot;leak_rate&quot;: leak_rate,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;snn_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.npz&quot;: model_dir / &quot;training_data.npz&quot;,
        },
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;snn_image_classification&quot;, &quot;model_type&quot;: &quot;SNN&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;snn-image-classification&quot;,
            model_version=model_version,
            metrics=metrics,
            params={&quot;n_features&quot;: N_FEATURES, &quot;n_classes&quot;: N_CLASSES, &quot;hidden_dim&quot;: hidden_dim, &quot;learning_rate&quot;: learning_rate, &quot;n_iterations&quot;: n_iterations, &quot;n_timesteps&quot;: n_timesteps},
            artifacts={&quot;model&quot;: str(model_path)},
            tags={&quot;model_type&quot;: &quot;snn&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description=&quot;Train SNN Image Classification model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;500&quot;)))
    parser.add_argument(&quot;--hidden-dim&quot;, type=int, default=int(os.getenv(&quot;HIDDEN_DIM&quot;, &quot;128&quot;)))
    parser.add_argument(&quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.01&quot;)))
    parser.add_argument(&quot;--n-iterations&quot;, type=int, default=int(os.getenv(&quot;N_ITERATIONS&quot;, &quot;200&quot;)))
    parser.add_argument(&quot;--n-timesteps&quot;, type=int, default=int(os.getenv(&quot;N_TIMESTEPS&quot;, &quot;10&quot;)))
    parser.add_argument(&quot;--weight-decay&quot;, type=float, default=float(os.getenv(&quot;WEIGHT_DECAY&quot;, &quot;0.0001&quot;)))
    parser.add_argument(&quot;--threshold&quot;, type=float, default=float(os.getenv(&quot;THRESHOLD&quot;, &quot;1.0&quot;)))
    parser.add_argument(&quot;--leak-rate&quot;, type=float, default=float(os.getenv(&quot;LEAK_RATE&quot;, &quot;0.9&quot;)))
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
        hidden_dim=args.hidden_dim,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        n_timesteps=args.n_timesteps,
        weight_decay=args.weight_decay,
        threshold=args.threshold,
        leak_rate=args.leak_rate,
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
<button class="copy-btn" onclick="copyCode('code-2469762392')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2469762392"><code class="language-python">&quot;&quot;&quot;Serving API for SNN Image Classification.&quot;&quot;&quot;

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
from ai_core.validation import DataValidator, create_snn_image_classification_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from snn_image_classification.data import N_CLASSES, N_FEATURES, generate_synthetic_data
from snn_image_classification.model import SNNImageClassification

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;SNN_METRICS_PORT&quot;, &quot;8031&quot;))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class PredictRequest(BaseModel):
    features: list[float] = Field(..., min_length=N_FEATURES, max_length=N_FEATURES)


class PredictResponse(BaseModel):
    predicted_class: int
    confidence: float
    class_probabilities: list[float]
    total_spikes: float
    model_version: str
    training_mode: str


class DriftResponse(BaseModel):
    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


class StatsResponse(BaseModel):
    n_features: int
    n_classes: int
    hidden_dim: int
    training_mode: str
    n_timesteps: int
    threshold: float
    leak_rate: float
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: SNNImageClassification | None = None
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
    _metrics = MetricsCollector(&quot;snn_image_classification&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_snn_image_classification_schema())
    feature_names = [f&quot;pixel_{i}&quot; for i in range(N_FEATURES)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: &quot;float&quot; for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;snn-image-classification&quot;,
        model_version=_model_version,
        model_type=&quot;classification&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;snn-image-classification&quot;, version=_model_version)

    yield
    logger.info(&quot;Shutting down snn-image-classification API&quot;)


def _load_model() -&gt; tuple[SNNImageClassification, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            nn_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;snn-image-classification&quot;]
            if nn_models:
                nn_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;snn_model_*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return SNNImageClassification.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;snn-image-classification&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;snn_model_*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return SNNImageClassification.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    npz_path = MODEL_DIR / &quot;snn_model.npz&quot;
    if npz_path.exists():
        return SNNImageClassification.load(str(npz_path)), &quot;legacy&quot;

    candidate_paths = [
        Path(&quot;/app/artifacts/models/snn_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;snn_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return SNNImageClassification.load(str(p)), &quot;1.0.0-bundled&quot;

    logger.warning(&quot;No pre-existing model found. Initializing baseline model.&quot;)
    X_base, y_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = SNNImageClassification(
        n_features=N_FEATURES,
        n_classes=N_CLASSES,
        hidden_dim=64,
        learning_rate=0.01,
        n_iterations=50,
        n_timesteps=5,
        random_seed=42,
    )
    model.fit(X_base, y_base)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    return X_base


app = FastAPI(
    title=&quot;SNN Image Classification API&quot;,
    description=&quot;Spiking neural networks using neuromorphic computing with discrete spikes mimicking biological neurons&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;snn_image_classification-api&quot;,
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
                model_name=&quot;snn-image-classification&quot;,
                model_version=_model_version,
                model_type=&quot;classification&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(&quot;Model reloaded&quot;, model=&quot;snn-image-classification&quot;, version=_model_version)
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
    if _model is None or not _model.layers:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    return StatsResponse(
        n_features=_model.n_features,
        n_classes=_model.n_classes,
        hidden_dim=_model.hidden_dim,
        training_mode=_model.training_mode,
        n_timesteps=_model.n_timesteps,
        threshold=_model.threshold,
        leak_rate=_model.leak_rate,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )


@app.post(&quot;/predict&quot;, response_model=PredictResponse)
def predict(body: PredictRequest):
    &quot;&quot;&quot;Classify an image using spiking neural network.&quot;&quot;&quot;
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    X = np.array(body.features).reshape(1, -1)
    validation = _validator.validate(X)
    if not validation.valid:
        raise HTTPException(status_code=422, detail=validation.errors)

    start = time.time()
    try:
        probs = _model.predict_proba(X)[0]
        pred = int(np.argmax(probs))
        confidence = float(np.max(probs))

        total_spikes = float(np.sum(probs))

        response = PredictResponse(
            predicted_class=pred,
            confidence=round(confidence, 4),
            class_probabilities=probs.tolist(),
            total_spikes=round(total_spikes, 4),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append(body.features)
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Prediction failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Prediction failed&quot;) from e</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-4236985254')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-4236985254"><code class="language-bash">uv run python -m snn_image_classification.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>
<div class="related-links">
<h3>Related Apps</h3>
<ul><li><a href="../image-generation/README.md">image-generation</a></li>
<li><a href="../spam-classification/README.md">spam-classification</a></li>
<li><a href="../classification-email-spam/README.md">classification-email-spam</a></li>
<li><a href="../vision-image-captioning/README.md">vision-image-captioning</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>snn-image-classification</strong></p>
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