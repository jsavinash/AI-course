<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>diffusion - AI App Documentation</title>
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
<p class="section-subtitle">Denoising Diffusion Probabilistic Model (DDPM) — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$q(x_t | x_{t-1}) = \mathcal{N}(x_t; \sqrt{1 - \beta_t}x_{t-1}, \beta_t I)$$</div>
<div class="math-block">$$p_\theta(x_{t-1} | x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \Sigma_\theta(x_t, t))$$</div>
<div class="math-block">$$\mathcal{L}_{simple} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t}x_0 + \sqrt{1-\bar{\alpha}_t}\epsilon, t) \|^2 \right]$$</div>
<div class="math-block">$$\bar{\alpha}_t = \prod_{s=1}^{t} (1 - \beta_s)$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>DDPM gradually corrupts data with Gaussian noise over $T$ steps. The model learns to reverse this process by predicting the noise $\epsilon$ at each step. Training minimizes the MSE between actual and predicted noise. Sampling iteratively denoises from pure Gaussian noise.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive forward/reverse process visualization; denoising trajectory viewer; noise schedule plot.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  DiffusionModel</pre>
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
<button class="copy-btn" onclick="copyCode('code-1429790945')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-1429790945"><code class="language-python">&quot;&quot;&quot;Training pipeline for Diffusion Image Generation.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_diffusion_image_generation_schema

from diffusion_image_generation.data import (
    IMAGE_SIZE,
    N_CHANNELS,
    load_training_data,
    save_training_data,
    train_test_split,
)
from diffusion_image_generation.model import DiffusionModel

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    n_filters: int = 8,
    kernel_size: int = 3,
    hidden_dim: int = 32,
    timesteps: int = 100,
    beta_start: float = 0.0001,
    beta_end: float = 0.02,
    learning_rate: float = 0.01,
    n_iterations: int = 200,
    weight_decay: float = 0.0001,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
) -&gt; dict:
    &quot;&quot;&quot;Train the diffusion model and save artifacts.&quot;&quot;&quot;
    X, y = load_training_data(data_path, n_samples=n_samples, random_seed=random_seed)
    logger.info(&quot;Loaded training data&quot;, n_samples=len(X), data_path=str(data_path))

    validator = DataValidator(create_diffusion_image_generation_schema())
    validation = validator.validate(X.reshape(-1, 1))
    if not validation.valid:
        logger.error(&quot;Training data validation failed&quot;, errors=validation.errors)
        raise ValueError(f&quot;Training data validation failed: {validation.errors}&quot;)
    logger.info(&quot;Training data validated&quot;, stats=validation.stats)

    X_train, X_test, _, _ = train_test_split(
        X, y, test_size=test_size, random_seed=random_seed
    )
    logger.info(&quot;Data split&quot;, n_train=len(X_train), n_test=len(X_test), test_size=test_size)

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / &quot;training_data.npz&quot;)

    model = DiffusionModel(
        img_size=IMAGE_SIZE,
        n_channels=N_CHANNELS,
        n_filters=n_filters,
        kernel_size=kernel_size,
        hidden_dim=hidden_dim,
        timesteps=timesteps,
        beta_start=beta_start,
        beta_end=beta_end,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X_train)

    model.evaluate(X_train)
    test_metrics = model.evaluate(X_test)

    logger.info(
        &quot;Training complete&quot;,
        training_mode=model.training_mode,
        n_epochs=len(model.loss_history),
        final_loss=model.loss_history[-1] if model.loss_history else 0.0,
        test_metrics=test_metrics,
    )

    model_path = model_dir / f&quot;diffusion_image_generation_model_v{model_version}.npz&quot;
    model.save(str(model_path))

    _save_chart(model, model_dir, model_version)

    metrics = {
        **test_metrics,
        &quot;training_mode&quot;: &quot;self-supervised&quot;,
        &quot;n_epochs_run&quot;: float(len(model.loss_history)),
        &quot;final_loss&quot;: model.loss_history[-1] if model.loss_history else 0.0,
        &quot;n_train_samples&quot;: float(len(X_train)),
        &quot;n_test_samples&quot;: float(len(X_test)),
        &quot;n_filters&quot;: float(n_filters),
        &quot;learning_rate&quot;: float(learning_rate),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;diffusion-image-generation&quot;,
        model_version=model_version,
        model_type=&quot;generative&quot;,
        metrics=metrics,
        parameters={
            &quot;img_size&quot;: IMAGE_SIZE,
            &quot;n_channels&quot;: N_CHANNELS,
            &quot;n_filters&quot;: n_filters,
            &quot;kernel_size&quot;: kernel_size,
            &quot;hidden_dim&quot;: hidden_dim,
            &quot;timesteps&quot;: timesteps,
            &quot;beta_start&quot;: beta_start,
            &quot;beta_end&quot;: beta_end,
            &quot;learning_rate&quot;: learning_rate,
            &quot;n_iterations&quot;: n_iterations,
            &quot;weight_decay&quot;: weight_decay,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;diffusion_image_generation_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.npz&quot;: model_dir / &quot;training_data.npz&quot;,
        },
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;diffusion_image_generation&quot;, &quot;model_type&quot;: &quot;Diffusion&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;diffusion-image-generation&quot;,
            model_version=model_version,
            metrics=metrics,
            params={
                &quot;img_size&quot;: IMAGE_SIZE,
                &quot;n_filters&quot;: n_filters,
                &quot;timesteps&quot;: timesteps,
                &quot;learning_rate&quot;: learning_rate,
                &quot;n_iterations&quot;: n_iterations,
            },
            artifacts={
                &quot;model&quot;: str(model_path),
                &quot;chart&quot;: str(model_dir / f&quot;diffusion_image_generation_v{model_version}.png&quot;),
            },
            tags={&quot;model_type&quot;: &quot;diffusion_image_generation&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )
        logger.info(&quot;Registered model to MLflow&quot;, model=&quot;diffusion-image-generation&quot;, version=model_version)

    return metrics


def _save_chart(model, output_dir: Path, version: str) -&gt; None:
    import matplotlib

    matplotlib.use(&quot;Agg&quot;)
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color=&quot;steelblue&quot;, linewidth=1.5)
    ax.set_xlabel(&quot;Training Iteration&quot;)
    ax.set_ylabel(&quot;Loss&quot;)
    ax.set_title(&quot;Diffusion Image Generation Training Loss&quot;)
    ax.grid(True, alpha=0.3)
    ax.set_yscale(&quot;log&quot;)
    plt.tight_layout()
    chart_path = output_dir / f&quot;diffusion_image_generation_v{version}.png&quot;
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info(&quot;Chart saved&quot;, path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description=&quot;Train Diffusion Image Generation model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;500&quot;)))
    parser.add_argument(&quot;--n-filters&quot;, type=int, default=int(os.getenv(&quot;N_FILTERS&quot;, &quot;8&quot;)))
    parser.add_argument(&quot;--kernel-size&quot;, type=int, default=int(os.getenv(&quot;KERNEL_SIZE&quot;, &quot;3&quot;)))
    parser.add_argument(&quot;--hidden-dim&quot;, type=int, default=int(os.getenv(&quot;HIDDEN_DIM&quot;, &quot;32&quot;)))
    parser.add_argument(&quot;--timesteps&quot;, type=int, default=int(os.getenv(&quot;TIMESTEPS&quot;, &quot;100&quot;)))
    parser.add_argument(&quot;--beta-start&quot;, type=float, default=float(os.getenv(&quot;BETA_START&quot;, &quot;0.0001&quot;)))
    parser.add_argument(&quot;--beta-end&quot;, type=float, default=float(os.getenv(&quot;BETA_END&quot;, &quot;0.02&quot;)))
    parser.add_argument(&quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.01&quot;)))
    parser.add_argument(&quot;--n-iterations&quot;, type=int, default=int(os.getenv(&quot;N_ITERATIONS&quot;, &quot;200&quot;)))
    parser.add_argument(&quot;--weight-decay&quot;, type=float, default=float(os.getenv(&quot;WEIGHT_DECAY&quot;, &quot;0.0001&quot;)))
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
        n_filters=args.n_filters,
        kernel_size=args.kernel_size,
        hidden_dim=args.hidden_dim,
        timesteps=args.timesteps,
        beta_start=args.beta_start,
        beta_end=args.beta_end,
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
<button class="copy-btn" onclick="copyCode('code-1097237156')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-1097237156"><code class="language-python">&quot;&quot;&quot;Serving API for Diffusion Image Generation.&quot;&quot;&quot;

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
from ai_core.validation import DataValidator, create_diffusion_image_generation_schema
from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, Field

from diffusion_image_generation.data import (
    IMAGE_SIZE,
    N_CHANNELS,
    N_FEATURES,
    generate_synthetic_data,
)
from diffusion_image_generation.model import DiffusionModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;DIFFUSION_IMAGE_GENERATION_METRICS_PORT&quot;, &quot;8024&quot;))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class PredictRequest(BaseModel):
    timesteps_to_run: int = Field(default=100, ge=1, le=1000)


class PredictBulkRequest(BaseModel):
    requests: list[dict] = Field(..., min_length=1, max_length=50)


class PredictResponse(BaseModel):
    generated_pixels: list[float]
    confidence: float
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
    img_size: int
    n_channels: int
    n_filters: int
    timesteps: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: DiffusionModel | None = None
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
    _metrics = MetricsCollector(&quot;diffusion_image_generation&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _validator = DataValidator(create_diffusion_image_generation_schema())
    feature_names = [f&quot;pixel_{i}&quot; for i in range(N_FEATURES)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: &quot;float&quot; for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;diffusion-image-generation&quot;,
        model_version=_model_version,
        model_type=&quot;generative&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;diffusion-image-generation&quot;, version=_model_version)

    yield
    logger.info(&quot;Shutting down diffusion-image-generation API&quot;)


def _load_model() -&gt; tuple[DiffusionModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            nn_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;diffusion-image-generation&quot;]
            if nn_models:
                nn_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;diffusion_image_generation_model_*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return DiffusionModel.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;diffusion-image-generation&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;diffusion_image_generation_model_*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return DiffusionModel.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    npz_path = MODEL_DIR / &quot;diffusion_image_generation_model.npz&quot;
    if npz_path.exists():
        return DiffusionModel.load(str(npz_path)), &quot;legacy&quot;

    candidate_paths = [
        Path(&quot;/app/artifacts/models/diffusion_image_generation_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;diffusion_image_generation_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return DiffusionModel.load(str(p)), &quot;1.0.0-bundled&quot;

    logger.warning(&quot;No pre-existing model found. Initializing baseline model.&quot;)
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    model = DiffusionModel(
        img_size=IMAGE_SIZE,
        n_channels=N_CHANNELS,
        n_filters=8,
        kernel_size=3,
        hidden_dim=32,
        timesteps=100,
        learning_rate=0.01,
        n_iterations=100,
        random_seed=42,
    )
    model.fit(X_base)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    return X_base


app = FastAPI(
    title=&quot;Diffusion Image Generation API&quot;,
    description=&quot;Generates images by systematically removing noise from a random starting state&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;diffusion_image_generation-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;training_mode&quot;: _model.training_mode if _model else &quot;unknown&quot;,
        &quot;n_features&quot;: N_FEATURES,
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
                model_name=&quot;diffusion-image-generation&quot;,
                model_version=_model_version,
                model_type=&quot;generative&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(&quot;Model reloaded dynamically&quot;, model=&quot;diffusion-image-generation&quot;, version=_model_version)
        return {&quot;status&quot;: &quot;reloaded&quot;, &quot;model_version&quot;: _model_version}
    except Exception as e:
        logger.exception(&quot;Model reload failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=f&quot;Reload failed: {e}&quot;) from e


@app.get(&quot;/drift&quot;, response_model=DriftResponse)
def drift_check():
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail=&quot;Drift detection not available&quot;)
    if len(_recent_predictions) &lt; 10:
        return {
            &quot;total_features&quot;: N_FEATURES,
            &quot;drifted_features&quot;: 0,
            &quot;drift_ratio&quot;: 0.0,
            &quot;drifted&quot;: [],
            &quot;all_results&quot;: [],
        }
    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)
    if _metrics:
        _metrics.set_drift_ratio(summary[&quot;drift_ratio&quot;])
    return summary


@app.get(&quot;/stats&quot;, response_model=StatsResponse)
def get_stats():
    if _model is None or _model.betas is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    return StatsResponse(
        img_size=_model.img_size,
        n_channels=_model.n_channels,
        n_filters=_model.n_filters,
        timesteps=_model.timesteps,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )


def _compute_prediction(request_body: dict):
    if _model is None or _metrics is None or _validator is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    request_body.get(&quot;timesteps_to_run&quot;, 100)
    start = time.time()
    try:
        pixels = _model.generate(n_samples=1, random_seed=42)[0]
        response = PredictResponse(
            generated_pixels=pixels.tolist(),
            confidence=round(float(np.max(np.abs(pixels))), 4),
            model_version=_model_version,
            training_mode=_model.training_mode,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append(pixels.tolist())
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Prediction failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Prediction failed&quot;) from e


@app.post(&quot;/predict&quot;, response_model=PredictResponse)
def predict(body: PredictRequest):
    &quot;&quot;&quot;Generate an image by iteratively denoising random noise.&quot;&quot;&quot;
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    return _compute_prediction({&quot;timesteps_to_run&quot;: body.timesteps_to_run})


@app.post(&quot;/predict/bulk&quot;, response_model=BulkPredictResponse)
def predict_bulk(body: PredictBulkRequest):
    &quot;&quot;&quot;Generate multiple images.&quot;&quot;&quot;
    global _recent_predictions
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    if len(body.requests) &lt; 1 or len(body.requests) &gt; 50:
        raise HTTPException(status_code=422, detail=&quot;Batch size must be between 1 and 50&quot;)

    predictions = []
    for req in body.requests:
        predictions.append(_compute_prediction(req))

    return BulkPredictResponse(predictions=predictions, model_version=_model_version)</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-2422502322')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2422502322"><code class="language-bash">uv run python -m diffusion.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>

</main>
<footer class="app-footer">
<p>Generated documentation for <strong>diffusion</strong></p>
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