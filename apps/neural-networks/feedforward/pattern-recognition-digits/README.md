<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pattern-recognition-digits - AI App Documentation</title>
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
<p class="section-subtitle">Digit Recognition / Classification — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$Z = WX + b$$</div>
<div class="math-block">$$A = \text{ReLU}(Z)$$</div>
<div class="math-block">$$\mathcal{L}_{CE} = -\sum_{i=1}^{C} y_i \log(\hat{y}_i)$$</div>
<div class="math-block">$$\hat{y} = \text{softmax}(Z_{out})$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>Feedforward networks learn hierarchical feature representations. Each layer computes a linear transformation followed by a non-linearity. Cross-entropy loss penalizes misclassification. Backpropagation computes gradients via the chain rule.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive decision boundary; feature visualization for hidden layers; confusion matrix explorer.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  DigitRecognitionNN</pre>
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
<button class="copy-btn" onclick="copyCode('code-2545722179')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2545722179"><code class="language-python">&quot;&quot;&quot;Training pipeline for handwritten digit recognition using a feedforward neural network.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from pattern_recognition_digits.data import (
    save_training_data,
    train_test_split,
)
from pattern_recognition_digits.model import DigitRecognitionNN

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 1000,
    hidden_dim: int = 64,
    learning_rate: float = 0.1,
    n_iterations: int = 1000,
    weight_decay: float = 0.0001,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    noise_level: float = 0.3,
    test_size: float = 0.2,
    random_seed: int = 42,
) -&gt; dict:
    &quot;&quot;&quot;Train the digit recognition neural network and save artifacts.

    Uses softmax cross-entropy loss for multi-class classification of handwritten digits (0-9).
    &quot;&quot;&quot;
    X, y = load_training_data_fn(data_path, n_samples, noise_level, random_seed)
    logger.info(&quot;Loaded training data&quot;, n_samples=len(X), data_path=str(data_path))

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

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / &quot;training_data.csv&quot;)

    model = DigitRecognitionNN(
        hidden_dim=hidden_dim,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X_train, y_train, X_val=X_test, y_val=y_test)

    train_metrics = model.evaluate(X_train, y_train)
    test_metrics = model.evaluate(X_test, y_test)

    logger.info(
        &quot;Training complete&quot;,
        training_mode=model.training_mode,
        n_epochs=len(model.loss_history),
        final_loss=model.loss_history[-1] if model.loss_history else 0.0,
        train_accuracy=train_metrics[&quot;accuracy&quot;],
        test_accuracy=test_metrics[&quot;accuracy&quot;],
    )

    model_path = model_dir / f&quot;digit_recognition_model_v{model_version}.npz&quot;
    model.save(str(model_path))

    _save_chart(model, model_dir, model_version)

    metrics = {
        **test_metrics,
        &quot;training_mode&quot;: &quot;supervised&quot;,
        &quot;n_epochs_run&quot;: float(len(model.loss_history)),
        &quot;final_loss&quot;: model.loss_history[-1] if model.loss_history else 0.0,
        &quot;train_accuracy&quot;: train_metrics[&quot;accuracy&quot;],
        &quot;train_macro_f1&quot;: train_metrics[&quot;macro_f1&quot;],
        &quot;n_train_samples&quot;: float(len(X_train)),
        &quot;n_test_samples&quot;: float(len(X_test)),
        &quot;hidden_dim&quot;: float(hidden_dim),
        &quot;learning_rate&quot;: float(learning_rate),
        &quot;weight_decay&quot;: float(weight_decay),
        &quot;n_features&quot;: float(X_train.shape[1]),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;handwritten-digit-recognition&quot;,
        model_version=model_version,
        model_type=&quot;pattern_recognition&quot;,
        metrics=metrics,
        parameters={
            &quot;hidden_dim&quot;: hidden_dim,
            &quot;learning_rate&quot;: learning_rate,
            &quot;n_iterations&quot;: n_iterations,
            &quot;weight_decay&quot;: weight_decay,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;digit_recognition_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.csv&quot;: model_dir / &quot;training_data.csv&quot;,
        },
        tags={
            &quot;framework&quot;: &quot;numpy&quot;,
            &quot;task&quot;: &quot;pattern_recognition&quot;,
            &quot;model_type&quot;: &quot;feedforward_neural_network&quot;,
        },
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;handwritten-digit-recognition&quot;,
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
                &quot;chart&quot;: str(model_dir / f&quot;handwritten_digit_recognition_v{model_version}.png&quot;),
                &quot;training_data&quot;: str(model_dir / &quot;training_data.csv&quot;),
            },
            tags={&quot;model_type&quot;: &quot;pattern_recognition&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )
        logger.info(
            &quot;Registered model to MLflow&quot;,
            model=&quot;handwritten-digit-recognition&quot;,
            version=model_version,
        )

    return metrics


def load_training_data_fn(data_path, n_samples, noise_level, random_seed):
    &quot;&quot;&quot;Wrapper to avoid circular import.&quot;&quot;&quot;
    from pattern_recognition_digits.data import load_training_data

    return load_training_data(data_path, n_samples, noise_level, random_seed)


def _save_chart(model: DigitRecognitionNN, output_dir: Path, version: str) -&gt; None:
    &quot;&quot;&quot;Save the training loss chart.&quot;&quot;&quot;
    import matplotlib

    matplotlib.use(&quot;Agg&quot;)
    import matplotlib.pyplot as plt

    if not model.loss_history:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(model.loss_history, color=&quot;steelblue&quot;, linewidth=1.5, label=&quot;Training Loss&quot;)
    if model.val_accuracy_history:
        ax2 = ax.twinx()
        ax2.plot(
            range(0, len(model.val_accuracy_history) * 50, 50),
            model.val_accuracy_history,
            color=&quot;coral&quot;,
            linewidth=1.5,
            label=&quot;Validation Accuracy&quot;,
        )
        ax2.set_ylabel(&quot;Validation Accuracy&quot;, color=&quot;coral&quot;)
        ax2.legend(loc=&quot;center right&quot;)
    ax.set_xlabel(&quot;Training Iteration&quot;)
    ax.set_ylabel(&quot;Loss (Cross-Entropy + L2)&quot;)
    ax.set_title(&quot;Handwritten Digit Recognition NN Training Loss&quot;)
    ax.grid(True, alpha=0.3)
    ax.set_yscale(&quot;log&quot;)

    plt.tight_layout()
    chart_path = output_dir / f&quot;handwritten_digit_recognition_v{version}.png&quot;
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info(&quot;Chart saved&quot;, path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(
        description=&quot;Train handwritten digit recognition neural network&quot;
    )
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;1000&quot;)))
    parser.add_argument(&quot;--hidden-dim&quot;, type=int, default=int(os.getenv(&quot;HIDDEN_DIM&quot;, &quot;64&quot;)))
    parser.add_argument(
        &quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.1&quot;))
    )
    parser.add_argument(&quot;--n-iterations&quot;, type=int, default=int(os.getenv(&quot;N_ITERATIONS&quot;, &quot;1000&quot;)))
    parser.add_argument(
        &quot;--weight-decay&quot;, type=float, default=float(os.getenv(&quot;WEIGHT_DECAY&quot;, &quot;0.0001&quot;))
    )
    parser.add_argument(&quot;--noise-level&quot;, type=float, default=float(os.getenv(&quot;NOISE_LEVEL&quot;, &quot;0.3&quot;)))
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
        noise_level=args.noise_level,
        test_size=args.test_size,
        random_seed=args.random_seed,
    )

    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-2467894986')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2467894986"><code class="language-python">&quot;&quot;&quot;Production serving API for handwritten digit recognition via feedforward neural network.&quot;&quot;&quot;

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

from pattern_recognition_digits.data import FEATURE_NAMES, N_CLASSES
from pattern_recognition_digits.model import DigitRecognitionNN

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;METRICS_PORT&quot;, os.getenv(&quot;DIGIT_RECOGNITION_METRICS_PORT&quot;, &quot;8011&quot;)))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class PredictRequest(BaseModel):
    &quot;&quot;&quot;Handwritten digit recognition request.&quot;&quot;&quot;

    pixels: list[float] = Field(
        ..., min_length=64, max_length=64, description=&quot;8x8=64 pixel values (0-1)&quot;
    )


class PredictBulkRequest(BaseModel):
    &quot;&quot;&quot;Bulk digit recognition request.&quot;&quot;&quot;

    requests: list[list[float]] = Field(..., min_length=1, max_length=50)


class PredictResponse(BaseModel):
    &quot;&quot;&quot;Digit recognition prediction response.&quot;&quot;&quot;

    digit: int
    confidence: float
    probabilities: dict[str, float]
    model_version: str
    training_mode: str


class BulkPredictResponse(BaseModel):
    &quot;&quot;&quot;Bulk digit recognition prediction response.&quot;&quot;&quot;

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
    n_classes: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: DigitRecognitionNN | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;handwritten_digit_recognition&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _drift_detector = DriftDetector(
        feature_names=FEATURE_NAMES,
        feature_types={f: &quot;float&quot; for f in FEATURE_NAMES},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;handwritten-digit-recognition&quot;,
        model_version=_model_version,
        model_type=&quot;pattern_recognition&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;handwritten-digit-recognition&quot;, version=_model_version)

    yield

    logger.info(&quot;Shutting down handwritten-digit-recognition API&quot;)


def _load_model() -&gt; tuple[DigitRecognitionNN, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            nn_models = [
                m for m in models if m.get(&quot;model_name&quot;) == &quot;handwritten-digit-recognition&quot;
            ]
            if nn_models:
                nn_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = nn_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;digit_recognition_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return DigitRecognitionNN.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;handwritten-digit-recognition&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;digit_recognition_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return DigitRecognitionNN.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    npz_path = MODEL_DIR / &quot;digit_recognition_model.npz&quot;
    if npz_path.exists():
        return DigitRecognitionNN.load(str(npz_path)), &quot;legacy&quot;

    candidate_paths = [
        Path(&quot;/app/artifacts/models/digit_recognition_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3]
        / &quot;artifacts&quot;
        / &quot;models&quot;
        / &quot;digit_recognition_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return DigitRecognitionNN.load(str(p)), &quot;1.0.0-bundled&quot;

    logger.warning(&quot;No pre-existing model found on disk. Initializing baseline model.&quot;)
    from pattern_recognition_digits.data import generate_synthetic_data

    X_base, y_base = generate_synthetic_data(n_samples=500, random_seed=42)
    model = DigitRecognitionNN(hidden_dim=64, learning_rate=0.1, n_iterations=500, random_seed=42)
    model.fit(X_base, y_base)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    candidate_csvs = [
        MODEL_DIR / &quot;handwritten-digit-recognition&quot; / _model_version / &quot;training_data.csv&quot;,
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

    from pattern_recognition_digits.data import generate_synthetic_data

    X_base, _ = generate_synthetic_data(n_samples=500, random_seed=42)
    return X_base


app = FastAPI(
    title=&quot;Handwritten Digit Recognition API&quot;,
    description=&quot;Feedforward neural network for recognizing handwritten digits (0-9) from 8x8 pixel images&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;handwritten-digit-recognition-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;training_mode&quot;: _model.training_mode if _model else &quot;unknown&quot;,
        &quot;n_classes&quot;: N_CLASSES,
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
                model_name=&quot;handwritten-digit-recognition&quot;,
                model_version=_model_version,
                model_type=&quot;pattern_recognition&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(
            &quot;Model reloaded dynamically&quot;,
            model=&quot;handwritten-digit-recognition&quot;,
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
        n_classes=_model.n_classes,
        training_mode=_model.training_mode,
        n_epochs_run=len(_model.loss_history),
        final_loss=_model.loss_history[-1] if _model.loss_history else 0.0,
        model_version=_model_version,
    )


def _compute_digit(pixels: list[float]) -&gt; PredictResponse:
    if _model is None or _metrics is None or _drift_detector is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    X = np.array([pixels])

    start = time.time()
    try:
        probs = _model.predict_proba(X)[0]
        digit = int(np.argmax(probs))
        confidence = float(np.max(probs))
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append(pixels)
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        prob_dict = {str(i): round(float(probs[i]), 4) for i in range(N_CLASSES)}

        return PredictResponse(
            digit=digit,
            confidence=round(confidence, 4),
            probabilities=prob_dict,
            model_version=_model_version,
            training_mode=_model.training_mode if _model else &quot;unknown&quot;,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Prediction failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Prediction failed&quot;) from e


@app.post(&quot;/predict&quot;, response_model=PredictResponse)
def predict_digit(body: PredictRequest):
    &quot;&quot;&quot;Recognize a single handwritten digit.&quot;&quot;&quot;
    return _compute_digit(body.pixels)


@app.post(&quot;/predict/bulk&quot;, response_model=BulkPredictResponse)
def predict_digit_bulk(body: PredictBulkRequest):
    &quot;&quot;&quot;Recognize multiple handwritten digits.&quot;&quot;&quot;
    global _recent_predictions
    if _model is None or _metrics is None or _drift_detector is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    if len(body.requests) &lt; 1 or len(body.requests) &gt; 50:
        raise HTTPException(status_code=422, detail=&quot;Batch size must be between 1 and 50&quot;)

    X = np.array(body.requests)

    start = time.time()
    try:
        all_probs = _model.predict_proba(X)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.extend(body.requests)
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions = _recent_predictions[-1000:]

        predictions = []
        for probs in all_probs:
            digit = int(np.argmax(probs))
            confidence = float(np.max(probs))
            prob_dict = {str(i): round(float(probs[i]), 4) for i in range(N_CLASSES)}
            predictions.append(
                PredictResponse(
                    digit=digit,
                    confidence=round(confidence, 4),
                    probabilities=prob_dict,
                    model_version=_model_version,
                    training_mode=_model.training_mode if _model else &quot;unknown&quot;,
                )
            )

        return BulkPredictResponse(predictions=predictions, model_version=_model_version)
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Bulk prediction failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Bulk prediction failed&quot;) from e</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-1500908013')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-1500908013"><code class="language-bash">uv run python -m pattern_recognition_digits.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>
<div class="related-links">
<h3>Related Apps</h3>
<ul><li><a href="../capsnet-text-recognition/README.md">capsnet-text-recognition</a></li>
<li><a href="../cnn-facial-recognition/README.md">cnn-facial-recognition</a></li>
<li><a href="../speech-audio-recognition/README.md">speech-audio-recognition</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>pattern-recognition-digits</strong></p>
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