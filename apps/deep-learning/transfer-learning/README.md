<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>transfer-learning - AI App Documentation</title>
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
<p class="section-subtitle">Transfer Learning — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$\mathcal{L} = \mathcal{L}_{task} + \lambda \mathcal{L}_{distill}$$</div>
<div class="math-block">$$\mathcal{L}_{distill} = \text{KL}(p_{\text{teacher}} \| p_{\text{student}})$$</div>
<div class="math-block">$$p_i = \frac{\exp(z_i / T)}{\sum_j \exp(z_j / T)}$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>Transfer learning reuses features from a source domain for a target task. Fine-tuning updates only the final layers to adapt to new data. Knowledge distillation transfers dark knowledge from a large teacher to a compact student via softened probabilities.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive feature reuse heatmap; layer freezing/unfreezing timeline; teacher vs student probability comparison.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  DenseLayer
  MultiHeadAttention
  AddNorm
  FeedForward
  TransformerBlock
  BaseModel
  TransferClassifier
  TransferLearningModel</pre>
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
<button class="copy-btn" onclick="copyCode('code-4258450362')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-4258450362"><code class="language-python">&quot;&quot;&quot;Training pipeline for Transfer Learning.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry
from ai_core.validation import DataValidator, create_transfer_learning_schema

from transfer_learning.data import (
    generate_synthetic_data,
    load_dataset,
    save_dataset,
    train_test_split,
)
from transfer_learning.model import TransferLearningModel

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    vocab_size: int = 1000,
    seq_len: int = 32,
    d_model: int = 128,
    n_heads: int = 4,
    n_base_layers: int = 2,
    d_ff: int = 512,
    max_seq_len: int = 32,
    n_classes: int = 10,
    freeze_base: bool = True,
    fine_tune_layers: int = 0,
    learning_rate: float = 0.001,
    fine_tune_lr: float = 0.0001,
    n_iterations: int = 100,
    weight_decay: float = 0.01,
    model_version: str = &quot;1.0.0&quot;,
    fine_tune_at: int | None = None,
    random_seed: int = 42,
    register_to_mlflow: bool = False,
) -&gt; dict:
    logger.info(&quot;Loading data&quot;, n_samples=n_samples)
    if data_path and Path(data_path).exists():
        X, y = load_dataset(data_path)
    else:
        X, y = generate_synthetic_data(n_samples=n_samples, vocab_size=vocab_size, seq_len=seq_len, n_classes=n_classes, random_seed=random_seed)

    validator = DataValidator(create_transfer_learning_schema())
    validation = validator.validate(X.reshape(-1, X.shape[-1]))
    if not validation.valid:
        logger.error(&quot;Data validation failed&quot;, errors=validation.errors)
        raise ValueError(f&quot;Data validation failed: {validation.errors}&quot;)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_seed=random_seed)
    logger.info(&quot;Data split&quot;, n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(X, y, model_dir / &quot;training_data.npz&quot;)

    model = TransferLearningModel(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_base_layers=n_base_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        n_classes=n_classes,
        freeze_base=freeze_base,
        fine_tune_layers=fine_tune_layers,
        learning_rate=learning_rate,
        fine_tune_lr=fine_tune_lr,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )

    logger.info(&quot;Starting transfer learning training&quot;, freeze_base=freeze_base, fine_tune_layers=fine_tune_layers)
    model.fit(X_train, y_train, n_iterations=n_iterations, fine_tune_at=fine_tune_at)

    test_metrics = model.evaluate(X_test, y_test)
    logger.info(&quot;Training complete&quot;, final_loss=model.loss_history[-1], test_accuracy=test_metrics[&quot;accuracy&quot;])

    model_path = model_dir / f&quot;transfer_learning_v{model_version}.npz&quot;
    model.save(str(model_path))

    metrics = {
        **test_metrics,
        &quot;n_epochs_run&quot;: float(len(model.loss_history)),
        &quot;final_loss&quot;: model.loss_history[-1] if model.loss_history else 0.0,
        &quot;n_train_samples&quot;: float(len(X_train)),
        &quot;n_test_samples&quot;: float(len(X_test)),
        &quot;vocab_size&quot;: float(vocab_size),
        &quot;d_model&quot;: float(d_model),
        &quot;n_base_layers&quot;: float(n_base_layers),
        &quot;d_ff&quot;: float(d_ff),
        &quot;n_classes&quot;: float(n_classes),
        &quot;freeze_base&quot;: 1.0 if freeze_base else 0.0,
        &quot;fine_tune_layers&quot;: float(fine_tune_layers),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;transfer-learning&quot;,
        model_version=model_version,
        model_type=&quot;classification&quot;,
        metrics=metrics,
        parameters={
            &quot;vocab_size&quot;: vocab_size,
            &quot;d_model&quot;: d_model,
            &quot;n_heads&quot;: n_heads,
            &quot;n_base_layers&quot;: n_base_layers,
            &quot;d_ff&quot;: d_ff,
            &quot;max_seq_len&quot;: max_seq_len,
            &quot;n_classes&quot;: n_classes,
            &quot;freeze_base&quot;: freeze_base,
            &quot;fine_tune_layers&quot;: fine_tune_layers,
            &quot;learning_rate&quot;: learning_rate,
            &quot;fine_tune_lr&quot;: fine_tune_lr,
            &quot;n_iterations&quot;: n_iterations,
            &quot;weight_decay&quot;: weight_decay,
            &quot;random_seed&quot;: random_seed,
            &quot;fine_tune_at&quot;: fine_tune_at,
        },
        artifacts={
            f&quot;transfer_learning_v{model_version}.npz&quot;: model_path,
            &quot;training_data.npz&quot;: model_dir / &quot;training_data.npz&quot;,
        },
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;transfer_learning&quot;, &quot;model_type&quot;: &quot;TransferLearning&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;transfer-learning&quot;,
            model_version=model_version,
            metrics=metrics,
            params={&quot;vocab_size&quot;: vocab_size, &quot;d_model&quot;: d_model, &quot;freeze_base&quot;: freeze_base, &quot;fine_tune_layers&quot;: fine_tune_layers},
            artifacts={&quot;model&quot;: str(model_path)},
            tags={&quot;model_type&quot;: &quot;transfer_learning&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description=&quot;Train Transfer Learning Model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;500&quot;)))
    parser.add_argument(&quot;--vocab-size&quot;, type=int, default=int(os.getenv(&quot;VOCAB_SIZE&quot;, str(1000))))
    parser.add_argument(&quot;--seq-len&quot;, type=int, default=int(os.getenv(&quot;SEQ_LEN&quot;, &quot;32&quot;)))
    parser.add_argument(&quot;--d-model&quot;, type=int, default=int(os.getenv(&quot;D_MODEL&quot;, &quot;128&quot;)))
    parser.add_argument(&quot;--n-heads&quot;, type=int, default=int(os.getenv(&quot;N_HEADS&quot;, &quot;4&quot;)))
    parser.add_argument(&quot;--n-base-layers&quot;, type=int, default=int(os.getenv(&quot;N_BASE_LAYERS&quot;, &quot;2&quot;)))
    parser.add_argument(&quot;--d-ff&quot;, type=int, default=int(os.getenv(&quot;D_FF&quot;, &quot;512&quot;)))
    parser.add_argument(&quot;--max-seq-len&quot;, type=int, default=int(os.getenv(&quot;MAX_SEQ_LEN&quot;, &quot;32&quot;)))
    parser.add_argument(&quot;--n-classes&quot;, type=int, default=int(os.getenv(&quot;N_CLASSES&quot;, &quot;10&quot;)))
    parser.add_argument(&quot;--freeze-base&quot;, action=&quot;store_true&quot;, default=os.getenv(&quot;FREEZE_BASE&quot;, &quot;true&quot;).lower() == &quot;true&quot;)
    parser.add_argument(&quot;--no-freeze-base&quot;, dest=&quot;freeze_base&quot;, action=&quot;store_false&quot;)
    parser.add_argument(&quot;--fine-tune-layers&quot;, type=int, default=int(os.getenv(&quot;FINE_TUNE_LAYERS&quot;, &quot;0&quot;)))
    parser.add_argument(&quot;--fine-tune-at&quot;, type=int, default=None)
    parser.add_argument(&quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.001&quot;)))
    parser.add_argument(&quot;--fine-tune-lr&quot;, type=float, default=float(os.getenv(&quot;FINE_TUNE_LR&quot;, &quot;0.0001&quot;)))
    parser.add_argument(&quot;--n-iterations&quot;, type=int, default=int(os.getenv(&quot;N_ITERATIONS&quot;, &quot;100&quot;)))
    parser.add_argument(&quot;--weight-decay&quot;, type=float, default=float(os.getenv(&quot;WEIGHT_DECAY&quot;, &quot;0.01&quot;)))
    parser.add_argument(&quot;--model-version&quot;, type=str, default=os.getenv(&quot;MODEL_VERSION&quot;, &quot;1.0.0&quot;))
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
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_base_layers=args.n_base_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        n_classes=args.n_classes,
        freeze_base=args.freeze_base,
        fine_tune_layers=args.fine_tune_layers,
        learning_rate=args.learning_rate,
        fine_tune_lr=args.fine_tune_lr,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        fine_tune_at=args.fine_tune_at,
        random_seed=args.random_seed,
        register_to_mlflow=args.register_mlflow,
    )
    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-1134446121')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-1134446121"><code class="language-python">&quot;&quot;&quot;Serving API for Transfer Learning.&quot;&quot;&quot;

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

from transfer_learning.data import VOCAB_SIZE
from transfer_learning.model import TransferLearningModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;TRANSFER_LEARNING_METRICS_PORT&quot;, &quot;8013&quot;))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class PredictRequest(BaseModel):
    tokens: list[int] = Field(..., min_length=1, max_length=32)
    max_len: int = Field(default=1, ge=1, le=1)


class PredictResponse(BaseModel):
    predicted_class: int
    confidence: float
    class_probabilities: list[float]
    model_version: str
    base_model_frozen: bool
    fine_tune_layers: int


class DriftResponse(BaseModel):
    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


class StatsResponse(BaseModel):
    vocab_size: int
    d_model: int
    n_base_layers: int
    d_ff: int
    n_classes: int
    freeze_base: bool
    fine_tune_layers: int
    learning_rate: float
    fine_tune_lr: float
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: TransferLearningModel | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;transfer_learning&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    feature_names = [f&quot;token_{i}&quot; for i in range(VOCAB_SIZE)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: &quot;float&quot; for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;transfer-learning&quot;,
        model_version=_model_version,
        model_type=&quot;classification&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;transfer-learning&quot;, version=_model_version)

    yield
    logger.info(&quot;Shutting down transfer-learning API&quot;)


def _load_model() -&gt; tuple[TransferLearningModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            tl_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;transfer-learning&quot;]
            if tl_models:
                tl_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = tl_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;transfer_learning_v*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return TransferLearningModel.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;transfer-learning&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;transfer_learning_v*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return TransferLearningModel.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    npz_path = MODEL_DIR / &quot;transfer_learning.npz&quot;
    if npz_path.exists():
        return TransferLearningModel.load(str(npz_path)), &quot;legacy&quot;

    candidate_paths = [
        Path(&quot;/app/artifacts/models/transfer_learning_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;transfer_learning_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return TransferLearningModel.load(str(p)), &quot;1.0.0-bundled&quot;

    logger.warning(&quot;No pre-existing model found. Initializing baseline model.&quot;)
    from transfer_learning.data import generate_synthetic_data
    X_base, y_base = generate_synthetic_data(n_samples=100, random_seed=42)
    model = TransferLearningModel(
        vocab_size=100,
        d_model=64,
        n_heads=4,
        n_base_layers=1,
        d_ff=256,
        max_seq_len=32,
        n_classes=10,
        freeze_base=True,
        fine_tune_layers=0,
        learning_rate=0.001,
        fine_tune_lr=0.0001,
        n_iterations=10,
        random_seed=42,
    )
    model.fit(X_base, y_base, n_iterations=10)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    from transfer_learning.data import generate_synthetic_data
    X_base, _ = generate_synthetic_data(n_samples=100, random_seed=42)
    return X_base.astype(float)


app = FastAPI(
    title=&quot;Transfer Learning API&quot;,
    description=&quot;Transfer Learning model with frozen base model and trainable classification head, supporting fine-tuning of top layers&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;transfer_learning-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;base_model_frozen&quot;: _model.base_model.frozen if _model and _model.base_model else True,
        &quot;fine_tune_layers&quot;: _model.fine_tune_layers if _model else 0,
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
        &quot;base_model_frozen&quot;: _model.base_model.frozen if _model and _model.base_model else True,
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
                model_name=&quot;transfer-learning&quot;,
                model_version=_model_version,
                model_type=&quot;classification&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(&quot;Model reloaded&quot;, model=&quot;transfer-learning&quot;, version=_model_version)
        return {&quot;status&quot;: &quot;reloaded&quot;, &quot;model_version&quot;: _model_version}
    except Exception as e:
        logger.exception(&quot;Model reload failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=f&quot;Reload failed: {e}&quot;) from e


@app.get(&quot;/drift&quot;, response_model=DriftResponse)
def drift_check():
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail=&quot;Drift detection not available&quot;)
    if len(_recent_predictions) &lt; 10:
        return {&quot;total_features&quot;: VOCAB_SIZE, &quot;drifted_features&quot;: 0, &quot;drift_ratio&quot;: 0.0, &quot;drifted&quot;: [], &quot;all_results&quot;: []}
    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)
    if _metrics:
        _metrics.set_drift_ratio(summary[&quot;drift_ratio&quot;])
    return summary


@app.get(&quot;/stats&quot;, response_model=StatsResponse)
def get_stats():
    if _model is None or _model.base_model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    info = _model.to_dict()
    return StatsResponse(
        vocab_size=info[&quot;vocab_size&quot;],
        d_model=info[&quot;d_model&quot;],
        n_base_layers=info[&quot;n_base_layers&quot;],
        d_ff=info[&quot;d_ff&quot;],
        n_classes=info[&quot;n_classes&quot;],
        freeze_base=bool(info[&quot;freeze_base&quot;]),
        fine_tune_layers=info[&quot;fine_tune_layers&quot;],
        learning_rate=info[&quot;learning_rate&quot;],
        fine_tune_lr=info[&quot;fine_tune_lr&quot;],
        n_epochs_run=info[&quot;n_epochs_run&quot;],
        final_loss=info[&quot;final_loss&quot;],
        model_version=_model_version,
    )


@app.post(&quot;/predict&quot;, response_model=PredictResponse)
def predict(body: PredictRequest):
    &quot;&quot;&quot;Predict class using transfer learning model with frozen base and trainable head.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    X = np.array(body.tokens).reshape(1, -1)

    start = time.time()
    try:
        probs = _model.predict_proba(X)
        predicted_class = int(np.argmax(probs[0]))
        confidence = float(probs[0][predicted_class])

        response = PredictResponse(
            predicted_class=predicted_class,
            confidence=round(confidence, 4),
            class_probabilities=probs[0].tolist(),
            model_version=_model_version,
            base_model_frozen=_model.base_model.frozen if _model.base_model else True,
            fine_tune_layers=_model.fine_tune_layers,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append([float(t) for t in body.tokens])
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
<button class="copy-btn" onclick="copyCode('code-916501537')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-916501537"><code class="language-bash">uv run python -m transfer_learning.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>

</main>
<footer class="app-footer">
<p>Generated documentation for <strong>transfer-learning</strong></p>
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