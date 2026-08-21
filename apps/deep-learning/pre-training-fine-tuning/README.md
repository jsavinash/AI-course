<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>pre-training-fine-tuning - AI App Documentation</title>
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
<p class="section-subtitle">Pre-training and Fine-Tuning — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$\mathcal{L}_{MLM} = -\sum_{i \in M} \log P(x_i | x_{\setminus M})$$</div>
<div class="math-block">$$\mathcal{L}_{NSP} = \log P(\text{IsNext} | [CLS])$$</div>
<div class="math-block">$$\mathcal{L}_{total} = \mathcal{L}_{MLM} + \mathcal{L}_{NSP}$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>Pre-training learns general representations from large unlabeled corpora. Masked Language Modeling (MLM) predicts randomly masked tokens. Next Sentence Prediction (NSP) learns inter-sentence coherence. Fine-tuning adapts pre-trained weights to downstream tasks with minimal labeled data.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive MLM token prediction explorer; attention head visualization; layer-wise transfer analysis.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  MultiHeadAttention
  FeedForward
  AddNorm
  LoRAAdapter
  MLMHead
  NTPHead
  ClassificationHead
  Transformer</pre>
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
<tr><td><code>POST</code></td><td><code>/pretrain</code></td></tr>
<tr><td><code>POST</code></td><td><code>/finetune</code></td></tr>
<tr><td><code>GET</code></td><td><code>/drift</code></td></tr>
<tr><td><code>POST</code></td><td><code>/reload</code></td></tr></tbody>
</table>
</section>
<section id="usage" class="section usage-section">
<h2><span class="section-icon">▶</span> Usage</h2>
<p class="section-subtitle">Code examples and CLI commands</p>
<h3>Training Script</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-299627409')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-299627409"><code class="language-python">&quot;&quot;&quot;Training pipeline for Pre-training and Fine-tuning.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from pre_training_fine_tuning.data import (
    VOCAB_SIZE,
    generate_mlm_data,
    generate_ntp_data,
    generate_synthetic_data,
    save_training_data,
    train_test_split,
)
from pre_training_fine_tuning.model import Transformer

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 500,
    d_model: int = 128,
    n_heads: int = 4,
    n_encoder_layers: int = 2,
    n_decoder_layers: int = 2,
    d_ff: int = 512,
    max_seq_len: int = 32,
    learning_rate: float = 0.001,
    n_iterations: int = 100,
    weight_decay: float = 0.01,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
    phase: str = &quot;pretrain&quot;,
    objective: str = &quot;ntp&quot;,
    strategy: str = &quot;full&quot;,
) -&gt; dict:
    &quot;&quot;&quot;Train model for pre-training or fine-tuning.

    Args:
        phase: &quot;pretrain&quot; or &quot;finetune&quot;
        objective: &quot;mlm&quot; or &quot;ntp&quot; (for pre-training)
        strategy: &quot;full&quot;, &quot;feature_extraction&quot;, &quot;partial&quot;, &quot;peft&quot; (for fine-tuning)
    &quot;&quot;&quot;
    if phase == &quot;pretrain&quot;:
        if objective == &quot;mlm&quot;:
            X, y, mask_positions = generate_mlm_data(n_samples=n_samples, vocab_size=VOCAB_SIZE, random_seed=random_seed)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_seed=random_seed)
            logger.info(&quot;Generated MLM pre-training data&quot;, n_samples=n_samples, mask_prob=0.15)
        else:
            X, y = generate_ntp_data(n_samples=n_samples, vocab_size=VOCAB_SIZE, random_seed=random_seed)
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_seed=random_seed)
            mask_positions = None
            logger.info(&quot;Generated NTP pre-training data&quot;, n_samples=n_samples)
    else:
        X, y = generate_synthetic_data(n_samples=n_samples, vocab_size=VOCAB_SIZE, random_seed=random_seed, phase=&quot;finetune&quot;)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_seed=random_seed)
        mask_positions = None
        logger.info(&quot;Generated fine-tuning data&quot;, n_samples=n_samples, strategy=strategy)

    model_dir.mkdir(parents=True, exist_ok=True)
    save_training_data(X, y, model_dir / &quot;training_data.npz&quot;)

    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=d_model,
        n_heads=n_heads,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        max_seq_len=max_seq_len,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )

    model.fit(
        X_train,
        y_train,
        phase=phase,
        objective=objective,
        strategy=strategy,
        n_iterations=n_iterations,
        learning_rate=learning_rate,
        mask_positions=mask_positions if phase == &quot;pretrain&quot; and objective == &quot;mlm&quot; else None,
    )

    test_metrics = model.evaluate(X_test, y_test, phase=phase)
    logger.info(&quot;Training complete&quot;, training_mode=model.training_mode, final_loss=model.loss_history[-1])

    model_path = model_dir / f&quot;model_v{model_version}.npz&quot;
    model.save(str(model_path))

    metrics = {
        **test_metrics,
        &quot;training_mode&quot;: phase,
        &quot;objective&quot;: objective if phase == &quot;pretrain&quot; else &quot;finetune&quot;,
        &quot;strategy&quot;: strategy if phase == &quot;finetune&quot; else &quot;none&quot;,
        &quot;n_epochs_run&quot;: float(len(model.loss_history)),
        &quot;final_loss&quot;: model.loss_history[-1] if model.loss_history else 0.0,
        &quot;n_train_samples&quot;: float(len(X_train)),
        &quot;n_test_samples&quot;: float(len(X_test)),
        &quot;vocab_size&quot;: float(VOCAB_SIZE),
        &quot;d_model&quot;: float(d_model),
        &quot;n_heads&quot;: float(n_heads),
        &quot;n_encoder_layers&quot;: float(n_encoder_layers),
        &quot;n_decoder_layers&quot;: float(n_decoder_layers),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;pre-training-fine-tuning&quot;,
        model_version=model_version,
        model_type=&quot;transformer&quot;,
        metrics=metrics,
        parameters={
            &quot;vocab_size&quot;: VOCAB_SIZE,
            &quot;d_model&quot;: d_model,
            &quot;n_heads&quot;: n_heads,
            &quot;n_encoder_layers&quot;: n_encoder_layers,
            &quot;n_decoder_layers&quot;: n_decoder_layers,
            &quot;d_ff&quot;: d_ff,
            &quot;max_seq_len&quot;: max_seq_len,
            &quot;learning_rate&quot;: learning_rate,
            &quot;n_iterations&quot;: n_iterations,
            &quot;weight_decay&quot;: weight_decay,
            &quot;random_seed&quot;: random_seed,
            &quot;phase&quot;: phase,
            &quot;objective&quot;: objective,
            &quot;strategy&quot;: strategy,
        },
        artifacts={
            f&quot;model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.npz&quot;: model_dir / &quot;training_data.npz&quot;,
        },
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;pre_training_fine_tuning&quot;, &quot;model_type&quot;: &quot;Transformer&quot;, &quot;phase&quot;: phase},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;pre-training-fine-tuning&quot;,
            model_version=model_version,
            metrics=metrics,
            params={&quot;vocab_size&quot;: VOCAB_SIZE, &quot;d_model&quot;: d_model, &quot;n_heads&quot;: n_heads, &quot;n_iterations&quot;: n_iterations},
            artifacts={&quot;model&quot;: str(model_path)},
            tags={&quot;model_type&quot;: &quot;transformer&quot;, &quot;framework&quot;: &quot;numpy&quot;, &quot;phase&quot;: phase},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description=&quot;Train Pre-training and Fine-tuning Transformer&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;500&quot;)))
    parser.add_argument(&quot;--d-model&quot;, type=int, default=int(os.getenv(&quot;D_MODEL&quot;, &quot;128&quot;)))
    parser.add_argument(&quot;--n-heads&quot;, type=int, default=int(os.getenv(&quot;N_HEADS&quot;, &quot;4&quot;)))
    parser.add_argument(&quot;--n-encoder-layers&quot;, type=int, default=int(os.getenv(&quot;N_ENCODER_LAYERS&quot;, &quot;2&quot;)))
    parser.add_argument(&quot;--n-decoder-layers&quot;, type=int, default=int(os.getenv(&quot;N_DECODER_LAYERS&quot;, &quot;2&quot;)))
    parser.add_argument(&quot;--d-ff&quot;, type=int, default=int(os.getenv(&quot;D_FF&quot;, &quot;512&quot;)))
    parser.add_argument(&quot;--max-seq-len&quot;, type=int, default=int(os.getenv(&quot;MAX_SEQ_LEN&quot;, &quot;32&quot;)))
    parser.add_argument(&quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.001&quot;)))
    parser.add_argument(&quot;--n-iterations&quot;, type=int, default=int(os.getenv(&quot;N_ITERATIONS&quot;, &quot;100&quot;)))
    parser.add_argument(&quot;--weight-decay&quot;, type=float, default=float(os.getenv(&quot;WEIGHT_DECAY&quot;, &quot;0.01&quot;)))
    parser.add_argument(&quot;--model-version&quot;, type=str, default=os.getenv(&quot;MODEL_VERSION&quot;, &quot;1.0.0&quot;))
    parser.add_argument(&quot;--test-size&quot;, type=float, default=float(os.getenv(&quot;TEST_SIZE&quot;, &quot;0.2&quot;)))
    parser.add_argument(&quot;--random-seed&quot;, type=int, default=int(os.getenv(&quot;RANDOM_SEED&quot;, &quot;42&quot;)))
    parser.add_argument(&quot;--phase&quot;, type=str, default=os.getenv(&quot;PHASE&quot;, &quot;pretrain&quot;), choices=[&quot;pretrain&quot;, &quot;finetune&quot;])
    parser.add_argument(&quot;--objective&quot;, type=str, default=os.getenv(&quot;OBJECTIVE&quot;, &quot;ntp&quot;), choices=[&quot;mlm&quot;, &quot;ntp&quot;])
    parser.add_argument(&quot;--strategy&quot;, type=str, default=os.getenv(&quot;STRATEGY&quot;, &quot;full&quot;), choices=[&quot;full&quot;, &quot;feature_extraction&quot;, &quot;partial&quot;, &quot;peft&quot;])
    parser.add_argument(&quot;--register-mlflow&quot;, action=&quot;store_true&quot;, default=os.getenv(&quot;REGISTER_MLFLOW&quot;, &quot;false&quot;).lower() == &quot;true&quot;)
    parser.add_argument(&quot;--log-level&quot;, type=str, default=os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        data_path=args.data_path,
        n_samples=args.n_samples,
        d_model=args.d_model,
        n_heads=args.n_heads,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_ff=args.d_ff,
        max_seq_len=args.max_seq_len,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        test_size=args.test_size,
        random_seed=args.random_seed,
        phase=args.phase,
        objective=args.objective,
        strategy=args.strategy,
    )
    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-650346947')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-650346947"><code class="language-python">&quot;&quot;&quot;Serving API for Pre-training and Fine-tuning.&quot;&quot;&quot;

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

from pre_training_fine_tuning.data import VOCAB_SIZE, generate_synthetic_data
from pre_training_fine_tuning.model import Transformer, softmax

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;PRETRAIN_METRICS_PORT&quot;, &quot;8012&quot;))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class PretrainRequest(BaseModel):
    tokens: list[int] = Field(..., min_length=1, max_length=64)
    objective: str = Field(default=&quot;ntp&quot;, pattern=&quot;^(mlm|ntp)$&quot;)
    mask_positions: list[int] | None = Field(default=None)


class FinetuneRequest(BaseModel):
    tokens: list[int] = Field(..., min_length=1, max_length=64)
    label: int = Field(..., ge=0, le=9)
    strategy: str = Field(default=&quot;partial&quot;, pattern=&quot;^(full|feature_extraction|partial|peft)$&quot;)
    learning_rate: float = Field(default=0.0001, gt=0)
    n_iterations: int = Field(default=50, ge=1, le=500)


class PredictRequest(BaseModel):
    tokens: list[int] = Field(..., min_length=1, max_length=64)
    max_len: int = Field(default=10, ge=1, le=32)
    phase: str = Field(default=&quot;pretrain&quot;, pattern=&quot;^(pretrain|finetune)$&quot;)


class PredictResponse(BaseModel):
    generated_tokens: list[int]
    predicted_class: int | None = None
    confidence: float | None = None
    model_version: str
    training_mode: str
    objective: str | None = None
    strategy: str | None = None


class StatsResponse(BaseModel):
    vocab_size: int
    d_model: int
    n_heads: int
    n_encoder_layers: int
    n_decoder_layers: int
    d_ff: int
    training_mode: str
    pretraining_objective: str
    fine_tuning_strategy: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: Transformer | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;pre_training_fine_tuning&quot;, port=METRICS_PORT)
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
        model_name=&quot;pre-training-fine-tuning&quot;,
        model_version=_model_version,
        model_type=&quot;transformer&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;pre-training-fine-tuning&quot;, version=_model_version, mode=_model.training_mode)

    yield
    logger.info(&quot;Shutting down pre-training-fine-tuning API&quot;)


def _load_model() -&gt; tuple[Transformer, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            pt_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;pre-training-fine-tuning&quot;]
            if pt_models:
                pt_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = pt_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;model_v*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return Transformer.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;pre-training-fine-tuning&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;model_v*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return Transformer.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    npz_path = MODEL_DIR / &quot;model.npz&quot;
    if npz_path.exists():
        return Transformer.load(str(npz_path)), &quot;legacy&quot;

    candidate_paths = [
        Path(&quot;/app/artifacts/models/model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return Transformer.load(str(p)), &quot;1.0.0-bundled&quot;

    logger.warning(&quot;No pre-existing model found. Initializing baseline model.&quot;)
    X_base, y_base = generate_synthetic_data(n_samples=100, vocab_size=VOCAB_SIZE, random_seed=42, phase=&quot;pretrain&quot;)
    model = Transformer(
        vocab_size=VOCAB_SIZE,
        d_model=64,
        n_heads=4,
        n_encoder_layers=1,
        n_decoder_layers=1,
        d_ff=256,
        learning_rate=0.001,
        n_iterations=50,
        random_seed=42,
    )
    model.fit(X_base, y_base, phase=&quot;pretrain&quot;, objective=&quot;ntp&quot;)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    X_base, _ = generate_synthetic_data(n_samples=100, vocab_size=VOCAB_SIZE, random_seed=42, phase=&quot;pretrain&quot;)
    return X_base.astype(float)


app = FastAPI(
    title=&quot;Pre-training and Fine-tuning API&quot;,
    description=&quot;Pre-training (MLM, NTP) and fine-tuning (full, feature extraction, partial, PEFT) for deep learning models&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;pre_training_fine_tuning-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;training_mode&quot;: _model.training_mode if _model else &quot;unknown&quot;,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;pretrain&quot;: &quot;POST /pretrain&quot;,
            &quot;finetune&quot;: &quot;POST /finetune&quot;,
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


@app.post(&quot;/pretrain&quot;)
def pretrain(body: PretrainRequest):
    &quot;&quot;&quot;Run pre-training step (simulated inference) with MLM or NTP objective.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    X = np.array(body.tokens).reshape(1, -1)
    start = time.time()

    try:
        if body.objective == &quot;mlm&quot;:
            from pre_training_fine_tuning.data import generate_mlm_data
            _, y_orig, _ = generate_mlm_data(n_samples=1, vocab_size=VOCAB_SIZE, random_seed=42)
            y_orig = y_orig[:1]
            _model.pretraining_objective = &quot;mlm&quot;
            logits = _model.mlm_head.forward(_model._embed(X))
            predicted = np.argmax(softmax(logits[0]), axis=-1).tolist()
        else:
            _model.pretraining_objective = &quot;ntp&quot;
            generated = _model.predict(X, max_len=min(10, X.shape[1] if X.ndim &gt; 1 else 10), phase=&quot;pretrain&quot;)
            predicted = generated.tolist()

        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append([float(t) for t in body.tokens])
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return {
            &quot;predicted_tokens&quot;: predicted,
            &quot;objective&quot;: body.objective,
            &quot;model_version&quot;: _model_version,
            &quot;training_mode&quot;: _model.training_mode,
        }
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;pretrain&quot;)
        logger.exception(&quot;Pre-training inference failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Pre-training inference failed&quot;) from e


@app.post(&quot;/finetune&quot;)
def finetune(body: FinetuneRequest):
    &quot;&quot;&quot;Fine-tune model on a single sample (simulated).&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    X = np.array(body.tokens).reshape(1, -1)
    y = np.array([body.label])

    start = time.time()
    try:
        _model.fit(
            X,
            y,
            phase=&quot;finetune&quot;,
            strategy=body.strategy,
            n_iterations=body.n_iterations,
            learning_rate=body.learning_rate,
        )

        preds = _model.predict(X, phase=&quot;finetune&quot;)
        predicted_class = int(preds[0]) if len(preds) &gt; 0 else 0

        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        return {
            &quot;predicted_class&quot;: predicted_class,
            &quot;true_label&quot;: body.label,
            &quot;strategy&quot;: body.strategy,
            &quot;model_version&quot;: _model_version,
            &quot;training_mode&quot;: _model.training_mode,
            &quot;final_loss&quot;: _model.loss_history[-1] if _model.loss_history else 0.0,
        }
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;finetune&quot;)
        logger.exception(&quot;Fine-tuning failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Fine-tuning failed&quot;) from e


@app.post(&quot;/predict&quot;, response_model=PredictResponse)
def predict(body: PredictRequest):
    &quot;&quot;&quot;Generate predictions using pre-trained or fine-tuned model.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    X = np.array(body.tokens).reshape(1, -1)
    start = time.time()

    try:
        if body.phase == &quot;finetune&quot;:
            preds = _model.predict(X, phase=&quot;finetune&quot;)
            predicted_class = int(preds[0]) if len(preds) &gt; 0 else 0
            logits = _model.classification_head.forward(np.mean(_model._embed(X), axis=1))
            probs = softmax(logits[0])
            confidence = float(probs[predicted_class]) if predicted_class &lt; len(probs) else 0.0

            response = PredictResponse(
                generated_tokens=[],
                predicted_class=predicted_class,
                confidence=round(confidence, 4),
                model_version=_model_version,
                training_mode=_model.training_mode,
                strategy=_model.fine_tuning_strategy,
            )
        else:
            generated = _model.predict(X, max_len=body.max_len, phase=&quot;pretrain&quot;)
            logits = _model._embed(X) @ _model.W_out.T
            probs = softmax(logits.flatten())
            predicted_token = int(generated[0]) if len(generated) &gt; 0 else 0
            confidence = float(probs[predicted_token]) if predicted_token &lt; len(probs) else 0.0

            response = PredictResponse(
                generated_tokens=generated.tolist(),
                predicted_class=None,
                confidence=round(confidence, 4),
                model_version=_model_version,
                training_mode=_model.training_mode,
                objective=_model.pretraining_objective,
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
        raise HTTPException(status_code=500, detail=&quot;Prediction failed&quot;) from e


@app.get(&quot;/drift&quot;)
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
    if _model is None or not _model.encoder_layers:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    info = _model.to_dict()
    return StatsResponse(
        vocab_size=info[&quot;vocab_size&quot;],
        d_model=info[&quot;d_model&quot;],
        n_heads=info[&quot;n_heads&quot;],
        n_encoder_layers=info[&quot;n_encoder_layers&quot;],
        n_decoder_layers=info[&quot;n_decoder_layers&quot;],
        d_ff=info[&quot;d_ff&quot;],
        training_mode=info[&quot;training_mode&quot;],
        pretraining_objective=info[&quot;pretraining_objective&quot;],
        fine_tuning_strategy=info[&quot;fine_tuning_strategy&quot;],
        n_epochs_run=info[&quot;n_epochs_run&quot;],
        final_loss=info[&quot;final_loss&quot;],
        model_version=_model_version,
    )


@app.post(&quot;/reload&quot;)
def reload_model():
    global _model, _model_version, _reference_data
    try:
        _model, _model_version = _load_model()
        if _metrics:
            _metrics.set_model_version(_model_version)
            _metrics.set_model_info(
                model_name=&quot;pre-training-fine-tuning&quot;,
                model_version=_model_version,
                model_type=&quot;transformer&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(&quot;Model reloaded&quot;, model=&quot;pre-training-fine-tuning&quot;, version=_model_version)
        return {&quot;status&quot;: &quot;reloaded&quot;, &quot;model_version&quot;: _model_version, &quot;training_mode&quot;: _model.training_mode}
    except Exception as e:
        logger.exception(&quot;Model reload failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=f&quot;Reload failed: {e}&quot;) from e</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-1279103830')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-1279103830"><code class="language-bash">uv run python -m pre_training_fine_tuning.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>

</main>
<footer class="app-footer">
<p>Generated documentation for <strong>pre-training-fine-tuning</strong></p>
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