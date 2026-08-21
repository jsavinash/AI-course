<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>multimodal-llm - AI App Documentation</title>
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
<p class="section-subtitle">Multimodal Learning — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$h = \text{CrossAttention}(Q_{\text{text}}, K_{\text{image}}, V_{\text{image}})$$</div>
<div class="math-block">$$\mathcal{L} = \mathcal{L}_{\text{image-text}} + \lambda_1 \mathcal{L}_{\text{image}} + \lambda_2 \mathcal{L}_{\text{text}}$$</div>
<div class="math-block">$$\text{cosine}(u, v) = \frac{u^T v}{\|u\| \|v\|}$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>Multimodal models align representations from different modalities in a shared embedding space. Cross-attention allows one modality to query another. Contrastive learning pulls matched pairs together and pushes unmatched pairs apart. The total loss balances cross-modal alignment with unimodal task losses.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive embedding alignment plot; cross-attention weight heatmap; modality contribution explorer.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  TextEncoder
  ImageEncoder
  AudioEncoder
  Connector
  FusionMechanism
  MultiHeadAttention
  FeedForward
  AddNorm
  TransformerEncoder
  TransformerDecoder
  LLMBackbone
  MultimodalLLM</pre>
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
<button class="copy-btn" onclick="copyCode('code-836150620')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-836150620"><code class="language-python">&quot;&quot;&quot;Training pipeline for Multimodal Language Modeling.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from multimodal_llm.data import (
    VOCAB_SIZE,
    generate_synthetic_multimodal_data,
    save_multimodal_data,
    train_test_split_multimodal,
)
from multimodal_llm.model import MultimodalLLM

logger = get_logger(__name__)


def train(
    model_dir: Path,
    n_samples: int = 500,
    vocab_size: int = VOCAB_SIZE,
    seq_len: int = 64,
    d_model: int = 256,
    text_encoder_dim: int = 256,
    image_encoder_dim: int = 768,
    audio_encoder_dim: int = 80,
    connector_dim: int = 512,
    fusion_type: str = &quot;hybrid&quot;,
    max_seq_len: int = 128,
    n_encoder_layers: int = 2,
    n_decoder_layers: int = 2,
    d_ff: int = 512,
    learning_rate: float = 0.001,
    n_iterations: int = 100,
    weight_decay: float = 0.01,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    test_size: float = 0.2,
    random_seed: int = 42,
    include_image: bool = True,
    include_audio: bool = True,
) -&gt; dict:
    logger.info(&quot;Generating multimodal training data&quot;, n_samples=n_samples, include_image=include_image, include_audio=include_audio)
    data = generate_synthetic_multimodal_data(
        n_samples=n_samples,
        vocab_size=vocab_size,
        seq_len=seq_len,
        random_seed=random_seed,
        include_image=include_image,
        include_audio=include_audio,
    )

    train_data, test_data = train_test_split_multimodal(data, test_size=test_size, random_seed=random_seed)
    logger.info(&quot;Data split&quot;, n_train=len(train_data[&quot;text_tokens&quot;]), n_test=len(test_data[&quot;text_tokens&quot;]))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_multimodal_data(data, model_dir / &quot;training_data.npz&quot;)

    X_train_text = train_data[&quot;text_tokens&quot;]
    y_train = train_data[&quot;text_targets&quot;]
    X_train_image = train_data.get(&quot;image_patches&quot;)
    X_train_audio = train_data.get(&quot;mel_spectrogram&quot;)

    model = MultimodalLLM(
        vocab_size=vocab_size,
        d_model=d_model,
        text_encoder_dim=text_encoder_dim,
        image_encoder_dim=image_encoder_dim,
        audio_encoder_dim=audio_encoder_dim,
        connector_dim=connector_dim,
        fusion_type=fusion_type,
        max_seq_len=max_seq_len,
        n_encoder_layers=n_encoder_layers,
        n_decoder_layers=n_decoder_layers,
        d_ff=d_ff,
        learning_rate=learning_rate,
        n_iterations=n_iterations,
        weight_decay=weight_decay,
        random_seed=random_seed,
    )
    model.fit(X_train_text, y_train, image_patches=X_train_image, mel_spectrogram=X_train_audio)

    X_test_text = test_data[&quot;text_tokens&quot;]
    y_test = test_data[&quot;text_targets&quot;]
    X_test_image = test_data.get(&quot;image_patches&quot;)
    X_test_audio = test_data.get(&quot;mel_spectrogram&quot;)

    model.predict(X_test_text[:5], image_patches=X_test_image[:5] if X_test_image is not None else None, mel_spectrogram=X_test_audio[:5] if X_test_audio is not None else None)
    test_metrics = model.evaluate(X_test_text, y_test)
    logger.info(&quot;Training complete&quot;, training_mode=model.training_mode, final_loss=model.loss_history[-1])

    model_path = model_dir / f&quot;multimodal_llm_v{model_version}.npz&quot;
    model.save(str(model_path))

    metrics = {
        **test_metrics,
        &quot;training_mode&quot;: &quot;supervised&quot;,
        &quot;n_epochs_run&quot;: float(len(model.loss_history)),
        &quot;final_loss&quot;: model.loss_history[-1] if model.loss_history else 0.0,
        &quot;n_train_samples&quot;: float(len(X_train_text)),
        &quot;n_test_samples&quot;: float(len(X_test_text)),
        &quot;vocab_size&quot;: float(vocab_size),
        &quot;d_model&quot;: float(d_model),
        &quot;connector_dim&quot;: float(connector_dim),
        &quot;fusion_type&quot;: fusion_type,
        &quot;n_encoder_layers&quot;: float(n_encoder_layers),
        &quot;n_decoder_layers&quot;: float(n_decoder_layers),
    }

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;multimodal-llm&quot;,
        model_version=model_version,
        model_type=&quot;classification&quot;,
        metrics=metrics,
        parameters={
            &quot;vocab_size&quot;: vocab_size,
            &quot;d_model&quot;: d_model,
            &quot;text_encoder_dim&quot;: text_encoder_dim,
            &quot;image_encoder_dim&quot;: image_encoder_dim,
            &quot;audio_encoder_dim&quot;: audio_encoder_dim,
            &quot;connector_dim&quot;: connector_dim,
            &quot;fusion_type&quot;: fusion_type,
            &quot;max_seq_len&quot;: max_seq_len,
            &quot;n_encoder_layers&quot;: n_encoder_layers,
            &quot;n_decoder_layers&quot;: n_decoder_layers,
            &quot;d_ff&quot;: d_ff,
            &quot;learning_rate&quot;: learning_rate,
            &quot;n_iterations&quot;: n_iterations,
            &quot;weight_decay&quot;: weight_decay,
            &quot;random_seed&quot;: random_seed,
            &quot;include_image&quot;: include_image,
            &quot;include_audio&quot;: include_audio,
        },
        artifacts={
            f&quot;multimodal_llm_v{model_version}.npz&quot;: model_path,
            &quot;training_data.npz&quot;: model_dir / &quot;training_data.npz&quot;,
        },
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;multimodal_llm&quot;, &quot;model_type&quot;: &quot;MLLM&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;multimodal-llm&quot;,
            model_version=model_version,
            metrics=metrics,
            params={&quot;vocab_size&quot;: vocab_size, &quot;d_model&quot;: d_model, &quot;fusion_type&quot;: fusion_type, &quot;n_iterations&quot;: n_iterations},
            artifacts={&quot;model&quot;: str(model_path)},
            tags={&quot;model_type&quot;: &quot;multimodal_llm&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )

    return metrics


def main():
    parser = argparse.ArgumentParser(description=&quot;Train Multimodal LLM&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;500&quot;)))
    parser.add_argument(&quot;--vocab-size&quot;, type=int, default=int(os.getenv(&quot;VOCAB_SIZE&quot;, str(VOCAB_SIZE))))
    parser.add_argument(&quot;--seq-len&quot;, type=int, default=int(os.getenv(&quot;SEQ_LEN&quot;, &quot;64&quot;)))
    parser.add_argument(&quot;--d-model&quot;, type=int, default=int(os.getenv(&quot;D_MODEL&quot;, &quot;256&quot;)))
    parser.add_argument(&quot;--text-encoder-dim&quot;, type=int, default=int(os.getenv(&quot;TEXT_ENCODER_DIM&quot;, &quot;256&quot;)))
    parser.add_argument(&quot;--image-encoder-dim&quot;, type=int, default=int(os.getenv(&quot;IMAGE_ENCODER_DIM&quot;, &quot;768&quot;)))
    parser.add_argument(&quot;--audio-encoder-dim&quot;, type=int, default=int(os.getenv(&quot;AUDIO_ENCODER_DIM&quot;, &quot;80&quot;)))
    parser.add_argument(&quot;--connector-dim&quot;, type=int, default=int(os.getenv(&quot;CONNECTOR_DIM&quot;, &quot;512&quot;)))
    parser.add_argument(&quot;--fusion-type&quot;, type=str, default=os.getenv(&quot;FUSION_TYPE&quot;, &quot;hybrid&quot;), choices=[&quot;early&quot;, &quot;late&quot;, &quot;hybrid&quot;])
    parser.add_argument(&quot;--max-seq-len&quot;, type=int, default=int(os.getenv(&quot;MAX_SEQ_LEN&quot;, &quot;128&quot;)))
    parser.add_argument(&quot;--n-encoder-layers&quot;, type=int, default=int(os.getenv(&quot;N_ENCODER_LAYERS&quot;, &quot;2&quot;)))
    parser.add_argument(&quot;--n-decoder-layers&quot;, type=int, default=int(os.getenv(&quot;N_DECODER_LAYERS&quot;, &quot;2&quot;)))
    parser.add_argument(&quot;--d-ff&quot;, type=int, default=int(os.getenv(&quot;D_FF&quot;, &quot;512&quot;)))
    parser.add_argument(&quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.001&quot;)))
    parser.add_argument(&quot;--n-iterations&quot;, type=int, default=int(os.getenv(&quot;N_ITERATIONS&quot;, &quot;100&quot;)))
    parser.add_argument(&quot;--weight-decay&quot;, type=float, default=float(os.getenv(&quot;WEIGHT_DECAY&quot;, &quot;0.01&quot;)))
    parser.add_argument(&quot;--model-version&quot;, type=str, default=os.getenv(&quot;MODEL_VERSION&quot;, &quot;1.0.0&quot;))
    parser.add_argument(&quot;--test-size&quot;, type=float, default=float(os.getenv(&quot;TEST_SIZE&quot;, &quot;0.2&quot;)))
    parser.add_argument(&quot;--random-seed&quot;, type=int, default=int(os.getenv(&quot;RANDOM_SEED&quot;, &quot;42&quot;)))
    parser.add_argument(&quot;--register-mlflow&quot;, action=&quot;store_true&quot;, default=os.getenv(&quot;REGISTER_MLFLOW&quot;, &quot;false&quot;).lower() == &quot;true&quot;)
    parser.add_argument(&quot;--no-image&quot;, action=&quot;store_true&quot;, help=&quot;Disable image modality&quot;)
    parser.add_argument(&quot;--no-audio&quot;, action=&quot;store_true&quot;, help=&quot;Disable audio modality&quot;)
    parser.add_argument(&quot;--log-level&quot;, type=str, default=os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    args = parser.parse_args()

    setup_logging(args.log_level)
    args.model_dir.mkdir(parents=True, exist_ok=True)

    metrics = train(
        model_dir=args.model_dir,
        n_samples=args.n_samples,
        vocab_size=args.vocab_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        text_encoder_dim=args.text_encoder_dim,
        image_encoder_dim=args.image_encoder_dim,
        audio_encoder_dim=args.audio_encoder_dim,
        connector_dim=args.connector_dim,
        fusion_type=args.fusion_type,
        max_seq_len=args.max_seq_len,
        n_encoder_layers=args.n_encoder_layers,
        n_decoder_layers=args.n_decoder_layers,
        d_ff=args.d_ff,
        learning_rate=args.learning_rate,
        n_iterations=args.n_iterations,
        weight_decay=args.weight_decay,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        test_size=args.test_size,
        random_seed=args.random_seed,
        include_image=not args.no_image,
        include_audio=not args.no_audio,
    )
    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-3843070131')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3843070131"><code class="language-python">&quot;&quot;&quot;Serving API for Multimodal Language Modeling.&quot;&quot;&quot;

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

from multimodal_llm.data import VOCAB_SIZE, generate_synthetic_multimodal_data
from multimodal_llm.model import MultimodalLLM, softmax

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;MULTIMODAL_LLM_METRICS_PORT&quot;, &quot;8012&quot;))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class MultimodalPredictRequest(BaseModel):
    text_tokens: list[int] = Field(..., min_length=1, max_length=64)
    image_patches: list[list[float]] | None = Field(default=None)
    mel_spectrogram: list[list[float]] | None = Field(default=None)
    max_len: int = Field(default=10, ge=1, le=32)


class MultimodalPredictResponse(BaseModel):
    generated_tokens: list[int]
    predicted_token: int
    confidence: float
    model_version: str
    training_mode: str
    modalities_used: list[str]


class DriftResponse(BaseModel):
    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


class StatsResponse(BaseModel):
    vocab_size: int
    d_model: int
    connector_dim: int
    fusion_type: str
    n_encoder_layers: int
    n_decoder_layers: int
    training_mode: str
    n_epochs_run: int
    final_loss: float
    model_version: str


_model: MultimodalLLM | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;multimodal_llm&quot;, port=METRICS_PORT)
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
        model_name=&quot;multimodal-llm&quot;,
        model_version=_model_version,
        model_type=&quot;multimodal&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;multimodal-llm&quot;, version=_model_version)

    yield
    logger.info(&quot;Shutting down multimodal-llm API&quot;)


def _load_model() -&gt; tuple[MultimodalLLM, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            mm_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;multimodal-llm&quot;]
            if mm_models:
                mm_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = mm_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;multimodal_llm_v*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return MultimodalLLM.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;multimodal-llm&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;multimodal_llm_v*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return MultimodalLLM.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    npz_path = MODEL_DIR / &quot;multimodal_llm.npz&quot;
    if npz_path.exists():
        return MultimodalLLM.load(str(npz_path)), &quot;legacy&quot;

    candidate_paths = [
        Path(&quot;/app/artifacts/models/multimodal_llm_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;multimodal_llm_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return MultimodalLLM.load(str(p)), &quot;1.0.0-bundled&quot;

    logger.warning(&quot;No pre-existing model found. Initializing baseline model.&quot;)
    X_base, y_base = generate_synthetic_multimodal_data(n_samples=100, random_seed=42)
    model = MultimodalLLM(
        vocab_size=VOCAB_SIZE,
        d_model=128,
        text_encoder_dim=128,
        image_encoder_dim=256,
        audio_encoder_dim=64,
        connector_dim=256,
        fusion_type=&quot;hybrid&quot;,
        max_seq_len=64,
        n_encoder_layers=1,
        n_decoder_layers=1,
        d_ff=256,
        learning_rate=0.001,
        n_iterations=50,
        random_seed=42,
    )
    model.fit(X_base[&quot;text_tokens&quot;], y_base, image_patches=X_base.get(&quot;image_patches&quot;), mel_spectrogram=X_base.get(&quot;mel_spectrogram&quot;))
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    X_base, _ = generate_synthetic_multimodal_data(n_samples=100, random_seed=42)
    return X_base[&quot;text_tokens&quot;].astype(float)


app = FastAPI(
    title=&quot;Multimodal LLM API&quot;,
    description=&quot;Multimodal Large Language Model that integrates text, image, and audio inputs using modality encoders, connectors, fusion mechanisms, and LLM backbone&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;multimodal_llm-api&quot;,
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
                model_name=&quot;multimodal-llm&quot;,
                model_version=_model_version,
                model_type=&quot;multimodal&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(&quot;Model reloaded&quot;, model=&quot;multimodal-llm&quot;, version=_model_version)
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
    if _model is None or not _model.text_encoder:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    info = _model.to_dict()
    return StatsResponse(
        vocab_size=info[&quot;vocab_size&quot;],
        d_model=info[&quot;d_model&quot;],
        connector_dim=info[&quot;connector_dim&quot;],
        fusion_type=info[&quot;fusion_type&quot;],
        n_encoder_layers=info[&quot;n_encoder_layers&quot;],
        n_decoder_layers=info[&quot;n_decoder_layers&quot;],
        training_mode=info[&quot;training_mode&quot;],
        n_epochs_run=info[&quot;n_epochs_run&quot;],
        final_loss=info[&quot;final_loss&quot;],
        model_version=_model_version,
    )


@app.post(&quot;/predict&quot;, response_model=MultimodalPredictResponse)
def predict(body: MultimodalPredictRequest):
    &quot;&quot;&quot;Generate next-token prediction using multimodal LLM with text, image, and audio inputs.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    text_tokens = np.array(body.text_tokens).reshape(1, -1)
    image_patches = np.array(body.image_patches).reshape(1, -1, 768) if body.image_patches else None
    mel_spectrogram = np.array(body.mel_spectrogram).reshape(1, -1, 80) if body.mel_spectrogram else None

    modalities_used = [&quot;text&quot;]
    if image_patches is not None:
        modalities_used.append(&quot;image&quot;)
    if mel_spectrogram is not None:
        modalities_used.append(&quot;audio&quot;)

    start = time.time()
    try:
        generated = _model.predict(text_tokens, image_patches=image_patches, mel_spectrogram=mel_spectrogram, max_len=body.max_len)
        predicted_token = int(generated[0]) if len(generated) &gt; 0 else 0

        logits = _model.llm_backbone.forward(text_tokens)
        probs = softmax(logits.flatten())
        confidence = float(probs[predicted_token]) if predicted_token &lt; len(probs) else 0.0

        response = MultimodalPredictResponse(
            generated_tokens=generated.tolist(),
            predicted_token=predicted_token,
            confidence=round(confidence, 4),
            model_version=_model_version,
            training_mode=_model.training_mode,
            modalities_used=modalities_used,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append([float(t) for t in body.text_tokens])
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
<button class="copy-btn" onclick="copyCode('code-3793007334')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-3793007334"><code class="language-bash">uv run python -m multimodal_llm.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>

</main>
<footer class="app-footer">
<p>Generated documentation for <strong>multimodal-llm</strong></p>
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