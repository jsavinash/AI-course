<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>video-generation - AI App Documentation</title>
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
<p class="section-subtitle">Video Generation — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$P(x_{1:T}) = \prod_{t=1}^{T} P(x_t | x_{<t})$$</div>
<div class="math-block">$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{<t}; \theta)$$</div>
<div class="math-block">$$\text{SSIM}(x, \hat{x}) = \frac{(2\mu_x \mu_{\hat{x}} + c_1)(2\sigma_{x\hat{x}} + c_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + c_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + c_2)}$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>Video generation extends sequence modeling to spatiotemporal data. 3D convolutions or factored spatial-temporal attention capture motion. Temporal consistency is enforced via warping or predictive coding. Frame-wise perceptual losses improve visual quality.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive frame-by-frame playback with generated vs real overlay; optical flow visualization; temporal consistency score.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  VideoTokenizer
  MultiHeadAttention
  AddNorm
  FeedForward
  TransformerBlock
  TextConditioning
  LatentVideoEncoder
  SpatiotemporalDiffusionModel
  VideoGenerationModel</pre>
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
<tr><td><code>GET</code></td><td><code>/metrics</code></td></tr></tbody>
</table>
</section>
<section id="usage" class="section usage-section">
<h2><span class="section-icon">▶</span> Usage</h2>
<p class="section-subtitle">Code examples and CLI commands</p>
<h3>Training Script</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-349085834')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-349085834"><code class="language-python">&quot;&quot;&quot;Training pipeline for Video Generation.&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

import numpy as np
from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from video_generation.data import load_video_dataset, save_dataset, train_test_split_videos
from video_generation.model import VideoGenerationModel

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path | None = None,
    n_samples: int = 200,
    img_size: int = 32,
    n_frames: int = 8,
    latent_dim: int = 64,
    model_id: str = &quot;video-generation-v1&quot;,
    n_diffusion_steps: int = 1000,
    model_version: str = &quot;1.0.0&quot;,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -&gt; dict:
    logger.info(&quot;Loading video dataset&quot;, n_samples=n_samples, n_frames=n_frames)
    videos, prompts = load_video_dataset(data_path=data_path, n_samples=n_samples, random_seed=random_seed)

    X_train, X_test, prompts_train, prompts_test = train_test_split_videos(videos, prompts, test_size=0.2, random_seed=random_seed)
    logger.info(&quot;Data split&quot;, n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(videos, prompts, model_dir / &quot;training_data.npz&quot;)

    model = VideoGenerationModel(
        model_id=model_id,
        img_size=img_size,
        n_frames=n_frames,
        latent_dim=latent_dim,
        n_diffusion_steps=n_diffusion_steps,
        random_seed=random_seed,
    )
    model._init()

    X_train_flat = X_train.reshape(len(X_train), -1)
    X_test_flat = X_test.reshape(len(X_test), -1)
    metrics = model.fit(X_train_flat, np.zeros(len(X_train_flat)), n_iterations=10)
    logger.info(&quot;Training finished&quot;, metrics=metrics)

    eval_metrics = model.evaluate(X_test_flat, np.zeros(len(X_test_flat)))
    logger.info(&quot;Evaluation metrics&quot;, metrics=eval_metrics)

    model_path = model_dir / f&quot;video_generation_v{model_version}.npz&quot;
    model.save(str(model_path))

    combined_metrics = {**metrics, **eval_metrics}
    combined_metrics.update({
        &quot;img_size&quot;: float(img_size),
        &quot;n_frames&quot;: float(n_frames),
        &quot;latent_dim&quot;: float(latent_dim),
        &quot;n_diffusion_steps&quot;: float(n_diffusion_steps),
        &quot;n_samples&quot;: float(n_samples),
    })

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;video-generation&quot;,
        model_version=model_version,
        model_type=&quot;generative&quot;,
        metrics=combined_metrics,
        parameters={
            &quot;model_id&quot;: model_id,
            &quot;img_size&quot;: img_size,
            &quot;n_frames&quot;: n_frames,
            &quot;latent_dim&quot;: latent_dim,
            &quot;n_diffusion_steps&quot;: n_diffusion_steps,
            &quot;n_samples&quot;: n_samples,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={f&quot;video_generation_v{model_version}.npz&quot;: model_path, &quot;training_data.npz&quot;: model_dir / &quot;training_data.npz&quot;},
        tags={&quot;framework&quot;: &quot;numpy&quot;, &quot;task&quot;: &quot;video_generation&quot;, &quot;model_type&quot;: &quot;VideoGeneration&quot;},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;video-generation&quot;,
            model_version=model_version,
            metrics=combined_metrics,
            params={&quot;model_id&quot;: model_id, &quot;img_size&quot;: img_size, &quot;n_frames&quot;: n_frames, &quot;latent_dim&quot;: latent_dim, &quot;n_diffusion_steps&quot;: n_diffusion_steps, &quot;n_samples&quot;: n_samples},
            artifacts={&quot;model&quot;: str(model_path)},
            tags={&quot;model_type&quot;: &quot;video_generation&quot;, &quot;framework&quot;: &quot;numpy&quot;},
        )

    return combined_metrics


def main():
    parser = argparse.ArgumentParser(description=&quot;Train Video Generation Model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--n-samples&quot;, type=int, default=int(os.getenv(&quot;N_SAMPLES&quot;, &quot;200&quot;)))
    parser.add_argument(&quot;--img-size&quot;, type=int, default=int(os.getenv(&quot;IMG_SIZE&quot;, &quot;32&quot;)))
    parser.add_argument(&quot;--n-frames&quot;, type=int, default=int(os.getenv(&quot;N_FRAMES&quot;, &quot;8&quot;)))
    parser.add_argument(&quot;--latent-dim&quot;, type=int, default=int(os.getenv(&quot;LATENT_DIM&quot;, &quot;64&quot;)))
    parser.add_argument(&quot;--model-id&quot;, type=str, default=os.getenv(&quot;MODEL_ID&quot;, &quot;video-generation-v1&quot;))
    parser.add_argument(&quot;--n-diffusion-steps&quot;, type=int, default=int(os.getenv(&quot;N_DIFFUSION_STEPS&quot;, &quot;1000&quot;)))
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
        img_size=args.img_size,
        n_frames=args.n_frames,
        latent_dim=args.latent_dim,
        model_id=args.model_id,
        n_diffusion_steps=args.n_diffusion_steps,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )
    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-1774562669')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-1774562669"><code class="language-python">&quot;&quot;&quot;Serving API for Video Generation.&quot;&quot;&quot;

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

from video_generation.data import DEFAULT_IMG_SIZE, DEFAULT_LATENT_DIM, DEFAULT_N_FRAMES
from video_generation.model import VideoGenerationModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;VIDEO_GENERATION_METRICS_PORT&quot;, &quot;9026&quot;))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))


class GenerateVideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    n_steps: int = Field(default=50, ge=1, le=200)
    mode: str = Field(default=&quot;text-to-video&quot;, pattern=&quot;^(text-to-video|image-to-video)$&quot;)


class GenerateVideoResponse(BaseModel):
    video_shape: tuple[int, int, int, int]
    prompt: str
    mode: str
    model_version: str


class AnimateImageRequest(BaseModel):
    image_data: list[float] = Field(..., min_length=1)
    motion_prompt: str = Field(..., min_length=1)
    n_steps: int = Field(default=50, ge=1, le=200)


class AnimateImageResponse(BaseModel):
    video_shape: tuple[int, int, int, int]
    motion_prompt: str
    model_version: str


class StatsResponse(BaseModel):
    model_id: str
    img_size: int
    n_frames: int
    latent_dim: int
    n_diffusion_steps: int
    model_version: str


_model: VideoGenerationModel | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;video_generation&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    feature_names = [f&quot;latent_{i}&quot; for i in range(DEFAULT_LATENT_DIM)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: &quot;float&quot; for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;video-generation&quot;,
        model_version=_model_version,
        model_type=&quot;generative&quot;,
    )

    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;video-generation&quot;, version=_model_version)

    yield
    logger.info(&quot;Shutting down video-generation API&quot;)


def _load_model() -&gt; tuple[VideoGenerationModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            vg_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;video-generation&quot;]
            if vg_models:
                vg_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = vg_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;video_generation_v*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return VideoGenerationModel.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;video-generation&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;video_generation_v*.npz&quot;)) + list(model_dir.glob(&quot;*.npz&quot;))
                if npz_files:
                    return VideoGenerationModel.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    npz_path = MODEL_DIR / &quot;video_generation.npz&quot;
    if npz_path.exists():
        return VideoGenerationModel.load(str(npz_path)), &quot;legacy&quot;

    candidate_paths = [
        Path(&quot;/app/artifacts/models/video_generation_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;video_generation_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return VideoGenerationModel.load(str(p)), &quot;1.0.0-bundled&quot;

    logger.warning(&quot;No pre-existing model found. Initializing baseline model.&quot;)
    model = VideoGenerationModel(model_id=&quot;baseline&quot;, img_size=DEFAULT_IMG_SIZE, n_frames=DEFAULT_N_FRAMES, latent_dim=DEFAULT_LATENT_DIM)
    model._init()
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    from video_generation.data import generate_synthetic_videos
    videos, _ = generate_synthetic_videos(n_samples=100, random_seed=42)
    return videos.reshape(100, -1).astype(float)


app = FastAPI(
    title=&quot;Video Generation API&quot;,
    description=&quot;Spatiotemporal diffusion model for text-to-video and image-to-video generation&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    return {
        &quot;service&quot;: &quot;video-generation-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;generate&quot;: &quot;POST /generate&quot;,
            &quot;animate&quot;: &quot;POST /animate&quot;,
            &quot;stats&quot;: &quot;GET /stats&quot;,
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
        &quot;model_id&quot;: _model.model_id if _model else &quot;unknown&quot;,
    }


@app.get(&quot;/metrics&quot;)
def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(&quot;/generate&quot;, response_model=GenerateVideoResponse)
def generate_video(body: GenerateVideoRequest):
    &quot;&quot;&quot;Generate a video from a text prompt.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    start = time.time()
    try:
        video = _model.generate_from_text(body.prompt, n_steps=body.n_steps)
        response = GenerateVideoResponse(
            video_shape=video.shape,
            prompt=body.prompt,
            mode=body.mode,
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append([float(body.n_steps)])
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;generation&quot;)
        logger.exception(&quot;Video generation failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Video generation failed&quot;) from e


@app.post(&quot;/animate&quot;, response_model=AnimateImageResponse)
def animate_image(body: AnimateImageRequest):
    &quot;&quot;&quot;Animate a static image using a motion prompt (image-to-video).&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    start = time.time()
    try:
        expected_size = DEFAULT_IMG_SIZE * DEFAULT_IMG_SIZE * 3
        if len(body.image_data) != expected_size:
            raise HTTPException(status_code=400, detail=f&quot;image_data must have {expected_size} elements for {DEFAULT_IMG_SIZE}x{DEFAULT_IMG_SIZE} RGB image&quot;)
        image = np.array(body.image_data).reshape(DEFAULT_IMG_SIZE, DEFAULT_IMG_SIZE, 3)
        video = _model.animate_from_image(image, body.motion_prompt, n_steps=body.n_steps)
        response = AnimateImageResponse(
            video_shape=video.shape,
            motion_prompt=body.motion_prompt,
            model_version=_model_version,
        )
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        _recent_predictions.append([float(body.n_steps)])
        if len(_recent_predictions) &gt; 1000:
            _recent_predictions.pop(0)

        return response
    except HTTPException:
        raise
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type=&quot;generation&quot;)
        logger.exception(&quot;Image animation failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Image animation failed&quot;) from e


@app.get(&quot;/stats&quot;, response_model=StatsResponse)
def get_stats():
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    info = _model.to_dict()
    return StatsResponse(
        model_id=info.get(&quot;model_id&quot;, &quot;unknown&quot;),
        img_size=info.get(&quot;img_size&quot;, DEFAULT_IMG_SIZE),
        n_frames=info.get(&quot;n_frames&quot;, DEFAULT_N_FRAMES),
        latent_dim=info.get(&quot;latent_dim&quot;, DEFAULT_LATENT_DIM),
        n_diffusion_steps=info.get(&quot;n_diffusion_steps&quot;, 1000),
        model_version=_model_version,
    )</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-4077587483')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-4077587483"><code class="language-bash">uv run python -m video_generation.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>
<div class="related-links">
<h3>Related Apps</h3>
<ul><li><a href="../code-generation/README.md">code-generation</a></li>
<li><a href="../image-generation/README.md">image-generation</a></li>
<li><a href="../retrieval-augmented-generation/README.md">retrieval-augmented-generation</a></li>
<li><a href="../text-generation/README.md">text-generation</a></li></ul>
</div>
</main>
<footer class="app-footer">
<p>Generated documentation for <strong>video-generation</strong></p>
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