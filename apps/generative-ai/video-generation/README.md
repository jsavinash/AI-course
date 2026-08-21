# video-generation

## ∫ Mathematics & Theory

Video Generation — Underlying equations and derivations

$$P(x_{1:T}) = \prod_{t=1}^{T} P(x_t | x_{

$$\mathcal{L} = -\sum_{t=1}^{T} \log P(x_t | x_{

$$\text{SSIM}(x, \hat{x}) = \frac{(2\mu_x \mu_{\hat{x}} + c_1)(2\sigma_{x\hat{x}} + c_2)}{(\mu_x^2 + \mu_{\hat{x}}^2 + c_1)(\sigma_x^2 + \sigma_{\hat{x}}^2 + c_2)}$$

### Step-by-Step Derivation

Video generation extends sequence modeling to spatiotemporal data. 3D convolutions or factored spatial-temporal attention capture motion. Temporal consistency is enforced via warping or predictive coding. Frame-wise perceptual losses improve visual quality.

### Interactive Visualization

Interactive frame-by-frame playback with generated vs real overlay; optical flow visualization; temporal consistency score.

## ⚙ Architecture

Model structure, data flow, and layer breakdown

### Class Hierarchy

```
  VideoTokenizer
  MultiHeadAttention
  AddNorm
  FeedForward
  TransformerBlock
  TextConditioning
  LatentVideoEncoder
  SpatiotemporalDiffusionModel
  VideoGenerationModel
```

### Data Flow

```mermaid
graph TD
  A[Input Data] --> B[Preprocessing]
  B --> C[Model Training]
  C --> D[Evaluation]
  D --> E[Model Registry]
  E --> F[Serving API]
```

## ⚡ API Reference

FastAPI endpoints and model interfaces

| Method | Endpoint |
| --- | --- |
| `GET` | `/` |
| `GET` | `/health` |
| `GET` | `/metrics` |

## ▶ Usage

Code examples and CLI commands

### Training Script

```python
"""Training pipeline for Video Generation."""

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
    model_id: str = "video-generation-v1",
    n_diffusion_steps: int = 1000,
    model_version: str = "1.0.0",
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -> dict:
    logger.info("Loading video dataset", n_samples=n_samples, n_frames=n_frames)
    videos, prompts = load_video_dataset(data_path=data_path, n_samples=n_samples, random_seed=random_seed)

    X_train, X_test, prompts_train, prompts_test = train_test_split_videos(videos, prompts, test_size=0.2, random_seed=random_seed)
    logger.info("Data split", n_train=len(X_train), n_test=len(X_test))

    model_dir.mkdir(parents=True, exist_ok=True)
    save_dataset(videos, prompts, model_dir / "training_data.npz")

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
    logger.info("Training finished", metrics=metrics)

    eval_metrics = model.evaluate(X_test_flat, np.zeros(len(X_test_flat)))
    logger.info("Evaluation metrics", metrics=eval_metrics)

    model_path = model_dir / f"video_generation_v{model_version}.npz"
    model.save(str(model_path))

    combined_metrics = {**metrics, **eval_metrics}
    combined_metrics.update({
        "img_size": float(img_size),
        "n_frames": float(n_frames),
        "latent_dim": float(latent_dim),
        "n_diffusion_steps": float(n_diffusion_steps),
        "n_samples": float(n_samples),
    })

    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="video-generation",
        model_version=model_version,
        model_type="generative",
        metrics=combined_metrics,
        parameters={
            "model_id": model_id,
            "img_size": img_size,
            "n_frames": n_frames,
            "latent_dim": latent_dim,
            "n_diffusion_steps": n_diffusion_steps,
            "n_samples": n_samples,
            "random_seed": random_seed,
        },
        artifacts={f"video_generation_v{model_version}.npz": model_path, "training_data.npz": model_dir / "training_data.npz"},
        tags={"framework": "numpy", "task": "video_generation", "model_type": "VideoGeneration"},
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="video-generation",
            model_version=model_version,
            metrics=combined_metrics,
            params={"model_id": model_id, "img_size": img_size, "n_frames": n_frames, "latent_dim": latent_dim, "n_diffusion_steps": n_diffusion_steps, "n_samples": n_samples},
            artifacts={"model": str(model_path)},
            tags={"model_type": "video_generation", "framework": "numpy"},
        )

    return combined_metrics

def main():
    parser = argparse.ArgumentParser(description="Train Video Generation Model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--n-samples", type=int, default=int(os.getenv("N_SAMPLES", "200")))
    parser.add_argument("--img-size", type=int, default=int(os.getenv("IMG_SIZE", "32")))
    parser.add_argument("--n-frames", type=int, default=int(os.getenv("N_FRAMES", "8")))
    parser.add_argument("--latent-dim", type=int, default=int(os.getenv("LATENT_DIM", "64")))
    parser.add_argument("--model-id", type=str, default=os.getenv("MODEL_ID", "video-generation-v1"))
    parser.add_argument("--n-diffusion-steps", type=int, default=int(os.getenv("N_DIFFUSION_STEPS", "1000")))
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument("--register-mlflow", action="store_true", default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true")
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
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
    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))

if __name__ == "__main__":
    main()
```

### API Server

```python
"""Serving API for Video Generation."""

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

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("VIDEO_GENERATION_METRICS_PORT", "9026"))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))

class GenerateVideoRequest(BaseModel):
    prompt: str = Field(..., min_length=1)
    n_steps: int = Field(default=50, ge=1, le=200)
    mode: str = Field(default="text-to-video", pattern="^(text-to-video|image-to-video)$")

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
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[float]] = []

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _model, _model_version, _metrics, _drift_detector, _reference_data

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("video_generation", port=METRICS_PORT)
    app.state.metrics = _metrics

    feature_names = [f"latent_{i}" for i in range(DEFAULT_LATENT_DIM)]
    _drift_detector = DriftDetector(
        feature_names=feature_names,
        feature_types={f: "float" for f in feature_names},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="video-generation",
        model_version=_model_version,
        model_type="generative",
    )

    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="video-generation", version=_model_version)

    yield
    logger.info("Shutting down video-generation API")

def _load_model() -> tuple[VideoGenerationModel, str]:
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            vg_models = [m for m in models if m.get("model_name") == "video-generation"]
            if vg_models:
                vg_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = vg_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("video_generation_v*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return VideoGenerationModel.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "video-generation" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("video_generation_v*.npz")) + list(model_dir.glob("*.npz"))
                if npz_files:
                    return VideoGenerationModel.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    npz_path = MODEL_DIR / "video_generation.npz"
    if npz_path.exists():
        return VideoGenerationModel.load(str(npz_path)), "legacy"

    candidate_paths = [
        Path("/app/artifacts/models/video_generation_v1.0.0.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "video_generation_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return VideoGenerationModel.load(str(p)), "1.0.0-bundled"

    logger.warning("No pre-existing model found. Initializing baseline model.")
    model = VideoGenerationModel(model_id="baseline", img_size=DEFAULT_IMG_SIZE, n_frames=DEFAULT_N_FRAMES, latent_dim=DEFAULT_LATENT_DIM)
    model._init()
    return model, "1.0.0-baseline"

def _load_reference_data() -> np.ndarray | None:
    from video_generation.data import generate_synthetic_videos
    videos, _ = generate_synthetic_videos(n_samples=100, random_seed=42)
    return videos.reshape(100, -1).astype(float)

app = FastAPI(
    title="Video Generation API",
    description="Spatiotemporal diffusion model for text-to-video and image-to-video generation",
    version="1.0.0",
    lifespan=lifespan,
)

add_observability_middleware(app)

@app.get("/")
def read_root():
    return {
        "service": "video-generation-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "endpoints": {
            "health": "/health",
            "generate": "POST /generate",
            "animate": "POST /animate",
            "stats": "GET /stats",
            "metrics": "/metrics",
        },
    }

@app.get("/health")
def health_check():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": _model_version,
        "model_id": _model.model_id if _model else "unknown",
    }

@app.get("/metrics")
def metrics():
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/generate", response_model=GenerateVideoResponse)
def generate_video(body: GenerateVideoRequest):
    """Generate a video from a text prompt."""
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

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
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return response
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="generation")
        logger.exception("Video generation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Video generation failed") from e

@app.post("/animate", response_model=AnimateImageResponse)
def animate_image(body: AnimateImageRequest):
    """Animate a static image using a motion prompt (image-to-video)."""
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        expected_size = DEFAULT_IMG_SIZE * DEFAULT_IMG_SIZE * 3
        if len(body.image_data) != expected_size:
            raise HTTPException(status_code=400, detail=f"image_data must have {expected_size} elements for {DEFAULT_IMG_SIZE}x{DEFAULT_IMG_SIZE} RGB image")
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
        if len(_recent_predictions) > 1000:
            _recent_predictions.pop(0)

        return response
    except HTTPException:
        raise
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="generation")
        logger.exception("Image animation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Image animation failed") from e

@app.get("/stats", response_model=StatsResponse)
def get_stats():
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    info = _model.to_dict()
    return StatsResponse(
        model_id=info.get("model_id", "unknown"),
        img_size=info.get("img_size", DEFAULT_IMG_SIZE),
        n_frames=info.get("n_frames", DEFAULT_N_FRAMES),
        latent_dim=info.get("latent_dim", DEFAULT_LATENT_DIM),
        n_diffusion_steps=info.get("n_diffusion_steps", 1000),
        model_version=_model_version,
    )
```

### CLI Commands

```bash
uv run python -m video_generation.train --model-dir ./artifacts/models
```

## 📊 Benchmarks

Test results and performance metrics

Run `pytest tests/test_models.py` and `pytest tests/test_apis.py` for detailed metrics.

### Related Apps

- [code-generation](../code-generation/README.md)

- [image-generation](../image-generation/README.md)

- [retrieval-augmented-generation](../retrieval-augmented-generation/README.md)

- [text-generation](../text-generation/README.md)

Generated documentation for **video-generation**
