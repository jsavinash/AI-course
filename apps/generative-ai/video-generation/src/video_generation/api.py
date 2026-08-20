"""Serving API for Video Generation."""

import os
import time
from contextlib import asynccontextmanager
from pathlib import Path

import numpy as np
from fastapi import FastAPI, HTTPException, Response
from mlops_shared.drift import DriftDetector
from mlops_shared.fastapi_middleware import add_observability_middleware
from mlops_shared.logging import get_logger, setup_logging
from mlops_shared.metrics import MetricsCollector
from mlops_shared.model_registry import ModelRegistry
from pydantic import BaseModel, Field

from video_generation.data import DEFAULT_IMG_SIZE, DEFAULT_LATENT_DIM, DEFAULT_N_FRAMES
from video_generation.model import VideoGenerationModel

logger = get_logger(__name__)

MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("VIDEO_GENERATION_METRICS_PORT", "8017"))
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
