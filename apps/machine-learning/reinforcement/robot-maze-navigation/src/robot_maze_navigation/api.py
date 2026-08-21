"""Production serving API for robot maze navigation (Q-Learning)."""

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

from robot_maze_navigation.data import (
    ACTION_NAMES,
    generate_maze,
    get_goal_positions,
    get_start_position,
)
from robot_maze_navigation.model import QLearningAgent

logger = get_logger(__name__)

# Configuration
MODEL_DIR = Path(os.getenv("MODEL_DIR", "/models"))
MODEL_VERSION = os.getenv("MODEL_VERSION", "latest")
METRICS_PORT = int(os.getenv("METRICS_PORT", os.getenv("ROBOT_MAZE_METRICS_PORT", "8006")))
DRIFT_THRESHOLD = float(os.getenv("DRIFT_THRESHOLD", "0.2"))
MAZE_SIZE = int(os.getenv("MAZE_SIZE", "8"))


class SolveRequest(BaseModel):
    """Request to solve a maze."""

    maze_size: int = Field(8, ge=4, le=20, description="Size of the square maze")
    max_steps: int = Field(200, ge=10, le=1000, description="Maximum steps to solve")


class SolveResponse(BaseModel):
    """Response with maze solution."""

    path: list[list[int]]
    success: bool
    steps: int
    start: list[int]
    goals: list[list[int]]
    model_version: str


class StepRequest(BaseModel):
    """Request to compute next action for a given state."""

    row: int = Field(..., ge=0, description="Current row")
    col: int = Field(..., ge=0, description="Current column")
    maze_size: int = Field(8, ge=4, le=20, description="Maze size for state indexing")


class StepResponse(BaseModel):
    """Response with recommended action."""

    action: str
    action_code: int
    next_row: int
    next_col: int
    q_values: list[float]
    model_version: str


class TrainRequest(BaseModel):
    """Request to trigger training."""

    n_episodes: int = Field(100, ge=10, le=10000, description="Number of training episodes")
    mode: str = Field("online", description="Training mode: online or offline")


class TrainResponse(BaseModel):
    """Response from training."""

    metrics: dict
    model_version: str


class StatsResponse(BaseModel):
    """Model statistics response."""

    n_states: int
    n_actions: int
    learning_rate: float
    discount_factor: float
    epsilon: float
    mode: str
    n_episodes_trained: int
    n_steps_total: int
    mean_training_error: float
    mean_episode_reward: float
    mean_episode_length: float
    model_version: str


class DriftResponse(BaseModel):
    """Drift detection response."""

    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


# Global model state
_model: QLearningAgent | None = None
_model_version: str = "unknown"
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[int]] = []
_maze_size: int = MAZE_SIZE


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model at startup and clean up at shutdown."""
    global _model, _model_version, _metrics, _drift_detector, _reference_data, _maze_size

    setup_logging(os.getenv("LOG_LEVEL", "INFO"))
    _metrics = MetricsCollector("robot_maze", port=METRICS_PORT)
    app.state.metrics = _metrics

    _drift_detector = DriftDetector(
        feature_names=["row", "col"],
        feature_types={"row": "float", "col": "float"},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name="robot-maze-navigation",
        model_version=_model_version,
        model_type="reinforcement_learning",
    )

    # Load reference data for drift detection
    _reference_data = _load_reference_data()
    logger.info("Model loaded", model="robot-maze-navigation", version=_model_version)

    yield

    logger.info("Shutting down robot-maze-navigation API")


def _load_model() -> tuple[QLearningAgent, str]:
    """Load the latest model from the registry or model directory with resilient fallback."""
    # 1. Try model registry
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == "latest":
            models = registry.list_models()
            rm_models = [m for m in models if m.get("model_name") == "robot-maze-navigation"]
            if rm_models:
                rm_models.sort(key=lambda m: m["model_version"], reverse=True)
                latest = rm_models[0]
                model_dir = Path(latest["artifact_path"])
                npz_files = list(model_dir.glob("robot_maze_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return QLearningAgent.load(str(npz_files[0])), latest["model_version"]
        else:
            model_dir = MODEL_DIR / "robot-maze-navigation" / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob("robot_maze_model_*.npz")) + list(
                    model_dir.glob("*.npz")
                )
                if npz_files:
                    return QLearningAgent.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f"Registry lookup failed: {e}")

    # 2. Try direct model in MODEL_DIR
    npz_path = MODEL_DIR / "robot_maze_model.npz"
    if npz_path.exists():
        return QLearningAgent.load(str(npz_path)), "legacy"

    # 3. Try bundled artifacts directory
    candidate_paths = [
        Path("/app/artifacts/models/robot_maze_model_v1.0.0.npz"),
        Path(__file__).resolve().parents[3]
        / "artifacts"
        / "models"
        / "robot_maze_model_v1.0.0.npz",
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info("Loading bundled baseline model", path=str(p))
            return QLearningAgent.load(str(p)), "1.0.0-bundled"

    # 4. In-memory baseline fallback (never crash cold start)
    logger.warning("No pre-existing model found on disk. Initializing baseline Q-learning model.")
    maze = generate_maze(_maze_size, _maze_size, 42)
    n_states = maze.shape[0] * maze.shape[1]
    model = QLearningAgent(n_states=n_states, n_actions=4, mode="online", seed=42)
    return model, "1.0.0-baseline"


def _load_reference_data() -> np.ndarray | None:
    """Load reference training data for drift detection."""
    candidate_npzs = [
        MODEL_DIR / "robot-maze-navigation" / _model_version / "training_data.npz",
        MODEL_DIR / "training_data.npz",
        Path("/app/artifacts/models/training_data.npz"),
        Path(__file__).resolve().parents[3] / "artifacts" / "models" / "training_data.npz",
    ]
    for npz_path in candidate_npzs:
        if npz_path.exists():
            try:
                data = np.load(npz_path)
                if "states" in data:
                    return data["states"]
            except Exception as e:
                logger.warning("Could not read reference npz", path=str(npz_path), error=str(e))

    # Generate reference data
    maze = generate_maze(_maze_size, _maze_size, 42)
    from robot_maze_navigation.data import generate_transitions

    states, _, _, _, _ = generate_transitions(maze, 1000, 42)
    return states


# Create FastAPI app
app = FastAPI(
    title="Robot Maze Navigation API",
    description="Q-Learning Reinforcement Learning for robot maze navigation",
    version="1.0.0",
    lifespan=lifespan,
)

# Add observability middleware
add_observability_middleware(app)


@app.get("/")
def read_root():
    """Service information."""
    return {
        "service": "robot-maze-navigation-api",
        "version": "1.0.0",
        "model_version": _model_version,
        "mode": _model.mode if _model else "unknown",
        "endpoints": {
            "health": "/health",
            "solve": "POST /solve",
            "step": "POST /step",
            "train": "POST /train",
            "stats": "GET /stats",
            "drift": "GET /drift",
            "metrics": "/metrics",
        },
    }


@app.get("/health")
def health_check():
    """Kubernetes liveness/readiness probe."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {
        "status": "healthy",
        "model_loaded": True,
        "model_version": _model_version,
        "mode": _model.mode,
    }


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post("/reload")
def reload_model():
    """Dynamically reload the model from disk/registry."""
    global _model, _model_version, _reference_data
    try:
        _model, _model_version = _load_model()
        if _metrics:
            _metrics.set_model_version(_model_version)
            _metrics.set_model_info(
                model_name="robot-maze-navigation",
                model_version=_model_version,
                model_type="reinforcement_learning",
            )
        _reference_data = _load_reference_data()
        logger.info(
            "Model reloaded dynamically", model="robot-maze-navigation", version=_model_version
        )
        return {"status": "reloaded", "model_version": _model_version}
    except Exception as e:
        logger.exception("Model reload failed", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reload failed: {e}") from e


@app.get("/drift", response_model=DriftResponse)
def drift_check():
    """Check for data drift between reference and recent predictions."""
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail="Drift detection not available")

    if len(_recent_predictions) < 10:
        return DriftResponse(
            total_features=2,
            drifted_features=0,
            drift_ratio=0.0,
            drifted=[],
            all_results=[],
        )

    current = np.array(_recent_predictions[-100:])
    results = _drift_detector.detect_drift(_reference_data, current)
    summary = _drift_detector.summarize(results)

    if _metrics:
        _metrics.set_drift_ratio(summary["drift_ratio"])

    return DriftResponse(**summary)


@app.get("/stats", response_model=StatsResponse)
def get_stats():
    """Return model statistics."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    return StatsResponse(
        n_states=_model.n_states,
        n_actions=_model.n_actions,
        learning_rate=_model.learning_rate,
        discount_factor=_model.discount_factor,
        epsilon=_model.epsilon,
        mode=_model.mode,
        n_episodes_trained=_model.n_episodes_trained,
        n_steps_total=_model.n_steps_total,
        mean_training_error=float(np.mean(_model.training_errors))
        if _model.training_errors
        else 0.0,
        mean_episode_reward=float(np.mean(_model.episode_rewards))
        if _model.episode_rewards
        else 0.0,
        mean_episode_length=float(np.mean(_model.episode_lengths))
        if _model.episode_lengths
        else 0.0,
        model_version=_model_version,
    )


@app.post("/solve", response_model=SolveResponse)
def solve_maze(body: SolveRequest):
    """Solve the maze using the learned Q-table."""
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        maze = generate_maze(body.maze_size, body.maze_size, seed=42)
        path, success, steps = _model.solve_maze(maze, max_steps=body.max_steps)
        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        return SolveResponse(
            path=[[int(r), int(c)] for r, c in path],
            success=success,
            steps=steps,
            start=[int(get_start_position(maze)[0]), int(get_start_position(maze)[1])],
            goals=[[int(r), int(c)] for r, c in get_goal_positions(maze)],
            model_version=_model_version,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Maze solving failed", error=str(e))
        raise HTTPException(status_code=500, detail="Maze solving failed") from e


@app.post("/step", response_model=StepResponse)
def compute_step(body: StepRequest):
    """Compute the next action for a given state."""
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        state_idx = body.row * body.maze_size + body.col
        action = _model.get_action(state_idx, training=False)
        q_values = _model.q_table[state_idx].tolist() if _model.q_table is not None else [0.0] * 4

        dr, dc = [(0, -1), (0, 1), (-1, 0), (1, 0)][action]
        next_row = body.row + dr
        next_col = body.col + dc

        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        return StepResponse(
            action=ACTION_NAMES[action],
            action_code=action,
            next_row=next_row,
            next_col=next_col,
            q_values=[round(float(q), 4) for q in q_values],
            model_version=_model_version,
        )
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="prediction")
        logger.exception("Step computation failed", error=str(e))
        raise HTTPException(status_code=500, detail="Step computation failed") from e


@app.post("/train", response_model=TrainResponse)
def trigger_training(body: TrainRequest):
    """Trigger online or offline training."""
    if _model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    start = time.time()
    try:
        maze = generate_maze(_maze_size, _maze_size, 42)
        goal_positions = get_goal_positions(maze)
        start_pos = get_start_position(maze)

        if body.mode == "online":

            def env_func():
                return start_pos, goal_positions, maze

            metrics = _model.train_online(env_func, n_episodes=body.n_episodes, max_steps=200)
        else:
            from robot_maze_navigation.data import load_training_data

            states, actions, rewards, next_states, dones = load_training_data(
                None, maze_size=_maze_size, n_transitions=body.n_episodes * 20, seed=42
            )
            metrics = _model.train_offline(
                states,
                actions,
                rewards,
                next_states,
                dones,
                n_epochs=max(1, body.n_episodes // 100),
                cols=_maze_size,
            )

        duration = time.time() - start
        _metrics.record_prediction(model_version=_model_version, duration=duration)

        return TrainResponse(metrics=metrics, model_version=_model_version)
    except Exception as e:
        _metrics.record_error(model_version=_model_version, error_type="training")
        logger.exception("Training failed", error=str(e))
        raise HTTPException(status_code=500, detail="Training failed") from e
