<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>robot-maze-navigation - AI App Documentation</title>
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
<p class="section-subtitle">Reinforcement Learning (Q-Learning) — Underlying equations and derivations</p>
<div class="math-content">
<div class="equations"><div class="math-block">$$G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}$$</div>
<div class="math-block">$$Q^\pi(s, a) = \mathbb{E}_\pi \left[ \sum_{k=0}^{\infty} \gamma^k R_{t+k+1} \bigg| S_t=s, A_t=a \right]$$</div>
<div class="math-block">$$Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right]$$</div>
<div class="math-block">$$\pi^*(a|s) = \arg\max_{a} Q^*(s, a)$$</div></div>
<div class="derivation">
<h3>Step-by-Step Derivation</h3>
<p>RL agents learn by interacting with an environment. The return $G_t$ is the discounted sum of future rewards. The Bellman equation decomposes $Q^\pi$ into immediate reward plus discounted future value. Q-learning updates action-values toward the Bellman optimality target.</p>
</div>
<div class="viz-desc">
<h3>Interactive Visualization</h3>
<p>Interactive grid world with agent path; Q-value heatmap; episode reward curves; epsilon-greedy action distribution.</p>
</div>
</div>
</section>
<section id="architecture" class="section arch-section">
<h2><span class="section-icon">⚙</span> Architecture</h2>
<p class="section-subtitle">Model structure, data flow, and layer breakdown</p>
<div class="arch-diagram">
<h3>Class Hierarchy</h3>
<pre class="ascii-diagram">  QLearningAgent</pre>
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
<button class="copy-btn" onclick="copyCode('code-2697970622')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2697970622"><code class="language-python">&quot;&quot;&quot;Production training pipeline for robot maze navigation (Q-Learning).&quot;&quot;&quot;

import argparse
import os
from pathlib import Path

import numpy as np
from ai_core.logging import get_logger, setup_logging
from ai_core.model_registry import ModelRegistry

from robot_maze_navigation.data import (
    generate_maze,
    get_goal_positions,
    get_start_position,
    load_training_data,
    save_training_data,
)
from robot_maze_navigation.model import QLearningAgent

logger = get_logger(__name__)


def train(
    model_dir: Path,
    data_path: Path,
    maze_size: int,
    n_episodes: int,
    max_steps: int,
    learning_rate: float,
    discount_factor: float,
    epsilon_decay: float,
    mode: str,
    model_version: str,
    register_to_mlflow: bool = False,
    random_seed: int = 42,
) -&gt; dict:
    &quot;&quot;&quot;Train the robot maze navigation Q-learning model.

    Args:
        model_dir: Directory to save model artifacts
        data_path: Optional path to CSV transition data
        maze_size: Size of the square maze
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        learning_rate: Q-learning learning rate (alpha)
        discount_factor: Future reward discount (gamma)
        epsilon_decay: Exploration rate decay per episode
        mode: &quot;online&quot; or &quot;offline&quot;
        model_version: Model version string
        register_to_mlflow: Whether to register to MLflow
        random_seed: Random seed

    Returns:
        Dictionary with training metrics
    &quot;&quot;&quot;
    # Generate maze
    maze = generate_maze(maze_size, maze_size, random_seed)
    n_states = maze.shape[0] * maze.shape[1]
    goal_positions = get_goal_positions(maze)
    start = get_start_position(maze)

    logger.info(
        &quot;Generated maze&quot;,
        maze_size=maze_size,
        n_states=n_states,
        start=start,
        goals=goal_positions,
        n_walls=int(np.sum(maze)),
    )

    # Create agent
    agent = QLearningAgent(
        n_states=n_states,
        n_actions=4,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon_decay=epsilon_decay,
        mode=mode,
        seed=random_seed,
    )

    if mode == &quot;online&quot;:
        # Online RL: agent learns by interacting with environment
        logger.info(&quot;Starting online RL training&quot;, n_episodes=n_episodes)

        def env_func():
            return start, goal_positions, maze

        train_metrics = agent.train_online(env_func, n_episodes=n_episodes, max_steps=max_steps)
    else:
        # Offline RL: agent learns from fixed dataset
        logger.info(&quot;Starting offline RL training&quot;, n_episodes=n_episodes)

        states, actions, rewards, next_states, dones = load_training_data(
            data_path, maze_size=maze_size, n_transitions=n_episodes * max_steps, seed=random_seed
        )
        logger.info(&quot;Loaded offline dataset&quot;, n_transitions=len(states))

        save_training_data(
            states, actions, rewards, next_states, dones, model_dir / &quot;training_data.npz&quot;
        )
        train_metrics = agent.train_offline(
            states,
            actions,
            rewards,
            next_states,
            dones,
            n_epochs=max(1, n_episodes // 100),
            cols=maze_size,
        )

    logger.info(&quot;Training complete&quot;, **train_metrics)

    # Evaluate learned policy
    eval_metrics = agent.evaluate(maze, n_episodes=100, max_steps=max_steps)
    logger.info(&quot;Evaluation metrics&quot;, **eval_metrics)

    # Model validation
    if eval_metrics[&quot;success_rate&quot;] &lt; 0.5:
        logger.warning(
            &quot;Policy success rate below threshold&quot;,
            success_rate=eval_metrics[&quot;success_rate&quot;],
            threshold=0.5,
        )

    # Save model
    model_path = model_dir / f&quot;robot_maze_model_v{model_version}.npz&quot;
    agent.save(str(model_path))

    # Save maze visualization
    _save_chart(agent, maze, model_dir, model_version)

    # Combined metrics
    training_metrics = {
        **train_metrics,
        **eval_metrics,
        &quot;maze_size&quot;: float(maze_size),
        &quot;n_states&quot;: float(n_states),
        &quot;n_walls&quot;: float(int(np.sum(maze))),
    }

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name=&quot;robot-maze-navigation&quot;,
        model_version=model_version,
        model_type=&quot;reinforcement_learning&quot;,
        metrics=training_metrics,
        parameters={
            &quot;maze_size&quot;: maze_size,
            &quot;n_episodes&quot;: n_episodes,
            &quot;max_steps&quot;: max_steps,
            &quot;learning_rate&quot;: learning_rate,
            &quot;discount_factor&quot;: discount_factor,
            &quot;epsilon_decay&quot;: epsilon_decay,
            &quot;mode&quot;: mode,
            &quot;random_seed&quot;: random_seed,
        },
        artifacts={
            f&quot;robot_maze_model_v{model_version}.npz&quot;: model_path,
            &quot;training_data.npz&quot;: model_dir / &quot;training_data.npz&quot;,
        },
        tags={
            &quot;framework&quot;: &quot;numpy&quot;,
            &quot;task&quot;: &quot;reinforcement_learning&quot;,
            &quot;method&quot;: &quot;q_learning&quot;,
            &quot;mode&quot;: mode,
        },
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name=&quot;robot-maze-navigation&quot;,
            model_version=model_version,
            metrics=training_metrics,
            params={
                &quot;maze_size&quot;: maze_size,
                &quot;n_episodes&quot;: n_episodes,
                &quot;max_steps&quot;: max_steps,
                &quot;learning_rate&quot;: learning_rate,
                &quot;discount_factor&quot;: discount_factor,
                &quot;epsilon_decay&quot;: epsilon_decay,
                &quot;mode&quot;: mode,
                &quot;random_seed&quot;: random_seed,
            },
            artifacts={
                &quot;model&quot;: str(model_path),
                &quot;chart&quot;: str(model_dir / f&quot;robot_maze_v{model_version}.png&quot;),
                &quot;training_data&quot;: str(model_dir / &quot;training_data.npz&quot;),
            },
            tags={
                &quot;model_type&quot;: &quot;reinforcement_learning&quot;,
                &quot;framework&quot;: &quot;numpy&quot;,
                &quot;method&quot;: &quot;q_learning&quot;,
                &quot;mode&quot;: mode,
            },
        )
        logger.info(
            &quot;Registered model to MLflow&quot;, model=&quot;robot-maze-navigation&quot;, version=model_version
        )

    return training_metrics


def _save_chart(agent: QLearningAgent, maze: np.ndarray, output_dir: Path, version: str) -&gt; None:
    &quot;&quot;&quot;Save the maze solution visualization and learning curves.&quot;&quot;&quot;
    import matplotlib

    matplotlib.use(&quot;Agg&quot;)
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Maze with solution path
    ax1 = axes[0]
    ax1.imshow(maze, cmap=&quot;binary&quot;, interpolation=&quot;none&quot;)
    path, success, steps = agent.solve_maze(maze)
    if len(path) &gt; 1:
        path_rows = [p[0] for p in path]
        path_cols = [p[1] for p in path]
        ax1.plot(path_cols, path_rows, &quot;b-&quot;, linewidth=2, alpha=0.7, label=&quot;Path&quot;)
    ax1.scatter(
        [get_start_position(maze)[1]],
        [get_start_position(maze)[0]],
        c=&quot;green&quot;,
        s=200,
        marker=&quot;o&quot;,
        label=&quot;Start&quot;,
    )
    goals = get_goal_positions(maze)
    ax1.scatter(
        [g[1] for g in goals],
        [g[0] for g in goals],
        c=&quot;red&quot;,
        s=200,
        marker=&quot;*&quot;,
        label=&quot;Goal&quot;,
    )
    ax1.set_title(f&quot;Maze Solution - v{version} (success={success}, steps={steps})&quot;)
    ax1.legend()
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Plot 2: Learning curve (episode rewards)
    ax2 = axes[1]
    if agent.episode_rewards:
        window = min(50, len(agent.episode_rewards) // 10)
        if window &gt; 0:
            moving_avg = np.convolve(agent.episode_rewards, np.ones(window) / window, mode=&quot;valid&quot;)
            ax2.plot(agent.episode_rewards, alpha=0.3, color=&quot;steelblue&quot;, label=&quot;Episode reward&quot;)
            ax2.plot(
                range(window - 1, len(agent.episode_rewards)),
                moving_avg,
                color=&quot;red&quot;,
                label=f&quot;MA({window})&quot;,
            )
    ax2.set_xlabel(&quot;Episode&quot;)
    ax2.set_ylabel(&quot;Reward&quot;)
    ax2.set_title(&quot;Learning Curve&quot;)
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Q-value heatmap (max Q per state)
    ax3 = axes[2]
    if agent.q_table is not None:
        q_max = np.max(agent.q_table, axis=1).reshape(maze.shape)
        q_max_masked = np.where(maze == 1, np.nan, q_max)
        im = ax3.imshow(q_max_masked, cmap=&quot;hot&quot;, interpolation=&quot;none&quot;)
        plt.colorbar(im, ax=ax3, label=&quot;Max Q-value&quot;)
        ax3.set_title(&quot;Max Q-value per State&quot;)
        ax3.set_xticks([])
        ax3.set_yticks([])

    plt.tight_layout()
    chart_path = output_dir / f&quot;robot_maze_v{version}.png&quot;
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info(&quot;Chart saved&quot;, path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description=&quot;Train robot maze navigation Q-learning model&quot;)
    parser.add_argument(&quot;--model-dir&quot;, type=Path, default=Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;)))
    parser.add_argument(&quot;--data-path&quot;, type=Path, default=None)
    parser.add_argument(&quot;--maze-size&quot;, type=int, default=int(os.getenv(&quot;MAZE_SIZE&quot;, &quot;8&quot;)))
    parser.add_argument(&quot;--n-episodes&quot;, type=int, default=int(os.getenv(&quot;N_EPISODES&quot;, &quot;500&quot;)))
    parser.add_argument(&quot;--max-steps&quot;, type=int, default=int(os.getenv(&quot;MAX_STEPS&quot;, &quot;200&quot;)))
    parser.add_argument(
        &quot;--learning-rate&quot;, type=float, default=float(os.getenv(&quot;LEARNING_RATE&quot;, &quot;0.1&quot;))
    )
    parser.add_argument(
        &quot;--discount-factor&quot;, type=float, default=float(os.getenv(&quot;DISCOUNT_FACTOR&quot;, &quot;0.99&quot;))
    )
    parser.add_argument(
        &quot;--epsilon-decay&quot;, type=float, default=float(os.getenv(&quot;EPSILON_DECAY&quot;, &quot;0.995&quot;))
    )
    parser.add_argument(
        &quot;--mode&quot;, type=str, default=os.getenv(&quot;RL_MODE&quot;, &quot;online&quot;), choices=[&quot;online&quot;, &quot;offline&quot;]
    )
    parser.add_argument(&quot;--model-version&quot;, type=str, default=os.getenv(&quot;MODEL_VERSION&quot;, &quot;1.0.0&quot;))
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
        maze_size=args.maze_size,
        n_episodes=args.n_episodes,
        max_steps=args.max_steps,
        learning_rate=args.learning_rate,
        discount_factor=args.discount_factor,
        epsilon_decay=args.epsilon_decay,
        mode=args.mode,
        model_version=args.model_version,
        register_to_mlflow=args.register_mlflow,
        random_seed=args.random_seed,
    )

    logger.info(&quot;Training finished&quot;, metrics=metrics, model_dir=str(args.model_dir))


if __name__ == &quot;__main__&quot;:
    main()</code></pre>
</div><h3>API Server</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-2526109603')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-2526109603"><code class="language-python">&quot;&quot;&quot;Production serving API for robot maze navigation (Q-Learning).&quot;&quot;&quot;

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
MODEL_DIR = Path(os.getenv(&quot;MODEL_DIR&quot;, &quot;/models&quot;))
MODEL_VERSION = os.getenv(&quot;MODEL_VERSION&quot;, &quot;latest&quot;)
METRICS_PORT = int(os.getenv(&quot;METRICS_PORT&quot;, os.getenv(&quot;ROBOT_MAZE_METRICS_PORT&quot;, &quot;8006&quot;)))
DRIFT_THRESHOLD = float(os.getenv(&quot;DRIFT_THRESHOLD&quot;, &quot;0.2&quot;))
MAZE_SIZE = int(os.getenv(&quot;MAZE_SIZE&quot;, &quot;8&quot;))


class SolveRequest(BaseModel):
    &quot;&quot;&quot;Request to solve a maze.&quot;&quot;&quot;

    maze_size: int = Field(8, ge=4, le=20, description=&quot;Size of the square maze&quot;)
    max_steps: int = Field(200, ge=10, le=1000, description=&quot;Maximum steps to solve&quot;)


class SolveResponse(BaseModel):
    &quot;&quot;&quot;Response with maze solution.&quot;&quot;&quot;

    path: list[list[int]]
    success: bool
    steps: int
    start: list[int]
    goals: list[list[int]]
    model_version: str


class StepRequest(BaseModel):
    &quot;&quot;&quot;Request to compute next action for a given state.&quot;&quot;&quot;

    row: int = Field(..., ge=0, description=&quot;Current row&quot;)
    col: int = Field(..., ge=0, description=&quot;Current column&quot;)
    maze_size: int = Field(8, ge=4, le=20, description=&quot;Maze size for state indexing&quot;)


class StepResponse(BaseModel):
    &quot;&quot;&quot;Response with recommended action.&quot;&quot;&quot;

    action: str
    action_code: int
    next_row: int
    next_col: int
    q_values: list[float]
    model_version: str


class TrainRequest(BaseModel):
    &quot;&quot;&quot;Request to trigger training.&quot;&quot;&quot;

    n_episodes: int = Field(100, ge=10, le=10000, description=&quot;Number of training episodes&quot;)
    mode: str = Field(&quot;online&quot;, description=&quot;Training mode: online or offline&quot;)


class TrainResponse(BaseModel):
    &quot;&quot;&quot;Response from training.&quot;&quot;&quot;

    metrics: dict
    model_version: str


class StatsResponse(BaseModel):
    &quot;&quot;&quot;Model statistics response.&quot;&quot;&quot;

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
    &quot;&quot;&quot;Drift detection response.&quot;&quot;&quot;

    total_features: int
    drifted_features: int
    drift_ratio: float
    drifted: list[dict]
    all_results: list[dict]


# Global model state
_model: QLearningAgent | None = None
_model_version: str = &quot;unknown&quot;
_metrics: MetricsCollector | None = None
_drift_detector: DriftDetector | None = None
_reference_data: np.ndarray | None = None
_recent_predictions: list[list[int]] = []
_maze_size: int = MAZE_SIZE


@asynccontextmanager
async def lifespan(app: FastAPI):
    &quot;&quot;&quot;Load model at startup and clean up at shutdown.&quot;&quot;&quot;
    global _model, _model_version, _metrics, _drift_detector, _reference_data, _maze_size

    setup_logging(os.getenv(&quot;LOG_LEVEL&quot;, &quot;INFO&quot;))
    _metrics = MetricsCollector(&quot;robot_maze&quot;, port=METRICS_PORT)
    app.state.metrics = _metrics

    _drift_detector = DriftDetector(
        feature_names=[&quot;row&quot;, &quot;col&quot;],
        feature_types={&quot;row&quot;: &quot;float&quot;, &quot;col&quot;: &quot;float&quot;},
        psi_threshold=DRIFT_THRESHOLD,
    )

    _model, _model_version = _load_model()
    _metrics.set_model_version(_model_version)
    _metrics.set_model_info(
        model_name=&quot;robot-maze-navigation&quot;,
        model_version=_model_version,
        model_type=&quot;reinforcement_learning&quot;,
    )

    # Load reference data for drift detection
    _reference_data = _load_reference_data()
    logger.info(&quot;Model loaded&quot;, model=&quot;robot-maze-navigation&quot;, version=_model_version)

    yield

    logger.info(&quot;Shutting down robot-maze-navigation API&quot;)


def _load_model() -&gt; tuple[QLearningAgent, str]:
    &quot;&quot;&quot;Load the latest model from the registry or model directory with resilient fallback.&quot;&quot;&quot;
    # 1. Try model registry
    registry = ModelRegistry(base_dir=MODEL_DIR)
    try:
        if MODEL_VERSION == &quot;latest&quot;:
            models = registry.list_models()
            rm_models = [m for m in models if m.get(&quot;model_name&quot;) == &quot;robot-maze-navigation&quot;]
            if rm_models:
                rm_models.sort(key=lambda m: m[&quot;model_version&quot;], reverse=True)
                latest = rm_models[0]
                model_dir = Path(latest[&quot;artifact_path&quot;])
                npz_files = list(model_dir.glob(&quot;robot_maze_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return QLearningAgent.load(str(npz_files[0])), latest[&quot;model_version&quot;]
        else:
            model_dir = MODEL_DIR / &quot;robot-maze-navigation&quot; / MODEL_VERSION
            if model_dir.exists():
                npz_files = list(model_dir.glob(&quot;robot_maze_model_*.npz&quot;)) + list(
                    model_dir.glob(&quot;*.npz&quot;)
                )
                if npz_files:
                    return QLearningAgent.load(str(npz_files[0])), MODEL_VERSION
    except Exception as e:
        logger.warning(f&quot;Registry lookup failed: {e}&quot;)

    # 2. Try direct model in MODEL_DIR
    npz_path = MODEL_DIR / &quot;robot_maze_model.npz&quot;
    if npz_path.exists():
        return QLearningAgent.load(str(npz_path)), &quot;legacy&quot;

    # 3. Try bundled artifacts directory
    candidate_paths = [
        Path(&quot;/app/artifacts/models/robot_maze_model_v1.0.0.npz&quot;),
        Path(__file__).resolve().parents[3]
        / &quot;artifacts&quot;
        / &quot;models&quot;
        / &quot;robot_maze_model_v1.0.0.npz&quot;,
    ]
    for p in candidate_paths:
        if p.exists():
            logger.info(&quot;Loading bundled baseline model&quot;, path=str(p))
            return QLearningAgent.load(str(p)), &quot;1.0.0-bundled&quot;

    # 4. In-memory baseline fallback (never crash cold start)
    logger.warning(&quot;No pre-existing model found on disk. Initializing baseline Q-learning model.&quot;)
    maze = generate_maze(_maze_size, _maze_size, 42)
    n_states = maze.shape[0] * maze.shape[1]
    model = QLearningAgent(n_states=n_states, n_actions=4, mode=&quot;online&quot;, seed=42)
    return model, &quot;1.0.0-baseline&quot;


def _load_reference_data() -&gt; np.ndarray | None:
    &quot;&quot;&quot;Load reference training data for drift detection.&quot;&quot;&quot;
    candidate_npzs = [
        MODEL_DIR / &quot;robot-maze-navigation&quot; / _model_version / &quot;training_data.npz&quot;,
        MODEL_DIR / &quot;training_data.npz&quot;,
        Path(&quot;/app/artifacts/models/training_data.npz&quot;),
        Path(__file__).resolve().parents[3] / &quot;artifacts&quot; / &quot;models&quot; / &quot;training_data.npz&quot;,
    ]
    for npz_path in candidate_npzs:
        if npz_path.exists():
            try:
                data = np.load(npz_path)
                if &quot;states&quot; in data:
                    return data[&quot;states&quot;]
            except Exception as e:
                logger.warning(&quot;Could not read reference npz&quot;, path=str(npz_path), error=str(e))

    # Generate reference data
    maze = generate_maze(_maze_size, _maze_size, 42)
    from robot_maze_navigation.data import generate_transitions

    states, _, _, _, _ = generate_transitions(maze, 1000, 42)
    return states


# Create FastAPI app
app = FastAPI(
    title=&quot;Robot Maze Navigation API&quot;,
    description=&quot;Q-Learning Reinforcement Learning for robot maze navigation&quot;,
    version=&quot;1.0.0&quot;,
    lifespan=lifespan,
)

# Add observability middleware
add_observability_middleware(app)


@app.get(&quot;/&quot;)
def read_root():
    &quot;&quot;&quot;Service information.&quot;&quot;&quot;
    return {
        &quot;service&quot;: &quot;robot-maze-navigation-api&quot;,
        &quot;version&quot;: &quot;1.0.0&quot;,
        &quot;model_version&quot;: _model_version,
        &quot;mode&quot;: _model.mode if _model else &quot;unknown&quot;,
        &quot;endpoints&quot;: {
            &quot;health&quot;: &quot;/health&quot;,
            &quot;solve&quot;: &quot;POST /solve&quot;,
            &quot;step&quot;: &quot;POST /step&quot;,
            &quot;train&quot;: &quot;POST /train&quot;,
            &quot;stats&quot;: &quot;GET /stats&quot;,
            &quot;drift&quot;: &quot;GET /drift&quot;,
            &quot;metrics&quot;: &quot;/metrics&quot;,
        },
    }


@app.get(&quot;/health&quot;)
def health_check():
    &quot;&quot;&quot;Kubernetes liveness/readiness probe.&quot;&quot;&quot;
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)
    return {
        &quot;status&quot;: &quot;healthy&quot;,
        &quot;model_loaded&quot;: True,
        &quot;model_version&quot;: _model_version,
        &quot;mode&quot;: _model.mode,
    }


@app.get(&quot;/metrics&quot;)
def metrics():
    &quot;&quot;&quot;Prometheus metrics endpoint.&quot;&quot;&quot;
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post(&quot;/reload&quot;)
def reload_model():
    &quot;&quot;&quot;Dynamically reload the model from disk/registry.&quot;&quot;&quot;
    global _model, _model_version, _reference_data
    try:
        _model, _model_version = _load_model()
        if _metrics:
            _metrics.set_model_version(_model_version)
            _metrics.set_model_info(
                model_name=&quot;robot-maze-navigation&quot;,
                model_version=_model_version,
                model_type=&quot;reinforcement_learning&quot;,
            )
        _reference_data = _load_reference_data()
        logger.info(
            &quot;Model reloaded dynamically&quot;, model=&quot;robot-maze-navigation&quot;, version=_model_version
        )
        return {&quot;status&quot;: &quot;reloaded&quot;, &quot;model_version&quot;: _model_version}
    except Exception as e:
        logger.exception(&quot;Model reload failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=f&quot;Reload failed: {e}&quot;) from e


@app.get(&quot;/drift&quot;, response_model=DriftResponse)
def drift_check():
    &quot;&quot;&quot;Check for data drift between reference and recent predictions.&quot;&quot;&quot;
    if _drift_detector is None or _reference_data is None:
        raise HTTPException(status_code=503, detail=&quot;Drift detection not available&quot;)

    if len(_recent_predictions) &lt; 10:
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
        _metrics.set_drift_ratio(summary[&quot;drift_ratio&quot;])

    return DriftResponse(**summary)


@app.get(&quot;/stats&quot;, response_model=StatsResponse)
def get_stats():
    &quot;&quot;&quot;Return model statistics.&quot;&quot;&quot;
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

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


@app.post(&quot;/solve&quot;, response_model=SolveResponse)
def solve_maze(body: SolveRequest):
    &quot;&quot;&quot;Solve the maze using the learned Q-table.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

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
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Maze solving failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Maze solving failed&quot;) from e


@app.post(&quot;/step&quot;, response_model=StepResponse)
def compute_step(body: StepRequest):
    &quot;&quot;&quot;Compute the next action for a given state.&quot;&quot;&quot;
    if _model is None or _metrics is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

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
        _metrics.record_error(model_version=_model_version, error_type=&quot;prediction&quot;)
        logger.exception(&quot;Step computation failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Step computation failed&quot;) from e


@app.post(&quot;/train&quot;, response_model=TrainResponse)
def trigger_training(body: TrainRequest):
    &quot;&quot;&quot;Trigger online or offline training.&quot;&quot;&quot;
    if _model is None:
        raise HTTPException(status_code=503, detail=&quot;Model not loaded&quot;)

    start = time.time()
    try:
        maze = generate_maze(_maze_size, _maze_size, 42)
        goal_positions = get_goal_positions(maze)
        start_pos = get_start_position(maze)

        if body.mode == &quot;online&quot;:

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
        _metrics.record_error(model_version=_model_version, error_type=&quot;training&quot;)
        logger.exception(&quot;Training failed&quot;, error=str(e))
        raise HTTPException(status_code=500, detail=&quot;Training failed&quot;) from e</code></pre>
</div>
<h3>CLI Commands</h3>
<div class="code-block-wrapper">
<button class="copy-btn" onclick="copyCode('code-75592677')" title="Copy to clipboard">&#x2398;</button>
<pre class="code-block" id="code-75592677"><code class="language-bash">uv run python -m robot_maze_navigation.train --model-dir ./artifacts/models</code></pre>
</div>
</section>
<section id="benchmarks" class="section bench-section">
<h2><span class="section-icon">📊</span> Benchmarks</h2>
<p class="section-subtitle">Test results and performance metrics</p>
<p class="muted">Run <code>pytest tests/test_models.py</code> and <code>pytest tests/test_apis.py</code> for detailed metrics.</p>
</section>

</main>
<footer class="app-footer">
<p>Generated documentation for <strong>robot-maze-navigation</strong></p>
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