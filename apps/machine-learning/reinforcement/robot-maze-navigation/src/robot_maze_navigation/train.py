"""Production training pipeline for robot maze navigation (Q-Learning)."""

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
) -> dict:
    """Train the robot maze navigation Q-learning model.

    Args:
        model_dir: Directory to save model artifacts
        data_path: Optional path to CSV transition data
        maze_size: Size of the square maze
        n_episodes: Number of training episodes
        max_steps: Maximum steps per episode
        learning_rate: Q-learning learning rate (alpha)
        discount_factor: Future reward discount (gamma)
        epsilon_decay: Exploration rate decay per episode
        mode: "online" or "offline"
        model_version: Model version string
        register_to_mlflow: Whether to register to MLflow
        random_seed: Random seed

    Returns:
        Dictionary with training metrics
    """
    # Generate maze
    maze = generate_maze(maze_size, maze_size, random_seed)
    n_states = maze.shape[0] * maze.shape[1]
    goal_positions = get_goal_positions(maze)
    start = get_start_position(maze)

    logger.info(
        "Generated maze",
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

    if mode == "online":
        # Online RL: agent learns by interacting with environment
        logger.info("Starting online RL training", n_episodes=n_episodes)

        def env_func():
            return start, goal_positions, maze

        train_metrics = agent.train_online(env_func, n_episodes=n_episodes, max_steps=max_steps)
    else:
        # Offline RL: agent learns from fixed dataset
        logger.info("Starting offline RL training", n_episodes=n_episodes)

        states, actions, rewards, next_states, dones = load_training_data(
            data_path, maze_size=maze_size, n_transitions=n_episodes * max_steps, seed=random_seed
        )
        logger.info("Loaded offline dataset", n_transitions=len(states))

        save_training_data(
            states, actions, rewards, next_states, dones, model_dir / "training_data.npz"
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

    logger.info("Training complete", **train_metrics)

    # Evaluate learned policy
    eval_metrics = agent.evaluate(maze, n_episodes=100, max_steps=max_steps)
    logger.info("Evaluation metrics", **eval_metrics)

    # Model validation
    if eval_metrics["success_rate"] < 0.5:
        logger.warning(
            "Policy success rate below threshold",
            success_rate=eval_metrics["success_rate"],
            threshold=0.5,
        )

    # Save model
    model_path = model_dir / f"robot_maze_model_v{model_version}.npz"
    agent.save(str(model_path))

    # Save maze visualization
    _save_chart(agent, maze, model_dir, model_version)

    # Combined metrics
    training_metrics = {
        **train_metrics,
        **eval_metrics,
        "maze_size": float(maze_size),
        "n_states": float(n_states),
        "n_walls": float(int(np.sum(maze))),
    }

    # Register model
    registry = ModelRegistry(base_dir=model_dir)
    registry.save_model(
        model_name="robot-maze-navigation",
        model_version=model_version,
        model_type="reinforcement_learning",
        metrics=training_metrics,
        parameters={
            "maze_size": maze_size,
            "n_episodes": n_episodes,
            "max_steps": max_steps,
            "learning_rate": learning_rate,
            "discount_factor": discount_factor,
            "epsilon_decay": epsilon_decay,
            "mode": mode,
            "random_seed": random_seed,
        },
        artifacts={
            f"robot_maze_model_v{model_version}.npz": model_path,
            "training_data.npz": model_dir / "training_data.npz",
        },
        tags={
            "framework": "numpy",
            "task": "reinforcement_learning",
            "method": "q_learning",
            "mode": mode,
        },
    )

    if register_to_mlflow:
        registry.log_to_mlflow(
            model_name="robot-maze-navigation",
            model_version=model_version,
            metrics=training_metrics,
            params={
                "maze_size": maze_size,
                "n_episodes": n_episodes,
                "max_steps": max_steps,
                "learning_rate": learning_rate,
                "discount_factor": discount_factor,
                "epsilon_decay": epsilon_decay,
                "mode": mode,
                "random_seed": random_seed,
            },
            artifacts={
                "model": str(model_path),
                "chart": str(model_dir / f"robot_maze_v{model_version}.png"),
                "training_data": str(model_dir / "training_data.npz"),
            },
            tags={
                "model_type": "reinforcement_learning",
                "framework": "numpy",
                "method": "q_learning",
                "mode": mode,
            },
        )
        logger.info(
            "Registered model to MLflow", model="robot-maze-navigation", version=model_version
        )

    return training_metrics


def _save_chart(agent: QLearningAgent, maze: np.ndarray, output_dir: Path, version: str) -> None:
    """Save the maze solution visualization and learning curves."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    # Plot 1: Maze with solution path
    ax1 = axes[0]
    ax1.imshow(maze, cmap="binary", interpolation="none")
    path, success, steps = agent.solve_maze(maze)
    if len(path) > 1:
        path_rows = [p[0] for p in path]
        path_cols = [p[1] for p in path]
        ax1.plot(path_cols, path_rows, "b-", linewidth=2, alpha=0.7, label="Path")
    ax1.scatter(
        [get_start_position(maze)[1]],
        [get_start_position(maze)[0]],
        c="green",
        s=200,
        marker="o",
        label="Start",
    )
    goals = get_goal_positions(maze)
    ax1.scatter(
        [g[1] for g in goals],
        [g[0] for g in goals],
        c="red",
        s=200,
        marker="*",
        label="Goal",
    )
    ax1.set_title(f"Maze Solution - v{version} (success={success}, steps={steps})")
    ax1.legend()
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Plot 2: Learning curve (episode rewards)
    ax2 = axes[1]
    if agent.episode_rewards:
        window = min(50, len(agent.episode_rewards) // 10)
        if window > 0:
            moving_avg = np.convolve(agent.episode_rewards, np.ones(window) / window, mode="valid")
            ax2.plot(agent.episode_rewards, alpha=0.3, color="steelblue", label="Episode reward")
            ax2.plot(
                range(window - 1, len(agent.episode_rewards)),
                moving_avg,
                color="red",
                label=f"MA({window})",
            )
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Reward")
    ax2.set_title("Learning Curve")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Q-value heatmap (max Q per state)
    ax3 = axes[2]
    if agent.q_table is not None:
        q_max = np.max(agent.q_table, axis=1).reshape(maze.shape)
        q_max_masked = np.where(maze == 1, np.nan, q_max)
        im = ax3.imshow(q_max_masked, cmap="hot", interpolation="none")
        plt.colorbar(im, ax=ax3, label="Max Q-value")
        ax3.set_title("Max Q-value per State")
        ax3.set_xticks([])
        ax3.set_yticks([])

    plt.tight_layout()
    chart_path = output_dir / f"robot_maze_v{version}.png"
    plt.savefig(str(chart_path), dpi=100)
    plt.close()
    logger.info("Chart saved", path=str(chart_path))


def main():
    parser = argparse.ArgumentParser(description="Train robot maze navigation Q-learning model")
    parser.add_argument("--model-dir", type=Path, default=Path(os.getenv("MODEL_DIR", "/models")))
    parser.add_argument("--data-path", type=Path, default=None)
    parser.add_argument("--maze-size", type=int, default=int(os.getenv("MAZE_SIZE", "8")))
    parser.add_argument("--n-episodes", type=int, default=int(os.getenv("N_EPISODES", "500")))
    parser.add_argument("--max-steps", type=int, default=int(os.getenv("MAX_STEPS", "200")))
    parser.add_argument(
        "--learning-rate", type=float, default=float(os.getenv("LEARNING_RATE", "0.1"))
    )
    parser.add_argument(
        "--discount-factor", type=float, default=float(os.getenv("DISCOUNT_FACTOR", "0.99"))
    )
    parser.add_argument(
        "--epsilon-decay", type=float, default=float(os.getenv("EPSILON_DECAY", "0.995"))
    )
    parser.add_argument(
        "--mode", type=str, default=os.getenv("RL_MODE", "online"), choices=["online", "offline"]
    )
    parser.add_argument("--model-version", type=str, default=os.getenv("MODEL_VERSION", "1.0.0"))
    parser.add_argument("--random-seed", type=int, default=int(os.getenv("RANDOM_SEED", "42")))
    parser.add_argument(
        "--register-mlflow",
        action="store_true",
        default=os.getenv("REGISTER_MLFLOW", "false").lower() == "true",
    )
    parser.add_argument("--log-level", type=str, default=os.getenv("LOG_LEVEL", "INFO"))
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

    logger.info("Training finished", metrics=metrics, model_dir=str(args.model_dir))


if __name__ == "__main__":
    main()
