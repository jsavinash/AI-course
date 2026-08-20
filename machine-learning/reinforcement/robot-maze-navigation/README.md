# Robot Maze Navigation (Reinforcement Learning)

Production-ready Q-Learning agent for robot maze navigation.

## Learning Type: Reinforcement Learning

This example demonstrates **reinforcement learning** — the agent learns by interacting with an environment, receiving rewards/penalties for its actions, and gradually improving its policy to maximize cumulative reward.

## Problem

Navigate a robot through a maze to reach a goal position using Q-Learning. The agent must learn the optimal path without being explicitly told the correct actions.

## Approach

- **Algorithm**: Tabular Q-Learning (model-free, off-policy)
- **Exploration**: Epsilon-greedy strategy with exponential decay
- **Reward shaping**: Positive rewards for reaching goal, negative for hitting walls/steps
- **Training modes**: 
  - **Online**: Learn interactively by running episodes
  - **Offline**: Learn from a fixed dataset of state transitions
- **Eligibility traces**: Optional SARSA-style updates
- **Implementation**: Pure NumPy, no external RL libraries

## Architecture

```
machine-learning/reinforcement/robot-maze-navigation/
├── pyproject.toml              # Package configuration
├── src/robot_maze/
│   ├── __init__.py             # Package exports
│   ├── data.py                 # Maze generation and environment
│   ├── model.py                # QLearningAgent (from scratch)
│   ├── train.py                # Training pipeline script
│   └── api.py                  # FastAPI serving API
└── README.md                   # This file
```

## Maze Environment

- Grid-based maze (e.g., 6x6 cells)
- Wall cells are impassable
- Robot starts at a designated position
- Goal position is marked
- Actions: up, down, left, right (4 possible)
- Rewards:
  - **+100** for reaching the goal
  - **-1** for each step (encourages shorter paths)
  - **-10** for hitting a wall

## Quick Start

### Training

```bash
make train-robot-maze
```

Or directly:

```bash
uv run python -m robot_maze.train --model-dir ./artifacts/models --model-version 1.0.0
```

### Serving

```bash
make serve-robot-maze
```

Or directly:

```bash
MODEL_DIR=./artifacts/models uv run uvicorn robot_maze.api:app --host 0.0.0.0 --port 8005
```

### API Usage

```bash
# Health check
curl http://localhost:8005/health

# Solve a maze
curl -X POST http://localhost:8005/solve \
  -H "Content-Type: application/json" \
  -d '{"maze_size": 6, "max_steps": 200}'

# Get next action for a given state
curl -X POST http://localhost:8005/step \
  -H "Content-Type: application/json" \
  -d '{"row": 1, "col": 1, "maze_size": 8}'

# Online training endpoint
curl -X POST http://localhost:8005/train \
  -H "Content-Type: application/json" \
  -d '{"n_episodes": 10, "mode": "online"}'

# Get model statistics
curl http://localhost:8005/stats
```

## API Endpoints

| Method | Path            | Description                     |
|--------|-----------------|---------------------------------|
| GET    | `/health`       | Health check                    |
| POST   | `/solve`        | Solve a maze end-to-end         |
| POST   | `/step`         | Get next action for a state     |
| POST   | `/train`        | Train the agent online          |
| GET    | `/stats`        | Model statistics                |
| GET    | `/metrics`      | Prometheus metrics              |
| POST   | `/reload`       | Reload model                    |

## Model Parameters

| Parameter           | Default  | Description                          |
|---------------------|----------|--------------------------------------|
| `n_states`          | -        | Total number of states (maze_size²) |
| `n_actions`         | `4`      | Number of actions (up/down/left/right) |
| `learning_rate`     | `0.1`    | Alpha — Q-value update step size    |
| `discount_factor`   | `0.99`   | Gamma — future reward discount      |
| `epsilon`           | `1.0`    | Initial exploration rate            |
| `epsilon_min`       | `0.01`   | Minimum exploration rate            |
| `epsilon_decay`     | `0.995`  | Epsilon decay per episode           |
| `mode`              | `online` | Learning mode: "online" or "offline" |
| `seed`              | `42`     | Random seed                         |

## How It Works

1. **Q-Table**: The agent maintains a table of Q-values Q(s, a) for each state-action pair.
2. **Exploration**: With probability ε, the agent takes a random action; otherwise, it takes the action with the highest Q-value.
3. **Update Rule**: After each step, the Q-value is updated:
   ```
   Q(s,a) ← Q(s,a) + α * [r + γ * max(Q(s',a')) - Q(s,a)]
   ```
4. **Convergence**: Over many episodes, ε decays and the agent shifts from exploration to exploitation, converging to an optimal policy.

## Key Concepts Demonstrated

- Q-Learning (tabular, model-free, off-policy)
- Epsilon-greedy exploration with decay
- Reward shaping (positive/negative rewards)
- Online vs. offline learning modes
- Production API serving with FastAPI
- Model versioning with registry
