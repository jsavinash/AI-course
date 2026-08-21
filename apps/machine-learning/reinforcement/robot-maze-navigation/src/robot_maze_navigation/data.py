"""Maze environment and data generation for robot maze navigation.

Implements a grid-world maze environment with:
- Procedural maze generation using recursive backtracking
- Multiple maze layouts for training and evaluation
- State representation as (row, col) coordinates
- Action space: up, down, left, right
- Reward shaping with positive and negative reinforcement
"""

from pathlib import Path

import numpy as np

# Action constants
UP = 0
DOWN = 1
LEFT = 2
RIGHT = 3

ACTION_NAMES = {UP: "up", DOWN: "down", LEFT: "left", RIGHT: "right"}
ACTION_DELTAS = {UP: (-1, 0), DOWN: (1, 0), LEFT: (0, -1), RIGHT: (0, 1)}

# Default maze size
DEFAULT_MAZE_SIZE = 8
DEFAULT_N_EPISODES = 500


def generate_maze(
    rows: int = DEFAULT_MAZE_SIZE, cols: int = DEFAULT_MAZE_SIZE, seed: int = 42
) -> np.ndarray:
    """Generate a random maze using recursive backtracking.

    0 = open path, 1 = wall

    Args:
        rows: Number of rows in the maze
        cols: Number of columns in the maze
        seed: Random seed for reproducibility

    Returns:
        2D numpy array representing the maze
    """
    rng = np.random.default_rng(seed)

    # Ensure odd dimensions for proper maze generation
    if rows % 2 == 0:
        rows += 1
    if cols % 2 == 0:
        cols += 1

    maze = np.ones((rows, cols), dtype=int)

    def carve(r: int, c: int) -> None:
        maze[r, c] = 0
        directions = [(0, 2), (0, -2), (2, 0), (-2, 0)]
        rng.shuffle(directions)
        for dr, dc in directions:
            nr, nc = r + dr, c + dc
            if 0 < nr < rows - 1 and 0 < nc < cols - 1 and maze[nr, nc] == 1:
                maze[r + dr // 2, c + dc // 2] = 0
                carve(nr, nc)

    carve(1, 1)

    # Ensure start (top-left) and goal (bottom-right) are open
    maze[1, 1] = 0
    maze[rows - 2, cols - 2] = 0

    return maze


def get_start_position(maze: np.ndarray) -> tuple[int, int]:
    """Get the start position (first open cell from top-left)."""
    rows, cols = maze.shape
    for r in range(rows):
        for c in range(cols):
            if maze[r, c] == 0:
                return (r, c)
    return (1, 1)


def get_goal_positions(maze: np.ndarray) -> list[tuple[int, int]]:
    """Get all goal positions (open cells in bottom-right region)."""
    rows, cols = maze.shape
    goals = []
    for r in range(rows - 3, rows - 1):
        for c in range(cols - 3, cols - 1):
            if 0 <= r < rows and 0 <= c < cols and maze[r, c] == 0:
                goals.append((r, c))
    return goals if goals else [(rows - 2, cols - 2)]


def state_to_index(state: tuple[int, int], cols: int) -> int:
    """Convert (row, col) state to flat index."""
    return state[0] * cols + state[1]


def index_to_state(index: int, cols: int) -> tuple[int, int]:
    """Convert flat index back to (row, col) state."""
    return (index // cols, index % cols)


def get_reward(
    state: tuple[int, int],
    goal_positions: list[tuple[int, int]],
    maze: np.ndarray,
    reward_positive: float = 10.0,
    reward_negative: float = -1.0,
    reward_wall: float = -5.0,
    reward_step: float = -0.1,
) -> float:
    """Compute reward for a state using positive and negative reinforcement.

    Positive reinforcement:
        - Large reward for reaching the goal

    Negative reinforcement:
        - Penalty for hitting walls
        - Small time-step penalty to encourage shorter paths
        - Penalty proportional to distance from goal (optional shaping)

    Args:
        state: Current (row, col) position
        goal_positions: List of goal positions
        maze: Maze grid
        reward_positive: Reward for reaching goal
        reward_negative: Penalty for invalid move (wall/out of bounds)
        reward_wall: Penalty for hitting a wall
        reward_step: Per-step penalty to encourage efficiency

    Returns:
        Reward value
    """
    if state in goal_positions:
        return reward_positive

    rows, cols = maze.shape
    r, c = state

    if not (0 <= r < rows and 0 <= c < cols) or maze[r, c] == 1:
        return reward_wall

    return reward_step


def get_next_state(state: tuple[int, int], action: int, maze: np.ndarray) -> tuple[int, int]:
    """Get next state after taking an action.

    If the action leads to a wall or out of bounds, stay in the same state.

    Args:
        state: Current (row, col) position
        action: Action to take (UP, DOWN, LEFT, RIGHT)
        maze: Maze grid

    Returns:
        Next (row, col) position
    """
    dr, dc = ACTION_DELTAS[action]
    nr, nc = state[0] + dr, state[1] + dc
    rows, cols = maze.shape

    if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0:
        return (nr, nc)

    return state


def generate_transitions(
    maze: np.ndarray,
    n_transitions: int = 10000,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate transition data for offline RL training.

    Collects (state, action, reward, next_state, done) tuples by
    running a random policy in the maze.

    Args:
        maze: Maze grid
        n_transitions: Number of transitions to generate
        seed: Random seed

    Returns:
        Tuple of arrays: (states, actions, rewards, next_states, dones)
    """
    rng = np.random.default_rng(seed)
    rows, cols = maze.shape
    goal_positions = get_goal_positions(maze)
    start = get_start_position(maze)

    states = np.zeros((n_transitions, 2), dtype=int)
    next_states = np.zeros((n_transitions, 2), dtype=int)
    actions = np.zeros(n_transitions, dtype=int)
    rewards = np.zeros(n_transitions, dtype=float)
    dones = np.zeros(n_transitions, dtype=bool)

    state = start
    t = 0
    for i in range(n_transitions):
        # Random action
        action = int(rng.integers(0, 4))
        next_state = get_next_state(state, action, maze)
        reward = get_reward(next_state, goal_positions, maze)
        done = next_state in goal_positions

        states[i] = state
        next_states[i] = next_state
        actions[i] = action
        rewards[i] = reward
        dones[i] = done

        t += 1
        if done or t > rows * cols * 2:
            state = start
            t = 0
        else:
            state = next_state

    return states, actions, rewards, next_states, dones


def load_training_data(
    data_path: Path | None = None,
    maze_size: int = DEFAULT_MAZE_SIZE,
    n_transitions: int = DEFAULT_N_EPISODES * 20,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load or generate maze transition data for offline RL.

    Args:
        data_path: Optional path to saved transition data
        maze_size: Size of the maze (rows = cols)
        n_transitions: Number of transitions to generate
        seed: Random seed

    Returns:
        Tuple of arrays: (states, actions, rewards, next_states, dones)
    """
    if data_path and Path(data_path).exists():
        data = np.load(data_path)
        return (
            data["states"],
            data["actions"],
            data["rewards"],
            data["next_states"],
            data["dones"],
        )

    maze = generate_maze(maze_size, maze_size, seed)
    return generate_transitions(maze, n_transitions, seed)


def save_training_data(
    states: np.ndarray,
    actions: np.ndarray,
    rewards: np.ndarray,
    next_states: np.ndarray,
    dones: np.ndarray,
    path: Path,
) -> None:
    """Save transition data to NPZ for reproducibility."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        states=states,
        actions=actions,
        rewards=rewards,
        next_states=next_states,
        dones=dones,
    )
