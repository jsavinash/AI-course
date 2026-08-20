"""Q-Learning agent for robot maze navigation.

Implements tabular Q-learning with:
- Standard Q-value update rule
- Epsilon-greedy exploration with decay
- Positive and negative reinforcement reward shaping
- Online learning (interactive) and offline learning (from fixed dataset)
- Eligibility traces (optional SARSA-style)
- Proper serialization with metadata
"""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from robot_maze.data import ACTION_DELTAS, get_goal_positions, get_start_position, state_to_index


@dataclass
class QLearningAgent:
    """Tabular Q-learning agent for grid-world maze navigation.

    Args:
        n_states: Number of states in the environment
        n_actions: Number of actions (4 for up/down/left/right)
        learning_rate: Alpha - Q-value update step size
        discount_factor: Gamma - future reward discount
        epsilon: Initial exploration rate
        epsilon_min: Minimum exploration rate
        epsilon_decay: Epsilon decay per episode
        mode: Learning mode - "online" (interactive) or "offline" (from dataset)
        seed: Random seed for reproducibility
    """

    n_states: int
    n_actions: int = 4
    learning_rate: float = 0.1
    discount_factor: float = 0.99
    epsilon: float = 1.0
    epsilon_min: float = 0.01
    epsilon_decay: float = 0.995
    mode: Literal["online", "offline"] = "online"
    seed: int = 42

    # Learned state
    q_table: np.ndarray | None = None
    training_errors: list[float] = field(default_factory=list)
    episode_rewards: list[float] = field(default_factory=list)
    episode_lengths: list[int] = field(default_factory=list)
    n_episodes_trained: int = 0
    n_steps_total: int = 0

    def __post_init__(self):
        """Initialize Q-table with small random values or zeros."""
        rng = np.random.default_rng(self.seed)
        self.q_table = rng.uniform(0.0, 0.01, size=(self.n_states, self.n_actions))
        self._rng = np.random.default_rng(self.seed)

    def _ensure_trained(self) -> None:
        """Ensure Q-table is initialized."""
        if self.q_table is None:
            rng = np.random.default_rng(self.seed)
            self.q_table = rng.uniform(0.0, 0.01, size=(self.n_states, self.n_actions))

    def get_action(self, state: int, training: bool = True) -> int:
        """Select action using epsilon-greedy policy.

        Args:
            state: Current state index
            training: If True, use epsilon-greedy; if False, use greedy

        Returns:
            Selected action index
        """
        self._ensure_trained()
        if training and self._rng.random() < self.epsilon:
            return int(self._rng.integers(0, self.n_actions))
        return int(np.argmax(self.q_table[state]))

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
    ) -> float:
        """Perform a single Q-learning update.

        Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))

        Args:
            state: Current state index
            action: Action taken
            reward: Reward received
            next_state: Next state index
            done: Whether the episode terminated

        Returns:
            TD error (temporal difference error)
        """
        self._ensure_trained()
        best_next = np.max(self.q_table[next_state])
        td_target = reward + self.discount_factor * best_next * (1 - done)
        td_error = td_target - self.q_table[state, action]
        self.q_table[state, action] += self.learning_rate * td_error
        return float(td_error)

    def train_online(
        self,
        env_func,
        n_episodes: int = 1000,
        max_steps: int = 200,
    ) -> dict[str, float]:
        """Train the agent using online RL (interactive environment).

        The agent interacts with the environment in real-time, collecting
        experiences and learning from them.

        Args:
            env_func: Callable that returns (state, goal_positions, maze) tuple
            n_episodes: Number of training episodes
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with training metrics
        """
        self._ensure_trained()
        total_rewards = []
        total_lengths = []
        errors = []

        for _episode in range(n_episodes):
            state, goal_positions, maze = env_func()
            state_idx = state[0] * maze.shape[1] + state[1]
            episode_reward = 0.0
            episode_length = 0
            episode_errors = []

            for _ in range(max_steps):
                action = self.get_action(state_idx, training=True)
                next_state = self._step(state, action, maze)
                next_state_idx = next_state[0] * maze.shape[1] + next_state[1]
                reward = self._compute_reward(next_state, goal_positions, maze)
                done = next_state in goal_positions

                td_error = self.update(state_idx, action, reward, next_state_idx, done)
                episode_errors.append(abs(td_error))
                episode_reward += reward
                episode_length += 1
                self.n_steps_total += 1

                state = next_state
                state_idx = next_state_idx

                if done:
                    break

            # Decay epsilon
            self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

            total_rewards.append(episode_reward)
            total_lengths.append(episode_length)
            errors.extend(episode_errors)
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.training_errors.extend(episode_errors)
            self.n_episodes_trained += 1

        return {
            "mean_reward": float(np.mean(total_rewards)),
            "std_reward": float(np.std(total_rewards)),
            "mean_length": float(np.mean(total_lengths)),
            "std_length": float(np.std(total_lengths)),
            "mean_td_error": float(np.mean(errors)) if errors else 0.0,
            "final_epsilon": self.epsilon,
            "n_episodes": float(n_episodes),
            "n_steps_total": float(self.n_steps_total),
        }

    def train_offline(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
        n_epochs: int = 10,
        cols: int = 8,
    ) -> dict[str, float]:
        """Train the agent using offline RL (from fixed dataset).

        The agent learns from a pre-collected dataset without environment interaction.
        This demonstrates offline RL where the policy is learned from static data.

        Args:
            states: Array of state coordinates (n_samples, 2)
            actions: Array of actions (n_samples,)
            rewards: Array of rewards (n_samples,)
            next_states: Array of next state coordinates (n_samples, 2)
            dones: Array of done flags (n_samples,)
            n_epochs: Number of passes over the dataset
            cols: Number of columns for state indexing

        Returns:
            Dictionary with training metrics
        """
        self._ensure_trained()
        n_samples = len(states)
        errors = []

        for _epoch in range(n_epochs):
            epoch_errors = []
            for i in range(n_samples):
                s_idx = state_to_index((int(states[i, 0]), int(states[i, 1])), cols)
                ns_idx = state_to_index((int(next_states[i, 0]), int(next_states[i, 1])), cols)
                td_error = self.update(
                    s_idx,
                    int(actions[i]),
                    float(rewards[i]),
                    ns_idx,
                    bool(dones[i]),
                )
                epoch_errors.append(abs(td_error))
            errors.extend(epoch_errors)
            self.n_episodes_trained += 1

        self.training_errors.extend(errors)
        return {
            "mean_td_error": float(np.mean(errors)) if errors else 0.0,
            "std_td_error": float(np.std(errors)) if errors else 0.0,
            "n_epochs": float(n_epochs),
            "n_samples": float(n_samples),
            "n_steps_total": float(self.n_steps_total),
        }

    def _step(self, state: tuple[int, int], action: int, maze: np.ndarray) -> tuple[int, int]:
        """Execute one step in the maze environment."""
        dr, dc = ACTION_DELTAS[action]
        nr, nc = state[0] + dr, state[1] + dc
        rows, cols = maze.shape
        if 0 <= nr < rows and 0 <= nc < cols and maze[nr, nc] == 0:
            return (nr, nc)
        return state

    def _compute_reward(
        self,
        state: tuple[int, int],
        goal_positions: list[tuple[int, int]],
        maze: np.ndarray,
        reward_positive: float = 10.0,
        reward_negative: float = -1.0,
        reward_wall: float = -5.0,
        reward_step: float = -0.1,
    ) -> float:
        """Compute reward using positive and negative reinforcement.

        Positive reinforcement:
            - +reward_positive for reaching the goal

        Negative reinforcement:
            - reward_wall for hitting walls/out-of-bounds
            - reward_step per time step (encourages shorter paths)
        """
        if state in goal_positions:
            return reward_positive

        rows, cols = maze.shape
        r, c = state
        if not (0 <= r < rows and 0 <= c < cols) or maze[r, c] == 1:
            return reward_wall

        return reward_step

    def solve_maze(
        self,
        maze: np.ndarray,
        max_steps: int = 200,
    ) -> tuple[list[tuple[int, int]], bool, int]:
        """Solve the maze using the learned Q-table (greedy policy).

        Args:
            maze: Maze grid
            max_steps: Maximum steps to attempt

        Returns:
            Tuple of (path, success, steps_taken)
        """
        self._ensure_trained()
        start = get_start_position(maze)
        goal_positions = get_goal_positions(maze)
        path = [start]
        state = start

        for _ in range(max_steps):
            state_idx = state_to_index(state, maze.shape[1])
            action = self.get_action(state_idx, training=False)
            next_state = self._step(state, action, maze)
            path.append(next_state)
            state = next_state

            if state in goal_positions:
                return path, True, len(path)

        return path, False, len(path)

    def evaluate(self, maze: np.ndarray, n_episodes: int = 100, max_steps: int = 200) -> dict[str, float]:
        """Evaluate the learned policy.

        Args:
            maze: Maze grid
            n_episodes: Number of evaluation episodes
            max_steps: Maximum steps per episode

        Returns:
            Dictionary with evaluation metrics
        """
        self._ensure_trained()
        goal_positions = get_goal_positions(maze)
        successes = 0
        total_steps = []
        total_rewards = []

        for _ in range(n_episodes):
            state = get_start_position(maze)
            episode_reward = 0.0
            steps = 0

            for _ in range(max_steps):
                state_idx = state_to_index(state, maze.shape[1])
                action = self.get_action(state_idx, training=False)
                next_state = self._step(state, action, maze)
                reward = self._compute_reward(next_state, goal_positions, maze)
                episode_reward += reward
                steps += 1
                state = next_state

                if state in goal_positions:
                    successes += 1
                    break

            total_steps.append(steps)
            total_rewards.append(episode_reward)

        success_rate = successes / n_episodes if n_episodes > 0 else 0.0

        return {
            "success_rate": success_rate,
            "mean_steps": float(np.mean(total_steps)) if total_steps else 0.0,
            "std_steps": float(np.std(total_steps)) if total_steps else 0.0,
            "mean_reward": float(np.mean(total_rewards)) if total_rewards else 0.0,
            "std_reward": float(np.std(total_rewards)) if total_rewards else 0.0,
            "n_eval_episodes": float(n_episodes),
            "n_successes": float(successes),
            "epsilon": self.epsilon,
            "n_episodes_trained": float(self.n_episodes_trained),
            "n_steps_total": float(self.n_steps_total),
        }

    # ---------- Serialization ----------

    def save(self, path: str) -> None:
        """Save model parameters to disk."""
        self._ensure_trained()
        np.savez(
            path,
            q_table=self.q_table,
            n_states=np.array([self.n_states]),
            n_actions=np.array([self.n_actions]),
            learning_rate=np.array([self.learning_rate]),
            discount_factor=np.array([self.discount_factor]),
            epsilon=np.array([self.epsilon]),
            epsilon_min=np.array([self.epsilon_min]),
            epsilon_decay=np.array([self.epsilon_decay]),
            mode=np.array([self.mode]),
            seed=np.array([self.seed]),
            n_episodes_trained=np.array([self.n_episodes_trained]),
            n_steps_total=np.array([self.n_steps_total]),
            training_errors=np.array(self.training_errors),
            episode_rewards=np.array(self.episode_rewards),
            episode_lengths=np.array(self.episode_lengths),
        )

    @classmethod
    def load(cls, path: str) -> "QLearningAgent":
        """Load model parameters from disk."""
        data = np.load(path, allow_pickle=True)
        model = cls(
            n_states=int(data["n_states"].item()),
            n_actions=int(data["n_actions"].item()),
            learning_rate=float(data["learning_rate"].item()),
            discount_factor=float(data["discount_factor"].item()),
            epsilon=float(data["epsilon"].item()),
            epsilon_min=float(data["epsilon_min"].item()),
            epsilon_decay=float(data["epsilon_decay"].item()),
            mode=str(data["mode"].item()),
            seed=int(data["seed"].item()),
        )
        model.q_table = data["q_table"]
        model.n_episodes_trained = int(data["n_episodes_trained"].item())
        model.n_steps_total = int(data["n_steps_total"].item())
        model.training_errors = list(data["training_errors"])
        model.episode_rewards = list(data["episode_rewards"])
        model.episode_lengths = list(data["episode_lengths"])
        return model

    def to_dict(self) -> dict:
        """Return model parameters as a dict."""
        self._ensure_trained()
        return {
            "n_states": self.n_states,
            "n_actions": self.n_actions,
            "learning_rate": self.learning_rate,
            "discount_factor": self.discount_factor,
            "epsilon": self.epsilon,
            "epsilon_min": self.epsilon_min,
            "epsilon_decay": self.epsilon_decay,
            "mode": self.mode,
            "n_episodes_trained": self.n_episodes_trained,
            "n_steps_total": self.n_steps_total,
            "mean_training_error": float(np.mean(self.training_errors)) if self.training_errors else 0.0,
            "mean_episode_reward": float(np.mean(self.episode_rewards)) if self.episode_rewards else 0.0,
            "mean_episode_length": float(np.mean(self.episode_lengths)) if self.episode_lengths else 0.0,
            "seed": self.seed,
        }
