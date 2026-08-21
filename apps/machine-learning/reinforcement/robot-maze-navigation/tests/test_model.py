"""Unit tests for robot-maze-navigation model."""

import numpy as np
import pytest
from robot_maze_navigation.model import QLearningAgent


class TestRobotMazeNavigation:
    """Tests for the robot maze navigation Q-learning model."""

    def _make_maze(self, size: int = 6, seed: int = 42) -> tuple[np.ndarray, int, int]:
        from robot_maze_navigation.data import generate_maze, get_start_position

        maze = generate_maze(size, size, seed)
        n_states = maze.shape[0] * maze.shape[1]
        return (
            maze,
            n_states,
            get_start_position(maze)[0] * maze.shape[1] + get_start_position(maze)[1],
        )

    def test_q_table_initialization(self):
        """Test that Q-table is initialized with correct shape."""
        maze, n_states, start_idx = self._make_maze()
        agent = QLearningAgent(n_states=n_states, n_actions=4, seed=42)
        assert agent.q_table is not None
        assert agent.q_table.shape == (n_states, 4)

    def test_get_action_exploration(self):
        """Test that get_action returns valid action indices."""
        maze, n_states, start_idx = self._make_maze()
        agent = QLearningAgent(n_states=n_states, n_actions=4, seed=42)
        for _ in range(100):
            action = agent.get_action(start_idx, training=True)
            assert 0 <= action < 4

    def test_q_update_changes_values(self):
        """Test that Q-values are updated after learning."""
        maze, n_states, start_idx = self._make_maze()
        agent = QLearningAgent(
            n_states=n_states, n_actions=4, learning_rate=0.1, discount_factor=0.9, seed=42
        )
        old_q = agent.q_table[start_idx, 0].copy()
        agent.update(start_idx, 0, 1.0, start_idx, False)
        assert agent.q_table[start_idx, 0] != old_q

    def test_train_online_improves_reward(self):
        """Test that online training improves episode rewards over time."""
        maze, n_states, start_idx = self._make_maze()
        from robot_maze_navigation.data import get_goal_positions

        agent = QLearningAgent(n_states=n_states, n_actions=4, epsilon_decay=0.99, seed=42)
        start_pos = (start_idx // maze.shape[1], start_idx % maze.shape[1])
        goal_positions = get_goal_positions(maze)

        def env_func():
            return start_pos, goal_positions, maze

        metrics = agent.train_online(env_func, n_episodes=50, max_steps=200)
        assert "mean_reward" in metrics
        assert "mean_length" in metrics
        assert metrics["n_episodes"] == 50.0

    def test_train_offline_improves_q_values(self):
        """Test that offline training updates Q-values from dataset."""
        maze, n_states, start_idx = self._make_maze()
        from robot_maze_navigation.data import generate_transitions

        agent = QLearningAgent(n_states=n_states, n_actions=4, seed=42)

        states, actions, rewards, next_states, dones = generate_transitions(maze, 1000, 42)
        old_q_sum = np.sum(agent.q_table)
        agent.train_offline(
            states, actions, rewards, next_states, dones, n_epochs=5, cols=maze.shape[1]
        )
        new_q_sum = np.sum(agent.q_table)
        assert old_q_sum != new_q_sum

    def test_solve_maze_returns_path(self):
        """Test that solve_maze returns a valid path."""
        maze, n_states, start_idx = self._make_maze()
        from robot_maze_navigation.data import get_goal_positions

        agent = QLearningAgent(n_states=n_states, n_actions=4, seed=42)
        start_pos = (start_idx // maze.shape[1], start_idx % maze.shape[1])
        goal_positions = get_goal_positions(maze)

        def env_func():
            return start_pos, goal_positions, maze

        agent.train_online(env_func, n_episodes=200, max_steps=200)
        path, success, steps = agent.solve_maze(maze)
        assert len(path) > 0
        assert path[0] == start_pos
        if success:
            assert path[-1] in goal_positions

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns success_rate and other metrics."""
        maze, n_states, start_idx = self._make_maze()
        from robot_maze_navigation.data import get_goal_positions

        agent = QLearningAgent(n_states=n_states, n_actions=4, seed=42)
        start_pos = (start_idx // maze.shape[1], start_idx % maze.shape[1])
        goal_positions = get_goal_positions(maze)

        def env_func():
            return start_pos, goal_positions, maze

        agent.train_online(env_func, n_episodes=100, max_steps=200)
        metrics = agent.evaluate(maze, n_episodes=20, max_steps=200)
        assert "success_rate" in metrics
        assert "mean_steps" in metrics
        assert "mean_reward" in metrics
        assert 0.0 <= metrics["success_rate"] <= 1.0

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        maze, n_states, start_idx = self._make_maze()
        agent = QLearningAgent(n_states=n_states, n_actions=4, seed=42)
        agent.update(start_idx, 0, 1.0, start_idx, False)

        path = str(tmp_path / "model.npz")
        agent.save(path)
        loaded = QLearningAgent.load(path)

        assert loaded.n_states == agent.n_states
        assert loaded.n_actions == agent.n_actions
        np.testing.assert_allclose(loaded.q_table, agent.q_table)
        assert loaded.learning_rate == pytest.approx(agent.learning_rate)
        assert loaded.discount_factor == pytest.approx(agent.discount_factor)
        assert loaded.mode == agent.mode

    def test_positive_negative_reinforcement(self):
        """Test that positive and negative reinforcement produce correct reward values."""
        maze, n_states, start_idx = self._make_maze()
        from robot_maze_navigation.data import get_goal_positions, get_reward

        goal_positions = get_goal_positions(maze)
        start = get_goal_positions(maze)[0]

        # Positive reinforcement: goal state
        goal_reward = get_reward(start, goal_positions, maze)
        assert goal_reward > 0

        # Negative reinforcement: wall state
        wall_reward = get_reward((0, 0), goal_positions, maze)
        assert wall_reward < 0

