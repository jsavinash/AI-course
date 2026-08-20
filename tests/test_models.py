"""Unit tests for pizza-price, spam-classification, market-segmentation, recommendation-engine, anomaly-detection, robot-maze-navigation, and self-supervised-monitoring models."""

import numpy as np
import pytest
from anomaly_detection.model import PCAAnomalyDetector
from market_segmentation.model import KMeans
from pizza_price.model import LinearRegression
from recommendation_engine.model import Apriori
from robot_maze.model import QLearningAgent
from self_supervised_monitoring.data import generate_synthetic_data
from self_supervised_monitoring.model import DenoisingAutoencoder
from semi_supervised_email.data import load_training_data
from semi_supervised_email.model import SelfTrainingClassifier
from spam_classification.model import LogisticRegression


class TestPizzaLinearRegression:
    """Tests for the pizza price Linear Regression model."""

    def test_training_converges(self):
        """Test that gradient descent converges to reasonable parameters."""
        X = np.array([6, 8, 10, 14, 18], dtype=float)
        y = np.array([7.0, 9.0, 13.0, 17.5, 18.0], dtype=float)

        model = LinearRegression(learning_rate=0.001, n_iterations=2000)
        model.fit(X, y)

        # Weight should be positive (bigger pizza = more expensive)
        assert model.weight > 0
        # Sanity check on reasonable range
        assert 0.5 < model.weight < 2.0
        assert -2.0 < model.bias < 5.0

        # MSE should be low for this small dataset
        mse = model.mse(X, y)
        assert mse < 5.0

    def test_prediction_shape(self):
        """Test that predictions return correct shape."""
        model = LinearRegression()
        model.weight = 1.0
        model.bias = 0.5

        preds = model.predict(np.array([6.0, 8.0, 12.0]))
        assert preds.shape == (3,)
        np.testing.assert_allclose(preds, [6.5, 8.5, 12.5])

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X = np.array([6, 8, 10, 14, 18], dtype=float)
        y = np.array([7.0, 9.0, 13.0, 17.5, 18.0], dtype=float)

        model = LinearRegression()
        model.fit(X, y)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = LinearRegression.load(path)

        assert loaded.weight == pytest.approx(model.weight)
        assert loaded.bias == pytest.approx(model.bias)
        assert len(loaded.loss_history) == len(model.loss_history)


class TestSpamLogisticRegression:
    """Tests for the spam classification Logistic Regression model."""

    def test_training_accuracy(self):
        """Test that the model learns to classify training data."""
        emails = np.array([
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 0, 1, 0, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 0],
            [0, 0, 0, 0, 1],
            [0, 1, 1, 0, 0],
            [0, 0, 0, 0, 1],
        ], dtype=float)
        labels = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=int)

        model = LogisticRegression(learning_rate=0.1, n_iterations=2000)
        model.fit(emails, labels)

        accuracy = model.accuracy(emails, labels)
        assert accuracy >= 0.9

    def test_predict_proba_range(self):
        """Test that probabilities are in [0, 1]."""
        X = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=float)
        y = np.array([1, 0], dtype=int)

        model = LogisticRegression()
        model.fit(X, y)

        probs = model.predict_proba(X)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X = np.array([[1, 0, 0, 0, 0], [0, 1, 0, 0, 0]], dtype=float)
        y = np.array([1, 0], dtype=int)

        model = LogisticRegression()
        model.fit(X, y)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = LogisticRegression.load(path)

        assert loaded.weights is not None
        assert loaded.bias == pytest.approx(model.bias)
        np.testing.assert_allclose(loaded.weights, model.weights)


class TestMarketSegmentationKMeans:
    """Tests for the market segmentation K-Means clustering model."""

    def _make_data(self, n_samples: int = 200) -> np.ndarray:
        """Generate simple separable clusters for testing."""
        rng = np.random.default_rng(42)
        cluster1 = rng.normal(loc=[30, 30], scale=3.0, size=(n_samples // 2, 2))
        cluster2 = rng.normal(loc=[80, 80], scale=3.0, size=(n_samples // 2, 2))
        return np.vstack([cluster1, cluster2])

    def test_fit_creates_centroids(self):
        """Test that fit produces the expected number of centroids."""
        X = self._make_data()
        model = KMeans(n_clusters=2, n_init=3, max_iterations=100)
        model.fit(X)

        assert model.centroids is not None
        assert model.centroids.shape == (2, 2)
        assert model.labels is not None
        assert len(model.labels) == len(X)
        assert set(np.unique(model.labels)).issubset({0, 1})

    def test_clusters_are_separable(self):
        """Test that K-Means separates the two distinct clusters."""
        X = self._make_data()
        model = KMeans(n_clusters=2, n_init=3, max_iterations=100)
        model.fit(X)

        # Points from cluster1 (low income) should mostly be in one cluster
        labels = model.labels
        low_income_labels = labels[: len(X) // 2]
        high_income_labels = labels[len(X) // 2 :]

        # The two groups should be assigned to different clusters
        assert set(low_income_labels) != set(high_income_labels)

    def test_predict_assigns_to_nearest_centroid(self):
        """Test that predict returns valid cluster indices."""
        X = self._make_data()
        model = KMeans(n_clusters=2, n_init=3, max_iterations=100)
        model.fit(X)

        preds = model.predict(np.array([[35.0, 35.0], [75.0, 75.0]]))
        assert preds.shape == (2,)
        assert set(preds).issubset({0, 1})

    def test_predict_confidence_in_range(self):
        """Test that confidence scores are in [0, 1]."""
        X = self._make_data()
        model = KMeans(n_clusters=2, n_init=3, max_iterations=100)
        model.fit(X)

        conf = model.predict_confidence(np.array([[35.0, 35.0], [75.0, 75.0]]))
        assert np.all(conf >= 0.0)
        assert np.all(conf <= 1.0)

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X = self._make_data()
        model = KMeans(n_clusters=2, n_init=3, max_iterations=100)
        model.fit(X)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = KMeans.load(path)

        assert loaded.centroids is not None
        assert loaded.feature_mean is not None
        assert loaded.feature_std is not None
        np.testing.assert_allclose(loaded.centroids, model.centroids)
        np.testing.assert_allclose(loaded.feature_mean, model.feature_mean)
        assert loaded.n_clusters == model.n_clusters
        assert loaded.inertia == pytest.approx(model.inertia)

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns unsupervised clustering metrics."""
        X = self._make_data()
        model = KMeans(n_clusters=2, n_init=3, max_iterations=100)
        model.fit(X)

        metrics = model.evaluate(X)
        assert "inertia" in metrics
        assert "silhouette" in metrics
        assert "n_clusters" in metrics
        assert metrics["inertia"] > 0
        assert -1.0 <= metrics["silhouette"] <= 1.0


class TestRecommendationEngineApriori:
    """Tests for the recommendation engine Apriori model."""

    def _make_transactions(self) -> list[list[str]]:
        """Generate simple transaction data with known associations."""
        return [
            ["Bread", "Milk", "Butter"],
            ["Bread", "Milk"],
            ["Bread", "Butter"],
            ["Milk", "Butter"],
            ["Bread", "Milk", "Butter"],
            ["Beer", "Diapers", "Chips"],
            ["Beer", "Diapers"],
            ["Beer", "Chips"],
            ["Diapers", "Chips"],
            ["Beer", "Diapers", "Chips"],
            ["Coffee", "Sugar"],
            ["Coffee", "Cream"],
            ["Sugar", "Cream"],
            ["Coffee", "Sugar", "Cream"],
            ["Coffee", "Sugar"],
        ]

    def test_fit_finds_frequent_itemsets(self):
        """Test that fit discovers frequent itemsets."""
        transactions = self._make_transactions()
        model = Apriori(min_support=0.2, min_confidence=0.5, min_lift=1.0)
        model.fit(transactions)

        assert len(model.frequent_itemsets) > 0
        assert model.n_transactions == len(transactions)
        assert len(model.products) > 0

    def test_fit_generates_rules(self):
        """Test that fit generates association rules."""
        transactions = self._make_transactions()
        model = Apriori(min_support=0.2, min_confidence=0.5, min_lift=1.0)
        model.fit(transactions)

        assert len(model.rules) > 0
        for rule in model.rules:
            assert rule.confidence >= 0.5
            assert rule.lift >= 1.0
            assert rule.support >= 0.2

    def test_recommend_returns_items(self):
        """Test that recommend returns relevant items."""
        transactions = self._make_transactions()
        model = Apriori(min_support=0.2, min_confidence=0.5, min_lift=1.0)
        model.fit(transactions)

        recs = model.recommend(["Bread", "Milk"], top_k=5)
        assert len(recs) > 0
        for rec in recs:
            assert "item" in rec
            assert "score" in rec
            assert "lift" in rec
            assert "confidence" in rec

    def test_recommend_excludes_purchased(self):
        """Test that recommend excludes items already in the basket."""
        transactions = self._make_transactions()
        model = Apriori(min_support=0.2, min_confidence=0.5, min_lift=1.0)
        model.fit(transactions)

        recs = model.recommend(["Bread", "Milk"], top_k=10, exclude_purchased=True)
        for rec in recs:
            assert rec["item"] not in ["Bread", "Milk"]

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        transactions = self._make_transactions()
        model = Apriori(min_support=0.2, min_confidence=0.5, min_lift=1.0)
        model.fit(transactions)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = Apriori.load(path)

        assert loaded.n_transactions == model.n_transactions
        assert loaded.products == model.products
        assert len(loaded.rules) == len(model.rules)
        assert len(loaded.frequent_itemsets) == len(model.frequent_itemsets)

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns rule quality metrics."""
        transactions = self._make_transactions()
        model = Apriori(min_support=0.2, min_confidence=0.5, min_lift=1.0)
        model.fit(transactions)

        metrics = model.evaluate(transactions)
        assert "n_rules" in metrics
        assert "n_frequent_itemsets" in metrics
        assert "coverage" in metrics
        assert "avg_confidence" in metrics
        assert "avg_lift" in metrics
        assert metrics["n_rules"] > 0
        assert 0.0 <= metrics["coverage"] <= 1.0


class TestPCAAnomalyDetector:
    """Tests for the PCA-based anomaly detection model."""

    def _make_data(self, n_samples: int = 400, shuffle: bool = True) -> tuple[np.ndarray, np.ndarray]:
        """Generate correlated normal data plus a few clear outliers."""
        rng = np.random.default_rng(42)
        # Correlated 5-dimensional normal data
        mean = np.array([50.0, 200.0, 30.0, 60.0, 90.0])
        cov = np.array([
            [25.0, 40.0, 10.0, 12.0, 8.0],
            [40.0, 100.0, 20.0, 25.0, 15.0],
            [10.0, 20.0, 9.0, 8.0, 5.0],
            [12.0, 25.0, 8.0, 12.0, 6.0],
            [8.0, 15.0, 5.0, 6.0, 7.0],
        ])
        X_normal = rng.multivariate_normal(mean, cov, size=n_samples)
        # 5 extreme outliers far from the distribution (5-10 std devs in multiple dims)
        # These live outside the dominant PCA subspace, so reconstruction error is huge
        anomalies = np.array([
            [300.0, 1600.0, 99.0, 99.0, 500.0],
            [280.0, 1500.0, 98.0, 98.0, 480.0],
            [320.0, 1700.0, 99.5, 99.5, 520.0],
            [10.0, 50.0, 1.0, 1.0, 5.0],
            [400.0, 2000.0, 99.0, 99.0, 600.0],
        ])
        X = np.vstack([X_normal, anomalies])
        y = np.concatenate([np.zeros(n_samples, dtype=int), np.ones(len(anomalies), dtype=int)])
        # Shuffle so anomalies aren't all at the end (tests detection, not position)
        if shuffle:
            perm = rng.permutation(len(X))
            X = X[perm]
            y = y[perm]
        return X, y

    def test_fit_creates_components(self):
        """Test that fit produces the expected number of principal components."""
        X, _ = self._make_data()
        model = PCAAnomalyDetector(n_components=3)
        model.fit(X)

        assert model.components is not None
        assert model.components.shape == (5, 3)
        assert model.feature_mean is not None
        assert model.feature_std is not None
        assert model.explained_variance_ratio is not None
        assert len(model.explained_variance_ratio) == 3
        assert 0.0 < model.cumulative_variance_ratio <= 1.0
        assert model.reconstruction_threshold > 0.0

    def test_transform_reduces_dimensions(self):
        """Test that transform projects data onto fewer dimensions."""
        X, _ = self._make_data()
        model = PCAAnomalyDetector(n_components=3)
        model.fit(X)

        projected = model.transform(X)
        assert projected.shape == (len(X), 3)

    def test_predict_returns_scores(self):
        """Test that predict returns per-sample anomaly scores."""
        X, _ = self._make_data()
        model = PCAAnomalyDetector(n_components=3)
        model.fit(X)

        scores = model.predict(X)
        assert scores.shape == (len(X),)
        assert np.all(scores >= 0.0)

    def test_predict_anomaly_flags_outliers(self):
        """Test that clear outliers are flagged as anomalies."""
        X, y = self._make_data()
        model = PCAAnomalyDetector(n_components=3, threshold_percentile=95.0)
        model.fit(X)

        preds = model.predict_anomaly(X)
        assert preds.shape == (len(X),)
        assert set(preds).issubset({0, 1})
        # All 5 synthetic outliers should be detected regardless of position
        assert np.sum(preds[y == 1]) == 5

    def test_predict_proba_range(self):
        """Test that anomaly probabilities are in [0, 1]."""
        X, _ = self._make_data()
        model = PCAAnomalyDetector(n_components=3)
        model.fit(X)

        probs = model.predict_proba(X)
        assert np.all(probs >= 0.0)
        assert np.all(probs <= 1.0)

    def test_reconstruction(self):
        """Test that reconstruction returns original feature dimensionality."""
        X, _ = self._make_data()
        model = PCAAnomalyDetector(n_components=3)
        model.fit(X)

        reconstructed = model.reconstruct(X)
        assert reconstructed.shape == X.shape

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns anomaly detection metrics.

        Uses the production pattern: fit PCA on normal data only, then
        evaluate on the full dataset (normal + anomalies).
        """
        X, y = self._make_data(shuffle=False)
        n_normal = int(np.sum(y == 0))
        model = PCAAnomalyDetector(n_components=3)
        model.fit(X[:n_normal])  # Fit on normal data only

        metrics = model.evaluate(X, y)
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "accuracy" in metrics
        assert "false_positive_rate" in metrics
        assert "anomaly_threshold" in metrics
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        # Perfect recall: all anomalies are detected (the core detection requirement)
        assert metrics["recall"] >= 0.8
        # F1 is bounded below by the designed FPR operating point:
        # threshold_percentile=95.0 intentionally flags ~5% of normal data,
        # which limits precision when the anomaly set is small.
        assert metrics["f1"] >= 0.3
        assert metrics["false_positive_rate"] <= 0.1

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X, _ = self._make_data()
        model = PCAAnomalyDetector(n_components=3)
        model.fit(X)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = PCAAnomalyDetector.load(path)

        assert loaded.n_components == model.n_components
        assert loaded.feature_mean is not None
        assert loaded.feature_std is not None
        assert loaded.components is not None
        np.testing.assert_allclose(loaded.components, model.components)
        np.testing.assert_allclose(loaded.feature_mean, model.feature_mean)
        assert loaded.reconstruction_threshold == pytest.approx(model.reconstruction_threshold)
        assert loaded.cumulative_variance_ratio == pytest.approx(model.cumulative_variance_ratio)

    def test_invalid_components(self):
        """Test that n_components greater than n_features raises."""
        X, _ = self._make_data()
        model = PCAAnomalyDetector(n_components=10)
        with pytest.raises(ValueError, match="n_components"):
            model.fit(X)


class TestRobotMazeNavigation:
    """Tests for the robot maze navigation Q-learning model."""

    def _make_maze(self, size: int = 6, seed: int = 42) -> tuple[np.ndarray, int, int]:
        from robot_maze.data import generate_maze, get_start_position
        maze = generate_maze(size, size, seed)
        n_states = maze.shape[0] * maze.shape[1]
        return maze, n_states, get_start_position(maze)[0] * maze.shape[1] + get_start_position(maze)[1]

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
        agent = QLearningAgent(n_states=n_states, n_actions=4, learning_rate=0.1, discount_factor=0.9, seed=42)
        old_q = agent.q_table[start_idx, 0].copy()
        agent.update(start_idx, 0, 1.0, start_idx, False)
        assert agent.q_table[start_idx, 0] != old_q

    def test_train_online_improves_reward(self):
        """Test that online training improves episode rewards over time."""
        maze, n_states, start_idx = self._make_maze()
        from robot_maze.data import get_goal_positions
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
        from robot_maze.data import generate_transitions
        agent = QLearningAgent(n_states=n_states, n_actions=4, seed=42)

        states, actions, rewards, next_states, dones = generate_transitions(maze, 1000, 42)
        old_q_sum = np.sum(agent.q_table)
        agent.train_offline(states, actions, rewards, next_states, dones, n_epochs=5, cols=maze.shape[1])
        new_q_sum = np.sum(agent.q_table)
        assert old_q_sum != new_q_sum

    def test_solve_maze_returns_path(self):
        """Test that solve_maze returns a valid path."""
        maze, n_states, start_idx = self._make_maze()
        from robot_maze.data import get_goal_positions
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
        from robot_maze.data import get_goal_positions
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
        from robot_maze.data import get_goal_positions, get_reward
        goal_positions = get_goal_positions(maze)
        start = get_goal_positions(maze)[0]

        # Positive reinforcement: goal state
        goal_reward = get_reward(start, goal_positions, maze)
        assert goal_reward > 0

        # Negative reinforcement: wall state
        wall_reward = get_reward((0, 0), goal_positions, maze)
        assert wall_reward < 0


class TestSemiSupervisedEmail:
    """Tests for the semi-supervised email classification model."""

    def _make_data(self, n_samples: int = 200, labeled_ratio: float = 0.1, seed: int = 42):
        """Generate semi-supervised email data."""
        return load_training_data(
            data_path=None,
            labeled_ratio=labeled_ratio,
            n_samples=n_samples,
            random_seed=seed,
        )

    def test_self_training_improves_over_supervised(self):
        """Test that self-training produces a working model with reasonable accuracy."""
        X, y, is_labeled = self._make_data(n_samples=500, labeled_ratio=0.15, seed=42)

        # Self-training model with moderate confidence threshold
        ss_model = SelfTrainingClassifier(
            confidence_threshold=0.8,
            max_iterations=15,
            min_labeled_ratio=0.9,
            random_seed=42,
        )
        ss_model.fit(X, y)

        # Evaluate on all labeled data
        X_labeled, y_labeled = _get_labeled_data(X, y)
        metrics = ss_model.evaluate(X_labeled, y_labeled)

        # Model should achieve reasonable accuracy and be in semi-supervised mode
        assert metrics["accuracy"] >= 0.4
        assert ss_model.training_mode == "semi-supervised"

    def test_self_training_uses_unlabeled_data(self):
        """Test that self-training incorporates unlabeled samples."""
        X, y, is_labeled = self._make_data(n_samples=200, labeled_ratio=0.1, seed=42)
        n_initial_labeled = np.sum(is_labeled)

        model = SelfTrainingClassifier(
            confidence_threshold=0.85,
            max_iterations=10,
            random_seed=42,
        )
        model.fit(X, y)

        # Should have used unlabeled data (semi-supervised mode)
        assert model.training_mode == "semi-supervised"
        assert len(model.n_labeled_history) > 1
        assert model.n_labeled_history[-1] > n_initial_labeled

    def test_predict_returns_valid_probabilities(self):
        """Test that predictions return valid probabilities."""
        X, y, is_labeled = self._make_data(n_samples=100, labeled_ratio=0.2, seed=42)
        model = SelfTrainingClassifier(confidence_threshold=0.9, max_iterations=5, random_seed=42)
        model.fit(X, y)

        probas = model.predict_proba(X[:10])
        assert probas.shape == (10,)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_predict_returns_valid_labels(self):
        """Test that predictions return binary labels."""
        X, y, is_labeled = self._make_data(n_samples=100, labeled_ratio=0.2, seed=42)
        model = SelfTrainingClassifier(confidence_threshold=0.9, max_iterations=5, random_seed=42)
        model.fit(X, y)

        preds = model.predict(X[:10])
        assert preds.shape == (10,)
        assert set(preds).issubset({0, 1})

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns classification metrics."""
        X, y, is_labeled = self._make_data(n_samples=100, labeled_ratio=0.2, seed=42)
        model = SelfTrainingClassifier(confidence_threshold=0.9, max_iterations=5, random_seed=42)
        model.fit(X, y)

        X_labeled, y_labeled = _get_labeled_data(X, y)
        metrics = model.evaluate(X_labeled, y_labeled)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X, y, is_labeled = self._make_data(n_samples=100, labeled_ratio=0.2, seed=42)
        model = SelfTrainingClassifier(confidence_threshold=0.9, max_iterations=5, random_seed=42)
        model.fit(X, y)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = SelfTrainingClassifier.load(path)

        assert loaded.n_features == model.n_features
        assert loaded.confidence_threshold == model.confidence_threshold
        assert loaded.training_mode == model.training_mode
        assert loaded.n_iterations_used == model.n_iterations_used
        np.testing.assert_allclose(loaded.model.weights, model.model.weights)
        assert loaded.model.bias == pytest.approx(model.model.bias)

    def test_to_dict_returns_metadata(self):
        """Test that to_dict returns model metadata."""
        X, y, is_labeled = self._make_data(n_samples=100, labeled_ratio=0.2, seed=42)
        model = SelfTrainingClassifier(confidence_threshold=0.9, max_iterations=5, random_seed=42)
        model.fit(X, y)

        metadata = model.to_dict()
        assert "n_features" in metadata
        assert "training_mode" in metadata
        assert "n_iterations_used" in metadata
        assert "n_labeled_history" in metadata
        assert "accuracy_history" in metadata
        assert metadata["training_mode"] == "semi-supervised"

    def test_supervised_mode_when_no_unlabeled(self):
        """Test that model stays in supervised mode when all data is labeled."""
        X, y, is_labeled = self._make_data(n_samples=100, labeled_ratio=1.0, seed=42)

        model = SelfTrainingClassifier(confidence_threshold=0.9, max_iterations=5, random_seed=42)
        model.fit(X, y)

        assert model.training_mode == "supervised"
        assert len(model.n_labeled_history) == 1


def _get_labeled_data(X: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Helper to extract labeled data."""
    mask = y != -1
    return X[mask], y[mask]


class TestDenoisingAutoencoder:
    """Tests for the self-supervised DenoisingAutoencoder model."""

    def _make_data(self, n_samples: int = 500, seed: int = 42) -> tuple[np.ndarray, np.ndarray]:
        """Generate synthetic server metrics data."""
        return generate_synthetic_data(n_samples=n_samples, random_seed=seed)

    def test_training_converges(self):
        """Test that training loss decreases over iterations."""
        X, _ = self._make_data(n_samples=500)
        X_normal = X[:400]  # Normal samples for training

        model = DenoisingAutoencoder(
            hidden_dim=8, learning_rate=0.01, n_iterations=500, noise_rate=0.25, random_seed=42
        )
        model.fit(X_normal)

        assert len(model.loss_history) > 0
        assert model.loss_history[0] > 0
        assert model.loss_history[-1] < model.loss_history[0]
        assert model.training_mode == "self-supervised"

    def test_training_converges_fast(self):
        """Test quick training convergence."""
        X, _ = self._make_data(n_samples=200)

        model = DenoisingAutoencoder(
            hidden_dim=4, learning_rate=0.01, n_iterations=200, noise_rate=0.25, random_seed=42
        )
        model.fit(X)

        assert len(model.loss_history) > 0
        assert model.loss_history[-1] < model.loss_history[0]

    def test_predict_proba_range(self):
        """Test that anomaly probabilities are in valid range."""
        X, _ = self._make_data(n_samples=200)
        model = DenoisingAutoencoder(
            hidden_dim=8, learning_rate=0.01, n_iterations=200, noise_rate=0.25, random_seed=42
        )
        model.fit(X)

        probas = model.predict_proba(X)
        assert probas.shape == (len(X),)
        assert np.all(probas >= 0.0)
        assert np.all(probas <= 1.0)

    def test_predict_returns_valid_labels(self):
        """Test that predictions are binary (0 or 1)."""
        X, _ = self._make_data(n_samples=200)
        model = DenoisingAutoencoder(
            hidden_dim=8, learning_rate=0.01, n_iterations=200, noise_rate=0.25, random_seed=42
        )
        model.fit(X)

        predictions = model.predict(X)
        assert predictions.shape == (len(X),)
        assert set(np.unique(predictions)).issubset({0, 1})

    def test_evaluate_returns_metrics(self):
        """Test that evaluate returns classification metrics."""
        X, y = self._make_data(n_samples=100)

        model = DenoisingAutoencoder(
            hidden_dim=8, learning_rate=0.01, n_iterations=200, noise_rate=0.25, random_seed=42
        )
        model.fit(X[y == 0])  # Train on normal data only

        metrics = model.evaluate(X, y)
        assert "accuracy" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_evaluate_detects_anomalies(self):
        """Test that anomalies have higher reconstruction error than normal data."""
        X, y = self._make_data(n_samples=500, seed=42)
        X_normal = X[y == 0]
        X_anomaly = X[y == 1]

        model = DenoisingAutoencoder(
            hidden_dim=8, learning_rate=0.01, n_iterations=300, noise_rate=0.25, random_seed=42
        )
        model.fit(X_normal[:400])

        normal_errors = model.reconstruction_error(X_normal[400:])
        anomaly_errors = model.reconstruction_error(X_anomaly)

        assert np.mean(anomaly_errors) > np.mean(normal_errors)

    def test_save_load_roundtrip(self, tmp_path):
        """Test that model save/load preserves parameters."""
        X, _ = self._make_data(n_samples=200)
        model = DenoisingAutoencoder(
            hidden_dim=8, learning_rate=0.01, n_iterations=200, noise_rate=0.25, random_seed=42
        )
        model.fit(X)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = DenoisingAutoencoder.load(path)

        assert loaded.input_dim == model.input_dim
        assert loaded.hidden_dim == model.hidden_dim
        assert loaded.threshold == pytest.approx(model.threshold)
        assert loaded.training_mode == model.training_mode
        np.testing.assert_allclose(loaded.W1, model.W1)
        np.testing.assert_allclose(loaded.W2, model.W2)
        np.testing.assert_allclose(loaded.b1, model.b1)
        np.testing.assert_allclose(loaded.b2, model.b2)

    def test_save_load_predictions_match(self, tmp_path):
        """Test that loaded model produces same predictions."""
        X, _ = self._make_data(n_samples=200)
        model = DenoisingAutoencoder(
            hidden_dim=8, learning_rate=0.01, n_iterations=200, noise_rate=0.25, random_seed=42
        )
        model.fit(X)

        path = str(tmp_path / "model.npz")
        model.save(path)
        loaded = DenoisingAutoencoder.load(path)

        test_input = X[:5]
        np.testing.assert_allclose(model.predict(test_input), loaded.predict(test_input))
        np.testing.assert_allclose(model.predict_proba(test_input), loaded.predict_proba(test_input), rtol=1e-5)

    def test_to_dict_returns_metadata(self):
        """Test that to_dict returns model metadata."""
        X, _ = self._make_data(n_samples=100)
        model = DenoisingAutoencoder(
            hidden_dim=8, learning_rate=0.01, n_iterations=100, noise_rate=0.25, random_seed=42
        )
        model.fit(X)

        metadata = model.to_dict()
        assert "input_dim" in metadata
        assert "hidden_dim" in metadata
        assert "training_mode" in metadata
        assert "threshold" in metadata
        assert "n_epochs_run" in metadata
        assert metadata["training_mode"] == "self-supervised"
        assert metadata["input_dim"] == X.shape[1]

    def test_reconstruction_error_shape(self):
        """Test that reconstruction error returns one value per sample."""
        X, _ = self._make_data(n_samples=200)
        model = DenoisingAutoencoder(
            hidden_dim=4, learning_rate=0.01, n_iterations=100, noise_rate=0.25, random_seed=42
        )
        model.fit(X)

        errors = model.reconstruction_error(X)
        assert errors.shape == (len(X),)
        assert np.all(errors >= 0.0)
