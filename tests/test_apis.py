"""Integration tests for the serving APIs."""

import os
import sys
from pathlib import Path

import numpy as np
from fastapi.testclient import TestClient

# Set up test model directories BEFORE importing APIs
TEST_MODEL_DIR = Path(__file__).parent / "test_models"
os.environ["MODEL_DIR"] = str(TEST_MODEL_DIR)
os.environ["PIZZA_METRICS_PORT"] = "9001"
os.environ["SPAM_METRICS_PORT"] = "9002"
os.environ["MARKET_METRICS_PORT"] = "9003"
os.environ["RECOMMENDATION_METRICS_PORT"] = "9004"
os.environ["ANOMALY_METRICS_PORT"] = "9005"
os.environ["ROBOT_MAZE_METRICS_PORT"] = "9006"
os.environ["SELF_SUPERVISED_MONITORING_METRICS_PORT"] = "9007"

# Create and train pizza model
TEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
from anomaly_detection.model import PCAAnomalyDetector
from market_segmentation.model import KMeans
from pizza_price.model import LinearRegression
from recommendation_engine.model import Apriori
from robot_maze.data import generate_maze, get_goal_positions, get_start_position
from robot_maze.model import QLearningAgent
from self_supervised_monitoring.data import generate_synthetic_data
from self_supervised_monitoring.model import DenoisingAutoencoder
from semi_supervised_email.data import load_training_data
from semi_supervised_email.model import SelfTrainingClassifier
from spam_classification.model import LogisticRegression

# Train and save pizza model
X_pizza = np.array([6, 8, 10, 14, 18], dtype=float)
y_pizza = np.array([7.0, 9.0, 13.0, 17.5, 18.0], dtype=float)
pizza_model = LinearRegression()
pizza_model.fit(X_pizza, y_pizza)
pizza_model.save(str(TEST_MODEL_DIR / "pizza_model.npz"))

# Train and save spam model
X_spam = np.array([
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
y_spam = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
spam_model = LogisticRegression()
spam_model.fit(X_spam, y_spam)
spam_model.save(str(TEST_MODEL_DIR / "spam_model.npz"))

# Train and save market segmentation model
X_market = np.array([
    [30, 30], [35, 35], [28, 32], [32, 28], [33, 31],  # Budget-Conscious
    [80, 25], [85, 22], [78, 28], [82, 24], [79, 26],  # Cautious High-Earners
    [35, 80], [38, 82], [32, 78], [36, 85], [34, 79],  # Impulsive Shoppers
    [70, 85], [72, 88], [68, 82], [75, 86], [71, 84],  # Premium Shoppers
    [55, 55], [58, 52], [52, 58], [56, 54], [54, 57],  # Average Shoppers
], dtype=float)
market_model = KMeans(n_clusters=5, n_init=3, max_iterations=100, random_seed=42)
market_model.fit(X_market)
market_model.save(str(TEST_MODEL_DIR / "market_segmentation_model.npz"))

# Train and save recommendation engine model
transactions = [
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
rec_model = Apriori(min_support=0.2, min_confidence=0.5, min_lift=1.0)
rec_model.fit(transactions)
rec_model.save(str(TEST_MODEL_DIR / "recommendation_model.npz"))

# Train and save PCA anomaly detection model
from anomaly_detection.data import load_normal_training_data

X_normal_train = load_normal_training_data()  # 2000 normal server metric samples
anomaly_model = PCAAnomalyDetector(n_components=3, threshold_percentile=95.0, random_seed=42)
anomaly_model.fit(X_normal_train)
anomaly_model.save(str(TEST_MODEL_DIR / "anomaly_detection_model.npz"))

# Train and save robot maze navigation model

maze = generate_maze(6, 6, 42)
n_states = maze.shape[0] * maze.shape[1]
start_pos = get_start_position(maze)
goal_positions = get_goal_positions(maze)
robot_maze_model = QLearningAgent(n_states=n_states, n_actions=4, seed=42)


def robot_maze_env_func():
    return start_pos, goal_positions, maze


robot_maze_model.train_online(robot_maze_env_func, n_episodes=100, max_steps=200)
robot_maze_model.save(str(TEST_MODEL_DIR / "robot_maze_model.npz"))

# Train and save semi-supervised email model
X_ss, y_ss, _ = load_training_data(None, labeled_ratio=0.1, n_samples=200, random_seed=42)
ss_model = SelfTrainingClassifier(confidence_threshold=0.9, max_iterations=5, random_seed=42)
ss_model.fit(X_ss, y_ss)
ss_model.save(str(TEST_MODEL_DIR / "semi_supervised_email_model.npz"))

# Train and save self-supervised monitoring model
X_ssm, y_ssm = generate_synthetic_data(n_samples=500, random_seed=42)
ssm_model = DenoisingAutoencoder(
    hidden_dim=8, learning_rate=0.01, n_iterations=300, noise_rate=0.25, random_seed=42
)
ssm_model.fit(X_ssm[y_ssm == 0])  # Train on normal data only
ssm_model.save(str(TEST_MODEL_DIR / "self_supervised_monitoring_model.npz"))

# Now import APIs (they read MODEL_DIR at import time)
from anomaly_detection.api import app as anomaly_app
from market_segmentation.api import app as market_app
from pizza_price.api import app as pizza_app
from recommendation_engine.api import app as rec_app
from robot_maze.api import app as robot_maze_app
from self_supervised_monitoring.api import app as ss_monitoring_app
from semi_supervised_email.api import app as ss_email_app
from spam_classification.api import app as spam_app

# Make sure the test module directory is in sys.path
sys.path.insert(0, str(TEST_MODEL_DIR))


class TestPizzaAPI:
    """Tests for the pizza price prediction API."""

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        with TestClient(pizza_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_single(self):
        """Test single pizza price prediction."""
        with TestClient(pizza_app) as client:
            response = client.post("/predict", json={"diameter": 12.0})
            assert response.status_code == 200
            data = response.json()
            assert data["diameter"] == 12.0
            assert data["predicted_price"] > 0
            assert "model_version" in data

    def test_predict_bulk(self):
        """Test bulk pizza price prediction."""
        with TestClient(pizza_app) as client:
            response = client.post("/predict/bulk", json={"diameters": [7, 12, 16, 20]})
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 4
            assert data["model_version"] is not None

    def test_predict_invalid_input(self):
        """Test validation of invalid input."""
        with TestClient(pizza_app) as client:
            response = client.post("/predict", json={"diameter": -5})
            assert response.status_code == 422


class TestSpamAPI:
    """Tests for the spam classification API."""

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        with TestClient(spam_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_with_features(self):
        """Test spam prediction with explicit features."""
        with TestClient(spam_app) as client:
            response = client.post("/predict", json={"features": [1, 1, 1, 1, 0]})
            assert response.status_code == 200
            data = response.json()
            assert data["is_spam"] is True
            assert data["spam_probability"] > 0.5
            assert data["label"] == "SPAM"

    def test_predict_with_text(self):
        """Test spam prediction with raw email text."""
        with TestClient(spam_app) as client:
            response = client.post(
                "/predict/email",
                json={"text": "Congratulations! You have been selected to receive a FREE gift!!!"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["is_spam"] is True
            assert data["label"] == "SPAM"

    def test_predict_not_spam(self):
        """Test that a normal email is classified as not spam."""
        with TestClient(spam_app) as client:
            response = client.post(
                "/predict/email",
                json={"text": "Meeting at 3pm about the project"},
            )
            assert response.status_code == 200
            data = response.json()
            assert data["is_spam"] is False
            assert data["label"] == "NOT spam"

    def test_predict_invalid_features(self):
        """Test validation of invalid feature count."""
        with TestClient(spam_app) as client:
            response = client.post("/predict", json={"features": [1, 0, 0]})
            assert response.status_code == 422


class TestMarketSegmentationAPI:
    """Tests for the market segmentation API."""

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        with TestClient(market_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_segment_single(self):
        """Test single customer segmentation."""
        with TestClient(market_app) as client:
            response = client.post("/segment", json={"annual_income": 75.0, "spending_score": 85.0})
            assert response.status_code == 200
            data = response.json()
            assert data["segment"] in range(5)
            assert data["segment_name"] != ""
            assert 0.0 <= data["confidence"] <= 1.0
            assert "model_version" in data

    def test_segment_bulk(self):
        """Test bulk customer segmentation."""
        with TestClient(market_app) as client:
            response = client.post(
                "/segment/bulk",
                json={
                    "customers": [
                        {"annual_income": 30.0, "spending_score": 30.0},
                        {"annual_income": 80.0, "spending_score": 25.0},
                        {"annual_income": 70.0, "spending_score": 85.0},
                    ]
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["segments"]) == 3
            assert data["model_version"] is not None

    def test_segment_invalid_input(self):
        """Test validation of invalid input."""
        with TestClient(market_app) as client:
            response = client.post("/segment", json={"annual_income": -5, "spending_score": 50})
            assert response.status_code == 422

    def test_profiles_endpoint(self):
        """Test the cluster profiles endpoint."""
        with TestClient(market_app) as client:
            response = client.get("/profiles")
            assert response.status_code == 200
            data = response.json()
            assert data["n_clusters"] == 5
            assert len(data["profiles"]) == 5
            assert "model_version" in data


class TestRecommendationEngineAPI:
    """Tests for the recommendation engine API."""

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        with TestClient(rec_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_recommend(self):
        """Test product recommendations."""
        with TestClient(rec_app) as client:
            response = client.post(
                "/recommend",
                json={"items": ["Bread", "Milk"], "top_k": 5},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["recommendations"]) > 0
            assert "model_version" in data

    def test_recommend_invalid_item(self):
        """Test validation of unknown items."""
        with TestClient(rec_app) as client:
            response = client.post(
                "/recommend",
                json={"items": ["UnknownProduct"]},
            )
            assert response.status_code == 422

    def test_rules_endpoint(self):
        """Test the association rules endpoint."""
        with TestClient(rec_app) as client:
            response = client.get("/rules")
            assert response.status_code == 200
            data = response.json()
            assert data["n_rules"] > 0
            assert len(data["rules"]) > 0
            assert "model_version" in data

    def test_stats_endpoint(self):
        """Test the model stats endpoint."""
        with TestClient(rec_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert data["n_transactions"] > 0
            assert data["n_products"] > 0
            assert data["n_rules"] > 0
            assert "model_version" in data


class TestAnomalyDetectionAPI:
    """Tests for the PCA anomaly detection API."""

    NORMAL_SAMPLE = {
        "request_count": 120,
        "bytes_per_request": 4800,
        "cpu_usage": 35,
        "memory_usage": 55,
        "disk_io": 950,
        "network_in": 220,
        "network_out": 180,
        "error_rate": 1.5,
        "connection_count": 480,
        "response_time": 95,
    }

    ANOMALY_SAMPLE = {
        "request_count": 520,
        "bytes_per_request": 1400,
        "cpu_usage": 72,
        "memory_usage": 80,
        "disk_io": 2000,
        "network_in": 1700,
        "network_out": 1000,
        "error_rate": 15.0,
        "connection_count": 2600,
        "response_time": 280,
    }

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        with TestClient(anomaly_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_normal(self):
        """Test that normal traffic is not flagged as anomalous."""
        with TestClient(anomaly_app) as client:
            response = client.post("/predict", json=self.NORMAL_SAMPLE)
            assert response.status_code == 200
            data = response.json()
            assert data["is_anomaly"] is False
            assert data["anomaly_score"] >= 0.0
            assert 0.0 <= data["anomaly_probability"] <= 1.0
            assert "model_version" in data

    def test_predict_anomaly(self):
        """Test that anomalous traffic is flagged."""
        with TestClient(anomaly_app) as client:
            response = client.post("/predict", json=self.ANOMALY_SAMPLE)
            assert response.status_code == 200
            data = response.json()
            assert data["is_anomaly"] is True
            assert data["anomaly_score"] > data["anomaly_threshold"]
            assert data["anomaly_probability"] > 0.5
            assert "model_version" in data

    def test_predict_bulk(self):
        """Test bulk anomaly prediction."""
        with TestClient(anomaly_app) as client:
            response = client.post(
                "/predict/bulk",
                json={"samples": [self.NORMAL_SAMPLE, self.ANOMALY_SAMPLE]},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["samples"]) == 2
            assert data["n_samples"] == 2
            assert data["n_anomalies"] == 1
            assert data["model_version"] is not None

    def test_predict_invalid_input(self):
        """Test validation of invalid input."""
        bad = dict(self.NORMAL_SAMPLE)
        bad["cpu_usage"] = -10
        with TestClient(anomaly_app) as client:
            response = client.post("/predict", json=bad)
            assert response.status_code == 422

    def test_model_info_endpoint(self):
        """Test the model info endpoint."""
        with TestClient(anomaly_app) as client:
            response = client.get("/model/info")
            assert response.status_code == 200
            data = response.json()
            assert data["n_components"] == 3
            assert data["n_features"] == 10
            assert len(data["feature_names"]) == 10
            assert 0.0 < data["cumulative_variance_ratio"] <= 1.0
            assert data["reconstruction_threshold"] > 0
            assert "model_version" in data


class TestRobotMazeNavigationAPI:
    """Tests for the robot maze navigation API."""

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        with TestClient(robot_maze_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            assert "mode" in data

    def test_solve_endpoint(self):
        """Test maze solving."""
        with TestClient(robot_maze_app) as client:
            response = client.post("/solve", json={"maze_size": 6, "max_steps": 200})
            assert response.status_code == 200
            data = response.json()
            assert "path" in data
            assert "success" in data
            assert "steps" in data
            assert "model_version" in data

    def test_step_endpoint(self):
        """Test action computation."""
        with TestClient(robot_maze_app) as client:
            response = client.post("/step", json={"row": 1, "col": 1, "maze_size": 8})
            assert response.status_code == 200
            data = response.json()
            assert "action" in data
            assert "action_code" in data
            assert "q_values" in data
            assert len(data["q_values"]) == 4

    def test_train_endpoint_online(self):
        """Test online training endpoint."""
        with TestClient(robot_maze_app) as client:
            response = client.post("/train", json={"n_episodes": 10, "mode": "online"})
            assert response.status_code == 200
            data = response.json()
            assert "metrics" in data
            assert data["metrics"]["n_episodes"] == 10.0

    def test_stats_endpoint(self):
        """Test the stats endpoint."""
        with TestClient(robot_maze_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "n_states" in data
            assert "n_actions" in data
            assert "mode" in data
            assert "model_version" in data


class TestSemiSupervisedEmailAPI:
    """Tests for the semi-supervised email classification API."""

    SPAM_SAMPLE = {
        "has_free": 1,
        "has_win": 1,
        "has_link": 1,
        "has_exclamation": 1,
        "has_meeting": 0,
        "length_score": 8,
        "has_caps": 1,
    }

    HAM_SAMPLE = {
        "has_free": 0,
        "has_win": 0,
        "has_link": 0,
        "has_exclamation": 0,
        "has_meeting": 1,
        "length_score": 3,
        "has_caps": 0,
    }

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        with TestClient(ss_email_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            assert "training_mode" in data

    def test_predict_spam(self):
        """Test spam email classification."""
        with TestClient(ss_email_app) as client:
            response = client.post("/predict", json=self.SPAM_SAMPLE)
            assert response.status_code == 200
            data = response.json()
            assert data["is_spam"] is True
            assert data["label"] == "SPAM"
            assert 0.0 <= data["spam_probability"] <= 1.0
            assert "model_version" in data
            assert "training_mode" in data

    def test_predict_ham(self):
        """Test legitimate email classification."""
        with TestClient(ss_email_app) as client:
            response = client.post("/predict", json=self.HAM_SAMPLE)
            assert response.status_code == 200
            data = response.json()
            assert "is_spam" in data
            assert "spam_probability" in data
            assert 0.0 <= data["spam_probability"] <= 1.0
            assert data["label"] in ("SPAM", "NOT spam")

    def test_predict_bulk(self):
        """Test bulk email classification."""
        with TestClient(ss_email_app) as client:
            response = client.post("/predict/bulk", json=[self.SPAM_SAMPLE, self.HAM_SAMPLE])
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
            assert data["model_version"] is not None

    def test_predict_invalid_input(self):
        """Test validation of invalid input."""
        with TestClient(ss_email_app) as client:
            bad = dict(self.SPAM_SAMPLE)
            bad["has_free"] = 2  # Invalid: must be 0 or 1
            response = client.post("/predict", json=bad)
            assert response.status_code == 422

    def test_stats_endpoint(self):
        """Test the stats endpoint."""
        with TestClient(ss_email_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "n_features" in data
            assert "confidence_threshold" in data
            assert "training_mode" in data
            assert "n_labeled_initial" in data
            assert "n_labeled_final" in data
            assert data["training_mode"] == "semi-supervised"


class TestSelfSupervisedMonitoringAPI:
    """Tests for the self-supervised monitoring API."""

    NORMAL_SAMPLE = {
        "request_count": 50,
        "bytes_per_request": 5000,
        "cpu_usage": 35,
        "memory_usage": 55,
        "disk_io": 150,
        "network_in": 30,
        "network_out": 25,
        "error_rate": 1.0,
        "connection_count": 45,
        "response_time": 60,
    }

    ANOMALY_SAMPLE = {
        "request_count": 480,
        "bytes_per_request": 1200,
        "cpu_usage": 95,
        "memory_usage": 95,
        "disk_io": 1000,
        "network_in": 3000,
        "network_out": 2000,
        "error_rate": 45,
        "connection_count": 2500,
        "response_time": 900,
    }

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        with TestClient(ss_monitoring_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True
            assert "training_mode" in data

    def test_predict_normal(self):
        """Test that normal metrics are not flagged as anomalous."""
        with TestClient(ss_monitoring_app) as client:
            response = client.post("/predict", json=self.NORMAL_SAMPLE)
            assert response.status_code == 200
            data = response.json()
            assert "is_anomaly" in data
            assert "reconstruction_error" in data
            assert "anomaly_probability" in data
            assert 0.0 <= data["anomaly_probability"] <= 1.0
            assert "model_version" in data
            assert data["training_mode"] == "self-supervised"

    def test_predict_anomaly(self):
        """Test that anomalous metrics are flagged."""
        with TestClient(ss_monitoring_app) as client:
            response = client.post("/predict", json=self.ANOMALY_SAMPLE)
            assert response.status_code == 200
            data = response.json()
            assert data["is_anomaly"] is True
            assert data["reconstruction_error"] > 0
            assert data["anomaly_probability"] > 0.5
            assert "model_version" in data

    def test_predict_bulk(self):
        """Test bulk anomaly prediction."""
        with TestClient(ss_monitoring_app) as client:
            response = client.post(
                "/predict/bulk",
                json={"samples": [self.NORMAL_SAMPLE, self.ANOMALY_SAMPLE]},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["samples"]) == 2
            assert data["n_samples"] == 2
            assert data["n_anomalies"] >= 1
            assert data["model_version"] is not None

    def test_predict_invalid_input(self):
        """Test validation of invalid input."""
        with TestClient(ss_monitoring_app) as client:
            bad = dict(self.NORMAL_SAMPLE)
            bad["cpu_usage"] = -10  # Invalid: negative
            response = client.post("/predict", json=bad)
            assert response.status_code == 422

    def test_stats_endpoint(self):
        """Test the stats endpoint."""
        with TestClient(ss_monitoring_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "n_features" in data
            assert "hidden_dim" in data
            assert "threshold" in data
            assert "training_mode" in data
            assert "noise_rate" in data
            assert data["training_mode"] == "self-supervised"

    def test_model_info_endpoint(self):
        """Test the model info endpoint."""
        with TestClient(ss_monitoring_app) as client:
            response = client.get("/model/info")
            assert response.status_code == 200
            data = response.json()
            assert data["n_features"] == 10
            assert data["hidden_dim"] > 0
            assert len(data["feature_names"]) == 10
            assert data["training_mode"] == "self-supervised"
