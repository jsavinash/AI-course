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
os.environ["EMAIL_SPAM_NN_METRICS_PORT"] = "9001"
os.environ["HOUSE_PRICE_METRICS_PORT"] = "9002"
os.environ["FRAUD_DETECTION_METRICS_PORT"] = "9003"
os.environ["DIGIT_RECOGNITION_METRICS_PORT"] = "9004"
os.environ["LANGUAGE_TRANSLATION_METRICS_PORT"] = "9005"
os.environ["SENTIMENT_ANALYSIS_METRICS_PORT"] = "9006"
os.environ["TEXT_GENERATION_METRICS_PORT"] = "9007"
os.environ["SPEECH_RECOGNITION_METRICS_PORT"] = "9008"
os.environ["MUSIC_GENERATION_METRICS_PORT"] = "9009"
os.environ["STOCK_PREDICTION_METRICS_PORT"] = "9010"
os.environ["WEATHER_FORECASTING_METRICS_PORT"] = "9011"
os.environ["IMAGE_CAPTIONING_METRICS_PORT"] = "9012"
os.environ["MEDICAL_IMAGING_METRICS_PORT"] = "9013"
os.environ["FACIAL_RECOGNITION_METRICS_PORT"] = "9014"
os.environ["VIDEO_SURVEILLANCE_METRICS_PORT"] = "9015"
os.environ["IMAGE_SUPER_RESOLUTION_METRICS_PORT"] = "9016"
os.environ["SEMANTIC_SEGMENTATION_METRICS_PORT"] = "9017"
os.environ["GENERATIVE_ART_METRICS_PORT"] = "9018"
os.environ["AUTONOMOUS_DRIVING_METRICS_PORT"] = "9019"
os.environ["MEDICAL_SCAN_ANALYSIS_METRICS_PORT"] = "9020"
os.environ["TEXT_CHAR_RECOGNITION_METRICS_PORT"] = "9021"

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
X_spam = np.array(
    [
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
    ],
    dtype=float,
)
y_spam = np.array([1, 0, 1, 0, 1, 0, 1, 0, 1, 0], dtype=int)
spam_model = LogisticRegression()
spam_model.fit(X_spam, y_spam)
spam_model.save(str(TEST_MODEL_DIR / "spam_model.npz"))

# Train and save market segmentation model
X_market = np.array(
    [
        [30, 30],
        [35, 35],
        [28, 32],
        [32, 28],
        [33, 31],  # Budget-Conscious
        [80, 25],
        [85, 22],
        [78, 28],
        [82, 24],
        [79, 26],  # Cautious High-Earners
        [35, 80],
        [38, 82],
        [32, 78],
        [36, 85],
        [34, 79],  # Impulsive Shoppers
        [70, 85],
        [72, 88],
        [68, 82],
        [75, 86],
        [71, 84],  # Premium Shoppers
        [55, 55],
        [58, 52],
        [52, 58],
        [56, 54],
        [54, 57],  # Average Shoppers
    ],
    dtype=float,
)
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

# Train and save email spam detection NN model
from email_spam_detection.data import generate_synthetic_data as generate_spam_data
from email_spam_detection.model import SpamDetectionNN

X_spam_nn, y_spam_nn = generate_spam_data(n_samples=200, random_seed=42)
spam_nn_model = SpamDetectionNN(hidden_dim=16, learning_rate=0.01, n_iterations=300, random_seed=42)
spam_nn_model.fit(X_spam_nn, y_spam_nn)
spam_nn_model.save(str(TEST_MODEL_DIR / "spam_detection_model.npz"))

# Train and save house price prediction NN model
from house_price_prediction.data import generate_synthetic_data as generate_house_data
from house_price_prediction.model import HousePriceNN

X_house_nn, y_house_nn = generate_house_data(n_samples=200, random_seed=42)
house_nn_model = HousePriceNN(hidden_dim=32, learning_rate=0.01, n_iterations=300, random_seed=42)
house_nn_model.fit(X_house_nn, y_house_nn)
house_nn_model.save(str(TEST_MODEL_DIR / "house_price_model.npz"))

# Train and save credit card fraud detection NN model
from credit_card_fraud_detection.data import generate_synthetic_data as generate_fraud_data
from credit_card_fraud_detection.model import FraudDetectionAutoencoder

X_fraud_nn, y_fraud_nn = generate_fraud_data(n_samples=400, random_seed=42)
X_fraud_normal = X_fraud_nn[y_fraud_nn == 0]
fraud_nn_model = FraudDetectionAutoencoder(
    hidden_dim=4, learning_rate=0.001, n_iterations=300, random_seed=42
)
fraud_nn_model.fit(X_fraud_normal)
fraud_nn_model.save(str(TEST_MODEL_DIR / "fraud_detection_model.npz"))

# Train and save handwritten digit recognition NN model
from handwritten_digit_recognition.data import generate_synthetic_data as generate_digits
from handwritten_digit_recognition.model import DigitRecognitionNN

X_digit_nn, y_digit_nn = generate_digits(n_samples=200, random_seed=42)
digit_nn_model = DigitRecognitionNN(
    hidden_dim=32, learning_rate=0.1, n_iterations=300, random_seed=42
)
digit_nn_model.fit(X_digit_nn, y_digit_nn)
digit_nn_model.save(str(TEST_MODEL_DIR / "digit_recognition_model.npz"))

# Train and save language translation RNN model
from language_translation.data import generate_synthetic_data as generate_translation_data
from language_translation.model import LanguageTranslationRNN

X_translation, y_translation = generate_translation_data(n_samples=50, random_seed=42)
translation_model = LanguageTranslationRNN(
    vocab_size=40, seq_len=8, hidden_dim=16, learning_rate=0.1, n_iterations=50, random_seed=42
)
translation_model.fit(X_translation, y_translation)
translation_model.save(str(TEST_MODEL_DIR / "language_translation_model.npz"))

# Train and save sentiment analysis RNN model
from sentiment_analysis.data import generate_synthetic_data as generate_sentiment_data
from sentiment_analysis.model import SentimentAnalysisRNN

X_sentiment, y_sentiment = generate_sentiment_data(n_samples=50, random_seed=42)
sentiment_model = SentimentAnalysisRNN(
    vocab_size=50, seq_len=10, hidden_dim=16, learning_rate=0.05, n_iterations=50, random_seed=42
)
sentiment_model.fit(X_sentiment, y_sentiment)
sentiment_model.save(str(TEST_MODEL_DIR / "sentiment_analysis_model.npz"))

# Train and save text generation RNN model
from text_generation.data import generate_synthetic_data as generate_text_data
from text_generation.model import TextGenerationRNN

X_text = generate_text_data(n_samples=50, random_seed=42)
text_model = TextGenerationRNN(
    vocab_size=26, seq_len=20, hidden_dim=16, learning_rate=0.1, n_iterations=50, random_seed=42
)
text_model.fit(X_text)
text_model.save(str(TEST_MODEL_DIR / "text_generation_model.npz"))

# Train and save speech recognition RNN model
from speech_recognition.data import generate_synthetic_data as generate_speech_data
from speech_recognition.model import SpeechRecognitionRNN

X_speech, y_speech = generate_speech_data(n_samples=50, random_seed=42)
speech_model = SpeechRecognitionRNN(
    n_features=16,
    seq_len=20,
    n_classes=10,
    hidden_dim=16,
    learning_rate=0.05,
    n_iterations=50,
    random_seed=42,
)
speech_model.fit(X_speech, y_speech)
speech_model.save(str(TEST_MODEL_DIR / "speech_recognition_model.npz"))

# Train and save music generation RNN model
from music_generation.data import generate_synthetic_data as generate_music_data
from music_generation.model import MusicGenerationRNN

X_music = generate_music_data(n_samples=50, random_seed=42)
music_model = MusicGenerationRNN(
    vocab_size=40, seq_len=20, hidden_dim=16, learning_rate=0.1, n_iterations=50, random_seed=42
)
music_model.fit(X_music)
music_model.save(str(TEST_MODEL_DIR / "music_generation_model.npz"))

# Train and save stock market prediction RNN model
from stock_market_prediction.data import generate_synthetic_data as generate_stock_data
from stock_market_prediction.model import StockMarketRNN

X_stock, y_stock = generate_stock_data(n_samples=50, random_seed=42)
stock_model = StockMarketRNN(
    n_features=5, seq_len=20, hidden_dim=16, learning_rate=0.01, n_iterations=50, random_seed=42
)
stock_model.fit(X_stock, y_stock)
stock_model.save(str(TEST_MODEL_DIR / "stock_market_model.npz"))

# Train and save weather forecasting RNN model
from weather_forecasting.data import generate_synthetic_data as generate_weather_data
from weather_forecasting.model import WeatherForecastingRNN

X_weather, y_weather = generate_weather_data(n_samples=50, random_seed=42)
weather_model = WeatherForecastingRNN(
    n_features=5, seq_len=30, hidden_dim=16, learning_rate=0.01, n_iterations=50, random_seed=42
)
weather_model.fit(X_weather, y_weather)
weather_model.save(str(TEST_MODEL_DIR / "weather_forecasting_model.npz"))

# Train and save image captioning RNN model
from image_captioning.data import generate_synthetic_data as generate_image_data
from image_captioning.model import ImageCaptioningRNN

X_image, y_image = generate_image_data(n_samples=50, random_seed=42)
image_model = ImageCaptioningRNN(
    n_pixels=64,
    vocab_size=20,
    caption_len=8,
    hidden_dim=16,
    learning_rate=0.05,
    n_iterations=50,
    random_seed=42,
)
image_model.fit(X_image, y_image)
image_model.save(str(TEST_MODEL_DIR / "image_captioning_model.npz"))

# Train and save medical imaging CNN model
from medical_imaging.data import generate_synthetic_data as generate_medical_data
from medical_imaging.model import MedicalImagingCNN

X_medical, y_medical = generate_medical_data(n_samples=200, random_seed=42)
medical_model = MedicalImagingCNN(
    n_filters=8, kernel_size=3, hidden_dim=32,
    learning_rate=0.05, n_iterations=100, random_seed=42
)
medical_model.fit(X_medical, y_medical)
medical_model.save(str(TEST_MODEL_DIR / "medical_imaging_model.npz"))

# Train and save facial recognition CNN model
from facial_recognition.data import generate_synthetic_data as generate_face_data
from facial_recognition.model import FacialRecognitionCNN

X_face, y_face = generate_face_data(n_samples=200, random_seed=42)
face_model = FacialRecognitionCNN(
    n_filters=8, kernel_size=3, hidden_dim=32,
    learning_rate=0.01, n_iterations=100, random_seed=42
)
face_model.fit(X_face, y_face)
face_model.save(str(TEST_MODEL_DIR / "facial_recognition_model.npz"))

# Train and save video surveillance CNN model
from video_surveillance.data import generate_synthetic_data as generate_surv_data
from video_surveillance.model import VideoSurveillanceCNN

X_surv, y_surv = generate_surv_data(n_samples=200, random_seed=42)
surv_model = VideoSurveillanceCNN(
    n_filters=8, kernel_size=3, hidden_dim=32,
    learning_rate=0.05, n_iterations=100, random_seed=42
)
surv_model.fit(X_surv, y_surv)
surv_model.save(str(TEST_MODEL_DIR / "video_surveillance_model.npz"))

# Train and save image super-resolution DN model
from image_super_resolution.data import generate_synthetic_data as generate_sr_data
from image_super_resolution.model import ImageSuperResolutionDN

X_sr, y_sr = generate_sr_data(n_samples=50, random_seed=42)
sr_model = ImageSuperResolutionDN(
    n_filters=4, kernel_size=3, learning_rate=0.01, n_iterations=30, random_seed=42
)
sr_model.fit(X_sr, y_sr)
sr_model.save(str(TEST_MODEL_DIR / "image_super_resolution_model.npz"))

# Train and save semantic segmentation DN model
from semantic_segmentation.data import generate_synthetic_data as generate_seg_data
from semantic_segmentation.model import SemanticSegmentationDN

X_seg, y_seg = generate_seg_data(n_samples=50, random_seed=42)
seg_model = SemanticSegmentationDN(
    n_filters=4, kernel_size=3, learning_rate=0.01, n_iterations=30, random_seed=42
)
seg_model.fit(X_seg, y_seg)
seg_model.save(str(TEST_MODEL_DIR / "semantic_segmentation_model.npz"))

# Train and save generative art DN model
from generative_art.data import generate_synthetic_data as generate_art_data
from generative_art.model import GenerativeArtDN

X_art, y_art = generate_art_data(n_samples=50, random_seed=42)
art_model = GenerativeArtDN(
    n_filters=4, kernel_size=3, learning_rate=0.01, n_iterations=30, random_seed=42
)
art_model.fit(X_art, y_art)
art_model.save(str(TEST_MODEL_DIR / "generative_art_model.npz"))

# Train and save autonomous driving CapsNet model
from autonomous_driving.data import generate_synthetic_data as generate_ad_data
from autonomous_driving.model import AutonomousDrivingCapsNet

X_ad, y_ad = generate_ad_data(n_samples=200, random_seed=42)
ad_model = AutonomousDrivingCapsNet(
    n_filters=8, kernel_size=3, learning_rate=0.05, n_iterations=100, random_seed=42
)
ad_model.fit(X_ad, y_ad)
ad_model.save(str(TEST_MODEL_DIR / "autonomous_driving_model.npz"))

# Train and save medical scan analysis CapsNet model
from medical_scan_analysis.data import generate_synthetic_data as generate_msa_data
from medical_scan_analysis.model import MedicalScanAnalysisCapsNet

X_msa, y_msa = generate_msa_data(n_samples=200, random_seed=42)
msa_model = MedicalScanAnalysisCapsNet(
    n_filters=8, kernel_size=3, learning_rate=0.05, n_iterations=100, random_seed=42
)
msa_model.fit(X_msa, y_msa)
msa_model.save(str(TEST_MODEL_DIR / "medical_scan_analysis_model.npz"))

# Train and save text/char recognition CapsNet model
from text_char_recognition.data import generate_synthetic_data as generate_tcr_data
from text_char_recognition.model import TextCharRecognitionCapsNet

X_tcr, y_tcr = generate_tcr_data(n_samples=360, random_seed=42)
tcr_model = TextCharRecognitionCapsNet(
    n_filters=8, kernel_size=3, learning_rate=0.1, n_iterations=100, random_seed=42
)
tcr_model.fit(X_tcr, y_tcr)
tcr_model.save(str(TEST_MODEL_DIR / "text_char_recognition_model.npz"))

# Now import APIs (they read MODEL_DIR at import time)
from anomaly_detection.api import app as anomaly_app
from autonomous_driving.api import app as autonomous_driving_app
from credit_card_fraud_detection.api import app as fraud_nn_app
from email_spam_detection.api import app as spam_nn_app
from facial_recognition.api import app as facial_recognition_app
from generative_art.api import app as generative_art_app
from handwritten_digit_recognition.api import app as digit_nn_app
from house_price_prediction.api import app as house_nn_app
from image_captioning.api import app as image_captioning_app
from image_super_resolution.api import app as image_super_resolution_app
from language_translation.api import app as translation_app
from market_segmentation.api import app as market_app
from medical_imaging.api import app as medical_imaging_app
from medical_scan_analysis.api import app as medical_scan_analysis_app
from music_generation.api import app as music_app
from pizza_price.api import app as pizza_app
from recommendation_engine.api import app as rec_app
from robot_maze.api import app as robot_maze_app
from self_supervised_monitoring.api import app as ss_monitoring_app
from semantic_segmentation.api import app as semantic_segmentation_app
from semi_supervised_email.api import app as ss_email_app
from sentiment_analysis.api import app as sentiment_app
from spam_classification.api import app as spam_app
from speech_recognition.api import app as speech_app
from stock_market_prediction.api import app as stock_app
from text_char_recognition.api import app as text_char_recognition_app
from text_generation.api import app as text_gen_app
from video_surveillance.api import app as video_surveillance_app
from weather_forecasting.api import app as weather_app

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


class TestEmailSpamDetectionNNAPI:
    """Tests for the email spam detection feedforward NN API."""

    SPAM_FEATURES = [1, 1, 1, 1, 0, 8, 1, 1, 5, 8, 1, 0.2]
    HAM_FEATURES = [0, 0, 0, 0, 1, 5, 0, 0, 0, 0, 0, 0.9]

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        with TestClient(spam_nn_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_spam(self):
        """Test spam classification."""
        with TestClient(spam_nn_app) as client:
            response = client.post("/predict", json={"features": self.SPAM_FEATURES})
            assert response.status_code == 200
            data = response.json()
            assert "is_spam" in data
            assert "spam_probability" in data
            assert "label" in data
            assert 0.0 <= data["spam_probability"] <= 1.0
            assert "model_version" in data

    def test_predict_ham(self):
        """Test ham classification."""
        with TestClient(spam_nn_app) as client:
            response = client.post("/predict", json={"features": self.HAM_FEATURES})
            assert response.status_code == 200
            data = response.json()
            assert "is_spam" in data
            assert "spam_probability" in data
            assert 0.0 <= data["spam_probability"] <= 1.0

    def test_predict_bulk(self):
        """Test bulk spam classification."""
        with TestClient(spam_nn_app) as client:
            response = client.post(
                "/predict/bulk",
                json={"requests": [self.SPAM_FEATURES, self.HAM_FEATURES]},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
            assert data["model_version"] is not None

    def test_predict_invalid_features(self):
        """Test validation of invalid feature count."""
        with TestClient(spam_nn_app) as client:
            response = client.post("/predict", json={"features": [1, 0, 0]})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        """Test the stats endpoint."""
        with TestClient(spam_nn_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "n_features" in data
            assert "hidden_dim" in data
            assert "training_mode" in data
            assert data["training_mode"] == "supervised"


class TestHousePricePredictionNNAPI:
    """Tests for the house price prediction feedforward NN API."""

    HOUSE_FEATURES = [2000, 3, 2, 85, 10, 2, 5000, 2014, 0, 8.5]

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        with TestClient(house_nn_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_single(self):
        """Test single house price prediction."""
        with TestClient(house_nn_app) as client:
            response = client.post("/predict", json={"features": self.HOUSE_FEATURES})
            assert response.status_code == 200
            data = response.json()
            assert "predicted_price" in data
            assert data["predicted_price"] > 0
            assert "model_version" in data

    def test_predict_bulk(self):
        """Test bulk house price prediction."""
        with TestClient(house_nn_app) as client:
            response = client.post(
                "/predict/bulk",
                json={"requests": [self.HOUSE_FEATURES, self.HOUSE_FEATURES]},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
            assert data["model_version"] is not None

    def test_predict_invalid_features(self):
        """Test validation of invalid feature count."""
        with TestClient(house_nn_app) as client:
            response = client.post("/predict", json={"features": [1, 0, 0]})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        """Test the stats endpoint."""
        with TestClient(house_nn_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "n_features" in data
            assert "hidden_dim" in data
            assert "training_mode" in data
            assert data["training_mode"] == "supervised"


class TestCreditCardFraudDetectionNNAPI:
    """Tests for the credit card fraud detection feedforward NN API."""

    NORMAL_TX = {
        "time_since_last_transaction": 30,
        "transaction_amount": 50,
        "merchant_category": 3,
        "merchant_risk_score": 0.2,
        "cardholder_risk_score": 0.15,
        "distance_from_home": 5,
        "is_online": 0,
        "is_foreign": 0,
        "hour_of_day": 14,
        "day_of_week": 3,
        "account_age_days": 365,
        "recent_transaction_count": 5,
        "avg_transaction_amount_24h": 60,
        "device_risk_score": 0.1,
        "ip_risk_score": 0.1,
    }

    FRAUD_TX = {
        "time_since_last_transaction": 5,
        "transaction_amount": 2000,
        "merchant_category": 5,
        "merchant_risk_score": 0.9,
        "cardholder_risk_score": 0.85,
        "distance_from_home": 300,
        "is_online": 1,
        "is_foreign": 1,
        "hour_of_day": 2,
        "day_of_week": 1,
        "account_age_days": 10,
        "recent_transaction_count": 30,
        "avg_transaction_amount_24h": 2000,
        "device_risk_score": 0.9,
        "ip_risk_score": 0.9,
    }

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        with TestClient(fraud_nn_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_normal(self):
        """Test that normal transaction is not flagged as fraud."""
        with TestClient(fraud_nn_app) as client:
            response = client.post("/predict", json=self.NORMAL_TX)
            assert response.status_code == 200
            data = response.json()
            assert "is_fraud" in data
            assert "fraud_probability" in data
            assert "reconstruction_error" in data
            assert "anomaly_threshold" in data
            assert 0.0 <= data["fraud_probability"] <= 1.0
            assert "model_version" in data

    def test_predict_fraud(self):
        """Test that fraudulent transaction is flagged."""
        with TestClient(fraud_nn_app) as client:
            response = client.post("/predict", json=self.FRAUD_TX)
            assert response.status_code == 200
            data = response.json()
            assert "is_fraud" in data
            assert "fraud_probability" in data

    def test_predict_bulk(self):
        """Test bulk fraud detection."""
        with TestClient(fraud_nn_app) as client:
            response = client.post(
                "/predict/bulk",
                json={"samples": [self.NORMAL_TX, self.FRAUD_TX]},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["samples"]) == 2
            assert data["n_samples"] == 2
            assert "model_version" in data

    def test_predict_invalid_input(self):
        """Test validation of invalid input."""
        with TestClient(fraud_nn_app) as client:
            response = client.post("/predict", json={})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        """Test the stats endpoint."""
        with TestClient(fraud_nn_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "n_features" in data
            assert "hidden_dim" in data
            assert "threshold" in data
            assert "training_mode" in data


class TestHandwrittenDigitRecognitionNNAPI:
    """Tests for the handwritten digit recognition feedforward NN API."""

    def _make_digit_features(self, digit: int = 0) -> list[float]:
        """Create 64 pixel features for a digit."""
        from handwritten_digit_recognition.data import _create_digit_template

        return _create_digit_template(digit).tolist()

    def test_health_endpoint(self):
        """Test the health check endpoint."""
        with TestClient(digit_nn_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_single(self):
        """Test single digit prediction."""
        pixels = self._make_digit_features(0)
        with TestClient(digit_nn_app) as client:
            response = client.post("/predict", json={"pixels": pixels})
            assert response.status_code == 200
            data = response.json()
            assert "digit" in data
            assert 0 <= data["digit"] <= 9
            assert "confidence" in data
            assert 0.0 <= data["confidence"] <= 1.0
            assert "probabilities" in data
            assert len(data["probabilities"]) == 10
            assert "model_version" in data

    def test_predict_bulk(self):
        """Test bulk digit prediction."""
        pixels0 = self._make_digit_features(0)
        pixels1 = self._make_digit_features(1)
        with TestClient(digit_nn_app) as client:
            response = client.post("/predict/bulk", json={"requests": [pixels0, pixels1]})
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
            assert data["model_version"] is not None

    def test_predict_invalid_features(self):
        """Test validation of invalid feature count."""
        with TestClient(digit_nn_app) as client:
            response = client.post("/predict", json={"pixels": [0.0] * 10})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        """Test the stats endpoint."""
        with TestClient(digit_nn_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "n_features" in data
            assert "hidden_dim" in data
            assert "n_classes" in data
            assert "training_mode" in data
            assert data["n_classes"] == 10


class TestLanguageTranslationRNNAPI:
    """Tests for the language translation RNN API."""

    def test_health_endpoint(self):
        with TestClient(translation_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_single(self):
        with TestClient(translation_app) as client:
            response = client.post("/predict", json={"tokens": [0, 1, 2, 3, 4, 5, 6, 7]})
            assert response.status_code == 200
            data = response.json()
            assert "translated_token" in data
            assert "confidence" in data
            assert 0.0 <= data["confidence"] <= 1.0
            assert "model_version" in data

    def test_predict_bulk(self):
        with TestClient(translation_app) as client:
            response = client.post(
                "/predict/bulk",
                json={"requests": [[0, 1, 2, 3, 4, 5, 6, 7], [7, 6, 5, 4, 3, 2, 1, 0]]},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
            assert data["model_version"] is not None

    def test_predict_invalid_tokens(self):
        with TestClient(translation_app) as client:
            response = client.post("/predict", json={"tokens": []})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        with TestClient(translation_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "vocab_size" in data
            assert "hidden_dim" in data
            assert "training_mode" in data


class TestSentimentAnalysisRNNAPI:
    """Tests for the sentiment analysis RNN API."""

    def test_health_endpoint(self):
        with TestClient(sentiment_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_positive(self):
        with TestClient(sentiment_app) as client:
            response = client.post("/predict", json={"tokens": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]})
            assert response.status_code == 200
            data = response.json()
            assert "sentiment" in data
            assert data["sentiment"] in ("positive", "negative")
            assert 0.0 <= data["positive_probability"] <= 1.0
            assert "model_version" in data

    def test_predict_bulk(self):
        with TestClient(sentiment_app) as client:
            response = client.post(
                "/predict/bulk",
                json={
                    "requests": [
                        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                        [10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
                    ]
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
            assert data["model_version"] is not None

    def test_predict_invalid_tokens(self):
        with TestClient(sentiment_app) as client:
            response = client.post("/predict", json={"tokens": []})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        with TestClient(sentiment_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "vocab_size" in data
            assert "hidden_dim" in data
            assert "training_mode" in data


class TestTextGenerationRNNAPI:
    """Tests for the text generation RNN API."""

    def test_health_endpoint(self):
        with TestClient(text_gen_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_single(self):
        with TestClient(text_gen_app) as client:
            response = client.post(
                "/predict",
                json={
                    "tokens": [
                        0,
                        1,
                        2,
                        3,
                        4,
                        5,
                        6,
                        7,
                        8,
                        9,
                        10,
                        11,
                        12,
                        13,
                        14,
                        15,
                        16,
                        17,
                        18,
                        19,
                    ],
                    "n_generate": 5,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "generated_tokens" in data
            assert len(data["generated_tokens"]) == 25
            assert "perplexity" in data
            assert "model_version" in data

    def test_predict_bulk(self):
        with TestClient(text_gen_app) as client:
            response = client.post(
                "/predict/bulk",
                json={
                    "requests": [
                        {
                            "tokens": [
                                0,
                                1,
                                2,
                                3,
                                4,
                                5,
                                6,
                                7,
                                8,
                                9,
                                10,
                                11,
                                12,
                                13,
                                14,
                                15,
                                16,
                                17,
                                18,
                                19,
                            ],
                            "n_generate": 3,
                        }
                    ]
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 1
            assert data["model_version"] is not None

    def test_predict_invalid_tokens(self):
        with TestClient(text_gen_app) as client:
            response = client.post("/predict", json={"tokens": []})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        with TestClient(text_gen_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "vocab_size" in data
            assert "hidden_dim" in data
            assert "training_mode" in data


class TestSpeechRecognitionRNNAPI:
    """Tests for the speech recognition RNN API."""

    def _make_features(self):
        return [[0.1 * i for i in range(16)] for _ in range(20)]

    def test_health_endpoint(self):
        with TestClient(speech_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_single(self):
        with TestClient(speech_app) as client:
            response = client.post("/predict", json={"audio_features": self._make_features()})
            assert response.status_code == 200
            data = response.json()
            assert "word" in data
            assert "word_index" in data
            assert "confidence" in data
            assert "model_version" in data

    def test_predict_bulk(self):
        with TestClient(speech_app) as client:
            response = client.post(
                "/predict/bulk",
                json={"requests": [self._make_features(), self._make_features()]},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
            assert data["model_version"] is not None

    def test_predict_invalid_input(self):
        with TestClient(speech_app) as client:
            response = client.post("/predict", json={"audio_features": []})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        with TestClient(speech_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "n_features" in data
            assert "n_classes" in data
            assert "training_mode" in data


class TestMusicGenerationRNNAPI:
    """Tests for the music generation RNN API."""

    def test_health_endpoint(self):
        with TestClient(music_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_single(self):
        with TestClient(music_app) as client:
            response = client.post(
                "/predict",
                json={
                    "seed_notes": [
                        0,
                        2,
                        4,
                        5,
                        7,
                        9,
                        11,
                        12,
                        14,
                        16,
                        17,
                        19,
                        20,
                        22,
                        24,
                        25,
                        26,
                        28,
                        29,
                        31,
                    ],
                    "n_generate": 5,
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert "generated_notes" in data
            assert len(data["generated_notes"]) == 25
            assert "perplexity" in data
            assert "model_version" in data

    def test_predict_bulk(self):
        with TestClient(music_app) as client:
            response = client.post(
                "/predict/bulk",
                json={
                    "requests": [
                        {
                            "seed_notes": [
                                0,
                                2,
                                4,
                                5,
                                7,
                                9,
                                11,
                                12,
                                14,
                                16,
                                17,
                                19,
                                20,
                                22,
                                24,
                                25,
                                26,
                                28,
                                29,
                                31,
                            ],
                            "n_generate": 3,
                        }
                    ]
                },
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 1
            assert data["model_version"] is not None

    def test_predict_invalid_input(self):
        with TestClient(music_app) as client:
            response = client.post("/predict", json={"seed_notes": []})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        with TestClient(music_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "vocab_size" in data
            assert "hidden_dim" in data
            assert "training_mode" in data


class TestStockMarketPredictionRNNAPI:
    """Tests for the stock market prediction RNN API."""

    def _make_features(self):
        return [[0.1 * i for i in range(5)] for _ in range(20)]

    def test_health_endpoint(self):
        with TestClient(stock_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_single(self):
        with TestClient(stock_app) as client:
            response = client.post("/predict", json={"feature_sequences": self._make_features()})
            assert response.status_code == 200
            data = response.json()
            assert "predicted_price" in data
            assert data["predicted_price"] > 0
            assert "model_version" in data

    def test_predict_bulk(self):
        with TestClient(stock_app) as client:
            response = client.post(
                "/predict/bulk",
                json={"requests": [self._make_features(), self._make_features()]},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
            assert data["model_version"] is not None

    def test_predict_invalid_input(self):
        with TestClient(stock_app) as client:
            response = client.post("/predict", json={"feature_sequences": []})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        with TestClient(stock_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "n_features" in data
            assert "hidden_dim" in data
            assert "training_mode" in data


class TestWeatherForecastingRNNAPI:
    """Tests for the weather forecasting RNN API."""

    def _make_features(self):
        return [[float(i + j) for j in range(5)] for i in range(20)]

    def test_health_endpoint(self):
        with TestClient(weather_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_single(self):
        with TestClient(weather_app) as client:
            response = client.post("/predict", json={"feature_sequences": self._make_features()})
            assert response.status_code == 200
            data = response.json()
            assert "predicted_weather" in data
            assert "model_version" in data

    def test_predict_bulk(self):
        with TestClient(weather_app) as client:
            response = client.post(
                "/predict/bulk",
                json={"requests": [self._make_features(), self._make_features()]},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
            assert data["model_version"] is not None

    def test_predict_invalid_input(self):
        with TestClient(weather_app) as client:
            response = client.post("/predict", json={"feature_sequences": []})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        with TestClient(weather_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "n_features" in data
            assert "hidden_dim" in data
            assert "training_mode" in data


class TestImageCaptioningRNNAPI:
    """Tests for the image captioning RNN API."""

    def _make_pixels(self):
        return [0.5] * 64

    def test_health_endpoint(self):
        with TestClient(image_captioning_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_single(self):
        with TestClient(image_captioning_app) as client:
            response = client.post("/predict", json={"pixels": self._make_pixels()})
            assert response.status_code == 200
            data = response.json()
            assert "caption_tokens" in data
            assert "caption" in data
            assert len(data["caption_tokens"]) == 8
            assert "model_version" in data

    def test_predict_bulk(self):
        with TestClient(image_captioning_app) as client:
            response = client.post(
                "/predict/bulk",
                json={"requests": [self._make_pixels(), self._make_pixels()]},
            )
            assert response.status_code == 200
            data = response.json()
            assert len(data["predictions"]) == 2
            assert data["model_version"] is not None

    def test_predict_invalid_input(self):
        with TestClient(image_captioning_app) as client:
            response = client.post("/predict", json={"pixels": [0.0] * 10})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        with TestClient(image_captioning_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "n_pixels" in data
            assert "vocab_size" in data
            assert "training_mode" in data


# ============================================================
# CNN / DN / CapsNet API Tests
# ============================================================

class TestMedicalImagingCNNAPI:
    """Tests for the medical imaging CNN API."""

    def test_health_endpoint(self):
        with TestClient(medical_imaging_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"
            assert data["model_loaded"] is True

    def test_predict_single(self):
        with TestClient(medical_imaging_app) as client:
            response = client.post("/predict", json={"condition": [0.5] * 64})
            assert response.status_code == 200
            data = response.json()
            assert "condition" in data
            assert "confidence" in data
            assert 0.0 <= data["confidence"] <= 1.0
            assert "model_version" in data

    def test_predict_invalid_pixels(self):
        with TestClient(medical_imaging_app) as client:
            response = client.post("/predict", json={"condition": [0.0] * 10})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        with TestClient(medical_imaging_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "img_size" in data
            assert "training_mode" in data


class TestFacialRecognitionCNNAPI:
    """Tests for the facial recognition CNN API."""

    def test_health_endpoint(self):
        with TestClient(facial_recognition_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    def test_predict_single(self):
        with TestClient(facial_recognition_app) as client:
            response = client.post("/predict", json={"is_owner": [0.5] * 64})
            assert response.status_code == 200
            data = response.json()
            assert "is_owner" in data
            assert "confidence" in data

    def test_predict_invalid_pixels(self):
        with TestClient(facial_recognition_app) as client:
            response = client.post("/predict", json={"is_owner": [0.0] * 10})
            assert response.status_code == 422


class TestVideoSurveillanceCNNAPI:
    """Tests for the video surveillance CNN API."""

    def test_health_endpoint(self):
        with TestClient(video_surveillance_app) as client:
            response = client.get("/health")
            assert response.status_code == 200

    def test_predict_single(self):
        with TestClient(video_surveillance_app) as client:
            response = client.post("/predict", json={"activity": [0.5] * 64})
            assert response.status_code == 200
            data = response.json()
            assert "activity" in data
            assert "model_version" in data


class TestImageSuperResolutionDNAPI:
    """Tests for the image super-resolution DN API."""

    def test_health_endpoint(self):
        with TestClient(image_super_resolution_app) as client:
            response = client.get("/health")
            assert response.status_code == 200

    def test_predict_single(self):
        with TestClient(image_super_resolution_app) as client:
            response = client.post("/predict", json={"high_res_pixels": [0.5] * 64})
            assert response.status_code == 200
            data = response.json()
            assert "high_res_pixels" in data
            assert "confidence" in data

    def test_predict_invalid_input(self):
        with TestClient(image_super_resolution_app) as client:
            response = client.post("/predict", json={"high_res_pixels": [0.0] * 10})
            assert response.status_code == 422


class TestSemanticSegmentationDNAPI:
    """Tests for the semantic segmentation DN API."""

    def test_health_endpoint(self):
        with TestClient(semantic_segmentation_app) as client:
            response = client.get("/health")
            assert response.status_code == 200

    def test_predict_single(self):
        with TestClient(semantic_segmentation_app) as client:
            response = client.post("/predict", json={"mask": [0.5] * 64})
            assert response.status_code == 200
            data = response.json()
            assert "mask" in data
            assert isinstance(data["mask"], list)


class TestGenerativeArtDNAPI:
    """Tests for the generative art DN API."""

    def test_health_endpoint(self):
        with TestClient(generative_art_app) as client:
            response = client.get("/health")
            assert response.status_code == 200

    def test_predict_single(self):
        with TestClient(generative_art_app) as client:
            response = client.post("/predict", json={"reconstructed_pixels": [0.5] * 64})
            assert response.status_code == 200
            data = response.json()
            assert "reconstructed_pixels" in data


class TestAutonomousDrivingCapsNetAPI:
    """Tests for the autonomous driving CapsNet API."""

    def test_health_endpoint(self):
        with TestClient(autonomous_driving_app) as client:
            response = client.get("/health")
            assert response.status_code == 200
            data = response.json()
            assert data["status"] == "healthy"

    def test_predict_single(self):
        with TestClient(autonomous_driving_app) as client:
            response = client.post("/predict", json={"object_type": [0.5] * 64})
            assert response.status_code == 200
            data = response.json()
            assert "object_type" in data
            assert "confidence" in data
            assert 0.0 <= data["confidence"] <= 1.0
            assert "model_version" in data

    def test_predict_invalid_pixels(self):
        with TestClient(autonomous_driving_app) as client:
            response = client.post("/predict", json={"object_type": [0.0] * 10})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        with TestClient(autonomous_driving_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "img_size" in data


class TestMedicalScanAnalysisCapsNetAPI:
    """Tests for the medical scan analysis CapsNet API."""

    def test_health_endpoint(self):
        with TestClient(medical_scan_analysis_app) as client:
            response = client.get("/health")
            assert response.status_code == 200

    def test_predict_single(self):
        with TestClient(medical_scan_analysis_app) as client:
            response = client.post("/predict", json={"structure": [0.5] * 64})
            assert response.status_code == 200
            data = response.json()
            assert "structure" in data
            assert "confidence" in data

    def test_predict_invalid_pixels(self):
        with TestClient(medical_scan_analysis_app) as client:
            response = client.post("/predict", json={"structure": [0.0] * 10})
            assert response.status_code == 422


class TestTextCharRecognitionCapsNetAPI:
    """Tests for the text/character recognition CapsNet API."""

    def test_health_endpoint(self):
        with TestClient(text_char_recognition_app) as client:
            response = client.get("/health")
            assert response.status_code == 200

    def test_predict_single(self):
        with TestClient(text_char_recognition_app) as client:
            response = client.post("/predict", json={"character": [0.5] * 64})
            assert response.status_code == 200
            data = response.json()
            assert "character" in data
            assert "confidence" in data

    def test_predict_invalid_pixels(self):
        with TestClient(text_char_recognition_app) as client:
            response = client.post("/predict", json={"character": [0.0] * 10})
            assert response.status_code == 422

    def test_stats_endpoint(self):
        with TestClient(text_char_recognition_app) as client:
            response = client.get("/stats")
            assert response.status_code == 200
            data = response.json()
            assert "img_size" in data
