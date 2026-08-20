"""Run all serving APIs concurrently (pizza on 8000, spam on 8001, market on 8002, recommendation on 8003, anomaly on 8004, robot maze on 8005, semi-supervised-email on 8006, self-supervised-monitoring on 8007, email-spam-detection-nn on 8008, house-price-prediction-nn on 8009, credit-card-fraud-detection-nn on 8010, handwritten-digit-recognition-nn on 8011, language-translation-rnn on 8012, sentiment-analysis-rnn on 8013, text-generation-rnn on 8014, speech-recognition-rnn on 8015, music-generation-rnn on 8016, stock-market-prediction-rnn on 8017, weather-forecasting-rnn on 8018, image-captioning-rnn on 8019)."""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "artifacts" / "models"

# Ensure model dir exists
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Environment for both servers
env = os.environ.copy()
env["MODEL_DIR"] = str(MODEL_DIR)

# Start both uvicorn servers as subprocesses
servers = [
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "pizza_price.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8000",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "spam_classification.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8001",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "market_segmentation.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8002",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "recommendation_engine.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8003",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "anomaly_detection.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8004",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "robot_maze.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8005",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "semi_supervised_email.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8006",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "self_supervised_monitoring.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8007",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "email_spam_detection.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8008",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "house_price_prediction.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8009",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "credit_card_fraud_detection.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8010",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "handwritten_digit_recognition.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8011",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "language_translation.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8012",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "sentiment_analysis.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8013",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "text_generation.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8014",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "speech_recognition.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8015",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "music_generation.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8016",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "stock_market_prediction.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8017",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "weather_forecasting.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8018",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "image_captioning.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8019",
        ],
        env=env,
    ),
]

print("Serving APIs:")
print("  Pizza Price Prediction API  -> http://localhost:8000")
print("  Spam Classification API     -> http://localhost:8001")
print("  Market Segmentation API     -> http://localhost:8002")
print("  Recommendation Engine API   -> http://localhost:8003")
print("  Anomaly Detection API       -> http://localhost:8004")
print("  Robot Maze Navigation API   -> http://localhost:8005")
print("  Semi-Supervised Email API   -> http://localhost:8006")
print("  Self-Supervised Monitoring API -> http://localhost:8007")
print("  Email Spam Detection NN API    -> http://localhost:8008")
print("  House Price Prediction NN API  -> http://localhost:8009")
print("  Credit Card Fraud Detection NN -> http://localhost:8010")
print("  Handwritten Digit Recognition NN -> http://localhost:8011")
print("  Language Translation RNN          -> http://localhost:8012")
print("  Sentiment Analysis RNN            -> http://localhost:8013")
print("  Text Generation RNN               -> http://localhost:8014")
print("  Speech Recognition RNN            -> http://localhost:8015")
print("  Music Generation RNN              -> http://localhost:8016")
print("  Stock Market Prediction RNN       -> http://localhost:8017")
print("  Weather Forecasting RNN           -> http://localhost:8018")
print("  Image Captioning RNN              -> http://localhost:8019")
print("Press Ctrl+C to stop all servers.\n")

try:
    # Wait for both processes
    for proc in servers:
        proc.wait()
except KeyboardInterrupt:
    print("\nShutting down servers...")
    for proc in servers:
        proc.terminate()
    for proc in servers:
        proc.wait()
    print("All servers stopped.")
    sys.exit(0)
