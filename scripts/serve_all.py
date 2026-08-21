"""Run all serving APIs concurrently.

Existing ML/RL/NN APIs (ports 8000-8019):
  pizza price            -> 8000
  spam classification    -> 8001
  market segmentation    -> 8002
  recommendation engine  -> 8003
  anomaly detection      -> 8004
  robot maze navigation  -> 8005
  semi-supervised email  -> 8006
  self-supervised monitor-> 8007
  email spam detection   -> 8008
  house price prediction -> 8009
  credit card fraud      -> 8010
  handwritten digit      -> 8011
  language translation   -> 8012
  sentiment analysis     -> 8013
  text generation RNN    -> 8014
  speech recognition     -> 8015
  music generation       -> 8016
  stock market prediction-> 8017
  weather forecasting    -> 8018
  image captioning       -> 8019

Generative-AI APIs (ports 8020-8026):
  code generation        -> 8020
  image generation       -> 8021
  prompt engineering     -> 8022
  retrieval-augmented gen-> 8023
  text gen (generative)  -> 8024
  tool use / func calling-> 8025
  video generation       -> 8026
"""

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
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "code_generation.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8020",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "image_generation.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8021",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "prompt_engineering.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8022",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "retrieval_augmented_generation.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8023",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "text_gen.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8024",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "tool_use_and_functional_calling.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8025",
        ],
        env=env,
    ),
    subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "video_generation.api:app",
            "--host",
            "0.0.0.0",
            "--port",
            "8026",
        ],
        env=env,
    ),
]

print("Serving APIs:")
print("  Pizza Price Prediction API       -> http://localhost:8000")
print("  Spam Classification API          -> http://localhost:8001")
print("  Market Segmentation API          -> http://localhost:8002")
print("  Recommendation Engine API        -> http://localhost:8003")
print("  Anomaly Detection API            -> http://localhost:8004")
print("  Robot Maze Navigation API        -> http://localhost:8005")
print("  Semi-Supervised Email API        -> http://localhost:8006")
print("  Self-Supervised Monitoring API   -> http://localhost:8007")
print("  Email Spam Detection NN API      -> http://localhost:8008")
print("  House Price Prediction NN API    -> http://localhost:8009")
print("  Credit Card Fraud Detection NN   -> http://localhost:8010")
print("  Handwritten Digit Recognition NN -> http://localhost:8011")
print("  Language Translation RNN        -> http://localhost:8012")
print("  Sentiment Analysis RNN          -> http://localhost:8013")
print("  Text Generation RNN             -> http://localhost:8014")
print("  Speech Recognition RNN          -> http://localhost:8015")
print("  Music Generation RNN            -> http://localhost:8016")
print("  Stock Market Prediction RNN     -> http://localhost:8017")
print("  Weather Forecasting RNN         -> http://localhost:8018")
print("  Image Captioning RNN            -> http://localhost:8019")
print("  Code Generation API             -> http://localhost:8020")
print("  Image Generation API            -> http://localhost:8021")
print("  Prompt Engineering API          -> http://localhost:8022")
print("  Retrieval-Augmented Generation  -> http://localhost:8023")
print("  Text Generation API (GenAI)     -> http://localhost:8024")
print("  Tool Use and Functional Calling -> http://localhost:8025")
print("  Video Generation API            -> http://localhost:8026")
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
