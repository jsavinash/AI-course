"""Run all serving APIs concurrently (pizza on 8000, spam on 8001, market on 8002, recommendation on 8003, anomaly on 8004, robot maze on 8005, semi-supervised-email on 8006, self-supervised-monitoring on 8007)."""

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
