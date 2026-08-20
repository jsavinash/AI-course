# Credit Card Fraud Detection (Feedforward Neural Network)

Production-ready feedforward autoencoder for credit card fraud detection, built from scratch with NumPy.

## Learning Type: Feedforward Neural Network - Anomaly Detection

This example demonstrates an **autoencoder-based feedforward neural network** for detecting fraudulent credit card transactions. The autoencoder is trained to reconstruct normal transactions; anomalous patterns produce high reconstruction errors and are flagged as potential fraud.

## Problem

Detect fraudulent credit card transactions without requiring labeled fraud examples for training. The model learns the patterns of legitimate transactions and flags deviations as potential fraud.

## Features

| Feature                        | Description                          |
|--------------------------------|--------------------------------------|
| `time_since_last_transaction`  | Minutes since last transaction       |
| `transaction_amount`           | Transaction amount in USD            |
| `merchant_category`            | Merchant category code (0-11)        |
| `merchant_risk_score`          | Merchant risk score (0-1)            |
| `cardholder_risk_score`        | Cardholder risk score (0-1)          |
| `distance_from_home`           | Distance from home in miles          |
| `is_online`                    | Whether transaction is online (0/1)  |
| `is_foreign`                   | Whether transaction is foreign (0/1) |
| `hour_of_day`                  | Hour of day (0-23)                   |
| `day_of_week`                  | Day of week (0-6)                    |
| `account_age_days`             | Account age in days                  |
| `recent_transaction_count`     | Recent transaction count             |
| `avg_transaction_amount_24h`     | Avg transaction amount (24h)         |
| `device_risk_score`            | Device risk score (0-1)              |
| `ip_risk_score`                | IP risk score (0-1)                  |

## Approach

- **Architecture**: Autoencoder with 1 hidden layer (bottleneck)
  - **Input layer**: 15 features
  - **Hidden layer (encoder)**: `hidden_dim` neurons with ReLU activation
  - **Output layer (decoder)**: 15 neurons with linear activation (reconstruction)
- **Training**: Supervised on normal transactions only — the model learns to reconstruct normal data
- **Loss**: Mean Squared Error (reconstruction loss)
- **Optimizer**: Batch Gradient Descent with He initialization
- **Anomaly detection**: Reconstruction error on test data is compared against a threshold (95th percentile of training errors)
- **Regularization**: L2 (weight decay)
- **Implementation**: Pure NumPy, no external ML libraries

## Architecture

```
apps/neural-networks/feedforward-neural-networks/anomaly-detection/credit-card-fraud-detection/
├── pyproject.toml                  # Package configuration
├── src/credit_card_fraud_detection/
│   ├── __init__.py                 # Package exports
│   ├── data.py                     # Synthetic transaction data
│   ├── model.py                    # FraudDetectionAutoencoder (from scratch)
│   ├── train.py                    # Training pipeline (reconstruction loss)
│   └── api.py                      # FastAPI serving with observability
└── README.md                       # This file
```

## Quick Start

### Training

```bash
make train-credit-card-fraud-detection-nn
```

Or directly:

```bash
uv run python -m credit_card_fraud_detection.train \
  --model-dir ./artifacts/models \
  --model-version 1.0.0
```

### Serving

```bash
MODEL_DIR=./artifacts/models uv run uvicorn credit_card_fraud_detection.api:app --host 0.0.0.0 --port 8010
```

### API Usage

```bash
# Health check
curl http://localhost:8010/health

# Detect fraud for a single transaction
curl -X POST http://localhost:8010/predict \
  -H "Content-Type: application/json" \
  -d '{
    "time_since_last_transaction": 5,
    "transaction_amount": 1200,
    "merchant_category": 3,
    "merchant_risk_score": 0.8,
    "cardholder_risk_score": 0.75,
    "distance_from_home": 200,
    "is_online": 1,
    "is_foreign": 1,
    "hour_of_day": 2,
    "day_of_week": 3,
    "account_age_days": 10,
    "recent_transaction_count": 30,
    "avg_transaction_amount_24h": 1500,
    "device_risk_score": 0.9,
    "ip_risk_score": 0.85
  }'

# Detect fraud for multiple transactions
curl -X POST http://localhost:8010/predict/bulk \
  -H "Content-Type: application/json" \
  -d '{"samples": [...]}'

# Get model statistics
curl http://localhost:8010/stats
```

## API Endpoints

| Method | Path             | Description                    |
|--------|------------------|--------------------------------|
| GET    | `/health`        | Health check                   |
| POST   | `/predict`       | Detect fraud (single)          |
| POST   | `/predict/bulk`  | Detect fraud (bulk)            |
| GET    | `/stats`         | Model statistics               |
| GET    | `/drift`         | Drift detection                |
| GET    | `/metrics`       | Prometheus metrics             |
| POST   | `/reload`        | Reload model                   |

## Model Parameters

| Parameter               | Default  | Description                              |
|-------------------------|----------|------------------------------------------|
| `hidden_dim`            | `8`      | Neurons in hidden (bottleneck) layer     |
| `learning_rate`         | `0.001`  | Gradient descent step size               |
| `n_iterations`          | `2000`   | Maximum training iterations              |
| `threshold_percentile`  | `95.0`   | Percentile for anomaly threshold         |
| `weight_decay`          | `0.0001` | L2 regularization strength               |
| `hidden_activation`     | `relu`   | Hidden layer activation                  |
| `random_seed`           | `42`     | Random seed for reproducibility          |

## How It Works

1. **Training**: The autoencoder is trained on **normal transactions only**. It learns to compress and reconstruct normal data patterns. The reconstruction loss decreases as the model learns the normal data manifold.

2. **Threshold Setting**: The anomaly threshold is set at the 95th percentile of reconstruction errors on training data.

3. **Inference**: When a new transaction comes in:
   - The model attempts to reconstruct it
   - Reconstruction error (MSE) is computed
   - If error > threshold → fraud detected

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Of predicted frauds, how many were truly fraudulent
- **Recall**: Of actual frauds, how many were detected
- **F1 Score**: Harmonic mean of precision and recall
- **False Positive Rate**: Rate of legitimate transactions flagged as fraud
- **Reconstruction Error**: MSE between input and reconstruction

## Key Concepts Demonstrated

- Feedforward autoencoder from scratch
- Anomaly detection via reconstruction error
- Encoder-decoder architecture (bottleneck)
- Unsupervised-style training (normal data only)
- He weight initialization
- Mean squared error loss
- L2 regularization
- Production API serving with FastAPI
