# Spam Classification (Supervised Learning)

Production-ready logistic regression model for email spam classification.

## Learning Type: Supervised (Binary Classification)

This example demonstrates **supervised learning** — the model is trained on labeled data where each input (email features) has a known label (spam or not spam).

## Problem

Classify emails as spam (1) or not spam (0) based on extracted features.

**Model**: Binary classification using logistic regression
```
z = X·w + b
p = sigmoid(z)
prediction = 1 if p >= 0.5 else 0
```

## Features

| Feature     | Description                          |
|-------------|--------------------------------------|
| `free`      | Email contains "free" keyword        |
| `win`       | Email contains "win" keyword         |
| `link`      | Email contains a link                |
| `!!!`       | Email has 3+ exclamation marks       |
| `meeting`   | Email contains "meeting" keyword     |

## Approach

- **Algorithm**: Logistic Regression trained via gradient descent on Binary Cross-Entropy loss
- **Implementation**: Pure NumPy, no scikit-learn dependency
- **Activation**: Numerically stable sigmoid function
- **Evaluation**: Accuracy, precision, recall, F1, ROC AUC

## Architecture

```
machine-learning/supervised/spam-classification/
├── pyproject.toml              # Package configuration
├── src/spam_classification/
│   ├── __init__.py             # Package exports
│   ├── data.py                 # Training data and text preprocessing
│   ├── model.py                # LogisticRegression (from scratch)
│   ├── train.py                # Training pipeline script
│   └── api.py                  # FastAPI serving API
└── README.md                   # This file
```

## Quick Start

### Training

```bash
make train-spam
```

Or directly:

```bash
uv run python -m spam_classification.train --model-dir ./artifacts/models --model-version 1.0.0
```

### Serving

```bash
make serve-spam
```

Or directly:

```bash
MODEL_DIR=./artifacts/models uv run uvicorn spam_classification.api:app --host 0.0.0.0 --port 8001
```

### API Usage

```bash
# Health check
curl http://localhost:8001/health

# Predict with features
curl -X POST http://localhost:8001/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1, 1, 1, 1, 0]}'

# Predict from raw email text
curl -X POST http://localhost:8001/predict/email \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You have been selected to receive a FREE gift!!!"}'
```

## API Endpoints

| Method | Path              | Description                     |
|--------|-------------------|---------------------------------|
| GET    | `/health`         | Health check                    |
| POST   | `/predict`         | Predict from feature vector     |
| POST   | `/predict/email`   | Predict from raw email text     |
| GET    | `/metrics`        | Prometheus metrics              |
| POST   | `/reload`         | Reload model                    |

## Model Parameters

| Parameter        | Default   | Description                       |
|------------------|-----------|-----------------------------------|
| `learning_rate`  | `0.1`     | Gradient descent step size        |
| `n_iterations`   | `2000`    | Number of training iterations     |

## Key Concepts Demonstrated

- Binary cross-entropy loss
- Sigmoid activation with numerical stability
- Multi-metric evaluation (precision, recall, F1, AUC)
- Text-to-feature preprocessing
- Feature validation with schemas
- Production API serving with FastAPI
- Prometheus metrics integration
- Model versioning with registry
