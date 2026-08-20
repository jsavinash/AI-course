# Semi-Supervised Email Classification

Production-ready self-training classifier for email spam detection using semi-supervised learning.

## Learning Type: Semi-Supervised Learning

This example demonstrates **semi-supervised learning** — the model leverages both a small amount of labeled data and a large amount of unlabeled data to improve classification accuracy. This is useful when labeling data is expensive but unlabeled data is abundant.

## Problem

Classify emails as spam (1) or ham (0) using semi-supervised self-training. A small labeled subset is used to bootstrap the model, which then iteratively labels high-confidence unlabeled samples to expand its training data.

## Features

| Feature        | Description                              |
|----------------|------------------------------------------|
| `has_free`     | Email contains "free" keyword            |
| `has_win`      | Email contains "win" keyword               |
| `has_link`     | Email contains a link                    |
| `has_exclamation` | Email has 3+ exclamation marks         |
| `has_meeting`  | Email contains "meeting" keyword         |
| `length_score` | Email length score (1-10)                |
| `has_caps`     | Email contains excessive caps            |

## Approach

- **Base Model**: Logistic Regression (from scratch, numpy-only)
- **Self-Training Algorithm**:
  1. Train base model on initial labeled data
  2. Predict probabilities on unlabeled data
  3. Add high-confidence predictions (above `confidence_threshold`) to labeled set
  4. Retrain on expanded labeled set
  5. Repeat until convergence or `max_iterations` reached
- **Training Modes**: 
  - `semi-supervised`: Pseudo-labels were added during training
  - `supervised`: No unlabeled data or no confident predictions made

## Architecture

```
apps/machine-learning/semi-supervised/semi-supervised-email/
├── pyproject.toml              # Package configuration
├── src/semi_supervised_email/
│   ├── __init__.py             # Package exports
│   ├── data.py                 # Synthetic email data generation
│   ├── model.py                # LogisticRegression + SelfTrainingClassifier
│   ├── train.py                # Training pipeline script
│   └── api.py                  # FastAPI serving API
└── README.md                   # This file
```

## Quick Start

### Training

```bash
make train-semi-supervised-email
```

Or directly:

```bash
uv run python -m semi_supervised_email.train \
  --model-dir ./artifacts/models \
  --model-version 1.0.0 \
  --labeled-ratio 0.1
```

### Serving

```bash
make serve-semi-supervised-email
```

Or directly:

```bash
MODEL_DIR=./artifacts/models uv run uvicorn semi_supervised_email.api:app --host 0.0.0.0 --port 8006
```

### API Usage

```bash
# Health check
curl http://localhost:8006/health

# Predict spam
curl -X POST http://localhost:8006/predict \
  -H "Content-Type: application/json" \
  -d '{
    "has_free": 1, "has_win": 1, "has_link": 1,
    "has_exclamation": 1, "has_meeting": 0,
    "length_score": 8, "has_caps": 1
  }'

# Bulk prediction
curl -X POST http://localhost:8006/predict/bulk \
  -H "Content-Type: application/json" \
  -d '[{...}, {...}]'

# Get training statistics
curl http://localhost:8006/stats
```

## API Endpoints

| Method | Path             | Description                |
|--------|------------------|----------------------------|
| GET    | `/health`        | Health check               |
| POST   | `/predict`       | Classify single email      |
| POST   | `/predict/bulk`  | Classify multiple emails   |
| GET    | `/stats`         | Training statistics        |
| GET    | `/drift`         | Drift detection            |
| GET    | `/metrics`       | Prometheus metrics         |
| POST   | `/reload`        | Reload model               |

## Model Parameters

### LogisticRegression (base model)

| Parameter        | Default   | Description                       |
|------------------|-----------|-----------------------------------|
| `learning_rate`  | `0.1`     | Gradient descent step size        |
| `n_iterations`   | `2000`    | Number of training iterations     |

### SelfTrainingClassifier

| Parameter                | Default   | Description                              |
|--------------------------|-----------|------------------------------------------|
| `confidence_threshold`   | `0.95`    | Min probability to add pseudo-label      |
| `max_iterations`         | `10`      | Max self-training iterations             |
| `min_labeled_ratio`      | `0.8`     | Stop if labeled ratio exceeds this       |
| `random_seed`            | `42`      | Random seed for reproducibility          |

## How Self-Training Works

1. **Initialization**: A small subset of data is labeled (e.g., 10% of all samples).
2. **Base Training**: The base logistic regression model is trained on labeled data.
3. **Pseudo-Labeling**: The model predicts on unlabeled data. Samples with confidence above the threshold are added to the labeled set with their predicted labels.
4. **Retraining**: The model is retrained on the expanded labeled set.
5. **Iteration**: Steps 3-4 repeat until convergence (no more confident predictions, labeled ratio exceeds threshold, or max iterations reached).

## Key Concepts Demonstrated

- Semi-supervised learning with self-training
- Pseudo-labeling with confidence thresholding
- Iterative model retraining
- Training mode tracking (supervised vs. semi-supervised)
- Labeled data growth history
- Production API serving with FastAPI
- Model versioning with registry
