# Email Spam Detection (Feedforward Neural Network)

Production-ready feedforward neural network for email spam classification, built from scratch with NumPy.

## Learning Type: Feedforward Neural Network - Binary Classification

This example demonstrates a **feedforward neural network** (multi-layer perceptron) for classifying emails as SPAM or NOT spam. The entire network — including forward propagation, backpropagation, and gradient descent training — is implemented from scratch using only NumPy.

## Problem

Classify emails into two categories: spam (unsolicited/bulk) or ham (legitimate). Spam emails often contain specific keywords, high urgency language, and suspicious links.

## Features

| Feature                | Description                          |
|------------------------|--------------------------------------|
| `has_free`             | Whether email contains "free"        |
| `has_win`              | Whether email contains "win"         |
| `has_link`             | Whether email contains a link        |
| `has_exclamation`      | Whether email has exclamation marks  |
| `has_meeting`          | Whether email mentions a meeting     |
| `email_length`         | Length of email (normalized score)   |
| `has_caps`             | Whether email has all-caps words     |
| `has_money`            | Whether email mentions money amounts |
| `num_links`            | Number of links in email             |
| `num_exclamations`     | Number of exclamation marks          |
| `has_urgent`           | Whether email has "urgent" language  |
| `sender_reputation`    | Sender reputation score (0-1)        |

## Approach

- **Architecture**: MLP with 1 hidden layer
  - **Input layer**: 12 features
  - **Hidden layer**: `hidden_dim` neurons with ReLU activation
  - **Output layer**: 1 neuron with Sigmoid activation (spam probability)
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Batch Gradient Descent with He initialization
- **Regularization**: L2 (weight decay) to prevent overfitting
- **Early stopping**: Stops when loss converges
- **Validation tracking**: Monitors validation accuracy during training
- **Implementation**: Pure NumPy, no external ML libraries

## Architecture

```
apps/neural-networks/feedforward-neural-networks/classification/email-spam-detection/
├── pyproject.toml              # Package configuration
├── src/email_spam_detection/
│   ├── __init__.py             # Package exports
│   ├── data.py                 # Synthetic email feature data
│   ├── model.py                # SpamDetectionNN (from scratch)
│   ├── train.py                # Training pipeline (BCE loss + backprop)
│   └── api.py                  # FastAPI serving with observability
└── README.md                   # This file
```

## Quick Start

### Training

```bash
make train-email-spam-detection-nn
```

Or directly:

```bash
uv run python -m email_spam_detection.train \
  --model-dir ./artifacts/models \
  --model-version 1.0.0
```

### Serving

```bash
MODEL_DIR=./artifacts/models uv run uvicorn email_spam_detection.api:app --host 0.0.0.0 --port 8008
```

### API Usage

```bash
# Health check
curl http://localhost:8008/health

# Predict spam for a single email (12 features)
curl -X POST http://localhost:8008/predict \
  -H "Content-Type: application/json" \
  -d '{"features": [1,1,1,1,0,8,1,1,5,8,1,0.2]}'

# Predict spam for multiple emails
curl -X POST http://localhost:8008/predict/bulk \
  -H "Content-Type: application/json" \
  -d '{"requests": [[1,1,1,1,0,8,1,1,5,8,1,0.2]]}'

# Get model statistics
curl http://localhost:8008/stats
```

## API Endpoints

| Method | Path             | Description              |
|--------|------------------|--------------------------|
| GET    | `/health`        | Health check             |
| POST   | `/predict`       | Classify single email    |
| POST   | `/predict/bulk`  | Classify multiple emails |
| GET    | `/stats`         | Model statistics         |
| GET    | `/drift`         | Drift detection          |
| GET    | `/metrics`       | Prometheus metrics       |
| POST   | `/reload`        | Reload model             |

## Model Parameters

| Parameter           | Default  | Description                              |
|---------------------|----------|------------------------------------------|
| `hidden_dim`        | `16`     | Neurons in hidden layer                  |
| `learning_rate`     | `0.01`   | Gradient descent step size               |
| `n_iterations`      | `1000`   | Maximum training iterations              |
| `weight_decay`      | `0.001`  | L2 regularization strength             |
| `hidden_activation` | `relu`   | Hidden layer activation                  |
| `random_seed`       | `42`     | Random seed for reproducibility          |

## How It Works

1. **Training**: The MLP is trained using supervised learning with binary cross-entropy loss. Backpropagation computes gradients through the network, and gradients are used to update weights via gradient descent.

2. **Prediction**: At inference time, the sigmoid output gives the probability the email is spam. A threshold of 0.5 determines the final classification.

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Precision**: Of predicted spams, how many were truly spam
- **Recall**: Of actual spams, how many were detected
- **F1 Score**: Harmonic mean of precision and recall
- **ROC-AUC**: Area under the ROC curve

## Key Concepts Demonstrated

- Feedforward neural network (MLP) from scratch
- Forward and backward propagation
- He weight initialization
- Binary cross-entropy loss
- L2 regularization (weight decay)
- Early stopping based on loss convergence
- Multi-feature binary classification
- Production API serving with FastAPI
- Model versioning with registry
