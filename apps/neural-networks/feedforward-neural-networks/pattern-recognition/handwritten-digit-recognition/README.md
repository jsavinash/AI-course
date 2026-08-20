# Handwritten Digit Recognition (Feedforward Neural Network)

Production-ready feedforward neural network for handwritten digit recognition (0-9), built from scratch with NumPy using softmax cross-entropy.

## Learning Type: Feedforward Neural Network - Multi-class Classification

This example demonstrates a **feedforward neural network** (multi-layer perceptron) for recognizing handwritten digits from 8x8 pixel images. The network uses softmax activation in the output layer for multi-class classification, with the entire implementation built from scratch using only NumPy.

## Problem

Classify handwritten digit images (represented as 8x8 = 64 pixel grids) into one of 10 classes (digits 0-9). Each pixel value ranges from 0 to 1, where 0 is background and 1 is ink.

## Features

The input is a flattened 8x8 grayscale image represented as 64 features:

| Feature Names         | Description                          |
|-----------------------|--------------------------------------|
| `pixel_0` through `pixel_63` | Pixel values (0.0 to 1.0) from an 8x8 grid |

## Approach

- **Architecture**: MLP with 1 hidden layer
  - **Input layer**: 64 pixels (8x8 image)
  - **Hidden layer**: `hidden_dim` neurons with ReLU activation
  - **Output layer**: 10 neurons with Softmax activation (class probabilities)
- **Loss**: Categorical Cross-Entropy (softmax)
- **Optimizer**: Batch Gradient Descent with He initialization
- **Regularization**: L2 (weight decay)
- **Early stopping**: Stops when loss converges
- **Validation tracking**: Monitors validation accuracy during training
- **Implementation**: Pure NumPy, no external ML libraries

## Architecture

```
apps/neural-networks/feedforward-neural-networks/pattern-recognition/handwritten-digit-recognition/
├── pyproject.toml                    # Package configuration
├── src/handwritten_digit_recognition/
│   ├── __init__.py                   # Package exports
│   ├── data.py                       # Synthetic 8x8 pixel digit images
│   ├── model.py                      # DigitRecognitionNN (from scratch)
│   ├── train.py                      # Training pipeline (cross-entropy + backprop)
│   └── api.py                        # FastAPI serving with observability
└── README.md                         # This file
```

## Quick Start

### Training

```bash
make train-handwritten-digit-recognition-nn
```

Or directly:

```bash
uv run python -m handwritten_digit_recognition.train \
  --model-dir ./artifacts/models \
  --model-version 1.0.0
```

### Serving

```bash
MODEL_DIR=./artifacts/models uv run uvicorn handwritten_digit_recognition.api:app --host 0.0.0.0 --port 8011
```

### API Usage

```bash
# Health check
curl http://localhost:8011/health

# Predict digit from 64 pixel values
curl -X POST http://localhost:8011/predict \
  -H "Content-Type: application/json" \
  -d '{"pixels": [0,0,1,1,1,1,0,0, 0,1,0,0,0,0,1,0, ...]}'  # 64 values

# Predict multiple digits
curl -X POST http://localhost:8011/predict/bulk \
  -H "Content-Type: application/json" \
  -d '{"requests": [[0,0,1,1,1,1,0,0,...]]}'

# Get model statistics
curl http://localhost:8011/stats
```

## API Endpoints

| Method | Path             | Description              |
|--------|------------------|--------------------------|
| GET    | `/health`        | Health check             |
| POST   | `/predict`       | Recognize single digit   |
| POST   | `/predict/bulk`  | Recognize multiple digits|
| GET    | `/stats`         | Model statistics         |
| GET    | `/drift`         | Drift detection          |
| GET    | `/metrics`       | Prometheus metrics       |
| POST   | `/reload`        | Reload model             |

## Model Parameters

| Parameter           | Default  | Description                              |
|---------------------|----------|------------------------------------------|
| `hidden_dim`        | `64`     | Neurons in hidden layer                  |
| `learning_rate`     | `0.1`    | Gradient descent step size               |
| `n_iterations`      | `1000`   | Maximum training iterations              |
| `weight_decay`      | `0.0001` | L2 regularization strength               |
| `hidden_activation` | `relu`   | Hidden layer activation                  |
| `random_seed`       | `42`     | Random seed for reproducibility          |

## How It Works

1. **Training**: The MLP is trained using supervised multi-class classification with categorical cross-entropy loss and softmax activation. Backpropagation computes gradients through the network, and gradient descent updates the weights.

2. **Prediction**: The softmax output layer produces a probability distribution over the 10 digit classes. The class with the highest probability is the predicted digit.

3. **Data Generation**: Each digit has a template pattern (8x8 grid) with added Gaussian noise to simulate handwriting variation.

## Evaluation Metrics

- **Accuracy**: Overall classification accuracy
- **Macro Precision**: Average precision across all 10 classes
- **Macro Recall**: Average recall across all 10 classes
- **Macro F1**: Average F1 score across all 10 classes
- **Confusion Matrix**: Per-class true/false positives/negatives

## Key Concepts Demonstrated

- Feedforward neural network (MLP) from scratch
- Multi-class classification with softmax
- Forward and backward propagation
- Categorical cross-entropy loss
- He weight initialization
- L2 regularization (weight decay)
- Early stopping based on loss convergence
- Softmax activation for multi-class output
- Per-class precision, recall, and F1
- Confusion matrix computation
- Production API serving with FastAPI
