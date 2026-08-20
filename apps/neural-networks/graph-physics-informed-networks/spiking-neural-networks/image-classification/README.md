# SNN Image Classification

Spiking Neural Networks that use neuromorphic computing where neurons communicate via discrete spikes, closely mimicking biological brain activity.

## Network Type
SNN (Spiking Neural Network)

## Architecture
- Input: (batch, 64) flattened 8x8 images with rate encoding
- LIF Layer 1: 64 -> 128 neurons (Leaky Integrate-and-Fire + ReLU)
- LIF Layer 2: 128 -> 128 neurons (Leaky Integrate-and-Fire + ReLU)
- Output: 128 -> 10 classes (softmax)

## Key Features
- **Membrane dynamics**: tau_m * dv/dt = -(v - v_rest) + R * I
- **Spike threshold**: v >= v_threshold triggers a spike
- **Temporal coding**: Multiple simulation timesteps per forward pass
- **Surrogate gradient**: Allows backpropagation through discontinuous spike events

## Training
```bash
snn_image_classification-train --model-dir ./artifacts/models --n-iterations 200 --n-timesteps 10
```

## Serving API
```bash
uvicorn snn_image_classification.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Classify flattened 8x8 image (64 pixels)
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Input Format
64 pixels (8x8 image) normalized to [0, 1]

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
