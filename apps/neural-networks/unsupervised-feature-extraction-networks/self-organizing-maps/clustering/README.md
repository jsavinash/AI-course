# Self-Organizing Maps

Unsupervised networks that produce a low-dimensional, discretized representation of the input space of the training samples.

## Network Type
SOM (Self-Organizing Map)

## Architecture
- Input: 32 feature vectors
- Competitive layer: 5x5 grid of neurons (25 neurons)
- Each neuron has a weight vector of dimension 32
- Best Matching Unit (BMU) found via Euclidean distance
- Neighborhood updated using Gaussian function

## Training
```bash
self_organizing_maps-train --model-dir ./artifacts/models --n-iterations 300 --n-samples 500
```

## Serving API
```bash
uvicorn self_organizing_maps.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Find BMU for input features (32 features)
- `POST /predict/bulk` - Batch predictions (up to 50)
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Classes
[0-5] (5 cluster centers)

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
