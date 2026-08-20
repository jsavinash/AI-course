# RBM Feature Learning

Stochastic neural networks that can learn a probability distribution over its set of inputs using Contrastive Divergence.

## Network Type
RBM (Restricted Boltzmann Machine)

## Architecture
- Binary visible units (32) <-> Binary hidden units (16)
- Fully connected undirected bipartite graph
- No visible-visible or hidden-hidden connections

## Training
```bash
rbm_feature_learning-train --model-dir ./artifacts/models --n-epochs 100 --n-samples 500
```

## Serving API
```bash
uvicorn rbm_feature_learning.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Extract features and reconstruct (32 binary features)
- `POST /predict/bulk` - Batch predictions (up to 50)
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Classes
[] (binary feature values [0, 1])

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
