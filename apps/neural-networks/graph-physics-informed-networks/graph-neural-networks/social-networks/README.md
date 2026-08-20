# GNN Social Networks

Graph Neural Networks designed to optimize directly on graph structures, highly effective for social networks or molecular mapping.

## Network Type
GNN (Graph Neural Network)

## Architecture
- Input: Node features (32-dim) + Adjacency matrix (20x20)
- GCN Layer 1: 32 -> 16 features (ReLU)
- GCN Layer 2: 16 -> 16 features (ReLU)
- Output: 16 -> 2 classes (softmax)

## Training
```bash
gnn_social_networks-train --model-dir ./artifacts/models --n-nodes 20 --n-iterations 200
```

## Serving API
```bash
uvicorn gnn_social_networks.api:app --host 0.0.0.0 --port 8010
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Node classification (32 features + adjacency row)
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Classes
[0, 1] binary community labels

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
