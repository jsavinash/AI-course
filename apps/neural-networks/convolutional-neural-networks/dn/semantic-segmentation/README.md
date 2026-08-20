# Semantic Segmentation

Maps object boundaries in synthetic images for autonomous driving and satellite mapping

## Network Type
DN

## Architecture
- Input: 1 channel x 8x8 grayscale images (64 pixels)
- Convolution: 1 filters, 3x3 kernel, ReLU activation
- Pooling: MaxPool2D (2x2)
- Dense layers: 32 hidden units + 1 output units (sigmoid)

## Training
```bash
semantic_segmentation-train --model-dir ./artifacts/models --n-iterations 300 --n-samples 500
```

## Serving API
```bash
uvicorn semantic_segmentation.api:app --host 0.0.0.0 --port 8011
```

### Endpoints
- `GET /` - Service info
- `GET /health` - Health check
- `POST /predict` - Single prediction (64 pixel values)
- `POST /predict/bulk` - Batch predictions (up to 50)
- `GET /stats` - Model statistics
- `GET /drift` - Drift detection
- `GET /metrics` - Prometheus metrics

## Classes
['background', 'foreground']

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
