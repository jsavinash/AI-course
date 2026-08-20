# Medical Scan Analysis

Identifies overlapping body structures in complex 3D medical scans

## Network Type
CapsNet

## Architecture
- Input: 1 channel x 8x8 grayscale images (64 pixels)
- Convolution: 4 filters, 3x3 kernel, ReLU activation
- Pooling: MaxPool2D (2x2)
- Dense layers: 32 hidden units + 4 output units (softmax)

## Training
```bash
medical_scan_analysis-train --model-dir ./artifacts/models --n-iterations 300 --n-samples 500
```

## Serving API
```bash
uvicorn medical_scan_analysis.api:app --host 0.0.0.0 --port 8013
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
['bone', 'organ', 'vessel', 'tissue']

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
