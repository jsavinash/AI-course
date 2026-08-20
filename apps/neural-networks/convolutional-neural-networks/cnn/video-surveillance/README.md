# Video Surveillance

Tracks crowd movement and spots security threats in real-time frame images

## Network Type
CNN

## Architecture
- Input: 1 channel x 8x8 grayscale images (64 pixels)
- Convolution: 3 filters, 3x3 kernel, ReLU activation
- Pooling: MaxPool2D (2x2)
- Dense layers: 32 hidden units + 3 output units (softmax)

## Training
```bash
video_surveillance-train --model-dir ./artifacts/models --n-iterations 300 --n-samples 500
```

## Serving API
```bash
uvicorn video_surveillance.api:app --host 0.0.0.0 --port 8008
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
['normal', 'activity', 'threat']

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
