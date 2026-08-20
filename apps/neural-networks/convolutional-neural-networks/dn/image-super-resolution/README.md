# Image Super-Resolution

Upscales low-resolution photos and blurry security footage into sharp images

## Network Type
DN

## Architecture
- Input: 1 channel x 8x8 grayscale images (64 pixels)
- Convolution: 0 filters, 3x3 kernel, ReLU activation
- Pooling: MaxPool2D (2x2)
- Dense layers: 32 hidden units + 1 output units (linear)

## Training
```bash
image_super_resolution-train --model-dir ./artifacts/models --n-iterations 300 --n-samples 500
```

## Serving API
```bash
uvicorn image_super_resolution.api:app --host 0.0.0.0 --port 8010
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
[]

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
