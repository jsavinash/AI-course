# Text and Character Recognition

Reads distorted captchas and complex handwritten text keeping part-to-whole relationships

## Network Type
CapsNet

## Architecture
- Input: 1 channel x 8x8 grayscale images (64 pixels)
- Convolution: 36 filters, 3x3 kernel, ReLU activation
- Pooling: MaxPool2D (2x2)
- Dense layers: 32 hidden units + 36 output units (softmax)

## Training
```bash
text_char_recognition-train --model-dir ./artifacts/models --n-iterations 300 --n-samples 500
```

## Serving API
```bash
uvicorn text_char_recognition.api:app --host 0.0.0.0 --port 8014
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
[str(i) for i in range(10)] + [chr(c) for c in range(ord('A'), ord('Z')+1)]

## Dependencies
- Python >= 3.11
- NumPy, Pydantic, FastAPI
- mlops-shared
