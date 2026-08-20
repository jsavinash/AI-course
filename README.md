# MLOps Monorepo: Supervised, Unsupervised, Self-Supervised, Semi-Supervised & Reinforcement Learning Examples

A **production-ready MLOps monorepo** containing supervised, unsupervised, reinforcement, self-supervised, and semi-supervised learning examples deployed on Kubernetes, built with the `uv` package manager.

## 📚 Examples by Learning Type

| Example | Learning Type | Task | Model | Metrics |
|---------|--------------|------|-------|---------|
| **Pizza Price Prediction** | Supervised Learning - Regression | Predict pizza price from diameter | Linear Regression + Gradient Descent | MSE, RMSE, MAE, R² |
| **Spam Email Classification** | Supervised Learning - Binary Classification | Classify emails as SPAM or NOT spam | Logistic Regression + Gradient Descent | Accuracy, Precision, Recall, F1, ROC-AUC |
| **Market Segmentation** | Unsupervised Learning - Clustering | Segment customers into behavioral groups | K-Means | Inertia, Silhouette |
| **Recommendation Engine** | Unsupervised Learning - Association Rules | Recommend products based on transaction patterns | Apriori | Confidence, Lift, Support |
| **Anomaly Detection** | Unsupervised Learning - Dimensionality Reduction | Detect anomalous server metrics | PCA + Reconstruction Error | Precision, Recall, F1, Accuracy, FPR |
| **Robot Maze Navigation** | Reinforcement Learning - Q-Learning | Navigate a robot through a maze to reach the goal | Tabular Q-Learning | Success Rate, Mean Steps, Mean Reward, TD Error |
| **Semi-Supervised Email Classification** | Semi-Supervised Learning - Self-Training | Classify emails as SPAM/HAM with limited labeled data | Self-Training + Logistic Regression | Accuracy, Precision, Recall, F1, Pseudo-Label Count |
| **Self-Supervised Monitoring** | Self-Supervised Learning - Denoising Autoencoder | Detect anomalous server metrics without labeled anomalies | Denoising Autoencoder + Reconstruction Error | Accuracy, Precision, Recall, F1, Reconstruction Error |

---

## 🏗️ Architecture

```
├── pyproject.toml                    # Root uv workspace config
├── uv.lock                           # Locked dependencies (reproducible)
├── shared/                           # Shared MLOps utilities
│   └── mlops_shared/
│       ├── config.py                 # 12-factor config (env/YAML/validation)
│       ├── logging.py                # Structured JSON logging + correlation IDs + redaction
│       ├── metrics.py                # Production Prometheus metrics
│       ├── model_registry.py         # Model versioning + lifecycle + MLflow integration
│       ├── validation.py             # Data schema validation
│       ├── drift.py                  # Data drift detection (PSI, KS test, chi-squared)
│       └── fastapi_middleware.py     # Request logging, security headers, observability
│
├── machine-learning/
│   ├── supervised/
│   │   ├── pizza-price/              # Supervised Learning - Regression
│   │   │   └── src/pizza_price/
│   │   │       ├── model.py          # Linear Regression (MSE, RMSE, R², MAE)
│   │   │       ├── data.py           # Data loading + train/test split
│   │   │       ├── train.py          # Production training pipeline
│   │   │       └── api.py            # FastAPI serving with observability
│   │   │
│   │   └── spam-classification/      # Supervised Learning - Binary Classification
│   │       └── src/spam_classification/
│   │           ├── model.py          # Logistic Regression (Acc, Prec, Rec, F1, AUC)
│   │           ├── data.py           # Data loading + feature extraction
│   │           ├── train.py          # Production training pipeline
│   │           └── api.py            # FastAPI serving with observability
│   │
│   ├── unsupervised/
│   │   ├── market-segmentation/      # Unsupervised Learning - Clustering
│   │   │   └── src/market_segmentation/
│   │   │       ├── model.py          # K-Means clustering (Inertia, Silhouette)
│   │   │       ├── data.py           # Synthetic customer data generator
│   │   │       ├── train.py          # Production training pipeline
│   │   │       └── api.py            # FastAPI serving with observability
│   │   │
│   │   ├── recommendation-engine/    # Unsupervised Learning - Association Rules
│   │   │   └── src/recommendation_engine/
│   │   │       ├── model.py          # Apriori (Confidence, Lift, Support)
│   │   │       ├── data.py           # Synthetic transaction generator
│   │   │       ├── train.py          # Production training pipeline
│   │   │       └── api.py            # FastAPI serving with observability
│   │   │
│   │   └── anomaly-detection/        # Unsupervised Learning - Dimensionality Reduction
│   │       └── src/anomaly_detection/
│   │           ├── model.py          # PCA + reconstruction error (Prec, Rec, F1)
│   │           ├── data.py           # Synthetic server monitoring metrics
│   │           ├── train.py          # Production training pipeline
│   │           └── api.py            # FastAPI serving with observability
│   │
│   ├── reinforcement/
│   │   └── robot-maze-navigation/    # Reinforcement Learning - Q-Learning
│   │       └── src/robot_maze/
│   │           ├── model.py          # Q-Learning agent (Success Rate, Mean Steps, Reward)
│   │           ├── data.py           # Maze generation + reward shaping
│   │           ├── train.py          # Production training pipeline
│   │           └── api.py            # FastAPI serving with observability
│   │
│   ├── semi-supervised/
│   │   └── semi-supervised-email/    # Semi-Supervised Learning - Self-Training
│   │       └── src/semi_supervised_email/
│   │           ├── model.py          # Self-Training + Logistic Regression
│   │           ├── data.py           # Email data with labeled/unlabeled split
│   │           ├── train.py          # Production training pipeline
│   │           └── api.py            # FastAPI serving with observability
│   │
│   └── self-supervised/
│       └── self-supervised-monitoring/  # Self-Supervised Learning - Autoencoder
│           └── src/self_supervised_monitoring/
│               ├── model.py          # Denoising Autoencoder (Reconstruction Error)
│               ├── data.py           # Synthetic server monitoring metrics
│               ├── train.py          # Production training pipeline
│               └── api.py            # FastAPI serving with observability
│
├── docker/
│   ├── train.Dockerfile              # Multi-stage, non-root, optimized training image
│   └── serve.Dockerfile              # Multi-stage, non-root, optimized serving image
│
├── k8s/
│   ├── namespace.yaml                # mlops namespace
│   ├── storage.yaml                  # PVCs for models & MLflow
│   ├── secrets.yaml                  # Grafana secrets
│   ├── mlflow.yaml                   # MLflow tracking server
│   ├── training-jobs.yaml            # K8s Jobs for all models
│   ├── serving.yaml                  # Deployments + Services for all APIs
│   ├── autoscaling.yaml              # HPA (CPU/Memory + scale behavior)
│   └── monitoring.yaml               # Prometheus + Grafana
│
├── .github/workflows/ci-cd.yml       # CI/CD: lint → test → build → deploy
├── tests/                            # Unit & integration tests
├── Makefile                          # Automation commands
└── README.md
```

---

## 🔬 Production-Grade MLOps Pipeline

### 1. Experiment Tracking (MLflow)
- **MLflow Tracking Server** deployed in-cluster (`k8s/mlflow.yaml`)
- Tracks parameters, metrics, and artifacts for every training run
- Model Registry integration for versioned model storage
- Persistent storage for experiment history

### 2. Training Pipeline (K8s Jobs)
- All models trained as **Kubernetes Jobs** (`k8s/training-jobs.yaml`)
- **Train/Test split** with configurable `test_size` and `random_seed` (supervised)
- **Comprehensive metrics**: MSE/RMSE/MAE/R² (regression), Acc/Precision/Recall/F1/AUC (classification), Inertia/Silhouette (clustering), Success Rate/Mean Steps (RL)
- **Data validation** before training (schema + range checks)
- **Reproducible**: versions pinned in `uv.lock`, random seeds configurable
- **MLflow registration** with lifecycle management (staging → production → archived)
- **TTL cleanup** (`ttlSecondsAfterFinished: 3600`) for completed job pods
- **Security**: runs as non-root, proper fsGroup permissions

### 3. Serving Pipeline (K8s Deployments)
- **FastAPI** production servers with Pydantic validation
- **2 replicas** by default with **Horizontal Pod Autoscaling** (CPU 70%, Memory 80%)
- **Health probes**: startup, liveness, readiness
- **Observability middleware**:
  - Request correlation IDs (`X-Request-ID` headers)
  - Structured JSON request logging
  - Security headers (X-Frame-Options, X-Content-Type-Options, HSTS)
  - Active request tracking
- **Model versioning**: loads `latest` from registry, falls back to legacy paths

### 4. Monitoring & Observability
- **Prometheus** scrapes prediction metrics:
  - `{service}_predictions_total{model_version}`
  - `{service}_prediction_duration_seconds`
  - `{service}_prediction_errors_total`
  - `{service}_requests_total` (HTTP metrics)
  - `{service}_feature_drift_ratio`
- **Grafana** dashboards for visualization
- **Data drift detection** via `/drift` endpoint (PSI, KS test, chi-squared)

### 5. CI/CD Pipeline
GitHub Actions workflow (`.github/workflows/ci-cd.yml`):
1. **Lint & Test** — ruff, mypy, pytest (with coverage)
2. **Build & Push** — Docker images to GHCR
3. **Deploy** — Apply K8s manifests, train models, deploy serving APIs, verify

### 6. Security
- **Non-root containers** (`runAsNonRoot: true`, `runAsUser: 1000`)
- **Security headers** on all API responses
- **Secret redaction** in structured logs
- **fsGroup** for proper volume permissions
- **Minimal base images**: `python:3.11-slim`

---

## 🚀 Quick Start

### Prerequisites
- [uv](https://github.com/astral-sh/uv) ≥ 0.5
- Python ≥ 3.11
- Kubernetes cluster (kind, minikube, EKS, GKE, etc.)
- kubectl configured
- Docker (for building images)

### 1. Install Dependencies
```bash
make install
```

### 2. Train Models Locally
```bash
make train-pizza                  # Supervised - Regression
make train-spam                   # Supervised - Classification
make train-market-segmentation    # Unsupervised - Clustering
make train-recommendation-engine  # Unsupervised - Association Rules
make train-anomaly-detection      # Unsupervised - Dimensionality Reduction
make train-robot-maze             # Reinforcement Learning - Q-Learning
make train-all                    # Train all models
```

### 3. Run API Locally
```bash
make serve-pizza                  # http://localhost:8000
make serve-spam                   # http://localhost:8001
make serve-market-segmentation    # http://localhost:8002
make serve-recommendation-engine  # http://localhost:8003
make serve-anomaly-detection      # http://localhost:8004
make serve-robot-maze             # http://localhost:8005
make serve-all                    # Run all APIs locally
```

### 4. Test Everything
```bash
make test          # Run pytest suite
make lint          # Ruff linting
make typecheck     # Mypy type checking
```

### 5. Build Docker Images
```bash
make docker-build
```

### 6. Deploy to Kubernetes
```bash
make k8s-apply
```

### 7. Verify Deployment
```bash
make k8s-status
```

### 8. Access Services
```bash
make k8s-port-forward-pizza       # Pizza API  → localhost:8080
make k8s-port-forward-spam        # Spam API   → localhost:8081
make k8s-port-forward-mlflow      # MLflow     → localhost:5000
make k8s-port-forward-grafana     # Grafana    → localhost:3000 (admin/mlops-admin)
make k8s-port-forward-prometheus  # Prometheus → localhost:9090
```

---

## 📡 API Endpoints

### Pizza Price Prediction API (Supervised - Regression)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness/readiness probe |
| `/predict` | POST | Predict price from diameter |
| `/predict/bulk` | POST | Predict prices for multiple diameters |
| `/drift` | GET | Check data drift |
| `/metrics` | GET | Prometheus metrics |

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"diameter": 12}'
```

### Spam Classification API (Supervised - Binary Classification)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness/readiness probe |
| `/predict` | POST | Classify with explicit features |
| `/predict/email` | POST | Classify raw email text |
| `/drift` | GET | Check data drift |
| `/metrics` | GET | Prometheus metrics |

```bash
curl -X POST http://localhost:8081/predict/email \
  -H "Content-Type: application/json" \
  -d '{"text": "WIN a FREE iPhone!!! Click http://bit.ly now!!"}'
```

### Anomaly Detection API (Unsupervised - Dimensionality Reduction)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness/readiness probe |
| `/predict` | POST | Score a single server metrics sample |
| `/predict/bulk` | POST | Score multiple server metrics samples |
| `/model/info` | GET | PCA model details (components, variance, threshold) |
| `/drift` | GET | Check data drift |
| `/metrics` | GET | Prometheus metrics |

```bash
curl -X POST http://localhost:8004/predict \
  -H "Content-Type: application/json" \
  -d '{
    "request_count": 480,
    "bytes_per_request": 1500,
    "cpu_usage": 68,
    "memory_usage": 75,
    "disk_io": 1800,
    "network_in": 1500,
    "network_out": 900,
    "error_rate": 12.0,
    "connection_count": 2400,
    "response_time": 250
  }'
```

### Robot Maze Navigation API (Reinforcement Learning - Q-Learning)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Liveness/readiness probe |
| `/solve` | POST | Solve maze and return path |
| `/step` | POST | Get next action for a given state |
| `/train` | POST | Trigger online/offline training |
| `/stats` | GET | Q-table statistics and training metrics |
| `/drift` | GET | Check data drift |
| `/metrics` | GET | Prometheus metrics |

```bash
# Solve a 6x6 maze
curl -X POST http://localhost:8005/solve \
  -H "Content-Type: application/json" \
  -d '{"maze_size": 6, "max_steps": 200}'

# Get next action for state (row=1, col=1)
curl -X POST http://localhost:8005/step \
  -H "Content-Type: application/json" \
  -d '{"row": 1, "col": 1, "maze_size": 8}'
```

---

## 🧪 Postman Collection

A ready-to-use **Postman collection** is included at [`postman/mlops.postman_collection.json`](postman/mlops.postman_collection.json). It covers every endpoint for the serving APIs:

- **Pizza Price Prediction API** — root, health, predict, predict/bulk, drift, metrics, reload
- **Spam Classification API** — root, health, predict (features), predict/email (raw text), drift, metrics, reload
- **Anomaly Detection API** — root, health, predict, predict/bulk, model/info, drift, metrics, reload
- **Robot Maze Navigation API** — root, health, solve, step, train, stats, drift, metrics, reload

### Import & Use

1. Open Postman → **Import** → select `postman/mlops.postman_collection.json`.
2. The collection uses variables for the base URLs (e.g., `pizza_base_url`, `spam_base_url`, `anomaly_base_url`, `robot_maze_base_url`).
3. Start the APIs (`make serve-all`) or port-forward from Kubernetes, then run any request.

> Tip: When using Kubernetes port-forwards, set the base URLs to the corresponding localhost ports.

---

## 🧠 Model Details

### Pizza Price Prediction (Supervised - Regression)
- **Features**: diameter (inches)
- **Architecture**: `price = weight * diameter + bias`
- **Loss**: Mean Squared Error
- **Optimizer**: Gradient Descent
- **Metrics**: MSE, RMSE, MAE, R²

### Spam Email Classification (Supervised - Binary Classification)
- **Features**: 5 binary indicators (free, win, link, !!!, meeting)
- **Architecture**: `p = sigmoid(X·w + b)`
- **Loss**: Binary Cross-Entropy
- **Optimizer**: Gradient Descent
- **Metrics**: Accuracy, Precision, Recall, F1, ROC-AUC

### Market Segmentation (Unsupervised - Clustering)
- **Features**: annual_income (k$), spending_score (0-100)
- **Architecture**: K-Means clustering with k-means++ initialization
- **Metrics**: Inertia, Silhouette Score

### Recommendation Engine (Unsupervised - Association Rules)
- **Features**: transaction items
- **Architecture**: Apriori association-rule mining
- **Metrics**: Support, Confidence, Lift

### Anomaly Detection (Unsupervised - Dimensionality Reduction)
- **Features**: 10 server monitoring metrics (request_count, cpu_usage, memory_usage, network_in/out, error_rate, etc.)
- **Architecture**: Principal Component Analysis (dimensionality reduction) + reconstruction error threshold
- **Unsupervised**: labels are never used during training — anomalies are identified by
  high reconstruction error when projecting data onto a reduced PCA subspace
- **Metrics**: Precision, Recall, F1, Accuracy, False Positive Rate, Cumulative Variance Ratio

### Robot Maze Navigation (Reinforcement Learning - Q-Learning)
- **State Space**: Grid coordinates (row, col) in an N×N maze
- **Action Space**: 4 discrete actions (up, down, left, right)
- **Architecture**: Tabular Q-Learning with epsilon-greedy exploration
- **Reward Shaping**:
  - **Positive Reinforcement**: +10.0 reward for reaching the goal
  - **Negative Reinforcement**: -5.0 penalty for hitting walls, -0.1 per-step penalty to encourage shorter paths
- **Learning Modes**:
  - **Online RL**: Agent learns by interacting with the maze environment in real-time
  - **Offline RL**: Agent learns from a fixed dataset of transitions without environment interaction
- **Metrics**: Success Rate, Mean Steps, Mean Reward, TD Error, Epsilon Decay

### Semi-Supervised Email Classification (Semi-Supervised - Self-Training)
- **Features**: 7 email features (has_free, has_win, has_link, has_exclamation, has_meeting, length_score, has_caps)
- **Architecture**: Self-training with Logistic Regression base model
- **Learning Paradigm**: Starts with small labeled dataset, iteratively labels high-confidence unlabeled samples
- **Key Concepts**:
  - **Pseudo-labeling**: High-confidence predictions on unlabeled data are added to training set
  - **Confidence Threshold**: Only predictions above threshold (default 0.95) are trusted
  - **Iterative Refinement**: Model retrains on expanded labeled set until convergence
- **Metrics**: Accuracy, Precision, Recall, F1, Pseudo-Label Count, Training Mode

### Robot Maze Navigation (Reinforcement Learning - Q-Learning)
- **State Space**: Grid coordinates (row, col) in an N×N maze
- **Action Space**: 4 discrete actions (up, down, left, right)
- **Architecture**: Tabular Q-Learning with epsilon-greedy exploration
- **Reward Shaping**:
  - **Positive Reinforcement**: +10.0 reward for reaching the goal
  - **Negative Reinforcement**: -5.0 penalty for hitting walls, -0.1 per-step penalty to encourage shorter paths
- **Learning Modes**:
  - **Online RL**: Agent learns by interacting with the maze environment in real-time
  - **Offline RL**: Agent learns from a fixed dataset of transitions without environment interaction
- **Metrics**: Success Rate, Mean Steps, Mean Reward, TD Error, Epsilon Decay

---

## 📁 Artifact Structure

Models are stored in a versioned directory structure:

```
/models/
├── pizza-price/
│   └── 1.0.0/
│       ├── pizza_model_v1.0.0.npz
│       ├── training_data.csv
│       ├── pizza_regression_v1.0.0.png
│       └── model_info.json          # Registry metadata
├── spam-classification/
│   └── 1.0.0/
│       ├── spam_model_v1.0.0.npz
│       ├── training_data.csv
│       ├── spam_classification_v1.0.0.png
│       └── model_info.json
├── anomaly-detection/
│   └── 1.0.0/
│       ├── anomaly_detection_model_v1.0.0.npz
│       ├── training_data.csv
│       ├── anomaly_detection_v1.0.0.png
│       └── model_info.json
└── robot-maze-navigation/
    └── 1.0.0/
        ├── robot_maze_model_v1.0.0.npz
        ├── training_data.npz
        ├── robot_maze_v1.0.0.png
        └── model_info.json
└── semi-supervised-email/
    └── 1.0.0/
        ├── semi_supervised_email_model_v1.0.0.npz
        ├── training_data.csv
        ├── semi_supervised_email_v1.0.0.png
        └── model_info.json
```

---

## 🔑 Kubernetes Secrets Required for CI/CD

- `KUBE_CONFIG` — base64-encoded kubeconfig

---

## 📊 Prometheus Metrics

| Metric | Type | Labels | Description |
|--------|------|--------|-------------|
| `{svc}_predictions_total` | Counter | model_version | Total predictions |
| `{svc}_prediction_duration_seconds` | Histogram | model_version | Prediction latency |
| `{svc}_prediction_errors_total` | Counter | model_version, error_type | Prediction errors |
| `{svc}_requests_total` | Counter | method, endpoint, status | HTTP requests |
| `{svc}_request_duration_seconds` | Histogram | method, endpoint | HTTP latency |
| `{svc}_feature_drift_ratio` | Gauge | - | Data drift |
| `{svc}_model_version` | Gauge | - | Current model version |

---

## 🛠️ Development Workflow

1. **Develop** — modify model, data, or API code
2. **Test** — `make test` (unit + integration)
3. **Lint** — `make lint && make typecheck`
4. **Train** — `make train-all` (locally or in K8s)
5. **Build** — `make docker-build`
6. **Deploy** — `make k8s-apply`
7. **Monitor** — Grafana + Prometheus + MLflow

---

## 📝 Makefile Commands

```bash
make install                  # Install all dependencies
make lint                     # Ruff linting
make format                   # Ruff formatting
make typecheck                # Mypy type checking
make test                     # Run pytest suite

# Training
make train-pizza              # Train pizza model locally
make train-spam               # Train spam model locally
make train-market-segmentation # Train market segmentation model locally
make train-recommendation-engine # Train recommendation engine model locally
make train-anomaly-detection  # Train anomaly detection model locally
make train-robot-maze         # Train robot maze navigation model locally
make train-semi-supervised-email # Train semi-supervised email model locally
make train-all                # Train all models

# Serving
make serve-pizza              # Run pizza API locally (port 8000)
make serve-spam               # Run spam API locally (port 8001)
make serve-market-segmentation # Run market segmentation API locally (port 8002)
make serve-recommendation-engine # Run recommendation engine API locally (port 8003)
make serve-anomaly-detection  # Run anomaly detection API locally (port 8004)
make serve-robot-maze         # Run robot maze navigation API locally (port 8005)
make serve-semi-supervised-email # Run semi-supervised email API locally (port 8006)
make serve-all                # Run all APIs locally

# Docker
make docker-build             # Build Docker images
make docker-clean             # Remove Docker artifacts

# Kubernetes
make k8s-apply                # Deploy everything to Kubernetes
make k8s-delete               # Delete Kubernetes namespace
make k8s-status               # Show Kubernetes status
make k8s-port-forward-pizza       # → localhost:8080
make k8s-port-forward-spam        # → localhost:8081
make k8s-port-forward-mlflow      # → localhost:5000
make k8s-port-forward-grafana     # → localhost:3000
make k8s-port-forward-prometheus  # → localhost:9090
make k8s-describe             # Describe pods
make k8s-events               # Show cluster events
```

---

## 📄 License
Educational purposes - demonstrates production-ready MLOps patterns with real-world scenarios.
# AI-course
