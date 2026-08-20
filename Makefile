# MLOps Monorepo Makefile
# =========================

PYTHON := uv run python
KUBECTL := kubectl
NAMESPACE := mlops
IMG_TRAIN := mlops/train
IMG_SERVE := mlops/serve

# --- Python / uv ---
.PHONY: install
install:
	uv sync --all-packages

.PHONY: lint
lint:
	uv run ruff check .

.PHONY: format
format:
	uv run ruff format .

.PHONY: typecheck
typecheck:
	uv run mypy shared machine-learning --ignore-missing-imports

.PHONY: test
test:
	uv run pytest

.PHONY: uv-sync
uv-sync:
	uv sync --all-packages --reinstall

.PHONY: train-pizza
train-pizza:
	uv run python -m pizza_price.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-spam
train-spam:
	uv run python -m spam_classification.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-market-segmentation
train-market-segmentation:
	uv run python -m market_segmentation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-recommendation-engine
train-recommendation-engine:
	uv run python -m recommendation_engine.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-anomaly-detection
train-anomaly-detection:
	uv run python -m anomaly_detection.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-robot-maze
train-robot-maze:
	uv run python -m robot_maze.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-semi-supervised-email
train-semi-supervised-email:
	uv run python -m semi_supervised_email.train --model-dir ./artifacts/models --model-version 1.0.0 --labeled-ratio 0.1

.PHONY: train-self-supervised-monitoring
train-self-supervised-monitoring:
	uv run python -m self_supervised_monitoring.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-all
train-all: train-pizza train-spam train-market-segmentation train-recommendation-engine train-anomaly-detection train-robot-maze train-semi-supervised-email train-self-supervised-monitoring

.PHONY: serve-pizza
serve-pizza:
	MODEL_DIR=./artifacts/models uv run uvicorn pizza_price.api:app --host 0.0.0.0 --port 8000

.PHONY: serve-spam
serve-spam:
	MODEL_DIR=./artifacts/models uv run uvicorn spam_classification.api:app --host 0.0.0.0 --port 8001

.PHONY: serve-market-segmentation
serve-market-segmentation:
	MODEL_DIR=./artifacts/models uv run uvicorn market_segmentation.api:app --host 0.0.0.0 --port 8002

.PHONY: serve-recommendation-engine
serve-recommendation-engine:
	MODEL_DIR=./artifacts/models uv run uvicorn recommendation_engine.api:app --host 0.0.0.0 --port 8003

.PHONY: serve-anomaly-detection
serve-anomaly-detection:
	MODEL_DIR=./artifacts/models uv run uvicorn anomaly_detection.api:app --host 0.0.0.0 --port 8004

.PHONY: serve-robot-maze
serve-robot-maze:
	MODEL_DIR=./artifacts/models uv run uvicorn robot_maze.api:app --host 0.0.0.0 --port 8005

.PHONY: serve-semi-supervised-email
serve-semi-supervised-email:
	MODEL_DIR=./artifacts/models uv run uvicorn semi_supervised_email.api:app --host 0.0.0.0 --port 8006

.PHONY: serve-self-supervised-monitoring
serve-self-supervised-monitoring:
	MODEL_DIR=./artifacts/models uv run uvicorn self_supervised_monitoring.api:app --host 0.0.0.0 --port 8007

.PHONY: serve-all
serve-all:
	uv run python scripts/serve_all.py

# --- Docker ---
.PHONY: docker-build
docker-build: docker-build-train docker-build-serve

.PHONY: docker-build-train
docker-build-train:
	docker build -t $(IMG_TRAIN):latest -f docker/train.Dockerfile .

.PHONY: docker-build-serve
docker-build-serve:
	docker build -t $(IMG_SERVE):latest -f docker/serve.Dockerfile .

.PHONY: docker-clean
docker-clean:
	docker system prune -af --volumes

# --- Kubernetes ---
.PHONY: k8s-apply
k8s-apply:
	$(KUBECTL) apply -f k8s/namespace.yaml
	$(KUBECTL) apply -f k8s/storage.yaml
	$(KUBECTL) apply -f k8s/secrets.yaml
	$(KUBECTL) apply -f k8s/mlflow.yaml
	$(KUBECTL) apply -f k8s/training-jobs.yaml
	$(KUBECTL) apply -f k8s/serving.yaml
	$(KUBECTL) apply -f k8s/autoscaling.yaml
	$(KUBECTL) apply -f k8s/monitoring.yaml

.PHONY: k8s-delete
k8s-delete:
	$(KUBECTL) delete namespace $(NAMESPACE) --ignore-not-found=true

.PHONY: k8s-clean
k8s-clean: k8s-delete

.PHONY: k8s-status
k8s-status:
	$(KUBECTL) get all -n $(NAMESPACE)
	@echo "--- HPA ---"
	$(KUBECTL) get hpa -n $(NAMESPACE)
	@echo "--- PVC ---"
	$(KUBECTL) get pvc -n $(NAMESPACE)

.PHONY: k8s-logs-pizza
k8s-logs-pizza:
	$(KUBECTL) logs -n $(NAMESPACE) deployment/pizza-price-api -f

.PHONY: k8s-logs-spam
k8s-logs-spam:
	$(KUBECTL) logs -n $(NAMESPACE) deployment/spam-classification-api -f

.PHONY: k8s-logs-train-pizza
k8s-logs-train-pizza:
	$(KUBECTL) logs -n $(NAMESPACE) job/train-pizza-price -f

.PHONY: k8s-logs-train-spam
k8s-logs-train-spam:
	$(KUBECTL) logs -n $(NAMESPACE) job/train-spam-classification -f

.PHONY: k8s-logs-train-market-segmentation
k8s-logs-train-market-segmentation:
	$(KUBECTL) logs -n $(NAMESPACE) job/train-market-segmentation -f

.PHONY: k8s-port-forward-pizza
k8s-port-forward-pizza:
	$(KUBECTL) port-forward -n $(NAMESPACE) service/pizza-price-api 8080:80

.PHONY: k8s-port-forward-spam
k8s-port-forward-spam:
	$(KUBECTL) port-forward -n $(NAMESPACE) service/spam-classification-api 8081:80

.PHONY: k8s-port-forward-mlflow
k8s-port-forward-mlflow:
	$(KUBECTL) port-forward -n $(NAMESPACE) service/mlflow 5000:5000

.PHONY: k8s-port-forward-grafana
k8s-port-forward-grafana:
	$(KUBECTL) port-forward -n $(NAMESPACE) service/grafana 3000:3000

.PHONY: k8s-port-forward-prometheus
k8s-port-forward-prometheus:
	$(KUBECTL) port-forward -n $(NAMESPACE) service/prometheus 9090:9090

.PHONY: k8s-describe
k8s-describe:
	$(KUBECTL) describe pods -n $(NAMESPACE)

.PHONY: k8s-events
k8s-events:
	$(KUBECTL) get events -n $(NAMESPACE) --sort-by=.lastTimestamp

.PHONY: help
help:
	@echo "MLOps Monorepo Commands:"
	@echo ""
	@echo "Python:"
	@echo "  make install              - Install dependencies with uv"
	@echo "  make lint                 - Run ruff linter"
	@echo "  make format               - Run ruff formatter"
	@echo "  make typecheck            - Run mypy type checker"
	@echo "  make test                 - Run pytest tests"
	@echo "  make train-pizza          - Train pizza-price model locally"
	@echo "  make train-spam           - Train spam-classification model locally"
	@echo "  make train-market-segmentation - Train market-segmentation model locally"
	@echo "  make train-recommendation-engine - Train recommendation-engine model locally"
	@echo "  make train-anomaly-detection - Train anomaly-detection model locally"
	@echo "  make train-robot-maze - Train robot-maze-navigation model locally"
	@echo "  make train-semi-supervised-email - Train semi-supervised-email model locally"
	@echo "  make train-self-supervised-monitoring - Train self-supervised-monitoring model locally"
	@echo "  make train-all            - Train all models"
	@echo "  make serve-pizza          - Run pizza-price API locally (port 8000)"
	@echo "  make serve-spam           - Run spam-classification API locally (port 8001)"
	@echo "  make serve-market-segmentation - Run market-segmentation API locally (port 8002)"
	@echo "  make serve-recommendation-engine - Run recommendation-engine API locally (port 8003)"
	@echo "  make serve-anomaly-detection - Run anomaly-detection API locally (port 8004)"
	@echo "  make serve-robot-maze - Run robot-maze-navigation API locally (port 8005)"
	@echo "  make serve-semi-supervised-email - Run semi-supervised-email API locally (port 8006)"
	@echo "  make serve-self-supervised-monitoring - Run self-supervised-monitoring API locally (port 8007)"
	@echo "  make serve-all            - Run all APIs locally"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build         - Build both Docker images"
	@echo "  make docker-clean         - Remove all Docker artifacts"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make k8s-apply            - Deploy everything to Kubernetes"
	@echo "  make k8s-delete           - Delete Kubernetes namespace"
	@echo "  make k8s-status           - Show Kubernetes status"
	@echo "  make k8s-port-forward-pizza   - Port-forward pizza API to localhost:8080"
	@echo "  make k8s-port-forward-spam    - Port-forward spam API to localhost:8081"
	@echo "  make k8s-port-forward-mlflow  - Port-forward MLflow to localhost:5000"
	@echo "  make k8s-port-forward-grafana - Port-forward Grafana to localhost:3000"
	@echo "  make k8s-port-forward-prometheus - Port-forward Prometheus to localhost:9090"