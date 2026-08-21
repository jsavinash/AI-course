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
	uv run mypy shared apps/machine-learning apps/neural-networks --ignore-missing-imports

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

.PHONY: train-email-spam-detection-nn
train-email-spam-detection-nn:
	uv run python -m email_spam_detection.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-house-price-prediction-nn
train-house-price-prediction-nn:
	uv run python -m house_price_prediction.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-credit-card-fraud-detection-nn
train-credit-card-fraud-detection-nn:
	uv run python -m credit_card_fraud_detection.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-handwritten-digit-recognition-nn
train-handwritten-digit-recognition-nn:
	uv run python -m handwritten_digit_recognition.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-language-translation-rnn
train-language-translation-rnn:
	uv run python -m language_translation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-sentiment-analysis-rnn
train-sentiment-analysis-rnn:
	uv run python -m sentiment_analysis.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-text-generation-rnn
train-text-generation-rnn:
	uv run python -m text_generation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-speech-recognition-rnn
train-speech-recognition-rnn:
	uv run python -m speech_recognition.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-music-generation-rnn
train-music-generation-rnn:
	uv run python -m music_generation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-stock-market-prediction-rnn
train-stock-market-prediction-rnn:
	uv run python -m stock_market_prediction.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-weather-forecasting-rnn
train-weather-forecasting-rnn:
	uv run python -m weather_forecasting.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-image-captioning-rnn
train-image-captioning-rnn:
	uv run python -m image_captioning.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-code-generation
train-code-generation:
	uv run python -m code_generation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-image-generation
train-image-generation:
	uv run python -m image_generation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-prompt-engineering
train-prompt-engineering:
	uv run python -m prompt_engineering.train --model-dir ./artifacts/models --model-version 1.0.0 --technique chain-of-thought

.PHONY: train-retrieval-augmented-generation
train-retrieval-augmented-generation:
	uv run python -m retrieval_augmented_generation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-text-gen
train-text-gen:
	uv run python -m text_gen.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-tool-use
train-tool-use:
	uv run python -m tool_use_and_functional_calling.train --model-dir ./artifacts/models --model-version 1.0.0 --n-tools 5

.PHONY: train-video-generation
train-video-generation:
	uv run python -m video_generation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-all
train-all: train-pizza train-spam train-market-segmentation train-recommendation-engine train-anomaly-detection train-robot-maze train-semi-supervised-email train-self-supervised-monitoring train-email-spam-detection-nn train-house-price-prediction-nn train-credit-card-fraud-detection-nn train-handwritten-digit-recognition-nn train-language-translation-rnn train-sentiment-analysis-rnn train-text-generation-rnn train-speech-recognition-rnn train-music-generation-rnn train-stock-market-prediction-rnn train-weather-forecasting-rnn train-image-captioning-rnn train-code-generation train-image-generation train-prompt-engineering train-retrieval-augmented-generation train-text-gen train-tool-use train-video-generation

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

.PHONY: serve-email-spam-detection-nn
serve-email-spam-detection-nn:
	MODEL_DIR=./artifacts/models uv run uvicorn email_spam_detection.api:app --host 0.0.0.0 --port 8008

.PHONY: serve-house-price-prediction-nn
serve-house-price-prediction-nn:
	MODEL_DIR=./artifacts/models uv run uvicorn house_price_prediction.api:app --host 0.0.0.0 --port 8009

.PHONY: serve-credit-card-fraud-detection-nn
serve-credit-card-fraud-detection-nn:
	MODEL_DIR=./artifacts/models uv run uvicorn credit_card_fraud_detection.api:app --host 0.0.0.0 --port 8010

.PHONY: serve-handwritten-digit-recognition-nn
serve-handwritten-digit-recognition-nn:
	MODEL_DIR=./artifacts/models uv run uvicorn handwritten_digit_recognition.api:app --host 0.0.0.0 --port 8011

.PHONY: serve-language-translation-rnn
serve-language-translation-rnn:
	MODEL_DIR=./artifacts/models uv run uvicorn language_translation.api:app --host 0.0.0.0 --port 8012

.PHONY: serve-sentiment-analysis-rnn
serve-sentiment-analysis-rnn:
	MODEL_DIR=./artifacts/models uv run uvicorn sentiment_analysis.api:app --host 0.0.0.0 --port 8013

.PHONY: serve-text-generation-rnn
serve-text-generation-rnn:
	MODEL_DIR=./artifacts/models uv run uvicorn text_generation.api:app --host 0.0.0.0 --port 8014

.PHONY: serve-speech-recognition-rnn
serve-speech-recognition-rnn:
	MODEL_DIR=./artifacts/models uv run uvicorn speech_recognition.api:app --host 0.0.0.0 --port 8015

.PHONY: serve-music-generation-rnn
serve-music-generation-rnn:
	MODEL_DIR=./artifacts/models uv run uvicorn music_generation.api:app --host 0.0.0.0 --port 8016

.PHONY: serve-stock-market-prediction-rnn
serve-stock-market-prediction-rnn:
	MODEL_DIR=./artifacts/models uv run uvicorn stock_market_prediction.api:app --host 0.0.0.0 --port 8017

.PHONY: serve-weather-forecasting-rnn
serve-weather-forecasting-rnn:
	MODEL_DIR=./artifacts/models uv run uvicorn weather_forecasting.api:app --host 0.0.0.0 --port 8018

.PHONY: serve-image-captioning-rnn
serve-image-captioning-rnn:
	MODEL_DIR=./artifacts/models uv run uvicorn image_captioning.api:app --host 0.0.0.0 --port 8019

.PHONY: serve-code-generation
serve-code-generation:
	MODEL_DIR=./artifacts/models uv run uvicorn code_generation.api:app --host 0.0.0.0 --port 8020

.PHONY: serve-image-generation
serve-image-generation:
	MODEL_DIR=./artifacts/models uv run uvicorn image_generation.api:app --host 0.0.0.0 --port 8021

.PHONY: serve-prompt-engineering
serve-prompt-engineering:
	MODEL_DIR=./artifacts/models uv run uvicorn prompt_engineering.api:app --host 0.0.0.0 --port 8022

.PHONY: serve-retrieval-augmented-generation
serve-retrieval-augmented-generation:
	MODEL_DIR=./artifacts/models uv run uvicorn retrieval_augmented_generation.api:app --host 0.0.0.0 --port 8023

.PHONY: serve-text-gen
serve-text-gen:
	MODEL_DIR=./artifacts/models uv run uvicorn text_gen.api:app --host 0.0.0.0 --port 8024

.PHONY: serve-tool-use
serve-tool-use:
	MODEL_DIR=./artifacts/models uv run uvicorn tool_use_and_functional_calling.api:app --host 0.0.0.0 --port 8025

.PHONY: serve-video-generation
serve-video-generation:
	MODEL_DIR=./artifacts/models uv run uvicorn video_generation.api:app --host 0.0.0.0 --port 8026

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
	@echo "  make train-email-spam-detection-nn - Train email-spam-detection NN model locally"
	@echo "  make train-house-price-prediction-nn - Train house-price-prediction NN model locally"
	@echo "  make train-credit-card-fraud-detection-nn - Train fraud-detection NN model locally"
	@echo "  make train-handwritten-digit-recognition-nn - Train digit-recognition NN model locally"
	@echo "  make train-language-translation-rnn - Train language-translation RNN model locally"
	@echo "  make train-sentiment-analysis-rnn  - Train sentiment-analysis RNN model locally"
	@echo "  make train-text-generation-rnn     - Train text-generation RNN model locally"
	@echo "  make train-speech-recognition-rnn  - Train speech-recognition RNN model locally"
	@echo "  make train-music-generation-rnn    - Train music-generation RNN model locally"
	@echo "  make train-stock-market-prediction-rnn - Train stock-market-prediction RNN model locally"
	@echo "  make train-weather-forecasting-rnn      - Train weather-forecasting RNN model locally"
	@echo "  make train-image-captioning-rnn          - Train image-captioning RNN model locally"
	@echo "  make train-code-generation               - Train code-generation model locally"
	@echo "  make train-image-generation              - Train image-generation model locally"
	@echo "  make train-prompt-engineering            - Train prompt-engineering model locally"
	@echo "  make train-retrieval-augmented-generation - Train RAG model locally"
	@echo "  make train-text-gen                      - Train text-gen (generative) model locally"
	@echo "  make train-tool-use                      - Train tool-use model locally"
	@echo "  make train-video-generation              - Train video-generation model locally"
	@echo "  make train-all            - Train all models"
	@echo "  make serve-pizza          - Run pizza-price API locally (port 8000)"
	@echo "  make serve-spam           - Run spam-classification API locally (port 8001)"
	@echo "  make serve-market-segmentation - Run market-segmentation API locally (port 8002)"
	@echo "  make serve-recommendation-engine - Run recommendation-engine API locally (port 8003)"
	@echo "  make serve-anomaly-detection - Run anomaly-detection API locally (port 8004)"
	@echo "  make serve-robot-maze - Run robot-maze-navigation API locally (port 8005)"
	@echo "  make serve-semi-supervised-email - Run semi-supervised-email API locally (port 8006)"
	@echo "  make serve-self-supervised-monitoring - Run self-supervised-monitoring API locally (port 8007)"
	@echo "  make serve-email-spam-detection-nn    - Run email-spam-detection NN API locally (port 8008)"
	@echo "  make serve-house-price-prediction-nn  - Run house-price-prediction NN API locally (port 8009)"
	@echo "  make serve-credit-card-fraud-detection-nn - Run fraud-detection NN API locally (port 8010)"
	@echo "  make serve-handwritten-digit-recognition-nn - Run digit-recognition NN API locally (port 8011)"
	@echo "  make serve-language-translation-rnn   - Run language-translation RNN API locally (port 8012)"
	@echo "  make serve-sentiment-analysis-rnn     - Run sentiment-analysis RNN API locally (port 8013)"
	@echo "  make serve-text-generation-rnn        - Run text-generation RNN API locally (port 8014)"
	@echo "  make serve-speech-recognition-rnn     - Run speech-recognition RNN API locally (port 8015)"
	@echo "  make serve-music-generation-rnn       - Run music-generation RNN API locally (port 8016)"
	@echo "  make serve-stock-market-prediction-rnn - Run stock-market-prediction RNN API locally (port 8017)"
	@echo "  make serve-weather-forecasting-rnn    - Run weather-forecasting RNN API locally (port 8018)"
	@echo "  make serve-image-captioning-rnn       - Run image-captioning RNN API locally (port 8019)"
	@echo "  make serve-code-generation            - Run code-generation API locally (port 8020)"
	@echo "  make serve-image-generation           - Run image-generation API locally (port 8021)"
	@echo "  make serve-prompt-engineering         - Run prompt-engineering API locally (port 8022)"
	@echo "  make serve-retrieval-augmented-generation - Run RAG API locally (port 8023)"
	@echo "  make serve-text-gen                   - Run text-gen (generative) API locally (port 8024)"
	@echo "  make serve-tool-use                   - Run tool-use API locally (port 8025)"
	@echo "  make serve-video-generation           - Run video-generation API locally (port 8026)"
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