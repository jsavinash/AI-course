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
	uv run mypy packages/ai-core apps/machine-learning apps/neural-networks apps/deep-learning apps/generative-ai --ignore-missing-imports

.PHONY: test
test:
	uv run pytest

.PHONY: uv-sync
uv-sync:
	uv sync --all-packages --reinstall

.PHONY: train-pizza-price
train-pizza-price:
	uv run python -m pizza_price.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-spam-classification
train-spam-classification:
	uv run python -m spam_classification.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-market-segmentation
train-market-segmentation:
	uv run python -m market_segmentation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-recommendation-engine
train-recommendation-engine:
	uv run python -m recommendation_engine.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-anomaly-detection-pca
train-anomaly-detection-pca:
	uv run python -m anomaly_detection.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-robot-maze-navigation
train-robot-maze-navigation:
	uv run python -m robot_maze_navigation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-semi-supervised-email
train-semi-supervised-email:
	uv run python -m semi_supervised_email.train --model-dir ./artifacts/models --model-version 1.0.0 --labeled-ratio 0.1

.PHONY: train-self-supervised-monitoring
train-self-supervised-monitoring:
	uv run python -m self_supervised_monitoring.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-classification-email-spam
train-classification-email-spam:
	uv run python -m classification_email_spam.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-regression-house-price
train-regression-house-price:
	uv run python -m regression_house_price.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-anomaly-detection-fraud
train-anomaly-detection-fraud:
	uv run python -m anomaly_detection_fraud.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-pattern-recognition-digits
train-pattern-recognition-digits:
	uv run python -m pattern_recognition_digits.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-nlp-language-translation
train-nlp-language-translation:
	uv run python -m nlp_language_translation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-nlp-sentiment-analysis
train-nlp-sentiment-analysis:
	uv run python -m nlp_sentiment_analysis.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-nlp-text-generation
train-nlp-text-generation:
	uv run python -m nlp_text_generation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-speech-audio-recognition
train-speech-audio-recognition:
	uv run python -m speech_audio_recognition.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-speech-audio-music
train-speech-audio-music:
	uv run python -m speech_audio_music.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-time-series-stock
train-time-series-stock:
	uv run python -m time_series_stock.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-time-series-weather
train-time-series-weather:
	uv run python -m time_series_weather.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-vision-image-captioning
train-vision-image-captioning:
	uv run python -m vision_image_captioning.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-cnn-medical-imaging
train-cnn-medical-imaging:
	uv run python -m cnn_medical_imaging.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-cnn-facial-recognition
train-cnn-facial-recognition:
	uv run python -m cnn_facial_recognition.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-cnn-video-surveillance
train-cnn-video-surveillance:
	uv run python -m cnn_video_surveillance.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-advanced-super-resolution
train-advanced-super-resolution:
	uv run python -m advanced_super_resolution.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-advanced-semantic-segmentation
train-advanced-semantic-segmentation:
	uv run python -m advanced_semantic_segmentation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-advanced-generative-art
train-advanced-generative-art:
	uv run python -m advanced_generative_art.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-capsnet-autonomous-driving
train-capsnet-autonomous-driving:
	uv run python -m capsnet_autonomous_driving.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-capsnet-medical-scan
train-capsnet-medical-scan:
	uv run python -m capsnet_medical_scan.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-capsnet-text-recognition
train-capsnet-text-recognition:
	uv run python -m capsnet_text_recognition.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-gnn-social-networks
train-gnn-social-networks:
	uv run python -m gnn_social_networks.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-pinn-heat-equation
train-pinn-heat-equation:
	uv run python -m pinn_heat_equation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-snn-image-classification
train-snn-image-classification:
	uv run python -m snn_image_classification.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-diffusion
train-diffusion:
	uv run python -m diffusion_image_generation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-gan
train-gan:
	uv run python -m gan_image_generation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-transformers
train-transformers:
	uv run python -m transformer_language_modeling.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-vae
train-vae:
	uv run python -m vae_data_generation.train --model-dir ./artifacts/models --model-version 1.0.0

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

.PHONY: train-text-generation
train-text-generation:
	uv run python -m text_generation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-tool-use-functional-calling
train-tool-use-functional-calling:
	uv run python -m tool_use_and_functional_calling.train --model-dir ./artifacts/models --model-version 1.0.0 --n-tools 5

.PHONY: train-video-generation
train-video-generation:
	uv run python -m video_generation.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-autoencoders-dimensionality-reduction
train-autoencoders-dimensionality-reduction:
	uv run python -m autoencoders_dimensionality_reduction.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-deep-belief-networks
train-deep-belief-networks:
	uv run python -m deep_belief_networks.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-restricted-boltzmann-machines
train-restricted-boltzmann-machines:
	uv run python -m rbm_feature_learning.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-self-organizing-maps
train-self-organizing-maps:
	uv run python -m self_organizing_maps.train --model-dir ./artifacts/models --model-version 1.0.0

.PHONY: train-all
train-all: train-pizza-price train-spam-classification train-market-segmentation train-recommendation-engine train-anomaly-detection-pca train-robot-maze-navigation train-semi-supervised-email train-self-supervised-monitoring train-classification-email-spam train-regression-house-price train-anomaly-detection-fraud train-pattern-recognition-digits train-nlp-language-translation train-nlp-sentiment-analysis train-nlp-text-generation train-speech-audio-recognition train-speech-audio-music train-time-series-stock train-time-series-weather train-vision-image-captioning train-cnn-medical-imaging train-cnn-facial-recognition train-cnn-video-surveillance train-advanced-super-resolution train-advanced-semantic-segmentation train-advanced-generative-art train-capsnet-autonomous-driving train-capsnet-medical-scan train-capsnet-text-recognition train-gnn-social-networks train-pinn-heat-equation train-snn-image-classification train-diffusion train-gan train-transformers train-vae train-code-generation train-image-generation train-prompt-engineering train-retrieval-augmented-generation train-text-generation train-tool-use-functional-calling train-video-generation train-autoencoders-dimensionality-reduction train-deep-belief-networks train-restricted-boltzmann-machines train-self-organizing-maps

.PHONY: serve-pizza-price
serve-pizza-price:
	MODEL_DIR=./artifacts/models uv run uvicorn pizza_price.api:app --host 0.0.0.0 --port 8000

.PHONY: serve-spam-classification
serve-spam-classification:
	MODEL_DIR=./artifacts/models uv run uvicorn spam_classification.api:app --host 0.0.0.0 --port 8001

.PHONY: serve-market-segmentation
serve-market-segmentation:
	MODEL_DIR=./artifacts/models uv run uvicorn market_segmentation.api:app --host 0.0.0.0 --port 8002

.PHONY: serve-recommendation-engine
serve-recommendation-engine:
	MODEL_DIR=./artifacts/models uv run uvicorn recommendation_engine.api:app --host 0.0.0.0 --port 8003

.PHONY: serve-anomaly-detection-pca
serve-anomaly-detection-pca:
	MODEL_DIR=./artifacts/models uv run uvicorn anomaly_detection.api:app --host 0.0.0.0 --port 8004

.PHONY: serve-robot-maze-navigation
serve-robot-maze-navigation:
	MODEL_DIR=./artifacts/models uv run uvicorn robot_maze_navigation.api:app --host 0.0.0.0 --port 8005

.PHONY: serve-semi-supervised-email
serve-semi-supervised-email:
	MODEL_DIR=./artifacts/models uv run uvicorn semi_supervised_email.api:app --host 0.0.0.0 --port 8006

.PHONY: serve-self-supervised-monitoring
serve-self-supervised-monitoring:
	MODEL_DIR=./artifacts/models uv run uvicorn self_supervised_monitoring.api:app --host 0.0.0.0 --port 8007

.PHONY: serve-classification-email-spam
serve-classification-email-spam:
	MODEL_DIR=./artifacts/models uv run uvicorn classification_email_spam.api:app --host 0.0.0.0 --port 8008

.PHONY: serve-regression-house-price
serve-regression-house-price:
	MODEL_DIR=./artifacts/models uv run uvicorn regression_house_price.api:app --host 0.0.0.0 --port 8009

.PHONY: serve-anomaly-detection-fraud
serve-anomaly-detection-fraud:
	MODEL_DIR=./artifacts/models uv run uvicorn anomaly_detection_fraud.api:app --host 0.0.0.0 --port 8010

.PHONY: serve-pattern-recognition-digits
serve-pattern-recognition-digits:
	MODEL_DIR=./artifacts/models uv run uvicorn pattern_recognition_digits.api:app --host 0.0.0.0 --port 8011

.PHONY: serve-nlp-language-translation
serve-nlp-language-translation:
	MODEL_DIR=./artifacts/models uv run uvicorn nlp_language_translation.api:app --host 0.0.0.0 --port 8012

.PHONY: serve-nlp-sentiment-analysis
serve-nlp-sentiment-analysis:
	MODEL_DIR=./artifacts/models uv run uvicorn nlp_sentiment_analysis.api:app --host 0.0.0.0 --port 8013

.PHONY: serve-nlp-text-generation
serve-nlp-text-generation:
	MODEL_DIR=./artifacts/models uv run uvicorn nlp_text_generation.api:app --host 0.0.0.0 --port 8014

.PHONY: serve-speech-audio-recognition
serve-speech-audio-recognition:
	MODEL_DIR=./artifacts/models uv run uvicorn speech_audio_recognition.api:app --host 0.0.0.0 --port 8015

.PHONY: serve-speech-audio-music
serve-speech-audio-music:
	MODEL_DIR=./artifacts/models uv run uvicorn speech_audio_music.api:app --host 0.0.0.0 --port 8016

.PHONY: serve-time-series-stock
serve-time-series-stock:
	MODEL_DIR=./artifacts/models uv run uvicorn time_series_stock.api:app --host 0.0.0.0 --port 8017

.PHONY: serve-time-series-weather
serve-time-series-weather:
	MODEL_DIR=./artifacts/models uv run uvicorn time_series_weather.api:app --host 0.0.0.0 --port 8018

.PHONY: serve-vision-image-captioning
serve-vision-image-captioning:
	MODEL_DIR=./artifacts/models uv run uvicorn vision_image_captioning.api:app --host 0.0.0.0 --port 8019

.PHONY: serve-cnn-medical-imaging
serve-cnn-medical-imaging:
	MODEL_DIR=./artifacts/models uv run uvicorn cnn_medical_imaging.api:app --host 0.0.0.0 --port 8020

.PHONY: serve-cnn-facial-recognition
serve-cnn-facial-recognition:
	MODEL_DIR=./artifacts/models uv run uvicorn cnn_facial_recognition.api:app --host 0.0.0.0 --port 8021

.PHONY: serve-cnn-video-surveillance
serve-cnn-video-surveillance:
	MODEL_DIR=./artifacts/models uv run uvicorn cnn_video_surveillance.api:app --host 0.0.0.0 --port 8022

.PHONY: serve-advanced-super-resolution
serve-advanced-super-resolution:
	MODEL_DIR=./artifacts/models uv run uvicorn advanced_super_resolution.api:app --host 0.0.0.0 --port 8023

.PHONY: serve-advanced-semantic-segmentation
serve-advanced-semantic-segmentation:
	MODEL_DIR=./artifacts/models uv run uvicorn advanced_semantic_segmentation.api:app --host 0.0.0.0 --port 8024

.PHONY: serve-advanced-generative-art
serve-advanced-generative-art:
	MODEL_DIR=./artifacts/models uv run uvicorn advanced_generative_art.api:app --host 0.0.0.0 --port 8025

.PHONY: serve-capsnet-autonomous-driving
serve-capsnet-autonomous-driving:
	MODEL_DIR=./artifacts/models uv run uvicorn capsnet_autonomous_driving.api:app --host 0.0.0.0 --port 8026

.PHONY: serve-capsnet-medical-scan
serve-capsnet-medical-scan:
	MODEL_DIR=./artifacts/models uv run uvicorn capsnet_medical_scan.api:app --host 0.0.0.0 --port 8027

.PHONY: serve-capsnet-text-recognition
serve-capsnet-text-recognition:
	MODEL_DIR=./artifacts/models uv run uvicorn capsnet_text_recognition.api:app --host 0.0.0.0 --port 8028

.PHONY: serve-gnn-social-networks
serve-gnn-social-networks:
	MODEL_DIR=./artifacts/models uv run uvicorn gnn_social_networks.api:app --host 0.0.0.0 --port 8029

.PHONY: serve-pinn-heat-equation
serve-pinn-heat-equation:
	MODEL_DIR=./artifacts/models uv run uvicorn pinn_heat_equation.api:app --host 0.0.0.0 --port 8030

.PHONY: serve-snn-image-classification
serve-snn-image-classification:
	MODEL_DIR=./artifacts/models uv run uvicorn snn_image_classification.api:app --host 0.0.0.0 --port 8031

.PHONY: serve-diffusion
serve-diffusion:
	MODEL_DIR=./artifacts/models uv run uvicorn diffusion_image_generation.api:app --host 0.0.0.0 --port 8032

.PHONY: serve-gan
serve-gan:
	MODEL_DIR=./artifacts/models uv run uvicorn gan_image_generation.api:app --host 0.0.0.0 --port 8033

.PHONY: serve-transformers
serve-transformers:
	MODEL_DIR=./artifacts/models uv run uvicorn transformer_language_modeling.api:app --host 0.0.0.0 --port 8034

.PHONY: serve-vae
serve-vae:
	MODEL_DIR=./artifacts/models uv run uvicorn vae_data_generation.api:app --host 0.0.0.0 --port 8035

.PHONY: serve-code-generation
serve-code-generation:
	MODEL_DIR=./artifacts/models uv run uvicorn code_generation.api:app --host 0.0.0.0 --port 8036

.PHONY: serve-image-generation
serve-image-generation:
	MODEL_DIR=./artifacts/models uv run uvicorn image_generation.api:app --host 0.0.0.0 --port 8037

.PHONY: serve-prompt-engineering
serve-prompt-engineering:
	MODEL_DIR=./artifacts/models uv run uvicorn prompt_engineering.api:app --host 0.0.0.0 --port 8038

.PHONY: serve-retrieval-augmented-generation
serve-retrieval-augmented-generation:
	MODEL_DIR=./artifacts/models uv run uvicorn retrieval_augmented_generation.api:app --host 0.0.0.0 --port 8039

.PHONY: serve-text-generation
serve-text-generation:
	MODEL_DIR=./artifacts/models uv run uvicorn text_generation.api:app --host 0.0.0.0 --port 8040

.PHONY: serve-tool-use-functional-calling
serve-tool-use-functional-calling:
	MODEL_DIR=./artifacts/models uv run uvicorn tool_use_and_functional_calling.api:app --host 0.0.0.0 --port 8041

.PHONY: serve-video-generation
serve-video-generation:
	MODEL_DIR=./artifacts/models uv run uvicorn video_generation.api:app --host 0.0.0.0 --port 8042

.PHONY: serve-autoencoders-dimensionality-reduction
serve-autoencoders-dimensionality-reduction:
	MODEL_DIR=./artifacts/models uv run uvicorn autoencoders_dimensionality_reduction.api:app --host 0.0.0.0 --port 8043

.PHONY: serve-deep-belief-networks
serve-deep-belief-networks:
	MODEL_DIR=./artifacts/models uv run uvicorn deep_belief_networks.api:app --host 0.0.0.0 --port 8044

.PHONY: serve-restricted-boltzmann-machines
serve-restricted-boltzmann-machines:
	MODEL_DIR=./artifacts/models uv run uvicorn rbm_feature_learning.api:app --host 0.0.0.0 --port 8045

.PHONY: serve-self-organizing-maps
serve-self-organizing-maps:
	MODEL_DIR=./artifacts/models uv run uvicorn self_organizing_maps.api:app --host 0.0.0.0 --port 8046

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
	@echo "  make train-pizza-price                         - Train pizza-price model locally"
	@echo "  make train-spam-classification                 - Train spam-classification model locally"
	@echo "  make train-market-segmentation                 - Train market-segmentation model locally"
	@echo "  make train-recommendation-engine               - Train recommendation-engine model locally"
	@echo "  make train-anomaly-detection-pca               - Train anomaly-detection-pca model locally"
	@echo "  make train-robot-maze-navigation               - Train robot-maze-navigation model locally"
	@echo "  make train-semi-supervised-email               - Train semi-supervised-email model locally"
	@echo "  make train-self-supervised-monitoring          - Train self-supervised-monitoring model locally"
	@echo "  make train-classification-email-spam           - Train classification-email-spam model locally"
	@echo "  make train-regression-house-price              - Train regression-house-price model locally"
	@echo "  make train-anomaly-detection-fraud             - Train anomaly-detection-fraud model locally"
	@echo "  make train-pattern-recognition-digits          - Train pattern-recognition-digits model locally"
	@echo "  make train-nlp-language-translation            - Train nlp-language-translation model locally"
	@echo "  make train-nlp-sentiment-analysis              - Train nlp-sentiment-analysis model locally"
	@echo "  make train-nlp-text-generation                 - Train nlp-text-generation model locally"
	@echo "  make train-speech-audio-recognition            - Train speech-audio-recognition model locally"
	@echo "  make train-speech-audio-music                  - Train speech-audio-music model locally"
	@echo "  make train-time-series-stock                   - Train time-series-stock model locally"
	@echo "  make train-time-series-weather                 - Train time-series-weather model locally"
	@echo "  make train-vision-image-captioning             - Train vision-image-captioning model locally"
	@echo "  make train-cnn-medical-imaging                 - Train cnn-medical-imaging model locally"
	@echo "  make train-cnn-facial-recognition              - Train cnn-facial-recognition model locally"
	@echo "  make train-cnn-video-surveillance              - Train cnn-video-surveillance model locally"
	@echo "  make train-advanced-super-resolution           - Train advanced-super-resolution model locally"
	@echo "  make train-advanced-semantic-segmentation      - Train advanced-semantic-segmentation model locally"
	@echo "  make train-advanced-generative-art             - Train advanced-generative-art model locally"
	@echo "  make train-capsnet-autonomous-driving          - Train capsnet-autonomous-driving model locally"
	@echo "  make train-capsnet-medical-scan                - Train capsnet-medical-scan model locally"
	@echo "  make train-capsnet-text-recognition            - Train capsnet-text-recognition model locally"
	@echo "  make train-gnn-social-networks                 - Train gnn-social-networks model locally"
	@echo "  make train-pinn-heat-equation                  - Train pinn-heat-equation model locally"
	@echo "  make train-snn-image-classification            - Train snn-image-classification model locally"
	@echo "  make train-diffusion                           - Train diffusion model locally"
	@echo "  make train-gan                                 - Train gan model locally"
	@echo "  make train-transformers                        - Train transformers model locally"
	@echo "  make train-vae                                 - Train vae model locally"
	@echo "  make train-code-generation                     - Train code-generation model locally"
	@echo "  make train-image-generation                    - Train image-generation model locally"
	@echo "  make train-prompt-engineering                  - Train prompt-engineering model locally"
	@echo "  make train-retrieval-augmented-generation      - Train retrieval-augmented-generation model locally"
	@echo "  make train-text-generation                     - Train text-generation model locally"
	@echo "  make train-tool-use-functional-calling         - Train tool-use-functional-calling model locally"
	@echo "  make train-video-generation                    - Train video-generation model locally"
	@echo "  make train-autoencoders-dimensionality-reduction - Train autoencoders-dimensionality-reduction model locally"
	@echo "  make train-deep-belief-networks                - Train deep-belief-networks model locally"
	@echo "  make train-restricted-boltzmann-machines       - Train restricted-boltzmann-machines model locally"
	@echo "  make train-self-organizing-maps                - Train self-organizing-maps model locally"
	@echo "  make train-all            - Train all models"
	@echo "  make serve-pizza-price                         - Run pizza-price API locally (port 8000)"
	@echo "  make serve-spam-classification                 - Run spam-classification API locally (port 8001)"
	@echo "  make serve-market-segmentation                 - Run market-segmentation API locally (port 8002)"
	@echo "  make serve-recommendation-engine               - Run recommendation-engine API locally (port 8003)"
	@echo "  make serve-anomaly-detection-pca               - Run anomaly-detection-pca API locally (port 8004)"
	@echo "  make serve-robot-maze-navigation               - Run robot-maze-navigation API locally (port 8005)"
	@echo "  make serve-semi-supervised-email               - Run semi-supervised-email API locally (port 8006)"
	@echo "  make serve-self-supervised-monitoring          - Run self-supervised-monitoring API locally (port 8007)"
	@echo "  make serve-classification-email-spam           - Run classification-email-spam API locally (port 8008)"
	@echo "  make serve-regression-house-price              - Run regression-house-price API locally (port 8009)"
	@echo "  make serve-anomaly-detection-fraud             - Run anomaly-detection-fraud API locally (port 8010)"
	@echo "  make serve-pattern-recognition-digits          - Run pattern-recognition-digits API locally (port 8011)"
	@echo "  make serve-nlp-language-translation            - Run nlp-language-translation API locally (port 8012)"
	@echo "  make serve-nlp-sentiment-analysis              - Run nlp-sentiment-analysis API locally (port 8013)"
	@echo "  make serve-nlp-text-generation                 - Run nlp-text-generation API locally (port 8014)"
	@echo "  make serve-speech-audio-recognition            - Run speech-audio-recognition API locally (port 8015)"
	@echo "  make serve-speech-audio-music                  - Run speech-audio-music API locally (port 8016)"
	@echo "  make serve-time-series-stock                   - Run time-series-stock API locally (port 8017)"
	@echo "  make serve-time-series-weather                 - Run time-series-weather API locally (port 8018)"
	@echo "  make serve-vision-image-captioning             - Run vision-image-captioning API locally (port 8019)"
	@echo "  make serve-cnn-medical-imaging                 - Run cnn-medical-imaging API locally (port 8020)"
	@echo "  make serve-cnn-facial-recognition              - Run cnn-facial-recognition API locally (port 8021)"
	@echo "  make serve-cnn-video-surveillance              - Run cnn-video-surveillance API locally (port 8022)"
	@echo "  make serve-advanced-super-resolution           - Run advanced-super-resolution API locally (port 8023)"
	@echo "  make serve-advanced-semantic-segmentation      - Run advanced-semantic-segmentation API locally (port 8024)"
	@echo "  make serve-advanced-generative-art             - Run advanced-generative-art API locally (port 8025)"
	@echo "  make serve-capsnet-autonomous-driving          - Run capsnet-autonomous-driving API locally (port 8026)"
	@echo "  make serve-capsnet-medical-scan                - Run capsnet-medical-scan API locally (port 8027)"
	@echo "  make serve-capsnet-text-recognition            - Run capsnet-text-recognition API locally (port 8028)"
	@echo "  make serve-gnn-social-networks                 - Run gnn-social-networks API locally (port 8029)"
	@echo "  make serve-pinn-heat-equation                  - Run pinn-heat-equation API locally (port 8030)"
	@echo "  make serve-snn-image-classification            - Run snn-image-classification API locally (port 8031)"
	@echo "  make serve-diffusion                           - Run diffusion API locally (port 8032)"
	@echo "  make serve-gan                                 - Run gan API locally (port 8033)"
	@echo "  make serve-transformers                        - Run transformers API locally (port 8034)"
	@echo "  make serve-vae                                 - Run vae API locally (port 8035)"
	@echo "  make serve-code-generation                     - Run code-generation API locally (port 8036)"
	@echo "  make serve-image-generation                    - Run image-generation API locally (port 8037)"
	@echo "  make serve-prompt-engineering                  - Run prompt-engineering API locally (port 8038)"
	@echo "  make serve-retrieval-augmented-generation      - Run retrieval-augmented-generation API locally (port 8039)"
	@echo "  make serve-text-generation                     - Run text-generation API locally (port 8040)"
	@echo "  make serve-tool-use-functional-calling         - Run tool-use-functional-calling API locally (port 8041)"
	@echo "  make serve-video-generation                    - Run video-generation API locally (port 8042)"
	@echo "  make serve-autoencoders-dimensionality-reduction - Run autoencoders-dimensionality-reduction API locally (port 8043)"
	@echo "  make serve-deep-belief-networks                - Run deep-belief-networks API locally (port 8044)"
	@echo "  make serve-restricted-boltzmann-machines       - Run restricted-boltzmann-machines API locally (port 8045)"
	@echo "  make serve-self-organizing-maps                - Run self-organizing-maps API locally (port 8046)"
	@echo "  make serve-all            - Run all APIs locally"
	@echo ""
	@echo "Docker:"
	@echo "  make docker-build         - Build Docker images"
	@echo "  make docker-clean         - Remove Docker artifacts"
	@echo ""
	@echo "Kubernetes:"
	@echo "  make k8s-apply            - Deploy everything to Kubernetes"
	@echo "  make k8s-delete           - Delete Kubernetes namespace"
	@echo "  make k8s-status           - Show Kubernetes status"
