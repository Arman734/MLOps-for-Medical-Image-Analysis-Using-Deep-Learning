# MedMLOps developer shortcuts. Windows (Git Bash) and POSIX friendly.
SHELL := /bin/sh

ifeq ($(OS),Windows_NT)
PY := .venv/Scripts/python.exe
else
PY := .venv/bin/python
endif

PYTHON ?= $(PY)
MEDMLOPS := $(PYTHON) -m medmlops.cli
DVC := $(PYTHON) -m dvc
MLFLOW := $(PYTHON) -m mlflow
PYTEST := $(PYTHON) -m pytest
RUFF := $(PYTHON) -m ruff

.DEFAULT_GOAL := help

.PHONY: help venv install download data train train-seg train-clf evaluate \
        serve api web-install web-build dev test lint repro mlflow docker-build \
        docker-up docker-down docker-logs smoke

help: ## Show available targets
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

venv: ## Create the Python 3.12 virtual environment
	uv venv --python 3.12 .venv

install: ## Install the package with dev, DVC and DagsHub extras + web deps
	uv pip install --python $(PYTHON) -e ".[dagshub,dvc,dev]"
	cd web && npm install

download: ## Download ISIC 2018 and PneumoniaMNIST raw data
	$(MEDMLOPS) data download --config configs/isic_unet.yaml
	$(MEDMLOPS) data prepare --config configs/pneumonia_densenet.yaml

data: ## Preprocess both datasets (npz splits, manifests, drift baselines)
	$(MEDMLOPS) data prepare --config configs/isic_unet.yaml
	$(MEDMLOPS) data prepare --config configs/pneumonia_densenet.yaml

train-seg: ## Train the U-Net segmentation model
	$(MEDMLOPS) train --config configs/isic_unet.yaml

train-clf: ## Train the DenseNet-121 classification model
	$(MEDMLOPS) train --config configs/pneumonia_densenet.yaml

train: train-seg train-clf ## Train both models

evaluate: ## Evaluate both checkpoints on the test split
	$(MEDMLOPS) evaluate --config configs/isic_unet.yaml --split test
	$(MEDMLOPS) evaluate --config configs/pneumonia_densenet.yaml --split test

serve: ## Start the FastAPI API + built React console on :8000
	$(MEDMLOPS) serve --host 127.0.0.1 --port 8000

api: ## Start the FastAPI dev server with autoreload
	$(PYTHON) -m uvicorn api.app:app --host 127.0.0.1 --port 8000 --reload

web-install: ## Install frontend dependencies
	cd web && npm install

web-build: ## Type-check and build the React console into web/dist
	cd web && npm run build

dev: ## Run the Vite dev server (proxies /api to :8000)
	cd web && npm run dev

test: ## Run the test suite
	$(PYTEST)

lint: ## Lint with ruff
	$(RUFF) check src api tests scripts

repro: ## Reproduce the whole DVC pipeline
	PATH="$(CURDIR)/.venv/Scripts:$(CURDIR)/.venv/bin:$$PATH" $(DVC) repro

mlflow: ## Serve the local MLflow UI on :5000
	$(MLFLOW) server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts --host 127.0.0.1 --port 5000

docker-build: ## Build the API and web images
	docker compose build

docker-up: ## Start api, web, mlflow, prometheus and grafana
	docker compose up -d

docker-down: ## Stop the compose stack
	docker compose down

docker-logs: ## Tail compose logs
	docker compose logs -f --tail=100

smoke: ## Run a fast end-to-end smoke test against the local API
	$(PYTHON) scripts/smoke_test.py
