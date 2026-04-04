# Makefile
.PHONY: help install dev test lint run docker-up docker-down

help:  ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | sort | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies
	pip install -e ".[dev]"

test:  ## Run tests
	pytest tests/ -v --cov=src/rag_engine --cov-report=term-missing

lint:  ## Run linters
	ruff check src/ tests/
	mypy src/

run:  ## Run API locally
	uvicorn rag_engine.api.app:app --reload --port 8000

docker-up:  ## Start all services
	docker compose -f docker/docker-compose.yml up --build -d

docker-down:  ## Stop all services
	docker compose -f docker/docker-compose.yml down