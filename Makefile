.PHONY: dev run download install lint format check test smoke swagger clean help

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-15s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (including dev)
	uv sync

download: ## Download model files
	bash scripts/download.sh

dev: ## Start dev server with auto-reload
	uv run uvicorn slm_server.app:app --reload --host 0.0.0.0 --port 8000

run: ## Start server via start.sh
	bash scripts/start.sh

lint: ## Run ruff linter
	uv run ruff check slm_server/

format: ## Run ruff formatter
	uv run ruff format slm_server/

check: lint ## Run linter + formatter check
	uv run ruff format --check slm_server/

smoke: ## Smoke-test the running server APIs with curl
	bash scripts/smoke.sh

test: ## Run tests with coverage
	uv run pytest tests/ -v --cov=slm_server --cov-report=term-missing

swagger: ## Refresh OpenAPI spec from running server
	curl -sf http://localhost:8000/openapi.json | uv run python -c "import sys,json,yaml;yaml.dump(json.load(sys.stdin),sys.stdout,default_flow_style=False,sort_keys=False,allow_unicode=True)" > swagger/openapi.yaml
	@echo "swagger/openapi.yaml updated"

clean: ## Remove caches and build artifacts
	rm -rf __pycache__ .pytest_cache .ruff_cache .coverage htmlcov build dist *.egg-info
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
