.DEFAULT_GOAL := help
.PHONY: help sync lock test test-unit test-integration test-ui test-e2e test-live coverage lint format typecheck security check eval build ui api demo graph compose-up compose-down

help:
	@printf '%s\n' \
	  'sync             Install the frozen development environment' \
	  'demo             Run a complete keyless research demo' \
	  'ui               Start Research Desk on localhost:8501' \
	  'api              Start the API on localhost:8000' \
	  'check            Run formatting, lint, types, tests, and security checks' \
	  'eval             Run deterministic research-quality evaluations' \
	  'test-e2e         Run opted-in Playwright browser tests' \
	  'test-live        Run opted-in live-provider tests' \
	  'compose-up       Build and start the containerized workbench'

sync:
	uv sync --frozen --all-groups

lock:
	uv lock --check

test:
	uv run pytest --cov=research_system --cov-report=term-missing

test-unit:
	uv run pytest tests/unit tests/contract -q

test-integration:
	uv run pytest tests/integration tests/api -q

test-ui:
	uv run pytest tests/ui -q

test-e2e:
	uv run pytest --force-enable-socket -m e2e tests/e2e -q

test-live:
	uv run pytest --force-enable-socket -m live tests/live -q

coverage:
	uv run pytest --cov=research_system --cov-branch --cov-report=term-missing --cov-report=xml

lint:
	uv run ruff check .
	uv run ruff format --check .

format:
	uv run ruff check --fix .
	uv run ruff format .

typecheck:
	uv run mypy src

security:
	uv run bandit -q -r src
	uv run pip-audit

check: lock lint typecheck coverage security

eval:
	uv run pytest tests/eval -q

build:
	uv build

ui:
	uv run research-desk ui

api:
	uv run research-desk api

demo:
	uv run research-desk demo

graph:
	uv run research-desk graph

compose-up:
	docker compose up --build -d

compose-down:
	docker compose down
