set quiet
set dotenv-load := false

# ─── Default ──────────────────────────────────────────────────────────────────

# List all available recipes
default:
    @just --list --unsorted

# ─── Environment Setup ────────────────────────────────────────────────────────

# Install all dependencies (dev tools + optional deps)
install:
    uv sync --all-extras --all-groups

# Install core dependencies only (no optional deps)
install-core:
    uv sync --group dev

# Install documentation dependencies only
install-docs:
    uv sync --group docs

# ─── Code Quality ─────────────────────────────────────────────────────────────

# Format code with ruff
format:
    uv run ruff format .

# Check formatting without modifying files
format-check:
    uv run ruff format --check .

# Lint code with ruff
lint:
    uv run ruff check .

# Lint and auto-fix issues
lint-fix:
    uv run ruff check . --fix

# Run ty type checker on maseval and tests
typecheck:
    uv run ty check maseval tests

# Run all quality checks (format, lint, typecheck)
check: format-check lint typecheck

# Format and fix lint issues
fix: format lint-fix

# ─── Testing — Core ──────────────────────────────────────────────────────────

# Run default test suite (excludes slow, credentialed, smoke)
test *ARGS:
    uv run pytest -v {{ ARGS }}

# Run core tests only (no optional deps needed)
test-core *ARGS:
    uv run pytest -m core -v {{ ARGS }}

# Run interface tests (requires optional deps)
test-interface *ARGS:
    uv run pytest -m interface -v {{ ARGS }}

# Run cross-implementation contract tests
test-contract *ARGS:
    uv run pytest -m contract -v {{ ARGS }}

# Run fast benchmark tests (excludes slow/live)
test-benchmark *ARGS:
    uv run pytest -m "benchmark and not (slow or live)" -v {{ ARGS }}

# ─── Testing — Frameworks ────────────────────────────────────────────────────

# Run smolagents framework tests
test-smolagents *ARGS:
    uv run pytest -m smolagents -v {{ ARGS }}

# Run langgraph framework tests
test-langgraph *ARGS:
    uv run pytest -m langgraph -v {{ ARGS }}

# Run llamaindex framework tests
test-llamaindex *ARGS:
    uv run pytest -m llamaindex -v {{ ARGS }}

# Run gaia2 benchmark tests
test-gaia2 *ARGS:
    uv run pytest -m gaia2 -v {{ ARGS }}

# Run camel framework tests
test-camel *ARGS:
    uv run pytest -m camel -v {{ ARGS }}

# ─── Testing — Special Modes ─────────────────────────────────────────────────

# Run slow tests — data downloads + integrity checks (needs network)
test-slow *ARGS:
    uv run pytest -m "(slow or live) and not credentialed" -v {{ ARGS }}

# Run credentialed tests — live API calls (needs API keys)
test-credentialed *ARGS:
    uv run pytest -m "credentialed and not smoke" -v {{ ARGS }}

# Run smoke tests — full end-to-end pipeline validation
test-smoke *ARGS:
    uv run pytest -m smoke -v {{ ARGS }}

# Run fully offline tests (no network access)
test-offline *ARGS:
    uv run pytest -m "not live" -v {{ ARGS }}

# Run all tests except smoke
test-all *ARGS:
    uv run pytest -m "not smoke" -v {{ ARGS }}

# ─── All-in-One ───────────────────────────────────────────────────────────────

# Fix, check, and run default tests
all: fix typecheck test

# Fix, check, and run all tests including slow (excludes credentialed/smoke)
all-slow: fix typecheck test-all test-slow

# ─── Coverage ─────────────────────────────────────────────────────────────────

# Run default tests with coverage and print report
coverage *ARGS:
    uv run coverage run -m pytest -v {{ ARGS }}
    uv run coverage report

# Generate HTML coverage report and open it
coverage-html: coverage
    uv run coverage html
    open htmlcov/index.html

# ─── Documentation ────────────────────────────────────────────────────────────

# Serve documentation locally (http://127.0.0.1:8000)
docs:
    uv run mkdocs serve

# Build documentation with strict checking
docs-build:
    uv run mkdocs build --strict

# ─── Utilities ────────────────────────────────────────────────────────────────

# Find tests without any CI-mapped marker (potential orphans)
orphans:
    #!/usr/bin/env bash
    uv run pytest --collect-only -q -m "not (core or benchmark or interface or slow or live or credentialed or smoke)" && \
        echo "⚠ Orphaned tests found above — add appropriate markers" || \
        echo "No orphaned tests found"

# Remove build artifacts and caches
clean:
    rm -rf .coverage .coverage.* htmlcov/ site/ dist/ .pytest_cache/
    find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true

# Install pre-commit hooks
pre-commit:
    uv run pre-commit install
