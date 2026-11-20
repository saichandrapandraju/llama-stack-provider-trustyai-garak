RHAI_INDEX_URL := https://console.redhat.com/api/pypi/public-rhai/rhoai/3.6/cpu-ubi9/simple/

.PHONY: test coverage lint format typecheck check build lock install install-dev e2e e2e-sweep e2e-sweep-full

test:
	pytest tests -v -m "not e2e"

# EvalHub contract tests: real eval-hub-server (local mode) + adapter + garak vs a mock model.
# Needs garak and eval-hub-server installed; see tests/e2e/README.md.
e2e:
	E2E_REQUIRED=1 pytest tests/e2e -m "e2e and not e2e_sweep"

# Every taxonomy benchmark: capped (one prompt per probe; what PRs run) or full (nightly).
e2e-sweep:
	E2E_REQUIRED=1 E2E_SWEEP=capped pytest tests/e2e -m e2e_sweep

e2e-sweep-full:
	E2E_REQUIRED=1 E2E_SWEEP=full pytest tests/e2e -m e2e_sweep

coverage:
	pytest tests -v -m "not e2e" --cov=llama_stack_provider_trustyai_garak --cov-report=term-missing

lint:
	ruff check src/ tests/

format:
	ruff format src/ tests/

typecheck:
	mypy src/

check: lint typecheck test

build:
	docker build -f Containerfile -t trustyai-garak:dev .

lock:
	uv pip compile \
		--python-platform linux \
		--extra inline \
		--index-strategy first-index \
		--emit-index-url \
		--default-index $(RHAI_INDEX_URL) \
		pyproject.toml \
		--index-url $(RHAI_INDEX_URL) \
		--constraint constraints.txt \
		-o requirements.txt

install:
	pip install -e .

install-dev:
	pip install -e ".[dev]"
	pre-commit install
