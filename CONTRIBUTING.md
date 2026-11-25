# Contributing

## Prerequisites

- Python 3.12+
- [uv](https://github.com/astral-sh/uv) (for lockfile generation)
- Docker/Podman (for container builds)

## Setup

```bash
git clone https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak.git
cd llama-stack-provider-trustyai-garak

python -m venv .venv
source .venv/bin/activate

pip install -e ".[dev]"
pre-commit install
```

## Dependency Management

Dependencies are declared in `pyproject.toml` with optional extras:

| Extra | Install Command | What You Get |
|-------|----------------|-------------|
| (none) | `pip install -e .` | Core provider (Llama Stack remote mode) |
| `[inline]` | `pip install -e ".[inline]"` | Core + garak for local scans |
| `[dev]` | `pip install -e ".[dev]"` | Tests + ruff + pre-commit |
| `[server]` | `pip install -e ".[server]"` | Llama Stack server |

**Lockfile**: `requirements.txt` is a pinned lockfile generated from the
[RH AI PyPI index](https://console.redhat.com/api/pypi/public-rhai/rhoai/3.6/cpu-ubi9/simple/)
via `uv pip compile`. It is the single source of truth for package versions: downstream
production builds prefetch it hermetically, and CI plus the dev `Containerfile` install
exactly the same pins from public indexes with `scripts/install-locked-deps.sh` (the
production index is reserved for product builds). Regenerate with `make lock`.

**Container image**: The dev `Containerfile` installs the locked set from public
indexes (CPU torch, garak from the fork's tag for the locked version, the rest from PyPI).
Production images use the AIPCC base image with the RH AI index. See `Containerfile` for the dev build and
`Dockerfile.konflux` (downstream) for production.

## Development Workflow

### Running Tests

```bash
make test          # unit tests (pytest -m "not e2e")
make coverage      # pytest with coverage report
make e2e           # EvalHub contract tests
```

Unit tests need no cluster, GPU, or network access; garak is mocked and need not be
installed. The EvalHub contract tests (`make e2e`) run the real eval-hub server, the
adapter and garak against a mock model, without a cluster; see
[`tests/e2e/README.md`](tests/e2e/README.md) for setup.

### Linting & Formatting

```bash
make lint          # ruff check
make format        # ruff format (auto-fix)
```

Pre-commit hooks run `ruff` automatically on staged files.

### Building the Container Image

```bash
make build         # builds dev Containerfile
```

The dev image installs deps from PyPI and garak from the midstream git branch.
Production images use the AIPCC base image with the RH AI PyPI index.

### Regenerating requirements.txt

When you change `pyproject.toml`, the pre-commit hook automatically regenerates
`requirements.txt` via `uv pip compile`. If you need to do it manually:

```bash
make lock
```

For pull requests from forks, CI regenerates `requirements.txt` and checks that
the committed file matches in a separate job with read-only repository permissions.
Run `make lock` and commit the result before pushing. Pull requests from branches
in this repository retain automatic lockfile commits in a write-enabled job
that only runs for same-repository pull requests.

The downstream image is built for x86_64, aarch64, ppc64le and s390x, but its
pull-request builds cover x86_64 and aarch64 only. CI therefore also checks that the
index has a wheel of every locked version each of those architectures installs (`make check-arch-wheels`,
`scripts/check_arch_wheels.py`), on pull requests that change the lock and weekly.
If it fails, that version is missing from the AIPCC index for that architecture:
pick a version that has it, or ask the AIPCC team to build it.

## Project Structure

See [ARCHITECTURE.md](ARCHITECTURE.md) for a detailed module-by-module breakdown.

```
src/llama_stack_provider_trustyai_garak/
├── core/           # Shared logic (config resolution, commands, runner, pipeline steps)
├── evalhub/        # Eval-Hub adapter (simple + KFP modes)
├── config.py       # Benchmark profiles (GarakScanConfig)
└── resources/      # Report templates and chart specs
tests/              # Unit tests (no external deps needed)
tests/e2e/          # EvalHub contract tests (real eval-hub server + garak, mock model)
```

Both execution modes (simple and KFP) support the **intents benchmark**; see
ARCHITECTURE.md for details.

## Adding a Benchmark Profile

1. Add the profile to `GarakScanConfig` in `config.py`: `FRAMEWORK_PROFILES` for a
   framework selected by garak taxonomy tags, `SCAN_PROFILES` for an explicit probe list
2. Add tests in `tests/test_evalhub_adapter.py`
3. List it in the benchmark table in `README.md`
4. Add it to eval-hub's garak catalog (`garak.yaml`) so eval-hub serves it. Until then
   the e2e sweep runs it as `<id>-adapter-only` and warns (see `tests/e2e/README.md`)

## Code Conventions

- Use type annotations for function signatures
- Config merging uses `deep_merge_dicts` — only leaf values are replaced
- `api_key` fields use `__FROM_ENV__` placeholder for K8s secret injection
- Tests mock garak — never import garak directly in test code

## Submitting Changes

1. Create a feature branch from `main`
2. Make your changes with tests
3. Run `make lint test` to validate locally
4. Push and open a PR — CI will run unit tests, security scans, and dep validation
5. Fill out the PR template checklist

## Reporting Bugs

Use the [bug report template](https://github.com/trustyai-explainability/llama-stack-provider-trustyai-garak/issues/new?template=bug_report.yml)
on GitHub Issues.
