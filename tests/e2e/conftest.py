"""Fixtures for the EvalHub contract e2e tests.

These tests run the *real* stack without a cluster:

    pytest ──REST──▶ eval-hub-server (-local) ──sh -c──▶ garak adapter ──▶ garak ──▶ mock model
                         ▲                                    │
                         └──────────── /events callbacks ─────┘

so they exercise the job.json contract, the status/result callbacks, the server's
result handling (metrics, pass criteria) and garak itself, which unit tests mock.

Inputs (all optional):

- ``EVALHUB_SERVER_BIN``     eval-hub server binary; defaults to ``eval-hub-server`` on PATH
                             (installed by the ``eval-hub-server`` wheel).
- ``E2E_GARAK_PROVIDER_YAML`` path or http(s) URL of the garak provider definition to load
                             (benchmark catalog, primary score, pass criteria). Defaults to
                             eval-hub's ``config/providers/garak.yaml`` at the tag matching the
                             installed ``eval-hub-server`` (pinned in the ``test`` extra), so the
                             catalog always matches the server. Required with ``EVALHUB_SERVER_BIN``;
                             the nightly canary points it at the eval-hub checkout it builds.
- ``E2E_REQUIRED=1``          fail instead of skip when the stack is not installed (CI sets it,
                             so a broken CI environment cannot silently skip every test).
- ``E2E_ARTIFACTS_DIR``       where to copy server/adapter logs for upload; defaults to a tmp dir.
- ``E2E_SWEEP=capped|full``   also run the ``e2e_sweep`` tests (every taxonomy benchmark): ``capped``
                             (or ``1``) caps each probe to one prompt and one generation (minutes);
                             ``full`` runs each profile as shipped (much longer).
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from .mock_openai_server import MockOpenAIServer

log = logging.getLogger("e2e")
PROGRESS_EVERY = float(os.getenv("E2E_PROGRESS_EVERY", "30"))
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
# garak prints one "<probe> <detector>: PASS|FAIL  ok on N/M" line per finished probe/detector pair.
_RESULT_LINE = re.compile(r"\b(?:PASS|FAIL)\b.*?ok on\s+\d+\s*/\s*\d+")

HERE = Path(__file__).parent
CATALOG_URL = "https://raw.githubusercontent.com/eval-hub/eval-hub/v{version}/config/providers/garak.yaml"
MOCK_API_KEY = "e2e-model-key"
# partially_failed: a multi-benchmark job where some, not all, benchmarks failed.
TERMINAL_STATES = {"completed", "failed", "cancelled", "partially_failed"}
# eval-hub's local runtime writes job specs and adapter logs here (not configurable).
LOCAL_JOBS_DIR = Path("/tmp/evalhub-jobs")


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _stack_problem() -> str | None:
    """Return why the e2e stack cannot run here, or None if it can."""
    try:
        import garak  # noqa: F401
        import yaml  # noqa: F401
        from evalhub.adapter import FrameworkAdapter  # noqa: F401
    except ImportError as exc:
        return f"e2e dependencies missing ({exc}); install the provider with garak and eval-hub-server"
    server = os.getenv("EVALHUB_SERVER_BIN") or shutil.which("eval-hub-server")
    if not server:
        return "eval-hub-server not found; pip install eval-hub-server or set EVALHUB_SERVER_BIN"
    return None


def pytest_report_header(config: pytest.Config) -> list[str]:
    """Record exactly what a run tested: which adapter install and which versions."""
    from importlib.metadata import PackageNotFoundError, version

    lines = []
    try:
        import llama_stack_provider_trustyai_garak as provider

        lines.append(f"e2e adapter package: {Path(provider.__file__).parent}")
    except ImportError:
        pass
    versions = []
    for pkg in ("garak", "eval-hub-sdk", "eval-hub-server", "transformers", "torch"):
        try:
            versions.append(f"{pkg}=={version(pkg)}")
        except PackageNotFoundError:
            versions.append(f"{pkg}=(missing)")
    lines.append("e2e versions: " + ", ".join(versions))
    if os.getenv("EVALHUB_SERVER_BIN"):
        lines.append(f"e2e server binary: {os.environ['EVALHUB_SERVER_BIN']}")
    return lines


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    if os.getenv("E2E_SWEEP") not in ("1", "capped", "full"):
        skip_sweep = pytest.mark.skip(reason="benchmark sweep; set E2E_SWEEP=capped or E2E_SWEEP=full to run")
        for item in items:
            if item.get_closest_marker("e2e_sweep"):
                item.add_marker(skip_sweep)
    e2e_items = [item for item in items if item.get_closest_marker("e2e")]
    if not e2e_items:
        return
    problem = _stack_problem()
    if problem is None:
        return
    if os.getenv("E2E_REQUIRED") == "1":
        raise pytest.UsageError(f"E2E_REQUIRED=1 but the e2e stack is unavailable: {problem}")
    skip = pytest.mark.skip(reason=problem)
    for item in e2e_items:
        item.add_marker(skip)


def _default_catalog_source() -> str:
    if os.getenv("EVALHUB_SERVER_BIN"):
        raise RuntimeError(
            "EVALHUB_SERVER_BIN is set, so the server version is unknown: set E2E_GARAK_PROVIDER_YAML "
            "to the garak.yaml of the eval-hub source the binary was built from"
        )
    from importlib.metadata import version

    return CATALOG_URL.format(version=version("eval-hub-server"))


def load_provider_definition() -> dict[str, Any]:
    import yaml

    source = os.getenv("E2E_GARAK_PROVIDER_YAML") or _default_catalog_source()
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source, timeout=30) as resp:  # noqa: S310 - test input
            text = resp.read().decode()
    else:
        text = Path(source).read_text()
    definition = yaml.safe_load(text)
    definition["_source"] = source
    return definition


@pytest.fixture(scope="session")
def artifacts_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    path = Path(os.getenv("E2E_ARTIFACTS_DIR") or tmp_path_factory.mktemp("e2e-artifacts"))
    path.mkdir(parents=True, exist_ok=True)
    return path


@pytest.fixture(scope="session")
def mock_model() -> Iterator[MockOpenAIServer]:
    """The target model. Requires MOCK_API_KEY, so every job also proves credential plumbing."""
    with MockOpenAIServer(api_key=MOCK_API_KEY) as server:
        yield server


@pytest.fixture(scope="session")
def model_secret_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """A local-mode model secret: ``model.auth.secret_ref: file:///<dir>`` with an ``api-key`` file."""
    path = tmp_path_factory.mktemp("model-secret")
    (path / "api-key").write_text(MOCK_API_KEY)
    return path


# Benchmarks with their own tests in test_evalhub_local.py; every other benchmark is swept.
DEDICATED_BENCHMARKS = {"quick", "intents"}
_PROFILE_PREFIX = "trustyai_garak::"


def adapter_profiles() -> dict[str, dict[str, Any]]:
    """The adapter's benchmark profiles, by eval-hub benchmark id."""
    from llama_stack_provider_trustyai_garak.config import GarakScanConfig

    config = GarakScanConfig()
    profiles = {**config.FRAMEWORK_PROFILES, **config.SCAN_PROFILES}
    return {key.removeprefix(_PROFILE_PREFIX): profile for key, profile in profiles.items()}


def adapter_only_benchmarks(definition: dict[str, Any]) -> list[str]:
    """Adapter profiles the eval-hub catalog doesn't list yet.

    Benchmarks are added here first, then to eval-hub's catalog (and the operator), so
    a new profile is normally adapter-only for a while.
    """
    catalog = {b["id"] for b in definition["benchmarks"]}
    return sorted(bid for bid in adapter_profiles() if bid not in catalog)


def with_adapter_only_benchmarks(definition: dict[str, Any]) -> dict[str, Any]:
    """The catalog plus a stand-in entry for each adapter-only profile.

    Lets the server accept, and the sweep run, a benchmark before eval-hub lists it. The
    stand-in copies a real taxonomy entry (same primary score and pass criteria), so it
    tests the adapter, not catalog metadata that doesn't exist yet.
    """
    extra = adapter_only_benchmarks(definition)
    if not extra:
        return definition
    benchmarks = definition["benchmarks"]
    template = next((b for b in benchmarks if b["id"] not in DEDICATED_BENCHMARKS), benchmarks[0])
    profiles = adapter_profiles()
    stand_ins = []
    for bid in extra:
        entry = {k: v for k, v in template.items() if k not in ("agent", "url")}
        entry.update(
            id=bid,
            name=profiles[bid].get("name") or bid,
            description=(profiles[bid].get("description") or bid)[:1024],
            tags=["adapter-only"],
        )
        stand_ins.append(entry)
    return {**definition, "benchmarks": [*benchmarks, *stand_ins]}


@pytest.fixture(scope="session")
def provider_definition() -> dict[str, Any]:
    """eval-hub's garak catalog, as published (contract tests check it as is)."""
    return load_provider_definition()


class EvalHub:
    """Minimal REST client for the eval-hub API, plus local-mode diagnostics."""

    def __init__(self, base_url: str, artifacts_dir: Path) -> None:
        self.base_url = base_url
        self.artifacts_dir = artifacts_dir

    def request(self, method: str, path: str, body: dict[str, Any] | None = None) -> tuple[int, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            f"{self.base_url}{path}", data=data, method=method, headers={"Content-Type": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310 - local test server
                raw = resp.read()
                return resp.status, json.loads(raw) if raw else None
        except urllib.error.HTTPError as exc:
            raw = exc.read()
            try:
                return exc.code, json.loads(raw)
            except json.JSONDecodeError:
                return exc.code, raw.decode(errors="replace")

    def get(self, path: str) -> Any:
        status, body = self.request("GET", path)
        assert status == 200, f"GET {path} -> {status}: {body}"
        return body

    def submit(self, payload: dict[str, Any]) -> str:
        status, body = self.request("POST", "/api/v1/evaluations/jobs", payload)
        assert status in (200, 201, 202), f"job submission rejected ({status}): {body}"
        return body["resource"]["id"]

    def job(self, job_id: str) -> dict[str, Any]:
        return self.get(f"/api/v1/evaluations/jobs/{job_id}")

    def wait(self, job_id: str, timeout: float = 600, poll: float = 2) -> dict[str, Any]:
        """Wait for a terminal state.

        The timeout is load-bearing: eval-hub's local runtime ignores the adapter's exit
        status, so an adapter that dies without reporting leaves the job ``running``
        forever. On timeout or failure the adapter log is attached to the assertion.
        """
        start = time.monotonic()
        deadline = start + timeout
        next_progress = start + PROGRESS_EVERY
        job = self.job(job_id)
        while job["status"]["state"] not in TERMINAL_STATES:
            now = time.monotonic()
            if now > deadline:
                # Read the logs first: cancelling makes the server delete the job directory
                # (/tmp/evalhub-jobs/<id>) before it responds.
                self.collect_logs(job_id)
                tail = self.adapter_log_tail(job_id)
                # Cancel it, so a hung scan can't keep calling the shared mock model and
                # skew the request counters later tests assert on.
                self.request("DELETE", f"/api/v1/evaluations/jobs/{job_id}")
                pytest.fail(f"job {job_id} still '{job['status']['state']}' after {timeout}s\n{tail}")
            if now >= next_progress:
                # Benchmarks run for minutes with no pytest output; with live logging on
                # (-o log_cli=true) this shows they are moving.
                log.info(
                    "%s: %s for %ds | %s",
                    job.get("name", job_id),
                    job["status"]["state"],
                    now - start,
                    self.latest_garak_progress(job_id),
                )
                next_progress = now + PROGRESS_EVERY
            time.sleep(poll)
            job = self.job(job_id)
        log.info("%s: %s after %ds", job.get("name", job_id), job["status"]["state"], time.monotonic() - start)
        self.collect_logs(job_id)
        return job

    def latest_garak_progress(self, job_id: str) -> str:
        """Finished probe/detector results so far, and garak's latest output line."""
        for log_file in self.adapter_logs(job_id):
            try:
                text = _ANSI.sub("", log_file.read_text(errors="replace"))
            except OSError:
                continue
            results = len(_RESULT_LINE.findall(text))
            latest = "no garak output yet"
            for line in reversed(re.split(r"[\r\n]+", text[-65536:])):
                for marker in ("garak[progress] ", "garak[stdout] "):
                    if marker in line and line.split(marker, 1)[1].strip():
                        latest = line.split(marker, 1)[1].strip()[:140]
                        break
                else:
                    continue
                break
            return f"{results} probe/detector results so far | {latest}"
        return "adapter not started yet"

    def adapter_logs(self, job_id: str) -> list[Path]:
        return sorted((LOCAL_JOBS_DIR / job_id).glob("**/jobrun.log"))

    def adapter_log_tail(self, job_id: str, lines: int = 60) -> str:
        chunks = []
        for log in self.adapter_logs(job_id):
            tail = log.read_text(errors="replace").splitlines()[-lines:]
            chunks.append(f"--- {log} (last {lines} lines) ---\n" + "\n".join(tail))
        return "\n".join(chunks) or f"(no adapter log under {LOCAL_JOBS_DIR / job_id})"

    def collect_logs(self, job_id: str) -> None:
        dest = self.artifacts_dir / "jobs" / job_id
        for log in self.adapter_logs(job_id):
            target = dest / log.relative_to(LOCAL_JOBS_DIR / job_id)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(log, target)
        job_json = LOCAL_JOBS_DIR / job_id
        for spec in job_json.glob("**/meta/job.json"):
            target = dest / spec.relative_to(job_json)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(spec, target)


@pytest.fixture(scope="session")
def evalhub(
    tmp_path_factory: pytest.TempPathFactory, provider_definition: dict[str, Any], artifacts_dir: Path
) -> Iterator[EvalHub]:
    """Start ``eval-hub-server -local`` with the garak provider wired to the real adapter."""
    import yaml

    config_dir = tmp_path_factory.mktemp("evalhub-config")
    scan_dir = tmp_path_factory.mktemp("garak-scans")
    port = _free_port()
    (config_dir / "config.yaml").write_text(
        yaml.safe_dump(
            {
                "service": {"port": port, "host": "127.0.0.1", "termination_file": str(config_dir / "termination")},
                "database": {"driver": "sqlite", "url": "file::eval_hub:?mode=memory&cache=shared"},
                "prometheus": {"enabled": False},
            }
        )
    )

    # The server also offers adapter profiles the catalog doesn't list yet, so they can be swept.
    served = with_adapter_only_benchmarks(provider_definition)
    provider = {k: v for k, v in served.items() if not k.startswith("_")}
    venv_bin = Path(sys.executable).parent
    # garak_runner invokes `garak` via PATH, so the adapter's own environment must come first.
    provider.setdefault("runtime", {})["local"] = {
        "command": f'PATH="{venv_bin}:$PATH" "{sys.executable}" -m llama_stack_provider_trustyai_garak.evalhub',
        "env": [
            {"name": "GARAK_SCAN_DIR", "value": str(scan_dir)},
            {"name": "LOG_LEVEL", "value": "INFO"},
        ],
    }
    providers_dir = config_dir / "providers"
    providers_dir.mkdir()
    (providers_dir / "garak.yaml").write_text(yaml.safe_dump(provider, sort_keys=False))

    server_bin = os.getenv("EVALHUB_SERVER_BIN") or shutil.which("eval-hub-server")
    assert server_bin, "eval-hub-server not found"
    log_path = artifacts_dir / "eval-hub-server.log"
    log_file = log_path.open("w")
    proc = subprocess.Popen(  # noqa: S603 - trusted test binary
        [server_bin, "-local", "-configdir", str(config_dir)],
        stdout=log_file,
        stderr=subprocess.STDOUT,
        env={**os.environ, "PORT": str(port)},
    )
    client = EvalHub(f"http://127.0.0.1:{port}", artifacts_dir)
    try:
        deadline = time.monotonic() + 60
        while True:
            if proc.poll() is not None:
                pytest.fail(f"eval-hub-server exited with {proc.returncode}:\n{log_path.read_text()[-4000:]}")
            try:
                if client.request("GET", "/api/v1/health")[0] == 200:
                    break
            except (OSError, ValueError):  # not listening yet, or a non-JSON answer while starting
                pass
            if time.monotonic() > deadline:
                pytest.fail(f"eval-hub-server not healthy after 60s:\n{log_path.read_text()[-4000:]}")
            time.sleep(0.5)
    except BaseException:
        proc.kill()
        log_file.close()
        raise

    try:
        yield client
    finally:
        proc.terminate()
        try:
            proc.wait(timeout=15)
        except subprocess.TimeoutExpired:
            proc.kill()
        log_file.close()


@pytest.fixture
def model_ref(mock_model: MockOpenAIServer, model_secret_dir: Path):
    """Build the ``model`` block of a job for a mock persona, with the credential attached."""

    def _ref(name: str, secret_dir: Path | None = None, url: str | None = None) -> dict[str, Any]:
        return {
            "url": url or mock_model.base_url,
            "name": name,
            "auth": {"secret_ref": f"file://{secret_dir or model_secret_dir}"},
        }

    return _ref


def benchmark_result(job: dict[str, Any], index: int = 0) -> dict[str, Any]:
    benchmarks = job.get("results", {}).get("benchmarks") or []
    assert len(benchmarks) > index, f"no result for benchmark {index}: {json.dumps(job, indent=1)[:2000]}"
    return benchmarks[index]
