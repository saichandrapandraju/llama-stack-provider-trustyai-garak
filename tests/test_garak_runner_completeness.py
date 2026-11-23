"""run_garak_scan must not report success for a scan garak did not finish.

garak exits 0 when it aborts (unreachable target, rejected credentials, a probe
whose generator fails to load). These tests run the real runner against a fake
``garak`` executable that reproduces the report garak leaves behind in each case.
"""

import json
import os
import stat
import sys
from pathlib import Path

import pytest

from llama_stack_provider_trustyai_garak.core.garak_runner import run_garak_scan

SETUP = {"entry_type": "start_run setup"}
INIT = {"entry_type": "init"}
ATTEMPT = {"entry_type": "attempt", "status": 2, "outputs": [{"text": "I can't help with that.", "lang": "en"}]}
# What garak 0.17 records when the target errors on a request (e.g. HTTP 500): no output.
FAILED_ATTEMPT = {"entry_type": "attempt", "status": 2, "outputs": [None]}
COMPLETION = {"entry_type": "completion"}
DIGEST = {"entry_type": "digest"}

FAKE_GARAK = f"""#!{sys.executable}
import json, os, sys
entries = json.loads(os.environ["FAKE_GARAK_ENTRIES"])
if entries is not None:
    with open(os.environ["FAKE_GARAK_REPORT"], "w") as f:
        for entry in entries:
            f.write(json.dumps(entry) + "\\n")
log = os.environ.get("GARAK_LOG_FILE")
if log and os.environ.get("FAKE_GARAK_LOG"):
    with open(log, "w") as f:
        f.write(os.environ["FAKE_GARAK_LOG"])
sys.exit(int(os.environ.get("FAKE_GARAK_RC", "0")))
"""


@pytest.fixture
def fake_garak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "garak"
    exe.write_text(FAKE_GARAK)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")

    def _run(entries: list[dict] | None, rc: int = 0, log: str = ""):
        report_prefix = tmp_path / "scan"
        config = tmp_path / "config.json"
        config.write_text("{}")
        env = {
            "FAKE_GARAK_ENTRIES": json.dumps(entries),
            "FAKE_GARAK_REPORT": str(report_prefix.with_suffix(".report.jsonl")),
            "FAKE_GARAK_RC": str(rc),
            "FAKE_GARAK_LOG": log,
        }
        return run_garak_scan(
            config_file=config,
            timeout_seconds=60,
            report_prefix=report_prefix,
            env=env,
            log_file=tmp_path / "scan.log",
        )

    return _run


def test_finished_scan_succeeds(fake_garak):
    result = fake_garak([SETUP, INIT, ATTEMPT, ATTEMPT, COMPLETION, DIGEST])
    assert result.success
    assert result.incomplete_reason is None


def test_digest_alone_marks_a_finished_scan(fake_garak):
    # The trimmed garak 0.17 fixtures in tests/_resources carry a digest but no completion entry.
    assert fake_garak([SETUP, ATTEMPT, DIGEST]).success


def test_missing_report_fails(fake_garak):
    """Unreachable target: garak exits 0 before writing a report."""
    result = fake_garak(None)
    assert not result.success
    assert "no report" in result.incomplete_reason


def test_report_aborted_before_any_attempt_fails(fake_garak):
    """A probe's generator failed to load: setup entries only."""
    result = fake_garak([SETUP, INIT, {"entry_type": "plugin_cache"}])
    assert not result.success
    assert "no completion record" in result.incomplete_reason
    assert "after 0 attempt(s)" in result.incomplete_reason


def test_report_truncated_mid_scan_fails(fake_garak):
    """Some probes ran, then garak aborted: attempts but no completion record."""
    result = fake_garak([SETUP, ATTEMPT, ATTEMPT])
    assert not result.success
    assert "after 2 attempt(s)" in result.incomplete_reason


def test_finished_scan_without_attempts_fails(fake_garak):
    result = fake_garak([SETUP, COMPLETION, DIGEST])
    assert not result.success
    assert "no attempts" in result.incomplete_reason


def test_target_failing_every_request_fails(fake_garak):
    """HTTP 500 on every request: garak gives up per prompt but still finishes the run."""
    result = fake_garak([SETUP, FAILED_ATTEMPT, FAILED_ATTEMPT, COMPLETION, DIGEST])
    assert not result.success
    assert "no output for any of the 2 attempt record(s)" in result.incomplete_reason


@pytest.mark.parametrize(
    "outputs",
    [[None, {"text": "", "lang": "en"}], ["plain string output (older garak)"]],
    ids=["one-of-several-generations", "string-output"],
)
def test_any_answered_attempt_counts(fake_garak, outputs):
    attempt = {"entry_type": "attempt", "status": 2, "outputs": outputs}
    assert fake_garak([SETUP, attempt, COMPLETION, DIGEST]).success


def test_some_failed_requests_do_not_fail_the_scan(fake_garak):
    """Errors on some prompts are recoverable; the answered ones are still scored."""
    result = fake_garak([SETUP, FAILED_ATTEMPT, ATTEMPT, COMPLETION, DIGEST])
    assert result.success


@pytest.mark.parametrize("outputs", [[None], [{"text": None}], [], None], ids=["none", "null-text", "empty", "missing"])
def test_attempts_without_output_do_not_count(fake_garak, outputs):
    attempt = {"entry_type": "attempt", "status": 1, "outputs": outputs}
    assert not fake_garak([SETUP, attempt, COMPLETION, DIGEST]).success


def test_recoverable_errors_do_not_fail_a_finished_scan(fake_garak):
    """garak logs ERROR for e.g. one prompt rejected by a content filter, then carries on."""
    log = "2026-11-19 10:00:00,000  ERROR  Bad request: 'some prompt'\n"
    result = fake_garak([SETUP, ATTEMPT, COMPLETION, DIGEST], log=log)
    assert result.success
    assert result.log_errors, "the error is still surfaced"


def test_nonzero_exit_is_a_failure_without_completeness_check(fake_garak):
    result = fake_garak(None, rc=3)
    assert not result.success
    assert result.returncode == 3
    assert result.incomplete_reason is None
