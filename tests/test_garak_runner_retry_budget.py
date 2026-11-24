"""The retry budget stops a scan that garak keeps retrying against a failing model.

garak retries HTTP 5xx, 429 and timeout errors with no limit, so a target that stays down
keeps a scan running indefinitely. These tests cover the watchdog that detects it (with a
fake clock) and run_garak_scan end to end against a fake ``garak`` executable.
"""

import os
import stat
import sys
import time
from pathlib import Path

import pytest

from llama_stack_provider_trustyai_garak.core import garak_runner
from llama_stack_provider_trustyai_garak.core.garak_runner import _RetryWatchdog, run_garak_scan

# What garak writes to scan.log while a request keeps failing (taken from a real 503 run).
HTTP_503 = (
    "2026-11-23 21:06:15,001  INFO  HTTP Request: POST http://127.0.0.1:8500/v1/chat/completions "
    '"HTTP/1.1 503 Service Unavailable"\n'
)
BACKOFF = (
    "2026-11-23 21:06:16,383  INFO  Backing off _call_model(...) for 1.3s (garak.exception.GeneratorBackoffTrigger)\n"
)
HTTP_200 = (
    '2026-11-23 21:06:17,001  INFO  HTTP Request: POST http://127.0.0.1:8500/v1/chat/completions "HTTP/1.1 200 OK"\n'
)


class FakeClock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


@pytest.fixture
def watch(tmp_path: Path):
    log, report = tmp_path / "scan.log", tmp_path / "scan.report.jsonl"
    log.write_text("")
    report.write_text('{"entry_type": "init"}\n')
    clock = FakeClock()
    watchdog = _RetryWatchdog(log, report, budget_seconds=600, gap_seconds=300, clock=clock)
    assert not watchdog.poll()  # baseline: existing report content is not progress

    def step(seconds: float, log_text: str = "", report_text: str = "") -> bool:
        clock.now += seconds
        with open(log, "a") as f:
            f.write(log_text)
        if report_text:
            with open(report, "a") as f:
                f.write(report_text)
        return watchdog.poll()

    return watchdog, step


def test_retries_without_progress_exceed_the_budget(watch):
    watchdog, step = watch
    exceeded = [step(60, HTTP_503 + BACKOFF) for _ in range(11)]  # one retry a minute
    assert exceeded[:10] == [False] * 10
    assert exceeded[10], "600 s of uninterrupted retries must exceed the 600 s budget"
    reason = watchdog.reason()
    assert "kept failing for 10 min" in reason
    assert "11 retries" in reason
    assert "HTTP 503 Service Unavailable" in reason
    assert "max_retry_seconds" in reason


def test_report_progress_resets_the_streak(watch):
    """Retries that eventually succeed write attempts to the report, so the scan is progressing."""
    _, step = watch
    for i in range(30):
        assert not step(60, HTTP_503 + BACKOFF, report_text='{"entry_type": "attempt"}\n' if i % 5 == 4 else "")


def test_a_quiet_gap_ends_the_streak(watch):
    """One transient retry followed by a long, healthy stretch (e.g. a slow probe) never trips it."""
    watchdog, step = watch
    assert not step(1, BACKOFF)
    for _ in range(20):
        assert not step(60, HTTP_200)
    assert watchdog.streak_start is None


def test_streak_restarts_after_a_gap(watch):
    _, step = watch
    assert not step(1, BACKOFF)
    assert not step(400, BACKOFF)  # 400 s since the last retry (> 300 s gap): a new streak starts
    assert not step(500, BACKOFF)  # again > 300 s: the streak starts over here
    for _ in range(9):
        assert not step(60, BACKOFF)  # 540 s into the streak
    assert step(60, BACKOFF)  # 600 s of retries, each within 300 s of the last


def test_no_retries_never_trip_it(watch):
    _, step = watch
    for _ in range(100):
        assert not step(60, HTTP_200)


def test_a_line_split_across_polls_is_read_once(watch):
    watchdog, step = watch
    step(1, BACKOFF[:20])
    assert watchdog.retries == 0
    step(1, BACKOFF[20:])
    assert watchdog.retries == 1


def test_missing_files_are_not_an_error(tmp_path):
    watchdog = _RetryWatchdog(tmp_path / "none.log", tmp_path / "none.jsonl", budget_seconds=1)
    assert not watchdog.poll()


# --------------------------------------------------------------------------- run_garak_scan

FAKE_GARAK = f"""#!{sys.executable}
import os, sys, time
log, report = os.environ["GARAK_LOG_FILE"], os.environ["FAKE_GARAK_REPORT"]
mode, seconds = os.environ["FAKE_GARAK_MODE"], float(os.environ["FAKE_GARAK_SECONDS"])
end = time.monotonic() + seconds
while time.monotonic() < end:
    with open(log, "a") as f:
        f.write({HTTP_503!r} + {BACKOFF!r})
    if mode == "progress":
        with open(report, "a") as f:
            f.write('{{"entry_type": "attempt"}}\\n')
    time.sleep(0.05)
sys.exit(0)
"""


@pytest.fixture
def fake_garak(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    exe = bin_dir / "garak"
    exe.write_text(FAKE_GARAK)
    exe.chmod(exe.stat().st_mode | stat.S_IEXEC)
    monkeypatch.setenv("PATH", f"{bin_dir}{os.pathsep}{os.environ['PATH']}")
    monkeypatch.setattr(garak_runner, "RETRY_WATCHDOG_POLL_SECONDS", 0.1)

    def _run(mode: str, seconds: float, max_retry_seconds: int | None, timeout_seconds: int = 0):
        report_prefix = tmp_path / "scan"
        config = tmp_path / "config.json"
        config.write_text("{}")
        started = time.monotonic()
        result = run_garak_scan(
            config_file=config,
            timeout_seconds=timeout_seconds,
            report_prefix=report_prefix,
            env={
                "FAKE_GARAK_REPORT": str(report_prefix.with_suffix(".report.jsonl")),
                "FAKE_GARAK_MODE": mode,
                "FAKE_GARAK_SECONDS": str(seconds),
            },
            log_file=tmp_path / "scan.log",
            max_retry_seconds=max_retry_seconds,
        )
        return result, time.monotonic() - started

    return _run


def test_scan_stuck_retrying_is_stopped(fake_garak):
    result, elapsed = fake_garak("stuck", seconds=60, max_retry_seconds=1)
    assert elapsed < 15, f"the watchdog should stop garak after ~1 s, took {elapsed:.1f} s"
    assert not result.success
    assert not result.timed_out
    assert "kept failing for 1 s" in result.retry_budget_reason
    assert "HTTP 503" in result.retry_budget_reason
    assert result.stderr == result.retry_budget_reason


def test_retries_with_progress_run_to_completion(fake_garak):
    result, _ = fake_garak("progress", seconds=2, max_retry_seconds=1)
    assert result.success
    assert result.retry_budget_reason is None


@pytest.mark.parametrize("max_retry_seconds", [0, None])
def test_budget_disabled(fake_garak, max_retry_seconds):
    result, _ = fake_garak("stuck", seconds=2, max_retry_seconds=max_retry_seconds)
    assert result.success
    assert result.retry_budget_reason is None


def test_timeout_still_applies_with_the_watchdog(fake_garak):
    result, elapsed = fake_garak("progress", seconds=60, max_retry_seconds=1, timeout_seconds=1)
    assert result.timed_out
    assert result.retry_budget_reason is None
    assert elapsed < 15
