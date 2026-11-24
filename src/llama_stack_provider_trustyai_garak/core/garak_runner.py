"""This module provides a sync subprocess runner that can be used by:
- eval-hub adapter (K8s Job)
- KFP components
"""

import logging
import os
import signal
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import re

from ..constants import DEFAULT_MAX_RETRY_SECONDS

logger = logging.getLogger(__name__)
_STDERR_WARNING_PATTERN = re.compile(
    r"(?:\berror\b|\bexception\b|\btraceback\b|\bfailed\b|\bfatal\b|\bwarn(?:ing)?\b|⚠)",
    re.IGNORECASE,
)
_SCAN_LOG_ISSUE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+(ERROR|WARNING)\s+(.+)$",
)
# garak retries a failed model request (HTTP 5xx, 429, timeouts) through the backoff library,
# which logs this line to scan.log before every wait, e.g.
#   Backing off _call_model(...) for 1.3s (garak.exception.GeneratorBackoffTrigger)
_BACKOFF_PATTERN = re.compile(r"\bBacking off \S+ for [\d.]+s\b(?: \((?P<exc>[^)]+)\))?")
# garak logs at DEBUG to scan.log, which includes the HTTP client's line for every model request:
#   httpx (openai client):  HTTP Request: POST <url> "HTTP/1.1 503 Service Unavailable"
#   urllib3 (requests, e.g. garak's REST generator):  http://host:8000 "POST /v1/x HTTP/1.1" 200 1234
_HTTP_STATUS_PATTERNS = (
    re.compile(r'HTTP Request: \S+ \S+ "HTTP/[\d.]+ (?P<code>\d{3})(?: (?P<reason>[^"]*))?"'),
    re.compile(r'"(?:GET|POST|PUT|PATCH|DELETE) \S+ HTTP/[\d.]+" (?P<code>\d{3}) '),
)

# How often the retry watchdog checks scan.log and the report.
RETRY_WATCHDOG_POLL_SECONDS = 5.0
# Fallback for logs without HTTP status lines (a client that doesn't log them): a retry streak
# also ends when no new "Backing off" line appears for this long.
RETRY_STREAK_GAP_SECONDS = 300.0


@dataclass
class GarakScanResult:
    """Result of a Garak scan execution."""

    returncode: int
    stdout: str
    stderr: str
    report_prefix: Path
    timed_out: bool = False
    log_errors: list[str] = field(default_factory=list)
    # Set when the scan was stopped because garak kept retrying failed model requests
    # for longer than the retry budget without making progress.
    retry_budget_reason: str | None = None

    @property
    def success(self) -> bool:
        """Check if the scan completed successfully."""
        return self.returncode == 0 and not self.timed_out and self.retry_budget_reason is None

    @property
    def report_jsonl(self) -> Path:
        """Path to the main report file."""
        return self.report_prefix.with_suffix(".report.jsonl")

    @property
    def avid_jsonl(self) -> Path:
        """Path to the AVID format report."""
        return self.report_prefix.with_suffix(".avid.jsonl")

    @property
    def hitlog_jsonl(self) -> Path:
        """Path to the hitlog (vulnerable attempts only)."""
        return self.report_prefix.with_suffix(".hitlog.jsonl")

    @property
    def report_html(self) -> Path:
        """Path to the HTML report."""
        return self.report_prefix.with_suffix(".report.html")

    def get_all_output_files(self) -> list[Path]:
        """Get all output files that exist."""
        candidates = [
            self.report_jsonl,
            self.avid_jsonl,
            self.hitlog_jsonl,
            self.report_html,
        ]
        return [f for f in candidates if f.exists()]


def _extract_scan_log_issues(log_file: Path) -> list[str]:
    """Extract ERROR/WARNING lines from garak's scan.log and log them.

    Garak writes its internal log to scan.log via GARAK_LOG_FILE but never
    outputs these to stdout/stderr. This function surfaces them so they
    appear in CLI and KFP step logs.
    """
    if not log_file.exists():
        return []
    issues: list[str] = []
    try:
        with open(log_file, encoding="utf-8", errors="replace") as f:
            for line in f:
                m = _SCAN_LOG_ISSUE_PATTERN.match(line.rstrip())
                if m:
                    level_str, message = m.group(1), m.group(2)
                    log_level = logging.ERROR if level_str == "ERROR" else logging.WARNING
                    logger.log(log_level, "garak[scan.log] %s", message)
                    issues.append(line.rstrip())
    except OSError as exc:
        logger.warning("Could not read scan.log: %s", exc)
    return issues


class _RetryWatchdog:
    """Detect garak retrying failed model requests with no scan progress.

    garak retries 5xx, 429 and timeout errors with no limit on attempts or total time, so a
    target that stays down keeps a scan running indefinitely. A wall-clock timeout can't tell
    that apart from a long scan against a slow model, so this watches for the failure itself:

    - a retry streak starts at the first "Backing off" line in scan.log;
    - it ends on any sign of progress: a successful (2xx) model response in scan.log, or the
      report growing (an attempt or evaluation was written);
    - the budget is exceeded when a streak has lasted ``budget_seconds``.

    Time between retries doesn't end a streak: a request that times out or hits a slow gateway
    error (e.g. a 504 after minutes) can take far longer than the backoff wait to fail. Only if
    scan.log has no HTTP status lines at all, so a success would be invisible, does a quiet gap
    of ``gap_seconds`` without a new retry end the streak instead.

    A successful response counts because the report alone is not a progress signal: the
    earlystop harness (intents) writes nothing until a whole probe finishes, which can take
    hours while the models answer normally. A slow but healthy model never triggers it (it
    causes no retries), and neither does rate limiting that lets some requests through.
    """

    def __init__(
        self,
        log_file: Path,
        report_file: Path,
        budget_seconds: float,
        gap_seconds: float = RETRY_STREAK_GAP_SECONDS,
        clock=time.monotonic,
    ) -> None:
        self.log_file = log_file
        self.report_file = report_file
        self.budget_seconds = budget_seconds
        self.gap_seconds = gap_seconds
        self._clock = clock
        self._log_pos = 0
        self._partial = ""
        self._report_size = 0
        self.streak_start: float | None = None
        self.last_backoff: float | None = None
        self.retries = 0
        self.last_http_error: str | None = None
        self.last_exception: str | None = None
        self.saw_http_status = False

    def _reset(self) -> None:
        self.streak_start = None
        self.last_backoff = None
        self.retries = 0
        self.last_http_error = None
        self.last_exception = None

    def _new_log_lines(self) -> list[str]:
        try:
            size = self.log_file.stat().st_size
        except OSError:
            return []
        if size < self._log_pos:  # truncated or replaced
            self._log_pos, self._partial = 0, ""
        if size == self._log_pos:
            return []
        try:
            with open(self.log_file, encoding="utf-8", errors="replace") as f:
                f.seek(self._log_pos)
                chunk = f.read()
                self._log_pos = f.tell()
        except OSError:
            return []
        lines = (self._partial + chunk).split("\n")
        self._partial = lines.pop()  # keep an unfinished last line for the next poll
        return lines

    def _report_grew(self) -> bool:
        try:
            size = self.report_file.stat().st_size
        except OSError:
            return False
        grew = size > self._report_size
        self._report_size = size
        return grew

    def poll(self) -> bool:
        """Read what garak wrote since the last poll; True when the retry budget is exceeded."""
        now = self._clock()
        if self._report_grew():
            self._reset()
        for line in self._new_log_lines():
            status = next((m for m in (p.search(line) for p in _HTTP_STATUS_PATTERNS) if m), None)
            if status:
                self.saw_http_status = True
                code = int(status.group("code"))
                if 200 <= code < 300:  # the model answered: not stuck
                    self._reset()
                    continue
                if code >= 400:
                    reason = status.groupdict().get("reason") or ""
                    self.last_http_error = f"HTTP {code} {reason}".strip()
            backoff = _BACKOFF_PATTERN.search(line)
            if backoff:
                if self._gap_expired(now):
                    self._reset()
                if self.streak_start is None:
                    self.streak_start = now
                self.last_backoff = now
                self.retries += 1
                if backoff.group("exc"):
                    self.last_exception = backoff.group("exc").rsplit(".", 1)[-1]
        if self.streak_start is None:
            return False
        if self._gap_expired(now):
            self._reset()
            return False
        return now - self.streak_start >= self.budget_seconds

    def _gap_expired(self, now: float) -> bool:
        return not self.saw_http_status and self.last_backoff is not None and now - self.last_backoff > self.gap_seconds

    def reason(self) -> str:
        minutes = self.budget_seconds / 60
        duration = f"{minutes:g} min" if minutes >= 1 else f"{self.budget_seconds:g} s"
        # GeneratorBackoffTrigger is garak's generic wrapper; the HTTP status says more.
        error = self.last_http_error or (
            self.last_exception if self.last_exception != "GeneratorBackoffTrigger" else None
        )
        last = f"; last error: {error}" if error else ""
        return (
            f"the model kept failing for {duration} with no scan progress "
            f"({self.retries} retries{last}). Check that the model endpoint is up, or raise "
            "max_retry_seconds (0 disables this check)."
        )


def _terminate_process_group(process: subprocess.Popen) -> None:
    """Stop garak and everything it started: SIGTERM, then SIGKILL after 5 s."""
    try:
        pgid = os.getpgid(process.pid)
        logger.info(f"Sending SIGTERM to process group {pgid}")
        os.killpg(pgid, signal.SIGTERM)

        try:
            process.wait(timeout=5)
            logger.info("Process terminated gracefully")
        except subprocess.TimeoutExpired:
            logger.warning("Process didn't terminate, sending SIGKILL")
            os.killpg(pgid, signal.SIGKILL)
            process.wait()

    except ProcessLookupError:
        logger.debug("Process already terminated")
    except Exception as e:
        logger.error(f"Error killing process group: {e}")
        # Try to kill just the main process
        try:
            process.kill()
            process.wait()
        except Exception:
            pass


def run_garak_scan(
    config_file: Path,
    timeout_seconds: int,
    report_prefix: Path,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
    max_retry_seconds: int | None = DEFAULT_MAX_RETRY_SECONDS,
) -> GarakScanResult:
    """Run a Garak scan with proper process group handling.

    This function:
    1. Creates a new process group for the garak process
    2. Handles timeout with graceful shutdown (SIGTERM → SIGKILL)
    3. Kills the entire process tree on timeout
    4. Streams process output to logs in real time
    5. Stops the scan when garak keeps retrying failed model requests for longer than
       ``max_retry_seconds`` with no progress (see ``_RetryWatchdog``)

    Args:
        config_file: The path to the Garak command config file
        timeout_seconds: Maximum execution time
        report_prefix: Expected report prefix path (for results)
        env: Environment variables (merged with os.environ)
        log_file: Path to write garak logs
        max_retry_seconds: Retry budget. None or 0 disables it; it also needs ``log_file``.

    Returns:
        GarakScanResult with execution details

    Raises:
        FileNotFoundError: If garak is not installed
        PermissionError: If garak cannot be executed
    """
    # Merge environment
    run_env = os.environ.copy()
    if env:
        run_env.update(env)
    # Improves real-time output behavior for Python subprocesses.
    run_env.setdefault("PYTHONUNBUFFERED", "1")

    # Set log file if provided
    if log_file:
        run_env["GARAK_LOG_FILE"] = str(log_file)

    timed_out = False
    output_tail_lines = int(run_env.get("GARAK_OUTPUT_TAIL_LINES", "500"))
    stdout_tail: deque[str] = deque(maxlen=output_tail_lines)
    stderr_tail: deque[str] = deque(maxlen=output_tail_lines)
    stdout = ""
    stderr = ""
    returncode = -1

    if not config_file.exists():
        raise FileNotFoundError(f"Garak command config file not found: {config_file}")

    cmd = ["garak", "--config", str(config_file)]

    # Start process in new process group
    # argument list with shell=False prevents shell injection
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
        env=run_env,
        preexec_fn=os.setsid,
        shell=False,
    )

    def _stream_pipe(
        pipe: subprocess.PIPE,
        tail: deque[str],
        level: int,
        stream_name: str,
    ) -> None:
        """Read pipe output line-by-line and stream to logger."""
        if pipe is None:
            return
        try:
            for line in iter(pipe.readline, ""):
                tail.append(line)
                text = line.rstrip()
                if text:
                    # Garak/tqdm writes progress bars to stderr; avoid classifying
                    # these as warnings unless the line looks like an actual issue.
                    is_stderr_issue = bool(stream_name == "stderr" and _STDERR_WARNING_PATTERN.search(text))
                    line_level = logging.WARNING if is_stderr_issue else level
                    stream_label = (
                        "stderr" if is_stderr_issue else ("progress" if stream_name == "stderr" else stream_name)
                    )
                    logger.log(line_level, "garak[%s] %s", stream_label, text)
        finally:
            try:
                pipe.close()
            except Exception:
                pass

    stdout_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stdout, stdout_tail, logging.INFO, "stdout"),
        daemon=True,
    )
    stderr_thread = threading.Thread(
        target=_stream_pipe,
        args=(process.stderr, stderr_tail, logging.INFO, "stderr"),
        daemon=True,
    )
    stdout_thread.start()
    stderr_thread.start()

    watchdog = (
        _RetryWatchdog(log_file, report_prefix.with_suffix(".report.jsonl"), max_retry_seconds)
        if log_file and max_retry_seconds and max_retry_seconds > 0
        else None
    )
    retry_budget_reason: str | None = None
    deadline = time.monotonic() + timeout_seconds if timeout_seconds and timeout_seconds > 0 else None

    try:
        while True:
            wait = None
            if watchdog:
                wait = RETRY_WATCHDOG_POLL_SECONDS
            if deadline is not None:
                remaining = max(0.0, deadline - time.monotonic())
                wait = remaining if wait is None else min(wait, remaining)
            try:
                process.wait(timeout=wait)
                break
            except subprocess.TimeoutExpired:
                if deadline is not None and time.monotonic() >= deadline:
                    logger.warning(f"Garak scan timed out after {timeout_seconds} seconds")
                    timed_out = True
                    break
                if watchdog and watchdog.poll():
                    retry_budget_reason = watchdog.reason()
                    logger.error("Stopping garak: %s", retry_budget_reason)
                    break

        if timed_out or retry_budget_reason:
            _terminate_process_group(process)
            returncode = process.returncode if process.returncode is not None else -1
        else:
            returncode = process.returncode
            if returncode == 0:
                logger.info("Garak scan completed successfully")
            else:
                logger.error("Garak scan failed with return code %s", returncode)
    finally:
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

    stdout = "".join(stdout_tail)
    stderr = "".join(stderr_tail)
    if timed_out and not stderr:
        stderr = "Scan timed out"
    if retry_budget_reason and not stderr:
        stderr = retry_budget_reason

    log_errors: list[str] = []
    if log_file:
        log_errors = _extract_scan_log_issues(log_file)

    return GarakScanResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        report_prefix=report_prefix,
        timed_out=timed_out,
        log_errors=log_errors,
        retry_budget_reason=retry_budget_reason,
    )


def convert_to_avid_report(report_path: Path) -> bool:
    """Convert a Garak report to AVID format.

    Args:
        report_path: Path to the .report.jsonl file

    Returns:
        True if conversion succeeded, False otherwise
    """
    if not report_path.exists():
        logger.error(f"Report file not found: {report_path}")
        return False

    try:
        from garak.report import Report

        report = Report(str(report_path)).load().get_evaluations()
        report.export()  # Creates .avid.jsonl file at same location as report_path
        logger.info(f"Converted report to AVID format: {report_path}")
        return True

    except ImportError:
        logger.error("AVID report module not found, cannot convert to AVID format")
        return False
    except Exception as e:
        logger.error(f"Failed to convert report to AVID format: {e}")
        return False
