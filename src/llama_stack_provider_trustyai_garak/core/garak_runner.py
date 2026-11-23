"""This module provides a sync subprocess runner that can be used by:
- eval-hub adapter (K8s Job)
- KFP components
"""

import json
import logging
import os
import signal
import subprocess
import threading
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
import re

logger = logging.getLogger(__name__)
_STDERR_WARNING_PATTERN = re.compile(
    r"(?:\berror\b|\bexception\b|\btraceback\b|\bfailed\b|\bfatal\b|\bwarn(?:ing)?\b|⚠)",
    re.IGNORECASE,
)
_SCAN_LOG_ISSUE_PATTERN = re.compile(
    r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2},\d+\s+(ERROR|WARNING)\s+(.+)$",
)


@dataclass
class GarakScanResult:
    """Result of a Garak scan execution."""

    returncode: int
    stdout: str
    stderr: str
    report_prefix: Path
    timed_out: bool = False
    log_errors: list[str] = field(default_factory=list)
    # Set when garak exited 0 but did not finish the scan (see _incomplete_scan_reason).
    incomplete_reason: str | None = None

    @property
    def success(self) -> bool:
        """Check if the scan completed successfully."""
        return self.returncode == 0 and not self.timed_out and self.incomplete_reason is None

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


# Report entries garak writes only when a run finishes: "completion" when the run ends
# and "digest" when the report is closed. An aborted run has neither.
_COMPLETION_ENTRY_TYPES = frozenset({"completion", "digest"})


def _incomplete_scan_reason(report_jsonl: Path) -> str | None:
    """Return why a scan that exited 0 is not a complete scan, or None if it is.

    garak exits 0 even when it aborts: an unreachable target, rejected credentials
    and a probe whose generator fails to load all end with exit code 0 and a missing
    or partial report. Scoring such a report yields attack_success_rate=0, i.e. a
    passing result for a scan that never ran. Require the report to show that the
    run reached its end, that at least one attempt was made, and that the target
    answered at least one of them.

    The last check catches a target that errors on every request (e.g. HTTP 500):
    garak gives up on each prompt, records the attempt with no output, and still
    finishes the run, so the report looks complete but nothing was evaluated.

    Per-prompt errors that garak recovers from (e.g. a content filter rejecting one
    prompt) do not affect these markers, so they do not fail the scan.
    """
    if not report_jsonl.exists():
        return f"garak wrote no report ({report_jsonl.name}); the scan did not start"
    attempts = 0
    answered = 0
    finished = False
    try:
        with open(report_jsonl, encoding="utf-8", errors="replace") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    entry = json.loads(line)
                    entry_type = entry.get("entry_type")
                except (json.JSONDecodeError, AttributeError):
                    continue
                if entry_type == "attempt":
                    attempts += 1
                    if _has_output(entry):
                        answered += 1
                elif entry_type in _COMPLETION_ENTRY_TYPES:
                    finished = True
    except OSError as exc:
        return f"could not read garak report {report_jsonl.name}: {exc}"
    if not finished:
        return f"garak's report has no completion record; the scan was aborted after {attempts} attempt(s)"
    if attempts == 0:
        return "garak's report contains no attempts; no probe ran against the target"
    if answered == 0:
        return (
            f"the target returned no output for any of the {attempts} attempt record(s); "
            "every request to the model failed (see scan.log for the HTTP errors)"
        )
    return None


def _has_output(attempt: dict) -> bool:
    """Whether a report attempt carries at least one response from the target.

    garak records a failed generation as a ``None`` output (0.17 writes outputs as
    message dicts, ``{"text": ...}``; older versions as plain strings).
    """
    for output in attempt.get("outputs") or []:
        if isinstance(output, dict):
            if output.get("text") is not None:
                return True
        elif output is not None:
            return True
    return False


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


def run_garak_scan(
    config_file: Path,
    timeout_seconds: int,
    report_prefix: Path,
    env: dict[str, str] | None = None,
    log_file: Path | None = None,
) -> GarakScanResult:
    """Run a Garak scan with proper process group handling.

    This function:
    1. Creates a new process group for the garak process
    2. Handles timeout with graceful shutdown (SIGTERM → SIGKILL)
    3. Kills the entire process tree on timeout
    4. Streams process output to logs in real time

    Args:
        config_file: The path to the Garak command config file
        timeout_seconds: Maximum execution time
        report_prefix: Expected report prefix path (for results)
        env: Environment variables (merged with os.environ)
        log_file: Path to write garak logs

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

    try:
        if timeout_seconds is None or timeout_seconds <= 0:
            effective_timeout = None
        else:
            effective_timeout = timeout_seconds
        process.wait(timeout=effective_timeout)
        returncode = process.returncode

        if returncode == 0:
            logger.info("Garak scan completed successfully")
        else:
            logger.error("Garak scan failed with return code %s", returncode)

    except subprocess.TimeoutExpired:
        logger.warning(f"Garak scan timed out after {timeout_seconds} seconds")
        timed_out = True

        # Kill the entire process group
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

        returncode = process.returncode if process.returncode is not None else -1
    finally:
        stdout_thread.join(timeout=5)
        stderr_thread.join(timeout=5)

    stdout = "".join(stdout_tail)
    stderr = "".join(stderr_tail)
    if timed_out and not stderr:
        stderr = "Scan timed out"

    log_errors: list[str] = []
    if log_file:
        log_errors = _extract_scan_log_issues(log_file)

    incomplete_reason = None
    if returncode == 0 and not timed_out:
        incomplete_reason = _incomplete_scan_reason(report_prefix.with_suffix(".report.jsonl"))
        if incomplete_reason:
            logger.error("Garak exited 0 but the scan is incomplete: %s", incomplete_reason)

    return GarakScanResult(
        returncode=returncode,
        stdout=stdout,
        stderr=stderr,
        report_prefix=report_prefix,
        timed_out=timed_out,
        log_errors=log_errors,
        incomplete_reason=incomplete_reason,
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
