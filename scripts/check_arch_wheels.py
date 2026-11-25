#!/usr/bin/env python3
"""Check that the lock's index has a wheel of every pin each target architecture installs.

The downstream image is built for several architectures from the AIPCC index that
requirements.txt names, but pull-request builds cover x86_64 only, so a wheel missing for
ppc64le or s390x is otherwise found when the release build fails.

The lock is resolved for x86_64 and carries no environment markers, so it lists packages
other architectures never install (triton is not a torch dependency on s390x, for example).
The check therefore re-resolves the lock with uv, constrained to the locked versions and
keeping the markers, and checks each pin only on the architectures its marker selects. It
asks the index directly (PEP 691 JSON).

    python scripts/check_arch_wheels.py [requirements.txt] [--arch s390x ...] [--python 3.12]

Needs `packaging`, and `uv` on PATH unless --no-resolve is given (the file is then checked
as it is, markers included). Exit status: 0 every needed wheel exists, 1 a wheel is missing,
2 the check could not be completed (index unreadable, resolution failed or differs from the
lock).
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from packaging.markers import Marker
from packaging.tags import Tag, compatible_tags, cpython_tags
from packaging.utils import InvalidWheelFilename, canonicalize_name, parse_wheel_filename
from packaging.version import InvalidVersion, Version

# What the downstream Konflux pipelines build (build-platforms in .tekton/).
DEFAULT_ARCHES = ("x86_64", "ppc64le", "s390x")
DEFAULT_PYTHON = "3.12"
# glibc of the RHEL 9 base images: manylinux_2_34 is the newest tag they can install.
GLIBC_MINOR = 34
# The extra the images install (`pip install ".[inline]"`), as `make lock` resolves it.
LOCK_EXTRA = "inline"

PIN = re.compile(
    r"^(?P<name>[A-Za-z0-9][A-Za-z0-9._-]*)(?:\[[^\]]*\])?==(?P<version>[^\s;#\\]+)\s*(?:;\s*(?P<marker>[^#]+))?"
)
INDEX_URL = re.compile(r"^(?:--index-url|-i)[=\s]+(?P<url>\S+)")

JSON_ACCEPT = "application/vnd.pypi.simple.v1+json"


@dataclass(frozen=True)
class Pin:
    name: str
    version: str
    line: int
    marker: str | None = None

    @property
    def key(self) -> tuple[str, str]:
        return canonicalize_name(self.name), self.version


@dataclass
class Result:
    pin: Pin
    needed_on: list[str] = field(default_factory=list)
    missing: list[str] = field(default_factory=list)
    error: str | None = None


def parse_requirements(text: str) -> tuple[str | None, list[Pin], list[tuple[int, str]]]:
    """The index URL, the `name==version` pins (with their markers), and any line that is not a pin."""
    index = None
    pins: list[Pin] = []
    unpinned: list[tuple[int, str]] = []
    for number, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("-"):
            match = INDEX_URL.match(line)
            if match:
                index = match.group("url")
            continue
        match = PIN.match(line)
        if match:
            marker = (match.group("marker") or "").strip() or None
            pins.append(Pin(match.group("name"), match.group("version"), number, marker))
        else:
            unpinned.append((number, line))
    return index, pins, unpinned


def marker_environment(arch: str, python: str) -> dict[str, str]:
    """PEP 508 marker values of a RHEL 9 CPython of this architecture."""
    return {
        "implementation_name": "cpython",
        "implementation_version": f"{python}.0",
        "os_name": "posix",
        "platform_machine": arch,
        "platform_python_implementation": "CPython",
        "platform_release": "",
        "platform_system": "Linux",
        "platform_version": "",
        "python_full_version": f"{python}.0",
        "python_version": python,
        "sys_platform": "linux",
    }


def needed_on(pin: Pin, arches: Iterable[str], python: str) -> list[str]:
    if not pin.marker:
        return list(arches)
    marker = Marker(pin.marker)
    return [arch for arch in arches if marker.evaluate(marker_environment(arch, python))]


def platform_tags(arch: str) -> list[str]:
    """Linux platform tags a RHEL 9 machine of this architecture can install."""
    tags = [f"manylinux_2_{minor}_{arch}" for minor in range(GLIBC_MINOR, 4, -1)]
    tags += [f"manylinux2014_{arch}", f"manylinux2010_{arch}", f"manylinux1_{arch}", f"linux_{arch}"]
    return tags


def supported_tags(arch: str, python: str) -> set[Tag]:
    version = tuple(int(part) for part in python.split("."))
    platforms = platform_tags(arch)
    interpreter = f"cp{version[0]}{version[1]}"
    return set(cpython_tags(version, platforms=platforms)) | set(
        compatible_tags(version, interpreter=interpreter, platforms=platforms)
    )


def fetch_files(index: str, name: str, attempts: int = 3) -> list[str]:
    """Filenames the index lists for a project, retrying transient failures."""
    url = f"{index.rstrip('/')}/{canonicalize_name(name)}/"
    request = urllib.request.Request(url, headers={"Accept": JSON_ACCEPT})
    for attempt in range(1, attempts + 1):
        try:
            with urllib.request.urlopen(request, timeout=60) as response:  # noqa: S310 - https index URL from the lock
                return [entry["filename"] for entry in json.load(response).get("files", [])]
        except urllib.error.HTTPError as exc:
            if exc.code == 404 or attempt == attempts:
                raise
        except (urllib.error.URLError, TimeoutError):
            if attempt == attempts:
                raise
        time.sleep(2 * attempt)
    raise AssertionError("unreachable")


def same_version(candidate: Version, pinned: str) -> bool:
    try:
        return candidate == Version(pinned)
    except InvalidVersion:
        return str(candidate) == pinned


def missing_arches(filenames: Iterable[str], pin: Pin, wanted: dict[str, set[Tag]]) -> list[str]:
    """The architectures with no wheel of exactly this version that they can install."""
    covered: set[str] = set()
    for filename in filenames:
        if not filename.endswith(".whl"):
            continue
        try:
            _, version, _, tags = parse_wheel_filename(filename)
        except InvalidWheelFilename:
            continue
        if not same_version(version, pin.version):
            continue
        covered.update(arch for arch, supported in wanted.items() if tags & supported)
    return [arch for arch in wanted if arch not in covered]


def check(
    index: str,
    pins: list[Pin],
    arches: Iterable[str],
    python: str,
    fetch: Callable[[str, str], list[str]] | None = None,
    workers: int = 16,
) -> list[Result]:
    fetch = fetch or fetch_files
    arches = list(arches)
    tags = {arch: supported_tags(arch, python) for arch in arches}

    def one(pin: Pin) -> Result:
        arches_for_pin = needed_on(pin, arches, python)
        result = Result(pin, needed_on=arches_for_pin)
        if not arches_for_pin:
            return result
        try:
            filenames = fetch(index, pin.name)
        except urllib.error.HTTPError as exc:
            if exc.code == 404:
                result.missing, result.error = arches_for_pin, "not on the index"
            else:
                result.error = f"index answered HTTP {exc.code}"
            return result
        except Exception as exc:  # noqa: BLE001 - reported per pin, the run goes on
            result.error = f"could not read the index: {exc}"
            return result
        result.missing = missing_arches(filenames, pin, {arch: tags[arch] for arch in arches_for_pin})
        return result

    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(one, pins))


def resolve_with_markers(lock: Path, index: str, python: str, uv: str = "uv") -> str:
    """The lock re-resolved as `make lock` does, pinned to its versions, with markers kept."""
    project = lock.parent
    command = [
        uv, "pip", "compile", "pyproject.toml",
        "--extra", LOCK_EXTRA,
        "--python-platform", "linux",
        "--python-version", python,
        "--index-strategy", "first-index",
        "--index-url", index,
        "--constraint", str(lock.resolve()),
        "--no-strip-markers", "--no-header", "--no-annotate", "--quiet",
    ]  # fmt: skip
    if (project / "constraints.txt").exists():
        command += ["--constraint", "constraints.txt"]
    with tempfile.TemporaryDirectory() as workdir:
        output = Path(workdir) / "requirements-markers.txt"
        subprocess.run([*command, "-o", str(output)], cwd=project, check=True)
        return output.read_text(encoding="utf-8")


def annotate(level: str, path: str, line: int, message: str) -> None:
    if os.environ.get("GITHUB_ACTIONS") == "true":
        print(f"::{level} file={path},line={line}::{message}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("requirements", nargs="?", default="requirements.txt")
    parser.add_argument(
        "--arch", action="append", dest="arches", help=f"repeatable; default {' '.join(DEFAULT_ARCHES)}"
    )
    parser.add_argument("--python", default=DEFAULT_PYTHON, help=f"target Python, default {DEFAULT_PYTHON}")
    parser.add_argument("--index", help="index to check instead of the --index-url in the file")
    parser.add_argument(
        "--no-resolve", action="store_true", help="check the file as it is instead of re-resolving it for markers"
    )
    args = parser.parse_args(argv)
    arches = args.arches or list(DEFAULT_ARCHES)
    lock = Path(args.requirements)

    index, pins, unpinned = parse_requirements(lock.read_text(encoding="utf-8"))
    index = args.index or index
    if not index:
        print(f"{lock} names no --index-url; pass --index", file=sys.stderr)
        return 2
    for number, line in unpinned:
        annotate("warning", str(lock), number, f"not an exact pin, not checked: {line}")
        print(f"warning: line {number} is not an exact pin, not checked: {line}")

    if not args.no_resolve:
        try:
            _, resolved, _ = parse_requirements(resolve_with_markers(lock, index, args.python))
        except (OSError, subprocess.CalledProcessError) as exc:
            print(f"could not re-resolve {lock} for its markers: {exc}", file=sys.stderr)
            return 2
        lines = {pin.key: pin.line for pin in pins}
        if set(lines) != {pin.key for pin in resolved}:
            extra = sorted({pin.key for pin in resolved} - set(lines))
            gone = sorted(set(lines) - {pin.key for pin in resolved})
            print(f"re-resolving {lock} gave different pins (lock out of date?): new {extra}, dropped {gone}")
            return 2
        # Markers come from the resolution; line numbers stay those of the lock.
        pins = [Pin(pin.name, pin.version, lines[pin.key], pin.marker) for pin in resolved]

    results = check(index, pins, arches, args.python)
    missing = [result for result in results if result.missing]
    unreadable = [result for result in results if result.error and not result.missing]
    not_needed = {arch: sorted(r.pin.name for r in results if arch not in r.needed_on) for arch in arches}

    print(f"{len(pins)} pins in {lock}, checked for {', '.join(arches)} (Python {args.python})")
    print(f"index: {index}")
    for arch, names in not_needed.items():
        if names:
            print(f"  not installed on {arch} (environment marker): {', '.join(names)}")
    for result in sorted(missing, key=lambda r: r.pin.name.lower()):
        reason = result.error or f"no installable wheel for {', '.join(result.missing)}"
        message = f"{result.pin.name}=={result.pin.version}: {reason}"
        print(f"  MISSING {message}")
        annotate("error", str(lock), result.pin.line, message)
    for result in sorted(unreadable, key=lambda r: r.pin.name.lower()):
        message = f"{result.pin.name}=={result.pin.version}: {result.error}"
        print(f"  UNKNOWN {message}")
        annotate("error", str(lock), result.pin.line, message)

    if missing:
        print(f"{len(missing)} pin(s) cannot be installed from this index on every architecture that needs them.")
        return 1
    if unreadable:
        print(f"{len(unreadable)} pin(s) could not be checked.")
        return 2
    print("Every architecture has a wheel of every pin it installs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
