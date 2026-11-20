#!/usr/bin/env bash
# Install the exact dependency set locked in requirements.txt from PUBLIC indexes.
#
# requirements.txt is the single source of truth for package versions, but it is
# compiled against the Red Hat AI production index (its --index-url line), which is
# reserved for product builds. Non-product images and CI use this script instead:
#
#   - torch   from the PyTorch CPU index (the locked version, CPU build)
#   - garak   from the trustyai-explainability fork at tag v<locked version>
#   - the rest from PyPI
#
# Everything is installed with --no-deps: the lock already lists every transitive
# dependency, and some production-index builds carry relaxed metadata (e.g. kfp
# accepting a newer kubernetes) that public metadata would refuse to resolve.
#
# --extras installs the packages of pyproject.toml extras (e.g. "test") on top, with
# the lock as a constraint, so a tool can add packages but never change a locked one.
# It does not install the project itself: callers run `pip install --no-deps .`,
# because a plain `pip install ".[test]"` re-resolves the project's dependencies
# against public metadata and silently changes locked versions.
#
# Usage: scripts/install-locked-deps.sh [--extras EXTRA[,EXTRA...]] [--extras-only] [requirements.txt]
#
# Environment overrides:
#   PYTHON              interpreter (default: python)
#   PIP                 pip command (default: $PYTHON -m pip)
#   PYPI_INDEX_URL      default https://pypi.org/simple
#   TORCH_INDEX_URL     default https://download.pytorch.org/whl/cpu
#   GARAK_REPO          default https://github.com/trustyai-explainability/garak.git
#   GARAK_REF           git ref for garak; default v<version locked in requirements.txt>
#   LOCK_EXTRAS         extras requirements.txt is compiled with (default: inline, as `make lock`)
#   PIP_EXTRA_ARGS      extra arguments for every pip install (e.g. --no-cache-dir)
#
# After installing the locked set, the script checks that every dependency pyproject.toml
# declares (plus LOCK_EXTRAS) is satisfied, so a lock that wasn't regenerated after a
# pyproject change fails here with "run make lock" instead of an ImportError later.
set -euo pipefail

usage() { sed -n '2,/^set -euo/p' "$0" | sed '$d; s/^# \{0,1\}//'; exit "${1:-0}"; }

REQUIREMENTS="requirements.txt"
EXTRAS=""
INSTALL_LOCKED=1
while [ $# -gt 0 ]; do
    case "$1" in
        --extras) EXTRAS="${2:?--extras needs a value}"; shift 2 ;;
        --extras-only) INSTALL_LOCKED=0; shift ;;
        -h|--help) usage ;;
        -*) echo "error: unknown option $1" >&2; usage 1 ;;
        *) REQUIREMENTS="$1"; shift ;;
    esac
done

PYTHON="${PYTHON:-python}"
PIP="${PIP:-$PYTHON -m pip}"
PYPI_INDEX_URL="${PYPI_INDEX_URL:-https://pypi.org/simple}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
GARAK_REPO="${GARAK_REPO:-https://github.com/trustyai-explainability/garak.git}"
PYPROJECT="$(dirname "$REQUIREMENTS")/pyproject.toml"
# Never empty: bash 3.2 (macOS /bin/bash) treats "${EXTRA[@]}" of an empty array as unbound under set -u.
read -r -a EXTRA <<<"--disable-pip-version-check ${PIP_EXTRA_ARGS:-}"

[ -f "$REQUIREMENTS" ] || { echo "error: $REQUIREMENTS not found" >&2; exit 1; }

pin() {  # pin <package>: the locked version, or empty
    sed -nE "s/^$1==([^[:space:]]+).*/\1/p" "$REQUIREMENTS" | head -n1
}

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

# pyproject.toml helper:
#   expand <pyproject> <extras>          print the requirements of the extras (self-references followed)
#   check  <pyproject> <extras> <skip>   verify the base dependencies + extras are installed and satisfied
cat >"$WORK/pyproject_reqs.py" <<'PY'
import re
import sys
import tomllib
from importlib.metadata import PackageNotFoundError, version

mode, path, extras = sys.argv[1], sys.argv[2], [e.strip() for e in sys.argv[3].split(",") if e.strip()]
skip = {s.strip().lower() for s in (sys.argv[4] if len(sys.argv) > 4 else "").split(",") if s.strip()}
project = tomllib.load(open(path, "rb"))["project"]
optional = project.get("optional-dependencies", {})
self_ref = re.compile(rf"^{re.escape(project['name'])}\[(.+)\]$")
seen, requirements = set(), []

def expand(extra: str) -> None:
    if extra in seen:
        return
    if extra not in optional:
        sys.exit(f"error: unknown extra '{extra}' (have: {', '.join(optional)})")
    seen.add(extra)
    for req in optional[extra]:
        if match := self_ref.match(req.replace(" ", "")):
            for nested in match.group(1).split(","):
                expand(nested.strip())
        else:
            requirements.append(req)

for extra in extras:
    expand(extra)

if mode == "expand":
    print("\n".join(requirements))
    sys.exit(0)

from packaging.requirements import Requirement  # in the lock

problems = []
for raw in project.get("dependencies", []) + requirements:
    req = Requirement(raw)
    if req.name.lower() in skip or (req.marker and not req.marker.evaluate()):
        continue
    try:
        installed = version(req.name)
    except PackageNotFoundError:
        problems.append(f"{req} (not in the lock)")
        continue
    if not req.specifier.contains(installed, prereleases=True):
        problems.append(f"{req} (lock has {installed})")
if problems:
    print("error: requirements.txt does not satisfy pyproject.toml; regenerate it with `make lock`:", file=sys.stderr)
    for problem in problems:
        print(f"  - {problem}", file=sys.stderr)
    sys.exit(1)
print(f"==> requirements.txt satisfies pyproject.toml (dependencies + [{','.join(extras)}])")
PY
# Drop pip options (the production --index-url) and the two packages sourced elsewhere.
grep -vE '^[[:space:]]*-|^(torch|garak)==' "$REQUIREMENTS" >"$WORK/public.txt"

if [ "$INSTALL_LOCKED" = 1 ]; then
    TORCH_VERSION="$(pin torch)"
    GARAK_VERSION="$(pin garak)"
    [ -n "$TORCH_VERSION" ] || { echo "error: no torch pin in $REQUIREMENTS" >&2; exit 1; }
    [ -n "$GARAK_VERSION" ] || { echo "error: no garak pin in $REQUIREMENTS" >&2; exit 1; }
    GARAK_REF="${GARAK_REF:-v${GARAK_VERSION}}"

    echo "==> torch==${TORCH_VERSION} from ${TORCH_INDEX_URL}"
    $PIP install "${EXTRA[@]}" --no-deps --index-url "$TORCH_INDEX_URL" "torch==${TORCH_VERSION}"

    echo "==> $(grep -cE '^[A-Za-z0-9]' "$WORK/public.txt") locked packages from ${PYPI_INDEX_URL}"
    $PIP install "${EXTRA[@]}" --no-deps --index-url "$PYPI_INDEX_URL" -r "$WORK/public.txt"

    echo "==> garak from ${GARAK_REPO}@${GARAK_REF} (locked: ${GARAK_VERSION})"
    $PIP install "${EXTRA[@]}" --no-deps --index-url "$PYPI_INDEX_URL" "garak @ git+${GARAK_REPO}@${GARAK_REF}"

    if [ -f "$PYPROJECT" ]; then
        # A custom GARAK_REF installs a different garak version on purpose; don't flag it.
        skip=""; [ "$GARAK_REF" = "v${GARAK_VERSION}" ] || skip="garak"
        $PYTHON "$WORK/pyproject_reqs.py" check "$PYPROJECT" "${LOCK_EXTRAS:-inline}" "$skip"
    fi
fi

if [ -n "$EXTRAS" ]; then
    [ -f "$PYPROJECT" ] || { echo "error: $PYPROJECT not found" >&2; exit 1; }
    # Expand the extras (following self-references such as "<project>[test]").
    $PYTHON "$WORK/pyproject_reqs.py" expand "$PYPROJECT" "$EXTRAS" >"$WORK/extras.txt"
    # torch stays in the constraints (keeps a CUDA build out); torch's +cpu local tag still matches.
    torch_pin="$(pin torch)"
    { cat "$WORK/public.txt"; [ -z "$torch_pin" ] || echo "torch==${torch_pin}"; } >"$WORK/constraints.txt"
    echo "==> extras [${EXTRAS}]: $(paste -sd' ' "$WORK/extras.txt") (constrained by the lock)"
    $PIP install "${EXTRA[@]}" --index-url "$PYPI_INDEX_URL" -c "$WORK/constraints.txt" -r "$WORK/extras.txt"
fi
