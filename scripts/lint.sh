#!/usr/bin/env bash
#
# Ruff over the package and the suite, which is what the lint workflow runs.
#
#   ./scripts/lint.sh          check, and say what is wrong
#   ./scripts/lint.sh --fix    format and apply what ruff can fix by itself
#
# `src/` and `tests/` only: the scripts beside this one and the documentation
# extension are read by people rather than shipped, and are not held to it.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATHS=("${ROOT}/src" "${ROOT}/tests")
RUFF=(ruff)
command -v ruff >/dev/null 2>&1 || RUFF=("${PYTHON:-python3}" -m ruff)

case "${1:-}" in
    --fix)
        "${RUFF[@]}" format "${PATHS[@]}"
        "${RUFF[@]}" check --fix "${PATHS[@]}"
        ;;
    "")
        # Format first: an unformatted file is usually also a long line, and
        # being told about the line is less useful than being told to run --fix.
        "${RUFF[@]}" format --check "${PATHS[@]}"
        "${RUFF[@]}" check "${PATHS[@]}"
        ;;
    -h|--help)
        sed -n '3,9p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'
        exit 0
        ;;
    *)
        echo "lint.sh: unknown argument $1 (--fix, --help)" >&2
        exit 2
        ;;
esac
