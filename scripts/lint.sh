#!/usr/bin/env bash
#
# Ruff over the package and the suite, and codespell over the repository:
# between them, what the lint and typos workflows run.
#
#   ./scripts/lint.sh          check, and say what is wrong
#   ./scripts/lint.sh --fix    format and apply what ruff can fix by itself
#
# Ruff sees `src/` and `tests/` only: the scripts beside this one and the
# documentation extension are read by people rather than shipped, and are not
# held to it.  codespell reads what to skip, and what is a word here rather
# than a typo, from `[tool.codespell]` in pyproject.toml -- the same place the
# workflow reads it, so the two cannot drift.
#
# Spelling is reported and never written, in either mode: codespell's
# corrections are good but not certain, and an identifier it rewrote would be
# a worse problem than the word it fixed.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PATHS=("${ROOT}/src" "${ROOT}/tests")
RUFF=(ruff)
command -v ruff >/dev/null 2>&1 || RUFF=("${PYTHON:-python3}" -m ruff)
SPELL=(codespell)
command -v codespell >/dev/null 2>&1 || SPELL=("${PYTHON:-python3}" -m codespell_lib)

spelling() {
    if ! "${SPELL[@]}" --version >/dev/null 2>&1; then
        echo "lint.sh: codespell is not installed, and the typos workflow runs it" >&2
        echo "  pip install codespell" >&2
        return 1
    fi
    # From the root, because the skips in pyproject.toml are written as the
    # `./`-prefixed paths codespell prints.
    ( cd "${ROOT}" && "${SPELL[@]}" )
}

case "${1:-}" in
    --fix)
        "${RUFF[@]}" format "${PATHS[@]}"
        "${RUFF[@]}" check --fix "${PATHS[@]}"
        spelling
        ;;
    "")
        # Format first: an unformatted file is usually also a long line, and
        # being told about the line is less useful than being told to run --fix.
        "${RUFF[@]}" format --check "${PATHS[@]}"
        "${RUFF[@]}" check "${PATHS[@]}"
        spelling
        ;;
    -h|--help)
        sed -n '3,18p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'
        exit 0
        ;;
    *)
        echo "lint.sh: unknown argument $1 (--fix, --help)" >&2
        exit 2
        ;;
esac
