#!/usr/bin/env bash
#
# Build the reference locally, the way the docs workflow builds it.
#
#   ./scripts/build_docs.sh                  render, warnings are errors
#   ./scripts/build_docs.sh --execute        run the gallery examples as well
#   ./scripts/build_docs.sh --clean --serve  start over, then serve the result
#
# Rendering needs neither torch nor the compiled library: `docs/conf.py`
# extracts the API from the checkout rather than from an installed package.
# Executing the gallery needs both, because the examples run BART.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/docs/_build/html"
PYTHON="${PYTHON:-python3}"

execute=0
clean=0
serve=""
strict="-W --keep-going"

usage() {
    sed -n '3,12p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'
    cat <<'EOF'

Options:
  --execute       Run the gallery examples and keep their figures.  Needs a
                  built library; point BARTORCH_LIBRARY at it, or let the
                  script find build/*/libbartorch.*.
  --online        Resolve intersphinx against python.org, numpy and torch.
                  Off by default so the build works without a network.
  --clean         Remove the built HTML and everything generated into the
                  source tree before building.
  --lax           Do not turn warnings into errors.  The workflow does, so
                  a build that needs this is a build that will fail there.
  --serve [PORT]  Serve the result on PORT (default 8000) when it is built.
  --install       Install the documentation requirements first.
  -h, --help      This.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --execute) execute=1 ;;
        --online) export BARTORCH_DOCS_ONLINE=1 ;;
        --clean) clean=1 ;;
        --lax) strict="" ;;
        --serve)
            serve=8000
            if [ "${2:-}" ] && [ -z "${2##[0-9]*}" ]; then serve="$2"; shift; fi
            ;;
        --install) "${PYTHON}" -m pip install -r "${ROOT}/docs/requirements.txt" ;;
        -h|--help) usage; exit 0 ;;
        *) echo "build_docs.sh: unknown argument $1" >&2; usage >&2; exit 2 ;;
    esac
    shift
done

if ! "${PYTHON}" -c "import sphinx, sphinx_gallery, sphinx_book_theme" 2>/dev/null; then
    echo "build_docs.sh: the documentation requirements are not installed." >&2
    echo "  ${PYTHON} -m pip install -r docs/requirements.txt" >&2
    echo "or pass --install." >&2
    exit 1
fi

if [ "${execute}" = 1 ]; then
    export BARTORCH_DOCS_EXECUTE=1
    if [ -z "${BARTORCH_LIBRARY:-}" ]; then
        # The library one of the build directories holds, newest first.
        found="$(ls -t "${ROOT}"/build/*/libbartorch.so "${ROOT}"/build/*/libbartorch.dylib \
                 2>/dev/null | head -1 || true)"
        if [ -n "${found}" ]; then
            export BARTORCH_LIBRARY="${found}"
            echo "build_docs.sh: running the examples against ${found}"
        else
            echo "build_docs.sh: --execute runs the examples, which need the compiled" >&2
            echo "library.  Build it with" >&2
            echo "  cmake -S . -B build/local && cmake --build build/local -j" >&2
            echo "or install the package, and set BARTORCH_LIBRARY if it is elsewhere." >&2
            exit 1
        fi
    fi
    export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
fi

if [ "${clean}" = 1 ]; then
    # The reference pages and the gallery are written into the source tree, so
    # a stale page for something that has since been renamed would survive an
    # ordinary rebuild and be linked from nothing.
    rm -rf "${ROOT}/docs/_build" \
           "${ROOT}/docs/api/generated" \
           "${ROOT}/docs/auto_examples" \
           "${ROOT}/docs/sg_execution_times.rst"
fi

# shellcheck disable=SC2086
"${PYTHON}" -m sphinx -b html ${strict} "${ROOT}/docs" "${OUT}"

echo
echo "build_docs.sh: ${OUT}/index.html"

if [ -n "${serve}" ]; then
    echo "build_docs.sh: serving on http://localhost:${serve}/ -- ^C to stop"
    "${PYTHON}" -m http.server "${serve}" --directory "${OUT}"
fi
