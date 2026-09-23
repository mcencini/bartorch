#!/usr/bin/env bash
#
# Build the reference locally, the way the docs workflow builds it.
#
#   ./scripts/build_docs.sh                  render, warnings are errors
#   ./scripts/build_docs.sh --execute        run the gallery's examples too
#   ./scripts/build_docs.sh --clean --serve  start over, then serve the result
#
# Rendering imports bartorch from src/, which needs torch but not the compiled
# library.  Executing the examples needs the compiled library, and the packages
# `docs/examples/README.rst` names.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${ROOT}/docs/_build/html"
PYTHON="${PYTHON:-python3}"

clean=0
serve=""
strict="-W --keep-going"

usage() {
    sed -n '3,11p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'
    cat <<'EOF'

Options:
  --execute       Run the gallery's examples and render their output.  Off by
                  default: it needs a built library and the example
                  dependencies, and it takes minutes rather than seconds.
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
        --execute) export BARTORCH_DOCS_EXECUTE=1 ;;
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

if ! "${PYTHON}" -c "import sphinx, sphinx_book_theme, myst_parser, sphinx_gallery, matplotlib, torch" 2>/dev/null; then
    echo "build_docs.sh: the documentation requirements are not installed." >&2
    echo "  ${PYTHON} -m pip install -r docs/requirements.txt" >&2
    echo "or pass --install." >&2
    exit 1
fi

if [ "${clean}" = 1 ]; then
    # The reference pages are written into the source tree, so a stale page for
    # something that has since been renamed would survive an ordinary rebuild
    # and be linked from nothing.
    rm -rf "${ROOT}/docs/_build" \
           "${ROOT}/docs/generated" \
           "${ROOT}/docs/api_objects.rst" \
           "${ROOT}/docs/auto_examples" \
           "${ROOT}/docs/sg_execution_times.rst"
fi

# The doctree cache is kept beside the output rather than inside it, so the
# directory the workflow publishes is the site and nothing else.
# shellcheck disable=SC2086
"${PYTHON}" -m sphinx -b html ${strict} -d "${ROOT}/docs/_build/doctrees" "${ROOT}/docs" "${OUT}"

echo
echo "build_docs.sh: ${OUT}/index.html"

if [ -n "${serve}" ]; then
    echo "build_docs.sh: serving on http://localhost:${serve}/ -- ^C to stop"
    "${PYTHON}" -m http.server "${serve}" --directory "${OUT}"
fi
