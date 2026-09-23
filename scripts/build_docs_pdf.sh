#!/usr/bin/env bash
#
# Build the documentation as one PDF, docs/_build/bartorch-docs.pdf.
#
#   ./scripts/build_docs_pdf.sh             the pages as a render-only build shows them
#   ./scripts/build_docs_pdf.sh --execute   with the gallery run, as a release is built
#
# Sphinx renders the sources as a single HTML page, with the page of every API
# object in it, and scripts/print_pdf.py prints that page with headless
# Chromium once MathJax has typeset it.  An executed gallery is reused when its
# outputs are current, so this after `build_docs.sh --execute` runs nothing
# twice.  Needs the documentation requirements and Playwright's Chromium:
#
#   pip install -r docs/requirements.txt && python -m playwright install chromium

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${ROOT}/docs/_build"
PYTHON="${PYTHON:-python3}"

while [ $# -gt 0 ]; do
    case "$1" in
        --execute) export BARTORCH_DOCS_EXECUTE=1 ;;
        -h|--help) sed -n '3,14p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'; exit 0 ;;
        *) echo "build_docs_pdf.sh: unknown argument $1" >&2; exit 2 ;;
    esac
    shift
done

version="$("${PYTHON}" -c 'import importlib.metadata as m; print(m.version("bartorch"))' 2>/dev/null || true)"

# Its own doctrees: the page tree differs from the site's, which leaves the
# object pages out of the navigation.
BARTORCH_DOCS_PDF=1 "${PYTHON}" -m sphinx -b singlehtml -W --keep-going \
    -d "${BUILD}/singlehtml-doctrees" "${ROOT}/docs" "${BUILD}/singlehtml"

"${PYTHON}" "${ROOT}/scripts/print_pdf.py" "${BUILD}/singlehtml" "${BUILD}/bartorch-docs.pdf" \
    --version "${version:+Version ${version}}"
