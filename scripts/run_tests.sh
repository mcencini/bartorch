#!/usr/bin/env bash
#
# Run the suite against a local build, without installing anything.
#
#   ./scripts/run_tests.sh                    build if needed, then run it all
#   ./scripts/run_tests.sh tests/test_solve.py -k pics
#   ./scripts/run_tests.sh --rebuild          reconfigure first
#
# The workflow installs the package and runs `pytest tests/`; this builds into
# `build/local` and puts `src/` on the path instead, so a change to the Python
# side is one command away from being tested and a change to the C side is a
# rebuild rather than a reinstall.  Arguments after the options go to pytest.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BARTORCH_BUILD_DIR:-${ROOT}/build/local}"
PYTHON="${PYTHON:-python3}"

rebuild=0
generate=1

usage() {
    sed -n '3,13p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'
    cat <<'EOF'

Options:
  --rebuild        Reconfigure the build directory before building.  Needed
                   after a source file moves, because CMake caches the list.
  --no-generate    Skip regenerating `_abi.py` and `_catalogue.py`.  They are
                   checked in and the suite fails when they are out of date,
                   so the default is to write them before running.
  --cuda           Configure with BART's CUDA kernels.  Implies --rebuild.
  -h, --help       This.

Everything else is passed to pytest.  PYTHON, CC, CXX and BARTORCH_BUILD_DIR
are honoured.
EOF
}

configure=("-DCMAKE_BUILD_TYPE=Release")
while [ $# -gt 0 ]; do
    case "$1" in
        --rebuild) rebuild=1 ;;
        --no-generate) generate=0 ;;
        --cuda) rebuild=1; configure+=("-DBARTORCH_CUDA=ON") ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        *) break ;;
    esac
    shift
done

if [ "${rebuild}" = 1 ] || [ ! -f "${BUILD}/CMakeCache.txt" ]; then
    cmake -S "${ROOT}" -B "${BUILD}" "${configure[@]}"
fi
cmake --build "${BUILD}" -j"$( (nproc || sysctl -n hw.ncpu || echo 4) 2>/dev/null )"

library="${BUILD}/libbartorch.so"
[ -f "${library}" ] || library="${BUILD}/libbartorch.dylib"
if [ ! -f "${library}" ]; then
    echo "run_tests.sh: no libbartorch in ${BUILD}" >&2
    exit 1
fi

if [ "${generate}" = 1 ]; then
    # Both are checked in and both have a test that fails when they drift, so
    # writing them here is the difference between a real failure and a stale one.
    "${PYTHON}" "${ROOT}/scripts/gen_abi.py"
    "${PYTHON}" "${ROOT}/scripts/gen_catalogue.py"
fi

export PYTHONPATH="${ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export BARTORCH_LIBRARY="${library}"

echo "run_tests.sh: ${library}"
exec "${PYTHON}" -m pytest "${@:-${ROOT}/tests}"
