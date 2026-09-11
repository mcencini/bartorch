#!/usr/bin/env bash
#
# Run the suite against a local build, without installing anything.
#
#   ./scripts/run_tests.sh                    build what changed, then run it all
#   ./scripts/run_tests.sh tests/test_solve.py -k pics
#   ./scripts/run_tests.sh --rebuild          configure from scratch first
#
# The workflow installs the package and runs `pytest tests/`; this builds into
# `build/local` and puts `src/` on the path instead.  The Python side needs
# nothing: it is read from `src/` on every run, the way an editable install
# reads it.  The C and C++ side is compiled, so it is built first -- only what
# changed, and nothing at all when nothing did.  Arguments after the options
# go to pytest.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD="${BARTORCH_BUILD_DIR:-${ROOT}/build/local}"
PYTHON="${PYTHON:-python3}"

rebuild=0
generate=1
build=1

usage() {
    sed -n '3,14p' "${BASH_SOURCE[0]}" | sed 's/^#\{1,\} \{0,1\}//'
    cat <<'EOF'

Options:
  --rebuild        Configure the build directory again before building.  Not
                   normally needed: CMake re-runs itself when CMakeLists.txt
                   changes and the globs over BART's sources are
                   CONFIGURE_DEPENDS, so an added or removed file is noticed.
  --no-build       Run against the library as it stands.  `tests/test_build.py`
                   then fails if it is older than the sources, which is the
                   point of having it.
  --no-generate    Skip regenerating `_abi.py` and `_catalogue.py`.  They are
                   checked in and the suite fails when they are out of date,
                   so the default is to write them before running.
  --cuda           Configure with BART's CUDA kernels.  Implies --rebuild.
  -h, --help       This.

Options come before pytest's arguments.  PYTHON, CC, CXX and BARTORCH_BUILD_DIR
are honoured.
EOF
}

configure=("-DCMAKE_BUILD_TYPE=Release")
while [ $# -gt 0 ]; do
    case "$1" in
        --rebuild) rebuild=1 ;;
        --no-build) build=0 ;;
        --no-generate) generate=0 ;;
        --cuda) rebuild=1; configure+=("-DBARTORCH_CUDA=ON") ;;
        -h|--help) usage; exit 0 ;;
        --) shift; break ;;
        *) break ;;
    esac
    shift
done

library="${BUILD}/libbartorch.so"
[ "$(uname -s)" = "Darwin" ] && library="${BUILD}/libbartorch.dylib"

if [ "${build}" = 1 ]; then
    if [ "${rebuild}" = 1 ] || [ ! -f "${BUILD}/CMakeCache.txt" ]; then
        cmake -S "${ROOT}" -B "${BUILD}" "${configure[@]}"
    fi
    # `scripts/sources.py` is the same list `tests/test_build.py` fails the
    # suite over, so what this calls up to date and what that calls up to date
    # are the same thing by construction.  The build below is incremental and
    # would do nothing anyway; asking first is what lets the script say so.
    if "${PYTHON}" "${ROOT}/scripts/sources.py" --newer-than "${library}" >/dev/null; then
        echo "run_tests.sh: the C and C++ sources are unchanged since the last build"
    else
        changed="$("${PYTHON}" "${ROOT}/scripts/sources.py" --newer-than "${library}" || true)"
        echo "run_tests.sh: ${changed:-the library} changed; rebuilding"
        cmake --build "${BUILD}" -j"$( (nproc || sysctl -n hw.ncpu || echo 4) 2>/dev/null )"
    fi
fi

if [ ! -f "${library}" ]; then
    echo "run_tests.sh: no libbartorch in ${BUILD}" >&2
    echo "  cmake -S . -B ${BUILD} && cmake --build ${BUILD} -j" >&2
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
