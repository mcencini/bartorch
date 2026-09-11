# Toolchain

Preinstall your chosen PyTorch build as described in
{doc}`../user/installation`.  Native development needs Git with submodules,
CMake 3.18+, Ninja or Make, and clang or GCC 14+.  The CMake configuration is
authoritative: clang compiles BART's nested functions as Blocks, GCC 14+ as
heap trampolines.  Linux OpenMP builds need the matching compiler's OpenMP
runtime.  CUDA builds also need a compatible toolkit with nvcc; a PyTorch CUDA
wheel is not a compiler.  Windows is not a target -- BART does not build on it
-- and WSL2 is a Linux toolchain like any other.

## Fork and clone

Fork [mcencini/bartorch](https://github.com/mcencini/bartorch) on GitHub, and
replace `YOUR-USERNAME` with your account name:

```bash
git clone --recurse-submodules https://github.com/YOUR-USERNAME/bartorch.git
cd bartorch
git remote add upstream https://github.com/mcencini/bartorch.git
git switch -c docs/my-improvement
python -m pip install -e '.[dev]'
```

For an existing clone, run `git submodule update --init --recursive`.  To choose
clang explicitly:

```bash
CC=clang CXX=clang++ python -m pip install -e '.[dev]'
```

For CUDA:

```bash
python -m pip install -e '.[dev,cufinufft]' \
    --config-settings=cmake.define.BARTORCH_CUDA=ON
```

Keep compiler and CUDA configurations in separate build directories.
`./scripts/run_tests.sh` builds into `build/local`, regenerates the committed
generated files, runs the suite against `src/`, and passes anything else to
pytest:

```bash
./scripts/run_tests.sh
./scripts/run_tests.sh tests/test_solve.py -k pics
BARTORCH_BUILD_DIR=$PWD/build-clang CC=clang ./scripts/run_tests.sh --rebuild
```

The same by hand:

```bash
cmake -S . -B build-docdev -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build-docdev -j
BARTORCH_LIBRARY=$PWD/build-docdev/libbartorch.so PYTHONPATH=src pytest tests/
```

The suffix above is Linux's; use `libbartorch.dylib` on macOS.

## Repository layout

Everything written here lives under `src/`, and everything that is not lives
under `external/`.

- `external/bart/`: the pinned BART submodule, compiled without source edits.
  `external/pocketfft/` and `external/blocksruntime/` are vendored with their
  licenses.
- `src/csrc/include/bartorch.h`: the plain C ABI, the only header a host sees.
  The implementation is in three parts: `abi/` is the boundary, `ops/` the
  operators and the solve the host drives, and `substitute/` what runs in
  BART's place -- its transforms, and the libraries it would otherwise link.
- `src/bartorch/`: the Python package.
- `scripts/`: everything run by hand.  `gen_catalogue.py` extracts BART's
  command tables into the committed catalogue and `gen_abi.py` the ctypes
  signatures from the ABI header; `run_tests.sh` and `build_docs.sh` run the
  suite and build the documentation; `check_device.py` is what a machine with
  a card runs.
- `cmake/`: what the build system runs.
- `tests/`: numerical and interoperability checks.
- `docs/`: pages, gallery scripts and the build configuration.

The compiled library uses no Python or torch C API; Python passes pointers and
reversed dimension vectors through ctypes.  Keep BART changes at the
replacement and compile-configuration boundary described in `AGENTS.md`.
