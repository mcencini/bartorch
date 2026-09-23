# Editable and source installation

Fork [mcencini/bartorch](https://github.com/mcencini/bartorch) and clone the
fork with its submodules:

```bash
git clone --recurse-submodules https://github.com/YOUR-USERNAME/bartorch.git
cd bartorch
git remote add upstream https://github.com/mcencini/bartorch.git
```

An existing clone is completed with `git submodule update --init --recursive`.

## Editable installation

```bash
CC=clang CXX=clang++ python -m pip install -e '.[dev]'
```

compiles the library through scikit-build-core into `build/<wheel tag>/` and
installs the package in editable mode, with pytest, numpydoc and ruff.  The
pre-commit hooks, the spelling check and the documentation build need

```bash
python -m pip install pre-commit 'codespell[toml]' -r docs/requirements.txt
```

With CUDA:

```bash
python -m pip install -e '.[dev,cufinufft]' \
    --config-settings=cmake.define.BARTORCH_CUDA=ON
```

## Building and testing without installing

`./scripts/run_tests.sh` builds the library into `build/local`, regenerates the
checked-in generated files, and runs the suite against `src/`; arguments after
its options are passed to pytest:

```bash
./scripts/run_tests.sh
./scripts/run_tests.sh tests/test_solve.py -k pics
BARTORCH_BUILD_DIR=$PWD/build-clang CC=clang CXX=clang++ ./scripts/run_tests.sh --rebuild
./scripts/run_tests.sh --cuda          # configure with BART's CUDA kernels
```

The same steps by hand:

```bash
cmake -S . -B build/local -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
cmake --build build/local -j
BARTORCH_LIBRARY=$PWD/build/local/libbartorch.so PYTHONPATH=src pytest tests/
```

On macOS the library is `libbartorch.dylib`.  Keep builds with different
compilers or CUDA settings in separate build directories.  CUDA tests skip
without a device; on a machine with one, `python scripts/check_device.py`
checks the device path in dependency order.
