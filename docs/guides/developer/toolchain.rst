Development toolchain
=====================

Preinstall your chosen PyTorch build as described in :doc:`../user/installation`.
Native development needs Git with submodules, CMake 3.18+, Ninja or Make, and
Clang or GCC 14+. The CMake configuration is authoritative: Clang uses Blocks,
while GCC 14+ uses heap trampolines. Linux OpenMP builds need the matching
compiler's OpenMP runtime. CUDA builds also need a compatible toolkit with nvcc;
a PyTorch CUDA wheel alone is not a compiler. Windows is not a target -- BART
does not build on it -- and WSL2 is a Linux toolchain like any other.

Fork and clone
--------------

Fork `mcencini/bartpy <https://github.com/mcencini/bartpy>`_ on GitHub. Replace
``YOUR-USERNAME`` with your account name::

   git clone --recurse-submodules https://github.com/YOUR-USERNAME/bartpy.git
   cd bartpy
   git remote add upstream https://github.com/mcencini/bartpy.git
   git switch -c docs/my-improvement
   python -m pip install -e '.[dev]'

For an existing clone, run ``git submodule update --init --recursive``.
To choose Clang explicitly::

   CC=clang CXX=clang++ python -m pip install -e '.[dev]'

For CUDA::

   python -m pip install -e '.[dev,finufft,cufinufft]' \
       --config-settings=cmake.define.BARTORCH_CUDA=ON

Keep compiler/CUDA configurations in separate build directories.
``./scripts/run_tests.sh`` does the usual one -- build into ``build/local``,
regenerate the committed generated files, run the suite against ``src/`` -- and
passes anything else to pytest::

   ./scripts/run_tests.sh
   ./scripts/run_tests.sh tests/test_solve.py -k pics
   BARTORCH_BUILD_DIR=$PWD/build-clang CC=clang ./scripts/run_tests.sh --rebuild

The same by hand::

   cmake -S . -B build-docdev -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
   cmake --build build-docdev -j
   BARTORCH_LIBRARY=$PWD/build-docdev/libbartorch.so PYTHONPATH=src pytest tests/

The suffix above is for Linux; use ``libbartorch.dylib`` on macOS. Install NUFFT
extras for tests that require them.

Repository architecture
-----------------------

Everything written here lives under ``src/`` and everything that is not lives
under ``external/``.

* ``external/bart/``: pinned upstream submodule, compiled without local source
  edits.  ``external/pocketfft/`` and ``external/blocksruntime/`` are vendored
  with their licenses.
* ``src/csrc/include/bartorch.h``: the plain C ABI, and the only header a host
  sees.  The implementation beside it is in three parts: ``abi/`` is the
  boundary, ``ops/`` the operators and the solve the host drives, and
  ``substitute/`` what runs in BART's place -- its transforms, and the
  libraries it would otherwise have been linked against.
* ``src/bartorch/``: Python tools, dispatch, operators and utilities.
* ``scripts/``: everything run by hand.  ``gen_catalogue.py`` extracts BART's
  metadata into the committed catalogue and ``gen_abi.py`` the ctypes
  signatures from the ABI header; ``run_tests.sh`` and ``build_docs.sh`` are
  the suite and the reference; ``check_device.py`` is what a machine with a
  card runs.
* ``cmake/``: what the build system runs and a person does not.
* ``tests/``: numerical and interoperability checks.
* ``docs/``: guides, gallery scripts, build configuration and API extraction.

The compiled library uses no Python or torch C API. Python passes pointers and
reversed dimension vectors through ctypes. Keep BART changes at the existing
replacement/compile-configuration boundary described in ``AGENTS.md``.
