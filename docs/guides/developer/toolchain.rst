Development toolchain
=====================

Preinstall your chosen PyTorch build as described in :doc:`../user/installation`.
Native development needs Git with submodules, CMake 3.18+, Ninja or Make, and
Clang or GCC 14+. The CMake configuration is authoritative: Clang uses Blocks,
while GCC 14+ uses heap trampolines. Linux OpenMP builds need the matching
compiler's OpenMP runtime. CUDA builds also need a compatible toolkit with nvcc;
a PyTorch CUDA wheel alone is not a compiler. Windows support remains a
development item.

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

Keep compiler/CUDA configurations in separate build directories. For an explicit
CMake build::

   cmake -S . -B build-docdev -DCMAKE_BUILD_TYPE=Release \
       -DCMAKE_C_COMPILER=clang -DCMAKE_CXX_COMPILER=clang++
   cmake --build build-docdev -j
   BARTORCH_LIBRARY=$PWD/build-docdev/libbartorch.so PYTHONPATH=src pytest tests/

The suffix above is for Linux; use ``libbartorch.dylib`` on macOS. Install NUFFT
extras for tests that require them.

Repository architecture
-----------------------

* ``bart/``: pinned upstream submodule, compiled without local source edits.
* ``csrc/include/bartorch.h``: plain C ABI; ``csrc/`` implements array
  registration, backend integration, operators and execution.
* ``src/bartorch/``: Python tools, dispatch, operators and utilities.
* ``build_tools/gen_tools.py``: BART metadata extraction into committed wrappers
  and NumPy-style docstrings.
* ``tests/``: numerical and interoperability checks.
* ``docs/``: guides, gallery scripts, build configuration and API extraction.

The compiled library uses no Python or torch C API. Python passes pointers and
reversed dimension vectors through ctypes. Keep BART changes at the existing
replacement/compile-configuration boundary described in ``AGENTS.md``.
