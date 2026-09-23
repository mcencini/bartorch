# Repository layout

Code written for bartorch is under `src/`, with its tests under `tests/` and
its scripts under `scripts/`; code from other projects is under `external/`.

| Path | Contents |
| --- | --- |
| `external/bart/` | BART, a Git submodule, compiled without source changes |
| `external/pocketfft/`, `external/blocksruntime/` | Vendored FFT and Blocks runtime, with their licenses |
| `src/csrc/include/bartorch.h` | The C ABI, the only header the Python side sees |
| `src/csrc/abi/` | Command execution, the in-memory CFL registry, CUDA stream ordering |
| `src/csrc/ops/` | Operators, the encoding executor, and the configuration of BART's iterative solve as `pics` sets it up |
| `src/csrc/substitute/` | Components compiled in place of BART's: the FINUFFT NUFFT, point spread functions, the FFT and BLAS/LAPACK tables |
| `src/bartorch/` | The Python package; private modules begin with an underscore |
| `src/bartorch/_abi.py`, `_catalogue.py` | Generated from the header and from BART's command declarations |
| `scripts/` | Scripts run by hand: tests, lint, documentation, generators, device checks, artwork |
| `cmake/` | Build helpers |
| `tests/` | The test suite |
| `docs/` | Documentation sources, the example gallery under `docs/examples/`, and the Sphinx configuration |
| `docs/design/` | Design records for maintainers, excluded from the built documentation |
| `attic/prototype/` | An earlier pybind11 extension, kept for reference and not built |

The compiled library uses no Python or PyTorch C API.  Python passes data
pointers and reversed dimension vectors through ctypes, which is why one wheel
per platform serves every Python and PyTorch version.  Changes to BART's
behaviour are made by replacing a translation unit or by a compile definition,
never by editing the submodule; `AGENTS.md` lists each replacement.

## Documentation sources

| Path | Contents |
| --- | --- |
| `docs/index.md` | Landing page: the README and the top-level toctree |
| `docs/guides/user/`, `docs/guides/developer/` | User and developer guides |
| `docs/explanation/` | Conceptual explanations |
| `docs/examples/` | Example scripts, one directory per section, and the examples landing page |
| `docs/api/` | API category pages; their tables list the objects |
| `docs/api_objects.py` | Collects the objects from the API tables into `docs/api_objects.rst`, which generates one page per object under `docs/generated/` |
| `docs/_templates/autosummary/` | Templates of the generated object pages |
| `docs/misc/` | License, related projects, contributors and citation |
