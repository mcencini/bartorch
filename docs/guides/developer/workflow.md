# Development workflow

Describe a substantial change to the public API in an issue before opening a
pull request.  Small fixes and documentation corrections can go directly to a
pull request.  Work on a topic branch of a fork, one coherent change per
branch.

## Making a change

1. Read `AGENTS.md` and the implementation and tests of the code being changed.
2. Change the code and the documentation of any public behaviour it affects.
   Do not edit generated files by hand.
3. Run the checks:

   | Change | Command |
   | --- | --- |
   | Python | `./scripts/lint.sh` (`--fix` applies ruff's formatting and fixes) |
   | Any code | `./scripts/run_tests.sh`, or a subset of the suite during development |
   | C, CUDA or the ABI | The full suite; `python scripts/check_device.py` on a machine with a CUDA device |
   | Documentation | `./scripts/build_docs.sh`, and `./scripts/build_docs.sh --execute` for changed examples |

4. Commit, push to the fork, and open a pull request as described in
   {doc}`pull-requests`.

## Generated files

`src/bartorch/_abi.py` is generated from `src/csrc/include/bartorch.h` by
`python scripts/gen_abi.py`, and `src/bartorch/_catalogue.py` from BART's
command declarations by `python scripts/gen_catalogue.py`.  `run_tests.sh`
regenerates both before the suite, and `tests/test_abi.py` and
`tests/test_catalogue.py` fail when the checked-in files differ from what the
generators write.  The logo, the compact mark and the architecture figure in
`docs/_static/` are written by `python scripts/make_artwork.py`.

## Updating the BART submodule

Moving `external/bart` to a newer BART revision is a change of its own:
update the submodule pointer, regenerate the catalogue, and review the
resulting API differences.  `tests/test_tools.py` requires every BART command
to be wrapped by hand, generated into {mod}`bartorch.tools`, or listed as
private with a reason in `src/bartorch/_coverage.py`.

## Tests

A numerical test compares BART with a reference outside BART: an independent
FFT or DFT, an explicit sum, a closed-form signal, a finite difference or an
adjoint identity.  Comparing two entry points into the same BART
implementation establishes nothing about its correctness.  Tests are pytest
functions whose names state the property tested.

To update a topic branch with upstream changes:

```bash
git fetch upstream
git rebase upstream/main
git submodule update --init --recursive
```

Rebase only a branch whose history is yours to rewrite.
