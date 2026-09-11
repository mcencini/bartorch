# Contributions

For a substantial API change, describe the user workflow in an issue first.
Small fixes and documentation improvements can go directly to a pull request.
Work on a topic branch in your fork, and keep a pull request focused enough to
review on its own.

1. Read `AGENTS.md`, and the implementation and tests you are changing.
2. Make a coherent change, updating the documentation of any public behaviour
   it changes.  Do not edit generated files by hand.
3. Run the checks.  `./scripts/run_tests.sh` builds and runs the suite without
   installing anything and takes pytest arguments; Python changes also run
   `./scripts/lint.sh`, or `./scripts/lint.sh --fix` to have ruff write what it
   can.  Run the full suite for native changes and `python scripts/check_device.py`
   for device changes, on suitable hardware.
4. For documentation, run `./scripts/build_docs.sh`, the strict build described
   in {doc}`documentation`, and run changed examples with
   `./scripts/build_docs.sh --execute`.  State unavailable hardware or optional
   dependencies in the pull request.
5. Commit the intended files, push to your fork, and open a pull request
   against upstream's default branch, as a draft while work remains.

A pull request description states the problem, the resulting behaviour and how
it was validated, with reproducible commands and numerical tolerances where
relevant.  Cite the issue, and the paper for a new method.

To update your view of upstream:

```bash
git fetch upstream
git rebase upstream/main
git submodule update --init --recursive
```

Rebase only a topic branch whose history is yours to rewrite.  Updating the
BART submodule is deliberate work: regenerate with
`python scripts/gen_catalogue.py` and review the resulting API differences.
`./scripts/run_tests.sh` runs both generators before the suite, and
`tests/test_catalogue.py` and `tests/test_abi.py` fail when the checked-in files
have drifted.

A numerical test compares BART against something outside BART: an independent
FFT or DFT, a closed-form signal, a finite difference or an adjoint identity.
Comparing two entry points into the same implementation establishes nothing.
