Contributing and pull requests
==============================

For a substantial API change, describe the user workflow in an issue first.
Small fixes and documentation improvements can go directly to a PR. Work on a
topic branch in your fork, and keep a PR focused enough to review independently.

1. Read ``AGENTS.md`` and inspect the implementation and relevant tests.
2. Make a coherent change, updating documentation for public behavior changes.
   Avoid hand edits to generated wrappers.
3. Run appropriate checks. ``./scripts/run_tests.sh`` builds and runs the suite
   without installing anything, and takes pytest arguments; Python changes also
   run ``./scripts/lint.sh``, or ``./scripts/lint.sh --fix`` to have ruff write
   what it can.
   Run the full suite for broad native changes and
   ``python scripts/check_device.py`` for device changes on suitable hardware.
4. For documentation, run ``./scripts/build_docs.sh``, which is the strict build
   described in :doc:`documentation`, and execute changed runnable examples with
   ``./scripts/build_docs.sh --execute``. State unavailable hardware or optional
   dependencies in the PR validation notes.
5. Commit the intended files, push to your fork, and open a PR against upstream's
   default branch. Use a draft while work remains.

A PR description should explain the concrete problem, resulting behavior and
validation. Include reproducible commands and numerical tolerances where
relevant. Cite the issue and supporting paper for a new method. Resolve review
comments and conflicts without overwriting unrelated work.

To update your view of upstream::

   git fetch upstream
   git rebase upstream/main
   git submodule update --init --recursive

Rebase only a topic branch whose history is yours to rewrite. A submodule update
is intentional work: regenerate with ``python scripts/gen_catalogue.py`` and
review resulting API differences.  ``./scripts/run_tests.sh`` runs both
generators before the suite, and ``tests/test_catalogue.py`` and
``tests/test_abi.py`` fail when the checked-in files have drifted.

Numerical validation compares against an independent FFT/DFT, closed-form
signal, finite difference or adjoint identity. Comparing two entry points into
the same implementation cannot establish correctness on its own.
