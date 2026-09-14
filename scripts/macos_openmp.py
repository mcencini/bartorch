#!/usr/bin/env python3
"""Make torch's and FINUFFT's OpenMP runtimes one runtime, on macOS.

    python scripts/macos_openmp.py diagnose    # read only; says what it would do
    python scripts/macos_openmp.py patch       # repoint FINUFFT, then re-sign
    python scripts/macos_openmp.py verify      # the patch held, and both still run

The substitution does this for itself the first time it needs a NUFFT
(:func:`bartorch._macos_openmp.ensure`), so this is for seeing what it found,
for an environment where it could not, and for the macOS CI job, which runs
all three before the suite.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bartorch import _macos_openmp as mo  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument("mode", choices=("diagnose", "patch", "verify"))
    ap.add_argument("--site", type=Path, help="look here instead of at the installed packages")
    ap.add_argument("--dry-run", action="store_true", help="print the commands without running them")
    args = ap.parse_args(argv)

    if sys.platform != "darwin" and args.site is None:
        print("this is a macOS problem; nothing to do here", file=sys.stderr)
        return 0

    layout = mo.find_layout(args.site)
    decision = mo.describe(layout)
    print(f"\n{decision.action}: {decision.reason}")

    if args.mode == "diagnose":
        return 0 if decision.action in ("patch", "already") else 1

    if args.mode == "verify":
        if decision.action != "already":
            print("\nthe library is not pointed at torch's runtime", file=sys.stderr)
            return 1
        return 0 if mo.run_checks() else 1

    if decision.action == "already":
        return 0
    if decision.action == "refuse":
        return 1

    print("\npatching:")
    mo.apply(layout, decision, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
