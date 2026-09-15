#!/usr/bin/env python3
"""Timings for the Gauss-Newton step, with the encoding applied both ways.

One case per process, each printing the plan it was lowered into beside its
times, as ``scripts/benchmark_encodings.py`` does for the linear encodings.
The rows are ``docs/design/nonlinear-fusion.md``'s targets.

    python scripts/benchmark_newton.py                 every case
    python scripts/benchmark_newton.py cartesian       one of them
"""

from __future__ import annotations

import argparse
import sys
import time

import torch

import bartorch.tools as bt
from bartorch import linop, nlop


def _encoding(case: str, n: int, coils: int):
    shape = (coils, 1, n, n)
    if "cartesian" == case:
        return linop.FFT(shape, axes=(-1, -2))
    return linop.NUFFT(bt.traj(x=n, y=401), shape)


def _timed(fn, repeats: int) -> float:
    """The best of ``repeats``, which is what a scattered machine reports honestly."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def run(case: str, *, n: int, coils: int, steps: int, repeats: int) -> None:
    torch.manual_seed(0)
    model = nlop.CoilSense(_encoding(case, n, coils))
    schedule = nlop.IRGNM(iterations=steps, alpha=1.0, redu=2.0, cg_maxiter=30, cg_tol=0.0)

    times = {}
    for name, fuse in (("normal", True), ("paired", False)):
        step = schedule.operator(model, fuse=fuse)
        data = step.prepare(torch.randn(step.model.oshapes[0], dtype=torch.complex64))
        start = torch.zeros(step.state_shape, dtype=torch.complex64)
        start[: n * n] = 1.0

        step(data, start, start, 1.0)  # the first call plans
        times[name] = _timed(lambda: step(data, start, start, 1.0), repeats)

        def backward():
            iterate = start.clone().requires_grad_(True)
            step(data, iterate, start, 1.0).abs().square().sum().backward()

        times[name + " backward"] = _timed(backward, repeats)

        if "normal" == name:
            print(f"  plan {step.plan!r}")
        print(
            f"  {name:7s} data {tuple(data.shape)}"
            f"  forward {times[name]:.3f} s"
            f"  backward {times[name + ' backward']:.3f} s"
        )

    for what in ("", " backward"):
        ratio = times["paired" + what] / times["normal" + what]
        print(f"  normal domain is {ratio:.2f}x the paired transform{what or ' forward'}")


CASES = {
    "cartesian": {"n": 256, "coils": 8, "steps": 8},
    "noncartesian": {"n": 256, "coils": 8, "steps": 8},
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=sorted(CASES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--size", type=int, default=None, help="override the grid")
    parser.add_argument("--steps", type=int, default=None, help="override the Newton steps")
    args = parser.parse_args(argv)

    for name in [args.case] if args.case else sorted(CASES):
        settings = dict(CASES[name])
        if args.size is not None:
            settings["n"] = args.size
        if args.steps is not None:
            settings["steps"] = args.steps
        print(f"{name}: {settings}")
        run(name, repeats=args.repeats, **settings)
    return 0


if __name__ == "__main__":
    sys.exit(main())
