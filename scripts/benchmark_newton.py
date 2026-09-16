#!/usr/bin/env python3
"""Timings for Gauss-Newton steps, with the encoding applied both ways.

One case per process, each printing the plan it was lowered into beside its
times, as ``scripts/benchmark_encodings.py`` does for the linear encodings.
The rows are ``docs/design/nonlinear-fusion.md``'s targets.

    python scripts/benchmark_newton.py                 every case
    python scripts/benchmark_newton.py cartesian       one of them
    python scripts/benchmark_newton.py --device cuda   on a card
    python scripts/benchmark_newton.py --batch 8          eight items, stepped one by one
    python scripts/benchmark_newton.py --batch 8 --items  the same eight as one model

On a card the spread between runs of the *same* variant is wide enough to be
mistaken for a difference between two, so a range is printed rather than a best,
and ``--variant`` runs one of them alone: an A/B wants a process each, and its
order alternated between them.
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time

import torch

import bartorch.tools as bt
from bartorch import linop, nlop


def _encoding(case: str, n: int, coils: int, lead=()):
    shape = (*lead, coils, 1, n, n)
    if "cartesian" == case:
        return linop.FFT(shape, axes=(-1, -2))
    return linop.NUFFT(bt.traj(x=n, y=401), shape)


def _model(case: str, n: int, coils: int, batch: int, items: bool):
    """The coil model and the shape of its data: a batch in front, or items inside."""
    if items:
        model = nlop.CoilSense(_encoding(case, n, coils, (batch,)), items=True)
        return model, tuple(model.oshapes[0])
    model = nlop.CoilSense(_encoding(case, n, coils))
    lead = (batch,) if 1 < batch else ()
    return model, (*lead, *model.oshapes[0])


def _sync(device: str) -> None:
    """A card queues the work: without this the clock reads the launch, not the run."""
    if "cuda" == device:
        torch.cuda.synchronize()


def _timed(fn, repeats: int, device: str) -> tuple[float, float]:
    """The range over ``repeats``, which is what decides whether a difference is one."""
    times = []
    for _ in range(repeats):
        _sync(device)
        start = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - start)
    return min(times), max(times)


def run(
    case: str,
    *,
    n: int,
    coils: int,
    steps: int,
    repeats: int,
    device: str,
    variant: str,
    batch: int = 1,
    items: bool = False,
) -> None:
    torch.manual_seed(0)
    model, shape = _model(case, n, coils, batch, items)

    wanted = (("normal", True), ("paired", False))
    if "both" != variant:
        wanted = tuple(one for one in wanted if one[0] == variant)

    times = {}
    for name, fuse in wanted:
        block = nlop.IRGNMBlock(alpha=1.0, redu=2.0, cg_maxiter=30, cg_tol=0.0, fuse=fuse)
        y = torch.randn(shape, dtype=torch.complex64, device=device)
        initial = block.start(y, model)  # prepares the data, and plans
        data, start = initial.data, initial.x

        def stepped(x, initial=initial, block=block):
            state = dataclasses.replace(initial, x=x)
            for _ in range(steps):
                state = block(state, model)
            return state.x

        stepped(start)  # the first call builds what the steps apply
        times[name] = _timed(lambda: stepped(start), repeats, device)

        def backward():
            iterate = start.clone().requires_grad_(True)
            stepped(iterate).abs().square().sum().backward()

        times[name + " backward"] = _timed(backward, repeats, device)

        print(f"  plan {block.plan(model)!r}")
        print(
            f"  {name:7s} data {tuple(data.shape)}"
            f"  forward {times[name][0]:.3f}-{times[name][1]:.3f} s"
            f"  backward {times[name + ' backward'][0]:.3f}-{times[name + ' backward'][1]:.3f} s"
        )

    if "both" != variant:
        return
    for what in ("", " backward"):
        paired, normal = times["paired" + what], times["normal" + what]
        apart = "" if paired[0] > normal[1] or normal[0] > paired[1] else " -- inside the spread"
        print(
            f"  normal domain is {paired[0] / normal[0]:.2f}x the paired transform"
            f"{what or ' forward'}{apart}"
        )


CASES = {
    "cartesian": {"n": 256, "coils": 8, "steps": 8},
    "noncartesian": {"n": 256, "coils": 8, "steps": 8},
}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("case", nargs="?", choices=sorted(CASES))
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", default="cpu", help="cpu, or cuda for a card")
    parser.add_argument(
        "--variant", default="both", choices=["both", "normal", "paired"], help="one side alone"
    )
    parser.add_argument("--size", type=int, default=None, help="override the grid")
    parser.add_argument("--steps", type=int, default=None, help="override the Newton steps")
    parser.add_argument("--batch", type=int, default=1, help="independent items")
    parser.add_argument("--items", action="store_true", help="the batch as one model")
    args = parser.parse_args(argv)

    for name in [args.case] if args.case else sorted(CASES):
        settings = dict(CASES[name])
        if args.size is not None:
            settings["n"] = args.size
        if args.steps is not None:
            settings["steps"] = args.steps
        kind = "one model" if args.items else "one by one"
        print(f"{name}: {settings}, batch {args.batch} {kind}, on {args.device}")
        run(
            name,
            repeats=args.repeats,
            device=args.device,
            variant=args.variant,
            batch=args.batch,
            items=args.items,
            **settings,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
