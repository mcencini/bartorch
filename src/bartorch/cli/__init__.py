"""BART's command line, served by this package.

``bartorch <command> ...`` takes the arguments ``bart <command> ...`` takes and
answers what it answers, so a script that calls ``bart`` runs unchanged against
``bartorch``.  Where :mod:`bartorch.apps` has the pipeline, the command line is
read into a Python call and the app runs; everywhere else the command itself
runs, in this process.  The two answer the same bits, so which one ran is a
question about speed.

An argument the reader does not express is not an error: the whole command line
goes to BART instead, which is what declared it.  An input file that is not
there is the one thing named here rather than by BART, because a command that
fails while loading its arguments leaves the library unable to serve the next
call in the same process.
"""

from __future__ import annotations

import os
import sys

from bartorch import _call
from bartorch._catalogue import BART_VERSION, COMMANDS
from bartorch._dispatch import run_command
from bartorch._options import HELP_FLAGS, describe, options_by_name
from bartorch.apps.pics import image_shape
from bartorch.cli._apps import ADAPTERS
from bartorch.cli._argv import Unsupported, parse

__all__ = ["main", "read", "route"]

_USAGE = """\
Usage: bartorch <command> [options] [arguments]
       bartorch <command> --help
       bartorch --list

BART {version}, in-process.  Every command below takes the arguments
`bart <command>` takes.
"""


def _commands() -> list[str]:
    """Every command this can run, which is every command BART builds."""
    return sorted(name for name in COMMANDS if name != "bart")


def _listing() -> str:
    lines = [_USAGE.format(version=BART_VERSION), "Commands"]
    names = _commands()
    width = max(len(name) for name in names) + 2
    for name in names:
        lines.append(f"    {name:<{width}}{_call.summary(COMMANDS[name])}")
    return "\n".join(lines) + "\n"


def route(name: str, argv: list[str]) -> tuple[str, dict | None]:
    """Whether this command line runs as an app or as the command.

    The decision needs the arrays, because which axes a ``-R`` bitmask names
    depends on the image's rank, so the inputs are read here and handed on.

    Returns
    -------
    where : {'app', 'command'}
    plan : dict or None
        What the app is called with and where its answer goes, or ``None``
        under ``'command'``.
    """
    adapter = ADAPTERS.get(name)
    if adapter is None:
        return "command", None

    command = COMMANDS[name]
    try:
        options, files = parse(name, argv)
        if len(files) != len(command.arguments):
            raise Unsupported(f"{name} takes {len(command.arguments)} files, got {len(files)}")

        inputs = [
            read(path)
            for path, argument in zip(files, command.arguments)
            if argument.kind == "INFILE"
        ]
        outputs = [
            path for path, argument in zip(files, command.arguments) if argument.kind == "OUTFILE"
        ]
        # An option that names a file is an array like any other argument, so
        # it is loaded before the reader sees it.
        for keyword, values in options.items():
            option = options_by_name(name).get(keyword)
            if option is not None and option.kind == "INFILE":
                options[keyword] = [read(path) for path in values]

        trajectory = options.get("t", [None])[-1]
        shape = image_shape(inputs[0], inputs[1], trajectory)
        return "app", {
            "inputs": inputs,
            "outputs": outputs,
            "arguments": adapter(options, len(shape)),
        }
    except (Unsupported, OSError, ValueError):
        # An argument the reader cannot express, or a file it cannot read, is
        # the command's to answer: it declared them, and its message is the
        # one a script expects.
        return "command", None


#: What an array argument may be written as, beyond the name itself: BART opens
#: a CFL pair, a RA file or a COO file under the bare name.
_SUFFIXES = ("", ".cfl", ".hdr", ".ra", ".coo")


def _missing(name: str, argv: list[str]) -> list[str]:
    """The input files this command line names and the filesystem does not have.

    BART reports a missing input itself, and would be the one to ask -- except
    that a command which fails while loading its arguments leaves the library
    unable to serve the next call in the same process.  A caller who runs
    `main` twice would then hang rather than see the second answer, so the
    names are checked before BART is asked.
    """
    command = COMMANDS[name]
    try:
        options, files = parse(name, argv)
    except Unsupported:
        return []

    named = [
        path
        for path, argument in zip(files, command.arguments)
        if argument.kind in ("INFILE", "INOUTFILE")
    ]
    for keyword, values in options.items():
        option = options_by_name(name).get(keyword)
        if option is not None and option.kind in ("INFILE", "INOUTFILE"):
            named.extend(values)

    return [
        path
        for path in named
        if not path.startswith("-")
        and not any(os.path.exists(path + suffix) for suffix in _SUFFIXES)
    ]


def read(path: str):
    """One CFL pair as a tensor in this package's layout.

    ``readcfl`` answers in BART's own order, whose shape is the reverse of
    this one over the same bytes, so the transpose is the whole conversion.
    """
    import numpy as np
    import torch

    from bartorch.io import readcfl

    return torch.as_tensor(np.ascontiguousarray(readcfl(path).T))


def _run_app(name: str, plan: dict) -> int:
    """Call the app and write what it returns."""
    from bartorch import apps
    from bartorch.io import writecfl

    answer = getattr(apps, name)(*plan["inputs"], **plan["arguments"])
    writecfl(plan["outputs"][0], answer.detach().cpu().numpy().T)
    return 0


def main(argv: list[str] | None = None) -> int:
    """Run one command line.  Returns the exit code the command returns."""
    words = list(sys.argv[1:] if argv is None else argv)

    if not words or words[0] in HELP_FLAGS or words[0] == "--list":
        sys.stdout.write(_listing())
        return 0
    if words[0] in ("--version", "version"):
        sys.stdout.write(f"{BART_VERSION}\n")
        return 0

    name, rest = words[0], words[1:]
    if name not in COMMANDS:
        sys.stderr.write(f"bartorch: no command called {name!r}; bartorch --list has them\n")
        return 1
    if HELP_FLAGS.intersection(rest):
        # BART answers its own help by calling `exit`, which would take the
        # interpreter with it; the catalogue says the same thing.
        sys.stdout.write(describe(name))
        return 0

    absent = _missing(name, rest)
    if absent:
        for path in absent:
            sys.stderr.write(f"bartorch: {name}: no such input: {path}\n")
        return 1

    where, plan = route(name, rest)
    if where == "app":
        return _run_app(name, plan)

    code, printed, failure = run_command([name, *rest])
    if printed:
        sys.stdout.write(printed)
    if failure.strip() and failure.strip() not in printed:
        sys.stderr.write(failure if failure.endswith("\n") else failure + "\n")
    # BART answers a failure with a negative code, which a shell reads as 255.
    return 0 if code == 0 else 1
