"""Hand-written wrapper marking, and wrappers derived from BART's command catalogue.

A derived wrapper has a real signature, BART's help as a numpydoc docstring, and
each keyword routed to the flag BART spells it with.
"""

from __future__ import annotations

import inspect
import keyword
from typing import Any

import torch

from bartorch._catalogue import COMMANDS, Command, Option
from bartorch._dispatch import dispatch
from bartorch._options import python_name

__all__ = ["build", "curated", "signature_for"]


def curated(*commands: str):
    """Mark a hand-written wrapper as running BART's ``commands``; the first is its main one.

    The coverage audit reads the mark.
    """

    def mark(function):
        function.bart_command = commands[0]
        function.bart_commands = commands
        function.is_derived = False
        return function

    return mark


#: What a catalogue kind is in Python.
ANNOTATIONS = {
    "INT": "int",
    "UINT": "int",
    "PINT": "int",
    "LONG": "int",
    "ULONG": "int",
    "ULLONG": "int",
    "FLOAT": "float",
    "DOUBLE": "float",
    "STRING": "str",
    "SET": "bool",
    "CLEAR": "bool",
    "SELECT": "bool",
    "INFILE": "torch.Tensor",
    "INOUTFILE": "torch.Tensor",
    # ARG_CFL / OPT_CFL is a complex number read from the command line.
    "CFL": "complex",
    "OUTFILE": "str",
    "SUBOPT": "str",
    "SPECIAL": "str",
    "VEC2": "tuple[int, int]",
    "VEC3": "tuple[int, int, int]",
    "VECN": "tuple[int, ...]",
    "FLVEC2": "tuple[float, float]",
    "FLVEC3": "tuple[float, float, float]",
    "FLVEC4": "tuple[float, float, float, float]",
    "FLVECN": "tuple[float, ...]",
    "DOVEC3": "tuple[float, float, float]",
    "DOVECN": "tuple[float, ...]",
}

#: An option that takes no value is a flag, and its default is off.
FLAG_KINDS = frozenset({"SET", "CLEAR", "SELECT"})


def annotation(kind: str) -> str:
    """The type annotation for one catalogue kind."""
    return ANNOTATIONS.get(kind, "Any")


def _parameters(command: Command) -> tuple[list[inspect.Parameter], dict[str, Option]]:
    """The signature of a derived wrapper, and the option each keyword stands for.

    A positional BART does not require becomes keyword-only rather than a
    positional with a default, because BART puts optional arguments before
    required ones -- ``bart copy [dim pos]... <input> <output>`` -- and Python
    has no way to spell that.  Where each value goes in the argument vector is
    read off the catalogue, not off the signature, so moving it is free.
    """
    parameters: list[inspect.Parameter] = []
    deferred: list[inspect.Parameter] = []
    for argument in command.arguments:
        if argument.kind == "OUTFILE":
            continue
        kind = "torch.Tensor" if argument.is_array else annotation(argument.kind)
        if argument.required:
            parameters.append(
                inspect.Parameter(
                    _identifier(argument.name),
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                    annotation=kind,
                )
            )
        else:
            deferred.append(
                inspect.Parameter(
                    _identifier(argument.name),
                    inspect.Parameter.KEYWORD_ONLY,
                    default=None,
                    annotation=f"{kind} | None",
                )
            )
    parameters.extend(deferred)

    options: dict[str, Option] = {}
    for option in command.options:
        keyword = _identifier(python_name(option))
        if keyword in options or any(p.name == keyword for p in parameters):
            continue
        options[keyword] = option
        flag = option.kind in FLAG_KINDS
        parameters.append(
            inspect.Parameter(
                keyword,
                inspect.Parameter.KEYWORD_ONLY,
                default=False if flag else None,
                annotation="bool" if flag else f"{annotation(option.kind)} | None",
            )
        )
    parameters.append(inspect.Parameter("extra", inspect.Parameter.VAR_KEYWORD, annotation="Any"))
    return parameters, options


def _identifier(name: str) -> str:
    """A BART name as something Python can be given as a keyword.

    A flag BART spells with a digit takes the ``flag_`` prefix the rest of the
    package already uses for one: ``-3`` is ``flag_3``.
    """
    cleaned = "".join(c if c.isalnum() or c == "_" else "_" for c in name)
    if not cleaned:
        cleaned = "arg"
    if cleaned[0].isdigit():
        cleaned = f"flag_{cleaned}"
    return f"{cleaned}_" if keyword.iskeyword(cleaned) else cleaned


def signature_for(name: str) -> inspect.Signature:
    """The signature a derived wrapper for *name* has."""
    parameters, _ = _parameters(COMMANDS[name])
    return inspect.Signature(parameters, return_annotation="torch.Tensor | tuple | str | None")


def _docstring(command: Command, parameters: list[inspect.Parameter]) -> str:
    """BART's own help, as numpydoc."""
    lines = [command.help.strip(), "", f"Runs ``bart {command.name}``.", ""]
    lines += ["Parameters", "----------"]
    by_name = {p.name: p for p in parameters}
    for argument in command.arguments:
        if argument.kind == "OUTFILE":
            continue
        parameter = by_name.get(_identifier(argument.name))
        if parameter is None:
            continue
        lines.append(f"{parameter.name} : {parameter.annotation}")
        what = "Input array." if argument.is_array else f"Positional {argument.kind.lower()}."
        lines.append(f"    {what}" + ("" if argument.required else "  Optional."))
    for option in command.options:
        keyword = _identifier(python_name(option))
        parameter = by_name.get(keyword)
        if parameter is None or parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            continue
        lines.append(f"{keyword} : {parameter.annotation}")
        said = option.help.strip() or f"BART's {option.flag}."
        lines.append(f"    {said}  (``{option.flag}``)")
    lines += ["**extra : Any", "    Further BART flags, passed through by name."]

    outputs = command.outputs
    lines += ["", "Returns", "-------"]
    if not outputs:
        lines.append("None\n    This command writes no array; its printed text is returned.")
    elif len(outputs) == 1:
        lines.append(f"torch.Tensor\n    {outputs[0].name}")
    else:
        names = ", ".join(a.name for a in outputs)
        lines.append(f"tuple of torch.Tensor\n    {names}")
    return "\n".join(lines) + "\n"


def build(name: str, module: str):
    """The derived wrapper for BART command ``name``, reported as defined in ``module``."""
    command = COMMANDS[name]
    parameters, options = _parameters(command)
    # BART's own order, which is what the argument vector needs.
    arrays = [_identifier(a.name) for a in command.inputs]
    values = [_identifier(a.name) for a in command.values]
    # An output BART does not require is one the caller has to ask for, and a
    # derived wrapper has no way to be asked: `ecalib` writes eigenvalues only
    # when given somewhere to put them.
    n_out = len([a for a in command.outputs if a.required])

    def call(*args: Any, **kwargs: Any):
        bound = call.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        given = dict(bound.arguments)
        passed_through = given.pop("extra", {}) or {}

        inputs = [given[n] for n in arrays if given.get(n) is not None]
        positional = [given[n] for n in values if given.get(n) is not None]

        flags = {
            name: value
            for name, value in given.items()
            if name in options and value not in (None, False)
        }
        flags.update(passed_through)
        return dispatch(
            command.name,
            [x for x in inputs if x is not None],
            None if n_out else False,
            _pos=positional,
            _n_out=max(1, n_out),
            **flags,
        )

    call.__name__ = name
    call.__qualname__ = name
    call.__module__ = module
    call.__signature__ = inspect.Signature(
        parameters, return_annotation="torch.Tensor | tuple | str | None"
    )
    call.__doc__ = _docstring(command, parameters)
    #: Which BART command this is, for the coverage audit to read.
    call.bart_command = command.name
    call.is_derived = True
    return call


# Torch is imported for the annotations the built functions carry.
_ = torch
