"""Hand-written wrapper marking, and wrappers derived from BART's command catalogue.

A derived wrapper has a real signature, BART's help as a numpydoc docstring, and
each keyword routed to the flag BART spells it with.  An argument BART takes
as a dimension number or a bitmask is taken here as axes, indices or
:mod:`bartorch.priors` terms, as :data:`TRANSLATED` says.
"""

from __future__ import annotations

import dataclasses
import inspect
import keyword
import re
from typing import Any

import torch

from bartorch._catalogue import COMMANDS, Command, Option
from bartorch._dispatch import dispatch
from bartorch._flags import _axes_to_dims, _indices_to_flags
from bartorch._options import options_by_name, python_name

__all__ = ["TRANSLATED", "build", "curated", "signature_for", "translate"]


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


@dataclasses.dataclass(frozen=True)
class _Takes:
    """How Python gives one argument that BART spells as dimensions or a bitmask.

    ``what`` is ``"axes"`` (C-order axes, sent as a bitmask), ``"axis"`` (one
    C-order axis, sent as BART's dimension number), ``"indices"`` (a set of
    indices that are not axes, sent as a bitmask) or ``"regularizers"``
    (:mod:`bartorch.priors` terms, sent as ``-R`` arguments).  Axes count along
    the command's input number ``input``; with None, or with that input not
    given, only negative axes are accepted.  ``kinds`` lists the terms a
    command's parser knows.
    """

    what: str
    help: str
    input: int | None = 0
    kinds: frozenset[str] | None = None

    @property
    def annotation(self) -> str:
        return {
            "axes": "int | tuple[int, ...]",
            "axis": "int",
            "indices": "int | tuple[int, ...]",
            "regularizers": "Regularizer | list[Regularizer]",
        }[self.what]

    def convert(self, value: Any, inputs: list, command: str) -> Any:
        if self.what == "indices":
            return _indices_to_flags(value)
        array = inputs[self.input] if self.input is not None and self.input < len(inputs) else None
        ndim = array.ndim if isinstance(array, torch.Tensor) else None
        if self.what == "regularizers":
            from bartorch.priors.base import _as_terms, _command_line

            arguments, shared = _command_line(_as_terms(value), ndim, command, self.kinds)
            if shared:
                raise ValueError(
                    f"{command} is given {', '.join(sorted(shared))} once for every term, "
                    "and its wrapper keeps BART's default for each; use terms that do too"
                )
            return arguments
        if self.what == "axis":
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{command} takes one axis here, not {value!r}")
            return _axes_to_dims(value, ndim)[0]
        return sum(1 << d for d in _axes_to_dims(value, ndim))


#: BART's term letters by parser: ``pics``, ``wshfl`` and ``denoise`` share
#: one; ``sqpics`` and ``moba`` each have their own.
_PICS_KINDS = frozenset("W H N L T G C V P S Q F I R1 R2".split())
_SQPICS_KINDS = frozenset("W L T I Q R1 R2".split())
_MOBA_KINDS = frozenset("W T Q".split())

#: Each argument of a public command that BART takes as dimensions or a
#: bitmask, by (command, flag or positional name).  The hand-written wrappers
#: turn their own; this table covers the derived wrappers and what a curated
#: one passes through by name.
TRANSLATED: dict[tuple[str, str], _Takes] = {
    ("bin", "-l"): _Takes("axis", "Bin according to the labels, clustered along this axis."),
    ("ccapply", "-A"): _Takes("axis", "Align the coil sensitivities along this axis."),
    ("coils", "-b"): _Takes("indices", "Channels to keep, by index."),
    ("ecalib", "-e"): _Takes("axis", "Split the second step along this axis."),
    ("homodyne", "dim"): _Takes("axis", "Axis the partial-Fourier fraction is along."),
    ("lrmatrix", "-m"): _Takes("axes", "Axes reshaped into the matrix columns."),
    ("lrmatrix", "-f"): _Takes("axes", "Axes the multi-scale partition is along."),
    ("moba", "-r"): _Takes("regularizers", "Regularization terms.", kinds=_MOBA_KINDS),
    ("moba", "--positive-maps"): _Takes(
        "indices", "Parameter maps constrained to be positive, by index."
    ),
    ("moba", "--l2-on-parameters"): _Takes("indices", "Parameter maps with an l2 norm, by index."),
    ("mobafit", "--min-flag"): _Takes(
        "indices", "Parameter maps with a minimum constraint, by index."
    ),
    ("mobafit", "--max-flag"): _Takes(
        "indices", "Parameter maps with a maximum constraint, by index."
    ),
    ("mobafit", "--max-mag-flag"): _Takes(
        "indices", "Parameter maps with a maximum magnitude constraint, by index."
    ),
    ("ncalib", "--shared-img-dims"): _Takes("axes", "Axes the image is shared along."),
    ("ncalib", "--shared-col-dims"): _Takes(
        "axes", "Axes the coil sensitivities are shared along."
    ),
    ("ncalib", "--scale-loop-dims"): _Takes(
        "axes", "Scale the parameters as if ncalib were looped over these axes."
    ),
    ("nlinv", "-s"): _Takes("axes", "Axes the sensitivities are constant along."),
    ("pattern", "-s"): _Takes("axes", "Axes to squash."),
    ("pics", "-L"): _Takes("axes", "Axes reconstructed one at a time (batch mode)."),
    ("pics", "--shared-img-dims"): _Takes("axes", "Axes the image is shared along."),
    ("pics", "--mpi"): _Takes("axes", "Axes distributed over MPI processes."),
    ("sqpics", "-R"): _Takes("regularizers", "Regularization terms.", kinds=_SQPICS_KINDS),
    ("ssa", "-g"): _Takes("indices", "Grouping, as a set of indices."),
    ("wshfl", "-R"): _Takes("regularizers", "Regularization terms.", kinds=_PICS_KINDS),
}


def _rule(command: str, keyword: str) -> _Takes | None:
    option = options_by_name(command).get(keyword)
    return TRANSLATED.get((command, option.flag if option is not None else keyword))


def translate(command: str, values: dict[str, Any], inputs: list) -> dict[str, Any]:
    """``values``, keyword to value, with each argument :data:`TRANSLATED` covers as BART spells it.

    ``inputs`` are the command's input arrays in its own order, which axes
    count along.
    """
    out = {}
    for name, value in values.items():
        rule = _rule(command, name)
        if rule is not None and value is not None and value is not False:
            value = rule.convert(value, inputs, command)
        out[name] = value
    return out


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
        rule = TRANSLATED.get((command.name, argument.name))
        if rule is not None:
            kind = rule.annotation
        elif argument.is_array:
            kind = "torch.Tensor"
        else:
            kind = annotation(argument.kind)
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
        rule = TRANSLATED.get((command.name, option.flag))
        kind = rule.annotation if rule is not None else annotation(option.kind)
        parameters.append(
            inspect.Parameter(
                keyword,
                inspect.Parameter.KEYWORD_ONLY,
                default=False if flag else None,
                annotation="bool" if flag else f"{kind} | None",
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


#: Commands whose only output BART marks optional and then creates anyway.
#
# `create_cfl` hands the name to `io_unlink_if_opened`, which calls `strcmp`
# on it, so an omitted name is a segmentation fault rather than a tool that
# quietly writes nothing -- `mobafit.c:398` and `morphop.c:77` are both
# unguarded.  Every other command with an optional output writes it through
# `anon_cfl` or behind an `if`, prints its answer instead when no name is
# given, and is left alone: passing one would silently turn the printed line
# `estdelay` and `measure` return here into a tensor.
_WRITES_ITS_OPTIONAL_OUTPUT = frozenset({"mobafit", "morphop"})


def _outputs(command: Command) -> int:
    """How many output arrays to ask the command for."""
    required = len([a for a in command.outputs if a.required])
    if required or command.name not in _WRITES_ITS_OPTIONAL_OUTPUT:
        return required
    return 1


def _signal_whole_echo_trains(flags: dict) -> None:
    """Refuse a ``signal -C`` whose echoes do not divide the measurements.

    ``ir_multi_grad_echo_model`` (``simu/signals.c:461-467``) fills
    ``(N / NE) * NE`` entries of the ``N``-entry array ``signal.c:234``
    leaves uninitialized, and ``get_signal`` averages all ``N`` of them into
    the output.  Without ``-m`` at all, ``NE`` is BART's ``-1``, the loop runs
    zero times and none of the array is written.  What comes back is finite
    stack memory, so nothing downstream notices.
    """
    if not flags.get("C"):
        return

    # `signal.c:59` starts `dims[TE_DIM]` at 100; the model is given
    # `dims[TE_DIM] * averaged_spokes` entries to fill.
    measurements = int(flags.get("n", 100)) * int(flags.get("av_spokes", 1))
    echoes = flags.get("m")

    if echoes is None:
        raise ValueError(
            "signal -C needs m=, the number of gradient echoes: BART leaves it at -1 and "
            "then writes none of the signal it returns, which is uninitialized memory and "
            "not an error"
        )
    if int(echoes) < 1 or measurements % int(echoes):
        raise ValueError(
            f"signal -C measures {measurements} points over {echoes} gradient echoes, and "
            f"BART writes only {measurements // int(echoes) * int(echoes)} of them; the "
            "rest of what it returns is uninitialized memory.  Give an m= that divides "
            "n= (times av_spokes=)"
        )


#: What a command's flags have to satisfy for BART to define what it does.
#: Checked before the command runs, because what BART does otherwise is read
#: memory nothing wrote and answer with it.
_PRECONDITIONS = {"signal": _signal_whole_echo_trains}


def _laid_out(line: str) -> bool:
    """Whether a line of BART's help is part of a list, a table or an example."""
    return line[:1] in (" ", "*", "-") or line.startswith("bart ")


def _help(text: str) -> list[str]:
    """BART's help as reStructuredText.

    Prose is joined into paragraphs; what BART lays out by hand -- the lists of
    conventions and dimensions, the example command lines -- is kept verbatim.
    """
    out: list[str] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        rows = block.expandtabs(4).splitlines()
        first = next((i for i, row in enumerate(rows) if _laid_out(row)), len(rows))
        if first:
            out += [" ".join(row.strip() for row in rows[:first]), ""]
        if first < len(rows):
            out += ["::", ""] + [f"    {row}" if row.strip() else "" for row in rows[first:]]
            out.append("")
    return out


def summary(command: Command) -> str:
    """The first paragraph of BART's help, on one line."""
    return " ".join(re.split(r"\n\s*\n", command.help.strip())[0].split())


def _docstring(command: Command, parameters: list[inspect.Parameter]) -> str:
    """BART's own help, as numpydoc."""
    lines = [*_help(command.help), f"Runs ``bart {command.name}``.", ""]
    lines += ["Parameters", "----------"]
    by_name = {p.name: p for p in parameters}
    for argument in command.arguments:
        if argument.kind == "OUTFILE":
            continue
        parameter = by_name.get(_identifier(argument.name))
        if parameter is None:
            continue
        lines.append(f"{parameter.name} : {parameter.annotation}{_default(parameter)}")
        rule = TRANSLATED.get((command.name, argument.name))
        if rule is not None:
            what = rule.help
        elif argument.is_array:
            what = "Input array."
        else:
            what = f"Positional {argument.kind.lower()}."
        lines.append(f"    {what}" + ("" if argument.required else "  Optional."))
    for option in command.options:
        keyword = _identifier(python_name(option))
        parameter = by_name.get(keyword)
        if parameter is None or parameter.kind is not inspect.Parameter.KEYWORD_ONLY:
            continue
        lines.append(f"{keyword} : {parameter.annotation}{_default(parameter)}")
        rule = TRANSLATED.get((command.name, option.flag))
        said = rule.help if rule is not None else option.help.strip() or f"BART's {option.flag}."
        lines.append(f"    {said}  (``{option.flag}``)")
    lines += ["**extra : Any", "    Further BART flags, passed through by name."]

    # What `build` asks the command for: the outputs BART requires, which an
    # optional output of the command is not.
    returned = command.outputs[: _outputs(command)]
    lines += ["", "Returns", "-------"]
    if not returned:
        lines.append("str\n    The command's printed text; it writes no array.")
    elif len(returned) == 1:
        lines.append(f"torch.Tensor\n    {returned[0].name}")
    else:
        names = ", ".join(a.name for a in returned)
        lines.append(f"tuple of torch.Tensor\n    {names}")
    return "\n".join(lines) + "\n"


def _default(parameter: inspect.Parameter) -> str:
    """The ``, default=...`` a parameter's type line carries, from its signature."""
    if parameter.default is inspect.Parameter.empty:
        return ""
    return f", default={parameter.default!r}"


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
    n_out = _outputs(command)

    def call(*args: Any, **kwargs: Any):
        bound = call.__signature__.bind(*args, **kwargs)
        bound.apply_defaults()
        given = dict(bound.arguments)
        passed_through = given.pop("extra", {}) or {}

        inputs = [given.get(n) for n in arrays]
        given = translate(command.name, given, inputs)
        passed_through = translate(command.name, passed_through, inputs)
        positional = [given[n] for n in values if given.get(n) is not None]

        flags = {
            name: value
            for name, value in given.items()
            if name in options and value is not None and value is not False
        }
        flags.update(passed_through)

        check = _PRECONDITIONS.get(command.name)
        if check is not None:
            check(flags)

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
