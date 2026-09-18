"""BART's own command line, read back into the values a Python call takes.

The grammar is `misc/opts.c`'s, and what each option means is the catalogue's,
so nothing here is a second list of BART's flags.  What it cannot express it
says so about, and the caller runs the command instead.
"""

from __future__ import annotations

from typing import Any

from bartorch._catalogue import COMMANDS, Option
from bartorch._options import python_name

__all__ = ["Unsupported", "axes", "parse", "regularizer"]


class Unsupported(Exception):
    """An argument this reader does not express, which is not an error.

    The command itself takes every argument it declares, so an argument the
    reader cannot turn into a Python value is a reason to run the command
    rather than the app -- the two answer the same bits.
    """


#: How an option's value is read, by the kind the catalogue gives it.  A kind
#: absent here takes no value; one BART declares and this does not read raises
#: :class:`Unsupported`.
_WITHOUT_VALUE = frozenset({"SET", "CLEAR", "SELECT"})
_SCALARS = {
    "INT": int,
    "PINT": int,
    "UINT": int,
    "LONG": int,
    "ULONG": int,
    "ULLONG": int,
    "FLOAT": float,
    "DOUBLE": float,
    "STRING": str,
    "INFILE": str,
    "OUTFILE": str,
    "INOUTFILE": str,
    "SPECIAL": str,
    "SUBOPT": str,
    "SUBOPT2": str,
}
_VECTORS = {
    "VEC2": (int, 2),
    "VEC3": (int, 3),
    "VECN": (int, 0),
    "FLVEC2": (float, 2),
    "FLVEC3": (float, 3),
    "FLVECN": (float, 0),
    "DOVEC3": (float, 3),
    "DOVECN": (float, 0),
}


def _value(option: Option, text: str) -> Any:
    if option.kind in _SCALARS:
        try:
            return _SCALARS[option.kind](text)
        except ValueError as problem:
            raise Unsupported(
                f"{option.flag} takes {option.kind.lower()}, got {text!r}"
            ) from problem
    if option.kind in _VECTORS:
        cast, count = _VECTORS[option.kind]
        parts = [cast(part) for part in text.split(":" if ":" in text else ",")]
        if count and len(parts) != count:
            raise Unsupported(f"{option.flag} takes {count} values, got {len(parts)}")
        return tuple(parts)
    raise Unsupported(f"{option.flag} is a {option.kind}, which this reader does not read")


def _tables(name: str) -> tuple[dict[str, Option], dict[str, Option]]:
    """The command's options by letter and by long name."""
    command = COMMANDS.get(name)
    if command is None:
        raise Unsupported(f"BART has no command called {name!r}")
    letters = {o.short: o for o in command.options if o.short}
    words = {o.long: o for o in command.options if o.long}
    return letters, words


def parse(name: str, argv: list[str]) -> tuple[dict[str, list[Any]], list[str]]:
    """``argv`` for command ``name``, as the values it names and the files it lists.

    Returns
    -------
    options : dict
        ``{python keyword: [value, ...]}``, a list because BART's ``-R`` and
        its kin may be given several times.  An option that takes no value
        carries ``True``, or ``False`` for a ``CLEAR``; a selector carries the
        spelling it was given by, so the caller can tell its arms apart.
    positional : list of str
        What is left, in order: the command's own arguments.

    Raises
    ------
    Unsupported
        An argument this reader does not express.
    """
    letters, words = _tables(name)
    options: dict[str, list[Any]] = {}
    positional: list[str] = []
    rest = list(argv)

    def take(option: Option, given: str, inline: str | None) -> None:
        if option.kind in _WITHOUT_VALUE:
            if inline is not None:
                raise Unsupported(f"{given} takes no value")
            value: Any = given if option.kind == "SELECT" else (option.kind == "SET")
        else:
            text = inline
            if text is None:
                if not rest:
                    raise Unsupported(f"{given} takes a value and none followed")
                text = rest.pop(0)
            value = _value(option, text)
        options.setdefault(python_name(option), []).append(value)

    only_files = False
    while rest:
        word = rest.pop(0)
        if only_files or word == "-" or not word.startswith("-"):
            positional.append(word)
            continue
        if word == "--":
            only_files = True
            continue
        if word.startswith("--"):
            spelling, _, inline = word[2:].partition("=")
            option = words.get(spelling)
            if option is None:
                raise Unsupported(f"--{spelling} is not an option of {name}")
            take(option, f"--{spelling}", inline or None)
            continue
        # A short flag, possibly with its value stuck to it, possibly several
        # value-less ones written together, which is what getopt accepts.
        body = word[1:]
        while body:
            option = letters.get(body[0])
            if option is None:
                raise Unsupported(f"-{body[0]} is not an option of {name}")
            if option.kind in _WITHOUT_VALUE:
                take(option, f"-{body[0]}", None)
                body = body[1:]
                continue
            take(option, f"-{body[0]}", body[1:] or None)
            break

    return options, positional


#: What each ``-R`` letter builds, and how many fields it carries after it.
#: The grammar is ``grecon/optreg.c``'s, which is what ``pics`` parses with.
_TERMS: dict[str, tuple[str, str]] = {
    "W": ("Wavelet", "xjw"),
    "T": ("TotalVariation", "xjw"),
    "L": ("LocallyLowRank", "xjw"),
    "P": ("Laplace", "xjw"),
    "F": ("FourierL1", "xjw"),
    "G": ("TotalGeneralizedVariation", "xjw"),
    "C": ("InfimalConvolutionTV", "xjw"),
    "V": ("InfimalConvolutionTGV", "xjw"),
    "H": ("WaveletNIHT", "xjk"),
    "N": ("ImageNIHT", "xjk"),
    "I": ("L1", "jw"),
    "R1": ("ImaginaryL1", "jw"),
    "R2": ("ImaginaryL2", "jw"),
    "Q": ("L2", "w"),
    "S": ("NonNegative", ""),
}


def axes(flags: int, ndim: int) -> tuple[int, ...]:
    """BART's bitmask as the axis indices a term takes.

    Bit ``i`` is the axis ``-(i + 1)``.  A bit past ``ndim`` is an axis BART
    carries as a singleton and the image here does not have at all -- a
    two-dimensional ``pics`` is written ``-R W:7:0:l`` as often as
    ``-R W:3:0:l``, and BART filters the bit out with ``md_nontriv_dims`` --
    so it is dropped rather than refused.
    """
    return tuple(-1 - bit for bit in range(ndim) if flags & (1 << bit))


def regularizer(text: str, ndim: int):
    """One ``-R`` argument as a :mod:`bartorch.priors` term over an ``ndim``-axis image.

    Raises
    ------
    Unsupported
        A letter this package does not offer, or a field that will not parse.
    """
    from bartorch import priors

    letter, _, fields = text.partition(":")
    if letter not in _TERMS:
        raise Unsupported(f"-R {letter} is not a term this package offers")
    name, layout = _TERMS[letter]
    parts = fields.split(":") if fields else []
    if len(parts) != len(layout):
        raise Unsupported(f"-R {letter} takes {len(layout)} fields, got {len(parts)}")

    made = dict(zip(layout, parts))
    term = getattr(priors, name)
    try:
        if layout == "xjw":
            return term(axes(int(made["x"]), ndim), float(made["w"]), axes(int(made["j"]), ndim))
        if layout == "xjk":
            return term(axes(int(made["x"]), ndim), int(made["k"]), axes(int(made["j"]), ndim))
        if layout == "jw":
            return term(float(made["w"]), axes(int(made["j"]), ndim))
        if layout == "w":
            return term(float(made["w"]))
        return term()
    except (TypeError, ValueError) as problem:
        raise Unsupported(f"-R {text}: {problem}") from problem
