"""Python keywords for BART options, looked up in the catalogue.

A keyword is the option's long name with underscores for hyphens, or its letter
when it has none.  The command-line spelling is always looked up rather than
derived from the keyword, because BART's long names mix hyphens and
underscores.
"""

from __future__ import annotations

from functools import lru_cache

from bartorch._catalogue import BART_VERSION, COMMANDS, Option

__all__ = ["HELP_FLAGS", "describe", "flag_for", "options_by_name", "python_name"]

#: Flags BART answers with its usage and ``exit``, which in this process would
#: end the interpreter.  :func:`describe` answers from the catalogue instead.
HELP_FLAGS = frozenset({"-h", "--help", "-?"})


def python_name(option: Option) -> str:
    """The keyword an option is given in Python.

    The long name where BART has one, with hyphens made underscores because a
    hyphen cannot be a keyword; otherwise the single letter, which stays as it
    is -- ``-R`` and ``-r`` are different options in ``pics`` and lowering the
    case would collide them.
    """
    if option.long:
        return option.long.replace("-", "_")
    return option.short


@lru_cache(maxsize=None)
def options_by_name(name: str) -> dict[str, Option]:
    """Every option of one command, by the keyword it is given in Python.

    A command whose long and short names collide -- one option called ``e``
    and another called ``e`` the long way -- keeps both, the long one under
    its own name and the short one under the letter, because the letter is
    always available as a fallback.
    """
    command = COMMANDS.get(name)
    if command is None:
        return {}
    table: dict[str, Option] = {}
    for option in command.options:
        table.setdefault(python_name(option), option)
    # The letter is a second way in wherever it is not already taken.
    for option in command.options:
        if option.short:
            table.setdefault(option.short, option)
    return table


def flag_for(command: str, keyword: str) -> str | None:
    """BART's spelling of the option ``keyword`` names, or None if the catalogue has none."""
    option = options_by_name(command).get(keyword)
    return option.flag if option is not None else None


def check(command: str, keywords) -> None:
    """Raise ValueError for an option ``command`` does not have, before BART sees it.

    BART answers an unknown option with ``error``, after which the next call into
    the library spins.  A command with no catalogue entry is not checked.
    """
    known = options_by_name(command)
    if not known:
        return
    for keyword in keywords:
        if keyword in known:
            continue
        # `R_1` and `R_2` are both `-R`; `flag_3` is `-3`.
        stem, _, suffix = keyword.rpartition("_")
        if stem and suffix.isdigit() and (stem in known or stem == "flag"):
            continue
        near = _closest(keyword, known)
        suggestion = f"; did you mean {near}=?" if near else ""
        raise ValueError(
            f"bart {command} has no option called {keyword!r}{suggestion}  "
            f"bartorch._options.describe({command!r}) lists the ones it has."
        )


def _closest(keyword: str, known: dict[str, Option]) -> str | None:
    import difflib

    matches = difflib.get_close_matches(keyword, [k for k in known if len(k) > 1], n=1)
    return matches[0] if matches else None


def describe(name: str) -> str:
    """What ``bart <name> -h`` prints, from the catalogue.

    BART's own help ends in ``exit``, which in this process would end the
    interpreter.

    Parameters
    ----------
    name : str
        A BART command.

    Returns
    -------
    str
        Its description, the arrays and values it takes, and every option with both
        spellings and BART's one-line description.

    Examples
    --------
    >>> print(describe("pics"))
    pics -- Parallel-imaging compressed-sensing reconstruction.
    ...
    """
    command = COMMANDS.get(name)
    if command is None:
        raise KeyError(f"BART {BART_VERSION} has no command called {name!r}")

    lines = [f"{command.name} -- {command.help.strip()}", ""]

    if command.inputs or command.values:
        lines.append("Takes")
        for argument in command.arguments:
            if argument.kind == "OUTFILE":
                continue
            what = "array" if argument.is_array else argument.kind.lower()
            tail = "" if argument.required else "  (optional)"
            lines.append(f"    {argument.name:<24}{what}{tail}")
        lines.append("")
    if command.outputs:
        lines.append("Returns")
        for argument in command.outputs:
            lines.append(f"    {argument.name:<24}array")
        lines.append("")

    choices = command.choices()
    grouped = {id(o) for arms in choices.values() for o in arms}
    if command.options:
        lines.append("Options")
        for option in command.options:
            if id(option) in grouped:
                continue
            lines.append(f"    {_spelling(option):<28}{option.help.strip()}")
        for arms in choices.values():
            names = ", ".join(python_name(o) for o in arms)
            lines.append(f"    one of: {names}")
            for option in arms:
                lines.append(f"        {_spelling(option):<24}{option.help.strip()}")
    return "\n".join(lines).rstrip() + "\n"


def _spelling(option: Option) -> str:
    """How an option is written in Python, with the value it takes.

    A letter BART also gives a word to is shown beside it, because that letter
    is what BART's own documentation and everyone's scripts use.
    """
    written = python_name(option)
    if option.takes_value:
        written += f"={option.arg or 'value'}"
    if option.short and option.long:
        written += f"  (-{option.short})"
    return written
