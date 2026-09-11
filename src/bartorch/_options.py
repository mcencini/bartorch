"""What a BART option is called in Python.

It sits beside the catalogue rather than under ``tools/`` because
``core.graph`` needs it to build an argument vector, and ``tools`` is built on
``core.graph``.

BART spells an option with a letter, a word, or both.  A wrapper wants the
word wherever there is one -- ``lowmem`` rather than ``U`` -- and the letter
only where there is nothing else.  What it must never do is *derive* the
command-line spelling back from the Python name: BART writes some long names
with hyphens and others with underscores, and a rule that turns one into the
other cannot be right for both.  So the flag is looked up rather than
reconstructed, and this module is the lookup.
"""

from __future__ import annotations

from functools import lru_cache

from bartorch._catalogue import BART_VERSION, COMMANDS, Option

__all__ = ["HELP_FLAGS", "describe", "flag_for", "options_by_name", "python_name"]

#: What BART treats as a request for help.  A tool that is given one prints
#: its usage and calls ``exit``, and BART is in this process, so the exit is
#: the interpreter's: `run_command(["pics", "-h"])` used to end the session
#: without a message.  :func:`describe` answers the same question from the
#: catalogue instead.
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
    """How BART spells the option *keyword* names, or ``None`` if it has no such option.

    ``None`` is not an error: a caller may be passing a flag through to a
    command this package has no catalogue entry for, and the old guess is
    better than nothing there.
    """
    option = options_by_name(command).get(keyword)
    return option.flag if option is not None else None


def check(command: str, keywords) -> None:
    """Refuse a flag the command does not have, before BART is asked.

    Not a courtesy.  BART answers an option it does not recognise by printing
    its usage and calling ``error``, which its own catcher turns into a return
    code -- and leaves the library in a state where the *next* tool call spins
    forever at full CPU.  One typo would end the session, so a flag the
    catalogue has no entry for never reaches BART.

    A command the catalogue does not know is not checked: there is nothing to
    check against.
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
            f"bartorch.tools.describe({command!r}) lists the ones it has."
        )


def _closest(keyword: str, known: dict[str, Option]) -> str | None:
    """The option a keyword was most likely meant to be."""
    import difflib

    matches = difflib.get_close_matches(keyword, [k for k in known if len(k) > 1], n=1)
    return matches[0] if matches else None


def describe(name: str) -> str:
    """What ``bart <name> -h`` would print, from the catalogue rather than from BART.

    BART's own help ends in ``exit``, which in this process is the
    interpreter's, so the question is answered here.

    Parameters
    ----------
    name : str
        A BART command.

    Returns
    -------
    str
        Its description, the arrays and values it takes, and every option with
        both spellings and BART's own one-line description of each.

    Examples
    --------
    >>> print(bartorch.tools.describe("pics"))
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
