#!/usr/bin/env python3
"""Generate ``src/bartorch/_catalogue.py`` from BART's sources.

Reads each command's argument and option tables: both spellings, the kind of
value, and BART's help text.  Run after updating BART::

    python scripts/gen_catalogue.py

``tests/test_catalogue.py`` fails when the checked-in file differs from this output.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
BART_SRC = ROOT / "external" / "bart" / "src"
OUTPUT = ROOT / "src" / "bartorch" / "_catalogue.py"

LINE_LENGTH = 100


class Unknown(Exception):
    """A macro the reader has not been taught."""


# --- what each macro's arguments mean --------------------------------------
#
# From external/bart/src/misc/opts.h.  The index of each field in the macro's argument
# list; a field the macro does not have is absent.  `c` is the short option
# character, `s` the long name, `argname` the metavar BART prints in its help,
# and `value` the constant a SELECT writes.

OPTION_MACROS: dict[str, dict[str, int]] = {
    # OPT_SET(c, ptr, descr) -- a flag that takes no value.
    "OPT_SET": {"c": 0, "descr": 2, "kind": -1},
    "OPT_CLEAR": {"c": 0, "descr": 2, "kind": -1},
    # OPTL_SET(c, s, ptr, descr)
    "OPTL_SET": {"c": 0, "s": 1, "descr": 3, "kind": -1},
    "OPTL_CLEAR": {"c": 0, "s": 1, "descr": 3, "kind": -1},
    # OPT_SELECT(c, T, ptr, value, descr) -- one arm of a choice.
    "OPT_SELECT": {"c": 0, "ptr": 2, "value": 3, "descr": 4, "kind": -1},
    # OPTL_SELECT(c, s, T, ptr, value, descr)
    "OPTL_SELECT": {"c": 0, "s": 1, "ptr": 3, "value": 4, "descr": 5, "kind": -1},
    # OPTL_SELECT_DEF(c, s, T, ptr, value, def, descr)
    "OPTL_SELECT_DEF": {"c": 0, "s": 1, "ptr": 3, "value": 4, "descr": 6, "kind": -1},
    # OPT_SUBOPT(c, argname, descr, NR, opts) -- a comma-separated sub-table.
    "OPT_SUBOPT": {"c": 0, "argname": 1, "descr": 2, "kind": -1},
    # OPTL_SUBOPT(c, s, argname, descr, NR, opts)
    "OPTL_SUBOPT": {"c": 0, "s": 1, "argname": 2, "descr": 3, "kind": -1},
    # OPTL_SUBOPT2(c, s, argname, descr, descr1, NR, opts)
    "OPTL_SUBOPT2": {"c": 0, "s": 1, "argname": 2, "descr": 4, "kind": -1},
}

#: The typed options, whose macro name after the prefix is the kind.
#: OPT_<KIND>(c, ptr, argname, descr) and OPTL_<KIND>(c, s, ptr, argname, descr).
TYPED_KINDS = (
    "INT UINT LONG ULONG ULLONG PINT FLOAT DOUBLE STRING "
    "INFILE OUTFILE INOUTFILE CFL "
    "VEC2 VEC3 VECN FLVEC2 FLVEC3 FLVEC4 FLVECN DOVEC3 DOVECN"
).split()

#: The two that take a variable number of values and so have no metavar of
#: their own -- BART prints `[f:]*f` for them.  One argument fewer than the
#: rest, and reading them like the rest puts the description in the metavar.
VARIADIC_KINDS = {"VECN", "FLVECN", "DOVECN"}

for _kind in TYPED_KINDS:
    if _kind in VARIADIC_KINDS:
        OPTION_MACROS[f"OPT_{_kind}"] = {"c": 0, "descr": 2, "kind": _kind}
        OPTION_MACROS[f"OPTL_{_kind}"] = {"c": 0, "s": 1, "descr": 3, "kind": _kind}
    else:
        OPTION_MACROS[f"OPT_{_kind}"] = {"c": 0, "argname": 2, "descr": 3, "kind": _kind}
        OPTION_MACROS[f"OPTL_{_kind}"] = {"c": 0, "s": 1, "argname": 3, "descr": 4, "kind": _kind}

#: ARG_<KIND>(required, ptr, argname).  ARG_TUPLE is variadic and is recorded
#: by its kind alone; nothing in the wrappers takes one yet.
ARGUMENT_KINDS = ("INFILE OUTFILE INOUTFILE CFL INT LONG ULONG FLOAT STRING VEC3 TUPLE").split()


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


#: What a C escape means in a one-line description: a break is a space.
ESCAPES = {"n": " ", "t": " ", "r": " ", "f": " ", "v": " ", "b": ""}


def unescape(body: str) -> str:
    """The text of one C string literal's body."""
    return re.sub(r"\\(.)", lambda m: ESCAPES.get(m.group(1), m.group(1)), body)


def joined_string(text: str) -> str:
    """Adjacent C string literals, which the compiler concatenates into one."""
    parts = re.findall(r'"((?:[^"\\]|\\.)*)"', text)
    return "".join(unescape(part) for part in parts).strip()


def split_arguments(text: str) -> list[str]:
    """Split a macro's argument list on the commas at depth zero."""
    out: list[str] = []
    depth = 0
    quoted = False
    escaped = False
    current = ""
    for ch in text:
        if escaped:
            current += ch
            escaped = False
            continue
        if ch == "\\" and quoted:
            current += ch
            escaped = True
            continue
        if ch == '"':
            quoted = not quoted
        if not quoted:
            if ch in "([{":
                depth += 1
            elif ch in ")]}":
                depth -= 1
            elif ch == "," and depth == 0:
                out.append(current)
                current = ""
                continue
        current += ch
    if current.strip():
        out.append(current)
    return out


def macro_calls(block: str, names) -> list[tuple[str, list[str]]]:
    """Every ``NAME(...)`` in *block* whose name is in *names*, with its arguments."""
    found: list[tuple[str, list[str]]] = []
    for match in re.finditer(r"\b([A-Z][A-Z0-9_]*)\s*\(", block):
        name = match.group(1)
        if name not in names:
            continue
        depth = 0
        start = match.end() - 1
        for index in range(start, len(block)):
            if block[index] == "(":
                depth += 1
            elif block[index] == ")":
                depth -= 1
                if depth == 0:
                    found.append((name, split_arguments(block[start + 1 : index])))
                    break
    return found


def table(code: str, declaration: str) -> str:
    """The body of one of BART's `{ ... }` tables, or an empty string."""
    match = re.search(declaration + r"\s*=\s*\{", code)
    if not match:
        return ""
    depth = 0
    start = match.end() - 1
    for index in range(start, len(code)):
        if code[index] == "{":
            depth += 1
        elif code[index] == "}":
            depth -= 1
            if depth == 0:
                return code[start + 1 : index]
    return ""


def read_option(name: str, args: list[str]) -> dict:
    """One option macro as a record."""
    fields = OPTION_MACROS[name]

    def field(key: str) -> str:
        index = fields.get(key)
        return args[index] if index is not None and index < len(args) else ""

    short = field("c").strip()
    short = short[1:-1] if short.startswith("'") and short.endswith("'") else ""
    if name.startswith("OPTL") and short in ("0", ""):
        short = ""

    kind = fields["kind"]
    if kind == -1:
        kind = name.removeprefix("OPTL_").removeprefix("OPT_")

    record = {
        "short": short,
        "long": joined_string(field("s")),
        "kind": kind,
        "arg": joined_string(field("argname")),
        "help": joined_string(field("descr")),
    }
    if kind == "SELECT":
        # A choice is several options writing one variable; the variable is
        # what groups them, and the value is which arm this one is.
        record["group"] = field("ptr").strip()
        record["value"] = field("value").strip()
    return record


#: The fields of an option BART writes out as a brace initialiser rather than
#: through a macro, from `struct opt_s` in misc/opts.h:
#: { char, long, arg_required, type, conv, ptr, argname, descr }
SPECIAL_FIELDS = {"c": 0, "s": 1, "argname": 6, "descr": 7}


def read_special(entry: str) -> dict | None:
    """One hand-written option table entry, or ``None`` if it cannot be read.

    `pics -R` is one of these -- the generalized regularization option, and the
    one people reach for most -- so they are worth reading even though they are
    not macros.
    """
    args = split_arguments(entry.strip().removeprefix("{").removesuffix("}"))
    if len(args) <= max(SPECIAL_FIELDS.values()):
        return None

    def field(key: str) -> str:
        return args[SPECIAL_FIELDS[key]]

    short = field("c").strip()
    short = short[1:-1] if short.startswith("'") and short.endswith("'") else ""
    long = joined_string(field("s"))
    if not short and not long:
        # The character is an expression rather than a literal -- `nlinv` picks
        # one of two depending on the compatibility version -- and there is no
        # word to fall back on.
        return None
    return {
        "short": short,
        "long": long,
        "kind": "SPECIAL",
        "arg": joined_string(field("argname")),
        "help": joined_string(field("descr")),
    }


def read_argument(name: str, args: list[str]) -> dict:
    """One positional-argument macro as a record."""
    kind = name.removeprefix("ARG_")
    required = args[0].strip() == "true" if args else False
    argname = joined_string(args[2]) if len(args) > 2 else ""
    if kind == "TUPLE":
        # ARG_TUPLE(required, ptr, N, ...) repeats a group; the name is the
        # first of the repeated entries.
        argname = joined_string(" ".join(args[3:])) or "tuple"
    return {"name": argname, "kind": kind, "required": required}


def read_command(name: str, source: Path) -> dict:
    """One BART tool, as its own source declares it."""
    code = strip_comments(source.read_text(errors="replace"))

    helped = re.search(r"help_str\[\]\s*=\s*((?:\s*\"(?:[^\"\\]|\\.)*\")+)", code)
    arguments = [
        read_argument(macro, args)
        for macro, args in macro_calls(
            table(code, r"struct\s+arg_s\s+args\s*\[\s*\]"),
            {f"ARG_{k}" for k in ARGUMENT_KINDS},
        )
    ]
    options: list[dict] = []
    unread: list[str] = []
    for entry in split_arguments(table(code, r"struct\s+opt_s\s+opts\s*\[\s*\]")):
        entry = entry.strip()
        if not entry or entry == "OPT_END":
            continue
        call = re.match(r"^([A-Z][A-Z0-9_]*)\s*\((.*)\)$", entry, re.S)
        if call and call.group(1) in OPTION_MACROS:
            options.append(read_option(call.group(1), split_arguments(call.group(2))))
        elif entry.startswith("{"):
            special = read_special(entry)
            if special is not None:
                options.append(special)
            else:
                unread.append(re.sub(r"\s+", " ", entry)[:60])
        else:
            unread.append(re.sub(r"\s+", " ", entry)[:60])

    return {
        "name": name,
        "help": joined_string(helped.group(1)) if helped else "",
        "arguments": arguments,
        "options": options,
        "unread": unread,
    }


def discover(bart_src: Path) -> list[tuple[str, Path]]:
    """Every tool BART builds, by the ``main_<name>`` it defines."""
    found = []
    for path in sorted(bart_src.glob("*.c")):
        match = re.search(r"^int main_(\w+)", path.read_text(errors="replace"), re.M)
        if match:
            found.append((match.group(1), path))
    return sorted(found)


def parse(bart_src: Path = BART_SRC) -> dict[str, dict]:
    if not (bart_src / "bart.c").exists():
        raise Unknown(f"BART sources not found at {bart_src}; run `git submodule update --init`")
    return {name: read_command(name, path) for name, path in discover(bart_src)}


# --- rendering --------------------------------------------------------------


def literal(value) -> str:
    return repr(value)


def call(name: str, parts: list[str], indent: int) -> list[str]:
    """One constructor call, on one line where it fits and broken up where it does not.

    A help string BART wrote can be longer than any line length on its own, so
    the break is where it helps and the file is exempt from the line-length
    lint rather than pretending otherwise.
    """
    pad = " " * indent
    one = f"{pad}{name}({', '.join(parts)}),"
    if len(one) <= LINE_LENGTH:
        return [one]
    inner = " " * (indent + 4)
    return [f"{pad}{name}("] + [f"{inner}{part}," for part in parts] + [f"{pad}),"]


def render(commands: dict[str, dict], bart_version: str) -> str:
    lines = [
        '"""Every BART command, as BART\'s own sources declare it.',
        "",
        "Generated from the BART submodule by ``scripts/gen_catalogue.py``.",
        "Do not edit: run the generator instead.  ``tests/test_catalogue.py`` fails",
        "when this file is not what the sources produce.",
        "",
        "This is a description of BART and not a Python API.  What each option is",
        "called on the command line, what kind of value it takes, and what BART's own",
        "help says about it -- the wrappers in this package are written by hand",
        "against it, so that a name in Python is chosen rather than transliterated,",
        "and so that a command this package does not wrap is still reachable and",
        "still countable.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "from dataclasses import dataclass",
        "",
        "#: The BART the catalogue was read from.",
        f"BART_VERSION = {literal(bart_version)}",
        "",
        "",
        "@dataclass(frozen=True)",
        "class Argument:",
        '    """One positional argument, in the order BART takes it."""',
        "",
        "    #: What BART calls it in its own help.",
        "    name: str",
        "    #: BART's ``ARG_`` kind: INFILE, OUTFILE, INOUTFILE, INT, CFL, ...",
        "    kind: str",
        "    #: Whether the command refuses to run without it.",
        "    required: bool",
        "",
        "    @property",
        "    def is_array(self) -> bool:",
        '        """Whether it is an array rather than a number or a string.',
        "",
        "        ``ARG_CFL`` is not one: it is a complex scalar read from the",
        "        command line, as in ``bart scale <factor> <input> <output>``.",
        '        """',
        '        return self.kind in ("INFILE", "OUTFILE", "INOUTFILE")',
        "",
        "",
        "@dataclass(frozen=True)",
        "class Option:",
        '    """One named option.',
        "",
        "    BART spells an option with a letter, a word, or both, and this keeps",
        "    whichever it has: a wrapper that wants a readable name has one wherever",
        "    BART does, and the wire spelling is still here for the ones it does not.",
        '    """',
        "",
        "    #: The single character, or an empty string for a long-only option.",
        "    short: str",
        "    #: The long name, or an empty string for a short-only option.",
        "    long: str",
        "    #: SET and CLEAR take no value; SELECT is one arm of a choice; SUBOPT",
        "    #: takes a comma-separated sub-table; the rest name a value's type.",
        "    kind: str",
        "    #: The metavar BART prints, where it takes a value.",
        "    arg: str",
        "    #: BART's own one-line description.",
        "    help: str",
        "    #: For a SELECT, the variable the arms share and the value this one",
        "    #: writes -- which is what turns a run of them into one choice.",
        "    group: str = ''",
        "    value: str = ''",
        "",
        "    @property",
        "    def flag(self) -> str:",
        '        """How it is spelled on the command line."""',
        '        return f"-{self.short}" if self.short else f"--{self.long}"',
        "",
        "    @property",
        "    def takes_value(self) -> bool:",
        '        return self.kind not in ("SET", "CLEAR", "SELECT")',
        "",
        "",
        "@dataclass(frozen=True)",
        "class Command:",
        '    """One BART tool."""',
        "",
        "    name: str",
        "    #: BART's own one-line description.",
        "    help: str",
        "    arguments: tuple[Argument, ...] = ()",
        "    options: tuple[Option, ...] = ()",
        "",
        "    @property",
        "    def inputs(self) -> tuple[Argument, ...]:",
        '        """The arrays it reads."""',
        '        reads = ("INFILE", "INOUTFILE")',
        "        return tuple(a for a in self.arguments if a.kind in reads)",
        "",
        "    @property",
        "    def outputs(self) -> tuple[Argument, ...]:",
        '        """The arrays it writes."""',
        '        return tuple(a for a in self.arguments if a.kind in ("OUTFILE", "INOUTFILE"))',
        "",
        "    @property",
        "    def values(self) -> tuple[Argument, ...]:",
        '        """The positional arguments that are not arrays."""',
        "        return tuple(a for a in self.arguments if not a.is_array)",
        "",
        "    def choices(self) -> dict[str, tuple[Option, ...]]:",
        '        """The SELECT options, grouped by the variable they share."""',
        "        groups: dict[str, list[Option]] = {}",
        "        for option in self.options:",
        '            if "SELECT" == option.kind:',
        "                groups.setdefault(option.group, []).append(option)",
        "        return {k: tuple(v) for k, v in groups.items()}",
        "",
        "",
        "COMMANDS: dict[str, Command] = {",
    ]

    for name, command in commands.items():
        lines.append(f"    {literal(name)}: Command(")
        lines.append(f"        name={literal(name)},")
        lines.append(f"        help={literal(command['help'])},")
        if command["arguments"]:
            lines.append("        arguments=(")
            for argument in command["arguments"]:
                lines.extend(
                    call(
                        "Argument",
                        [
                            literal(argument["name"]),
                            literal(argument["kind"]),
                            literal(argument["required"]),
                        ],
                        indent=12,
                    )
                )
            lines.append("        ),")
        if command["options"]:
            lines.append("        options=(")
            for option in command["options"]:
                parts = [
                    literal(option["short"]),
                    literal(option["long"]),
                    literal(option["kind"]),
                    literal(option["arg"]),
                    literal(option["help"]),
                ]
                if option.get("group") or option.get("value"):
                    parts.append(literal(option.get("group", "")))
                    parts.append(literal(option.get("value", "")))
                lines.extend(call("Option", parts, indent=12))
            lines.append("        ),")
        lines.append("    ),")

    lines.append("}")
    lines.append("")
    lines.append("#: Entries in a command's option table that the reader could not")
    lines.append("#: make sense of, by command.  Nothing is dropped quietly: a construct")
    lines.append("#: BART starts using shows up here and fails a test rather than")
    lines.append("#: leaving an option missing with nothing to say so.")
    lines.append("UNREAD: dict[str, tuple[str, ...]] = {")
    for name, command in commands.items():
        if command.get("unread"):
            lines.append(f"    {literal(name)}: (")
            for entry in command["unread"]:
                lines.append(f"        {literal(entry)},")
            lines.append("    ),")
    lines.append("}")
    lines.append("")
    lines.append("#: Every command BART builds a tool for.")
    lines.append("NAMES = tuple(COMMANDS)")
    lines.append("")
    return "\n".join(lines)


def bart_version(bart_root: Path) -> str:
    version = bart_root / "version.txt"
    if version.exists():
        return version.read_text().strip()
    return "unknown"


def generate(bart_src: Path = BART_SRC) -> str:
    return render(parse(bart_src), bart_version(bart_src.parent))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if the output is out of date")
    args = ap.parse_args()

    text = generate()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            print(
                f"{OUTPUT} is out of date; run python scripts/gen_catalogue.py",
                file=sys.stderr,
            )
            return 1
        return 0
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT.relative_to(ROOT)}: {len(parse())} commands")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
