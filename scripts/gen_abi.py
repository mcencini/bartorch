#!/usr/bin/env python3
"""Generate ``src/bartorch/_abi.py``, the ctypes signatures, from ``src/csrc/include/bartorch.h``.

The header is plain C -- no complex types, variable-length arrays or GNU
extensions -- which is what this parser handles.  Run after changing it::

    python scripts/gen_abi.py

``tests/test_abi.py`` fails when the checked-in file differs from this output.
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
HEADER = ROOT / "src" / "csrc" / "include" / "bartorch.h"
OUTPUT = ROOT / "src" / "bartorch" / "_abi.py"

#: The line length the project lints at, so the generated file passes too.
LINE_LENGTH = 100

# --- the C types the header uses ------------------------------------------
#
# Every one of these appears in bartorch.h; a type that does not is a type the
# generator has not been taught, and it stops rather than guessing.  Pointers
# to the opaque operator structs are void pointers here: the host never reads
# through them.

SCALARS = {
    "void": "None",
    "int": "ctypes.c_int",
    "long": "ctypes.c_long",
    "float": "ctypes.c_float",
    "double": "ctypes.c_double",
    "size_t": "ctypes.c_size_t",
    "unsigned long": "ctypes.c_ulong",
    "unsigned int": "ctypes.c_uint",
}

POINTERS = {
    "char*": "ctypes.c_char_p",
    "char**": "ctypes.POINTER(ctypes.c_char_p)",
    "void*": "ctypes.c_void_p",
    "void**": "ctypes.POINTER(ctypes.c_void_p)",
    "long*": "ctypes.POINTER(ctypes.c_long)",
    "int*": "ctypes.POINTER(ctypes.c_int)",
    "float*": "ctypes.POINTER(ctypes.c_float)",
    "double*": "ctypes.POINTER(ctypes.c_double)",
    "bartorch_linop*": "ctypes.c_void_p",
    "bartorch_linop**": "ctypes.POINTER(ctypes.c_void_p)",
    "bartorch_nlop*": "ctypes.c_void_p",
    "bartorch_nlop**": "ctypes.POINTER(ctypes.c_void_p)",
    "bartorch_noir*": "ctypes.c_void_p",
    "bartorch_noir_net*": "ctypes.c_void_p",
    "bartorch_prox*": "ctypes.c_void_p",
    "bartorch_prox**": "ctypes.POINTER(ctypes.c_void_p)",
}


class Unknown(Exception):
    """A construct the generator has not been taught to read."""


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//[^\n]*", " ", text)


def strip_directives(text: str) -> str:
    """Drop the preprocessor lines, including the two that define BARTORCH_API."""
    return "\n".join(line for line in text.splitlines() if not line.lstrip().startswith("#"))


def normalise(decl: str) -> str:
    """Collapse whitespace and drop the qualifiers ctypes has no use for."""
    decl = re.sub(r"\s+", " ", decl).strip()
    decl = re.sub(r"\bconst\b", "", decl)
    decl = re.sub(r"\bstruct\b", "", decl)
    return re.sub(r"\s+", " ", decl).strip()


#: Words that begin a type rather than name a parameter.
TYPE_WORDS = {"void", "int", "long", "float", "double", "size_t", "unsigned", "char"}


def ctype(decl: str, callbacks: dict[str, str]) -> str:
    """The ctypes spelling of one C type, with any parameter name already gone."""
    decl = normalise(decl)
    if decl in callbacks:
        return callbacks[decl]
    stars = decl.count("*")
    base = re.sub(r"\s+", " ", decl.replace("*", " ")).strip()
    if stars == 0:
        if base not in SCALARS:
            raise Unknown(f"scalar type {base!r}")
        return SCALARS[base]
    key = f"{base}{'*' * stars}"
    if key not in POINTERS:
        raise Unknown(f"pointer type {key!r}")
    return POINTERS[key]


def split_params(params: str) -> list[str]:
    """Split an argument list on the commas that are not inside parentheses."""
    out, depth, current = [], 0, ""
    for ch in params:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        if ch == "," and depth == 0:
            out.append(current)
            current = ""
        else:
            current += ch
    if current.strip():
        out.append(current)
    return out


def param_type(param: str, callbacks: dict[str, str]) -> str:
    """The ctypes spelling of one parameter, whose name is dropped.

    ``long dims[16]`` and ``const long* dims`` say the same thing to ctypes, so
    the brackets become a star and the stars are counted wherever they sit.
    """
    # An array is a pointer, and where the star ends up does not matter.
    param = re.sub(r"\[[^\]]*\]", "*", normalise(param))
    stars = param.count("*")
    words = param.replace("*", " ").split()
    if not words:
        raise Unknown("empty parameter")
    # The last word names the parameter unless it is part of the type.
    if len(words) > 1 and words[-1] not in TYPE_WORDS and words[-1] not in callbacks:
        words.pop()
    return ctype(" ".join(words) + "*" * stars, callbacks)


def field_type(decl: str) -> str:
    """The ctypes spelling of one structure field, whose name is dropped.

    Every pointer is ``c_void_p`` here rather than a typed pointer: the
    structure exists so that Python can fill it, and what it fills a pointer
    field with is the integer a tensor reports.
    """
    decl = re.sub(r"\[[^\]]*\]", "*", normalise(decl))
    if "*" in decl:
        return "ctypes.c_void_p"
    words = decl.split()
    if len(words) > 1 and words[-1] not in TYPE_WORDS:
        words.pop()
    base = " ".join(words)
    if base not in SCALARS:
        raise Unknown(f"field type {base!r}")
    return SCALARS[base]


def struct_name(c_name: str) -> str:
    """``bartorch_encoding`` as the generated class ``Encoding``."""
    return "".join(part.capitalize() for part in c_name.removeprefix("bartorch_").split("_"))


def parse(header: str) -> dict:
    """Read the header into the pieces the generated module is made of."""
    text = strip_comments(header)

    dims = re.search(r"#define\s+BARTORCH_DIMS\s+(\d+)", text)
    if not dims:
        raise Unknown("BARTORCH_DIMS")

    text = strip_directives(text)

    levels: list[tuple[str, int]] = []
    enum = re.search(r"enum\s+bartorch_log_level\s*\{(.*?)\}", text, re.S)
    if enum:
        for name, value in re.findall(r"(BARTORCH_LOG_\w+)\s*=\s*(\d+)", enum.group(1)):
            levels.append((name, int(value)))

    # Every other enum the header declares, so that a name Python has to
    # pass -- a transform, a counter -- is the header's name and not a number
    # written down twice.
    constants: list[tuple[str, int]] = []
    for name, body in re.findall(r"enum\s+(bartorch_\w+)\s*\{(.*?)\}", text, re.S):
        if "bartorch_log_level" == name:
            continue
        for member, value in re.findall(r"(BARTORCH_\w+)\s*=\s*(\d+)", body):
            constants.append((member, int(value)))

    # Structures a caller fills and passes by pointer.
    structs: list[tuple[str, str, list[tuple[str, str]]]] = []
    for name, body in re.findall(r"struct\s+(bartorch_\w+)\s*\{(.*?)\n\}\s*;", text, re.S):
        fields = []
        for field in body.split(";"):
            if not field.strip():
                continue
            member = normalise(field).replace("[", " [").split()[-1]
            member = re.sub(r"[*\[\]0-9]", "", member)
            fields.append((member, field_type(field)))
        structs.append((struct_name(name), name, fields))
        POINTERS[f"{name}*"] = f"ctypes.POINTER({struct_name(name)})"

    # Callback typedefs, in the order they appear, so a later signature can
    # name an earlier one.
    callbacks: dict[str, str] = {}
    emitted: list[tuple[str, str, list[str]]] = []
    for ret, name, params in re.findall(
        r"typedef\s+([\w\s*]+?)\s*\(\s*\*\s*(bartorch_\w+)\s*\)\s*\((.*?)\)\s*;", text, re.S
    ):
        py = name.removeprefix("bartorch_").upper()
        argtypes = [param_type(p, callbacks) for p in split_params(params)]
        emitted.append((py, ctype(ret, callbacks), argtypes))
        callbacks[name] = py

    functions: list[tuple[str, str, list[str]]] = []
    for decl in re.findall(r"BARTORCH_API\s+(.*?);", text, re.S):
        m = re.match(r"^(.*?)\b(bartorch_\w+)\s*\((.*)\)$", normalise(decl), re.S)
        if not m:
            raise Unknown(f"declaration {decl.strip()!r}")
        ret, name, params = m.group(1), m.group(2), m.group(3)
        # A `*` between the return type and the name belongs to the type.
        ret = ret.strip()
        argtypes = (
            []
            if normalise(params) in ("", "void")
            else [param_type(p, callbacks) for p in split_params(params)]
        )
        functions.append((name, ctype(ret, callbacks), argtypes))

    return {
        "dims": int(dims.group(1)),
        "levels": levels,
        "callbacks": emitted,
        "constants": constants,
        "structs": structs,
        "functions": functions,
    }


def render(spec: dict) -> str:
    lines = [
        '"""The C ABI, as ctypes sees it.',
        "",
        "Generated from ``src/csrc/include/bartorch.h`` by ``scripts/gen_abi.py``.",
        "Do not edit: run the generator instead.  ``tests/test_abi.py`` fails when",
        "this file is not what the header produces.",
        '"""',
        "",
        "from __future__ import annotations",
        "",
        "import ctypes",
        "",
        "#: Dimensions BART carries for every array.",
        f"DIMS = {spec['dims']}",
        "",
    ]

    if spec["levels"]:
        lines.append("#: BART's debug levels, by the header's own names.")
        lines.append("LOG_LEVELS = {")
        for name, value in spec["levels"]:
            lines.append(f'    "{name.removeprefix("BARTORCH_LOG_").lower()}": {value},')
        lines.append("}")
        lines.append("")

    if spec["constants"]:
        lines.append("#: The header's own enumerators, by their own names.")
        for name, value in spec["constants"]:
            lines.append(f"{name} = {value}")
        lines.append("")
        lines.append("")

    for name, c_name, fields in spec["structs"]:
        lines.append(f"class {name}(ctypes.Structure):")
        lines.append(f'    """``struct {c_name}``, field for field."""')
        lines.append("")
        lines.append("    _fields_ = [")
        for member, ctype_name in fields:
            lines.append(f'        ("{member}", {ctype_name}),')
        lines.append("    ]")
        lines.append("")
        lines.append("")

    for name, restype, argtypes in spec["callbacks"]:
        args = "".join(f"\n    {a}," for a in argtypes)
        lines.append(f"{name} = ctypes.CFUNCTYPE(\n    {restype},{args}\n)")
    lines.append("")

    names = ",\n".join(f'    "{n}"' for n, _, _ in spec["functions"])
    lines.append("#: Every entry point the header exports.")
    lines.append(f"SYMBOLS = (\n{names},\n)")
    lines.append("")
    lines.append("")
    lines.append("def bind(lib: ctypes.CDLL) -> ctypes.CDLL:")
    lines.append('    """Give every entry point its signature, and return the library."""')
    for name, restype, argtypes in spec["functions"]:
        lines.append(f"    lib.{name}.restype = {restype}")
        one = f"    lib.{name}.argtypes = [{', '.join(argtypes)}]"
        if len(one) <= LINE_LENGTH:
            lines.append(one)
        else:
            lines.append(f"    lib.{name}.argtypes = [")
            for a in argtypes:
                lines.append(f"        {a},")
            lines.append("    ]")
    lines.append("    return lib")
    lines.append("")
    return "\n".join(lines)


def generate(header: Path = HEADER) -> str:
    return render(parse(header.read_text()))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="fail if the output is out of date")
    args = ap.parse_args()

    text = generate()
    if args.check:
        if not OUTPUT.exists() or OUTPUT.read_text() != text:
            print(f"{OUTPUT} is out of date; run python scripts/gen_abi.py", file=sys.stderr)
            return 1
        return 0
    OUTPUT.write_text(text)
    print(f"wrote {OUTPUT.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
