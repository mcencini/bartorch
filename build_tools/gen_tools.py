"""build_tools/gen_tools.py — Generate bartorch/tools/_generated.py.

Parses BART source C files (``bart/src/<tool>.c`` in the git submodule) to
extract each tool's positional arguments, named options, and description.
Produces a ``bartorch/tools/_generated.py`` with one properly typed Python
function per BART command.

The generated functions:

* Have named keyword arguments matching every BART ``-flag`` option.
* Carry Python type hints inferred from BART's opt-type macros.
* Include a NumPy-style docstring generated from BART's ``help_str`` and the
  option descriptions extracted from source.
* Are decorated with ``@bart_op`` so tensor inputs are normalised to
  ``complex64`` automatically.
* Accept ``**extra_flags`` for any BART option not covered by a named param.

Usage
-----
From the repository root::

    python build_tools/gen_tools.py [--bart-src PATH] [--out PATH]

``--bart-src`` defaults to ``bart/src`` relative to the repository root.
``--out`` defaults to ``bartorch/tools/_generated.py``.

If ``bart/src`` is absent the script aborts with an error — initialise the
submodule first::

    git submodule update --init --recursive

Generated file
--------------
``bartorch/tools/_generated.py`` is **committed to the repository** and tracks
the pinned BART submodule commit.  Re-run this script whenever the BART
submodule is updated to keep the generated wrappers in sync.
"""

from __future__ import annotations

import argparse
import keyword
import re
import sys
import textwrap
from pathlib import Path

# ---------------------------------------------------------------------------
# Tools with hand-written richer Python wrappers — skip in generated file
# ---------------------------------------------------------------------------

_SKIP_TOOLS: set[str] = set()

# ---------------------------------------------------------------------------
# Tool categories for Sphinx API grouping
# ---------------------------------------------------------------------------

_TOOL_CATEGORIES: dict[str, list[str]] = {
    "Fourier Transforms": ["fft", "fftmod", "fftshift"],
    "Simulation / Phantoms": ["phantom", "signal", "bloch"],
    "Sampling / Trajectories": ["traj", "poisson", "pattern", "upat", "estdelay"],
    "Non-Cartesian": ["nufft", "nufftbase"],
    "Coil Processing": ["coils", "cc", "ccapply", "walsh", "whiten", "rss"],
    "Calibration": ["ecalib", "caldir", "calmat", "ecaltwo", "estvar", "estdims"],
    "Image Processing": [
        "resize",
        "reshape",
        "crop",
        "flip",
        "circshift",
        "transpose",
        "repmat",
        "join",
        "split",
        "extract",
        "squeeze",
        "flatten",
        "copy",
        "slice",
        "fmac",
        "scale",
        "saxpy",
    ],
    "Wavelets / Sparsity": ["wavelet", "cdf97", "threshold", "tgv", "rof"],
    "Low-Rank / Subspace": ["svd", "lrmatrix", "casorati", "ssa"],
    "Iterative Algorithms": [
        "pics",
        "pocsense",
        "itsense",
        "sake",
        "nlinv",
        "moba",
        "mobafit",
        "irgnm",
    ],
    "Utilities": [
        "bitmask",
        "avg",
        "noise",
        "nrmse",
        "normalize",
        "std",
        "sdot",
        "zeros",
        "delta",
        "ones",
        "rand",
        "randn",
        "cabs",
        "carg",
        "creal",
        "conj",
        "cpyphs",
        "spow",
        "zexp",
        "poly",
        "calc",
        "invert",
        "window",
        "filter",
        "homodyne",
        "rmfreq",
        "mandelbrot",
    ],
}

# Tools whose BART implementation supports GPU via the -g flag
_GPU_CAPABLE_TOOLS: frozenset[str] = frozenset(
    {
        "nlinv",
        "moba",
        "mobafit",
        "nufft",
        "nufftbase",
        "pocsense",
        "itsense",
    }
)


def _cuda_note(tool_name: str, opts: list) -> str:
    """Return a Sphinx note about CUDA compatibility for *tool_name*."""
    # Check if the tool has a -g GPU flag in its opts
    has_gpu_flag = any(raw_key == "g" for _, _, raw_key, _, _ in opts)
    if tool_name in _GPU_CAPABLE_TOOLS or has_gpu_flag:
        return (
            "    .. note::\n"
            "        **CUDA:** GPU-capable when BART is compiled with\n"
            "        ``USE_CUDA=ON`` (pass ``g=True`` to enable).\n"
        )
    return (
        "    .. note::\n"
        "        **CUDA:** CPU only.  CUDA tensors are automatically moved to\n"
        "        CPU before dispatch and returned to the original device.\n"
    )


# ---------------------------------------------------------------------------
# BART OPT type → (Python annotation, default value string)
# ---------------------------------------------------------------------------

_OPT_TYPE_MAP: dict[str, tuple[str, str]] = {
    "bool": ("bool", "False"),
    "INT": ("int | None", "None"),
    "UINT": ("int | None", "None"),
    "LONG": ("int | None", "None"),
    "ULONG": ("int | None", "None"),
    "ULLONG": ("int | None", "None"),
    "PINT": ("int | None", "None"),
    "FLOAT": ("float | None", "None"),
    "DOUBLE": ("float | None", "None"),
    "STRING": ("str | None", "None"),
    "INFILE": ("str | None", "None"),
    "OUTFILE": ("str | None", "None"),
    "INOUTFILE": ("str | None", "None"),
    "VEC2": ("tuple[int, int] | None", "None"),
    "VEC3": ("tuple[int, int, int] | None", "None"),
    "VECN": ("tuple[int, ...] | None", "None"),
    "FLVEC2": ("tuple[float, float] | None", "None"),
    "FLVEC3": ("tuple[float, float, float] | None", "None"),
    "FLVEC4": ("tuple[float, float, float, float] | None", "None"),
    "FLVECN": ("tuple[float, ...] | None", "None"),
}

# ARG types that represent CFL / tensor inputs
_TENSOR_ARG_TYPES: frozenset[str] = frozenset({"INFILE", "OUTFILE", "INOUTFILE", "CFL"})

# ARG types that represent plain-value positional args (not tensors)
_VALUE_ARG_TYPE_MAP: dict[str, str] = {
    "ULONG": "int",
    "LONG": "int",
    "INT": "int",
    "UINT": "int",
    "FLOAT": "float",
    "STRING": "str",
    "VEC2": "tuple[int, int]",
    "VEC3": "tuple[int, int, int]",
}

# Python keywords and common builtins that need renaming
_PYTHON_RESERVED: frozenset[str] = frozenset(
    set(keyword.kwlist)
    | {
        "input",
        "type",
        "list",
        "dict",
        "set",
        "id",
        "map",
        "filter",
        "min",
        "max",
        "sum",
        "len",
        "all",
        "any",
        "zip",
        "range",
        "help",
    }
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _to_py_ident(s: str) -> str:
    """Convert a raw string into a valid Python identifier.

    Rules:
    * Replace hyphens and dots with underscores.
    * Remove any remaining non-alphanumeric/underscore characters.
    * Prefix digit-starting names with ``flag_``.
    * Append ``_`` to names that clash with Python keywords or common builtins.
    """
    s = s.replace("-", "_").replace(".", "_").replace("/", "_per_")
    s = re.sub(r"[^a-zA-Z0-9_]", "_", s)
    if s and s[0].isdigit():
        s = "flag_" + s
    if s in _PYTHON_RESERVED:
        s = s + "_"
    return s


def _strip_c_comments(text: str) -> str:
    """Remove C block comments and C++ line comments."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.DOTALL)
    text = re.sub(r"//[^\n]*", " ", text)
    return text


# ---------------------------------------------------------------------------
# BART C-source parser
# ---------------------------------------------------------------------------


def _parse_tool_source(tool_name: str, src_path: Path) -> dict:
    """Extract description, positional args, and named opts from a BART ``.c`` file.

    Returns a dict with keys:
    * ``name`` — tool name (str)
    * ``help`` — description (str, may be empty)
    * ``tensor_inputs`` — list of ``(name,)`` for ARG_INFILE / ARG_INOUTFILE
    * ``value_args``    — list of ``(py_type, name, description)`` for non-CFL
                          positional args (ARG_ULONG, ARG_INT, …)
    * ``opts``          — list of ``(py_type, py_key, orig_key, argname, descr)``
    * ``has_cfl_output``— True if the tool writes a CFL output file (ARG_OUTFILE)
    """
    content = src_path.read_text(errors="replace")
    code = _strip_c_comments(content)

    def _clean_c_str(s: str) -> str:
        """Clean a C string literal for use in a Python docstring.

        Processes escape sequences in a single pass to avoid double-escaping:
        - ``\\n``, ``\\r``, ``\\t``, ``\\f``, ``\\v`` → space
        - ``\\\\`` (C escaped backslash) → ``\\\\`` (escaped for Python docstring)
        - Any other ``\\X`` → ``\\\\X`` (escape the backslash for docstring safety)
        """

        def _replace(m: re.Match) -> str:
            c = m.group(1)
            if c in "nrtfv":
                return " "
            if c == "\\":
                return "\\\\"
            return "\\\\" + c

        return re.sub(r"\\(.)", _replace, s).strip().replace("|", "\\\\|")

    # 1. Description (help_str)
    m = re.search(
        r'static\s+const\s+char\s+help_str\[\]\s*=\s*"((?:[^"\\]|\\.)*?)"',
        code,
        re.DOTALL,
    )
    help_str = _clean_c_str(m.group(1)) if m else ""

    # 2. Positional args (struct arg_s args[] = { ... };)
    args_m = re.search(
        r"struct\s+arg_s\s+args\s*\[\s*\]\s*=\s*\{(.*?)\};",
        code,
        re.DOTALL,
    )
    tensor_inputs: list[tuple[str]] = []
    value_args: list[tuple[str, str, str]] = []
    n_outputs: int = 0
    if args_m:
        for m in re.finditer(
            r"ARG_(\w+)\s*\(\s*(true|false)\s*,\s*[^,]+,\s*\"([^\"]+)\"",
            args_m.group(1),
        ):
            kind, required, argname = m.group(1), m.group(2), m.group(3)
            if kind == "OUTFILE":
                # Every required output array is returned; an optional one
                # needs a flag the caller passes explicitly.
                if required == "true":
                    n_outputs += 1
            elif kind in _TENSOR_ARG_TYPES:
                tensor_inputs.append((argname,))
            elif kind in _VALUE_ARG_TYPE_MAP:
                py_type = _VALUE_ARG_TYPE_MAP[kind]
                value_args.append((py_type, _to_py_ident(argname), argname))

    # 3. Named options (const struct opt_s opts[] = { ... };)
    opts_m = re.search(
        r"const\s+struct\s+opt_s\s+opts\s*\[\s*\]\s*=\s*\{(.*?)\};",
        code,
        re.DOTALL,
    )
    opts: list[tuple[str, str, str, str, str]] = []
    seen_keys: set[str] = set()

    def _add_opt(py_type: str, raw_key: str, argname: str, descr: str) -> None:
        key = _to_py_ident(raw_key)
        if key in seen_keys or not key:
            return
        seen_keys.add(key)
        opts.append((py_type, key, raw_key, argname, _clean_c_str(descr)))

    # Helper: a regex fragment that matches a quoted C string allowing \" inside
    _CSTR = r'"((?:[^"\\]|\\.)*)"'

    if opts_m:
        block = opts_m.group(1)

        # OPT_SET / OPT_CLEAR  (char, ptr, descr)
        for m in re.finditer(
            r"OPT_(?:SET|CLEAR)\s*\(\s*'([^']+)'\s*,[^,]+,\s*" + _CSTR,
            block,
        ):
            _add_opt("bool", m.group(1), "", m.group(2))

        # OPT_<TYPE>  (char, ptr, argname, descr)
        for m in re.finditer(
            r"OPT_("
            r"INT|UINT|LONG|ULONG|ULLONG|PINT|FLOAT|DOUBLE|"
            r"STRING|INFILE|OUTFILE|INOUTFILE|"
            r"VEC2|VEC3|VECN|FLVEC2|FLVEC3|FLVEC4|FLVECN"
            r")\s*\(\s*'([^']+)'\s*,[^,]+,\s*" + _CSTR + r"\s*,\s*" + _CSTR,
            block,
        ):
            _add_opt(m.group(1), m.group(2), m.group(3), m.group(4))

        # OPTL_SET / OPTL_CLEAR  (char_or_0, long_name, ptr, descr)
        for m in re.finditer(
            r"OPTL_(?:SET|CLEAR)\s*\(\s*(?:'([^']+)'|0)\s*,\s*" + _CSTR + r"\s*,[^,]+,\s*" + _CSTR,
            block,
        ):
            char = m.group(1) or ""
            long_name = m.group(2)
            raw_key = char if char else long_name
            _add_opt("bool", raw_key, "", m.group(3))

        # OPTL_<TYPE>  (char_or_0, long_name, ptr, argname, descr)
        for m in re.finditer(
            r"OPTL_("
            r"INT|UINT|LONG|ULONG|ULLONG|PINT|FLOAT|DOUBLE|"
            r"STRING|INFILE|OUTFILE|INOUTFILE|"
            r"VEC2|VEC3|VECN|FLVEC2|FLVEC3|FLVEC4|FLVECN"
            r")\s*\(\s*(?:'([^']+)'|0)\s*,\s*"
            + _CSTR
            + r"\s*,[^,]+,\s*"
            + _CSTR
            + r"\s*,\s*"
            + _CSTR,
            block,
        ):
            char = m.group(2) or ""
            long_name = m.group(3)
            raw_key = char if char else long_name
            _add_opt(m.group(1), raw_key, m.group(4), m.group(5))

    return {
        "name": tool_name,
        "help": help_str,
        "tensor_inputs": tensor_inputs,
        "value_args": value_args,
        "opts": opts,
        "n_outputs": n_outputs,
    }


def _discover_tools(bart_src: Path) -> list[str]:
    """Return sorted list of BART tool names from ``main_<name>`` functions."""
    tools: list[str] = []
    if not bart_src.is_dir():
        return tools
    for path in sorted(bart_src.glob("*.c")):
        content = path.read_text(errors="replace")
        m = re.search(r"^int main_(\w+)", content, re.MULTILINE)
        if m:
            tools.append(m.group(1))
    return sorted(tools)


# ---------------------------------------------------------------------------
# Code generation
# ---------------------------------------------------------------------------


def _opt_type_hint(py_type: str) -> str:
    return _OPT_TYPE_MAP.get(py_type, ("Any", "None"))[0]


def _opt_default(py_type: str) -> str:
    return _OPT_TYPE_MAP.get(py_type, ("Any", "None"))[1]


def _generate_func(info: dict) -> str:
    """Return the Python source for one generated function."""
    name = info["name"]
    safe_name = name.replace("-", "_")
    tensor_inputs = info["tensor_inputs"]  # [(argname,), ...]
    value_args = info["value_args"]  # [(py_type, py_key, orig_name), ...]
    opts = info["opts"]  # [(py_type, py_key, orig_key, argname, descr), ...]
    help_str = info["help"]

    lines: list[str] = []

    # ---- decorator ----
    lines.append("@bart_op")

    # ---- signature ----
    sig_lines: list[str] = [f"def {safe_name}("]

    # positional tensor inputs
    for (argname,) in tensor_inputs:
        py_param = _to_py_ident(argname)
        sig_lines.append(f"    {py_param}: torch.Tensor,")

    # positional non-tensor args
    for py_type, py_key, _orig in value_args:
        sig_lines.append(f"    {py_key}: {py_type},")

    # keyword-only separator
    sig_lines.append("    *,")
    sig_lines.append("    output_dims: list[int] | None = None,")

    # named opt kwargs
    for py_type, py_key, _orig_key, _argname, _descr in opts:
        type_hint = _opt_type_hint(py_type)
        default = _opt_default(py_type)
        sig_lines.append(f"    {py_key}: {type_hint} = {default},")

    sig_lines.append("    **extra_flags: Any,")
    sig_lines.append(") -> torch.Tensor | tuple[torch.Tensor, ...]:")

    lines.extend(sig_lines)

    # ---- docstring ----
    doc_lines: list[str] = ['    """']
    if help_str:
        # Wrap long descriptions at 76 chars; preserve original punctuation
        wrapped = textwrap.fill(help_str, width=76, subsequent_indent="    ")
        doc_lines.append("    " + wrapped.lstrip())
    else:
        doc_lines.append(f"    Wraps BART's ``{name}`` command.")
    doc_lines.append("")
    doc_lines.append(f"    Equivalent to calling ``bart {name}`` with the given arguments.")
    doc_lines.append("    See the BART documentation for full details.")
    doc_lines.append("")
    doc_lines.append(_cuda_note(name, opts))
    doc_lines.append("    Parameters")
    doc_lines.append("    ----------")

    for (argname,) in tensor_inputs:
        py_param = _to_py_ident(argname)
        doc_lines.append(f"    {py_param} : torch.Tensor")
        doc_lines.append(f"        Input CFL array ``{argname}``.")

    for py_type, py_key, orig_name in value_args:
        doc_lines.append(f"    {py_key} : {py_type}")
        doc_lines.append(f"        Positional argument ``{orig_name}``.")

    doc_lines.append("    output_dims : list[int], optional")
    doc_lines.append("        Expected output shape; ``None`` to infer at runtime.")

    for py_type, py_key, orig_key, argname, descr in opts:
        type_hint = _opt_type_hint(py_type)
        default = _opt_default(py_type)
        flag_str = f"-{orig_key}" if len(orig_key) == 1 else f"--{orig_key}"
        if argname:
            flag_str += f" {argname}"
        doc_lines.append(f"    {py_key} : {type_hint}")
        wrapped_descr = textwrap.fill(
            f"        BART flag ``{flag_str}``: {descr.strip()} Default ``{default}``.",
            width=79,
            subsequent_indent="        ",
        )
        doc_lines.append(wrapped_descr)

    doc_lines.append("    **extra_flags : Any")
    doc_lines.append(f"        Additional BART ``{name}`` flags forwarded directly.")
    doc_lines.append("")
    doc_lines.append("    Returns")
    doc_lines.append("    -------")
    doc_lines.append("    torch.Tensor")
    doc_lines.append("        Result array in C-order ``complex64``.")
    doc_lines.append('    """')

    lines.extend(doc_lines)

    # ---- body ----
    tensor_arg_names = [_to_py_ident(a[0]) for a in tensor_inputs]
    inputs_expr = "[" + ", ".join(tensor_arg_names) + "]"

    # Positional scalar args (e.g. bitmask for fft) go in _pos=[…] so that
    # run() can insert them into argv *without* a -flag prefix.
    pos_expr = "[" + ", ".join(py_key for _, py_key, _ in value_args) + "]"

    # Build the explicit kwargs dict passed to dispatch.
    # Bool opts: pass as `True` only when set (truthy check).
    # Optional opts: pass through as-is (None is ignored by dispatch).
    kw_parts: list[str] = []
    for py_type, py_key, orig_key, _argname, _descr in opts:
        if py_type == "bool":
            kw_parts.append(f"{py_key}={py_key} or None")
        else:
            kw_parts.append(f"{py_key}={py_key}")

    kw_str = ", ".join(kw_parts)
    pos_arg = f"_pos={pos_expr}, " if value_args else ""
    # A tool without an output array returns its printed text: output_dims
    # is passed as False so no output name is appended to argv.  A tool with
    # several output arrays returns a tuple.
    n_outputs = info.get("n_outputs", 1)
    output_dims_val = "output_dims" if n_outputs > 0 else "False"
    if n_outputs > 1:
        pos_arg = f"_n_out={n_outputs}, " + pos_arg
    if kw_str:
        body = (
            f"    return dispatch({name!r}, {inputs_expr}, {output_dims_val},\n"
            f"                    {pos_arg}{kw_str}, **extra_flags)"
        )
    else:
        body = (
            f"    return dispatch({name!r}, {inputs_expr}, {output_dims_val},\n"
            f"                    {pos_arg}**extra_flags)"
        )

    lines.append(body)
    lines.append("")

    return "\n".join(lines)


def _generate_stub(name: str) -> str:
    """Generate a minimal stub when source is unavailable."""
    safe = name.replace("-", "_")
    return (
        f"@bart_op\n"
        f"def {safe}(*inputs: torch.Tensor, output_dims: list[int] | None = None,\n"
        f"           **extra_flags: Any) -> torch.Tensor | tuple[torch.Tensor, ...]:\n"
        f'    """Wraps BART\'s ``{name}`` command.\n\n'
        f"    Run ``bart {name} -h`` for full documentation.\n\n"
        f"    Parameters\n"
        f"    ----------\n"
        f"    *inputs : torch.Tensor\n"
        f"        Positional tensor inputs.\n"
        f"    output_dims : list[int], optional\n"
        f"        Expected output shape.\n"
        f"    **extra_flags : Any\n"
        f"        BART flags forwarded directly.\n\n"
        f"    Returns\n"
        f"    -------\n"
        f"    torch.Tensor\n"
        f'    """\n'
        f"    return dispatch({name!r}, list(inputs), output_dims, **extra_flags)\n\n"
    )


def _generate_file(out_path: Path, bart_src: Path | None) -> None:
    """Write *out_path* with one generated function per BART tool."""

    if bart_src is None or not bart_src.is_dir():
        msg = (
            f"BART source directory not found: {bart_src}\n"
            "Initialise the git submodule first:\n"
            "    git submodule update --init --recursive"
        )
        raise SystemExit(f"ERROR: {msg}")

    tools = _discover_tools(bart_src)
    print(f"Found {len(tools)} tools in {bart_src}")

    # Filter out skipped tools (hand-written wrappers) and internal tools
    _INTERNAL = {"bart", "bench", "version"}
    emit_tools = [t for t in tools if t not in _SKIP_TOOLS and t not in _INTERNAL]

    # Build grouped autosummary for Sphinx
    tool_set = set(emit_tools)
    categorised: set[str] = set()
    autosummary_sections: list[str] = []
    for cat_name, cat_tools in _TOOL_CATEGORIES.items():
        in_cat = [t.replace("-", "_") for t in cat_tools if t in tool_set]
        if in_cat:
            categorised.update(t for t in cat_tools if t in tool_set)
            entries = "\n".join(f"   {t}" for t in in_cat)
            autosummary_sections.append(
                f".. rubric:: {cat_name}\n\n.. autosummary::\n   :nosignatures:\n\n{entries}\n"
            )
    misc_tools = [t.replace("-", "_") for t in emit_tools if t not in categorised]
    if misc_tools:
        entries = "\n".join(f"   {t}" for t in misc_tools)
        autosummary_sections.append(
            f".. rubric:: Miscellaneous\n\n.. autosummary::\n   :nosignatures:\n\n{entries}\n"
        )
    sphinx_toc = "\n".join(autosummary_sections)

    source_note = "yes (bart/src/ parsed)"
    header = (
        '"""bartorch.tools._generated — Auto-generated BART CLI wrappers.\n'
        "\n"
        "Generated by ``build_tools/gen_tools.py``.  Do not edit by hand.\n"
        "\n"
        "Each function wraps one BART command.  All functions:\n"
        "\n"
        "* Accept named keyword arguments matching each BART flag (with type hints).\n"
        "* Are decorated with ``@bart_op`` for automatic ``complex64`` normalisation.\n"
        "* Accept ``**extra_flags`` for flags not listed as named parameters.\n"
        "* Return a plain ``complex64 torch.Tensor`` (or tuple) in C-order.\n"
        "* Carry a CUDA compatibility note in their docstring.\n"
        "\n"
        f"Source-based generation: {source_note}\n"
        "\n" + sphinx_toc + '\n"""\n'
        "\n"
        "from __future__ import annotations\n"
        "\n"
        "from typing import Any\n"
        "\n"
        "import torch\n"
        "\n"
        "from bartorch.core.graph import dispatch\n"
        "from bartorch.core.tensor import bart_op\n"
        "\n"
    )

    func_lines: list[str] = []
    exported: list[str] = []

    for name in emit_tools:
        safe_name = name.replace("-", "_")
        exported.append(safe_name)

        src_path = bart_src / (name + ".c")
        if src_path.exists():
            try:
                info = _parse_tool_source(name, src_path)
                func_lines.append(_generate_func(info))
                continue
            except Exception as exc:
                print(f"  WARNING: parse error for {name}: {exc}; using stub.")

        func_lines.append(_generate_stub(name))

    all_lines = [header]
    all_lines.extend(func_lines)
    all_lines.append("__all__ = [")
    for e in exported:
        all_lines.append(f"    {e!r},")
    all_lines.append("]")
    all_lines.append("")

    out_path.write_text("\n".join(all_lines), encoding="utf-8")
    print(f"Generated {out_path} with {len(exported)} tool wrappers.")


# ---------------------------------------------------------------------------
# Fallback tool list (used when bart/src/ is absent)
# ---------------------------------------------------------------------------

_FALLBACK_TOOLS: list[str] = [
    "affinereg",
    "avg",
    "bitmask",
    "bloch",
    "cabs",
    "calc",
    "caldir",
    "calmat",
    "carg",
    "casorati",
    "cc",
    "ccapply",
    "cdf97",
    "circshift",
    "coils",
    "compress",
    "conj",
    "conv",
    "copy",
    "cpyphs",
    "creal",
    "crop",
    "delta",
    "ecalib",
    "ecaltwo",
    "estdelay",
    "estdims",
    "estmat",
    "estvar",
    "extract",
    "fft",
    "fftmod",
    "fftshift",
    "filter",
    "flatten",
    "flip",
    "fmac",
    "homodyne",
    "invert",
    "itsense",
    "join",
    "looklocker",
    "lrmatrix",
    "mandelbrot",
    "meanshape",
    "moba",
    "mobafit",
    "morphop",
    "multicfl",
    "nlinv",
    "noise",
    "normalize",
    "nrmse",
    "nufft",
    "nufftbase",
    "onehotenc",
    "pattern",
    "phantom",
    "pics",
    "pocsense",
    "poisson",
    "poly",
    "repmat",
    "reshape",
    "resize",
    "rmfreq",
    "rof",
    "rss",
    "sake",
    "saxpy",
    "scale",
    "sdot",
    "signal",
    "slice",
    "spow",
    "squeeze",
    "ssa",
    "std",
    "svd",
    "tgv",
    "threshold",
    "traj",
    "transpose",
    "twixread",
    "upat",
    "walsh",
    "wave",
    "wavelet",
    "wavepsf",
    "whiten",
    "window",
    "zeros",
    "zexp",
]


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    repo_root = Path(__file__).resolve().parent.parent

    parser = argparse.ArgumentParser(
        description="Generate bartorch/tools/_generated.py from BART source."
    )
    parser.add_argument(
        "--bart-src",
        default=str(repo_root / "bart" / "src"),
        help="Path to BART src/ directory (default: bart/src relative to repo root)",
    )
    parser.add_argument(
        "--out",
        default=str(repo_root / "src" / "bartorch" / "tools" / "_generated.py"),
        help="Output path for _generated.py",
    )
    args = parser.parse_args(argv)

    bart_src = Path(args.bart_src)
    out_path = Path(args.out)

    _generate_file(out_path, bart_src)
    return 0


if __name__ == "__main__":
    sys.exit(main())
