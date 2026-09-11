"""Extract the checkout's public API without importing its runtime dependencies.

This deliberately handles this package's explicit re-exports, not arbitrary
Python execution. Unsupported public exports fail the build instead of silently
disappearing from the reference.
"""

import ast
import copy
import textwrap
from pathlib import Path

from sphinx.ext.napoleon.docstring import NumpyDocstring


class ExampleOrder:
    """Sort by lesson number, independently of the execution/dependency prefix."""

    def __init__(self, src_dir):
        self.src_dir = src_dir

    def __call__(self, filename):
        return Path(filename).name.partition("_")[2]


def read_module(root, module):
    """Return an AST and repository-relative source path."""
    path = root / "src" / Path(*module.split("."))
    path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
    return ast.parse(path.read_text()), path.relative_to(root)


def submodules(root, module):
    """The names under *module* that are modules of their own.

    A package re-exports its subpackages -- ``bartorch.linop``, ``bartorch.alg``
    and the rest -- and those are pages, not entries on this one.  Reading them
    off the checkout rather than off a list is what keeps a new subpackage from
    failing the build on the day it is added.
    """
    path = root / "src" / Path(*module.split("."))
    if not path.is_dir():
        return set()
    return {
        child.stem if child.is_file() else child.name
        for child in path.iterdir()
        if not child.name.startswith("_")
        and (child.suffix == ".py" or (child.is_dir() and (child / "__init__.py").exists()))
    }


def definitions(root, module, seen=None):
    """Resolve local definitions and explicit package re-exports statically.

    Modules re-export each other -- a subpackage's ``__init__`` names what its
    own modules define, and those import back from the package -- so a module
    already being resolved is skipped rather than followed round again.
    """
    seen = set() if seen is None else seen
    if module in seen:
        return {}
    seen = seen | {module}
    tree, path = read_module(root, module)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found[node.name] = (node, path)
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            # A module-level constant is part of the interface too, where
            # __all__ names one: `alg.ALGORITHMS` is what the solver argument
            # is checked against, and a reader wants to see the list.
            targets = [node.target] if isinstance(node, ast.AnnAssign) else node.targets
            for target in targets:
                if isinstance(target, ast.Name) and not target.id.startswith("_"):
                    found.setdefault(target.id, (node, path))
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("bartorch."):
            exports = definitions(root, node.module, seen)
            for alias in node.names:
                if alias.name == "*":
                    found.update({k: v for k, v in exports.items() if not k.startswith("_")})
                elif alias.name in exports:
                    found[alias.asname or alias.name] = exports[alias.name]
    return found


def export_names(tree, bindings=None):
    """Read a literal __all__, allowing explicitly bound starred re-exports."""
    bindings = bindings or {}
    for node in tree.body:
        targets = node.targets if isinstance(node, ast.Assign) else []
        if isinstance(node, ast.AnnAssign):
            targets = [node.target]
        if any(isinstance(t, ast.Name) and t.id == "__all__" for t in targets):
            names = []
            for item in node.value.elts:
                if isinstance(item, ast.Starred) and isinstance(item.value, ast.Name):
                    names.extend(bindings[item.value.id])
                else:
                    names.append(ast.literal_eval(item))
            return names
    return None


def signature(node, name, method=False):
    args = copy.deepcopy(node.args)
    if method and args.args and args.args[0].arg in {"self", "cls"}:
        args.args.pop(0)
    result = f"{name}({ast.unparse(args)})"
    if node.returns:
        result += f" -> {ast.unparse(node.returns)}"
    return result


def comment_doc(root, path, lineno):
    """The ``#:`` lines immediately above a module-level assignment.

    A constant has nowhere to put a docstring, so the convention the package
    follows is the one Sphinx reads: a run of ``#:`` comments above it.
    """
    lines = (root / path).read_text().splitlines()
    said = []
    index = lineno - 2
    while index >= 0 and lines[index].lstrip().startswith("#:"):
        said.insert(0, lines[index].lstrip()[2:].strip())
        index -= 1
    return " ".join(said)


def render(node, name, module, path, app):
    if isinstance(node, (ast.Assign, ast.AnnAssign)):
        root = Path(app.confdir).parent
        doc = comment_doc(root, path, node.lineno) or "A constant of this module."
        value = ast.unparse(node.value) if node.value is not None else ""
        lines = [f".. py:data:: {name}", f"   :module: {module}"]
        if value and len(value) <= 200:
            lines += [f"   :value: {value}"]
        lines += ["", textwrap.indent(str(NumpyDocstring(doc, config=app.config)), "   "), ""]
        lines += [f"   Source: ``{path}:{node.lineno}``.", ""]
        return "\n".join(lines)
    kind = "class" if isinstance(node, ast.ClassDef) else "function"
    sig = name if kind == "class" else signature(node, name)
    doc = ast.get_docstring(node) or "No narrative docstring is present in this checkout."
    override = Path(app.confdir) / "_docstrings" / f"{module}.{name}.txt"
    if override.exists():
        doc = override.read_text()
    # This prose currently sits between parameters in dispatch's docstring.
    # Move it to the summary for Napoleon, without editing runtime source.
    if module == "bartorch" and name == "dispatch":
        paragraph = (
            "Inputs are copied before the tool runs unless :func:`set_copy_inputs`\n"
            "turned that off, because a BART tool may write into its inputs.\n"
        )
        if paragraph in doc:
            doc = doc.replace(paragraph, "").replace(
                "Parameters\n", paragraph + "\nParameters\n", 1
            )
    # Public overrides sometimes link to their generated implementation.
    # Keep those links within the public reference instead of exposing both APIs.
    doc = doc.replace(":func:`_generated.", ":func:`bartorch.tools.")
    lines = [f".. py:{kind}:: {sig}", f"   :module: {module}", ""]
    lines += textwrap.indent(str(NumpyDocstring(doc, config=app.config)), "   ").splitlines()
    lines += ["", f"   Source: ``{path}:{node.lineno}``.", ""]
    if override.exists():
        lines += [f"   Documentation override: ``docs/_docstrings/{override.name}``.", ""]
    if kind == "class":
        methods = {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
        for child in node.body:
            # LinearOperator.forward is the un-recorded __call__.
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.Name):
                for target in child.targets:
                    if isinstance(target, ast.Name) and child.value.id in methods:
                        methods[target.id] = methods[child.value.id]
        for method_name, method in methods.items():
            if method_name.startswith("_") and method_name not in {
                "__call__",
                "__matmul__",
                "__add__",
            }:
                continue
            lines += [f"   .. py:method:: {signature(method, method_name, method=True)}"]
            if any(
                isinstance(d, ast.Name) and d.id == "classmethod" for d in method.decorator_list
            ):
                lines += ["      :classmethod:"]
            method_doc = ast.get_docstring(method) or "See the class overview and encoding tour."
            lines += [
                "",
                textwrap.indent(str(NumpyDocstring(method_doc, config=app.config)), "      "),
                "",
            ]
    return "\n".join(lines)


def catalogue(root):
    """The checked-in command catalogue, loaded without importing the package.

    ``src/bartorch/_catalogue.py`` is generated data and imports nothing but
    the standard library, so it can be read on its own -- which matters here
    because importing ``bartorch`` would pull in torch and the compiled
    library, and the reference is extracted from the checkout rather than from
    a working installation.
    """
    import importlib.util
    import sys

    name = "_bartorch_catalogue"
    path = root / "src" / "bartorch" / "_catalogue.py"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    # `dataclass` resolves a class's annotations through its module, so the
    # module has to be findable while its body runs.
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module.COMMANDS


def not_exposed(root):
    """The commands named in ``tools/_coverage.py`` as not exposed, and why."""
    tree, _ = read_module(root, "bartorch.tools._coverage")
    for node in tree.body:
        targets = [node.target] if isinstance(node, ast.AnnAssign) else getattr(node, "targets", [])
        if any(isinstance(t, ast.Name) and t.id == "NOT_EXPOSED" for t in targets):
            return {k.value: v.value for k, v in zip(node.value.keys, node.value.values)}
    raise ValueError("tools/_coverage.py no longer states which commands are not exposed")


def rst(text):
    """Text BART wrote, as inline reStructuredText rather than as markup.

    Its help strings carry things docutils reads: ``cabs`` describes itself
    with ``|<input>|``, which is a substitution reference and an error when
    nothing defines it.
    """
    for char in "\\*`|_":
        text = text.replace(char, "\\" + char)
    return text.strip()


def from_catalogue(command):
    """A page for a wrapper built from its catalogue entry rather than by hand.

    There is no source to read for one of these: it is made at import from what
    BART's own declaration says, and takes what the command line takes.  So the
    page says the same thing the declaration does.
    """
    lines = [
        f".. py:function:: {command.name}(...)",
        "   :module: bartorch.tools",
        "",
        f"   {rst(command.help) or 'BART states no description for this command.'}",
        "",
        "   Built from BART's own declaration of the command, so it is shaped",
        "   like the command line: positional arrays in the order ``bart"
        f" {command.name}`` takes them, then every option under its own name.",
        f"   ``bartorch.tools.describe({command.name!r})`` prints the whole of it.",
        "",
    ]
    if command.arguments:
        lines += ["   **Arguments**", ""]
        for argument in command.arguments:
            kind = "array" if argument.is_array else argument.kind.lower()
            need = "" if argument.required else ", optional"
            lines += [f"   * ``{argument.name}`` -- {kind}{need}"]
        lines += [""]
    if command.options:
        lines += ["   **Options**", ""]
        for option in command.options:
            name = option.long or option.short
            value = f" ``{option.arg}``" if option.takes_value and option.arg else ""
            lines += [f"   * ``{name}``{value} -- {rst(option.help)} (``{option.flag}``)"]
        lines += [""]
    return "\n".join(lines)


def write(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists() or path.read_text() != content:
        path.write_text(content)


def generate(app):
    root = Path(app.confdir).parent
    out = Path(app.srcdir) / "api" / "generated"
    modules = {
        "package": ("bartorch", "Package and diagnostics"),
        "linops": ("bartorch.linop", "Linear operators"),
        "nlops": ("bartorch.nlop", "Nonlinear operators"),
        "prox": ("bartorch.prox", "Regularization terms"),
        "alg": ("bartorch.alg", "BART's own solve"),
        "interop": ("bartorch.interop", "Handing operators to other libraries"),
        "cuda": ("bartorch.cuda", "CUDA controls"),
        "finufft": ("bartorch.finufft", "NUFFT backend controls"),
        "cfl": ("bartorch.utils.cfl", "CFL file interoperability"),
    }
    for slug, (module, title) in modules.items():
        entries = definitions(root, module)
        tree, _ = read_module(root, module)
        public = export_names(tree)
        names = public if public is not None else entries
        # These exports are modules or version data, documented on pages of
        # their own rather than as entries on this one.
        noncallables = submodules(root, module) | {"__version__"}
        missing = set(names) - entries.keys() - noncallables
        if missing:
            raise ValueError(f"Unsupported public exports in {module}: {sorted(missing)}")
        body = f"{title}\n{'=' * len(title)}\n\n.. py:module:: {module}\n\n"
        for name in names:
            if name.startswith("_") or name in noncallables:
                continue
            node, path = entries[name]
            body += render(node, name, module, path, app) + "\n"
        write(out / f"{slug}.rst", body)

    # One page per command, from the wrapper where one is written by hand and
    # from BART's own declaration where the wrapper is built from it.
    curated = definitions(root, "bartorch.tools")
    commands = catalogue(root)
    excluded = not_exposed(root)
    public = sorted({"describe", "ifft"} | (set(commands) - set(excluded)))

    body = "Tools\n=====\n\n.. py:module:: bartorch.tools\n\n"
    body += (
        "Every BART command, as a function on tensors.  A wrapper written by\n"
        "hand takes axes rather than bitmasks and one argument where BART has\n"
        "several flags; the rest are built from BART's own declaration and are\n"
        "shaped like the command line.  ``describe`` prints what either takes.\n"
        "A listed wrapper does not establish validation of every option,\n"
        "optional output, or device path.\n\n"
        ".. toctree::\n   :maxdepth: 1\n\n"
    )
    for name in public:
        page = f"{name}\n{'=' * len(name)}\n\n.. py:currentmodule:: bartorch.tools\n\n"
        if name in curated:
            node, path = curated[name]
            page += render(node, name, "bartorch.tools", path, app)
        else:
            page += from_catalogue(commands[name])
        write(out / "tools" / f"{name}.rst", page)
        body += f"   tools/{name}\n"
    write(out / "tools.rst", body)

    # And a page naming the commands that are not exposed, with the reason,
    # so that the audit `tests/test_tools.py` makes is readable here too.
    page = "Commands not exposed\n====================\n\n"
    page += (
        "Each of these reads or writes something that is not an array, so there\n"
        "is nothing for a wrapper on tensors to hand back.  ``bartorch`` runs\n"
        "BART in-process, so a command that wants a file wants a file.\n\n"
    )
    for name, reason in sorted(excluded.items()):
        page += f"``{name}``\n    {rst(reason)}\n\n"
    write(out / "not_exposed.rst", page)


def setup(app):
    app.connect("builder-inited", generate)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
