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


def read_module(root, module):
    """Return an AST and repository-relative source path."""
    path = root / "src" / Path(*module.split("."))
    path = path / "__init__.py" if path.is_dir() else path.with_suffix(".py")
    return ast.parse(path.read_text()), path.relative_to(root)


def definitions(root, module):
    """Resolve local definitions and explicit package re-exports statically."""
    tree, path = read_module(root, module)
    found = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            found[node.name] = (node, path)
        elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("bartorch."):
            exports = definitions(root, node.module)
            for alias in node.names:
                if alias.name == "*":
                    found.update({k: v for k, v in exports.items() if not k.startswith("_")})
                elif alias.name in exports:
                    found[alias.asname or alias.name] = exports[alias.name]
    return found


def signature(node, name, method=False):
    args = copy.deepcopy(node.args)
    if method and args.args and args.args[0].arg in {"self", "cls"}:
        args.args.pop(0)
    result = f"{name}({ast.unparse(args)})"
    if node.returns:
        result += f" -> {ast.unparse(node.returns)}"
    return result


def render(node, name, module, path, app):
    kind = "class" if isinstance(node, ast.ClassDef) else "function"
    sig = name if kind == "class" else signature(node, name)
    doc = ast.get_docstring(node) or "No narrative docstring is present in this checkout."
    # Public overrides sometimes link to their generated implementation.
    # Keep those links within the public reference instead of exposing both APIs.
    doc = doc.replace(":func:`_generated.", ":func:`bartorch.tools.")
    lines = [f".. py:{kind}:: {sig}", f"   :module: {module}", ""]
    lines += textwrap.indent(str(NumpyDocstring(doc, config=app.config)), "   ").splitlines()
    lines += ["", f"   Source: ``{path}:{node.lineno}``.", ""]
    if kind == "class":
        methods = {n.name: n for n in node.body if isinstance(n, ast.FunctionDef)}
        for child in node.body:
            # LinearOperator.forward is an alias for __call__.
            if isinstance(child, ast.Assign) and isinstance(child.value, ast.Name):
                for target in child.targets:
                    if isinstance(target, ast.Name) and child.value.id in methods:
                        methods[target.id] = methods[child.value.id]
        for method_name, method in methods.items():
            if method_name.startswith("_") and method_name not in {"__call__", "__matmul__", "__add__"}:
                continue
            lines += [f"   .. py:method:: {signature(method, method_name, method=True)}"]
            if any(isinstance(d, ast.Name) and d.id == "classmethod" for d in method.decorator_list):
                lines += ["      :classmethod:"]
            method_doc = ast.get_docstring(method) or "See the class overview and encoding tour."
            lines += ["", textwrap.indent(str(NumpyDocstring(method_doc, config=app.config)), "      "), ""]
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
        "operators": ("bartorch.ops", "Operators and solvers"),
        "cuda": ("bartorch.cuda", "CUDA controls"),
        "finufft": ("bartorch.finufft", "NUFFT backend controls"),
        "cfl": ("bartorch.utils.cfl", "CFL file interoperability"),
    }
    for slug, (module, title) in modules.items():
        entries = definitions(root, module)
        tree, _ = read_module(root, module)
        public = None
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "__all__" for t in node.targets):
                public = ast.literal_eval(node.value)
        names = public if public is not None else entries
        # These exports are modules/version data, documented in their own pages.
        noncallables = {"cuda", "finufft", "__version__"}
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

    entries = definitions(root, "bartorch.tools._commands")
    # The generated export list is literal; overrides add ifft and scale.
    tree, _ = read_module(root, "bartorch.tools._generated")
    public = next(ast.literal_eval(n.value) for n in tree.body if isinstance(n, ast.Assign)
                  and any(isinstance(t, ast.Name) and t.id == "__all__" for t in n.targets))
    public = sorted(set(public) | {"ifft", "scale"})
    body = "Tools\n=====\n\n.. py:module:: bartorch.tools\n\n"
    body += ("Public Python signatures take precedence over generated BART options.\n"
             "Descriptions originate in the checked-in wrappers; many are generated\n"
             "from BART C help strings. A listed wrapper does not establish validation\n"
             "of every option, optional output, or device path.\n\n"
             ".. toctree::\n   :maxdepth: 1\n\n")
    for name in public:
        node, path = entries[name]
        page = f"{name}\n{'=' * len(name)}\n\n.. py:currentmodule:: bartorch.tools\n\n"
        page += render(node, name, "bartorch.tools", path, app)
        write(out / "tools" / f"{name}.rst", page)
        body += f"   tools/{name}\n"
    write(out / "tools.rst", body)


def setup(app):
    app.connect("builder-inited", generate)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
