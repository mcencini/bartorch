"""Collect the API pages' object tables into one page outside the navigation.

Each page under ``docs/api/`` lists its objects in Markdown tables whose first
column is an ``{obj}`` role naming the object in full.  The stub page of every
object is generated from the holder page this module writes, rather than from
the API pages themselves, so that the stubs stay out of the toctree the sidebar
is built from: a sidebar that showed every class, function and method would not
show the six sections the documentation is organized into.

The object lists are read from the API pages, which remain the only place they
are written.  ``tests/test_docs.py`` holds them against each module's
``__all__``.
"""

from __future__ import annotations

import inspect
import re
from importlib import import_module
from pathlib import Path

#: An object link in the first column of a Markdown table row.
_OBJECT = re.compile(r"^\|\s*\{obj\}`~?(bartorch(?:\.\w+)*)`\s*\|", re.M)

#: The modules whose objects the reference documents.
MODULES = (
    "bartorch",
    "bartorch.linop",
    "bartorch.nlop",
    "bartorch.optim",
    "bartorch.priors",
    "bartorch.learning",
    "bartorch.apps",
    "bartorch.tools",
    "bartorch.io",
    "bartorch.cli",
    "bartorch.interop",
)


def split(dotted: str) -> tuple[str, str]:
    """``bartorch.linop.CartesianSense`` as ``("bartorch.linop", "CartesianSense")``.

    The module is the longest prefix that is one of :data:`MODULES`, so that
    ``bartorch.fft`` resolves to the top-level namespace.
    """
    for module in sorted(MODULES, key=len, reverse=True):
        if dotted.startswith(module + "."):
            return module, dotted[len(module) + 1 :]
    raise ValueError(f"{dotted} is not in one of the documented modules")


def collect(pages: Path) -> dict[str, list[str]]:
    """Every API page's object tables, as ``{module: [name, ...]}`` in page order."""
    blocks: dict[str, list[str]] = {}
    for page in sorted(pages.glob("*.md")):
        for dotted in _OBJECT.findall(page.read_text(encoding="utf-8")):
            module, name = split(dotted)
            names = blocks.setdefault(module, [])
            if name not in names:
                names.append(name)
    return blocks


def render(blocks: dict[str, list[str]]) -> str:
    """The holder page: every table again, this time writing the stubs."""
    lines = [
        ":orphan:",
        "",
        "API object index",
        "================",
        "",
        "Every documented object, by module.  The pages of the API reference list",
        "the same objects in tables; this page generates the page of each.",
        "",
    ]
    for module in MODULES:
        names = blocks.get(module)
        if not names:
            continue
        lines += [module, "-" * len(module), "", f".. currentmodule:: {module}", ""]
        lines += [".. autosummary::", "   :toctree: generated", "   :nosignatures:", ""]
        lines += [f"   {name}" for name in names] + [""]
    return "\n".join(lines) + "\n"


def write(into: str | Path) -> int:
    """Write the holder page under ``into``; return the number of objects.

    The page sits at the top of the source directory, so that its ``:toctree:``
    resolves to ``generated/``.  It is rewritten only when its text changes,
    which keeps Sphinx from rereading it and every stub on each build.
    """
    root = Path(into)
    blocks = collect(root / "api")
    (root / "generated").mkdir(parents=True, exist_ok=True)
    target = root / "api_objects.rst"
    text = render(blocks)
    if not target.exists() or target.read_text(encoding="utf-8") != text:
        target.write_text(text, encoding="utf-8")
    return sum(len(names) for names in blocks.values())


# -- What a class page documents ------------------------------------------------


def _public(cls: type) -> bool:
    """Whether ``cls`` is documented on a page of its own."""
    module = getattr(cls, "__module__", "")
    if not module.startswith("bartorch") or cls.__name__.startswith("_"):
        return False
    for name in MODULES:
        namespace = import_module(name)
        if getattr(namespace, cls.__name__, None) is cls and cls.__name__ in namespace.__all__:
            return True
    return False


def _kind(member) -> str | None:
    """The autodoc directive for a class attribute, or None for one to leave out."""
    if isinstance(member, property):
        return "autoproperty"
    if isinstance(member, (classmethod, staticmethod)) or inspect.isfunction(member):
        return "automethod"
    return None


def class_members(fullname: str) -> dict[str, object]:
    """The members a class page documents, and the public classes it inherits from.

    A member is documented on the page of the class that defines it.  Members a
    class inherits from a private base are documented on the class's own page,
    since the private base has none; members inherited from a public class are
    left to that class's page, which the class page links instead.  ``__call__``
    is documented wherever it is, being how a solver or a network is applied.
    """
    module, name = split(fullname)
    cls = getattr(import_module(module), name)
    own: list[tuple[str, str]] = []
    seen: set[str] = set()
    inherited: list[str] = []
    for klass in cls.__mro__:
        if klass is object:
            break
        if not klass.__module__.startswith("bartorch"):
            # torch.nn.Module and other foreign bases are documented by their
            # own projects.
            continue
        if klass is not cls and _public(klass):
            # Everything behind the first public base is on that base's page.
            inherited.append(f"{_documented_module(klass)}.{klass.__name__}")
            break
        for attribute, member in vars(klass).items():
            if attribute in seen:
                continue
            if attribute.startswith("_") and attribute != "__call__":
                continue
            kind = _kind(member)
            if kind is None:
                continue
            seen.add(attribute)
            own.append((kind, attribute))
    return {"members": own, "inherits": inherited}


def _documented_module(cls: type) -> str:
    """The public module a class is documented under."""
    for name in MODULES:
        namespace = import_module(name)
        if getattr(namespace, cls.__name__, None) is cls and cls.__name__ in namespace.__all__:
            return name
    return cls.__module__


def class_context(blocks: dict[str, list[str]]) -> dict[str, dict[str, object]]:
    """:func:`class_members` for every class the reference documents."""
    context = {}
    for module, names in blocks.items():
        namespace = import_module(module)
        for name in names:
            if inspect.isclass(getattr(namespace, name, None)):
                context[f"{module}.{name}"] = class_members(f"{module}.{name}")
    return context
