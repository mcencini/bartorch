"""Every default a docstring documents is the one the signature has.

A parameter with a default states it in its Parameters entry as
``type, default=value``, and ``value`` is what ``inspect.signature`` reports;
``optional`` alone does not say which value is taken, so it is not used.  A
parameter without a default states neither.  The docstrings are parsed by
numpydoc, the same reading the reference is rendered from.

The functions of :mod:`bartorch.tools` derived from BART's catalogue build
their docstrings from the parameters their signatures are built from
(``_call._docstring``), so they are held to the same rule rather than
exempted from it.
"""

from __future__ import annotations

import ast
import inspect
import math
import re
from importlib import import_module

import pytest

numpydoc = pytest.importorskip("numpydoc.docscrape")

#: The modules whose public names the reference documents.
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

#: Constructors that pass ``**kwargs`` on and document what they pass, by the
#: callable whose signature holds those parameters' defaults.
FORWARDED = {
    "bartorch.linop.mri.CartesianSense": "bartorch.linop.mri._GridSense",
}

#: Names a documented default may use beyond Python literals.
_NAMES = {"sqrt": math.sqrt, "pi": math.pi}

_DEFAULT = re.compile(r"\bdefault\s*=\s*(.+?)\s*$")
_OPTIONAL = re.compile(r"\boptional\b")


def _targets(module: str):
    """The module's public functions and classes, and their classes' public methods.

    Methods a class inherits from a private base are documented on its page,
    so they are included; those of a foreign base, such as ``nn.Module``, are
    not.
    """
    namespace = import_module(module)
    for name in namespace.__all__:
        obj = getattr(namespace, name)
        if inspect.ismodule(obj) or not callable(obj):
            continue
        yield f"{module}.{name}", obj
        if not inspect.isclass(obj):
            continue
        seen = set()
        for klass in obj.__mro__:
            if klass is object or not klass.__module__.startswith("bartorch"):
                continue
            for attribute, member in vars(klass).items():
                if attribute in seen or (attribute.startswith("_") and attribute != "__call__"):
                    continue
                seen.add(attribute)
                function = getattr(member, "__func__", member)
                if inspect.isfunction(function):
                    yield f"{module}.{name}.{attribute}", function


def _parameters(obj) -> list:
    """The Parameters and Other Parameters entries of ``obj``'s docstring."""
    doc = numpydoc.NumpyDocString(inspect.getdoc(obj) or "")
    return list(doc["Parameters"]) + list(doc["Other Parameters"])


def _value(text: str):
    """A documented default as a value: a Python literal, or ``sqrt``/``pi`` of one."""
    try:
        return ast.literal_eval(text)
    except (ValueError, SyntaxError):
        return eval(compile(ast.parse(text, mode="eval"), "<default>", "eval"), {**_NAMES})


def _forwarded(obj) -> dict[str, inspect.Parameter]:
    target = FORWARDED.get(f"{getattr(obj, '__module__', '')}.{getattr(obj, '__qualname__', '')}")
    if target is None:
        return {}
    module, _, name = target.rpartition(".")
    return dict(inspect.signature(getattr(import_module(module), name)).parameters)


def _problems(qualified: str, obj) -> list[str]:
    try:
        signature = inspect.signature(obj)
    except (TypeError, ValueError):
        return []
    parameters = dict(signature.parameters)
    takes_kwargs = any(p.kind is p.VAR_KEYWORD for p in parameters.values())
    forwarded = _forwarded(obj) if takes_kwargs else {}
    problems = []
    for entry in _parameters(obj):
        for name in (n.strip().lstrip("*") for n in entry.name.split(",")):
            parameter = parameters.get(name) or forwarded.get(name)
            where = f"{qualified}: {name} : {entry.type}"
            if parameter is None:
                problems.append(f"{where} -- not a parameter of {signature}")
                continue
            if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
                continue
            stated = _DEFAULT.search(entry.type)
            if _OPTIONAL.search(entry.type):
                problems.append(f"{where} -- says 'optional' rather than its default")
            if parameter.default is inspect.Parameter.empty:
                if stated:
                    problems.append(f"{where} -- documents a default the signature lacks")
                continue
            if not stated:
                problems.append(f"{where} -- the signature's default {parameter.default!r}")
                continue
            try:
                value = _value(stated.group(1))
            except Exception as exc:  # noqa: BLE001  (the message is the finding)
                problems.append(f"{where} -- cannot read {stated.group(1)!r}: {exc}")
                continue
            if value != parameter.default or type(value) is not type(parameter.default):
                problems.append(f"{where} -- the signature's default is {parameter.default!r}")
    return problems


@pytest.mark.parametrize("module", MODULES)
def test_every_documented_default_is_the_signatures(module):
    problems = [p for qualified, obj in _targets(module) for p in _problems(qualified, obj)]
    assert not problems, "\n".join(problems)


def test_a_wrong_default_is_found():
    """The check reads a default rather than the presence of the word."""

    def f(x, *, maxiter=30, step=0.95, name=None):
        """Something.

        Parameters
        ----------
        x : torch.Tensor
        maxiter : int, default=20
        step : float, optional
        name : str
        """

    problems = _problems("f", f)
    assert any("maxiter" in p and "30" in p for p in problems)
    assert any("step" in p and "optional" in p for p in problems)
    assert any("name" in p and "None" in p for p in problems)
    assert not any(p.startswith("f: x") for p in problems)


def test_barts_laid_out_help_is_kept_verbatim():
    """A command's help lists its conventions and dimensions line by line, and
    a docstring joining those lines into one paragraph makes the lists
    unreadable; prose is joined and the lists are a literal block."""
    tools = pytest.importorskip("bartorch.tools")
    doc = numpydoc.NumpyDocString(tools.wave.__doc__)
    assert doc["Summary"] == ["Perform a wave-caipi reconstruction."]
    assert "::" in tools.wave.__doc__
    assert "\n      * (sx, sy, sz) - Spatial dimensions.\n" in tools.wave.__doc__
