"""Every public name has a place in the API reference (``docs/api/*.md``).

The API pages list their objects in tables whose first column is an ``{obj}``
role; ``docs/api_objects.py`` collects those tables into the page the
per-object stubs are generated from.
"""

import importlib.util
from importlib import import_module
from pathlib import Path

import pytest

DOCS = Path(__file__).resolve().parent.parent / "docs"
API = DOCS / "api"

#: Public modules and the names of theirs that are modules or data, not entries.
MODULES = {
    "bartorch": {
        "__version__",
        "apps",
        "cli",
        "interop",
        "io",
        "learning",
        "linop",
        "nlop",
        "optim",
        "priors",
        "tools",
    },
    "bartorch.learning": set(),
    "bartorch.linop": set(),
    "bartorch.nlop": set(),
    "bartorch.optim": set(),
    "bartorch.priors": set(),
    "bartorch.tools": set(),
    "bartorch.apps": set(),
    "bartorch.cli": set(),
    "bartorch.io": set(),
    "bartorch.interop": set(),
}


def _api_objects():
    """``docs/api_objects.py``, which is not a package module."""
    spec = importlib.util.spec_from_file_location("api_objects", DOCS / "api_objects.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _listed() -> dict[str, set[str]]:
    """Names listed in the API pages' object tables, by module."""
    return {module: set(names) for module, names in _api_objects().collect(API).items()}


@pytest.mark.parametrize("module", sorted(MODULES))
def test_every_public_name_is_in_the_reference(module):
    public = set(import_module(module).__all__) - MODULES[module]
    missing = public - _listed().get(module, set())
    assert not missing, f"{module}: not in docs/api: {sorted(missing)}"


@pytest.mark.parametrize("module", sorted(MODULES))
def test_the_reference_lists_nothing_that_is_not_public(module):
    extra = _listed().get(module, set()) - set(import_module(module).__all__)
    assert not extra, f"{module}: listed in docs/api but not public: {sorted(extra)}"


def test_the_object_index_generates_a_page_for_every_listed_object():
    """The holder page carries every table entry under its module's autosummary."""
    objects = _api_objects()
    blocks = objects.collect(API)
    text = objects.render(blocks)
    assert text.startswith(":orphan:")
    for module, names in blocks.items():
        assert f".. currentmodule:: {module}" in text
        for name in names:
            assert f"\n   {name}\n" in text, f"{module}.{name} has no stub"


def test_api_pages_carry_no_visible_autosummary():
    """Object lists are human-written tables; autosummary runs on the hidden holder page."""
    for page in API.glob("*.md"):
        assert ".. autosummary::" not in page.read_text(), page.name
