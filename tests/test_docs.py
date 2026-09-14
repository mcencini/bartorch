"""Every public name has a place in the API reference (``docs/api/*.md``)."""

import re
from importlib import import_module
from pathlib import Path

import pytest

API = Path(__file__).resolve().parent.parent / "docs" / "api"

#: Public modules and the names of theirs that are modules or data, not entries.
MODULES = {
    "bartorch": {"__version__", "io", "linop", "nlop", "optim", "prox", "tools"},
    "bartorch.linop": set(),
    "bartorch.nlop": set(),
    "bartorch.optim": set(),
    "bartorch.prox": set(),
    "bartorch.tools": set(),
    "bartorch.io": set(),
}


def _listed() -> dict[str, set[str]]:
    """Names listed under each ``currentmodule`` in the API pages' autosummary blocks."""
    listed: dict[str, set[str]] = {}
    for page in API.glob("*.md"):
        module = None
        in_summary = False
        for line in page.read_text().splitlines():
            if m := re.match(r"\.\. currentmodule:: (\S+)", line):
                module = m.group(1)
            elif line.startswith(".. autosummary::"):
                in_summary = True
            elif line.startswith("```"):
                in_summary = False
            elif in_summary and (m := re.fullmatch(r"   ([A-Za-z_]\w*)", line)):
                listed.setdefault(module, set()).add(m.group(1))
    return listed


@pytest.mark.parametrize("module", sorted(MODULES))
def test_every_public_name_is_in_the_reference(module):
    public = set(import_module(module).__all__) - MODULES[module]
    missing = public - _listed().get(module, set())
    assert not missing, f"{module}: not in docs/api: {sorted(missing)}"


@pytest.mark.parametrize("module", sorted(MODULES))
def test_the_reference_lists_nothing_that_is_not_public(module):
    extra = _listed().get(module, set()) - set(import_module(module).__all__)
    assert not extra, f"{module}: listed in docs/api but not public: {sorted(extra)}"
