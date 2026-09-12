"""What the package asks to be installed with, and why.

FINUFFT is a dependency rather than an extra: BART's own gridder is not
reachable from this package's surface, so a bartorch without FINUFFT cannot do
non-Cartesian work at all.  It carries no marker, because a wheel is built only
for the platforms FINUFFT ships a wheel for too -- installing one never starts
a build, and anywhere else the install is from the sdist, where compiling BART
is already the price of entry.

deepinv is a dependency for a different reason.  It was an extra while it was
only an adapter -- `to_deepinv` handed an operator over and nothing imported
deepinv until someone asked -- and it stopped being that when `optim.IST`,
`optim.FISTA`, `optim.ADMM` and `optim.PRIDU` started running the iterations in
`bartorch.optim.iterators`, which are written as deepinv optimizers.  A solver
that runs a different loop depending on whether an extra happens to be
installed is worse than the install it saves.

cuFINUFFT is the one that stays optional: it serves a transform on a card, and
most machines have no card.

The promise is checked twice over -- against pyproject, and against the
metadata pip was actually given -- because a dependency that quietly became an
extra again would show up as seventeen failing tests and no explanation.
"""

import platform
import sys
from pathlib import Path

import pytest

from bartorch import _finufft

ROOT = Path(__file__).resolve().parent.parent


def _pyproject() -> dict:
    tomllib = pytest.importorskip("tomllib", reason="pyproject is read with tomllib, a 3.11 module")
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _requirements(specs: list[str], name: str) -> list:
    """Every requirement on *name* in *specs*, parsed."""
    packaging = pytest.importorskip("packaging.requirements")
    return [r for r in (packaging.Requirement(s) for s in specs) if r.name == name]


def test_finufft_is_a_dependency_and_not_an_extra():
    project = _pyproject()["project"]
    assert _requirements(project["dependencies"], "finufft"), (
        "FINUFFT computes every non-Cartesian transform; it belongs in "
        "dependencies, not in optional-dependencies"
    )
    assert "finufft" not in project["optional-dependencies"], (
        "an extra named finufft says it is optional, and it is not"
    )


def test_deepinv_is_a_dependency_and_not_an_extra():
    """The ordinary way to run a regularized reconstruction imports it."""
    project = _pyproject()["project"]
    assert _requirements(project["dependencies"], "deepinv"), (
        "the proximal solvers run iterations written as deepinv optimizers; "
        "deepinv belongs in dependencies, not in optional-dependencies"
    )
    assert "deepinv" not in project["optional-dependencies"], (
        "an extra named deepinv says the solvers are optional, and they are not"
    )


def test_the_solvers_really_do_reach_deepinv():
    """The claim the requirement rests on, rather than the requirement alone.

    If the iterations ever stop being deepinv's, this is the test that says so
    and the dependency can go back to being an extra.
    """
    import deepinv.optim.optim_iterators as di

    from bartorch.optim import iterators

    assert issubclass(iterators.ADMMIteration, di.OptimIterator)
    assert issubclass(iterators.FISTAIteration, di.OptimIterator)


def test_the_finufft_requirement_holds_on_every_platform():
    """No marker: a wheel is built only where FINUFFT ships one too, and every
    other install is already a source build."""
    project = _pyproject()["project"]
    for requirement in _requirements(project["dependencies"], "finufft"):
        assert requirement.marker is None, (
            f"{requirement} is conditional, and a platform where it does not "
            "apply is one where the package cannot do non-Cartesian work"
        )


def test_cufinufft_stays_optional():
    """It serves a transform on a card, and most machines have no card."""
    project = _pyproject()["project"]
    assert not _requirements(project["dependencies"], "cufinufft")
    assert _requirements(project["optional-dependencies"]["cufinufft"], "cufinufft")


def test_finufft_is_installed_wherever_this_package_says_it_will_be():
    """The promise the metadata makes, checked against this environment.

    Read off the installed distribution rather than off pyproject, so what is
    checked is what pip was actually told.  ``extra`` is empty in the
    environment, so a requirement belonging to an extra evaluates false and
    only the unconditional ones are counted.
    """
    from importlib import metadata

    packaging = pytest.importorskip("packaging.requirements")
    try:
        specs = metadata.requires("bartorch") or []
    except metadata.PackageNotFoundError:
        pytest.skip("bartorch is on the path but not installed, so it has no metadata")

    env = {
        "sys_platform": sys.platform,
        "platform_machine": platform.machine(),
        "extra": "",
    }
    required = [
        r
        for r in (packaging.Requirement(s) for s in specs)
        if r.name == "finufft" and (r.marker is None or r.marker.evaluate(env))
    ]
    assert required, (
        "the installed bartorch does not require FINUFFT; it was built from a "
        "pyproject where FINUFFT was still optional"
    )
    assert _finufft.available(), _finufft.required_but_missing()


def test_a_missing_finufft_is_reported_as_the_broken_install_it_is():
    said = _finufft.required_but_missing()
    assert "dependency" in said
    assert "bartorch[finufft]" not in said, "there is no such extra to point anyone at"
