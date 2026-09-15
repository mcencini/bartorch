"""What the package asks to be installed with, and why.

FINUFFT is a dependency rather than an extra: BART's own gridder is not
reachable from this package's surface, so a bartorch without FINUFFT cannot do
non-Cartesian work at all.  It carries no marker, because a wheel is built only
for the platforms FINUFFT ships a wheel for too -- installing one never starts
a build, and anywhere else the install is from the sdist, where compiling BART
is already the price of entry.

deepinv is an extra: denoisers, losses and samplers come from it through
`priors.ImplicitPrior` and `bartorch.interop`, and nothing else imports it.

torchsim is a dependency because `nlop.FromTorchSim`
turns any of its simulators into a BART nonlinear operator, and
`nlop.InversionRecovery`, `nlop.MultiEcho` and `nlop.Bloch` are `moba`'s
families written on it, so every quantitative reconstruction here imports it.

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


def test_deepinv_is_an_extra_and_not_a_dependency():
    project = _pyproject()["project"]
    assert not _requirements(project["dependencies"], "deepinv"), (
        "only the adapter and the denoisers a caller brings use deepinv; "
        "it belongs in optional-dependencies"
    )
    assert _requirements(project["optional-dependencies"]["deepinv"], "deepinv")


def test_torchsim_is_a_dependency_and_not_an_extra():
    """Every quantitative reconstruction here goes through it."""
    project = _pyproject()["project"]
    assert _requirements(project["dependencies"], "torchsim"), (
        "nlop.FromTorchSim turns a TorchSim simulator into a BART operator and "
        "the moba families are written on it; torchsim belongs in dependencies, "
        "not in optional-dependencies"
    )
    assert "torchsim" not in project["optional-dependencies"], (
        "an extra named torchsim says the signal models are optional, and they are not"
    )


def test_the_models_really_do_reach_torchsim():
    """The claim the requirement rests on, rather than the requirement alone."""
    from torchsim.recon import ModelOperator

    from bartorch import nlop

    model = nlop.MultiEcho((10.0, 40.0), (2,))
    assert isinstance(model.model, ModelOperator)


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
