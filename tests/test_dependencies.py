"""What the package asks to be installed with, and why.

FINUFFT is a dependency rather than an extra: BART's own gridder is not
reachable from this package's surface, so a bartorch without FINUFFT cannot do
non-Cartesian work at all.  The one thing that stops it being a plain
unconditional dependency is that FINUFFT ships no wheel for every platform
this package does -- Linux on aarch64, an Intel Mac -- and requiring it there
would make `pip install bartorch` build it from source, which needs a
toolchain nobody installing a wheel agreed to have.

So the requirement carries markers, `_finufft.WHEEL_PLATFORMS` is the same set
written in Python, and what is checked here is that the two say the same thing
-- and that where the metadata promises FINUFFT, it is actually here.
"""

import platform
import sys
from pathlib import Path

import pytest

from bartorch import _finufft

ROOT = Path(__file__).resolve().parent.parent

#: Every platform this package builds a wheel for, per .github/workflows/publish.yml,
#: plus the ones a source install can land on.
CANDIDATES = [
    {"sys_platform": "linux", "platform_machine": "x86_64"},
    {"sys_platform": "linux", "platform_machine": "aarch64"},
    {"sys_platform": "darwin", "platform_machine": "arm64"},
    {"sys_platform": "darwin", "platform_machine": "x86_64"},
    {"sys_platform": "win32", "platform_machine": "AMD64"},
]


def _pyproject() -> dict:
    tomllib = pytest.importorskip("tomllib", reason="pyproject is read with tomllib, a 3.11 module")
    return tomllib.loads((ROOT / "pyproject.toml").read_text())


def _requirements(specs: list[str], name: str) -> list:
    """Every requirement on *name* in *specs*, parsed."""
    packaging = pytest.importorskip("packaging.requirements")
    parsed = [packaging.Requirement(s) for s in specs]
    return [r for r in parsed if r.name == name]


def test_finufft_is_a_dependency_and_not_an_extra():
    project = _pyproject()["project"]
    assert _requirements(project["dependencies"], "finufft"), (
        "FINUFFT computes every non-Cartesian transform; it belongs in "
        "dependencies, not in optional-dependencies"
    )


def test_cufinufft_stays_optional():
    """It serves a transform on a card, and most machines have no card."""
    project = _pyproject()["project"]
    assert not _requirements(project["dependencies"], "cufinufft")
    assert _requirements(project["optional-dependencies"]["cufinufft"], "cufinufft")


@pytest.mark.parametrize(
    "env", CANDIDATES, ids=lambda e: f"{e['sys_platform']}-{e['platform_machine']}"
)
def test_the_markers_are_the_platforms_finufft_ships_a_wheel_for(env):
    """pyproject and ``_finufft.WHEEL_PLATFORMS`` are the same set, checked by
    evaluating rather than by reading, so a reworded marker cannot drift."""
    project = _pyproject()["project"]
    required = any(
        r.marker is None or r.marker.evaluate(env)
        for r in _requirements(project["dependencies"], "finufft")
    )
    expected = (env["sys_platform"], env["platform_machine"]) in _finufft.WHEEL_PLATFORMS
    assert required == expected, (
        f"pyproject {'requires' if required else 'does not require'} FINUFFT on "
        f"{env['sys_platform']}/{env['platform_machine']}, and WHEEL_PLATFORMS says "
        f"{'it ships a wheel' if expected else 'it does not'}"
    )


def test_the_extra_is_still_there_for_where_no_wheel_is():
    """An old ``pip install 'bartorch[finufft]'`` still means something, and on
    aarch64 it is the only way to ask."""
    project = _pyproject()["project"]
    assert _requirements(project["optional-dependencies"]["finufft"], "finufft")


def test_finufft_is_installed_wherever_this_package_says_it_will_be():
    """The promise the markers make, checked against this environment.

    Read off the installed distribution rather than off pyproject, so what is
    checked is what pip was actually told.  ``extra`` is empty in the
    environment, so a requirement that belongs to an extra evaluates false and
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
    if not required:
        pytest.skip(f"FINUFFT is not a dependency on {sys.platform}/{platform.machine()}")

    assert _finufft.available(), _finufft.required_but_missing()


def test_the_message_for_a_missing_finufft_says_which_case_this_is():
    said = _finufft.required_but_missing()
    if _finufft.ships_a_wheel():
        assert "should already be installed" in said
        assert "bartorch[finufft]" not in said
    else:
        assert "bartorch[finufft]" in said
        assert "compiler" in said
