"""The command surface: what is exposed, how, and that nothing went missing.

Every command BART builds is wrapped by hand, derived into a ``bartorch.tools``
section, or private with a reason.  The audit is that those three together are
all of them, so a command arriving with a BART update has to be placed.
"""

import inspect
from importlib import import_module
from pathlib import Path

import pytest
import torch

import bartorch
import bartorch.tools as bt
from bartorch import _catalogue as catalogue
from bartorch import _coverage
from bartorch._options import describe

ROOT = Path(__file__).resolve().parent.parent


# --- the audit --------------------------------------------------------------


def test_every_command_is_curated_derived_or_private():
    curated = _coverage.curated_names()
    derived = _coverage.derived_names()
    private = frozenset(_coverage.PRIVATE)
    missing = frozenset(catalogue.COMMANDS) - curated - derived - private
    assert not missing, f"commands in none of the three groups: {sorted(missing)}"
    assert not (curated & derived), sorted(curated & derived)
    assert not (curated & private), sorted(curated & private)
    assert not (derived & private), sorted(derived & private)


def test_every_private_command_gives_a_reason():
    for name, reason in _coverage.PRIVATE.items():
        assert len(reason) > 10, f"{name} is private without saying why"


def test_a_private_command_has_no_public_wrapper():
    public = set(bartorch.__all__) | set(bt.__all__) | set(bartorch.prox.__all__)
    for name in _coverage.PRIVATE:
        assert name not in public, f"{name} is private but exported"


@pytest.mark.parametrize("name", sorted(_coverage.derived_names()))
def test_every_derived_command_is_a_documented_callable(name):
    wrapper = getattr(bt, name)
    assert wrapper.is_derived and wrapper.bart_command == name
    assert wrapper.__doc__
    assert wrapper.__module__ in _coverage.TOOLS_MODULES
    assert inspect.signature(wrapper) is not None


def test_every_curated_wrapper_is_documented_and_exported_from_its_module():
    for command, wrappers in _coverage.curated_wrappers().items():
        for wrapper in wrappers:
            assert wrapper.__doc__, f"{wrapper.__name__} ({command}) has no docstring"
            module = import_module(wrapper.__module__)
            assert wrapper.__name__ in module.__all__


def test_a_curated_wrapper_says_it_is_one():
    assert not bt.pics.is_derived
    assert not bartorch.fft.is_derived
    assert bt.sim.is_derived


def test_no_tools_module_is_named_after_a_command():
    """``bartorch.tools.sim`` would be the module and the ``sim`` command."""
    for path in (ROOT / "src" / "bartorch" / "tools").glob("*.py"):
        stem = path.stem
        if stem.startswith("_"):
            continue
        assert stem not in catalogue.COMMANDS, f"{stem}.py collides with the {stem} command"


# --- the guard that keeps a typo from ending the session --------------------


def test_an_option_the_command_does_not_have_is_refused():
    """BART answers an unrecognised option by printing its usage and calling
    ``error``, which leaves the next tool call spinning at full CPU.  So one
    never reaches BART."""
    with pytest.raises(ValueError, match="has no option called"):
        bt.phantom(8, ncoils=2)
    with pytest.raises(ValueError, match="has no option called"):
        bt.sim(definitely_not_an_option=1)


def test_the_refusal_suggests_what_was_meant():
    with pytest.raises(ValueError, match="did you mean coil"):
        bt.phantom(8, ncoils=2)


def test_the_library_still_works_after_a_refusal():
    with pytest.raises(ValueError):
        bt.phantom(8, nonsense=1)
    assert tuple(bt.phantom(8).shape) == (8, 8)


def test_a_real_option_passed_through_by_name_is_not_refused():
    """A curated wrapper takes the command's other options by BART's name."""
    image = bt.phantom(8, k=True)
    assert tuple(image.shape) == (8, 8)


# --- what the curated wrappers buy ------------------------------------------


@pytest.mark.parametrize("solver", ["ist", "fista", "admm", "pridu"])
def test_pics_can_choose_its_solver(solver):
    """With a regularizer, because IST and FISTA assert on exactly one penalty."""
    kspace = bt.phantom(24, coils=2, kspace=True)
    maps = bt.ecalib(kspace, maps=1)
    image = bt.pics(kspace, maps, regularizers="W:3:0:0.01", solver=solver, maxiter=5)
    assert tuple(image.shape) == (24, 24)


def test_pics_refuses_a_solver_bart_does_not_have():
    kspace = bt.phantom(24, coils=2, kspace=True)
    maps = bt.ecalib(kspace, maps=1)
    with pytest.raises(ValueError, match="solver must be one of"):
        bt.pics(kspace, maps, solver="newton")


def test_an_axis_is_an_axis_and_not_a_bitmask():
    x = torch.randn(4, 8, dtype=torch.complex64)
    torch.testing.assert_close(bartorch.fft(x, axes=-1), bartorch.fft(x, axes=1))


def test_a_derived_wrapper_is_shaped_like_the_command_line():
    """``sim`` takes what ``bart sim`` takes, under BART's names."""
    parameters = inspect.signature(bt.sim).parameters
    assert "extra" in parameters


def test_describe_covers_a_private_command_too():
    assert describe("svd").startswith("svd --")
