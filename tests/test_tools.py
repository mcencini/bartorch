"""The command surface: what is exposed, how, and that nothing went missing.

Every command BART builds is curated, derived, or named as not exposed with a
reason.  The audit is that those three together are all of them -- so a command
that arrives with a submodule bump is reachable the same day, and one that
cannot work here has to be argued for rather than quietly dropped.
"""

import inspect
from pathlib import Path

import pytest
import torch

import bartorch.tools as bt
from bartorch import _catalogue as catalogue
from bartorch.tools import _coverage

ROOT = Path(__file__).resolve().parent.parent


# --- the audit --------------------------------------------------------------


def test_every_command_is_curated_derived_or_named_as_not_exposed():
    curated = _coverage.curated_names()
    derived = _coverage.derived_names()
    excluded = frozenset(_coverage.NOT_EXPOSED)
    assert curated | derived | excluded == frozenset(catalogue.COMMANDS), (
        "a BART command is in none of the three groups; add a wrapper, or name "
        "it in _coverage.NOT_EXPOSED with a reason"
    )
    assert not (curated & derived)
    assert not (curated & excluded)
    assert not (derived & excluded)


def test_nothing_is_excluded_that_bart_does_not_have():
    assert frozenset(_coverage.NOT_EXPOSED) <= frozenset(catalogue.COMMANDS)


def test_every_exclusion_gives_a_reason():
    for name, reason in _coverage.NOT_EXPOSED.items():
        assert len(reason) > 20, f"{name} is excluded without saying why"


def test_an_excluded_command_is_not_reachable_by_accident():
    for name in _coverage.NOT_EXPOSED:
        assert not hasattr(bt, name), f"{name} is excluded but exported anyway"


@pytest.mark.parametrize("name", sorted(_coverage.curated_names() | _coverage.derived_names()))
def test_every_exposed_command_is_a_documented_callable(name):
    wrapper = getattr(bt, name)
    assert callable(wrapper)
    assert wrapper.__doc__, f"{name} has no docstring"
    assert inspect.signature(wrapper) is not None
    assert wrapper.bart_command == name


def test_a_curated_wrapper_says_it_is_one():
    assert not bt.pics.is_derived
    assert not bt.fft.is_derived
    assert bt.svd.is_derived


def test_no_module_is_named_after_a_command():
    """``bartorch.tools.sim`` would be the module and the ``sim`` command."""
    for path in (ROOT / "src" / "bartorch" / "tools").glob("*.py"):
        stem = path.stem
        if stem.startswith("_"):
            continue
        assert stem not in catalogue.COMMANDS, f"{stem}.py collides with the {stem} command"


# --- the guard that keeps a typo from ending the session --------------------


def test_an_option_the_command_does_not_have_is_refused():
    """BART answers an unrecognised option by printing its usage and calling
    ``error``; what that leaves behind makes the *next* tool call spin forever
    at full CPU.  So one never reaches BART."""
    with pytest.raises(ValueError, match="has no option called"):
        bt.phantom(8, ncoils=2)
    with pytest.raises(ValueError, match="has no option called"):
        bt.svd(torch.eye(4, dtype=torch.complex64), definitely_not_an_option=1)


def test_the_refusal_suggests_what_was_meant():
    with pytest.raises(ValueError, match="did you mean coil"):
        bt.phantom(8, ncoils=2)


def test_the_library_still_works_after_a_refusal():
    """The point of refusing: what BART does instead is unrecoverable."""
    with pytest.raises(ValueError):
        bt.phantom(8, nonsense=1)
    assert tuple(bt.phantom(8).shape) == (8, 8)


def test_a_real_option_passed_through_by_name_is_not_refused():
    """A curated wrapper takes anything else the command has, by its own name."""
    # `k` is what the curated wrapper calls `kspace`; passing BART's own
    # spelling has to keep working.
    image = bt.phantom(8, k=True)
    assert tuple(image.shape) == (8, 8)


# --- what the curated wrappers buy ------------------------------------------


@pytest.mark.parametrize("solver", ["ist", "fista", "admm", "pridu"])
def test_pics_can_choose_its_solver(solver):
    """BART writes the choice as five separate flags into one variable, and
    reading them one at a time is how it stopped being reachable at all.

    With a regularizer throughout, because IST and FISTA are proximal methods
    and BART asserts on exactly one penalty.
    """
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
    torch.testing.assert_close(bt.fft(x, axes=-1), bt.fft(x, axes=1))


def test_scale_takes_a_number_rather_than_an_array():
    """``bart scale <factor> <input>`` reads the factor from the command line."""
    x = torch.randn(4, 4, dtype=torch.complex64)
    torch.testing.assert_close(bt.scale(x, 2.0), 2 * x)


def test_a_derived_wrapper_is_shaped_like_the_command_line():
    """Which is the honest thing for one to be: `svd` takes what `bart svd`
    takes, under the names BART gives them."""
    parameters = inspect.signature(bt.svd).parameters
    assert "input" in parameters
    assert "extra" in parameters


def test_describe_covers_a_derived_command_too():
    said = bt.describe("svd")
    assert said.startswith("svd --")
