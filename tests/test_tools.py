"""The command surface: what is exposed, how, and that nothing went missing.

Every command BART builds is wrapped by hand, derived into a ``bartorch.tools``
section, or private with a reason.  The audit is that those three together are
all of them, so a command arriving with a BART update has to be placed.
"""

import inspect
import re
from importlib import import_module
from pathlib import Path

import pytest
import torch

import bartorch
import bartorch.tools as bt
import bartorch.tools.recon as recon
from bartorch import _call, _coverage, prox
from bartorch import _catalogue as catalogue
from bartorch._dispatch import dispatch
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
    term = prox.Wavelet((-1, -2), 0.01)
    image = bt.pics(kspace, maps, regularizers=term, solver=solver, maxiter=5)
    assert tuple(image.shape) == (24, 24)


def test_pics_refuses_a_solver_bart_does_not_have():
    kspace = bt.phantom(24, coils=2, kspace=True)
    maps = bt.ecalib(kspace, maps=1)
    with pytest.raises(ValueError, match="solver must be one of"):
        bt.pics(kspace, maps, solver="newton")


def test_an_axis_is_an_axis_and_not_a_bitmask():
    x = torch.randn(4, 8, dtype=torch.complex64)
    torch.testing.assert_close(bartorch.fft(x, axes=-1), bartorch.fft(x, axes=1))


# --- axes, index sets and terms where BART takes bitmasks -------------------

#: What reads as dimensions in BART's help: a bitmask, flags, dims.
_READS_AS_DIMENSIONS = re.compile(
    r"bitmask|flags?\b|\bdims?\b|dimension|squash|shared|loop over|unknowns|subset|group|<T>",
    re.I,
)
_INTEGERS = frozenset({"INT", "UINT", "PINT", "LONG", "ULONG", "ULLONG"})

#: Arguments whose help reads as dimensions but which are not, with why.
_NOT_DIMENSIONS = {
    ("phantom", "-x"): "a size",
    ("poisson", "-Y"): "a size",
    ("poisson", "-Z"): "a size",
    ("seq", "-z"): "a count of partitions",
    ("pics", "-R"): "refused: pics takes the terms as its regularizers argument",
}


def _reachable_arguments():
    """(command, flag or positional name, BART's words for it) for each integer
    or ``-R`` argument a caller can give: every one of a derived wrapper, and
    the options of a curated one that passes the rest through by name."""

    def options(command):
        for option in command.options:
            if option.kind in _INTEGERS or "<T>" in option.arg:
                yield command.name, option.flag, f"{option.arg} {option.help}"

    for name in sorted(_coverage.derived_names()):
        command = catalogue.COMMANDS[name]
        for argument in command.arguments:
            if argument.kind in _INTEGERS:
                yield name, argument.name, argument.name
        yield from options(command)
    for name, wrappers in sorted(_coverage.curated_wrappers().items()):
        if any(
            p.kind is p.VAR_KEYWORD
            for w in wrappers
            for p in inspect.signature(w).parameters.values()
        ):
            yield from options(catalogue.COMMANDS[name])


def test_every_argument_bart_takes_as_dimensions_takes_axes():
    """Each is in ``_call.TRANSLATED`` or said here not to be dimensions."""
    missed = [
        (name, key, said)
        for name, key, said in _reachable_arguments()
        if _READS_AS_DIMENSIONS.search(said)
        and (name, key) not in _call.TRANSLATED
        and (name, key) not in _NOT_DIMENSIONS
    ]
    assert not missed, missed


def test_every_translated_argument_is_one_a_caller_can_give():
    reachable = {(name, key) for name, key, _ in _reachable_arguments()}
    assert not set(_call.TRANSLATED) - reachable


def test_a_derived_wrapper_takes_axes_where_bart_takes_a_bitmask():
    kspace = bt.phantom(24, coils=2, kspace=True)
    ours = bt.pattern(kspace, s=0)
    assert torch.equal(ours, dispatch("pattern", [kspace], None, s=8))
    assert "Axes to squash." in bt.pattern.__doc__
    assert "bitmask" not in bt.pattern.__doc__


def test_without_an_array_to_count_from_an_axis_is_negative():
    with pytest.raises(ValueError, match="negative"):
        bt.seq(raga_flags=1)


def test_what_a_curated_wrapper_passes_through_takes_axes(monkeypatch):
    seen = {}
    monkeypatch.setattr(recon, "dispatch", lambda *args, **kwargs: seen.update(kwargs))
    kspace = torch.zeros(2, 3, 8, 8, dtype=torch.complex64)
    bt.pics(kspace, kspace, L=-3)
    assert seen["L"] == 4
    bt.nlinv(kspace, s=(0, -1))
    assert seen["s"] == 8 | 1


def _pics_data():
    kspace = bt.phantom(24, coils=2, kspace=True)
    return kspace, bt.ecalib(kspace, maps=1)


@pytest.mark.parametrize(
    "term,flags",
    [
        (prox.Wavelet((-1, -2), 0.01, randshift=False), {"R": ["W:3:0:0.01"], "n": True}),
        (prox.LocallyLowRank((-1, -2), 0.01, block=4), {"R": ["L:3:0:0.01"], "b": 4}),
        (prox.TotalGeneralizedVariation((-1, -2), 0.01), {"R": ["G:3:0:0.01"]}),
    ],
    ids=["wavelet", "locally low rank", "tgv"],
)
def test_pics_is_given_each_term_as_bart_would_be(term, flags):
    """The same bits as the command line the term stands for, shared
    settings included."""
    kspace, maps = _pics_data()
    ours = bt.pics(kspace, maps, regularizers=term, maxiter=5)
    theirs = dispatch("pics", [kspace, maps], None, i=5, **flags)
    assert torch.equal(ours, theirs)


def test_pics_takes_terms_and_not_strings():
    kspace, maps = _pics_data()
    with pytest.raises(TypeError, match="prox.Wavelet"):
        bt.pics(kspace, maps, regularizers="W:3:0:0.01")
    with pytest.raises(TypeError, match="regularizers"):
        bt.pics(kspace, maps, R="W:3:0:0.01")
    with pytest.raises(TypeError, match="randshift"):
        bt.pics(kspace, maps, n=True)


def test_a_setting_pics_gives_once_has_to_agree_across_terms():
    kspace, maps = _pics_data()
    terms = [prox.Wavelet((-1, -2), 0.01), prox.Wavelet((-1, -2), 0.01, family="haar")]
    with pytest.raises(ValueError, match="family"):
        bt.pics(kspace, maps, regularizers=terms)


def test_a_command_refuses_a_term_its_parser_does_not_know():
    kspace, maps = _pics_data()
    with pytest.raises(TypeError, match="sqpics does not take"):
        bt.sqpics(kspace, maps, R=prox.TotalGeneralizedVariation((-1, -2), 0.01))


def test_a_derived_wrapper_is_shaped_like_the_command_line():
    """``sim`` takes what ``bart sim`` takes, under BART's names."""
    parameters = inspect.signature(bt.sim).parameters
    assert "extra" in parameters


def test_describe_covers_a_private_command_too():
    assert describe("svd").startswith("svd --")


def test_a_tool_resolves_a_conjugated_view_before_reading_it():
    """See tests/test_linop.py: a conjugation is a flag on shared storage.

    A tool reads its operand through the same pointer an operator does, so it
    had the same hole: ``flip`` of a conjugated tensor came back unconjugated.
    """
    x = torch.randn(4, 8, dtype=torch.complex64)
    torch.testing.assert_close(
        bartorch.flip(x.conj(), axes=-1), bartorch.flip(x.conj().resolve_conj(), axes=-1)
    )
    assert not torch.allclose(bartorch.flip(x.conj(), axes=-1), bartorch.flip(x, axes=-1))
