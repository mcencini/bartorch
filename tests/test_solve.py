"""The solve, which is BART's and not this package's.

``bartorch_solve`` hands BART the three things its own ``lsqr2`` takes -- the
encoding, the proximal operators its ``-R`` strings name, and which of its
iterations to run -- through ``opt_reg_configure`` and ``italgo_config``, the
same functions ``pics`` calls and in the same order.  Nothing here iterates.

What is checked is that the mechanism is BART's: that the loop does not cross
back into Python, that every one of BART's iterations is reachable, and that a
specification means here what it means on the command line.
"""

import pytest
import torch

import bartorch
import bartorch.tools as bt
from bartorch import alg, linop


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def _unitary():
    """An operator whose normal is the identity, so the answer is known."""
    return linop.FFT((8, 8), axes=(-1, -2))


# --- the loop stays in C ----------------------------------------------------


def test_a_bart_operator_is_handed_over_as_itself():
    """The requirement: an encoding BART built is given to BART's solver as it
    stands, so the iteration has nothing to call back into.  A wrapper around
    it would be a crossing per application, every step."""
    A = _unitary()
    assert A.as_bart() is A


def test_an_operator_written_here_is_the_one_that_costs_a_crossing():
    """And it is visible as one: the callbacks fire.  This is the price of
    putting something of one's own in the encoding, and it is paid per
    application rather than per solve."""
    A = _unitary()
    calls = []

    def forward(x):
        calls.append("forward")
        return A.forward(x)

    def adjoint(x):
        calls.append("adjoint")
        return A.adjoint(x)

    P = linop.Callback((8, 8), (8, 8), forward, adjoint)
    alg.solve(P, _rand(8, 8), solver="cg", maxiter=5)
    assert calls, "a Python operator was not called at all"


def test_the_solve_is_one_call_into_the_library():
    """However many iterations it runs."""
    from bartorch._lib import library

    A = _unitary()
    y = _rand(8, 8)
    lib = library()
    entered = []
    original = lib.bartorch_solve

    class Counting:
        def __call__(self, *args):
            entered.append(1)
            return original(*args)

    lib.bartorch_solve = Counting()
    try:
        alg.solve(A, y, solver="cg", maxiter=50)
    finally:
        lib.bartorch_solve = original
    assert entered == [1]


# --- it is BART's iteration -------------------------------------------------


def test_the_plain_solve_is_the_one_the_operator_already_had():
    """``lstsq`` and this reach the same answer by two of BART's own drivers,
    so they agree to what single precision allows and not by construction."""
    A = _unitary()
    y = _rand(8, 8)
    torch.testing.assert_close(
        alg.solve(A, y, solver="cg", maxiter=15),
        A.lstsq(y, maxiter=15),
        rtol=1e-4,
        atol=1e-5,
    )


def test_an_orthonormal_encoding_gives_back_what_it_was_given():
    A = _unitary()
    truth = _rand(8, 8)
    torch.testing.assert_close(
        alg.solve(A, A(truth), solver="cg", maxiter=40), truth, rtol=1e-3, atol=1e-4
    )


@pytest.mark.parametrize("solver", ["cg", "ist", "fista", "admm", "pridu"])
def test_every_iteration_bart_has_is_reachable(solver):
    """Including the ones the tool selects with a flag of their own, which is
    how BART writes a choice and why they were unreachable before."""
    A = _unitary()
    y = _rand(8, 8)
    regularizers = None if solver == "cg" else "W:3:0:0.01"
    got = alg.solve(A, y, solver=solver, regularizers=regularizers, maxiter=5)
    assert got.shape == A.ishape
    assert torch.isfinite(got.abs()).all()


def test_a_regularizer_means_what_it_means_on_the_command_line():
    """The strings are read by BART's own parser, so a heavier weight shrinks
    the answer, as it does for the tool."""
    A = _unitary()
    y = _rand(8, 8)
    # With the step spelled out.  This operator's normal is the identity, so
    # one is the right step; BART's default without `-s` or `-e` diverges here
    # and warns that it will, which is the tool's behaviour too.
    light = alg.solve(A, y, regularizers="W:3:0:0.001", solver="fista", maxiter=30, step=1.0)
    heavy = alg.solve(A, y, regularizers="W:3:0:0.5", solver="fista", maxiter=30, step=1.0)
    assert heavy.abs().sum() < light.abs().sum()


def test_several_regularizers_are_taken_together():
    A = _unitary()
    got = alg.solve(
        A, _rand(8, 8), regularizers=["W:3:0:0.01", "T:3:0:0.01"], solver="admm", maxiter=5
    )
    assert torch.isfinite(got.abs()).all()


# --- what it refuses --------------------------------------------------------


def test_an_iteration_bart_does_not_have_is_refused_here():
    with pytest.raises(ValueError, match="solver must be one of"):
        alg.solve(_unitary(), _rand(8, 8), solver="newton")


def test_a_regularizer_bart_does_not_have_is_refused_before_bart_sees_it():
    """BART answers an unrecognised one with ``error()``, and what that leaves
    behind makes the next call into the library spin forever -- so a typo in a
    regularizer would end the session, exactly as an unknown option used to."""
    with pytest.raises(ValueError, match="not a regularizer BART has"):
        alg.solve(_unitary(), _rand(8, 8), regularizers="NOTAREGULARIZER:1:2:3")
    assert tuple(bt.phantom(8).shape) == (8, 8)


def test_the_regularizers_are_the_ones_barts_parser_knows():
    """Read off ``grecon/optreg.c``, so the list cannot drift from it."""
    import re
    from pathlib import Path

    source = Path(__file__).resolve().parent.parent / "bart" / "src" / "grecon" / "optreg.c"
    if not source.exists():
        pytest.skip("the BART submodule is not checked out")
    found = set(re.findall(r'strcmp\(rt, "([A-Za-z0-9]+)"\)', source.read_text()))
    # `h` is BART's own request for help on the regularizers, not one of them.
    assert found - {"h"} == set(alg.solve.__globals__["REGULARIZERS"])


# --- against the tool -------------------------------------------------------


def test_an_assembled_sense_solve_reaches_the_same_image_as_pics():
    """Not the same number: ``pics`` conditions its k-space and estimates a
    scaling around the solve, and this is given the operator and the data as
    they are.  What is held here is that the assembled problem is the same
    problem -- the images agree to the scale of the reconstruction.
    """
    kspace = bt.phantom(24, coils=4, kspace=True)
    maps = bt.ecalib(kspace, maps=1)
    bartorch.set_coil_batch(1)
    A = linop.Sense(maps.squeeze(1), (4, 24, 24))

    theirs = bt.pics(kspace, maps, maxiter=20, w=1.0).squeeze()
    ours = alg.solve(A, kspace.squeeze(1), maxiter=20).squeeze()

    scale = float(theirs.abs().max())
    assert float((ours - theirs).abs().max()) / scale < 1e-2
