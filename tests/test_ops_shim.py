"""The operator surface as it was still works, and says it has moved.

``bartorch.ops`` is a deprecation shim.  Deleting it should be the whole of
removing the old surface, so what is checked here is that it adds nothing but
the old spellings: the classes are the same objects, an operator built either
way is the same operator, and every old constructor warns.
"""

import pytest
import torch

from bartorch import linop, nlop
from bartorch.ops import LinearOperator, NonlinearOperator


def _rand(*shape):
    return torch.randn(*shape, dtype=torch.complex64)


def test_the_classes_are_the_ones_that_moved():
    assert LinearOperator is linop.LinearOperator
    assert NonlinearOperator is nlop.NonlinearOperator


@pytest.mark.parametrize(
    ("call", "replacement"),
    [
        (lambda: LinearOperator.fft((8, 16), axes=-1), linop.FFT),
        (lambda: LinearOperator.diagonal(_rand(8, 16), (8, 16)), linop.Diagonal),
        (lambda: LinearOperator.sampling(torch.ones(8, 16), (8, 16)), linop.Sampling),
        (
            lambda: LinearOperator.multiply_sum(_rand(2, 8, 8), (1, 8, 8), (2, 8, 8)),
            linop.MultiplySum,
        ),
        (
            lambda: LinearOperator.from_callbacks((4, 4), (4, 4), lambda x: x, lambda y: y),
            linop.Callback,
        ),
        (
            lambda: NonlinearOperator.from_callbacks(
                (4,), (4,), lambda x: x, lambda d: d, lambda d: d
            ),
            nlop.Callback,
        ),
        (
            lambda: NonlinearOperator.from_torch(lambda p: p * 2, (4,), (4,)),
            nlop.FromTorch,
        ),
    ],
)
def test_an_old_constructor_warns_and_builds_the_class_that_replaced_it(call, replacement):
    with pytest.deprecated_call():
        op = call()
    assert isinstance(op, replacement)


def test_an_operator_built_the_old_way_is_the_same_operator():
    x = _rand(8, 16)
    with pytest.deprecated_call():
        old = LinearOperator.fft((8, 16), axes=-1)
    new = linop.FFT((8, 16), axes=-1)
    torch.testing.assert_close(old(x), new(x))


def test_the_shim_adds_nothing_but_the_old_names():
    import bartorch.ops as ops

    assert set(ops.__all__) == {"LinearOperator", "NonlinearOperator"}
