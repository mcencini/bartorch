"""ctypes marshalling of the ABI's arguments.

With :mod:`bartorch._abi` and :mod:`bartorch._lib`, the only modules that use
ctypes for bartorch's own library; :mod:`bartorch._backend` and
:mod:`bartorch._finufft` also resolve addresses in other libraries.
"""

from __future__ import annotations

import ctypes

from bartorch._abi import APPLY_FN, DIMS, RELEASE_FN

__all__ = [
    "DIMS",
    "address_of",
    "argv",
    "by_reference",
    "dim_vector",
    "dims",
    "float_buffer",
    "floats",
    "ints",
    "longs",
    "null_apply",
    "null_release",
    "out_pointer",
    "pointers",
    "padded_dims",
    "shape_from_dims",
    "text_buffer",
]


def padded_dims(shape: tuple[int, ...]) -> ctypes.Array:
    """BART dimension vector of a C-order shape: reversed, and padded with ones to :data:`DIMS`."""
    rev = list(shape)[::-1]
    if len(rev) > DIMS:
        raise ValueError(f"BART supports at most {DIMS} dimensions, got {len(rev)}")
    return (ctypes.c_long * DIMS)(*(rev + [1] * (DIMS - len(rev))))


def dims(shape: tuple[int, ...]) -> tuple[int, ctypes.Array]:
    """BART rank and dimension vector of a C-order shape, unpadded; a scalar has rank one."""
    rev = list(shape)[::-1] or [1]
    if len(rev) > DIMS:
        raise ValueError(f"BART supports at most {DIMS} dimensions, got {len(rev)}")
    return len(rev), (ctypes.c_long * len(rev))(*rev)


def shape_from_dims(vector: ctypes.Array, min_ndim: int) -> list[int]:
    """C-order shape of a BART dimension vector, dropping leading ones down to ``min_ndim`` axes."""
    rev = [int(vector[i]) for i in range(DIMS)][::-1]
    while len(rev) > max(1, min_ndim) and rev[0] == 1:
        rev.pop(0)
    return rev


def argv(args: list[str]) -> ctypes.Array:
    """A C argument vector of encoded strings."""
    return (ctypes.c_char_p * len(args))(*[a.encode() for a in args])


def longs(values) -> ctypes.Array:
    return (ctypes.c_long * len(values))(*[int(v) for v in values])


def ints(values) -> ctypes.Array:
    return (ctypes.c_int * len(values))(*[int(v) for v in values])


def floats(values) -> ctypes.Array:
    return (ctypes.c_float * len(values))(*[float(v) for v in values])


def pointers(values) -> ctypes.Array:
    return (ctypes.c_void_p * len(values))(*[int(v) for v in values])


def text_buffer(size: int) -> ctypes.Array:
    return ctypes.create_string_buffer(size)


def float_buffer(ptr: int, count: int) -> ctypes.Array:
    """A view of ``count`` floats at ``ptr``, without a copy."""
    return (ctypes.c_float * count).from_address(ptr)


def null_apply() -> ctypes.Array:
    """The NULL an apply callback the caller did not give is passed as.

    A function-pointer argument carries its callback type, which ctypes will
    not convert ``None`` to, so an absent one is a null instance of the type.
    """
    return APPLY_FN()


def null_release() -> ctypes.Array:
    """The NULL a release callback the caller did not give is passed as."""
    return RELEASE_FN()


def dim_vector() -> ctypes.Array:
    """A dimension vector of :data:`DIMS` entries for the library to fill in."""
    return (ctypes.c_long * DIMS)()


def out_pointer() -> ctypes.c_void_p:
    """An address the library writes into; read it back off ``.value``."""
    return ctypes.c_void_p()


def by_reference(value):
    """*value* passed by address, for an argument the library writes through."""
    return ctypes.byref(value)


def address_of(x) -> int:
    """The address a pointer argument carries, or zero."""
    return 0 if not x else ctypes.cast(x, ctypes.c_void_p).value or 0
