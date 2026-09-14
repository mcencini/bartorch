"""The torch layout of an MRI operator, and the BART dimensions it is built on.

Arrays are laid out the torch way, the axes that vary independently first and
the ones a transform reads last::

    image     (*batches, [sets,] *encoding, [z,] y, x)
    k-space   (*batches, coils, *encoding, shots, samples)
    trajectory           (*encoding, shots, samples, d)

BART lays the same arrays out by role instead -- coils on its COIL axis, sets
of maps on MAPS, frames on TE, coefficients on COEFF -- and its roles are in a
fixed order that puts coils before any encoding axis.  In memory the torch
layout has the coils after them.  So no single BART dimension vector describes
a whole array without a copy, while one block of it does: one batch item, and
inside the coil loop one coil.  An operator is therefore built for a block,
with each axis of the block placed on a BART dimension in the order the block
is laid out in memory, and ``bartorch_linop_blocks`` applies it block after
block.  A block is contiguous in both layouts, so nothing is copied.
"""

from __future__ import annotations

from bartorch._lib import DIMS

(
    READ,
    PHS1,
    PHS2,
    COIL,
    MAPS,
    TE,
    COEFF,
    COEFF2,
    ITER,
    CSHIFT,
    TIME,
    TIME2,
    LEVEL,
    SLICE,
    AVG,
    BATCH,
) = range(16)

#: The dimensions no transform here gives a role to, fastest first: where
#: encoding axes go, other than the one a basis contracts.
FREE = (COEFF2, ITER, CSHIFT, TIME, TIME2, LEVEL, SLICE, AVG, BATCH)


def vector(placed: dict[int, int]) -> tuple[int, ...]:
    """A BART dimension vector: ones, and ``placed[dim]`` where it is given."""
    v = [1] * DIMS
    for dim, n in placed.items():
        v[dim] = int(n)
    return tuple(v)


def encoding_dims(count: int, basis: bool) -> tuple[list[int], list[int]]:
    """Where ``count`` encoding axes go, in torch order, in k-space and in the image.

    With a basis the last encoding axis is the one it contracts: its frames on
    TE in k-space and its coefficients on COEFF in the image, which is where
    BART's NUFFT reads them.  It is the fastest of the encoding axes in memory,
    and TE and COEFF come before every free dimension, so the order holds.
    The other axes take the free dimensions, fastest first.
    """
    rest = count - 1 if (basis and count > 0) else count
    if rest > len(FREE):
        raise ValueError(
            f"{count} encoding axes is more than BART has dimensions for; at most "
            f"{len(FREE) + (1 if basis else 0)}"
        )
    kspace: list[int] = [0] * count
    image: list[int] = [0] * count
    for j in range(rest):
        kspace[rest - 1 - j] = image[rest - 1 - j] = FREE[j]
    if basis and count > 0:
        kspace[count - 1], image[count - 1] = TE, COEFF
    return kspace, image


def split(
    shape: tuple[int, ...], trailing: int, what: str
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """``(batches, rest)``: the leading axes of ``shape`` and its last ``trailing`` ones."""
    shape = tuple(shape)
    if len(shape) < trailing:
        raise ValueError(f"{what} {shape} has fewer than the {trailing} axes the operator reads")
    cut = len(shape) - trailing
    return shape[:cut], shape[cut:]


def count(shape: tuple[int, ...]) -> int:
    n = 1
    for s in shape:
        n *= int(s)
    return n
