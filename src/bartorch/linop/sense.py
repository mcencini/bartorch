"""The SENSE encodings, applied a slab of coils at a time, in the torch layout.

An image is ``(*batches, [sets,] *encoding, [z,] y, x)`` and its samples
``(*batches, coils, *encoding, shots, samples)`` off a grid, or
``(*batches, coils, *encoding, [z,] y, x)`` on one.  Sensitivities are
``([sets,] coils, [z,] y, x)``.
"""

from __future__ import annotations

import torch

from bartorch import _layout
from bartorch._dispatch import _lock
from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, as_operand, dims
from bartorch.linop.base import LinearOperator
from bartorch.linop.form import Array, Form

__all__ = ["NoncartesianSense"]


def _vector(v):
    """A BART dimension vector as the C entry points take it.

    ``dims`` reverses a C-order shape into BART's order, so a vector already in
    BART's order goes through reversed.
    """
    return dims(tuple(v)[::-1])


def _bank(sensitivities: torch.Tensor, ndim: int):
    """``(bank, has_sets, sets, coils, spatial, batch)`` from the bank's shape.

    ``([batch, sets,] coils, *spatial)``: an axis directly in front of the
    coils is a set, which the image carries and the samples do not, and one in
    front of that is a batch, which both carry.  A batch is written with its
    sets axis even where there is one set, so ``(3, c, y, x)`` is three sets
    and ``(3, 1, c, y, x)`` three batch items.

    ``spatial`` is the bank's own three spatial axes -- a kernel's extent for
    kernels -- with a two-dimensional bank's z written out as one.
    """
    s = as_operand(sensitivities, tuple(sensitivities.shape), "sensitivities")
    lead = s.ndim - ndim
    batch = 1
    if lead == 1:
        has_sets, sets, coils = False, 1, int(s.shape[0])
    elif lead == 2:
        has_sets, sets, coils = True, int(s.shape[0]), int(s.shape[1])
    elif lead == 3:
        has_sets, sets, coils = True, int(s.shape[1]), int(s.shape[2])
        batch = int(s.shape[0])
    else:
        raise ValueError(
            f"sensitivities of {tuple(s.shape)} are none of (coils, *spatial), "
            f"(sets, coils, *spatial) and (batch, sets, coils, *spatial) for "
            f"{ndim} spatial axes"
        )
    spatial = tuple(int(n) for n in s.shape[-ndim:])
    return s, has_sets, sets, coils, (spatial if ndim == 3 else (1, *spatial)), batch


class _SensitivityBatch:
    """Where a batch the sensitivities vary along sits, for an encoding that has one.

    The torch layout puts it above the coils, so it is the slowest dimension
    of one item -- and it lives inside the operator rather than around it,
    because ``bartorch_linop_blocks`` applies one operator to every block and
    one bank does not serve every item.
    """

    def _batch_placed(self) -> dict:
        return {} if self.sens_batch < 2 else {_layout.SENS_BATCH: self.sens_batch}

    def _batch_dim(self) -> int:
        return _layout.SENS_BATCH if self.sens_batch > 1 else -1


def _grid_ndim(sensitivities: torch.Tensor, image_shape: Shape, kernels: bool, ndim) -> int:
    """How many spatial axes a transform on a grid has.

    Given, it is given.  Sensitivities held as maps share the image's spatial
    axes, so the longer match says it.  Kernels share only their number, and a
    bank of four axes is either three spatial ones behind the coils or two
    behind sets and coils: that is what ``ndim`` is for.
    """
    if ndim is not None:
        if int(ndim) not in (2, 3):
            raise ValueError(f"a transform has two or three spatial axes, not {ndim}")
        return int(ndim)
    shape, image = tuple(sensitivities.shape), tuple(image_shape)
    if not kernels:
        if len(shape) >= 4 and len(image) >= 3 and shape[-3:] == image[-3:]:
            return 3
        if len(shape) >= 3 and len(image) >= 2 and shape[-2:] == image[-2:]:
            return 2
        raise ValueError(f"sensitivities of {shape} share no spatial axes with an image of {image}")
    if len(shape) == 3:
        return 2
    if len(shape) == 5:
        return 3
    raise ValueError(
        f"kernels of {shape} are three spatial axes behind the coils or two behind sets "
        "and coils; say which with ndim=3 or ndim=2"
    )


def sets_order(batches: Shape, sets: int, encoding: Shape, ndim: int):
    """The image's axes with the sets moved behind the encoding axes, or ``None``.

    BART holds several sets of maps on a lower dimension than any encoding
    axis, so in memory they vary faster than the encoding does: the reverse
    of the torch layout.  Where both are nontrivial the image is permuted
    into BART's order in front of the operator.
    """
    if sets < 2 or all(n == 1 for n in encoding):
        return None
    nb, ne = len(batches), len(encoding)
    return (*range(nb), *range(nb + 1, nb + 1 + ne), nb, *range(nb + 1 + ne, nb + 1 + ne + ndim))


def wrap_item(
    owner, lib, item: int, oshape: Shape, ishape: Shape, batches: Shape, device, order=None
) -> int:
    """The operator over every batch item, from one built for a single one.

    The item's operator is described over the plain reversal of the torch
    shapes, applied to each batch item in turn where there is more than one,
    and freed: the wrapper holds its own reference.  With ``order`` the item
    takes the image permuted that way (see :func:`sets_order`), and the
    permutation is put in front of it.
    """
    if not item:
        return 0
    inner = ishape if order is None else tuple(ishape[o] for o in order)
    try:
        n = _layout.count(batches)
        if n == 1:
            ptr = owner._under_lock(
                lib.bartorch_linop_reshaped, item, DIMS, dims(oshape), dims(inner), device=device
            )
        else:
            ptr = owner._under_lock(
                lib.bartorch_linop_blocks, item, DIMS, dims(oshape), dims(inner), n, device=device
            )
    finally:
        with _lock:
            lib.bartorch_linop_free(item)
    if order is None or not ptr:
        return ptr
    return _behind_permutation(owner, lib, ptr, ishape, order, device)


def build_form(owner, form: Form, image_shape=None, kspace_shape=None, encoding=None) -> Built:
    """One encoding built from its form, over every batch item.

    The form is lowered and built by the library's single encoding entry
    point; what comes back is the operator for one batch item, which
    :func:`wrap_item` spreads over the batches.  ``owner.plan`` is left saying
    which executor path took it.
    """
    image_shape = owner.image_shape if image_shape is None else image_shape
    kspace_shape = owner.kspace_shape if kspace_shape is None else kspace_shape
    encoding = owner.image_encoding if encoding is None else encoding

    lib = library()
    with _lock:
        item, executor = form.build(owner, owner.device)
        owner._plan = form.plan(executor)
        ptr = wrap_item(
            owner,
            lib,
            item,
            kspace_shape,
            image_shape,
            owner.batches,
            owner.device,
            sets_order(owner.batches, owner.sets, encoding, owner.ndim),
        )
    return Built(ptr, image_shape, kspace_shape, keep=form.keep(), device=owner.device)


class _Encoded(LinearOperator):
    """One encoding built from a form the planner lowered, rather than from arguments.

    It carries the shapes and the layout of the encoding it was matched
    against; what differs is the form, which has the element-wise factors the
    composition put on either side of the transform folded into it.
    """

    def __init__(self, source: LinearOperator, form: Form):
        self._form = form
        self._segments = form.contraction
        for name in ("image_shape", "kspace_shape", "batches", "sets", "ndim", "image_encoding"):
            setattr(self, name, getattr(source, name))
        self.ishape, self.oshape, self.device = source.ishape, source.oshape, source.device
        super().__init__()

    def _create(self) -> Built:
        return build_form(self, self._form)


def _behind_permutation(owner, lib, ptr: int, ishape: Shape, order, device) -> int:
    """``ptr`` applied to the image permuted by ``order``, its own normal kept.

    The normal is ``P^H N P`` with ``N`` the operator's normal, so a Toeplitz
    normal stays one.  ``ptr`` is freed.
    """
    from bartorch.linop.shape import Permute

    permute = Permute(ishape, order)._bart()
    p = permute._h.ptr
    made = [ptr]
    try:
        forward = owner._under_lock(lib.bartorch_linop_chain, p, ptr, device=device)
        normal = owner._under_lock(lib.bartorch_linop_normal_op, ptr, device=device)
        back = owner._under_lock(lib.bartorch_linop_adjoint_op, p, device=device)
        made += [forward, normal, back]
        if not (forward and normal and back):
            return 0
        permuted = owner._under_lock(lib.bartorch_linop_chain, p, normal, device=device)
        made.append(permuted)
        if not permuted:
            return 0
        full = owner._under_lock(lib.bartorch_linop_chain, permuted, back, device=device)
        made.append(full)
        if not full:
            return 0
        return owner._under_lock(lib.bartorch_linop_with_normal, forward, full, device=device)
    finally:
        with _lock:
            for h in made:
                if h:
                    lib.bartorch_linop_free(h)


class NoncartesianSense(_SensitivityBatch, LinearOperator):
    """Sensitivities followed by a NUFFT, applied ``coil_batch`` coils at a time.

    Memory held -- and the doubled grid the Toeplitz normal convolves on --
    scales with ``coil_batch`` rather than with the number of coils.

    On a grid the operator to use is
    :func:`~bartorch.linop.CartesianSense`, which is this one's Cartesian path
    under the name that says so.

    Parameters
    ----------
    sensitivities : tensor
        Coil sensitivities ``(coils, [z,] y, x)``, or their k-space kernels when
        ``kernels`` is set.  A bank on the host while the transform is on a
        card is transferred a slab at a time.

        Several sets of maps -- ESPIRiT's second, ENLIVE's relaxed model --
        are ``(sets, coils, [z,] y, x)``, which is what
        :func:`bartorch.tools.ecalib` and :func:`bartorch.tools.nlinv` return
        for ``maps > 1``.  The image then carries the sets and the samples do
        not: the encoding is ``y[c] = sum_m S[m, c] x[m]``.

        A batch goes in front of the sets, ``(batch, sets, coils, [z,] y, x)``,
        and its sets axis is written even where there is one set: independent
        slices each with their own maps are ``(nz, 1, coils, y, x)``, where
        ``(nz, coils, y, x)`` would be one set of maps per slice summed
        together.  Both the image and the samples carry a batch; the
        trajectory does not, being shared across the items, which is what lets
        one plan serve them all.
    image_shape : tuple of int
        Image shape ``(*batches, [batch,] [sets,] *encoding, [z,] y, x)``: the
        batches first, then the one the sensitivities vary along and the sets
        where they carry them, then the
        trajectory's encoding axes -- with a basis, its coefficients in place of
        the last -- then the spatial axes the trajectory has.
    traj : tensor
        Trajectory ``(*encoding, shots, samples, ndim)`` in grid units, as
        :func:`bartorch.tools.traj` produces: ``kx, ky`` or ``kx, ky, kz``.

        A stack on the image's z grid (``z`` blocks of shots at
        ``kz = j - z // 2``, each with the same in-plane trajectory and weights)
        is applied as an FFT along z over 2D transforms, given ``coil_batch=1``
        and one set; ``plan.cartesian`` reports it.
    kspace_shape : tuple of int, optional
        Sample shape; by default ``(*batches, coils, *encoding, shots,
        samples)``, and on a grid ``(*batches, coils, *encoding, [z,] y, x)``.
    kernels : bool
        Read ``sensitivities`` as k-space kernels, zero-padded to the image grid
        and transformed a slab at a time.  The operator then applies the maps
        band-limited to the kernel; :func:`bartorch.maps_to_kernels` makes such
        kernels and :func:`bartorch.kernels_to_maps` gives the maps they stand for.
    toeplitz : bool
        Apply the normal as a convolution with a point spread function.
    modulated : bool
        On a grid, answer in BART's own sample convention rather than the
        centred one -- a scale and a modulation folded into the sensitivities
        and the plain transform after them, which is what ``pics`` works in
        and what its k-space is written in.  The two differ by an ``fftmod``
        on the sample axes; the default is the centred convention, which is
        what :func:`bartorch.fft` produces and so what an operator chained
        against one expects.

        It does not depend on ``coil_batch``: every slab answers in the
        convention that was asked for.  At ``coil_batch=0`` this is BART's own
        operator, arithmetic and all, which is what reproduces ``pics`` to the
        last bit.  Refused off a grid, where there is only one convention, and
        with ``kernels``, because the modulation is the whole grid's and a
        kernel cannot carry it.
    weights : tensor, optional
        Diagonal in k-space, broadcast over ``(*encoding, shots, samples)``.
    basis : tensor, optional
        Subspace basis ``(coeffs, frames)`` over the last encoding axis, as
        :class:`~bartorch.linop.NUFFT` takes it.
    device : device, optional
        Where the operator is built and does its arithmetic; by default where the
        trajectory is, or the sensitivities for a Cartesian operator.  With a card
        here, operands may stay on the host: the image crosses once each way per
        application, the samples a slab at a time, and between applications the
        card holds the operator only.
    coil_batch : int
        Coils applied at once; 0 uses BART's own operator over all coils, which
        lays the samples out BART's way and so is refused where they have
        encoding axes.  A larger batch is faster and holds proportionally more.
        A single-coil operator is always BART's own.  A batch that does not
        divide the coils is cut down to one that does, because the loop steps
        by the slab and the transform is built for a slab.

        What it changes is residency, not arithmetic.  The sample convention
        is ``modulated``'s to say and not this one's.
    fold_maps : bool
        Apply the sensitivities inside the transform of the normal, which saves
        two coil images per batch.  Takes effect only where the transform works
        one coefficient at a time, the gathered arrangement of a compressed
        Toeplitz function.
    ndim : int, optional
        Spatial axes of a transform on a grid, where the sensitivities and the
        image do not say; off a grid the trajectory says.
    """

    #: Whether a trajectory is required.  The Cartesian encoding is the same
    #: operator over BART's own FFT, and reaches it by clearing this.
    _needs_traj = True

    #: Whether a stack on the image's own z grid is decoupled along z.
    _decouples = True

    #: How far a stack's kz may lie from a whole position of the image's grid.
    _on_grid = 1e-4

    def __init__(
        self,
        sensitivities: torch.Tensor,
        image_shape: Shape,
        traj: torch.Tensor | None = None,
        kspace_shape: Shape | None = None,
        kernels: bool = False,
        toeplitz: bool = True,
        modulated: bool = False,
        weights: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        device: torch.device | str | None = None,
        coil_batch: int = 1,
        fold_maps: bool = True,
        ndim: int | None = None,
    ):
        image_shape = tuple(image_shape)
        if traj is None and self._needs_traj:
            raise ValueError("this is the encoding off a grid; CartesianSense is the one on it")
        if traj is None and (weights is not None or basis is not None):
            raise ValueError("weights and a basis belong to a non-Cartesian transform")
        if traj is not None and modulated:
            raise ValueError("the modulated convention is the grid's; off it there is only one")
        if coil_batch < 0:
            raise ValueError(f"coil_batch is a number of coils, not {coil_batch}")

        self.kernels = bool(kernels)
        self.toeplitz = bool(toeplitz)
        self.modulated = bool(modulated)
        self.coil_batch = int(coil_batch)
        self.fold_maps = bool(fold_maps)

        self.traj = None if traj is None else as_operand(traj, tuple(traj.shape), "traj")
        if self.traj is not None:
            from bartorch import _finufft

            if self.traj.ndim < 3:
                raise ValueError(
                    "a trajectory is (*encoding, shots, samples, ndim), "
                    f"not {tuple(self.traj.shape)}"
                )
            self.traj = _finufft.three_components(self.traj)
            self.ndim = _finufft.spatial_ndim(self.traj)
            self.encoding = tuple(int(n) for n in self.traj.shape[:-3])
        else:
            self.ndim = _grid_ndim(sensitivities, image_shape, self.kernels, ndim)
            self.encoding = self._grid_encoding()

        (
            s,
            self.has_sets,
            self.sets,
            self.coils,
            self.sens_spatial,
            self.sens_batch,
        ) = _bank(sensitivities, self.ndim)
        self.sensitivities = s

        self.basis = None
        self.coeffs = None
        if basis is not None:
            b = as_operand(basis, tuple(basis.shape), "basis")
            if b.ndim < 2 or any(n != 1 for n in b.shape[2:]):
                raise ValueError(f"a basis is (coeffs, frames), not {tuple(b.shape)}")
            if not self.encoding:
                raise ValueError("a basis contracts an encoding axis, and the trajectory has none")
            coeffs, frames = int(b.shape[0]), int(b.shape[1])
            if frames != self.encoding[-1]:
                raise ValueError(
                    f"the basis has {frames} frames and the last encoding axis {self.encoding[-1]}"
                )
            self.basis = b.reshape(coeffs, frames)
            self.coeffs = coeffs

        self.image_encoding = self._image_encoding()
        lead = (
            (1 if self.sens_batch > 1 else 0)
            + (1 if self.has_sets else 0)
            + len(self.image_encoding)
            + self.ndim
        )
        batches, rest = _layout.split(image_shape, lead, "the image")
        spatial = tuple(rest[len(rest) - self.ndim :])
        want = (
            *((self.sens_batch,) if self.sens_batch > 1 else ()),
            *((self.sets,) if self.has_sets else ()),
            *self.image_encoding,
            *spatial,
        )
        if tuple(rest) != want:
            raise ValueError(
                f"the image {image_shape} does not end in "
                f"{'(batch, ' if self.sens_batch > 1 else '('}"
                f"{'sets, ' if self.has_sets else ''}*encoding {self.image_encoding}, "
                f"{self.ndim} spatial axes)"
            )
        self.batches = tuple(batches)
        self.spatial = spatial
        self.image_shape = image_shape

        if self.coil_batch == 0 and any(n > 1 for n in self.encoding):
            raise ValueError(
                "coil_batch=0 is BART's own operator, which lays samples out with the coils "
                "before the encoding axes; use a coil batch of one or more"
            )

        self.weights = None
        if weights is not None:
            per_coil = self._kspace_tail()
            w = as_operand(weights, tuple(weights.shape), "weights")
            got = (1,) * (len(per_coil) - w.ndim) + tuple(w.shape)
            if len(got) != len(per_coil) or any(g not in (1, f) for g, f in zip(got, per_coil)):
                raise ValueError(f"weights of {tuple(w.shape)} do not broadcast over {per_coil}")
            self.weights = w.reshape(got)

        batch_axis = (self.sens_batch,) if self.sens_batch > 1 else ()
        default = (*self.batches, *batch_axis, self.coils, *self._kspace_tail())
        self.kspace_shape = default if kspace_shape is None else tuple(kspace_shape)
        if self.kspace_shape != default:
            raise ValueError(f"the samples of this encoding are {default}, not {self.kspace_shape}")
        self.ishape = image_shape
        self.stack = 0 if self.traj is None else self._stack()

        # Where the operator is built follows the transform's own data, not
        # the sensitivities: a bank left on the host is the point.  Named, it
        # may be a card none of the inputs are on.
        if device is not None:
            built_on = torch.device(device)
            if built_on.type == "cuda" and built_on.index is None:
                built_on = torch.device("cuda", torch.cuda.current_device())
        else:
            built_on = s.device if self.traj is None else self.traj.device
        self.device = built_on

        super().__init__()

    # --- the layout ----------------------------------------------------------

    def _grid_encoding(self) -> tuple[int, ...]:
        """The encoding axes of the samples on a grid; none for the plain encoding."""
        return ()

    def _has_basis(self) -> bool:
        """Whether a basis contracts the last encoding axis."""
        return self.coeffs is not None

    def _image_encoding(self) -> tuple[int, ...]:
        """The image's encoding axes: the samples', with coefficients in place of frames."""
        if self.coeffs is None:
            return self.encoding
        return (*self.encoding[:-1], self.coeffs)

    def _kspace_tail(self) -> tuple[int, ...]:
        """One coil's samples."""
        if self.traj is None:
            return (*self.encoding, *self.spatial)
        return tuple(int(n) for n in self.traj.shape[:-1])

    def _spatial3(self) -> tuple[int, int, int]:
        return self.spatial if self.ndim == 3 else (1, *self.spatial)

    def _stack(self) -> int:
        """Positions along z the trajectory is a stack of, or 0 where it is not one.

        A stack is the shots in ``z`` contiguous blocks, block ``j`` at ``kz = j -
        z // 2`` with the same in-plane trajectory as every other block, and any
        weights the same in every block too.  A transform along z over a batch of
        in-plane transforms is then the same operator, and its normal is over x
        and y alone.
        """
        # The executor lays z out on the axis the sets would take, under a slab
        # of one coil.
        if not self._decouples or self.ndim != 3 or self.coil_batch != 1 or self.sets > 1:
            return 0
        z, shots = int(self.spatial[0]), int(self.traj.shape[-3])
        if z < 2 or shots % z:
            return 0
        t = self.traj.real if self.traj.is_complex() else self.traj
        blocks = t.reshape(*t.shape[:-3], z, shots // z, *t.shape[-2:])
        kz = torch.arange(z, dtype=blocks.dtype, device=blocks.device) - z // 2
        if float((blocks[..., 2] - kz[:, None, None]).abs().max()) > self._on_grid:
            return 0
        plane = blocks[..., :2]
        if float((plane - plane[..., :1, :, :, :]).abs().max()) > self._on_grid:
            return 0
        w = self.weights
        if w is not None and int(w.shape[-2]) > 1:
            by_block = w.reshape(*w.shape[:-2], z, int(w.shape[-2]) // z, int(w.shape[-1]))
            if not torch.equal(by_block, by_block[..., :1, :, :].expand_as(by_block)):
                return 0
        return z

    def _in_plane(self, values: torch.Tensor, tail: int) -> torch.Tensor:
        """The first position's block of ``values`` along the shots, contiguous.

        ``values`` is ``(*encoding, shots, samples, *tail)``; shots of one, which
        broadcast, are kept as they are.
        """
        spokes = int(self.traj.shape[-3]) // self.stack
        axis = values.ndim - 2 - tail
        return values.narrow(axis, 0, min(spokes, int(values.shape[axis]))).contiguous()

    def _encoding_placement(self):
        return _layout.encoding_dims(len(self.encoding), self._has_basis(), self.sens_batch > 1)

    def _max_vector(self):
        """The image of one batch item, with the coils BART counts beside it."""
        z, y, x = self._spatial3()
        v = {
            _layout.READ: x,
            _layout.PHS1: y,
            _layout.PHS2: z,
            _layout.COIL: self.coils,
            _layout.MAPS: self.sets,
            **self._batch_placed(),
        }
        _, idims = self._encoding_placement()
        for dim, n in zip(idims, self.image_encoding):
            v[dim] = n
        return _layout.vector(v)

    def _sens_vector(self):
        z, y, x = self.sens_spatial
        return _layout.vector(
            {
                _layout.READ: x,
                _layout.PHS1: y,
                _layout.PHS2: z,
                _layout.COIL: self.coils,
                _layout.MAPS: self.sets,
                **self._batch_placed(),
            }
        )

    def _encoding_vector(self, base: dict, sizes, batch: bool = True) -> tuple[int, ...]:
        kdims, _ = self._encoding_placement()
        v = {**base, **(self._batch_placed() if batch else {})}
        for dim, n in zip(kdims, sizes):
            v[dim] = n
        return _layout.vector(v)

    def _kspace_vector(self):
        if self.traj is None:
            z, y, x = self._spatial3()
            base = {_layout.READ: x, _layout.PHS1: y, _layout.PHS2: z, _layout.COIL: self.coils}
            return self._encoding_vector(base, self.encoding)
        shots, samples = int(self.traj.shape[-3]), int(self.traj.shape[-2])
        return self._encoding_vector(
            {1: samples, 2: shots, _layout.COIL: self.coils}, self.encoding
        )

    # --- building ------------------------------------------------------------

    def _create(self) -> Built:
        return build_form(self, self._form())

    def _basis_layout(self):
        """``(basis, vector)``: the basis the item is built with and its BART dimensions.

        A subspace basis lies along the frames and the coefficients.
        """
        b = self.basis
        if b is None:
            return None, None
        return b, _layout.vector({_layout.TE: b.shape[1], _layout.COEFF: b.shape[0]})

    def _form(self) -> Form:
        """This encoding as the form the library's one executor takes."""
        t, w = self.traj, self.weights
        b, bvec = self._basis_layout()
        tvec = wvec = None
        if self.stack:
            # One position's block: the transform is over x and y, and the
            # executor lays z out as a batch of it.
            t = self._in_plane(t, 1).clone()
            t[..., 2] = 0
            spokes, samples = int(t.shape[-3]), int(t.shape[-2])
            tvec = self._encoding_vector({0: 3, 1: samples, 2: spokes}, self.encoding, batch=False)
            if w is not None:
                wlead = tuple(w.shape[: len(self.encoding)])
                w = self._in_plane(w, 0)
                wvec = self._encoding_vector({1: w.shape[-1], 2: w.shape[-2]}, wlead, batch=False)
        else:
            if t is not None:
                shots, samples, d = (int(n) for n in t.shape[-3:])
                tvec = self._encoding_vector(
                    {0: d, 1: samples, 2: shots}, self.encoding, batch=False
                )
            if w is not None:
                wshape = tuple(w.shape)
                base = {1: wshape[-1], 2: wshape[-2]} if t is not None else {}
                wvec = self._encoding_vector(base, wshape[: len(self.encoding)], batch=False)
        return Form(
            transform="nufft" if t is not None else "fft",
            max_vector=self._max_vector(),
            kspace_vector=self._kspace_vector(),
            sensitivities=Array(self.sensitivities, self._sens_vector()),
            kernels=self.kernels,
            basis=None if b is None else Array(b, bvec),
            weights=None if w is None else Array(w, wvec),
            traj=None if t is None else Array(t, tvec),
            toeplitz=self.toeplitz,
            modulated=self.modulated,
            coil_batch=self.coil_batch,
            fold_maps=self.fold_maps,
            coils=self.coils,
            sets=self.sets,
            coeffs=1 if b is None else int(b.shape[0]),
            batch_dim=self._batch_dim(),
            stacked=bool(self.stack),
        )


class Coils(_SensitivityBatch, LinearOperator):
    """Coil sensitivities, without the transform that usually follows them.

    The multiply on its own: an image to coil images, and the conjugate
    sensitivities summed over the coils on the way back.  For an encoding
    whose transform is not a Fourier transform, and which therefore cannot be
    a SENSE operator.

    Held as maps it is what :class:`~bartorch.linop.MultiplySum` builds, and
    with ``coil_batch=0`` it *is* that operator.  Held as kernels, or walked a
    slab at a time, the bank is never resident whole -- the arrangement
    :class:`NoncartesianSense` uses.

    Parameters
    ----------
    sensitivities : tensor
        Coil sensitivities ``([batch, sets,] [sets,] coils, [z,] y, x)``, or
        their k-space kernels when ``kernels`` is set.
    image_shape : tuple of int
        Image shape ``(*batches, [batch,] [sets,] [coeffs,] [z,] y, x)``.  The
        coil images are ``(*batches, [batch,] coils, [coeffs,] [z,] y, x)``.
    kernels : bool
        Read ``sensitivities`` as k-space kernels, zero-padded to the image
        grid and transformed a slab at a time, as :class:`NoncartesianSense`
        reads them.
    device : device, optional
        Where the operator is built and does its arithmetic; by default where
        the sensitivities are.
    coil_batch : int
        Coils applied at once; 0 uses BART's own ``fmac`` over all of them.
    coeffs : int
        Subspace coefficients the image carries, on an axis of their own in
        front of the spatial ones: the sensitivities are the same for every
        coefficient, so nothing about the multiply changes.
    ndim : int, optional
        Spatial axes, where the sensitivities and the image do not say.

    Examples
    --------
    >>> S = Coils(kernels, (y, x), kernels=True, ndim=2)
    >>> A = FFT(S.oshape, axes=(-2, -1)) @ S
    """

    def __init__(
        self,
        sensitivities: torch.Tensor,
        image_shape: Shape,
        kernels: bool = False,
        device: torch.device | str | None = None,
        coil_batch: int = 1,
        coeffs: int = 1,
        ndim: int | None = None,
    ):
        image_shape = tuple(image_shape)
        if coeffs < 1:
            raise ValueError(f"coeffs is a number of coefficients, not {coeffs}")
        if coil_batch < 0:
            raise ValueError(f"coil_batch is a number of coils, not {coil_batch}")

        self.kernels = bool(kernels)
        self.coil_batch = int(coil_batch)
        self.coeffs = int(coeffs)
        self.ndim = _grid_ndim(sensitivities, image_shape, self.kernels, ndim)
        (
            s,
            self.has_sets,
            self.sets,
            self.coils,
            self.sens_spatial,
            self.sens_batch,
        ) = _bank(sensitivities, self.ndim)
        self.sensitivities = s

        encoding = (self.coeffs,) if self.coeffs > 1 else ()
        carried = (self.sens_batch,) if self.sens_batch > 1 else ()
        lead = len(carried) + (1 if self.has_sets else 0) + len(encoding) + self.ndim
        batches, rest = _layout.split(image_shape, lead, "the image")
        spatial = tuple(rest[len(rest) - self.ndim :])
        want = (*carried, *((self.sets,) if self.has_sets else ()), *encoding, *spatial)
        if tuple(rest) != want:
            raise ValueError(f"the image {image_shape} does not end in {want}")
        self.batches = tuple(batches)
        self.spatial = spatial
        self.image_shape = image_shape
        self.encoding = encoding
        self.ishape = image_shape
        self.oshape = (*self.batches, *carried, self.coils, *encoding, *spatial)
        self.device = torch.device(device) if device is not None else s.device

        super().__init__()

    def _create(self) -> Built:
        z, y, x = self.spatial if self.ndim == 3 else (1, *self.spatial)
        maxv = _layout.vector(
            {
                _layout.READ: x,
                _layout.PHS1: y,
                _layout.PHS2: z,
                _layout.COIL: self.coils,
                _layout.MAPS: self.sets,
                _layout.COEFF: self.coeffs,
                **self._batch_placed(),
            }
        )
        sz, sy, sx = self.sens_spatial
        sensv = _layout.vector(
            {
                _layout.READ: sx,
                _layout.PHS1: sy,
                _layout.PHS2: sz,
                _layout.COIL: self.coils,
                _layout.MAPS: self.sets,
                **self._batch_placed(),
            }
        )
        return build_form(
            self,
            Form(
                transform="none",
                max_vector=maxv,
                kspace_vector=maxv,
                sensitivities=Array(self.sensitivities, sensv),
                kernels=self.kernels,
                coil_batch=self.coil_batch,
                coils=self.coils,
                sets=self.sets,
                coeffs=self.coeffs,
                batch_dim=self._batch_dim(),
            ),
            image_shape=self.ishape,
            kspace_shape=self.oshape,
            encoding=self.encoding,
        )
