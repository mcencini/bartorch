"""Nonlinear MRI encoding operators: the image and the coils as two unknowns.

:class:`NonlinearSense` is BART's own model, from ``noir/model2.c``, built by
the same code ``nlinv`` runs::

    kspace = A[ (mask * image) * ifftuc(weights * ksens) ]

:func:`CartesianSense` and :func:`NoncartesianSense` name its two cases the
way the linear operators do.  :class:`CoilSense` is the general recipe
instead: any linear encoding from coil images to data, with a product of two
unknowns in front of it.
"""

from __future__ import annotations

import torch

from bartorch._dispatch import BartError, _ensure_ready, _lock, _on_device
from bartorch._lib import DIMS, library
from bartorch._operator import Built, Shape, _Handle, as_operand, dims
from bartorch.linop.base import LinearOperator
from bartorch.linop.mri import _spatial
from bartorch.nlop.base import NonlinearOperator, _built, _chain
from bartorch.nlop.basic import Multiply

__all__ = ["CartesianSense", "CoilSense", "NoncartesianSense", "NonlinearSense"]

#: BART's three spatial axes, as a flag word.
_FFT_FLAGS = 7

#: What ``bartorch_noir_dims`` calls each shape.
_SHAPES = {
    "kspace": 0,
    "coil_images": 1,
    "image": 2,
    "coils": 3,
    "coils_as_multiplied": 4,
    "pattern": 5,
    "trajectory": 6,
}


def _tensor(x, what: str):
    return None if x is None else as_operand(x, tuple(x.shape), what)


def _optional(x):
    """``(dims, pointer)`` for a tensor BART may be given or not."""
    return (None, None) if x is None else (dims(tuple(x.shape)), x.data_ptr())


class _Part(LinearOperator):
    """One of the linear operators the model is made of, held against its lifetime."""

    def __init__(self, fn, noir, ishape: Shape, oshape: Shape, device):
        self.fn, self.noir = fn, noir
        self._ishape, self._oshape = tuple(ishape), tuple(oshape)
        self._device = device
        super().__init__()

    def _create(self) -> Built:
        ptr = self._under_lock(self.fn, self.noir.ptr, device=self._device)
        return Built(ptr, self._ishape, self._oshape, keep=(self.noir,), device=self._device)


class NonlinearSense(NonlinearOperator):
    r"""Joint image and coil-sensitivity forward model, BART's ``noir``.

    The operator has **two** inputs, the image and the coil representation, and
    one output, the data.  Its derivative with respect to either input is a
    linear operator (:meth:`~bartorch.nlop.NonlinearOperator.linearize`), and a
    Gauss-Newton step solves the linearized problem over both jointly.

    The coil unknown is not the sensitivity maps.  It is a k-space
    representation :math:`\hat{s}` of them, from which the maps follow by the
    Sobolev weighting

    .. math::

        S = \mathcal{F}^{-1} \left[ \, \kappa \,
            (1 + a |k|^2)^{-b/2} \, \hat{s} \, \right]

    with ``sobolev`` giving :math:`(a, b)` and ``c`` the scale
    :math:`\kappa` (BART's ``noir_calc_weights``).  The weighting is part of
    the model, so the estimated variable is smoothness-regularized by
    construction and the Gauss-Newton step needs no separate penalty on the
    coils.  :attr:`coils` is the linear operator mapping fitted coefficients to
    sensitivities.

    Off the grid the model is asymmetric, as BART builds it: it returns gridded
    coil images rather than samples, so a measurement must be gridded to match.
    :meth:`prepare` does that, and is a no-op on a grid, where the model
    returns k-space directly.

    Parameters
    ----------
    image_shape : tuple of int
        Coil-image shape, ``(coils, *spatial)``, C order -- the same shape the
        linear encodings take.  The image itself is this with one coil.
    pattern : tensor, default=None
        Binary sampling mask, one at acquired positions and zero elsewhere,
        on a grid.  Without one the acquisition is treated as fully sampled.
    trajectory : tensor, default=None
        Trajectory ``(..., samples, 3)`` in grid units, ``kx, ky, kz``, as
        :func:`bartorch.tools.traj` produces.  Giving one selects the
        non-Cartesian model.
    kspace_shape : tuple of int, default=None
        Sample shape.  Off the grid it defaults to the trajectory's with the
        image's coil axes in front, as the NUFFT's does; on the grid it is the
        coil-image shape.
    coil_shape, coefficient_shape : tuple of int, default=None
        Where the sensitivities live and where their coefficients do; both
        default to the coil-image shape.
    weights : tensor, default=None
        Diagonal in k-space applied to the samples, off the grid; its
        conjugate is applied on the adjoint.
    basis : tensor, default=None
        Temporal subspace basis ``(coeffs, frames)`` over the last encoding
        axis, off the grid.  The Cartesian model does not take one.
    mask : tensor, default=None
        A support the image is restricted to.
    sobolev : tuple of float, default=(220.0, 32.0)
        ``(a, b)`` of the coil weighting ``c (1 + a |k|^2)^(-b/2)``; BART's
        ``220, 32``.
    c : float, default=1.0
        The scale on that weighting, BART's ``1``.
    real : bool, default=False
        Constrain the image to be real.
    sos : bool, default=False
        BART's sum-of-squares variant of the coil weighting.
    oversampling_coils : float, default=None
        Fit the coils on a finer grid than the image.  By default whichever
        ``nlinv`` uses: two off the grid, one on it.
    oversampled_coils : bool, default=False
        Return them on that grid rather than on the image's.
    toeplitz : bool, default=True
        Apply the normal in closed form rather than as the forward and
        adjoint applications: a convolution with a point spread function, off
        the grid.
    optimized : bool, default=False
        BART's ``noir2_noncart_optimized_create``, off the grid.

    Examples
    --------
    >>> F = NonlinearSense((8, 128, 128), pattern=mask)
    >>> F.ishapes
    ((1, 128, 128), (8, 128, 128))
    >>> image, coefficients = nlop.IRGNM()(kspace, F)
    """

    def __init__(
        self,
        image_shape: Shape,
        pattern: torch.Tensor | None = None,
        trajectory: torch.Tensor | None = None,
        *,
        kspace_shape: Shape | None = None,
        coil_shape: Shape | None = None,
        coefficient_shape: Shape | None = None,
        weights: torch.Tensor | None = None,
        basis: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        sobolev: tuple[float, float] = (220.0, 32.0),
        c: float = 1.0,
        real: bool = False,
        sos: bool = False,
        oversampling_coils: float | None = None,
        oversampled_coils: bool = False,
        toeplitz: bool = True,
        optimized: bool = False,
    ):
        self.image_shape = tuple(image_shape)
        # BART counts a coil on its fourth axis and a slice on its third, so a
        # two-dimensional problem has to have that third spatial axis written
        # out before the coils land where the model expects them.  This is the
        # same expansion the linear encodings do.
        coils, spatial = _spatial(self.image_shape)
        self.coil_image_shape = (coils, *spatial)

        self.noncart = trajectory is not None
        if pattern is None and not self.noncart:
            # BART's model always carries one; `nlinv` estimates it from the
            # measurement, and a fully sampled acquisition is all ones.
            pattern = torch.ones((1, *spatial), dtype=torch.complex64)
        self.pattern = _tensor(pattern, "pattern")
        self.trajectory = _tensor(trajectory, "trajectory")
        self.weights = _tensor(weights, "weights")
        self.basis = _tensor(basis, "basis")
        self.mask = _tensor(mask, "mask")

        if self.noncart and self.basis is not None and self.pattern is not None:
            raise ValueError("off the grid the pattern goes into the weights, not beside them")
        if not self.noncart and self.basis is not None:
            raise ValueError(
                "BART's Cartesian noir model refuses a basis; build the subspace encoding "
                "with CoilSense over a linear operator that carries it"
            )

        self.coil_shape = tuple(coil_shape) if coil_shape is not None else self.coil_image_shape
        self.coefficient_shape = (
            tuple(coefficient_shape) if coefficient_shape is not None else self.coil_shape
        )

        if kspace_shape is not None:
            self.kspace_shape = tuple(kspace_shape)
        elif self.noncart:
            # BART's noir model reads the samples in BART's order, with the
            # coils in front of the trajectory's axes and the coordinate axis a
            # singleton: `(coils, *trajectory[:-1], 1)`.
            self.kspace_shape = (
                self.coil_image_shape[0],
                *tuple(self.trajectory.shape)[:-1],
                1,
            )
        else:
            self.kspace_shape = self.coil_image_shape

        self.sobolev = (float(sobolev[0]), float(sobolev[1]))
        self.c = float(c)
        self.real = bool(real)
        self.sos = bool(sos)
        # `nlinv` fits the coils on a doubled grid off the grid and on the
        # image's own on it; None asks for whichever BART would use.
        self.oversampling_coils = (
            (2.0 if self.noncart else 1.0)
            if oversampling_coils is None
            else float(oversampling_coils)
        )
        self.oversampled_coils = bool(oversampled_coils)
        self.toeplitz = bool(toeplitz)
        self.optimized = bool(optimized)
        super().__init__()

    # --- construction ------------------------------------------------------

    def _image_only(self) -> Shape:
        """The coil-image shape with one coil, which is where the image lives."""
        return (1,) + self.coil_image_shape[1:]

    def _create(self) -> Built:
        _ensure_ready()
        lib = library()
        pat_dims, pattern = _optional(self.pattern)
        trj_dims, traj = _optional(self.trajectory)
        wgh_dims, weights = _optional(self.weights)
        bas_dims, basis = _optional(self.basis)
        msk_dims, mask = _optional(self.mask)
        a, b = self.sobolev
        device = None
        for t in (self.trajectory, self.pattern, self.weights, self.mask):
            if t is not None:
                device = t.device
                break
        with _lock, _on_device(device or torch.device("cpu")):
            ptr = lib.bartorch_noir_create(
                DIMS,
                dims(self.kspace_shape),
                dims(self.coil_image_shape),
                dims(self._image_only()),
                dims(self.coefficient_shape),
                dims(self.coil_shape),
                pat_dims,
                pattern,
                trj_dims,
                traj,
                wgh_dims,
                weights,
                bas_dims,
                basis,
                msk_dims,
                mask,
                int(self.noncart),
                int(self.optimized),
                int(self.toeplitz),
                _FFT_FLAGS,
                _FFT_FLAGS,
                int(self.real),
                int(self.sos),
                a,
                b,
                self.c,
                self.oversampling_coils,
                int(self.oversampled_coils),
            )
        keep = tuple(
            t
            for t in (self.pattern, self.trajectory, self.weights, self.basis, self.mask)
            if t is not None
        )
        self._noir = _Handle(ptr, lib.bartorch_noir_free, keep)
        self.device = device

        model = self._under_lock(lib.bartorch_noir_model, self._noir.ptr, device=device)
        # BART settles the shapes itself -- the coils may be oversampled, and
        # off the grid the model returns gridded coil images -- so they are
        # read back rather than assumed.
        data_shape = self._read(_SHAPES["coil_images"] if self.noncart else _SHAPES["kspace"])
        return _built(
            model,
            (self._image_only(), self.coefficient_shape),
            (data_shape,),
            keep=(self._noir,),
            device=device,
        )

    def _composition(self) -> NonlinearOperator:
        """The model written out: the two unknowns' linear parts, their product, the transform.

        ``noir2_join`` (``model2.c:164``) is these four operators, and off the
        grid its last stage is asymmetric -- see
        :class:`~bartorch.nlop.bundle.Asymmetric`.
        """
        from bartorch.linop.basic import Identity
        from bartorch.nlop.bundle import Asymmetric

        image, coils, transform = self.image, self.coils, self.transform
        product = Multiply(image.oshape, coils.oshape)
        made = _chain(coils.to_nonlinear(), product, output=0, input=1)
        made = _chain(image.to_nonlinear(), made, output=0, input=0)
        # The chain leaves the coil coefficients in front of the image.
        made = made._permute_inputs([1, 0])
        last = (
            Asymmetric(transform.gram(), Identity(product.oshape), source=transform)
            if self.noncart
            else transform.to_nonlinear()
        )
        return _chain(made, last, output=0, input=0)

    def _bundle(self):
        from bartorch.nlop.bundle import of_composition

        return of_composition(self, self._composition())

    def _read(self, which: int) -> Shape:
        """One of BART's dimension vectors, as a C-order shape.

        BART answers in all sixteen axes; what comes back has the rank of the
        image, or more when BART put something on an axis beyond it.
        """
        from bartorch import _marshal

        vector = _marshal.dim_vector()
        if library().bartorch_noir_dims(self._noir.ptr, which, DIMS, vector) < 0:
            raise BartError("BART would not report the model's dimensions")
        bart = [int(vector[i]) for i in range(DIMS)]
        used = [axis for axis, size in enumerate(bart) if 1 != size]
        rank = max(len(self.coil_image_shape), 1 + used[-1] if used else 1)
        return tuple(bart[:rank][::-1])

    # --- what the model is made of -----------------------------------------

    def _linop(self, fn, ishape: Shape, oshape: Shape):
        return _Part(fn, self._noir, ishape, oshape, self.device)

    @property
    def coils(self):
        """Coil coefficients to sensitivities: the Sobolev weighting and the transform.

        A fit returns coefficients; applying this operator gives the
        sensitivity maps they represent.
        """
        return self._linop(
            library().bartorch_noir_coils, self.ishapes[1], self._read(_SHAPES["coils"])
        )

    @property
    def image(self):
        """The image's own linear part -- the mask, and the real constraint if asked for."""
        shape = self.ishapes[0]
        return self._linop(library().bartorch_noir_image, shape, shape)

    @property
    def data(self):
        """From what the model returns to the measurement.

        Off the grid this is the NUFFT, whose adjoint grids a measurement into
        the coil images the model returns; on the grid it is the identity.
        :meth:`prepare` is the one call that matters.
        """
        return self._linop(library().bartorch_noir_data, self.oshapes[0], self.kspace_shape)

    @property
    def transform(self):
        """The transform from coil images to k-space, without the model in front of it."""
        return self._linop(
            library().bartorch_noir_transform,
            self._read(_SHAPES["coil_images"]),
            self.kspace_shape,
        )

    def prepare(self, kspace: torch.Tensor) -> torch.Tensor:
        """The measurement in the shape the model returns.

        Off the grid this is ``nufft^H(kspace)``, as ``noir2_recon`` applies
        before the first Gauss-Newton step; on the grid the measurement is
        returned unchanged.
        """
        if not self.noncart:
            return as_operand(kspace, self.kspace_shape, "kspace")
        return self.data.adjoint(kspace)

    def __repr__(self) -> str:
        where = "noncartesian" if self.noncart else "cartesian"
        return f"NonlinearSense({self.image_shape}, {where})"


def CartesianSense(  # noqa: N802  (it is a constructor)
    image_shape: Shape,
    pattern: torch.Tensor | None = None,
    **kwargs,
) -> NonlinearSense:
    """Cartesian joint image and coil-sensitivity forward model.

    The nonlinear counterpart of :func:`bartorch.linop.CartesianSense`, with
    the sensitivities a second unknown rather than a fixed tensor; this is the
    model ``nlinv`` inverts without a trajectory.  The coil unknown is the
    Sobolev-weighted k-space representation described in
    :class:`NonlinearSense`, which documents the arguments.
    """
    if kwargs.get("trajectory") is not None:
        raise ValueError("a trajectory makes the non-Cartesian model; use NoncartesianSense")
    return NonlinearSense(image_shape, pattern=pattern, **kwargs)


def NoncartesianSense(  # noqa: N802  (it is a constructor)
    trajectory: torch.Tensor,
    image_shape: Shape,
    **kwargs,
) -> NonlinearSense:
    """Non-Cartesian joint image and coil-sensitivity forward model.

    The nonlinear counterpart of :class:`bartorch.linop.NoncartesianSense`, the
    model ``nlinv`` inverts off the grid.  The coil unknown is the Sobolev-weighted
    k-space representation described in :class:`NonlinearSense`.  The model is
    asymmetric, as BART builds it: it returns gridded coil images, and
    :meth:`NonlinearSense.prepare` puts a measurement in that shape.
    """
    return NonlinearSense(image_shape, trajectory=trajectory, **kwargs)


def CoilSense(  # noqa: N802  (it is a constructor)
    encoding,
    image_shape: Shape | None = None,
    coil_shape: Shape | None = None,
    *,
    items: bool = False,
) -> NonlinearOperator:
    """Product of an image and unknown coil sensitivities, through any encoding.

    The general construction rather than BART's particular model: a product of
    two unknowns in front of any linear operator from coil images to data -- a
    wave encoding, a field-corrected one, a subspace one, or one of your own::

        _chain(Multiply(image_shape, coil_shape), encoding.to_nonlinear())

    Unlike :class:`NonlinearSense` there is no Sobolev weighting on the coils:
    the sensitivities themselves are the unknown, and nothing constrains them
    to be smooth.  Regularize them by chaining a smoothing operator onto the
    coil input, or use :class:`NonlinearSense`, which carries BART's.

    Parameters
    ----------
    encoding : LinearOperator
        From coil images to data.  Its domain is the coil-image shape.
    image_shape : tuple of int, default=None
        Where the image lives; by default the encoding's domain with one coil.
    coil_shape : tuple of int, default=None
        Where the sensitivities live; by default the encoding's domain.
    items : bool, default=False
        The encoding's leading axis holds independent items, each with its own
        image and coils; the coils are the next axis.  A Gauss-Newton step then
        applies the model to every item at once and solves each item's inner
        problem on its own.

    Returns
    -------
    NonlinearOperator
        Two inputs, the image and the coils, and one output, the data.

    Examples
    --------
    >>> E = linop.Diagonal(pattern, shape) @ linop.FFT(shape, axes=(-1, -2))
    >>> F = CoilSense(E)
    >>> F.ishapes
    ((1, 128, 128), (8, 128, 128))
    """
    cim = tuple(encoding.ishape)
    lead = cim[:1] if items else ()
    body = cim[1:] if items else cim
    image_shape = tuple(image_shape) if image_shape is not None else lead + (1,) + body[1:]
    coil_shape = tuple(coil_shape) if coil_shape is not None else cim
    if items and not (image_shape[0] == coil_shape[0] == cim[0]):
        raise ValueError(
            f"items lead every shape: the image {image_shape}, the coils {coil_shape} and "
            f"the encoding's domain {cim} begin with different counts"
        )
    made = _chain(Multiply(image_shape, coil_shape), encoding.to_nonlinear(), output=0, input=0)
    if items:
        made.items = cim[0]
    return made
