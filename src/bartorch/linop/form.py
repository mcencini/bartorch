"""The MRI encoding form, and the plan an encoding is lowered into.

Every encoding this library builds reduces to one expression, for coil ``c``,
encoding frame ``t`` and sample ``k``::

    y[c, t, k] = sum_a  O[a, t](k) . T_t( I[c, a, t](r) . x[a](r) )(k)

with ``I`` the image-side element-wise factors, ``T`` the transform, ``O`` the
k-space element-wise factors and ``sum_a`` a contraction.  :class:`Form` is
that expression as the library's one encoding entry point takes it, and
:class:`Plan` is what it reports about itself.

Composing in Python builds a description; matching and lowering happen once,
when the operator is built, and each application is then one call into the
library.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace

import torch

from bartorch import _abi, _marshal
from bartorch._lib import library
from bartorch._operator import dims

__all__ = ["Array", "Contraction", "Factor", "Form", "Plan"]

#: The transforms, by the name the form uses and the header's enumerator.
TRANSFORMS = {
    "none": _abi.BARTORCH_ENCODING_NONE,
    "fft": _abi.BARTORCH_ENCODING_FFT,
    "nufft": _abi.BARTORCH_ENCODING_NUFFT,
    "wave": _abi.BARTORCH_ENCODING_WAVE,
}


def _vector(v) -> tuple[int, ...]:
    """A BART dimension vector as the C entry points take it.

    ``dims`` reverses a C-order shape into BART's order, so a vector already
    in BART's order goes through reversed.
    """
    return dims(tuple(v)[::-1])


@dataclass(frozen=True)
class Array:
    """One array the form holds by pointer, with the BART dimensions it lies on.

    ``vector`` is in BART's order, so it is what :func:`bartorch._layout.vector`
    produces rather than a C-order shape.
    """

    tensor: torch.Tensor
    vector: tuple[int, ...]

    @property
    def pointer(self) -> int:
        return self.tensor.data_ptr()

    @property
    def dimensions(self) -> tuple[int, ...]:
        return _vector(self.vector)


@dataclass(frozen=True)
class Contraction:
    """A contraction over terms, each with a factor on either side of the transform.

    ``sum_l diag(sample_l) E diag(image_l)``: time segmentation for
    off-resonance, with ``count`` terms held contiguously one after another.
    """

    count: int
    sample: Array
    image: Array


@dataclass(frozen=True)
class Factor:
    """One element-wise factor of the form, as :class:`Plan` reports it.

    ``varies`` names the axis roles the factor differs along -- ``coils``,
    ``sets``, ``frames``, ``terms``, ``voxels``, ``samples`` -- which is what
    decides whether it can be streamed with a slab and whether it enters the
    normal's kernel.  ``held`` is ``"whole"`` unless the factor is compressed,
    as coil kernels and a table of phase-encode positions are.
    """

    name: str
    shape: tuple[int, ...]
    varies: tuple[str, ...] = ()
    held: str = "whole"


@dataclass(frozen=True)
class Plan:
    """What the planner chose for one encoding.

    Attributes
    ----------
    transform : str
        ``"none"``, ``"fft"``, ``"nufft"`` or ``"wave"``.
    image, kspace : tuple of Factor
        The element-wise factors on each side of the transform.
    contraction : str or None
        What the sum over ``a`` is: ``"subspace"`` where a basis contracts the
        coefficients, ``"segments"`` where terms are folded into the form's
        own contraction, ``"chained"`` where a sum of terms was left as BART's
        sum of chains because it did not match the form, and ``None`` where
        there is none.
    terms : int
        How many terms the contraction has; one without one.
    normal : str
        ``"kernel"``, ``"transform"`` or ``"applications"``; see
        :attr:`Form.normal`.
    coil_batch : int
        Coils in a slab; 0 is every coil at once, which is BART's own chain.
    streamed : tuple of str
        What is walked a slab at a time rather than held whole.
    executor : str
        ``"slab"`` where the slab loop took the form and ``"chain"`` where it
        could not and BART's plain chain of operators answers instead.  Read
        back from the library after the operator is built, so it is what ran
        and not what was intended.
    cartesian : tuple of str
        Axes of a trajectory found on the image's own grid and transformed by
        an FFT, the NUFFT then over the rest: ``("z",)`` for a stack.
    items : int
        Items along encoding axes the trajectory varies along, each its own
        transform and normal kernel under the coil loop; one without them.
    """

    transform: str
    image: tuple[Factor, ...] = ()
    kspace: tuple[Factor, ...] = ()
    contraction: str | None = None
    terms: int = 1
    normal: str = "applications"
    coil_batch: int = 1
    streamed: tuple[str, ...] = ()
    executor: str = "slab"
    cartesian: tuple[str, ...] = ()
    items: int = 1

    @property
    def fused(self) -> bool:
        """Whether every part of the plan runs in the slab executor.

        False where the slab loop could not take the form, and where a sum of
        terms was left as BART's sum of chains rather than folded into the
        form's contraction.
        """
        return self.executor == "slab" and self.contraction != "chained"

    def __repr__(self) -> str:
        parts = [f"transform={self.transform}"]
        if self.cartesian:
            parts.append("cartesian=" + ",".join(self.cartesian))
        if self.items > 1:
            parts.append(f"items={self.items}")
        if self.image:
            parts.append("image=" + "·".join(f.name for f in self.image))
        if self.kspace:
            parts.append("kspace=" + "·".join(f.name for f in self.kspace))
        if self.contraction is not None:
            parts.append(f"contraction={self.contraction}({self.terms})")
        parts.append(f"normal={self.normal}")
        parts.append(f"coil_batch={self.coil_batch}")
        if self.streamed:
            parts.append("streamed=" + ",".join(self.streamed))
        parts.append(f"executor={self.executor}")
        return f"Plan({', '.join(parts)})"


@dataclass
class Form:
    """One encoding, as ``struct bartorch_encoding`` takes it.

    The fields are the C structure's, in the library's own terms: BART
    dimension vectors and the tensors they describe.  :meth:`build` fills the
    structure and asks for the operator; :attr:`plan` says what the form is
    without building anything.

    Attributes
    ----------
    transform : str
        A key of :data:`TRANSFORMS`.
    max_vector : tuple of int
        Every axis of one batch item -- spatial, coils, sets, coefficients.
    kspace_vector : tuple of int
        One batch item's samples, with the coefficients a basis contracts
        still on them.  A grid transform works these out from ``max_vector``
        and leaves this as the image's.
    positions : tensor, optional
        Phase-encode places as int64 ``(frames, shots, components)``, ``-1``
        for padding, instead of a dense pattern.
    coeffs : int
        Coefficients the image carries, for the plan's report alone.
    stacked : bool
        A stack on the image's own z grid, decoupled: ``traj`` is its in-plane
        part over one position's samples, and z is a batch of the transform.
    stack_positions : tensor, optional
        int64 positions along z of a stack's blocks, one per block, where they
        are not every position in order.
    item_vector : tuple of int, optional
        The encoding axes each item of which has its own trajectory, in BART's
        order: their extents, and one elsewhere.
    """

    transform: str
    max_vector: tuple[int, ...]
    kspace_vector: tuple[int, ...]
    sensitivities: Array
    kernels: bool = False
    pattern: Array | None = None
    basis: Array | None = None
    weights: Array | None = None
    traj: Array | None = None
    stacked: bool = False
    stack_positions: torch.Tensor | None = None
    item_vector: tuple[int, ...] | None = None
    positions: torch.Tensor | None = None
    frames: int = 1
    shots: int = 0
    components: int = 0
    kspace_readout: bool = True
    readout: int = 0
    psf: torch.Tensor | None = None
    centred: bool = False
    contraction: Contraction | None = None
    slice_phase: Array | None = None
    batch_dim: int = -1
    toeplitz: bool = True
    modulated: bool = False
    coil_batch: int = 1
    fold_maps: bool = True
    coils: int = 1
    sets: int = 1
    coeffs: int = 1
    _keep: tuple = field(default_factory=tuple)

    def with_contraction(self, contraction: Contraction) -> Form:
        """This form with a contraction over terms around its transform."""
        return replace(self, contraction=contraction)

    # --- what the form is ----------------------------------------------------

    @property
    def normal(self) -> str:
        """How ``A^H A`` is applied.

        ``"kernel"`` where the k-space side collapses into one kernel applied
        between the transforms, or into a point spread function convolved
        with; ``"transform"`` where there is no k-space factor and the
        transform's own normal is the whole of it; ``"applications"`` where it
        is the forward followed by the adjoint.

        A contraction over terms has no closed form: the sum is BART's, and
        its normal is the two applications.  The wave transform puts its
        kernel between the phase-encode transforms, so a pattern that varies
        along the readout has no closed form either -- the readout is not the
        transform's to cancel.
        """
        if not self.toeplitz or self.contraction is not None or self.slice_phase is not None:
            return "applications"
        if self.transform == "nufft":
            return "kernel"
        if self.pattern is None and self.basis is None and self.positions is None:
            return "transform"
        if self.transform == "wave" and self.pattern is not None:
            return "kernel" if self.pattern.vector[0] == 1 else "applications"
        return "kernel"

    def _image_factors(self) -> tuple[Factor, ...]:
        varies = ("coils",) + (("sets",) if self.sets > 1 else ())
        out = [
            Factor(
                "sensitivities",
                tuple(self.sensitivities.tensor.shape),
                varies,
                "kernels" if self.kernels else "whole",
            )
        ]
        if self.contraction is not None:
            out.append(
                Factor(
                    "segment weights",
                    tuple(self.contraction.image.tensor.shape),
                    ("terms", "voxels"),
                )
            )
        return tuple(out)

    def _kspace_factors(self) -> tuple[Factor, ...]:
        out = []
        if self.basis is not None:
            out.append(Factor("basis", tuple(self.basis.tensor.shape), ("frames", "terms")))
        if self.pattern is not None:
            out.append(Factor("pattern", tuple(self.pattern.tensor.shape), ("samples",)))
        if self.positions is not None:
            out.append(
                Factor("positions", tuple(self.positions.shape), ("frames", "samples"), "table")
            )
        if self.weights is not None:
            out.append(Factor("weights", tuple(self.weights.tensor.shape), ("samples",)))
        if self.slice_phase is not None:
            out.append(
                Factor("slice phase", tuple(self.slice_phase.tensor.shape), ("sets", "samples"))
            )
        if self.contraction is not None:
            out.append(
                Factor(
                    "segment weights",
                    tuple(self.contraction.sample.tensor.shape),
                    ("terms", "samples"),
                )
            )
        return tuple(out)

    def plan(self, executor: str = "slab") -> Plan:
        """The plan this form stands for, with the executor path that ran it."""
        contraction, terms = None, 1
        if self.contraction is not None:
            contraction, terms = "segments", self.contraction.count
        elif self.slice_phase is not None:
            contraction, terms = "slices", self.sets
        elif self.basis is not None:
            contraction, terms = "subspace", self.coeffs

        streamed = ["coils"] if executor == "slab" and self.coil_batch else []
        if self.kernels:
            streamed.append("coil kernels")

        return Plan(
            transform=self.transform,
            image=self._image_factors(),
            kspace=self._kspace_factors(),
            contraction=contraction,
            terms=terms,
            normal=self.normal,
            coil_batch=self.coil_batch,
            streamed=tuple(streamed),
            executor=executor,
            cartesian=("z",) if self.stacked else (),
            items=1 if self.item_vector is None else math.prod(self.item_vector),
        )

    # --- building ------------------------------------------------------------

    def _structure(self):
        """The form as the C structure, with the dimension vectors it points at.

        The vectors are returned beside it: ctypes frees an array as soon as
        nothing holds it, and the structure keeps only their addresses.
        """
        held = []

        def vector(v) -> int:
            array = _vector(v)
            held.append(array)
            return _marshal.address_of(array)

        def array(a: Array | None) -> tuple[int, int]:
            return (0, 0) if a is None else (vector(a.vector), a.pointer)

        s = _abi.Encoding()
        s.transform = TRANSFORMS[self.transform]
        s.max_dims = vector(self.max_vector)
        s.ksp_dims = vector(self.kspace_vector)

        s.sens_dims, s.sens = array(self.sensitivities)
        s.kernels = int(self.kernels)
        s.pat_dims, s.pattern = array(self.pattern)
        s.bas_dims, s.basis = array(self.basis)
        s.wgh_dims, s.weights = array(self.weights)
        s.traj_dims, s.traj = array(self.traj)
        s.stacked = int(self.stacked)
        s.stack_count = 0 if self.stack_positions is None else int(self.stack_positions.numel())
        s.stack_positions = 0 if self.stack_positions is None else self.stack_positions.data_ptr()
        s.item_dims = 0 if self.item_vector is None else vector(self.item_vector)

        s.frames = int(self.frames)
        s.shots = int(self.shots)
        s.components = int(self.components)
        s.positions = 0 if self.positions is None else self.positions.data_ptr()
        s.kspace_readout = int(self.kspace_readout)

        s.readout = int(self.readout)
        s.psf = 0 if self.psf is None else self.psf.data_ptr()
        s.centred = int(self.centred)

        s.segments = 0 if self.contraction is None else int(self.contraction.count)
        if self.contraction is not None:
            s.segment_sample_dims, s.segment_sample = array(self.contraction.sample)
            s.segment_image_dims, s.segment_image = array(self.contraction.image)

        s.slice_dims, s.slice = array(self.slice_phase)
        s.batch_dim = int(self.batch_dim)

        s.toeplitz = int(self.toeplitz)
        s.modulated = int(self.modulated)
        s.coil_batch = int(self.coil_batch)
        s.fold_maps = int(self.fold_maps)
        return s, held

    def build(self, owner, device) -> tuple[int, str]:
        """``(operator, executor)`` for one batch item.

        The executor is read back from the library's counters rather than
        worked out again here, so it is the path that was taken.
        """
        lib = library()
        structure, held = self._structure()
        before = lib.bartorch_encoding_counter(_abi.BARTORCH_ENCODING_BUILT)
        item = owner._under_lock(
            lib.bartorch_linop_encoding, _marshal.by_reference(structure), device=device
        )
        after = lib.bartorch_encoding_counter(_abi.BARTORCH_ENCODING_BUILT)
        del held
        return item, ("slab" if after > before else "chain")

    def keep(self) -> tuple:
        """Every tensor the operator holds by pointer, so it outlives the handle."""
        arrays = (
            self.sensitivities,
            self.pattern,
            self.basis,
            self.weights,
            self.traj,
            self.slice_phase,
        )
        out = [a.tensor for a in arrays if a is not None]
        out += [t for t in (self.positions, self.psf, self.stack_positions) if t is not None]
        if self.contraction is not None:
            out += [self.contraction.sample.tensor, self.contraction.image.tensor]
        return (*out, *self._keep)
