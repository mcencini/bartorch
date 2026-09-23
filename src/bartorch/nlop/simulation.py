"""Signal models from TorchSim, as BART nonlinear operators.

:class:`SignalModel` bridges TorchSim's
:class:`~torchsim.recon.ModelOperator` -- a model's value, its
Jacobian-vector product and its adjoint product, none of which builds a
Jacobian -- onto the three things BART's ``nlop_s`` asks for.
:func:`InversionRecovery`, :func:`MultiEcho` and :func:`Bloch` are the
quantitative models built on TorchSim's simulators, in TorchSim's
parameterisation.  The suite holds their curves and their fits against BART's
own signal models.

Notes
-----
TorchSim's parameter maps are real and stacked on the **last** axis.  BART
works in ``complex float`` on a C-order shape with the channels in front, so
the bridge moves the axis and carries the maps in the real part of a complex
buffer.  The imaginary half is an exact null direction of the derivative --
nothing reads it and the adjoint returns zero there -- so an iterate that
starts real stays real to the bit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch

from bartorch._operator import Shape
from bartorch.nlop.callback import _Callback

__all__ = ["Bloch", "SignalModel", "InversionRecovery", "MultiEcho"]


def _operator(acquisition, unknown, bounds, scale, amplitude, subspace):
    from torchsim.recon import ModelOperator

    return ModelOperator(
        acquisition,
        *unknown,
        bounds=bounds,
        scale=scale,
        amplitude=amplitude,
        subspace=subspace,
    )


class SignalModel(_Callback):
    """A TorchSim signal model as a BART nonlinear operator.

    The operator maps parameter maps to one image per contrast.  What it does
    at each point is TorchSim's: :meth:`~torchsim.recon.ModelOperator.A` for
    the value, ``A_jvp`` for the derivative and ``A_vjp`` for its adjoint, none
    of which builds a Jacobian -- the model is voxel-diagonal, so one
    forward-mode pass gives the whole volume's derivative whatever the
    parameter count.

    Parameters
    ----------
    model : torchsim.recon.ModelOperator
        The signal model, with its unknowns, bounds and scales already set.
    shape : tuple of int, default=()
        The voxel shape, C order -- ``(y, x)``, ``(z, y, x)``, whatever the
        maps are.  The operator's domain is ``(channels, *shape)`` and its
        codomain ``(contrasts, *shape)``.
    contrasts : int, default=None
        How many images the model returns.  Measured from the model when it
        is not given.

    Attributes
    ----------
    channels : int
        Map channels the domain carries, in TorchSim's order.

    Examples
    --------
    >>> from torchsim.recon import ModelOperator
    >>> from torchsim.simulators import MultiEchoSimulator
    >>> model = ModelOperator(
    ...     MultiEchoSimulator(TE=echo_times), "T2", bounds={"T2": (10.0, 300.0)}
    ... )
    >>> M = SignalModel(model, (128, 128))
    >>> M.ishape, M.oshape
    ((3, 128, 128), (8, 128, 128))
    >>> images = M(M.initial(T2=80.0))

    Under an encoding:

    >>> F = encoding @ M
    >>> maps = nlop.IRGNM()(kspace, F, x0=M.initial(T2=80.0))
    >>> M.split(maps)["T2"]
    """

    def __init__(self, model, shape: Shape = (), contrasts: int | None = None):
        self.model = model
        self.voxels = tuple(shape)
        self.channels = int(model.channels)
        if contrasts is None:
            contrasts = int(model.A(model.initial(())).shape[-1])
        self.contrasts = int(contrasts)

        state: dict[str, torch.Tensor] = {}

        def forward(x: torch.Tensor) -> torch.Tensor:
            maps = self._to_maps(x)
            state["x"] = maps
            return self._to_bart(model.A(maps))

        def derivative(dx: torch.Tensor) -> torch.Tensor:
            return self._to_bart(model.A_jvp(state["x"], self._to_maps(dx)))

        def adjoint(dy: torch.Tensor) -> torch.Tensor:
            return self._to_bart(model.A_vjp(state["x"], self._cotangent(dy)))

        super().__init__(
            (self.contrasts, *self.voxels),
            (self.channels, *self.voxels),
            forward,
            derivative,
            adjoint,
        )

    # --- the two layouts ---------------------------------------------------

    def _to_maps(self, x: torch.Tensor) -> torch.Tensor:
        """BART's ``(channels, *voxels)`` complex to TorchSim's real ``(*voxels, channels)``."""
        return x.reshape(self.channels, *self.voxels).movedim(0, -1).real.contiguous()

    def _to_bart(self, x: torch.Tensor) -> torch.Tensor:
        """The way back, into the real part of a complex buffer."""
        return x.movedim(-1, 0).to(torch.complex64).contiguous()

    def _cotangent(self, dy: torch.Tensor) -> torch.Tensor:
        """One image per contrast, in TorchSim's layout."""
        return dy.reshape(self.contrasts, *self.voxels).movedim(0, -1).contiguous()

    def _bundle(self):
        """TorchSim takes the point as an argument already, so the bundle is its own pair.

        ``A_jvp`` and ``A_vjp`` are ``(x, v)`` throughout -- no Jacobian is
        built and no point is stored, as a bundle requires; the
        members are torch operators so that a step differentiating by the point
        differentiates them again rather than reading a derivative this
        recorded.
        """
        from bartorch.nlop.bundle import Bundle
        from bartorch.nlop.callback import TorchOperator

        model = self.model

        def derivative(dx: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self._to_bart(model.A_jvp(self._to_maps(x), self._to_maps(dx)))

        def adjoint(dy: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
            return self._to_bart(model.A_vjp(self._to_maps(x), self._cotangent(dy)))

        return Bundle(
            self,
            TorchOperator(derivative, [self.ishape, self.ishape], self.oshape),
            TorchOperator(adjoint, [self.oshape, self.ishape], self.ishape),
            source="torch",
        )

    @property
    def names(self) -> tuple[str, ...]:
        """What each channel of the domain is, in order."""
        return tuple(self.model.names)

    def initial(self, **values: Any) -> torch.Tensor:
        """Maps to start from, in this operator's layout.

        Accepts the arguments of
        :meth:`~torchsim.recon.ModelOperator.initial` -- ``{name: value}`` in
        each property's own units -- and returns a complex tensor of
        :attr:`ishape`.
        """
        maps = self.model.initial(self.voxels, **values)
        return maps.movedim(-1, 0).to(torch.complex64).contiguous()

    def split(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """The named maps ``x`` stands for, in their own units.

        The inverse of the packing :meth:`initial` does: what a fit returns is
        the variables actually solved for, and this turns them back
        into a T1 in milliseconds and a complex amplitude.
        """
        maps = x.reshape(self.channels, *self.voxels).movedim(0, -1).real.contiguous()
        return self.model.split(maps)

    def __repr__(self) -> str:
        return (
            f"SignalModel({type(self.model.acquisition).__name__}, {self.voxels}, "
            f"unknown={list(self.model.unknown)})"
        )


def _from(acquisition, unknown, shape, bounds, scale, amplitude, subspace, contrasts):
    return SignalModel(
        _operator(acquisition, unknown, bounds, scale, amplitude, subspace),
        shape,
        contrasts,
    )


def InversionRecovery(  # noqa: N802  (it is a constructor)
    TI: Sequence[float],  # noqa: N803  (BART and the literature spell it this way)
    shape: Shape = (),
    *,
    TR: float | None = None,  # noqa: N803
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    unknown: Sequence[str] = ("T1",),
    amplitude: bool = True,
    subspace: Any = None,
    **scale: float,
) -> SignalModel:
    """T1 from an inversion-recovery series.

    The longitudinal magnetization read at a series of inversion times.  The
    unknowns are ``T1`` and, with ``amplitude``, a complex amplitude; the
    Look-Locker parameterisation ``(Mss, M0, R1*)`` describes the same recovery
    in different variables.

    Parameters
    ----------
    TI : sequence of float
        Inversion times, in milliseconds.
    shape : tuple of int, default=()
        The voxel shape, C order.
    TR : float, default=None
        Repetition time in milliseconds; without one the magnetization is
        fully relaxed when each inversion arrives.
    bounds : dict, default=None
        ``{name: (low, high)}``; ``T1`` defaults to ``(10, 5000)`` ms, which
        keeps a Gauss-Newton iterate physical.
    unknown : sequence of str, default=('T1',)
        What to solve for.  ``inv_efficiency`` and ``offset`` are the other
        things the model exposes.
    amplitude : bool, default=True
        Carry a complex amplitude multiplying the recovery.
    subspace : torchsim.Subspace, default=None
        Solve in a temporal basis rather than in the contrasts.
    **scale
        The size of a step in a parameter left unbounded.
    """
    from torchsim.simulators import InversionRecoverySimulator

    bounds = {"T1": (10.0, 5000.0), **(bounds or {})}
    bounds = {name: bound for name, bound in bounds.items() if name in tuple(unknown)}
    sequence = (
        InversionRecoverySimulator(TI=TI)
        if TR is None
        else (InversionRecoverySimulator(TI=TI, TR=TR))
    )
    return _from(
        sequence, tuple(unknown), shape, bounds, scale or None, amplitude, subspace, len(TI)
    )


def MultiEcho(  # noqa: N802  (it is a constructor)
    TE: Sequence[float],  # noqa: N803
    shape: Shape = (),
    *,
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    unknown: Sequence[str] = ("T2",),
    amplitude: bool = True,
    subspace: Any = None,
    **scale: float,
) -> SignalModel:
    """T2 or T2* from a multi-echo readout.

    The transverse decay read at a series of echo times.  Which relaxation is
    measured is a property of the sequence that produced the data, not of the
    model: a spin-echo train measures T2 and a gradient-echo train T2*, and the
    exponential is the same either way.

    Parameters
    ----------
    TE : sequence of float
        Echo times, in milliseconds.
    shape : tuple of int, default=()
        The voxel shape, C order.
    bounds : dict, default=None
        ``{name: (low, high)}``; ``T2`` defaults to ``(1, 1000)`` ms.
    unknown : sequence of str, default=('T2',)
        What to solve for.  ``offset`` is the other thing the model exposes.
    amplitude : bool, default=True
        Carry a complex amplitude multiplying the decay.
    subspace : torchsim.Subspace, default=None
        Solve in a temporal basis rather than in the contrasts.
    **scale
        The size of a step in a parameter left unbounded.
    """
    from torchsim.simulators import MultiEchoSimulator

    bounds = {"T2": (1.0, 1000.0), **(bounds or {})}
    bounds = {name: bound for name, bound in bounds.items() if name in tuple(unknown)}
    return _from(
        MultiEchoSimulator(TE=TE),
        tuple(unknown),
        shape,
        bounds,
        scale or None,
        amplitude,
        subspace,
        len(TE),
    )


def Bloch(  # noqa: N802  (it is a constructor)
    acquisition,
    *unknown: str,
    shape: Shape = (),
    bounds: dict[str, tuple[float | None, float | None]] | None = None,
    amplitude: bool = True,
    subspace: Any = None,
    contrasts: int | None = None,
    **scale: float,
) -> SignalModel:
    """Any TorchSim sequence as a model operator, through Bloch simulation.

    Fits a Bloch simulation of the sequence rather than a closed-form signal
    equation, so it serves sequences that have none: an FSE train, a
    fingerprinting schedule, a bSSFP sweep, or a sequence of your own --
    anything with a ``simulate`` becomes an operator Gauss-Newton can solve.

    Parameters
    ----------
    acquisition : torchsim Simulator
        The sequence, with everything not being solved for already fixed on
        it.  A property bound as a map -- a measured B1, a known T1 -- is one
        value per voxel and rides along.
    *unknown : str
        The properties being solved for, in the order their channels appear.
    shape : tuple of int, default=()
        The voxel shape, C order.
    bounds : dict, default=None
        ``{name: (low, high)}``, either end ``None`` for unbounded.  A bound
        is kept by solving for a transformed variable, so no iterate leaves
        it.
    amplitude : bool, default=True
        Carry a complex amplitude multiplying the simulated signal.
    subspace : torchsim.Subspace, default=None
        Solve in a temporal basis rather than in the contrasts.
    contrasts : int, default=None
        How many images the sequence records; measured when not given.
    **scale
        The size of a step in a parameter left unbounded.

    Examples
    --------
    >>> from torchsim.simulators import FSESimulator
    >>> M = Bloch(
    ...     FSESimulator(flip=train, ESP=8.0, TR=3000.0),
    ...     "T1", "T2",
    ...     shape=(128, 128),
    ...     bounds={"T1": (100.0, 4000.0), "T2": (5.0, 500.0)},
    ... )
    """
    return _from(acquisition, unknown, shape, bounds, scale or None, amplitude, subspace, contrasts)
