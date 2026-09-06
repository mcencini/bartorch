"""BartLinearPhysics — deepinv LinearPhysics backed by a BartLinop.

This module is imported lazily; deepinv is an optional dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from bartorch.lib.linops import BartLinop


def _make_class() -> type:
    """Build and return the BartLinearPhysics class as a true subclass of
    ``deepinv.physics.LinearPhysics``.

    Construction is deferred to this function so that the class is only
    created when deepinv is actually installed — importing this module does
    not raise an ``ImportError`` if deepinv is absent.
    """
    try:
        from deepinv.physics import LinearPhysics  # noqa: PLC0415
    except ImportError as exc:
        raise ImportError(
            "deepinv is required for BartLinearPhysics.\n"
            "Install it with:  pip install bartorch[deepinv]"
        ) from exc

    class BartLinearPhysics(LinearPhysics):
        """deepinv ``LinearPhysics`` wrapping a :class:`~bartorch.lib.BartLinop`.

        Inherits from ``deepinv.physics.LinearPhysics`` and overrides:

        * ``A`` — BART SENSE forward operator
        * ``A_adjoint`` — BART SENSE adjoint operator
        * ``A_dagger`` — CG pseudo-inverse via BART's ``lsqr + iter_conjgrad``
          running entirely in C with no Python callbacks in the inner loop.

        Parameters
        ----------
        encoding_op : BartLinop
            The BART SENSE encoding operator returned by
            :func:`bartorch.lib.encoding_op`.
        noise_model : deepinv noise model, optional
            Noise model attached to this physics (default ``None``).
        maxiter : int, optional
            Default CG iterations for :meth:`A_dagger` (default 30).
        lam : float, optional
            Default Tikhonov weight for :meth:`A_dagger` (default 0.0).
        tol : float, optional
            Default CG tolerance for :meth:`A_dagger` (default 1e-6).

        Examples
        --------
        >>> import bartorch.tools as bt
        >>> import bartorch.lib as bl
        >>> from bartorch.interop import BartLinearPhysics
        >>>
        >>> ksp  = bt.phantom([128, 128], kspace=True, ncoils=8)
        >>> sens = bt.ecalib(ksp, calib_size=24, maps=1)
        >>> E    = bl.encoding_op(sens)
        >>> phys = BartLinearPhysics(E)
        >>>
        >>> img_gt  = bt.phantom([128, 128])
        >>> y       = phys.A(img_gt)       # forward: image → k-space
        >>> img_rec = phys.A_dagger(y)     # CG reconstruction
        """

        def __init__(
            self,
            encoding_op: BartLinop,
            noise_model: object = None,
            maxiter: int = 30,
            lam: float = 0.0,
            tol: float = 1e-6,
        ) -> None:
            super().__init__(noise_model=noise_model)
            self._op = encoding_op
            self._maxiter = maxiter
            self._lam = lam
            self._tol = tol
            self._ishape = encoding_op.ishape
            self._oshape = encoding_op.oshape

        # ── deepinv LinearPhysics interface ───────────────────────────────

        def A(self, x: torch.Tensor, **kwargs: object) -> torch.Tensor:  # noqa: N802
            """Forward operator ``y = A x``.

            Parameters
            ----------
            x : torch.Tensor
                Image tensor; trailing ``len(ishape)`` dims must match
                :attr:`ishape`.

            Returns
            -------
            torch.Tensor
                K-space tensor with trailing shape :attr:`oshape`.
            """
            return self._apply_batched(x, mode=0)

        def A_adjoint(  # noqa: N802
            self, y: torch.Tensor, **kwargs: object
        ) -> torch.Tensor:
            """Adjoint operator ``x = A^H y``.

            Parameters
            ----------
            y : torch.Tensor
                K-space tensor with trailing shape :attr:`oshape`.

            Returns
            -------
            torch.Tensor
                Image tensor with trailing shape :attr:`ishape`.
            """
            return self._apply_batched(y, mode=1)

        def A_dagger(  # noqa: N802
            self, y: torch.Tensor, **kwargs: object
        ) -> torch.Tensor:
            """CG pseudo-inverse ``x = (A^H A + lam I)^{-1} A^H y``.

            Keyword arguments override the defaults set at construction.

            Parameters
            ----------
            y : torch.Tensor
                K-space measurement tensor.
            maxiter : int, optional
                CG iterations.
            lam : float, optional
                Tikhonov weight.
            tol : float, optional
                Convergence tolerance.

            Returns
            -------
            torch.Tensor
                Reconstructed image tensor.
            """
            maxiter = int(kwargs.get("maxiter", self._maxiter))
            lam = float(kwargs.get("lam", self._lam))
            tol = float(kwargs.get("tol", self._tol))
            return self._solve_batched(y, maxiter=maxiter, lam=lam, tol=tol)

        # ── Helpers ───────────────────────────────────────────────────────

        def _apply_batched(self, x: torch.Tensor, mode: int) -> torch.Tensor:
            """Apply the operator to a possibly-batched tensor."""
            op_shape = self._oshape if mode == 0 else self._ishape
            in_shape = self._ishape if mode == 0 else self._oshape
            extra = x.shape[: x.ndim - len(in_shape)]
            flat = x.reshape(-1, *in_shape)
            results = [
                self._op._handle.apply(flat[i].to(torch.complex64), mode)
                for i in range(flat.shape[0])
            ]
            out = torch.stack(results).reshape(*extra, *op_shape)
            return out

        def _solve_batched(
            self, y: torch.Tensor, *, maxiter: int, lam: float, tol: float
        ) -> torch.Tensor:
            """Solve the normal equation for a possibly-batched tensor."""
            extra = y.shape[: y.ndim - len(self._oshape)]
            flat = y.reshape(-1, *self._oshape)
            results = [
                self._op._handle.solve(flat[i].to(torch.complex64), maxiter, lam, tol)
                for i in range(flat.shape[0])
            ]
            out = torch.stack(results).reshape(*extra, *self._ishape)
            return out

    return BartLinearPhysics


# Module-level attribute — constructed lazily on first access.
_BartLinearPhysics: type | None = None


def _get_class() -> type:
    global _BartLinearPhysics  # noqa: PLW0603
    if _BartLinearPhysics is None:
        _BartLinearPhysics = _make_class()
    return _BartLinearPhysics


class _LazyProxy:
    """Proxy that behaves like BartLinearPhysics but constructs it lazily."""

    def __call__(self, *args: object, **kwargs: object) -> object:
        return _get_class()(*args, **kwargs)

    def __instancecheck__(self, instance: object) -> bool:
        return isinstance(instance, _get_class())

    def __subclasscheck__(self, subclass: type) -> bool:
        return issubclass(subclass, _get_class())

    def __repr__(self) -> str:
        return "<class 'bartorch.interop._deepinv.BartLinearPhysics'>"


BartLinearPhysics: type = _LazyProxy()  # type: ignore[assignment]

__all__ = ["BartLinearPhysics"]
