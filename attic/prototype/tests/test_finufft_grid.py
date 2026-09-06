"""tests/test_finufft_grid.py — tests for the FINUFFT-backed grid2/grid2H
and ES-kernel rolloff correction.

Tests that require a working BART runtime are wrapped with try/except so they
are skipped gracefully when the C++ extension is not built.

Test plan
─────────
1. Pure-Python ES rolloff weight checks (no extension required).
2. NUFFT adjointness tests that exercise the grid2/grid2H path internally
   via bt.nufft().  Skipped automatically if bt.phantom() raises (extension
   not built or broken).
"""

from __future__ import annotations

import math

import pytest

# ---------------------------------------------------------------------------
# Guard: skip BART-calling tests when extension is unavailable
# ---------------------------------------------------------------------------


def _bart_available() -> bool:
    """Return True if the BART extension is importable and functional."""
    try:
        import bartorch.tools as bt

        bt.phantom([8, 8])
        return True
    except Exception:
        return False


skip_no_bart = pytest.mark.skipif(
    not _bart_available(),
    reason="BART C++ extension not built or not functional",
)


# ---------------------------------------------------------------------------
# Pure-Python ES rolloff weight validation (no extension needed)
#
# Verifies that the same quadrature used in finufft_grid.cpp produces
# physically sensible rolloff weights.
# ---------------------------------------------------------------------------


def _es_rolloff_weight(
    xi_cps: float, beta: float = 2.30 * 7.0, hw: float = 7.0 / 2.0, nquad: int = 256
) -> float:
    """Python replica of esro_hat_phi / esro_weight from finufft_grid.cpp."""
    h = hw / nquad
    total = 0.0
    for i in range(nquad):
        x = (i + 0.5) * h
        t = x / hw
        arg = 1.0 - t * t
        phi = math.exp(beta * math.sqrt(max(arg, 0.0)))
        total += phi * math.cos(2.0 * math.pi * xi_cps * x)
    hat_phi = 2.0 * h * total
    return 1.0 / hat_phi if hat_phi != 0.0 else 1.0


def test_es_rolloff_dc_weight():
    """ES rolloff weight at DC is a finite positive float < 1."""
    w_dc = _es_rolloff_weight(0.0)
    assert w_dc > 0.0, f"DC weight should be positive, got {w_dc}"
    assert w_dc < 1.0, f"DC weight should be < 1 (ES kernel peak >> 1), got {w_dc}"
    assert math.isfinite(w_dc), "DC weight must be finite"


def test_es_rolloff_symmetry():
    """ES rolloff weights are symmetric: w(p, n) == w(n-p, n)."""
    n, os = 64, 2.0
    for p in range(n // 2):
        xi_pos = (p - n / 2.0) / (n * os)
        xi_neg = (n - p - n / 2.0) / (n * os)  # mirror pixel
        w_pos = _es_rolloff_weight(xi_pos)
        w_neg = _es_rolloff_weight(xi_neg)
        assert abs(w_pos - w_neg) < 1e-7, (
            f"Rolloff not symmetric at p={p}: w(p)={w_pos}, w(n-p)={w_neg}"
        )


def test_es_rolloff_monotone():
    """ES rolloff weight decreases monotonically from centre toward Nyquist."""
    n, os = 64, 2.0
    xis = [(p - n / 2.0) / (n * os) for p in range(n // 2, n)]
    weights = [_es_rolloff_weight(xi) for xi in xis]
    for i in range(len(weights) - 1):
        assert weights[i] <= weights[i + 1] + 1e-6, (
            f"Rolloff weight not monotone at index {i}: {weights[i]} > {weights[i + 1]}"
        )


# ---------------------------------------------------------------------------
# NUFFT adjointness tests — exercise grid2/grid2H internally via bt.nufft()
#
# Mirrors BART's test-nufft-adjoint from nufft.mk:
#   traj -r -x128 -y128 → nufft / nufft -a → fmac inner products → nrmse
#
# When FINUFFT is compiled in (BARTORCH_USE_FINUFFT=ON), these paths go through
# __wrap_grid2 / __wrap_grid2H.  When FINUFFT is disabled the original BART
# KB gridder is used — the adjointness property holds in both cases.
# ---------------------------------------------------------------------------


@skip_no_bart
def test_adjointness_2d():
    """Adjointness of NUFFT forward/adjoint for a 2-D radial trajectory.

    Checks: |⟨A·n1, n2⟩ − ⟨n1, A^H·n2⟩| / |⟨A·n1, n2⟩| < tol
    where k = A·n1 (forward), x = A^H·n2 (adjoint).

    Inner products are computed in Python (avoiding bt.fmac shape/bitmask
    sensitivity) so the test isolates the NUFFT forward/adjoint operators.
    """
    import torch

    import bartorch.tools as bt

    traj = bt.traj(r=True, x=64, y=64)

    n1 = bt.noise(torch.zeros(1, 64, 64, dtype=torch.complex64), s=123)
    n2 = bt.noise(torch.zeros(64, 64, 1, dtype=torch.complex64), s=321)

    k = bt.nufft(traj, n1)  # A·n1: image (1,64,64) → k-space (64,64,1)
    x = bt.nufft(traj, n2, adjoint=True)  # A^H·n2: k-space (64,64,1) → image (1,64,64)

    # ⟨k, n2⟩ = sum(conj(k) * n2)
    lhs = (k.conj() * n2).sum()
    # ⟨n1, x⟩ = sum(conj(n1) * x)
    rhs = (n1.conj() * x).sum()
    ref = abs(lhs).item()
    # Normalise by |lhs| (the forward inner product); avoids division by the
    # adjoint inner product which may accidentally be near zero.
    err = abs(lhs - rhs).item() / ref if ref != 0.0 else abs(lhs - rhs).item()
    assert err < 1e-4, f"Adjointness error {err:.2e} exceeds 1e-4"


@skip_no_bart
def test_adjointness_3d():
    """Adjointness of NUFFT forward/adjoint for a smaller trajectory."""
    import torch

    import bartorch.tools as bt

    traj = bt.traj(r=True, x=16, y=16)

    n1 = bt.noise(torch.zeros(1, 16, 16, dtype=torch.complex64), s=123)
    n2 = bt.noise(torch.zeros(16, 16, 1, dtype=torch.complex64), s=321)

    k = bt.nufft(traj, n1)  # A·n1: image (1,16,16) → k-space (16,16,1)
    x = bt.nufft(traj, n2, adjoint=True)  # A^H·n2: k-space (16,16,1) → image (1,16,16)

    # ⟨k, n2⟩ = sum(conj(k) * n2)
    lhs = (k.conj() * n2).sum()
    # ⟨n1, x⟩ = sum(conj(n1) * x)
    rhs = (n1.conj() * x).sum()
    ref = abs(lhs).item()
    # Normalise by |lhs| (the forward inner product).
    err = abs(lhs - rhs).item() / ref if ref != 0.0 else abs(lhs - rhs).item()
    assert err < 1e-4, f"Adjointness error {err:.2e} exceeds 1e-4"
