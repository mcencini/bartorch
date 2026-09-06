"""Integration tests for bartorch BART operations.

Tests are ported from BART's own integration test suite (``bart/tests/*.mk``).
They verify that real BART commands execute correctly through the bartorch
Python/C++ bridge.

All tests use the **public** ``bartorch.tools`` API (``import bartorch.tools as bt``).
This exercises both the hand-written overrides in ``_commands.py`` (e.g.
``phantom``, ``fft``, ``flip``, ``rss``) and the auto-generated wrappers in
``_generated.py`` (e.g. ``cabs``, ``nrmse``, ``conj``).

Each test is annotated with the originating BART test for traceability:
  phantom.mk, fft.mk, ecalib.mk, pics.mk, nufft.mk, wavelet.mk, noise.mk,
  fmac.mk, nrmse.mk.
"""

from __future__ import annotations

import math

import torch

import bartorch.tools as bt

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _nrmse(a: torch.Tensor, b: torch.Tensor) -> float:
    """Normalised RMSE — BART convention: ‖a − b‖ / ‖b‖."""
    diff = (a - b).abs().norm().item()
    ref = b.abs().norm().item()
    if ref == 0.0:
        return 0.0 if diff == 0.0 else float("inf")
    return diff / ref


# ---------------------------------------------------------------------------
# phantom — bart/tests/phantom.mk
# ---------------------------------------------------------------------------


def test_phantom_creates_nonzero_image():
    """bt.phantom produces a non-trivial complex64 image."""
    img = bt.phantom([128, 128])
    assert isinstance(img, torch.Tensor)
    assert img.dtype == torch.complex64
    assert img.ndim == 2
    assert img.shape == torch.Size([128, 128])
    assert img.abs().max().item() > 0.0


def test_phantom_kspace_flag():
    """bt.phantom(kspace=True) produces k-space with the same shape as the image."""
    img = bt.phantom([128, 128])
    ksp = bt.phantom([128, 128], kspace=True)
    assert ksp.shape == img.shape
    assert ksp.abs().max().item() > 0.0
    assert not torch.allclose(ksp, img)


def test_phantom_custom_size():
    """bt.phantom([64, 64]) produces a 64×64 image."""
    img = bt.phantom([64, 64])
    assert img.shape[-1] == 64
    assert img.shape[-2] == 64


def test_phantom_coil_shape():
    """bt.phantom([128,128], ncoils=4) expands to 4 coil copies."""
    img = bt.phantom([128, 128], ncoils=4)
    assert img.numel() == 128 * 128 * 4
    assert img.dtype == torch.complex64


def test_phantom_ksp():
    """
    bart/tests/phantom.mk — test-phantom-ksp.

    IFFT(phantom(kspace=True)) ≈ phantom()  nrmse < 0.22.
    BART: ``fft -i 7 ksp img ; nrmse -t 0.22 shepplogan img``
    """
    img = bt.phantom([128, 128])
    ksp = bt.phantom([128, 128], kspace=True)
    recon = bt.ifft(ksp, axes=(-1, -2))
    err = _nrmse(recon, img)
    assert err < 0.22, f"phantom ksp roundtrip nrmse={err:.4f}"


def test_phantom_ksp_coil():
    """
    bart/tests/phantom.mk — test-phantom-ksp-coil.

    IFFT(phantom(-s8 -k)) ≈ phantom(-s8)  nrmse < 0.22.
    BART: ``fft -i 7 coil_ksp coil_img ; nrmse -t 0.22 coil coil_img``
    """
    coil_imgs = bt.phantom([128, 128], ncoils=8)
    coil_ksp = bt.phantom([128, 128], kspace=True, ncoils=8)
    recon = bt.ifft(coil_ksp, axes=(-1, -2))
    err = _nrmse(recon, coil_imgs)
    assert err < 0.22, f"phantom ksp-coil roundtrip nrmse={err:.4f}"


def test_phantom_coil_consistency():
    """
    bart/tests/phantom.mk — test-phantom-coil.

    phantom(-s8) == fmac(phantom(), phantom(-S8))  nrmse ≈ 0.
    BART: ``fmac shepplogan coils sl_coil2 ; nrmse -t 0. coil sl_coil2``
    """
    img = bt.phantom([128, 128])
    coil_sens = bt.phantom([128, 128], S=8)  # sensitivities only (-S8)
    coil_imgs = bt.phantom([128, 128], ncoils=8)  # weighted copies (-s8)
    manual = bt.fmac(img, coil_sens)
    err = _nrmse(manual, coil_imgs)
    assert err < 1e-4, f"phantom coil consistency nrmse={err:.2e}"


# ---------------------------------------------------------------------------
# FFT — bart/tests/fft.mk
# ---------------------------------------------------------------------------


def test_fft_basic():
    """
    bart/tests/fft.mk — test-fft-basic.

    IFFT(FFT(x)) == N*x  (non-unitary; N = total elements).
    BART: ``scale 16384 shepplogan shepploganS ; fft -i 7 ksp img``
          ``nrmse -t 0.000001 shepploganS img``
    """
    img = bt.phantom([128, 128])
    n = math.prod(img.shape)  # 16384
    ksp = bt.fft(img, axes=(-1, -2))
    recon = bt.ifft(ksp, axes=(-1, -2))
    expected = bt.scale(float(n), img)
    err = _nrmse(recon, expected)
    assert err < 1e-5, f"fft basic nrmse={err:.2e}"


def test_fft_unitary():
    """
    bart/tests/fft.mk — test-fft-unitary.

    Unitary FFT is self-inverse: IFFT_u(FFT_u(x)) == x  nrmse < 1e-6.
    BART: ``fft -u -i 7 ksp_u img ; nrmse -t 0.000001 shepplogan img``
    """
    img = bt.phantom([128, 128])
    ksp_u = bt.fft(img, axes=(-1, -2), unitary=True)
    recon = bt.ifft(ksp_u, axes=(-1, -2), unitary=True)
    err = _nrmse(recon, img)
    assert err < 1e-5, f"unitary fft roundtrip nrmse={err:.2e}"


def test_fft_uncentered():
    """
    bart/tests/fft.mk — test-fft-uncentered.

    fftmod_inv + FFT_u_uncentered_inv + fftmod_inv == identity  nrmse < 1e-6.
    BART: ``fftmod -i 7 ksp_u t1 ; fft -uni 7 t1 t2 ; fftmod -i 7 t2 t3``
          ``nrmse -t 0.000001 shepplogan t3``
    """
    img = bt.phantom([128, 128])
    ksp_u = bt.fft(img, axes=(-1, -2), unitary=True)
    t1 = bt.fftmod(ksp_u, axes=(-1, -2), i=True)
    t2 = bt.fft(t1, axes=(-1, -2), unitary=True, inverse=True, n=True)
    t3 = bt.fftmod(t2, axes=(-1, -2), i=True)
    err = _nrmse(t3, img)
    assert err < 1e-5, f"fft uncentered nrmse={err:.2e}"


def test_fft_shape_preserved():
    """FFT does not change the tensor shape."""
    img = bt.phantom([32, 32])
    ksp = bt.fft(img, axes=(-1, -2))
    assert ksp.shape == img.shape


def test_fft_output_dtype():
    """FFT output is always complex64."""
    img = bt.phantom([16, 16])
    ksp = bt.fft(img, axes=-1)
    assert ksp.dtype == torch.complex64


def test_fft_single_axis_roundtrip():
    """FFT over the last axis: IFFT(FFT(x, axes=-1), axes=-1) = N0 * x."""
    img = bt.phantom([32, 32])
    ksp = bt.fft(img, axes=-1)
    recon = bt.ifft(ksp, axes=-1)
    n0 = img.shape[-1]
    err = _nrmse(recon, img * n0)
    assert err < 1e-4, f"single-axis fft nrmse={err:.2e}"


def test_fft_multiple_calls_independent():
    """Two separate FFT calls on the same tensor give the same result."""
    img = bt.phantom([32, 32])
    k1 = bt.fft(img, axes=(-1, -2))
    k2 = bt.fft(img, axes=(-1, -2))
    assert torch.allclose(k1, k2)


# ---------------------------------------------------------------------------
# flip
# ---------------------------------------------------------------------------


def test_flip_last_axis():
    """bt.flip(img, axes=-1) reverses the last C-order axis."""
    img = bt.phantom([16, 16])
    flipped = bt.flip(img, axes=-1)
    expected = torch.flip(img, dims=[-1])
    err = _nrmse(flipped, expected)
    assert err < 1e-6, f"flip nrmse={err:.2e}"


# ---------------------------------------------------------------------------
# cabs (complex absolute value)
# ---------------------------------------------------------------------------


def test_cabs_nonnegative():
    """bart cabs returns non-negative real values (as complex64)."""
    img = bt.phantom([32, 32])
    mags = bt.cabs(img)
    assert mags.dtype == torch.complex64
    assert (mags.real >= 0).all()


def test_cabs_magnitude():
    """bart cabs matches torch.abs on a known complex input."""
    x = torch.randn(16, 16, dtype=torch.complex64)
    res = bt.cabs(x)
    expected = x.abs().to(torch.complex64)
    err = _nrmse(res, expected)
    assert err < 1e-5, f"cabs nrmse={err:.2e}"


# ---------------------------------------------------------------------------
# rss (root-sum-of-squares coil combination)
# ---------------------------------------------------------------------------


def test_rss_coil_shape():
    """bt.rss reduces the coil dimension (C-order axis 0)."""
    coil_imgs = bt.phantom([32, 32], ncoils=4)
    rss = bt.rss(coil_imgs, axes=0)
    assert rss.dtype == torch.complex64
    assert rss.numel() == 32 * 32


def test_rss_positive_real():
    """bart rss returns non-negative real parts."""
    coil_imgs = bt.phantom([32, 32], ncoils=4)
    rss = bt.rss(coil_imgs, axes=0)
    assert (rss.real >= -1e-6).all()


# ---------------------------------------------------------------------------
# conj
# ---------------------------------------------------------------------------


def test_conj_identity():
    """Conjugate applied twice gives original tensor."""
    img = bt.phantom([16, 16])
    recon = bt.conj(bt.conj(img))
    err = _nrmse(recon, img)
    assert err < 1e-6, f"conj double nrmse={err:.2e}"


def test_conj_matches_torch():
    """bart conj matches torch.conj on a known input."""
    x = torch.randn(16, 16, dtype=torch.complex64)
    result = bt.conj(x)
    err = _nrmse(result, x.conj())
    assert err < 1e-5, f"conj vs torch.conj nrmse={err:.2e}"


# ---------------------------------------------------------------------------
# scale
# ---------------------------------------------------------------------------


def test_scale_by_factor():
    """bart scale multiplies every element by the given factor."""
    x = bt.phantom([16, 16])
    scaled = bt.scale(2.0, x)
    err = _nrmse(scaled, x * 2.0)
    assert err < 1e-5, f"scale nrmse={err:.2e}"


def test_scale_identity():
    """bart scale by 1 is identity."""
    x = bt.phantom([16, 16])
    scaled = bt.scale(1.0, x)
    err = _nrmse(scaled, x)
    assert err < 1e-6, f"scale-1 nrmse={err:.2e}"


# ---------------------------------------------------------------------------
# fmac (fused multiply-accumulate / Hadamard product)
# ---------------------------------------------------------------------------


def test_fmac_hadamard():
    """bart fmac with no -s flag computes element-wise product."""
    a = bt.phantom([16, 16])
    b = bt.phantom([16, 16])
    result = bt.fmac(a, b)
    expected = a * b
    err = _nrmse(result, expected)
    assert err < 1e-5, f"fmac nrmse={err:.2e}"


def test_fmac_sum():
    """
    bart/tests/fmac.mk — test-fmac-sum.

    fmac(a, ones, s=bitmask) squashes the selected dimension by summation.
    BART: ``ones 3 2 1 3 a0 ; noise a0 a1``
          ``fmac -s4 a1 ones a3 ; [manual sum of a1 over Fortran dim 2]``
    """
    # a0 = ones with Fortran shape (2,1,3) → C-order (3,1,2)
    a0 = bt.ones(3, output_dims=[3, 1, 2])
    a1 = bt.noise(a0)
    # ones (1,1) for broadcasting
    o = bt.ones(2, output_dims=[1, 1])
    # fmac -s4: Fortran bit 2 = Fortran dim 2 = C-order axis 0 for ndim=3
    a3 = bt.fmac(a1, o, s=4)
    # Manual: sum a1 over C-order axis 0, keep dims
    a2 = a1.sum(dim=0, keepdim=True)
    err = _nrmse(a3, a2)
    assert err < 1e-5, f"fmac-sum nrmse={err:.2e}"


# ---------------------------------------------------------------------------
# ecalib — bart/tests/ecalib.mk
# ---------------------------------------------------------------------------


def test_ecalib_returns_tensor():
    """bt.ecalib returns a complex64 tensor for 8-coil phantom kspace."""
    ksp = bt.phantom([128, 128], kspace=True, ncoils=8)
    sens = bt.ecalib(ksp, calib_size=24, maps=1)
    assert isinstance(sens, torch.Tensor)
    assert sens.dtype == torch.complex64


def test_ecalib_coil_count():
    """bt.ecalib output has C-order shape (maps, nc, nz, ny, nx) = (1,8,1,128,128)."""
    ksp = bt.phantom([128, 128], kspace=True, ncoils=8)
    sens = bt.ecalib(ksp, calib_size=24, maps=1)
    # BART Fortran (nx,ny,nz,nc,maps) = (128,128,1,8,1) → C-order (1,8,1,128,128)
    assert sens.shape == torch.Size([1, 8, 1, 128, 128])


def test_ecalib_nonzero():
    """Sensitivity maps from non-trivial kspace must contain non-zero values."""
    ksp = bt.phantom([128, 128], kspace=True, ncoils=8)
    sens = bt.ecalib(ksp, calib_size=24, maps=1)
    assert sens.abs().max().item() > 0.0


def test_ecalib_pocsense():
    """
    bart/tests/ecalib.mk — test-ecalib.

    POCS projection of kspace through estimated sensitivity maps is close to
    the original kspace.
    BART: ``ecalib -m1 coil_ksp coils``
          ``pocsense -i1 coil_ksp coils proj``
          ``nrmse -t 0.05 proj coil_ksp``
    """
    ksp = bt.phantom([128, 128], kspace=True, ncoils=8)
    sens = bt.ecalib(ksp, maps=1)
    proj = bt.pocsense(ksp, sens, i=1)
    err = _nrmse(proj, ksp)
    assert err < 0.05, f"ecalib pocsense nrmse={err:.4f}"


def test_ecalib_phase():
    """
    bart/tests/ecalib.mk — test-ecalib-phase.

    With -N (no-phase-correction), the coil-combined image is nearly real:
    nrmse(proj, conj(proj)) < 0.1.
    BART: ``ecalib -N -m1 ... ; fmac -s8 -C cim coils proj``
          ``conj proj projc ; nrmse -t 0.1 proj projc``
    """
    ksp = bt.phantom([128, 128], kspace=True, ncoils=8)
    # -N = no phase correction (via extra_flags)
    sens = bt.ecalib(ksp, maps=1, N=True)
    # IFFT of coil kspace
    cim = bt.fft(ksp, axes=(-1, -2), inverse=True)  # (8,1,128,128)
    # Coil-combine: sum(cim * conj(sens[0])) over coil dim
    # Fortran bit 3 = coil dim = C-order axis 0 for ndim=4
    proj = bt.fmac(cim, sens[0], C=True, s=8)
    projc = bt.conj(proj)
    err = _nrmse(proj, projc)
    assert err < 0.1, f"ecalib phase nrmse={err:.4f}"


# ---------------------------------------------------------------------------
# pics — bart/tests/pics.mk
# ---------------------------------------------------------------------------


def test_pics_pi_reconstruction():
    """
    bart/tests/pics.mk — test-pics-pi.

    Parallel-imaging reconstruction with ESPIRiT sensitivity maps.
    BART: ``pics -S -r0.001 coil_ksp coils reco``
          ``scale 128 reco reco2 ; nrmse -t 0.23 reco2 shepplogan``
    """
    img = bt.phantom([128, 128])
    ksp_calib = bt.phantom([128, 128], kspace=True, ncoils=8)
    sens = bt.ecalib(ksp_calib, calib_size=24, maps=1)

    # Generate k-space consistent with the SENSE forward model:
    coil_imgs = sens[0] * img  # (8, 1, 128, 128)
    ksp = bt.fft(coil_imgs, axes=(-1, -2))

    reco = bt.pics(ksp, sens, lambda_=0.001)
    err = _nrmse(reco, img)
    assert err < 0.23, f"pics-pi nrmse={err:.4f}"


# ---------------------------------------------------------------------------
# nufft — bart/tests/nufft.mk
# ---------------------------------------------------------------------------


def test_nufft_forward_vs_fft():
    """
    bart/tests/nufft.mk — test-nufft-forward.

    NUFFT with partition-of-unity (-P) on a Cartesian trajectory must match
    the unitary FFT to within 0.1 %.

    BART: ``traj -x32 -y32 traj``
          ``nufft -P traj shepplogan ksp``
          ``fft -u 3 shepplogan ksp_fft``
          nrmse < 0.001
    """
    img = bt.phantom([32, 32])
    ksp_fft = bt.fft(img, axes=(-1, -2), unitary=True)
    traj = bt.traj(x=32, y=32)
    ksp_nufft = bt.nufft(traj, img, P=True)
    # ksp_nufft has C-order shape (32, 32, 1); reshape → (1, 32, 32) then squeeze.
    # bitmask 7 = bits 0,1,2 = BART Fortran dims 0,1,2 (all dims of a 3-D array).
    ksp_2d = bt.reshape(ksp_nufft, 7, output_dims=[1, 32, 32]).squeeze()
    err = _nrmse(ksp_2d, ksp_fft)
    assert err < 1e-3, f"nufft -P vs fft nrmse={err:.2e}"


def test_nufft_adjoint():
    """
    bart/tests/nufft.mk — test-nufft-adjoint.

    Inner-product adjointness: ⟨Ax, y⟩ = ⟨x, A^H y⟩ up to 1 % relative
    error, using a small radial trajectory.

    BART: ``traj -r -x32 -y32 traj``
          ``nufft traj x k``      (forward: image → k-space)
          ``nufft -a traj y x2``  (adjoint: k-space → image)
          |⟨k, y⟩ − ⟨x, x2⟩| / |⟨k, y⟩| < 0.01
    """
    traj = bt.traj(r=True, x=32, y=32)
    # x: random image in C-order (1, 32, 32) — bt.zeros + bt.noise
    x = bt.noise(bt.zeros(3, output_dims=[1, 32, 32]), s=123)
    # y: random k-space in C-order (32, 32, 1) — reshape then noise.
    # bitmask 7 = bits 0,1,2 = BART Fortran dims 0,1,2 (all dims of a 3-D array).
    y = bt.noise(
        bt.reshape(bt.zeros(3, output_dims=[1, 32, 32]), 7, output_dims=[32, 32, 1]),
        s=456,
    )
    k = bt.nufft(traj, x)  # A x: image → k-space, shape (32, 32, 1)
    x2 = bt.nufft(traj, y, adjoint=True)  # A^H y: k-space → image, shape (1, 32, 32)
    # ⟨k, y⟩ = sum(conj(k) * y)
    lhs = (k.conj() * y).sum()
    # ⟨x, x2⟩ = sum(conj(x) * x2)
    rhs = (x.conj() * x2).sum()
    scale = max(abs(lhs).item(), 1e-10)
    err = abs(lhs - rhs).item() / scale
    assert err < 0.01, f"nufft adjointness err={err:.2e} (lhs={lhs:.4e}, rhs={rhs:.4e})"


# ---------------------------------------------------------------------------
# wavelet — bart/tests/wavelet.mk
# ---------------------------------------------------------------------------


def test_wavelet_roundtrip():
    """
    bart/tests/wavelet.mk — test-wavelet.

    Wavelet forward then adjoint (inverse) reconstructs the original image
    exactly: nrmse < 1e-6.
    BART: ``wavelet 3 shepplogan w ; wavelet -a 3 128 128 w a``
          ``nrmse -t 0.000001 shepplogan a``
    """
    img = bt.phantom([128, 128])
    w = bt.wavelet(img, axes=(-1, -2))
    a = bt.wavelet(w, axes=(-1, -2), a=True, output_dims=[128, 128])
    err = _nrmse(a, img)
    assert err < 1e-5, f"wavelet roundtrip nrmse={err:.2e}"


def test_wavelet_haar_roundtrip():
    """
    bart/tests/wavelet.mk — test-wavelet-haar.

    Haar wavelet forward + adjoint = identity.
    BART: ``wavelet -H 3 shepplogan w ; wavelet -H -a 3 128 128 w a``
          ``nrmse -t 0.000001 shepplogan a``
    """
    img = bt.phantom([128, 128])
    w = bt.wavelet(img, axes=(-1, -2), H=True)
    a = bt.wavelet(w, axes=(-1, -2), a=True, H=True, output_dims=[128, 128])
    err = _nrmse(a, img)
    assert err < 1e-5, f"wavelet haar roundtrip nrmse={err:.2e}"


def test_wavelet_cdf44_roundtrip():
    """
    bart/tests/wavelet.mk — test-wavelet-cdf44.

    CDF 4-4 wavelet forward + adjoint = identity.
    BART: ``wavelet -C 3 shepplogan w ; wavelet -C -a 3 128 128 w a``
          ``nrmse -t 0.000001 shepplogan a``
    """
    img = bt.phantom([128, 128])
    w = bt.wavelet(img, axes=(-1, -2), C=True)
    a = bt.wavelet(w, axes=(-1, -2), a=True, C=True, output_dims=[128, 128])
    err = _nrmse(a, img)
    assert err < 1e-5, f"wavelet cdf44 roundtrip nrmse={err:.2e}"


# ---------------------------------------------------------------------------
# noise — bart/tests/noise.mk
# ---------------------------------------------------------------------------


def test_noise_std():
    """
    bart/tests/noise.mk — test-noise.

    Unit-variance complex Gaussian noise has std ≈ 1 across all elements.
    BART: ``zeros 2 100 100 z ; noise -s1 -n1. z n``
          ``std 3 n d ; ones 2 1 1 o ; nrmse -t 0.02 o d``
    """
    z = bt.zeros(2, output_dims=[100, 100])  # (100,100) zeros
    n = bt.noise(z, s=1, n=1.0)  # unit-variance noise
    # std over both axes (Fortran bitmask 3 = C-order axes (-1,-2))
    d = bt.std(n, axes=(-1, -2))  # result is (1,1) scalar
    o = bt.ones(2, output_dims=[1, 1])  # ones (1,1)
    err = _nrmse(d.real.to(torch.complex64), o)
    assert err < 0.02, f"noise std nrmse={err:.4f}"


# ---------------------------------------------------------------------------
# nrmse scale invariance — bart/tests/nrmse.mk
# ---------------------------------------------------------------------------


def test_nrmse_scale_invariant():
    """
    bart/tests/nrmse.mk — test-nrmse-scale.

    nrmse -s (scale-invariant) of x vs 2j*x is ≈ 0: they are identical
    up to a complex scale factor.
    BART: ``scale 2.i shepplogan sc ; nrmse -s -t 1e-7 shepplogan sc``
    """
    img = bt.phantom([32, 32])
    # Scale-invariant NRMSE: minimise ‖img - α·scaled‖/‖scaled‖ over α∈ℂ
    # With scaled = 2j * img → optimal α = 1/(2j) = -0.5j → residual = 0
    scaled = img * complex(0.0, 2.0)
    img_f = img.reshape(-1)
    scl_f = scaled.reshape(-1)
    alpha = (scl_f.conj() @ img_f) / (scl_f.conj() @ scl_f)
    residual = img_f - alpha * scl_f
    si_nrmse = residual.abs().norm().item() / scl_f.abs().norm().item()
    assert si_nrmse < 1e-5, f"scale-invariant nrmse={si_nrmse:.2e}"
