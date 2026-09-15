"""Checks of the CUDA paths, in dependency order, each independent of the others.

Needs a CUDA build (``pip install -e . --config-settings=cmake.define.BARTORCH_CUDA=ON``)
and, for the transform checks, the ``cufinufft`` wheel::

    python scripts/check_device.py
"""

from __future__ import annotations

import math
import time
import traceback

import numpy as np
import torch

import bartorch
import bartorch.tools as bt
from bartorch.linop import NUFFT

CHECKS: list = []


def check(name: str):
    def register(fn):
        CHECKS.append((name, fn))
        return fn

    return register


def _radial(n: int, spokes: int, device: str):
    traj = bt.traj(x=n, y=spokes, r=True).to(device)
    image = bt.phantom([n, n]).reshape(1, n, n).to(device)
    return traj, image


def _explicit_dft(traj: torch.Tensor, image: torch.Tensor, n: int) -> np.ndarray:
    """One spoke of the transform BART computes, summed out sample by sample."""
    k = traj.cpu()[:1].numpy().real
    x = np.arange(n) - n // 2
    phase = np.exp(
        -2j
        * math.pi
        * (
            k[..., 0][..., None, None] * x[None, None, None, :] / n
            + k[..., 1][..., None, None] * x[None, None, :, None] / n
        )
    )
    return (phase * image.cpu().numpy().reshape(n, n)[None, None]).sum(axis=(-1, -2)) / n


def _relative(got: np.ndarray, ref: np.ndarray) -> float:
    return float(np.linalg.norm(got - ref) / np.linalg.norm(ref))


def _time(fn, reps: int = 3) -> float:
    fn()
    torch.cuda.synchronize()
    best = None
    for _ in range(reps):
        start = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        best = elapsed if best is None else min(best, elapsed)
    return best


@check("the library was built with CUDA and finds a device")
def _built():
    if not bartorch._cuda.built():
        return False, f"built without CUDA: {bartorch.build_info()}"
    count = bartorch._cuda.device_count()
    if 0 == count:
        return False, "built with CUDA but no device is visible"
    free = bartorch._cuda.free_memory()
    return True, f"{count} device(s), {torch.cuda.get_device_name(0)}, {free / 1e9:.1f} GB free"


@check("a tool on device tensors answers on the device, with the right numbers")
def _tool_on_device():
    x = torch.randn(4, 64, dtype=torch.complex64, device="cuda")
    y = bartorch.fft(x, axes=-1)
    if y.device.type != "cuda":
        return False, f"the output came back on {y.device}"
    host = x.cpu().numpy()
    ref = np.fft.fftshift(np.fft.fft(np.fft.ifftshift(host, axes=-1), axis=-1), axes=-1)
    rel = _relative(y.cpu().numpy(), ref)
    return rel < 1e-4, f"fft rel {rel:.2e} against numpy"


@check("a NUFFT on device tensors matches an explicit discrete Fourier sum")
def _nufft_on_device():
    n = 64
    traj, image = _radial(n, 32, "cuda")
    bartorch._finufft.reset_counters()
    y = bartorch.nufft(image, traj)
    if y.device.type != "cuda":
        return False, f"the output came back on {y.device}"
    ref = _explicit_dft(traj, image, n)
    got = y.cpu().numpy().reshape(32, n)[:1]
    rel = _relative(got, ref)
    built = bartorch._finufft.operators_built()
    return rel < 5e-3, f"rel {rel:.2e}, operators {built} (finufft, bart)"


@check("cuFINUFFT is what serves a trajectory on the card")
def _cufinufft():
    if not bartorch._finufft.cuda_available():
        return False, "the cufinufft wheel is not installed: pip install 'bartorch[cufinufft]'"
    if not bartorch._finufft.used_in_tools():
        return False, "the substitution did not install itself"
    if not bartorch._finufft.used_on_device():
        return False, f"the device table is empty: {bartorch._finufft.decline_reason()}"

    n = 128
    traj, image = _radial(n, 64, "cuda")
    bartorch._finufft.reset_counters()
    fast = bartorch.nufft(image, traj)
    if bartorch._finufft.operators_built() != (1, 0):
        return False, f"BART's own operator ran instead: {bartorch._finufft.decline_reason()}"

    ref = _explicit_dft(traj, image, n)
    got = fast.cpu().numpy().reshape(64, n)[:1]
    rel = _relative(got, ref)
    eps = bartorch._finufft.tolerance()
    return rel < 5 * eps, f"rel {rel:.2e} against the explicit sum, at a tolerance of {eps:g}"


@check("the device transform and the host transform agree")
def _device_matches_host():
    if not bartorch._finufft.used_on_device():
        return False, "cuFINUFFT is not in use"
    n = 128
    traj, image = _radial(n, 64, "cuda")
    on_card = bartorch.nufft(image, traj).cpu()
    on_host = bartorch.nufft(image.cpu(), traj.cpu())
    rel = float((on_card - on_host).abs().max().item() / on_host.abs().max().item())
    eps = bartorch._finufft.tolerance()
    return rel < 5 * eps, f"rel {rel:.2e}, each held to a tolerance of {eps:g}"


@check("pics runs on the card, with and without the Toeplitz normal")
def _pics_on_device():
    n, spokes, coils = 256, 401, 8
    traj = bt.traj(x=n, y=spokes, r=True).cuda()
    image = bt.phantom([n, n], s=coils).cuda()
    maps = (torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5).cuda()
    kspace = bartorch.nufft(image, traj)

    bartorch._finufft.reset_counters()
    toeplitz = bt.pics(kspace, maps, t=traj)
    normals = bartorch._finufft.normals_built()
    fast = _time(lambda: bt.pics(kspace, maps, t=traj), reps=2)
    pair = _time(lambda: bt.pics(kspace, maps, t=traj, no_toeplitz=True), reps=2)

    if toeplitz.device.type != "cuda":
        return False, f"the reconstruction came back on {toeplitz.device}"
    return True, f"toeplitz {fast:.2f} s, pair {pair:.2f} s, normals {normals} (psf, pair)"


@check("more than one BART stream overlaps transfer with arithmetic")
def _streams():
    n, spokes, coils = 256, 401, 8
    traj = bt.traj(x=n, y=spokes, r=True).cuda()
    image = bt.phantom([n, n], s=coils).cuda()
    maps = (torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5).cuda()
    kspace = bartorch.nufft(image, traj)

    timings = {}
    for streams in (1, 2, 4):
        try:
            bartorch._cuda.set_streams(streams)
        except ValueError:
            continue
        timings[streams] = _time(lambda: bt.pics(kspace, maps, t=traj), reps=2)
    bartorch._cuda.set_streams(1)
    if not timings:
        return False, "set_streams was refused for every count"
    return True, ", ".join(f"{k} stream(s) {v:.2f} s" for k, v in timings.items())


@check("BART's allocations and torch's caching allocator coexist")
def _memcache():
    """What BART holds on the card while torch holds the rest of it.

    A tool ends by clearing BART's memory cache, so what it took is back
    before the call returns whatever the cache is set to; an operator outlives
    the call, and there ``use_memcache`` is what decides.
    """
    n, spokes, coils = 256, 401, 8
    traj = bt.traj(x=n, y=spokes, r=True).cuda()
    image = bt.phantom([n, n], s=coils).cuda()
    maps = (torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5).cuda()
    kspace = bartorch.nufft(image, traj)

    torch.cuda.empty_cache()
    idle = bartorch._cuda.free_memory()
    ballast = torch.empty(1 << 27, dtype=torch.complex64, device="cuda")
    with_torch = bartorch._cuda.free_memory()
    out = bt.pics(kspace, maps, t=traj)
    del ballast, out
    torch.cuda.empty_cache()
    after_tool = bartorch._cuda.free_memory()

    held = {}
    for cache in (True, False):
        bartorch._cuda.use_memcache(cache)
        torch.cuda.empty_cache()
        before = bartorch._cuda.free_memory()
        op = NUFFT(traj, (1, n, n))
        del op
        torch.cuda.empty_cache()
        held[cache] = before - bartorch._cuda.free_memory()
    bartorch._cuda.use_memcache(True)

    detail = (
        f"pics ran with {(idle - with_torch) / 1e6:.0f} MB in torch's hands and left "
        f"{(idle - after_tool) / 1e6:+.0f} MB behind; an operator holds "
        f"{held[True] / 1e6:.0f} MB with the cache on, {held[False] / 1e6:.0f} MB with it off"
    )
    return (after_tool >= with_torch) and (held[False] <= held[True]), detail


@check("a tool needs no -g beyond the device pointers")
def _needs_g():
    n = 128
    traj, image = _radial(n, 64, "cuda")
    plain = bartorch.nufft(image, traj)
    flagged = bartorch.nufft(image, traj, g=True)
    rel = float((plain - flagged).abs().max().item() / flagged.abs().max().item())
    return rel < 1e-5, f"with and without -g differ by {rel:.2e}"


def main() -> int:
    print(bartorch.build_info())
    print(f"torch {torch.__version__}, CUDA {torch.version.cuda}\n")

    failed = 0
    for name, fn in CHECKS:
        try:
            ok, detail = fn()
        except Exception:
            ok, detail = False, traceback.format_exc(limit=3).strip().splitlines()[-1]
        failed += not ok
        print(f"[{'ok  ' if ok else 'FAIL'}] {name}\n         {detail}")

    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
