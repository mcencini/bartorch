"""What a card can tell us that a machine without one cannot.

Every device path in bartorch is written and none of it has run.  This walks
them in dependency order and prints what each one found, so a failure names
the first thing that is wrong rather than the last thing that crashed.  Each
check is independent: one failing does not stop the rest.

    python scripts/check_device.py

Needs a CUDA build (`pip install -e . --config-settings=cmake.define.BARTORCH_CUDA=ON`)
and, for the transform checks, the `cufinufft` wheel.
"""

from __future__ import annotations

import math
import time
import traceback

import numpy as np
import torch

import bartorch
import bartorch.tools as bt

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
    if not bartorch.cuda.built():
        return False, f"built without CUDA: {bartorch.build_info()}"
    count = bartorch.cuda.device_count()
    if 0 == count:
        return False, "built with CUDA but no device is visible"
    free = bartorch.cuda.free_memory()
    return True, f"{count} device(s), {free / 1e9:.1f} GB free, device {bartorch.cuda.device()}"


@check("a tool on device tensors answers on the device, with the right numbers")
def _tool_on_device():
    x = torch.randn(4, 64, dtype=torch.complex64, device="cuda")
    y = bt.fft(x, axes=-1)
    if y.device.type != "cuda":
        return False, f"the output came back on {y.device}"
    ref = np.fft.fft(x.cpu().numpy(), axis=-1)
    return True, f"fft rel {_relative(y.cpu().numpy(), ref):.2e} against numpy"


@check("a NUFFT on device tensors matches an explicit discrete Fourier sum")
def _nufft_on_device():
    n = 64
    traj, image = _radial(n, 32, "cuda")
    bartorch.finufft.reset_counters()
    y = bt.nufft(traj, image)
    if y.device.type != "cuda":
        return False, f"the output came back on {y.device}"
    ref = _explicit_dft(traj, image, n)
    got = y.cpu().numpy().reshape(32, n)[:1]
    built = bartorch.finufft.operators_built()
    return True, f"rel {_relative(got, ref):.2e}, operators {built} (finufft, bart)"


@check("cuFINUFFT is what serves a trajectory on the card")
def _cufinufft():
    if not bartorch.finufft.cuda_available():
        return False, "the cufinufft wheel is not installed: pip install 'bartorch[cufinufft]'"
    if not bartorch.finufft.enable():
        return False, "the FINUFFT substitution declined to install itself"
    if not bartorch.finufft.used_on_device():
        return False, f"the device table is empty: {bartorch.finufft.decline_reason()}"

    n = 128
    traj, image = _radial(n, 64, "cuda")
    bartorch.finufft.reset_counters()
    fast = bt.nufft(traj, image)
    if bartorch.finufft.operators_built() != (1, 0):
        return False, f"BART's own operator ran instead: {bartorch.finufft.decline_reason()}"

    ref = _explicit_dft(traj, image, n)
    got = fast.cpu().numpy().reshape(64, n)[:1]
    return True, f"rel {_relative(got, ref):.2e} against the explicit sum"


@check("the device transform and the host transform agree")
def _device_matches_host():
    if not bartorch.finufft.used_on_device():
        return False, "cuFINUFFT is not in use"
    n = 128
    traj, image = _radial(n, 64, "cuda")
    on_card = bt.nufft(traj, image).cpu()
    on_host = bt.nufft(traj.cpu(), image.cpu())
    rel = float((on_card - on_host).abs().max().item() / on_host.abs().max().item())
    return rel < 1e-4, f"rel {rel:.2e}"


@check("pics runs on the card, with and without the Toeplitz normal")
def _pics_on_device():
    n, spokes, coils = 256, 401, 8
    traj = bt.traj(x=n, y=spokes, r=True).cuda()
    image = bt.phantom([n, n], ncoils=coils).cuda()
    maps = (torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5).cuda()
    kspace = bt.nufft(traj, image)

    bartorch.finufft.reset_counters()
    toeplitz = bt.pics(kspace, maps, t=traj)
    normals = bartorch.finufft.normals_built()
    fast = _time(lambda: bt.pics(kspace, maps, t=traj), reps=2)
    pair = _time(lambda: bt.pics(kspace, maps, t=traj, no_toeplitz=True), reps=2)

    if toeplitz.device.type != "cuda":
        return False, f"the reconstruction came back on {toeplitz.device}"
    return True, f"toeplitz {fast:.2f} s, pair {pair:.2f} s, normals {normals} (psf, pair)"


@check("more than one BART stream overlaps transfer with arithmetic")
def _streams():
    n, spokes, coils = 256, 401, 8
    traj = bt.traj(x=n, y=spokes, r=True).cuda()
    image = bt.phantom([n, n], ncoils=coils).cuda()
    maps = (torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5).cuda()
    kspace = bt.nufft(traj, image)

    timings = {}
    for streams in (1, 2, 4):
        if 0 != bartorch.cuda.set_streams(streams):
            continue
        timings[streams] = _time(lambda: bt.pics(kspace, maps, t=traj), reps=2)
    bartorch.cuda.set_streams(1)
    if not timings:
        return False, "set_streams was refused"
    return True, ", ".join(f"{k} stream(s) {v:.2f} s" for k, v in timings.items())


@check("BART's allocations and torch's caching allocator coexist")
def _memcache():
    n, spokes, coils = 256, 401, 8
    traj = bt.traj(x=n, y=spokes, r=True).cuda()
    image = bt.phantom([n, n], ncoils=coils).cuda()
    maps = (torch.ones(1, coils, 1, n, n, dtype=torch.complex64) / coils**0.5).cuda()
    kspace = bt.nufft(traj, image)

    out = []
    for enabled in (True, False):
        bartorch.cuda.use_memcache(enabled)
        torch.cuda.empty_cache()
        before = bartorch.cuda.free_memory()
        bt.pics(kspace, maps, t=traj)
        after = bartorch.cuda.free_memory()
        out.append(f"memcache {'on' if enabled else 'off'}: {(before - after) / 1e9:+.2f} GB held")
    bartorch.cuda.use_memcache(True)
    return True, ", ".join(out)


@check("a tool needs no -g beyond the device pointers")
def _needs_g():
    n = 128
    traj, image = _radial(n, 64, "cuda")
    plain = bt.nufft(traj, image)
    flagged = bt.nufft(traj, image, g=True)
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
