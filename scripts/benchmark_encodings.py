"""Timings of the MRI encodings, as docs/design/composed-encodings.md records them.

    python scripts/benchmark_encodings.py CASE

CASE is one of cart2d, cart2d_sub, cart3d, cart3d_sub, cart3d_sampled, wave3d,
wave3d_sub, wave3d_sampled, noncart3d_sub, field_cart3d, field_noncart3d,
field_wave3d.  Host arrays, the operator on the card: build, then forward,
adjoint and normal (the 3D subspace cases over a dense pattern time the normal
alone, their dense k-space not fitting in host memory), each after one warm-up
and repeated with a reused output where the operator takes ``out=``.  Prints
the plan the encoding was lowered into, min-max times, the device peak of each
phase, the device memory at rest and the host peak.  Run one case per process,
so that the peaks are the case's.
"""

import math, resource, sys, threading, time
import torch
from bartorch import linop
import bartorch.tools as bt

case = sys.argv[1]
torch.manual_seed(0)
C64 = torch.complex64
coils, R, accel = 8, 4, 4


def kern(ndim, c=coils):
    return torch.randn(c, *(24,) * ndim, dtype=C64)


def pe_pattern(lead, *pe):
    """Phase-encode undersampling at `accel` with a fully sampled centre; `lead` frames in front."""
    keep = torch.rand(*lead, *pe) < 1.0 / accel
    centre = tuple(slice(n // 2 - max(1, n // 16) // 2, n // 2 + (max(1, n // 16) + 1) // 2) for n in pe)
    keep[(..., *centre)] = True
    return keep.to(C64)


def cosine_basis(frames):
    b = torch.zeros(R, frames, dtype=C64)
    for c in range(R):
        b[c] = torch.cos(torch.pi * c * (torch.arange(frames) + 0.5) / frames)
    return b / b.norm(dim=1, keepdim=True)


def wave_psf(z, y, wx, cycles=7, amp=0.35):
    t = torch.arange(wx, dtype=torch.float64) / wx
    yy = (torch.arange(y, dtype=torch.float64) - y / 2)[None, :, None]
    zz = (torch.arange(z, dtype=torch.float64) - z / 2)[:, None, None]
    ph = 2 * math.pi * amp * (yy * torch.sin(2 * math.pi * cycles * t) + zz * torch.cos(2 * math.pi * cycles * t))
    return torch.exp(1j * ph).to(C64)  # (z, y, wx)


normal_only = case in ("cart3d_sub", "wave3d_sub", "noncart3d_sub")
frames = 80
if case == "cart2d":
    n = 320
    make = lambda: linop.CartesianSense(kern(2, 32), (n, n), pattern=pe_pattern((), n)[:, None], kernels=True, ndim=2, device="cuda")
elif case == "cart2d_sub":
    n = 256
    make = lambda: linop.CartesianSense(kern(2), (R, n, n), pattern=pe_pattern((frames,), n)[..., None],
                                        basis=cosine_basis(frames), kernels=True, ndim=2, device="cuda")
elif case in ("cart3d", "field_cart3d"):
    n = 256 if case == "cart3d" else 160
    make = lambda: linop.CartesianSense(kern(3), (n, n, n), pattern=pe_pattern((), n, n)[..., None], kernels=True, ndim=3, device="cuda")
elif case == "cart3d_sub":
    n = 256
    make = lambda: linop.CartesianSense(kern(3), (R, n, n, n), pattern=pe_pattern((frames,), n, n)[..., None],
                                        basis=cosine_basis(frames), kernels=True, ndim=3, device="cuda")
elif case == "cart3d_sampled":
    # t2sh-like: 362 phase encodes per frame of a 256 x 256 plane, 80 frames, whole readout.
    n, shots = 256, 362
    pos = torch.stack([torch.randperm(n * n)[:shots] for _ in range(frames)])
    positions = torch.stack([pos // n, pos % n], dim=-1)
    make = lambda: linop.CartesianSense(kern(3), (R, n, n, n), positions=positions, basis=cosine_basis(frames),
                                        kernels=True, ndim=3, device="cuda")
elif case in ("wave3d", "wave3d_sub", "wave3d_sampled", "field_wave3d"):
    n = 160 if case == "field_wave3d" else 192
    wx = 3 * n
    psf = wave_psf(n, n, wx)
    if case == "wave3d_sampled":
        shots = 362
        pos = torch.stack([torch.randperm(n * n)[:shots] for _ in range(frames)])
        positions = torch.stack([pos // n, pos % n], dim=-1)
        make = lambda: linop.WaveSense(kern(3), (R, n, n, n), psf=psf, readout=wx, positions=positions,
                                       basis=cosine_basis(frames), kernels=True, ndim=3, device="cuda")
    elif case == "wave3d_sub":
        make = lambda: linop.WaveSense(kern(3), (R, n, n, n), psf=psf, readout=wx, pattern=pe_pattern((frames,), n, n)[..., None],
                                       basis=cosine_basis(frames), kernels=True, ndim=3, device="cuda")
    else:
        make = lambda: linop.WaveSense(kern(3), (n, n, n), psf=psf, readout=wx, pattern=pe_pattern((), n, n)[..., None],
                                       kernels=True, ndim=3, device="cuda")
elif case == "noncart3d_sub":
    n, fr, shots = 256, 500, 48
    traj = bt.traj(x=n, y=shots * fr, r=True, flag_3=True).reshape(fr, shots, n, 3)
    make = lambda: linop.NoncartesianSense(kern(3), (R, n, n, n), traj=traj, basis=cosine_basis(fr), kernels=True, device="cuda")
elif case == "field_noncart3d":
    n, spokes = 160, 8000
    traj = bt.traj(x=n, y=spokes, r=True, flag_3=True).reshape(spokes, n, 3)
    make = lambda: linop.NoncartesianSense(kern(3), (n, n, n), traj=traj, kernels=True, device="cuda")
else:
    raise SystemExit(f"unknown case {case}")

if case.startswith("field"):
    make_enc, L = make, 6

    def make():
        E = make_enc()
        o = E.oshape
        if case == "field_cart3d":  # EPI-like: time runs along the phase-encode rows
            ny = o[-2]
            t = torch.arange(ny, dtype=torch.float64) / ny
            b = torch.stack([torch.exp(-2j * math.pi * (l + 1) * t) for l in range(L)]).to(C64).reshape(L, 1, 1, ny, 1)
        else:  # time runs along the readout samples
            ns = o[-1]
            t = torch.arange(ns, dtype=torch.float64) / ns
            b = torch.stack([torch.exp(-2j * math.pi * (l + 1) * t) for l in range(L)]).to(C64).reshape(L, *(1,) * (len(o) - 1), ns)
        c = torch.randn(L, *E.ishape, dtype=C64) * 0.1 + 1.0
        return linop.FieldCorrected(E, coefficients=(b, c))

torch.cuda.synchronize(); torch.cuda.empty_cache(); free0, _ = torch.cuda.mem_get_info()
peak = {}; phase = ["build"]; stop = threading.Event()


def watch():
    while not stop.is_set():
        f, _ = torch.cuda.mem_get_info(); p = phase[0]; peak[p] = max(peak.get(p, 0), free0 - f); time.sleep(0.002)


th = threading.Thread(target=watch, daemon=True); th.start()


def sync():
    torch.cuda.synchronize(); time.sleep(0.01)


def held():
    sync(); f, _ = torch.cuda.mem_get_info(); return (free0 - f) / 2**20


def reusing(fn, arg, shape):
    """`fn(arg, out=buffer)` with one buffer across calls where the operator takes one."""
    buf = torch.empty(shape, dtype=C64)
    try:
        fn(arg, out=buf)
        return (lambda: fn(arg, out=buf)), "out"
    except TypeError:
        return (lambda: fn(arg)), "fresh"


def timed(label, fn, shape, reps):
    phase[0] = label
    t0 = time.time(); call, how = reusing(fn, *shape); sync(); first = time.time() - t0
    ts = []
    for _ in range(reps):
        t0 = time.time(); call(); sync(); ts.append(time.time() - t0)
    return f"{label} first {first:5.2f} s, {how} {min(ts):5.2f}-{max(ts):5.2f} s"


parts = []
try:
    phase[0] = "build"; t0 = time.time(); op = make(); sync(); parts.append(f"build {time.time() - t0:5.2f} s")
    # A case that fell back to BART's plain chain is timing something else, so
    # the plan is printed beside the times rather than left to be inferred.
    plan = op.plan
    parts.append(f"plan {plan.transform}/{plan.contraction}/{plan.normal}/{plan.executor}"
                 + ("" if plan.fused else " NOT FUSED"))
    rest = held()
    x = torch.randn(*op.ishape, dtype=C64)
    if not normal_only:
        parts.append(timed("fwd", op.forward if hasattr(op, "forward") else op, (x, op.oshape), 3))
        y = torch.randn(*op.oshape, dtype=C64)
        parts.append(timed("adj", op.adjoint, (y, op.ishape), 3))
        del y
    parts.append(timed("nrm", op.normal, (x, op.ishape), 3))
    shapes = f"{tuple(op.ishape)}->{tuple(op.oshape)}"
except Exception as e:
    parts.append(f"FAILED {type(e).__name__}: {str(e)[:200]}"); shapes = ""; rest = 0
stop.set(); th.join()
host = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20
g = lambda k: peak.get(k, 0) / 2**20
temp = __import__("subprocess").run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader"], capture_output=True, text=True).stdout.strip()
print(f"{case:15s} {shapes} | " + " | ".join(parts)
      + f" | peak MiB build {g('build'):.0f} fwd {g('fwd'):.0f} adj {g('adj'):.0f} nrm {g('nrm'):.0f} | rest {rest:.0f} MiB | host {host:.2f} GiB | {temp} C",
      flush=True)
