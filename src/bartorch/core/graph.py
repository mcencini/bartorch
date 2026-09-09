"""Running a BART tool on tensors.

:func:`dispatch` is the single entry point every wrapper in
:mod:`bartorch.tools` calls.  It registers each input tensor's memory in the
compiled library's in-memory registry under a ``.mem`` name, assembles the
tool's argv, runs the tool in-process and hands back the tensor the tool
wrote, allocated by torch through the allocator callback.

A tool sees host memory.  BART's tools are command mains that map their
inputs the way the command line does and some read them there --
``estimate_im_dims`` in ``nufft``, the sort in ``pics``'s scaling estimate,
``gram_matrix`` in ``ecalib`` -- so a tensor on a card crosses to the host
first.  What puts the work back on the card is BART's own device path:
tensors on a device select it, and BART allocates there and runs its kernels
there for as long as the tool does.  The operator layer in
:mod:`bartorch.ops` is the one that takes device memory as it stands.

Axis convention: a C-order tensor of shape ``(a, b, c)`` and a BART array of
dims ``[c, b, a]`` are the same bytes, so a shape is reversed at this boundary
and nothing else happens to the data.
"""

from __future__ import annotations

import contextlib
import ctypes
import logging
import threading
from typing import Any

import numpy as np
import torch

from bartorch import _backend, _cuda
from bartorch._lib import ALLOC_FN, DIMS, FREE_FN, LOG_FN, library

__all__ = [
    "dispatch",
    "BartError",
    "set_debug_level",
    "get_debug_level",
    "set_num_threads",
    "set_copy_inputs",
]

_log = logging.getLogger("bartorch.bart")

_LEVELS = {
    0: logging.ERROR,
    1: logging.WARNING,
    2: logging.INFO,
}


class BartError(RuntimeError):
    """A BART tool exited with an error."""


class _Allocator:
    """Serves BART's own allocations with torch tensors and keeps them alive.

    They come from wherever the tool is working, because BART's `md_`
    operations take the host path unless every argument is on a device -- and
    take it silently, reading device memory from the host.  An output on the
    wrong side of the bus is a segmentation fault, not a slower answer.
    """

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.live: dict[int, torch.Tensor] = {}
        self._alloc_cb = ALLOC_FN(self._alloc)
        self._free_cb = FREE_FN(self._free)

    def _alloc(self, _ctx, D, dims):
        shape = [dims[i] for i in range(D)][::-1]
        try:
            t = torch.empty(shape, dtype=torch.complex64, device=self.device)
        except Exception:
            _log.exception("bartorch: allocation of %s failed", shape)
            return None
        ptr = t.data_ptr()
        self.live[ptr] = t
        return ptr

    def _free(self, _ctx, ptr):
        self.live.pop(ptr, None)

    def take(self, ptr: int) -> torch.Tensor:
        return self.live.pop(ptr)

    def install(self) -> None:
        library().bartorch_set_allocator(self._alloc_cb, self._free_cb, None)


def _on_log(_ctx, level, func, file, line, msg):
    text = msg.decode(errors="replace").rstrip("\n")
    _log.log(_LEVELS.get(level, logging.DEBUG), "%s", text)


_log_cb = LOG_FN(_on_log)
_allocator = _Allocator()
_lock = threading.RLock()
_ready = False
_call_id = 0
_copy_inputs = True


def _ensure_ready() -> None:
    global _ready
    if _ready:
        return
    with _lock:
        if _ready:
            return
        lib = library()
        _allocator.install()
        lib.bartorch_set_log_handler(_log_cb, None)
        lib.bartorch_set_debug_level(1)
        _backend.install()
        _ready = True

    # After _ready, because putting it in place runs a tool of its own.
    from bartorch import _finufft

    _finufft.install_once()


def set_debug_level(level: int) -> None:
    """Set BART's verbosity: 0 errors, 1 warnings, 2 info, 3 and up debug."""
    _ensure_ready()
    library().bartorch_set_debug_level(int(level))


def get_debug_level() -> int:
    _ensure_ready()
    return int(library().bartorch_get_debug_level())


def set_copy_inputs(copy: bool) -> None:
    """Choose whether a tool works on a private copy of each input.

    BART maps input files copy-on-write and some tools write into them, so
    by default every input tensor is copied before a tool runs.  Passing
    ``False`` hands the tool the tensor's own memory: no copy is made, and a
    tool that writes into its input changes the caller's tensor.  A tensor on
    a card crosses to the host either way, so this only decides what happens
    to a host tensor.
    """
    global _copy_inputs
    _copy_inputs = bool(copy)


def _resize_centre(x, spatial):
    """Crop or zero-pad the trailing axes about the centre the FFT uses.

    ``md_resize_center`` puts the centre at ``dim / 2``, so the offset between
    two sizes is the difference of their halves -- which is not the same as
    half their difference when one of them is odd.
    """
    import torch

    spatial = tuple(spatial)
    out = x
    for axis, want in zip(range(-len(spatial), 0), spatial, strict=True):
        have = out.shape[axis]
        if have == want:
            continue
        offset = abs(want // 2 - have // 2)
        if want < have:
            out = out.narrow(axis, offset, want)
        else:
            shape = list(out.shape)
            shape[axis] = want
            padded = torch.zeros(shape, dtype=out.dtype, device=out.device)
            padded.narrow(axis, offset, have).copy_(out)
            out = padded
    return out


def maps_to_kernels(maps, size):
    """The centre of each sensitivity's spectrum, which is all a smooth map carries.

    A bank of maps is one image per coil; the kernels are a few dozen samples
    across, so a bank that would not fit becomes one that costs nothing.  What
    is lost is everything above the kernel's own band, which for sensitivities
    from a calibration is close to nothing -- and the loss is measurable, by
    inflating the kernels again and comparing.

    Parameters
    ----------
    maps : tensor
        Sensitivities, coils first: ``(coils, *spatial)``.
    size : int or tuple of int
        The kernel's spatial size, per axis or the same for all.

    Returns
    -------
    tensor
        Kernels of shape ``(coils, *size)``.
    """
    import bartorch.tools as bt

    spatial = tuple(maps.shape[1:])
    size = (size,) * len(spatial) if isinstance(size, int) else tuple(size)
    if len(size) != len(spatial):
        raise ValueError(f"kernel size {size} does not match the map's {spatial} spatial axes")

    axes = tuple(range(-len(spatial), 0))
    spectrum = bt.fft(maps, axes=axes, unitary=True)
    return _resize_centre(spectrum, size)


def kernels_to_maps(kernels, spatial):
    """The maps a kernel bank stands for: padded back on to the grid and transformed.

    This is what the SENSE operator does to one slab at a time; doing it here
    is how the approximation is checked against the maps the kernels came from.
    """
    import bartorch.tools as bt

    spatial = tuple(spatial)
    axes = tuple(range(-len(spatial), 0))
    return bt.fft(_resize_centre(kernels, spatial), axes=axes, unitary=True, inverse=True)


def set_coil_batch(n: int) -> None:
    """How many coils a SENSE operator holds at once.

    The coils are independent until the sum that ends an adjoint, so the
    operator walks them a slab at a time and what is resident is a slab
    rather than the whole bank.  Behind a non-Cartesian transform that also
    shrinks the grid its Toeplitz normal convolves on, which is the largest
    thing a three-dimensional reconstruction allocates.

    One coil at a time is the smallest and the default.  More is faster --
    FINUFFT batches a plan across transforms, and a slab of one gives that up
    -- and proportionally larger.  Zero leaves BART its own operator over
    every coil at once, which is the fastest and the largest of all.
    """
    _ensure_ready()
    library().bartorch_sense_set_coil_batch(int(n))


def coil_batch() -> int:
    """Coils a SENSE operator holds at once; zero means all of them."""
    _ensure_ready()
    return int(library().bartorch_sense_coil_batch())


def set_fold_maps(enable: bool = True) -> None:
    """Apply the sensitivity inside the transform rather than beside it.

    A SENSE normal otherwise makes two coil images for every slab: one to
    multiply the map into, and one for the transform's answer to land in.  A
    transform that reads and writes one coefficient at a time can take the map
    itself -- on as a coefficient is read, conjugated as it is written -- and
    then neither is made.  At 256^3 over four coefficients each of them is half
    a gigabyte; at 224^3 over four, turning this off costs 1.13 GiB.

    On by default, and it applies only where the transform works a coefficient
    at a time, which is the gathered arrangement a compressed function uses.
    Elsewhere the coil images are made as before.
    """
    _ensure_ready()
    library().bartorch_sense_set_fold_maps(int(bool(enable)))


def fold_maps() -> bool:
    """Whether the sensitivity is applied inside the transform."""
    _ensure_ready()
    return bool(library().bartorch_sense_fold_maps())


def set_num_threads(n: int) -> None:
    """Set the number of threads BART, its FFT and FINUFFT use.

    One number for everything in the process that threads.  FINUFFT starts
    out choosing for itself, a thread per physical core, and
    :func:`bartorch.finufft.set_threads` with zero puts it back to that
    without giving BART a count of its own.
    """
    _ensure_ready()
    library().bartorch_set_num_threads(int(n))


# --- argv -------------------------------------------------------------------


@contextlib.contextmanager
def _on_device(device: torch.device):
    """Point BART at *device* for the duration, ordered against torch's stream."""
    if device.type != "cuda":
        yield
        return
    with _cuda.ordered(device):
        yield


def _value_str(val: Any) -> str:
    if isinstance(val, bool):
        return "1" if val else "0"
    if isinstance(val, (tuple, list)):
        return ":".join(_value_str(v) for v in val)
    if isinstance(val, float):
        return repr(val)
    return str(val)


def _expand_list_flags(kwargs: dict[str, Any]) -> list[tuple[str, Any]]:
    """Flatten ``R=[a, b]`` into repeated ``R`` entries, keeping order."""
    out: list[tuple[str, Any]] = []
    for key, val in kwargs.items():
        if isinstance(val, list):
            out.extend((key, v) for v in val)
        else:
            out.append((key, val))
    return out


def _flag_string(key: str) -> str:
    """The flag a keyword stands for.

    ``x`` is ``-x`` and ``psf_export`` is ``--psf-export``.  A flag BART
    spells with a digit takes a ``flag_`` prefix, so ``flag_3`` is ``-3``, and
    a trailing digit otherwise repeats a flag: ``R_1`` and ``R_2`` are both
    ``-R``.
    """
    stem, _, suffix = key.rpartition("_")
    if stem and stem != "flag" and suffix.isdigit():
        key = stem
    if key.startswith("flag_") and len(key) > 5:
        return "-" + key[5:]
    if len(key) == 1:
        return "-" + key
    return "--" + key.replace("_", "-")


def build_argv(
    op_name: str,
    input_names: list[str],
    output_names: list[str] | str | None,
    positional: list[Any],
    kwargs: dict[str, Any],
    flag_arrays: dict[int, str] | None = None,
) -> list[str]:
    """Assemble ``[tool, flags..., positionals..., inputs..., outputs...]``.

    A flag whose value is an array takes the name that array was registered
    under, which is how a tool reads a trajectory, a sampling pattern or a
    subspace basis.  ``flag_arrays`` maps the position of such a value in the
    expanded flag list to its name.
    """
    argv = [op_name]
    for index, (key, val) in enumerate(_expand_list_flags(kwargs)):
        if val is None or val is False:
            continue
        argv.append(_flag_string(key))
        if flag_arrays is not None and index in flag_arrays:
            argv.append(flag_arrays[index])
        elif val is not True:
            argv.append(_value_str(val))
    for val in positional:
        if val is not None:
            argv.append(_value_str(val))
    argv.extend(input_names)
    if isinstance(output_names, str):
        output_names = [output_names]
    argv.extend(output_names or [])
    return argv


# --- tensors ----------------------------------------------------------------


def _bart_dims(shape: tuple[int, ...]) -> tuple[int, ctypes.Array]:
    """BART rank and dimension vector of a C-order shape: reversed, at least one axis."""
    rev = list(shape)[::-1] or [1]
    if len(rev) > DIMS:
        raise ValueError(f"BART supports at most {DIMS} dimensions, got {len(rev)}")
    return len(rev), (ctypes.c_long * len(rev))(*rev)


# The tools that work on the memory they are handed.
#
# BART's `md_` operations take the host path unless *every* argument is on a
# device, and take it silently -- so a tool that allocates a temporary with
# `md_alloc` or `anon_cfl`, or that resets `bart_use_gpu` for itself, reads
# device memory from the host and dies rather than answering slowly.  Whether
# a tool does that is a property of its own code, not something to infer, so
# this list holds only what has been run on a card and checked against the
# same tool on the host.  Everything else is given host memory.
_ON_DEVICE = frozenset({"estdims", "fft", "ifft", "nufft", "pics", "rss"})

# Tools that calibrate before a reconstruction is attempted, at a resolution
# where the textbook grid costs nothing.
#
# The default transform is the cheap one, because what a reconstruction is
# held to is the data rather than the transform.  Coil sensitivities are not
# that: `nlinv` and its relatives fit them from a low-resolution image that
# everything after them is built on, and a grid twice over at FINUFFT's own
# tolerance is a few megabytes there.  So they get it.
_CALIBRATES = frozenset({"ncalib", "nlinv", "rtnlinv"})

_CAREFUL_TOLERANCE = 1e-6
_CAREFUL_UPSAMPLING = 2.0


@contextlib.contextmanager
def _transform_for(op_name: str):
    """The tolerance and grid this tool's transforms are planned with."""
    if op_name not in _CALIBRATES:
        yield
        return

    lib = library()
    was = (lib.bartorch_finufft_tolerance(), lib.bartorch_finufft_upsampling())
    lib.bartorch_finufft_set_tolerance(_CAREFUL_TOLERANCE)
    lib.bartorch_finufft_set_upsampling(_CAREFUL_UPSAMPLING)
    try:
        yield
    finally:
        lib.bartorch_finufft_set_tolerance(was[0])
        lib.bartorch_finufft_set_upsampling(was[1])


def _for_bart(x: torch.Tensor, op_name: str) -> torch.Tensor:
    """The tensor a tool is given: the caller's own, or a private copy.

    BART maps input files copy-on-write and some tools write into what they
    were given, so it is cloned unless :func:`set_copy_inputs` turned that
    off; a tensor moved off a card is a private copy already.
    """
    if (x.device.type != "cpu") and (op_name not in _ON_DEVICE):
        return x.cpu()
    return x.clone() if _copy_inputs else x


def _as_input(x: Any) -> torch.Tensor:
    if isinstance(x, np.ndarray):
        x = torch.from_numpy(np.ascontiguousarray(x, dtype=np.complex64))
    if not isinstance(x, torch.Tensor):
        raise TypeError(f"bartorch tools take tensors, got {type(x).__name__}")
    if x.dtype != torch.complex64:
        x = x.to(torch.complex64)
    return x.contiguous()


def _output_shape(dims: ctypes.Array, min_ndim: int) -> list[int]:
    rev = [int(dims[i]) for i in range(DIMS)][::-1]
    while len(rev) > max(1, min_ndim) and rev[0] == 1:
        rev.pop(0)
    return rev


def run_command(argv: list[str]) -> tuple[int, str, str]:
    """Run one tool with a fully formed argv; return (code, stdout, error text)."""
    _ensure_ready()
    lib = library()
    c_argv = (ctypes.c_char_p * len(argv))(*[a.encode() for a in argv])
    out = ctypes.create_string_buffer(1 << 16)
    err = ctypes.create_string_buffer(4096)
    code = lib.bartorch_command(len(argv), c_argv, out, len(out), err, len(err))
    return code, out.value.decode(errors="replace"), err.value.decode(errors="replace")


def dispatch(
    op_name: str,
    inputs: list[Any],
    output_dims: list[int] | bool | None,
    _pos: list[Any] | None = None,
    _n_out: int = 1,
    **kwargs: Any,
) -> torch.Tensor | tuple[torch.Tensor, ...] | str | None:
    """Run BART tool *op_name* on *inputs* and return its output.

    Parameters
    ----------
    op_name : str
        BART tool name, as on the command line.
    inputs : list of tensors
        Input arrays in the order the tool expects them.
    output_dims : list of int, None or False
        A hint for the number of leading singleton axes to keep on the
        result, or ``False`` for a tool that writes no array and whose
        printed text is returned instead.
    _pos : list, optional
        Scalar positional arguments placed between the flags and the
        input names.
    _n_out : int
        Number of output arrays the tool writes; more than one gives a tuple.

    Inputs are copied before the tool runs unless :func:`set_copy_inputs`
    turned that off, because a BART tool may write into its inputs.
    **kwargs
        Flags.  ``True`` gives a bare flag, ``None`` and ``False`` are
        skipped, a list repeats the flag, and a tuple joins its values with
        colons.

    Returns
    -------
    torch.Tensor, tuple of torch.Tensor, str or None
        The output arrays, on the device of the inputs, or the tool's text.

    Raises
    ------
    BartError
        When the tool exits with an error; the message carries BART's own.
    """
    global _call_id
    _ensure_ready()
    lib = library()
    tensors = [_as_input(x) for x in inputs]

    # A flag can carry an array too: `pics -t` takes a trajectory, `-p` a
    # sampling pattern, `-B` a basis.  Those are registered like any input and
    # the flag is given the name they were registered under.
    flag_arrays: dict[int, torch.Tensor] = {}
    for index, (_, value) in enumerate(_expand_list_flags(kwargs)):
        if isinstance(value, (torch.Tensor, np.ndarray)):
            flag_arrays[index] = _as_input(value)

    devices = {t.device for t in list(tensors) + list(flag_arrays.values())}
    if len(devices) > 1:
        raise ValueError("all inputs must live on the same device")
    device = devices.pop() if devices else torch.device("cpu")
    if device.type == "cuda" and not _cuda.available():
        raise ValueError(
            "this library has no CUDA support built in, or no device is present; "
            "move the tensors to the host with .cpu()"
        )
    tensors = [_for_bart(t, op_name) for t in tensors]
    flag_arrays = {i: _for_bart(t, op_name) for i, t in flag_arrays.items()}
    want_output = output_dims is not False
    min_ndim = len(output_dims) if isinstance(output_dims, (list, tuple)) else 1

    with _lock, _on_device(device), _transform_for(op_name):
        # What BART allocates for itself comes from torch, on the memory the
        # tool was actually given: an output on the other side of the bus from
        # its input is a segmentation fault, not a slower answer.
        _allocator.device = tensors[0].device if tensors else device
        _call_id += 1
        call = _call_id
        names = [f"_bt_{call}_in{i}.mem" for i in range(len(tensors))]
        out_names = [f"_bt_{call}_out{i}.mem" for i in range(_n_out)] if want_output else []
        flag_names = {i: f"_bt_{call}_flag{i}.mem" for i in flag_arrays}
        for name, t in list(zip(names, tensors)) + [
            (flag_names[i], flag_arrays[i]) for i in flag_arrays
        ]:
            rank, dims = _bart_dims(tuple(t.shape))
            lib.bartorch_register(name.encode(), rank, dims, t.data_ptr())
        argv = build_argv(op_name, names, out_names, list(_pos or []), kwargs, flag_names)
        try:
            code, text, err = run_command(argv)
            if code != 0:
                raise BartError(
                    f"bart {op_name} failed (code {code}): {err or text or 'no message'}"
                )
            if not want_output:
                return text.strip() if text else None
            results = []
            for out_name in out_names:
                dims = (ctypes.c_long * DIMS)()
                ptr = ctypes.c_void_p()
                if lib.bartorch_lookup(out_name.encode(), DIMS, dims, ctypes.byref(ptr)) != 0:
                    raise BartError(f"bart {op_name} did not write {out_name}")
                out = _allocator.take(ptr.value).reshape(_output_shape(dims, min_ndim))
                results.append(out.to(device) if device.type != "cpu" else out)
            return results[0] if len(results) == 1 else tuple(results)
        finally:
            _allocator.device = torch.device("cpu")
            for name in names + out_names + list(flag_names.values()):
                lib.bartorch_unlink(name.encode())
