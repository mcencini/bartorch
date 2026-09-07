"""Running a BART tool on tensors.

:func:`dispatch` is the single entry point every wrapper in
:mod:`bartorch.tools` calls.  It registers each input tensor's memory in the
compiled library's in-memory registry under a ``.mem`` name, assembles the
tool's argv, runs the tool in-process and hands back the tensor the tool
wrote, which was allocated by torch through the allocator callback and so was
never copied.

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
    """Serves BART's output allocations with torch tensors and keeps them alive."""

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
    tool that writes into its input changes the caller's tensor.
    """
    global _copy_inputs
    _copy_inputs = bool(copy)


def set_num_threads(n: int) -> None:
    """Set the number of threads BART and its FFT use."""
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
    base = key
    stem, _, suffix = key.rpartition("_")
    if stem and suffix.isdigit():
        base = stem
    if len(base) == 1:
        return "-" + base
    if base.startswith("flag_") and len(base) > 5:
        return "-" + base[5:]
    return "--" + base.replace("_", "-")


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

    if _copy_inputs:
        tensors = [t.clone() for t in tensors]
        flag_arrays = {i: t.clone() for i, t in flag_arrays.items()}
    devices = {t.device for t in list(tensors) + list(flag_arrays.values())}
    if len(devices) > 1:
        raise ValueError("all inputs must live on the same device")
    device = devices.pop() if devices else torch.device("cpu")
    if device.type == "cuda" and not _cuda.available():
        raise ValueError(
            "this library has no CUDA support built in, or no device is present; "
            "move the tensors to the host with .cpu()"
        )
    want_output = output_dims is not False
    min_ndim = len(output_dims) if isinstance(output_dims, (list, tuple)) else 1

    with _lock, _on_device(device):
        _call_id += 1
        call = _call_id
        names = [f"_bt_{call}_in{i}.mem" for i in range(len(tensors))]
        out_names = [f"_bt_{call}_out{i}.mem" for i in range(_n_out)] if want_output else []
        _allocator.device = device
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
                results.append(_allocator.take(ptr.value).reshape(_output_shape(dims, min_ndim)))
            return results[0] if len(results) == 1 else tuple(results)
        finally:
            for name in names + out_names + list(flag_names.values()):
                lib.bartorch_unlink(name.encode())
