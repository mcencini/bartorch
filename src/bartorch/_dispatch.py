"""Running a BART command in-process on tensors.

Inputs are registered in the library's in-memory file registry, the argv is
assembled, the command runs under BART's error catcher, and its outputs come
back as tensors torch allocated through the allocator callback.  A C-order
shape ``(a, b, c)`` is BART dims ``[c, b, a]``: the bytes are shared and only
the shape is reversed.
"""

from __future__ import annotations

import contextlib
import logging
import threading
from typing import Any

import numpy as np
import torch

from bartorch import _backend, _cuda, _marshal, _options
from bartorch._lib import ALLOC_FN, DIMS, FREE_FN, LOG_FN, library

__all__ = [
    "BartError",
    "dispatch",
    "get_debug_level",
    "run_command",
    "set_copy_inputs",
    "set_debug_level",
    "set_num_threads",
]

_log = logging.getLogger("bartorch.bart")

_LEVELS = {
    0: logging.ERROR,
    1: logging.WARNING,
    2: logging.INFO,
}


class BartError(RuntimeError):
    """A BART command or operator failed; the message is BART's."""


class _Allocator:
    """Serves BART's own allocations with torch tensors and keeps them alive until taken.

    They are allocated on the device the command runs on: BART's ``md_``
    operations silently take the host path unless every argument is on a device, so
    an allocation on the wrong side of the bus crashes rather than slows down.
    """

    def __init__(self) -> None:
        self.device = torch.device("cpu")
        self.live: dict[int, torch.Tensor] = {}
        self._alloc_cb = ALLOC_FN(self._alloc)
        self._free_cb = FREE_FN(self._free)

    def _alloc(self, _ctx, D, dims):
        shape = [dims[i] for i in range(D)][::-1]
        try:
            # Zeroed, as a new CFL file is: some commands (rof, tgv) start their
            # solver from what the output already holds.
            t = torch.zeros(shape, dtype=torch.complex64, device=self.device)
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
    """Whether a command works on a private copy of each input, which is the default.

    Some commands write into their inputs; with ``False`` a command is handed the
    tensor's own memory and may modify it.  A card tensor given to a command that
    runs on the host is copied to the host regardless.
    """
    global _copy_inputs
    _copy_inputs = bool(copy)


def set_coil_batch(n: int) -> None:
    """Set the coil batch of SENSE operators BART's tools build; 0 uses BART's own operator.

    :class:`bartorch.linop.NoncartesianSense` takes its own and restores this after building.
    """
    _ensure_ready()
    library().bartorch_sense_set_coil_batch(int(n))


def coil_batch() -> int:
    """Coil batch of SENSE operators BART's tools build; zero means BART's own operator."""
    _ensure_ready()
    return int(library().bartorch_sense_coil_batch())


def set_fold_maps(enable: bool = True) -> None:
    """Set ``fold_maps`` of SENSE operators BART's tools build; see ``linop.NoncartesianSense``."""
    _ensure_ready()
    library().bartorch_sense_set_fold_maps(int(bool(enable)))


def fold_maps() -> bool:
    """``fold_maps`` of SENSE operators BART's tools build."""
    _ensure_ready()
    return bool(library().bartorch_sense_fold_maps())


def set_num_threads(n: int) -> None:
    """Set the number of threads BART, its FFT and FINUFFT's host transforms use."""
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
    if isinstance(val, torch.Tensor):
        if val.numel() != 1:
            raise ValueError(f"a command-line value is one number, not an array of {val.numel()}")
        val = val.reshape(()).item()
    if isinstance(val, (tuple, list)):
        return ":".join(_value_str(v) for v in val)
    if isinstance(val, complex):
        # BART reads `1.5`, `2i` or `1.5+2i`, and nothing that looks like
        # Python's own `(1.5+2j)`.
        return f"{val.real!r}{'+' if val.imag >= 0 else '-'}{abs(val.imag)!r}i"
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


def _flag_string(key: str, op_name: str = "") -> str:
    """The command-line flag for keyword ``key`` of command ``op_name``.

    The catalogue is asked first, because BART's long options mix hyphens and
    underscores.  For a keyword it has no entry for: a single letter is ``-x``;
    ``flag_3`` is ``-3``; a trailing ``_<digit>`` repeats a flag (``R_1`` and
    ``R_2`` are both ``-R``); anything else is ``--key`` with hyphens for
    underscores.
    """
    stem, _, suffix = key.rpartition("_")
    repeated = bool(stem) and stem != "flag" and suffix.isdigit()
    if op_name:
        # The whole keyword first: BART has options called `--kfilter-1` and
        # `--kfilter-2`, and reading the digit as a repetition would send both
        # to a `--kfilter` that does not exist.
        for candidate in (key, stem) if repeated else (key,):
            known = _options.flag_for(op_name, candidate)
            if known is not None:
                return known
    if repeated:
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
    _options.check(op_name, kwargs)
    for index, (key, val) in enumerate(_expand_list_flags(kwargs)):
        if val is None or val is False:
            continue
        argv.append(_flag_string(key, op_name))
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


_bart_dims = _marshal.dims


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
    # See as_operand: a conjugated or negated view shares its storage and
    # reports itself contiguous, so it has to be resolved before the pointer
    # is handed over.
    return x.resolve_conj().resolve_neg().contiguous()


_output_shape = _marshal.shape_from_dims


def run_command(argv: list[str]) -> tuple[int, str, str]:
    """Run one command from a complete argv; return (exit code, printed text, error text).

    A help flag is refused: BART answers it by calling ``exit``, which would end
    this process.
    """
    _ensure_ready()
    asked_for_help = _options.HELP_FLAGS.intersection(argv[1:])
    if asked_for_help:
        flag = sorted(asked_for_help)[0]
        raise ValueError(
            f"bart {argv[0]} {flag} would print its usage and exit, which in this "
            f"process ends the interpreter; use bartorch._options.describe({argv[0]!r})"
        )
    lib = library()
    c_argv = _marshal.argv(argv)
    out = _marshal.text_buffer(1 << 16)
    err = _marshal.text_buffer(4096)
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
    """Run BART command ``op_name`` on tensors and return what it writes.

    Host inputs are copied first unless :func:`set_copy_inputs` turned that off,
    because some commands write into their inputs.

    Parameters
    ----------
    op_name : str
        Command name, as on the command line.
    inputs : list of torch.Tensor
        Input arrays, in the order the command takes them.
    output_dims : list of int, None or False
        ``False`` for a command that writes no array: its printed text is returned.
        A list keeps at least ``len(output_dims)`` axes on the output; otherwise
        leading singleton axes are dropped.
    _pos : list, optional
        Scalar positional arguments, placed after the flags and before the inputs.
    _n_out : int
        Output arrays the command writes; more than one returns a tuple.
    **kwargs
        Options.  ``True`` is a bare flag, ``None`` and ``False`` are omitted, a
        list repeats the option, a tuple is joined with colons, and a tensor is
        registered as an array.

    Returns
    -------
    torch.Tensor, tuple of torch.Tensor, str or None
        The outputs, on the inputs' device, or the printed text.

    Raises
    ------
    BartError
        The command exited with an error; the message is BART's.
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
                dims = _marshal.dim_vector()
                ptr = _marshal.out_pointer()
                found = lib.bartorch_lookup(
                    out_name.encode(), DIMS, dims, _marshal.by_reference(ptr)
                )
                if found != 0:
                    raise BartError(f"bart {op_name} did not write {out_name}")
                out = _allocator.take(ptr.value).reshape(_output_shape(dims, min_ndim))
                results.append(out.to(device) if device.type != "cpu" else out)
            return results[0] if len(results) == 1 else tuple(results)
        finally:
            _allocator.device = torch.device("cpu")
            for name in names + out_names + list(flag_names.values()):
                lib.bartorch_unlink(name.encode())
