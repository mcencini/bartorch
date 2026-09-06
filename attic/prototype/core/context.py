"""
bartorch.core.context — Thread-local execution context for batching BART calls.

The ``BartContext`` manages a session of BART in-memory CFL names so that
multiple chained operations share the same backing memory without
re-registering tensors between calls.
"""

from __future__ import annotations

import threading
import uuid
from contextlib import contextmanager
from typing import Generator

import torch


class BartContext:
    """Thread-local session state for the C++ extension dispatcher.

    Within an active context every tensor that enters a BART operation is
    registered in BART's in-memory CFL namespace under a deterministic
    ``_bt_<uuid>.mem`` name.  Intermediate results produced by chained
    operations are stored with matching names, allowing consecutive
    ``bart_command()`` calls to consume them without any Python↔C boundary
    crossings beyond the initial and final ones — no disk I/O, no
    ``/dev/shm`` writes.

    Usage::

        with bart_context() as ctx:
            result = bt.fft(bt.phantom([256, 256]), flags=3)
            # → no copies, two bart_command() calls, all in C

    Outside a context, the dispatcher creates and destroys a mini-session per
    operation.  Both styles produce identical results; the context simply
    eliminates redundant re-registration overhead for call chains.

    Requires the compiled ``_bartorch_ext`` C++ extension.
    """

    _local = threading.local()

    def __init__(self):
        self._registered: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Context manager protocol
    # ------------------------------------------------------------------

    def __enter__(self) -> "BartContext":
        BartContext._local.active = self
        return self

    def __exit__(self, *_):
        self._cleanup()
        BartContext._local.active = None

    # ------------------------------------------------------------------
    # Registry helpers
    # ------------------------------------------------------------------

    def fresh_name(self) -> str:
        """Return a unique ``*.mem`` name for the in-memory CFL registry."""
        return f"_bt_{uuid.uuid4().hex}.mem"

    def register(self, name: str, tensor: torch.Tensor) -> None:
        """Record that *tensor* is registered under *name* in BART's CFL map."""
        self._registered[name] = tensor

    def unregister(self, name: str) -> None:
        """Remove a name from the local tracking dict (does not call BART)."""
        self._registered.pop(name, None)

    def _cleanup(self) -> None:
        """Unlink all in-memory CFLs registered in this session.

        Calls into the C++ extension once compiled; no-op at stub stage.
        """
        self._registered.clear()

    # ------------------------------------------------------------------
    # Class-level helpers
    # ------------------------------------------------------------------

    @classmethod
    def current(cls) -> "BartContext | None":
        """Return the active context for the current thread, or ``None``."""
        return getattr(cls._local, "active", None)

    @classmethod
    def is_active(cls) -> bool:
        """Return ``True`` if a context is active on the current thread."""
        return cls.current() is not None


@contextmanager
def bart_context() -> Generator[BartContext, None, None]:
    """Convenience context manager that creates and activates a :class:`BartContext`.

    Example::

        with bart_context() as ctx:
            y = bt.fft(bt.phantom([256, 256]), flags=3)
    """
    ctx = BartContext()
    with ctx:
        yield ctx
