"""Running BART on a CUDA device.

A tool or operator given CUDA tensors runs on that device: BART recognises
the pointers, allocates its outputs through torch on the same device, and its
streams are ordered against torch's current stream, so nothing is copied
through the host and neither side synchronises the device.

``bartorch.cuda.available()`` says whether that path exists here.
:func:`set_streams` and :func:`use_memcache` tune how BART uses the card.
"""

from bartorch._cuda import (
    available,
    built,
    device_count,
    free_memory,
    ordered,
    set_streams,
    use_memcache,
)

__all__ = [
    "available",
    "built",
    "device_count",
    "free_memory",
    "ordered",
    "set_streams",
    "use_memcache",
]
