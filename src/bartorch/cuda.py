"""Running BART on a CUDA device.

A tensor on a card selects that card, and BART's streams are ordered against
torch's current stream by an event in each direction, so neither side
synchronises the device.  What crosses the host depends on which layer is
asked:

An **operator** takes the memory as it is.  BART's linops reach their
arguments through ``md_`` operations, which dispatch on where a pointer is,
so a device tensor is transformed where it lies and the result is written
back there.

A **tool** keeps the memory it was given where that is safe.  The few entry
points BART reads element by element rather than through ``md_`` -- sizing an
image from a trajectory, taking a median of k-space -- answer over a host copy
of that one array, the way BART already guards them against its own virtual
pointers.  What cannot be reached that way is a tool that allocates a
temporary of its own on the host and mixes it with what it was handed:
``md_`` operations take the host path unless every argument is on a device,
and take it silently.  So a tool is kept on the card only where it has been
run there and checked against the host, and the rest are given host memory and
their result handed back on the card.  The work in between is BART's own
device path either way, the one ``-g`` selects on the command line, so passing
``-g`` as well changes nothing.

``bartorch.cuda.available()`` says whether any of this exists here.
:func:`set_streams` and :func:`use_memcache` tune how BART uses the card.
"""

from bartorch._cuda import (
    available,
    built,
    device,
    device_count,
    free_memory,
    ordered,
    set_streams,
    streams,
    use_memcache,
)

__all__ = [
    "available",
    "built",
    "device",
    "device_count",
    "free_memory",
    "ordered",
    "set_streams",
    "streams",
    "use_memcache",
]
