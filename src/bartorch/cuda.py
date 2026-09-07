"""Running BART on a CUDA device.

A tensor on a card selects that card, and BART's streams are ordered against
torch's current stream by an event in each direction, so neither side
synchronises the device.  What crosses the host depends on which layer is
asked:

An **operator** takes the memory as it is.  BART's linops reach their
arguments through ``md_`` operations, which dispatch on where a pointer is,
so a device tensor is transformed where it lies and the result is written
back there.

A **tool** is given host memory.  BART's tools are command mains that map
their inputs the way the command line does, and several read them there --
``estimate_im_dims`` in ``nufft``, the sort in ``pics``'s scaling estimate,
``gram_matrix`` in ``ecalib`` -- so the tensors cross to the host and the
result crosses back.  The work in between is still BART's own device path,
the one ``-g`` selects on the command line, on the card the tensors came
from; passing ``-g`` as well changes nothing.

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
    "use_memcache",
]
