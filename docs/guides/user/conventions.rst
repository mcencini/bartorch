Tensors, tools, and operators
=============================

Shape and axis conventions
--------------------------

Python shapes are C order. The BART dimension vector is the reversed shape:

.. list-table::
   :header-rows: 1

   * - Data
     - Python shape
     - BART dimensions
   * - Cartesian coil k-space
     - ``(coils, z, y, x)``
     - ``[x, y, z, coils]``
   * - Radial trajectory
     - ``(spokes, samples, 3)``
     - ``[3, samples, spokes]``
   * - Radial coil samples
     - ``(coils, spokes, samples, 1)``
     - ``[1, samples, spokes, coils]``

Keep meaningful singleton axes when calling tools: BART's coil dimension is
always dimension 3, and map sets occupy dimension 4. Inspect returned shapes
instead of assuming every singleton is removed. Compact shapes such as
``(coils, y, x)`` work in explicitly constructed operators; this does not move
the fixed coil dimension in BART commands. Use ``squeeze()`` for display, after
reconstruction.

Python axis arguments accept indices, including negative indices:
``bt.fft(x, axes=(-2, -1))``. Raw option payloads such as ``R="W:7:0:0.005"``
retain BART's grammar: ``7`` is a BART bitmask, not a Python axis index.
Trajectories carry ``kx, ky, kz`` in grid units, not radians or cycles/metre.

Tools and composed apps
-----------------------

Functions in ``bartorch.tools`` take and return tensors. Dispatch normalizes
array inputs to contiguous ``complex64`` and copies tool inputs by default
because BART commands can mutate them. Keep that default while learning;
``set_copy_inputs(False)`` permits scratch-input semantics.

``LinearOperator`` supports forward application, ``adjoint``, ``normal``,
composition with ``@``, addition, and a conjugate-gradient ``lstsq`` solver.
``P @ F @ S`` applies sensitivity multiplication, Fourier encoding, and sampling
in that order. ``NonlinearOperator.from_torch`` provides derivatives of a
PyTorch signal model to BART's Gauss-Newton solver.

``LinearOperator.fft`` is centered and unitary by default. The FFT tool is
unnormalized unless ``unitary=True`` is passed. State centering and normalization
when comparing reconstructions; see the FFT tool reference for its options.

Autograd boundary
-----------------

Receiving a torch tensor does not establish a PyTorch backward graph through
the native library. Current operator applications write to newly allocated
buffers through C; ``from_torch`` uses autograd internally for BART's local
derivatives. Neither implies differentiation through a complete BART solve.
The DeepInverse gallery supplies an explicit, fixed-linear-operator autograd
adapter and checks its gradient. Training acquisition parameters or
differentiating a solver requires additional derivative definitions.

CFL interoperability
--------------------

``bartorch.utils.cfl.readcfl`` and ``writecfl`` use **BART/Fortran axis order**,
unlike tensor tools. Reverse the axes explicitly at that boundary:

.. code-block:: python

   import numpy as np
   import torch
   from bartorch.utils.cfl import readcfl, writecfl

   bart_array = readcfl("kspace")  # basename, without .cfl or .hdr
   tensor = torch.from_numpy(np.ascontiguousarray(bart_array.T))
   writecfl("result", tensor.detach().cpu().numpy().T)

For NumPy arrays, ``.T`` reverses all axes. These helpers exchange existing
files; tool-to-tool calls pass tensors in memory.
