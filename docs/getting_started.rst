Getting Started
===============

**bartorch** runs the `Berkeley Advanced Reconstruction Toolbox (BART)
<https://mrirecon.github.io/bart/>`_ inside a Python process on
``torch.Tensor`` objects.  Every BART tool is a function, every BART operator
is an object that applies to tensors, and an operator written in Python can
be handed to BART's solvers.

Installation
------------

.. code-block:: bash

   pip install bartorch

From source, with clang and CMake:

.. code-block:: bash

   git clone --recurse-submodules https://github.com/mcencini/bartpy
   cd bartpy
   pip install -e .

The wheel carries one C library with all of BART and depends on nothing but
``numpy`` and ``torch``.  BLAS and LAPACK come from the library torch already
loaded; the FFT is pocketfft.

Axis convention
---------------

Shapes are C order, so the last axis is the one BART calls the first:

.. code-block:: text

   bartorch shape: (coils, phase2, phase1, read)
   BART dims:      [read, phase1, phase2, coils]

Wherever a BART tool takes a bitmask of dimensions, the Python function takes
axis indices, negative ones included: ``bt.fft(x, axes=(-1, -2))``.

Tools
-----

.. code-block:: python

   import bartorch.tools as bt

   kspace = bt.phantom([256, 256], kspace=True, ncoils=8)
   maps = bt.ecalib(kspace, calib_size=24, maps=1)
   image = bt.pics(kspace, maps, R="W:7:0:0.005")

A tool receives a private copy of each input, because BART tools may write
into their inputs; :func:`bartorch.set_copy_inputs` turns that off for
callers who accept scratch inputs.

Operators
---------

.. code-block:: python

   from bartorch.ops import LinearOperator, NonlinearOperator

   S = LinearOperator.multiply_sum(maps, (1, 256, 256), (8, 256, 256))
   F = LinearOperator.fft((8, 256, 256), axes=(-1, -2))
   A = F @ S                          # apply S, then F
   x = A.lstsq(kspace, lambda_=1e-3)  # BART's conjugate gradients

An operator defined in Python enters BART the same way.  A torch function
becomes a nonlinear operator whose derivative and adjoint come from autograd,
which is how a signal model is fitted by BART's Gauss-Newton solver:

.. code-block:: python

   M = NonlinearOperator.from_torch(signal_model, params_shape, images_shape)
   x = (F @ M).irgnm(kspace, x0, iterations=10)
