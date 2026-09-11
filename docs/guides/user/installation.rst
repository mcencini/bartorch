Installation
============

Install PyTorch first
---------------------

Use the `PyTorch installation selector <https://pytorch.org/get-started/locally/>`_
to install the CPU or CUDA build you want in your active Python environment.
Then install bartorch into that same environment::

   python -m pip install bartorch

The metadata currently requires Python 3.10+, PyTorch 2.1+, NumPy 1.24+, and
SciPy 1.10+. NumPy and SciPy are installed as dependencies. A compatible wheel
includes the embedded BART library; wheel users do not need a separate BART
executable or a C compiler.

.. note::

   This checkout is pre-alpha. The command above is the intended release install
   path, not a claim that wheels for every platform are published. If pip cannot
   find a suitable distribution, use the :doc:`../developer/toolchain` source
   installation. A source archive requires the developer toolchain too.

Check the installation
----------------------

.. code-block:: python

   import torch
   import bartorch
   import bartorch.tools as bt

   print(torch.__version__)
   print(bartorch.__version__)
   print(bartorch.bart_version())
   print(bartorch.build_info())
   image = bt.phantom([32, 32])
   print(image.shape, image.dtype, image.device)

Devices and optional transforms
-------------------------------

CUDA requires both a CUDA-capable PyTorch installation and a bartorch library
built with CUDA. Check ``torch.cuda.is_available()`` and
``bartorch.cuda.available()`` independently. Move input tensors with
``tensor.to("cuda")``. Some tools stage work through CPU memory; tensor placement
alone does not guarantee every step executes on the GPU. CPU and CUDA are the
current device paths; Apple MPS is not supported by the wrappers.

Non-Cartesian reconstruction uses FINUFFT and, on CUDA, cuFINUFFT. FINUFFT is
a dependency rather than an extra -- it computes every non-Cartesian transform,
and BART's own gridder is not reachable from this package -- so ``pip install
bartorch`` already brings it and there is nothing to ask for. cuFINUFFT serves
a transform on a card and stays an extra::

   python -m pip install 'bartorch[cufinufft]'

Wheels are published for Linux x86_64 and macOS on Apple silicon, which are
the platforms FINUFFT ships wheels for too. Elsewhere -- Linux on aarch64, an
Intel Mac -- ``pip install bartorch`` builds from the source distribution, and
builds FINUFFT alongside it; that needs CMake, ninja and a C++ compiler, which
compiling BART needs anyway.

The NUFFT substitution raises when a required backend is missing or a
configuration cannot be served. See :doc:`../../api/generated/finufft` for
configuration and diagnostics. The ``mkl`` extra is optional on supported
platforms; it is not required for the Cartesian examples.

Next: :doc:`conventions` and :doc:`../../auto_examples/index`.
