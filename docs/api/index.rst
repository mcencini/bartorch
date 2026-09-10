API References
==============

The reference is extracted from this checkout at build time. Handwritten public
wrappers take precedence over the generated command signatures. NumPy-style
parameter documentation is rendered with Napoleon. See
:doc:`../guides/developer/documentation` for the C-help-to-Python pipeline.

.. toctree::
   :maxdepth: 1

   generated/package
   generated/operators
   generated/tools
   generated/cuda
   generated/finufft
   generated/cfl

Useful starting points
----------------------

* Simulation and Fourier transforms: :func:`bartorch.tools.phantom`,
  :func:`bartorch.tools.fft`, :func:`bartorch.tools.ifft`,
  :func:`bartorch.tools.traj`, :func:`bartorch.tools.nufft`.
* Coil preparation: :func:`bartorch.tools.whiten`, :func:`bartorch.tools.cc`,
  :func:`bartorch.tools.ccapply`, :func:`bartorch.tools.ecalib`.
* Reconstruction: :func:`bartorch.tools.pics`, :func:`bartorch.tools.nlinv`,
  :func:`bartorch.tools.moba`, :func:`bartorch.tools.wave`,
  :func:`bartorch.tools.wshfl`.
* Native apps: the operators in :mod:`bartorch.linop` and
  :mod:`bartorch.nlop`, each a class of its own over
  :class:`bartorch.linop.LinearOperator` and
  :class:`bartorch.nlop.NonlinearOperator`.

Read :doc:`../guides/user/conventions` before translating CLI arguments.
Some raw option strings retain BART's bitmasks even though axis-oriented wrappers
accept Python indices. C help text alone cannot establish shape contracts,
optional-output support or differentiability.
