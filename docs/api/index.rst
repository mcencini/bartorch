API References
==============

The reference is extracted from this checkout at build time. Handwritten public
wrappers take precedence over the generated command signatures. NumPy-style
parameter documentation is rendered with Napoleon. See
:doc:`../guides/developer/documentation` for the C-help-to-Python pipeline.

.. toctree::
   :maxdepth: 1

   generated/package
   generated/linops
   generated/nlops
   generated/prox
   generated/alg
   generated/interop
   generated/tools
   generated/not_exposed
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
* Assembling one: a regularization term from :mod:`bartorch.prox`, BART's own
  solve as :func:`bartorch.alg.solve`, and the scaling the tool puts around it
  as :func:`bartorch.alg.data_scaling`.

Read :doc:`../guides/user/conventions` before translating CLI arguments.
Some raw option strings retain BART's bitmasks even though axis-oriented wrappers
accept Python indices. C help text alone cannot establish shape contracts,
optional-output support or differentiability.
