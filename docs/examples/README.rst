Examples
========

Reconstruction workflows written with bartorch, grouped by what they build.

**Basics** covers the data layout, BART's applications as functions, the same
reconstruction assembled from an encoding operator and a solver, and noise
prewhitening.

**Non-Cartesian imaging** covers trajectories, the non-uniform Fourier
transform and its point spread function, and a radial SENSE reconstruction.

**Applications** covers two problems whose encoding is more than sampling and
coils: a dynamic series reconstructed from golden-angle radial data, and a
subspace-constrained :math:`T_1` mapping experiment.

**Model-based reconstruction** estimates the unknowns jointly with the coil
sensitivities, or estimates parameter maps directly from k-space through a
signal model.

**Deep learning** replaces a specified regularizer with a learned one, by
unrolling a BART iteration and training its proximal step through it.

The concepts these examples use -- the encoding model, regularized
least squares, non-uniform transforms, nonlinear inversion -- are introduced in
:doc:`../explanation/index`.

Running them needs a built ``bartorch``, ``brainweb-dl``, which every example
builds its phantom from, and ``matplotlib`` and ``cmap`` for the figures::

    pip install bartorch brainweb-dl matplotlib cmap

The deep-learning section additionally requires::

    pip install lightning torchio monai deepinv
