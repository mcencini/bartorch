Examples
========

Reconstruction workflows written with bartorch, grouped by what they build.

**Basics** covers the data layout, the preprocessing a reconstruction starts
from, BART's applications as functions, and the same reconstruction assembled
from an encoding operator and a solver.

**Non-Cartesian imaging** covers trajectories, the non-uniform Fourier
transform and its point spread function, and a radial SENSE reconstruction.

**Applications** covers two problems whose encoding is more than sampling and
coils: a dynamic series reconstructed from golden-angle radial data, and a
subspace-constrained :math:`T_1` mapping experiment.

**Model-based reconstruction** estimates the unknowns jointly with the coil
sensitivities, or estimates parameter maps directly from k-space through a
signal model.

The concepts these examples use -- the encoding model, regularized
least squares, non-uniform transforms, nonlinear inversion -- are introduced in
:doc:`../explanation/index`.

Running them needs a built ``bartorch`` and, for the examples that use an
anatomical phantom, ``brainweb-dl``::

    pip install bartorch matplotlib brainweb-dl
