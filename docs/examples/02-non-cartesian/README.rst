Non-Cartesian imaging
---------------------

Sampling off the Cartesian grid.

The first example is a tour of the trajectory and transform interfaces: the
trajectories :func:`bartorch.tools.traj` generates, the non-uniform Fourier
transform, density compensation, and the point spread function the normal
operator convolves with. The second reconstructs an undersampled radial
acquisition with :class:`bartorch.linop.NoncartesianSense`.
