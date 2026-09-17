Basics
------

The conventions, the preprocessing, and the two ways of writing a
reconstruction.

The first example follows undersampled Cartesian k-space to an image: the
sampling pattern, coil compression, ESPIRiT calibration, and reconstruction
with :func:`bartorch.tools.pics`. The second assembles the same problem out of
an encoding operator and a solver from :mod:`bartorch.optim`, which is what a
reconstruction that a BART application does not implement is written with.
