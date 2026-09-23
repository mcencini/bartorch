Basics
------

The conventions, the two ways of writing a reconstruction, and the channel
noise model of a SENSE reconstruction.

The first example follows undersampled Cartesian k-space to an image: the
sampling pattern, channel compression, ESPIRiT calibration, and reconstruction
with :func:`bartorch.tools.pics`. The second assembles the same problem out of
an encoding operator and a solver from :mod:`bartorch.optim`, which is how a
reconstruction BART has no application for is written. The third measures the effect
of correlated channel noise on a SENSE reconstruction, and its removal by
prewhitening with a noise measurement.
