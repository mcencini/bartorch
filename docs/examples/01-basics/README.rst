Basics
------

The conventions, the two ways of writing a reconstruction, and the array
operations around one.

The first example follows undersampled Cartesian k-space to an image: the
sampling pattern, channel compression, ESPIRiT calibration, and reconstruction
with :func:`bartorch.tools.pics`. The second assembles the same problem out of
an encoding operator and a solver from :mod:`bartorch.optim`, the route a
reconstruction BART has no application for takes. The third is the rest of the
toolbox: noise prewhitening, alignment, resizing, the image-quality measures,
and BART's file format.
