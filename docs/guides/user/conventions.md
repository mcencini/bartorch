# Conventions

## Shapes and axes

Python shapes are C order.  The BART dimension vector is the reversed shape:

| Data | Python shape | BART dimensions |
| --- | --- | --- |
| Cartesian coil k-space | `(coils, z, y, x)` | `[x, y, z, coils]` |
| Radial trajectory | `(spokes, samples, 3)` | `[3, samples, spokes]` |
| Radial coil samples | `(coils, spokes, samples, 1)` | `[1, samples, spokes, coils]` |

Keep meaningful singleton axes when calling BART's applications: BART's coil
dimension is always dimension 3, and sets of maps occupy dimension 4.  Inspect
returned shapes rather than assuming every singleton is removed.  Use
`squeeze()` for display, after reconstruction.  The operators in
`bartorch.linop` use a layout of their own; see Operator layout.

Axis arguments are indices, negative ones included:
`bartorch.fft(x, axes=(-2, -1))`, and no argument takes a BART bitmask.  A
set of indices that are not axes -- coil channels, parameter maps -- is a
tuple of indices too.  Regularization is a list of `bartorch.prox` terms,
never a `-R` string: `pics(..., regularizers=prox.Wavelet((-1, -2), 0.005))`.
A command that reads no array, such as `seq`, counts axes from the last one
and takes negative axes only.  Trajectories carry `kx, ky, kz` in grid units,
not radians or cycles per metre.

## Operator layout

The MRI operators in `bartorch.linop` lay arrays out the torch way, not in
BART's axis order: batches first, then coils, then the problem.

| Array | Shape |
| --- | --- |
| Image | `(*batches, [sets,] *encoding, [z,] y, x)` |
| Sensitivities | `([sets,] coils, [z,] y, x)` |
| Trajectory | `(*encoding, shots, samples, ndim)` |
| Non-Cartesian samples | `(*batches, coils, *encoding, shots, samples)` |
| Cartesian samples | `(*batches, coils, *encoding, [z,] y, x)` |
| NUFFT and FFT samples | `(*batches, *encoding, shots, samples)` |

Batches are volumes that share a trajectory or a pattern, such as slices or
averages, and are applied independently.  A NUFFT or an FFT without
sensitivities treats coils as one more batch axis.  Encoding axes are whatever
the trajectory has in front of its shots: frames, echoes, cardiac phases, in
any number.  A two-dimensional problem has no `z` axis anywhere.

A basis `(coeffs, frames)` contracts the last encoding axis: the image carries
coefficients where the samples carry frames.  A sampling pattern broadcasts
over one coil's samples, so `(y, 1)` undersamples a phase encode for every coil
and batch.  Several sets of maps together with an encoding axis cost one copy
of the image per application.

The commands in `bartorch.tools` keep BART's order; see Shapes and axes.

## Tools and operators

Functions take and return tensors.  Array inputs become contiguous `complex64`,
and a command works on a copy of each input by default, because some BART
commands write into their inputs; `bartorch.set_copy_inputs(False)` hands them
the tensor's own memory.

A {class}`~bartorch.linop.LinearOperator` has a forward, an `adjoint` and a
`normal`, composes with `@` and `+`, and is solved by the classes in
{mod}`bartorch.optim`.  `P @ F @ S` applies sensitivities, then the Fourier
transform, then sampling.  {class}`~bartorch.nlop.FromTorch` gives BART's
Gauss-Newton solver a PyTorch signal model with its derivatives.

{class}`~bartorch.linop.FFT` is centred and unitary by default;
{func}`bartorch.fft` is centred and unnormalized unless `unitary=True`.  State
centring and normalization when comparing reconstructions.

## Autograd

Applying an operator to a tensor that requires a gradient records it, and the
backward pass is the adjoint: for complex tensors, the conjugate Wirtinger
gradient torch expects.  A nonlinear operator's backward pass is the adjoint of
its derivative at the evaluated point.  BART's commands and solvers
({mod}`bartorch.tools`, {mod}`bartorch.optim`) record nothing, so no gradient
flows through a BART reconstruction or solve.

## CFL files

{func}`bartorch.io.readcfl` and {func}`bartorch.io.writecfl` use BART's axis
order, the reverse of a tensor's.  Reverse the axes at that boundary:

```python
import numpy as np
import torch
from bartorch.io import readcfl, writecfl

bart_array = readcfl("kspace")  # basename, without .cfl or .hdr
tensor = torch.from_numpy(np.ascontiguousarray(bart_array.T))
writecfl("result", tensor.detach().cpu().numpy().T)
```

For NumPy arrays, `.T` reverses all axes.  These read and write existing
files; calls between functions pass tensors in memory.
