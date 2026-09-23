# Data layout and conventions

## Array order and axes

Tensors are in C order, and a BART dimension vector is the reversed shape: the
last axis of a tensor is BART's first dimension.

| Data | Tensor shape | BART dimensions |
| --- | --- | --- |
| Cartesian coil k-space | `(coils, z, y, x)` | `[x, y, z, coils]` |
| Radial trajectory | `(spokes, samples, 3)` | `[3, samples, spokes]` |
| Radial coil samples | `(coils, spokes, samples, 1)` | `[1, samples, spokes, coils]` |

An axis argument is an index into the tensor's shape, negative indices
included: `bartorch.fft(x, axes=(-2, -1))`.  No argument takes a BART bitmask
or dimension number, and a set of indices that are not axes, such as coil
channels or parameter maps, is also a tuple of indices.  Regularization is
given as {mod}`bartorch.priors` terms rather than as `-R` strings:
`pics(kspace, maps, regularizers=priors.Wavelet((-1, -2), 0.005))`.  A command
that reads no array, such as `seq`, counts axes from the last one and accepts
negative axes only.

## Commands

The functions of {mod}`bartorch.tools` and the `bartorch` namespace keep BART's
dimension order.  BART's coil dimension is its dimension 3 and sets of
sensitivity maps its dimension 4, so the corresponding singleton axes are kept
in inputs and outputs; returned shapes are those BART produces, with trailing
BART singletons (leading tensor axes) removed.  Array inputs are converted to
contiguous `complex64`, and a command works on a copy of each input unless
{func}`bartorch.set_copy_inputs` is set to `False`, because some BART commands
write into their inputs.

## Operators

The MRI operators of {mod}`bartorch.linop` use a layout of their own: batch
axes first, then coils, then the encoding.

| Array | Shape |
| --- | --- |
| Image | `(*batches, [sets,] *encoding, [z,] y, x)` |
| Sensitivities | `([sets,] coils, [z,] y, x)` |
| Trajectory | `(*encoding, shots, samples, ndim)` |
| Non-Cartesian samples | `(*batches, coils, *encoding, shots, samples)` |
| Cartesian samples | `(*batches, coils, *encoding, [z,] y, x)` |
| Sampled phase-encode samples | `(*batches, coils, [frames,] shots, readout)` |
| Sampled phase-encode table | `([frames,] shots, d)` |
| NUFFT and FFT samples | `(*batches, *encoding, shots, samples)` |

A batch axis indexes independent volumes that share the trajectory or sampling
pattern, such as slices or averages; each item is encoded separately.
Encoding axes are the axes the trajectory has in front of its shots, such as
frames, echoes or cardiac phases.  A two-dimensional problem has no `z` axis.
A NUFFT or FFT without sensitivities treats coils as a batch axis.  A subspace
basis of shape `(coeffs, frames)` contracts the last encoding axis, so the
image has coefficients where the samples have frames.  A sampling pattern
broadcasts over one coil's samples: a pattern of shape `(y, 1)` selects phase
encodes for every coil and batch item.

## Trajectories

Trajectories hold `kx, ky, kz` in grid units, the k-space coordinate in units
of $1/\mathrm{FOV}$ of the image being encoded: a fully sampled readout of $N$
samples spans $-N/2$ to $N/2$.  A trajectory always has three components;
a `kz` that is zero throughout makes the transform two-dimensional.
{func}`bartorch.tools.traj` generates trajectories in these units.

## Fourier transform conventions

| Function or operator | Centring | Normalization |
| --- | --- | --- |
| {func}`bartorch.fft` | Centred unless `uncentred=True` | Unnormalized unless `unitary=True` |
| {class}`bartorch.linop.FFT` | Centred unless `centred=False` | Unitary |
| {func}`bartorch.nufft`, {class}`bartorch.linop.NUFFT` | — | $1/\sqrt{N}$ for $N$ image voxels; negative exponent in the forward transform |

State the centring and normalization when comparing reconstructions from
different software.

## CFL files

{func}`bartorch.io.readcfl` and {func}`bartorch.io.writecfl` exchange NumPy
arrays in BART's dimension order, so the axes are reversed at that boundary:

```python
import numpy as np
import torch
from bartorch.io import readcfl, writecfl

tensor = torch.from_numpy(np.ascontiguousarray(readcfl("kspace").T))  # kspace.hdr, kspace.cfl
writecfl("result", tensor.detach().cpu().numpy().T)
```

## Differentiation

Applying an operator, a nonlinear operator or a solver of {mod}`bartorch.optim`
to a tensor that requires a gradient records the operation for autograd; the
functions of {mod}`bartorch.tools` record nothing.
{doc}`../../explanation/differentiation` describes the backward pass of each.
