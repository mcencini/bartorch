# Linear operators

`bartorch.linop` represents a linear operator $A$ from a domain of C-order
shape `ishape` to a codomain of shape `oshape`, with its adjoint $A^H$ and its
normal operator $A^H A$.  Operators combine by `A @ B` (composition), `A + B`,
`A - B`, `c * A`, `A ** n`, `A.H` (adjoint), `A.T` (transpose) and `A[key]`
(restriction of the codomain); each result is a {obj}`~bartorch.linop.LinearOperator`,
and a composition of BART-backed operators is built as one BART operator.
{doc}`../explanation/encoding` describes the MRI encoding these operators
implement and {doc}`../explanation/differentiation` their autograd behaviour.

```{eval-rst}
.. currentmodule:: bartorch.linop
```

| Operation | Result |
| --- | --- |
| `A(x)`, `A.forward(x)` | $Ax$; `A(x)` is recorded by autograd, `forward` is not |
| `A.adjoint(y)`, `A.H(y)` | $A^H y$ |
| `A.normal(x)`, `A.gram()` | $A^H A x$, and $A^H A$ as an operator |
| `A.cogram()` | $A A^H$ as an operator |
| `A.plan` | The encoding form an MRI operator was lowered into, or `None` |

## MRI encoding operators

| Object | Transform | Model |
| --- | --- | --- |
| {obj}`~bartorch.linop.CartesianSense` | FFT | Cartesian SENSE, $A = PFS$, with optional subspace basis and sampled-encode table |
| {obj}`~bartorch.linop.NoncartesianSense` | NUFFT | Non-Cartesian SENSE, $A = W\,\mathrm{NUFFT}\,S$, with optional density weights and subspace basis |
| {obj}`~bartorch.linop.WaveSense` | Wave-encoded FFT | Wave-CAIPI and Wave-Shuffling encoding |
| {obj}`~bartorch.linop.FieldCorrected` | Any of the above | Off-resonance correction by time segmentation, $\sum_l \operatorname{diag}(b_l)\,E\,\operatorname{diag}(c_l)$ |

## Operator class

| Object | Description |
| --- | --- |
| {obj}`~bartorch.linop.LinearOperator` | Base class: forward, adjoint and normal applications, operator algebra, and operators defined by Python callbacks |

## Elementary operators

| Object | Description |
| --- | --- |
| {obj}`~bartorch.linop.Identity` | Identity on a shape |
| {obj}`~bartorch.linop.Zero` | Zero operator |
| {obj}`~bartorch.linop.Diagonal` | Pointwise multiplication by a broadcast tensor |
| {obj}`~bartorch.linop.ComponentDiagonal` | Separate real scalings of the real and imaginary parts (real-linear) |
| {obj}`~bartorch.linop.Conj` | Complex conjugation (real-linear) |
| {obj}`~bartorch.linop.Real` | Real part (real-linear) |
| {obj}`~bartorch.linop.FFT` | Unitary Fourier transform along axes, centred by default |
| {obj}`~bartorch.linop.NUFFT` | Non-uniform Fourier transform along a trajectory, with optional weights and basis |
| {obj}`~bartorch.linop.MultiplySum` | Multiplication by a tensor followed by summation over axes absent from the codomain |

## Matrix, convolution and finite-difference operators

| Object | Description |
| --- | --- |
| {obj}`~bartorch.linop.Matrix` | Multiplication by a matrix along one axis |
| {obj}`~bartorch.linop.Convolve` | Convolution with a fixed kernel |
| {obj}`~bartorch.linop.Gradient` | Forward finite differences with circular boundary, stacked on a new leading axis |

## Stacking and block composition

| Object | Description |
| --- | --- |
| {obj}`~bartorch.linop.concatenate` | Operators applied to one input, outputs concatenated along an axis |
| {obj}`~bartorch.linop.stack` | Operators applied to one input, outputs stacked on a new axis |
| {obj}`~bartorch.linop.hstack` | Input split between operators, outputs summed |
| {obj}`~bartorch.linop.block_diag` | Block-diagonal operator |
| {obj}`~bartorch.linop.block` | Block matrix of operators |

## Shape and indexing operators

| Object | Description |
| --- | --- |
| {obj}`~bartorch.linop.Reshape` | Same elements under another shape |
| {obj}`~bartorch.linop.Transpose` | Exchange of two axes |
| {obj}`~bartorch.linop.Permute` | Permutation of the axes |
| {obj}`~bartorch.linop.Flip` | Reversal along axes |
| {obj}`~bartorch.linop.Roll` | Cyclic shift along one axis |
| {obj}`~bartorch.linop.Pad` | Zero padding |
| {obj}`~bartorch.linop.Resize` | Centred cropping or zero filling |
| {obj}`~bartorch.linop.Extract` | Extraction of a block |
| {obj}`~bartorch.linop.Hankel` | Sliding-window (Hankel) embedding along one axis |
| {obj}`~bartorch.linop.Sum` | Sum over axes |
| {obj}`~bartorch.linop.ScaledSum` | Sum over axes divided by the square root of the number of terms |
| {obj}`~bartorch.linop.Mean` | Mean over axes |
| {obj}`~bartorch.linop.Repeat` | Repetition along axes |

## User-defined operators

{meth}`LinearOperator.from_callbacks` builds an operator from Python functions
for the forward, the adjoint and, optionally, the normal; BART calls them
through callbacks, one call into Python per application.  A subclass that
defines `forward` and `adjoint` in Python is equivalent.

Complete reconstructions with these operators are in
{doc}`../auto_examples/01-basics/02-operators-and-solvers` and
{doc}`../auto_examples/02-non-cartesian/02-radial-sense`.
