# Sampling off the Cartesian grid

A radial, spiral or otherwise non-Cartesian acquisition measures k-space at
positions that do not fall on a grid, so the fast Fourier transform does not
apply to it. This page states what replaces the FFT, what the replacement costs
and what it approximates, and how a reconstruction is built on it.

## The transform

The data of a non-Cartesian acquisition is

$$
y_j = \frac{1}{\sqrt{N}} \sum_{r} x_r \, e^{-2\pi i \, k_j \cdot r} ,
\qquad j = 1, \dots, M,
$$

a sum over the $N$ voxels of the image evaluated at $M$ arbitrary points $k_j$.
Computed directly this costs $O(NM)$, which for a 2D acquisition of realistic
size is minutes per application and therefore hours per reconstruction.

A **non-uniform fast Fourier transform** (NUFFT) computes it to a requested
accuracy in $O(N \log N + M)$. Every implementation of one is the same three
operations: a spreading or an interpolation between the samples and a grid
oversampled with respect to the image, an FFT of that grid, and a division by
the kernel's Fourier transform, which is the **deapodization** that undoes the
spreading. The kernel and the oversampling factor set the accuracy, and the
classical accounts of the method are about how to choose them.

In bartorch this transform is [FINUFFT](https://finufft.readthedocs.io), and
what a caller sees is not a kernel and an oversampling but a **tolerance**: the
relative accuracy the transform is planned for, from which the library sizes
its kernel. A type-2 transform (uniform to non-uniform) is the forward
operator and a type-1 (non-uniform to uniform) is its adjoint, with the sign of
the exponent reversed between them.

The default is a tolerance of $10^{-3}$ on a grid a quarter larger than the
image. That is looser than the gridding parameters MRI reconstruction usually
defaults to, and it is a deliberate choice: a reconstruction is limited by its
data, the noise in which is orders of magnitude above $10^{-3}$, so accuracy
bought in the transform is spent where it cannot be seen. Where it matters —
a validation against an analytical reference, a comparison across
implementations — the tolerance is a setting, and so are the oversampling
factor and the kernel width.

## The adjoint is not the inverse

$A^H y$ for a non-Cartesian $A$ sums each sample onto the grid. Where the
trajectory samples some regions more densely than others, the sum is weighted
by the sampling density: for a radial trajectory, every spoke passes through
the centre of k-space and they separate toward the periphery, so the centre is
measured once per spoke and the periphery once per spoke per ring. The adjoint
of such a trajectory is the image convolved with a strongly low-pass kernel.

**Density compensation** corrects this by weighting each sample by the
reciprocal of the local sampling density before the adjoint is applied — the
distance from the centre of k-space, for radial sampling. The result is the
**gridding reconstruction**, which is a reconstruction in the sense that it
produces an image and not in the sense that it inverts anything: it leaves
whatever the trajectory did not sample unrecovered, and at any real
acceleration that omission dominates the result.

An operator carries its weights itself rather than leaving them to the caller,
because the normal operator is built over them (see below) and a chain of
weights around a transform could not be. Weights that do not lie along k-space
are declined, with the reason stated.

## The normal operator and the point spread function

For an iterative reconstruction the quantity that matters is $A^H A$, applied
once per iteration. For a non-Cartesian transform this operator is a
**convolution**: it is translation invariant, because the trajectory does not
depend on where the object is, and a translation-invariant operator is a
multiplication in the Fourier domain.

The array it multiplies by is the **transfer function**, computed as the
adjoint transform of ones along the trajectory, and its inverse transform is
the **point spread function** — the image a single point would reconstruct
into, whose sidelobes are the streaks a radial reconstruction is known for. The
multiplication has to happen on a grid twice the size of the image, because the
convolution of two arrays of size $N$ has support $2N$ and computing it on the
smaller grid would alias.

The gain is substantial. One application of the Toeplitz normal is two FFTs on
a doubled grid and one multiplication; the transform pair it replaces is two
non-uniform transforms over every sample of every channel. Both are measured
side by side in
{doc}`../auto_examples/02-non-cartesian/01-trajectories-and-transforms`.

The cost is memory: the transfer function is an array of $2^d N$ complex
numbers per set of coefficients, which for a 3D subspace reconstruction is the
largest thing in the reconstruction. BART has several ways of storing it — as
a decomposition into $2^d$ shifted copies computed one at a time, as half of a
Hermitian array, as its real part, or compressed to the entries that are not
negligible — and bartorch keeps all of them, because what changes is where the
function comes from and not what is done with it afterwards.

## What the substitution replaces, and what it does not

BART has its own gridding implementation, and bartorch does not use it: the
entry point BART builds every NUFFT through is answered by the FINUFFT
substitution instead, so `nufft`, `pics`, `nlinv`, `moba` and the operators in
{mod}`bartorch.linop` all get the same transform. What stays BART's is
everything around it — the operator's structure, the Toeplitz machinery, the
solvers.

The substitution declines rather than falls back. An answer computed by a
different method, arriving with nothing to say so, is worse than no answer, so
a transform the substitution cannot serve is an error that names its reason
rather than a silent change of implementation. The cases are few and specific:
weights that do not lie along k-space, and an image that varies along an axis
the trajectory indexes, which would require a plan per item.

## Trajectories

{func}`bartorch.tools.traj` generates the trajectories BART supports, in grid
units: the coordinate of a sample in units of the k-space cell of the image it
encodes, so a readout of $N$ samples runs from $-N/2$ to $N/2$. The convention
matters because a transform has to know how many cells the trajectory spans;
radians and cycles per metre are conversions of it.

{func}`~bartorch.tools.traj` offers two orderings of a radial trajectory.
Spokes separated by $\pi/n$ tile k-space uniformly for a frame of $n$ spokes
and for no other number.
**Golden-angle** ordering separates successive spokes by $\pi$ times the golden
ratio conjugate, which tiles k-space approximately uniformly for *any* number
of consecutive spokes. A continuously acquired golden-angle scan can therefore
be cut into frames after the fact, at a frame duration chosen when the data is
reconstructed rather than when it is acquired, as
{doc}`../auto_examples/03-applications/01-dynamic-golden-angle` does.

## Devices

Where a transform runs is decided by where its arguments are: an operator
applied to tensors on a CUDA device plans on the device and executes there,
through cuFINUFFT, which is an optional extra. Without that extra a transform
BART would have run on a card stays on BART's own operator rather than quietly
moving to the host. The k-space and the image of one reconstruction can be in
different places, since BART hands an operator memory on either side, so an
operator holds one plan per place and builds each the first time it is needed.

## References

O'Sullivan JD. A fast sinc function gridding algorithm for Fourier inversion in
computer tomography. *IEEE Trans Med Imaging* 4(4):200-207 (1985).

Fessler JA, Sutton BP. Nonuniform fast Fourier transforms using min-max
interpolation. *IEEE Trans Signal Process* 51(2):560-574 (2003).

Beatty PJ, Nishimura DG, Pauly JM. Rapid gridding reconstruction with a minimal
oversampling ratio. *IEEE Trans Med Imaging* 24(6):799-808 (2005).

Barnett AH, Magland J, af Klinteberg L. A parallel nonuniform fast Fourier
transform library based on an "exponential of semicircle" kernel. *SIAM J Sci
Comput* 41(5):C479-C504 (2019).

Fessler JA, Lee S, Olafsson VT, Shi HR, Noll DC. Toeplitz-based iterative image
reconstruction for MRI with correction for magnetic field inhomogeneity. *IEEE
Trans Signal Process* 53(9):3393-3402 (2005).

Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O. An optimal radial
profile order based on the golden ratio for time-resolved MRI. *IEEE Trans Med
Imaging* 26(1):68-76 (2007).
