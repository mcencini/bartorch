# The MRI encoding operator

The forward operator of an MRI reconstruction says what signal an object would
have produced in a given experiment. This page builds it from the physics,
states the one expression the library represents every variant of it with, and
explains what that representation buys.

## Sensitivity encoding

The signal a receive coil measures is the transverse magnetization of the
object, weighted by the coil's spatial sensitivity and integrated against the
spatial phase the gradients have imposed. With $x(r)$ the object, $S_c(r)$ the
sensitivity of coil $c$ and $k_j$ the position in k-space reached at the $j$-th
sample,

$$
y[c, j] = \int S_c(r) \, x(r) \, e^{-2\pi i \, k_j \cdot r} \, \mathrm{d}r .
$$

Discretized on a voxel grid, this is three operations applied in turn:
multiplication by the sensitivities, a Fourier transform, and a restriction to
the sampled positions. Written as operators,

$$
A = P F S ,
$$

which is the **SENSE** forward model: $S$ maps one image to as many coil
images as there are channels, $F$ is the Fourier transform, and $P$ keeps the
samples the acquisition took.

The sensitivities are what makes an accelerated acquisition invertible. With
one channel, omitting phase encodes leaves a system with more unknowns than
equations and no way to choose among the solutions. With several, each channel
sees the aliased voxels through a different weight, and the aliasing can be
resolved — up to the conditioning of the resulting system, measured by the
geometry factor of parallel imaging. The sensitivities are smooth
functions of position, and the estimate of them is itself a problem:
{func}`~bartorch.tools.ecalib` solves it by ESPIRiT from a fully sampled
neighbourhood of the k-space centre, {func}`~bartorch.tools.ncalib` from
non-Cartesian samples, and {doc}`nonlinear` treats the case where they are
estimated jointly with the image.

## One expression for every encoding

Cartesian SENSE is the simplest member of a family. A wave-encoded acquisition
has a readout gradient that spreads each voxel along the readout; an
off-resonance-corrected reconstruction has a phase that accumulates with time
during the readout; a subspace reconstruction carries coefficients where the
data carries frames; simultaneous multislice excites several slices and adds
their signals. None of these is $PFS$, and each is a small change to it.

The library represents all of them as one expression. For coil $c$, encoding
frame $t$ and sample $k$,

$$
y[c, t, k] = \sum_a O[a, t](k) \; T_t \!\left( I[c, a, t](r) \, x[a](r)
    \right)\!(k) ,
$$

with

* $T$ the **transform**: a fast Fourier transform on the image's grid, a
  non-uniform transform along a trajectory, or a wave front;
* $I$ the **image-side factor**, multiplying the image before the transform:
  the coil sensitivities, and any weight that varies with position — a time
  segment's field phase, a slice selection;
* $O$ the **k-space-side factor**, multiplying the samples after it: the
  sampling pattern, density weights, a subspace basis, a per-shot phase;
* $\sum_a$ the **contraction**, summing terms that are transformed separately:
  the coefficients of a subspace, the segments of a time-segmented field
  correction, the slices of a multislice excitation.

Setting $O$ to a sampling pattern, $I$ to the sensitivities, $T$ to the FFT and
the contraction to nothing recovers $A = PFS$.

The constructors in {mod}`bartorch.linop` —
{func}`~bartorch.linop.CartesianSense`,
{class}`~bartorch.linop.NoncartesianSense`,
{func}`~bartorch.linop.WaveSense`, {func}`~bartorch.linop.FieldCorrected` —
each name a particular instance of that expression, and one executor applies
all of them.

## Composition and lowering

A reconstruction assembled by composition, `P @ F @ S`, would apply its factors
in sequence, each writing a full intermediate array: for a volume with many
channels those intermediates exhaust the memory before the arithmetic becomes
the limit. Recognizing the composition as one encoding lets it be executed
differently — one channel or one slab of channels at a time, with the
element-wise factors folded into the same pass as the transform, and the
intermediates never formed for the whole array at once.

So composition in {mod}`bartorch.linop` builds a *description* rather than an
operator. `@` and `+` record their operands and defer; the first thing that
needs a built operator — an application, an adjoint, a solve — offers the
whole description to a planner, which matches it against the expression above
and lowers it where it fits. Writing the terms of a time-segmented field
correction by hand therefore gives the same operator that
{func}`~bartorch.linop.FieldCorrected` gives, because both arrive at the
planner as the same description.

Some compositions do not fit. A weight that differs between sets of
sensitivities has nowhere to go in the expression, because the sets have
already been contracted by the time the image-side factor is applied; such a
composition is built as a plain sum of chains instead, which computes the same
numbers more slowly. Since the answer is the same either way, only a timing
would reveal which happened, so the operator reports it instead: `A.plan` names
the transform, the factors on each side, the contraction, and which executor
ran it, read back from the library after the build rather than predicted.
`plan.fused` is the field to check.

## The normal operator

A solver built on the normal equations applies $A^H A$ once per iteration
rather than $A$ and $A^H$ in turn, so the cost of a reconstruction is the cost
of the normal operator. Two structures make it cheaper than the two
applications it is defined as.

On a Cartesian grid with a sampling pattern, $A^H A$ is a multiplication by
$|P|^2$ between two transforms, and the sum over frames can be done once when
the operator is built rather than in every iteration.

Off the grid, $A^H A$ is a **convolution**: the adjoint of a non-uniform
transform followed by the transform itself is a translation-invariant operator,
so it can be applied as a multiplication in a doubled Fourier domain by a
single array — the **transfer function** of the trajectory, whose inverse
transform is its **point spread function**. Computing that array costs one
adjoint transform of ones, once, and every iteration afterwards
costs a pair of FFTs on a doubled grid instead of a pair of non-uniform
transforms over every sample of every channel. {doc}`non-cartesian` takes this
up in detail.

## Encoding axes and batches

Two axes are easy to confuse, and the library keeps them apart.

An **encoding axis** is an axis the transform sees: frames of a dynamic
acquisition, echoes, cardiac phases. The trajectory indexes it, so its samples
belong to one transform, and a subspace basis contracts it — the image carries
coefficients where the samples carry frames.

A **batch** is an axis the transform does not see: independent slices,
averages, repetitions that share a trajectory and a pattern. Each item is
encoded on its own, and off the grid one plan serves all of them: a batch of
transforms against one point set costs one plan rather than one per item.

The distinction has a consequence worth stating: a trajectory that varies
across frames is still one transform. Every sample the trajectory indexes
belongs to the same point set, so frames join the readout and the shots in it
rather than splitting it into a transform per frame. What cannot be expressed
this way is an *image* that varies along an axis the trajectory indexes, which
would need a plan per item; the library declines it rather than computing it
slowly.

## Where to go next

{doc}`../auto_examples/01-basics/02-operators-and-solvers` builds the Cartesian
encoding and checks it against the definition of an adjoint;
{doc}`../auto_examples/03-applications/02-subspace-t1-mapping` uses the
subspace contraction; the shapes each constructor expects are in
{doc}`../guides/user/conventions` and in the reference pages of
{mod}`bartorch.linop`.

## References

Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: sensitivity
encoding for fast MRI. *Magn Reson Med* 42(5):952-962 (1999).

Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig
M. ESPIRiT -- an eigenvalue approach to autocalibrating parallel MRI: where
SENSE meets GRAPPA. *Magn Reson Med* 71(3):990-1001 (2014).

Tamir JI, Uecker M, Chen W, Lai P, Alley MT, Vasanawala SS, Lustig M. T2
shuffling: sharp, multicontrast, volumetric fast spin-echo imaging. *Magn Reson
Med* 77(1):180-195 (2017).

Bilgic B, Gagoski BA, Cauley SF, Fan AP, Polimeni JR, Grant PE, Wald LL,
Setsompop K. Wave-CAIPI for highly accelerated 3D imaging. *Magn Reson Med*
73(6):2152-2162 (2015).
