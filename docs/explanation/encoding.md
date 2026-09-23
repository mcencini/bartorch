# The MRI encoding operator

```{admonition} TL;DR
:class: tldr

- The SENSE forward model is $A = PFS$; coil sensitivities make an undersampled acquisition solvable within the limits of the coil geometry.
- Every encoding bartorch builds is one expression, $y = \sum_a O \cdot T(I \cdot x_a)$: an image-side factor, a transform (FFT, NUFFT or wave), a k-space-side factor and a contraction, applied by one executor a slab of coils at a time.
- Compositions written with `@` and `+` are matched against that form when first used; `A.plan` reports the result, including a fallback to a chain of operators.
- The normal operator $A^H A$ is applied through a kernel — the sampling pattern on a grid, a point spread function on a doubled grid off it — or, where no kernel exists, as the forward operator followed by the adjoint.
```

The forward operator of an MRI reconstruction maps an object to the signal the
experiment would have measured.  This page derives it for parallel imaging,
states the single expression bartorch represents every encoding with, and
describes how compositions of operators are built into that expression and how
its normal operator is applied.

## Sensitivity encoding

A receive coil measures the transverse magnetization of the object, weighted
by the coil's spatial sensitivity and integrated against the phase imposed by
the gradients.  With $x(r)$ the object at position $r$, $S_c(r)$ the
sensitivity of coil $c$ and $k_j$ the k-space position of sample $j$, in cycles
per unit length,

$$
y_{c,j} = \int S_c(r)\, x(r)\, e^{-2\pi i\, k_j \cdot r}\, \mathrm{d}r .
$$

On a voxel grid this is the composition

$$
A = P F S ,
$$

the **SENSE** forward model:[^pruessmann1999] $S$ maps an image to one coil
image per channel, $F$ is the Fourier transform of each coil image, and $P$
selects the acquired samples.

With $R$-fold Cartesian undersampling, each voxel of a coil image obtained by
the inverse FFT of the undersampled data is the superposition of $R$ object
voxels separated by $\mathrm{FOV}/R$.  For $C$ coils the
aliased voxels satisfy $C$ equations in $R$ unknowns; they are determined when
$C \ge R$ and the sensitivity vectors of the aliased voxels are linearly
independent, and the conditioning of this small system — the g-factor — sets
the noise amplification.  Coil sensitivities therefore make an accelerated
acquisition solvable within the limits of the coil geometry; beyond them, and
outside the object, the system is underdetermined or poorly conditioned and a
regularized estimate is needed ({doc}`inverse-problems`).

The sensitivities are smooth functions of position, estimated from the data:
{func}`~bartorch.tools.ecalib` by ESPIRiT from a fully sampled region of the
k-space centre,[^espirit] {func}`~bartorch.tools.ncalib` from non-Cartesian
samples, and {doc}`nonlinear` treats their estimation jointly with the image.

## The encoding form

Wave-encoded, subspace-constrained, off-resonance-corrected and simultaneous
multislice acquisitions each modify $PFS$.  bartorch represents all of them by
one expression: for coil $c$, encoding frame $t$ and sample $j$,

$$
y[c, t, j] = \sum_a O[a, t](j)\; T_t\!\left( I[c, a, t](r)\, x[a](r) \right)\!(j).
$$

```{image} ../_static/encoding.svg
:class: only-light
:alt: The image or its coefficients x[a] is multiplied by the image-side factor I, transformed by T, multiplied by the k-space-side factor O, and summed over a to give the samples y
:width: 100%
```

```{image} ../_static/encoding-dark.svg
:class: only-dark
:alt: The image or its coefficients x[a] is multiplied by the image-side factor I, transformed by T, multiplied by the k-space-side factor O, and summed over a to give the samples y
:width: 100%
```

| Symbol | Name | Instances |
| --- | --- | --- |
| $I$ | Image-side factor | Coil sensitivities; spatial weights of the terms of a contraction, such as the field-map phase of a time segment |
| $T$ | Transform | FFT on the image grid; NUFFT along a trajectory ({doc}`non-cartesian`); wave encoding, an FFT with a point spread function between the readout and phase-encode transforms[^wave] |
| $O$ | K-space-side factor | Sampling pattern or table of sampled phase encodes; density weights; subspace basis;[^tamir] sample weights of the terms of a contraction |
| $\sum_a$ | Contraction | Subspace coefficients; time segments of an off-resonance correction;[^sutton] slices of a simultaneous multislice acquisition |

With $I$ the sensitivities, $T$ the FFT, $O$ a sampling pattern and no
contraction, the expression is $A = PFS$.  {func}`~bartorch.linop.CartesianSense`,
{class}`~bartorch.linop.NoncartesianSense`, {func}`~bartorch.linop.WaveSense`
and {func}`~bartorch.linop.FieldCorrected` each construct an instance, and one
executor applies all of them.  It processes the coils a slab at a time, so
that the coil images and their transforms are never held for all coils at
once, and keeps the element-wise factors in the same pass as the transform.

## Composition and lowering

Composing operators with `@` and `+` records a description instead of building
an operator.  The first operation that needs the built operator — an
application, `.H`, a solve, or reading `.plan` — passes the whole description
to a planner, which matches it against the encoding form.  Where it matches,
the composition is built as one encoding (**lowering**); a time-segmented
off-resonance correction written term by term is therefore built into the same
operator {func}`~bartorch.linop.FieldCorrected` gives.  Where it does not
match — for example, a spatial weight that differs between sets of
sensitivities, which the form cannot carry because the sets are contracted
before the image-side factor is applied — the composition is built as BART's
chain of operators, with the same result and a higher cost.

`A.plan` reports the outcome, read back from the library after the operator is
built:

| Field | Reports |
| --- | --- |
| `transform` | `"fft"`, `"nufft"`, `"wave"` or `"none"` |
| `image`, `kspace` | The element-wise factors on each side of the transform |
| `contraction`, `terms` | `"subspace"`, `"segments"`, `"slices"`, `"chained"` or `None`, and the number of terms |
| `normal` | How $A^H A$ is applied: `"kernel"`, `"transform"` or `"applications"` (below) |
| `coil_batch`, `streamed` | Coils per slab, and what is processed a slab at a time |
| `executor` | `"slab"` where the executor took the form, `"chain"` where BART's chain of operators runs instead |
| `fused` | `True` when the executor took the form and no sum of terms was left as a chain |

## The normal operator

CG and the proximal-gradient methods apply $A^H A$ once per iteration, and ADMM
once per iteration of its inner solve.  For the encoding form it is applied in
one of three ways, which `plan.normal` names.

**Cartesian sampling** (`"kernel"`).  With a binary pattern $P$,
$A^H A = S^H F^H P F S$: for each coil, multiplication by $S_c$, an FFT,
multiplication by the pattern, an inverse FFT, and multiplication by
$\overline{S_c}$, summed over the coils.  With a subspace basis $\Phi$ the
k-space factor becomes a kernel over pairs of coefficients,
$K_{aa'}(k) = \sum_t \overline{\Phi_{at}}\, \Phi_{a't}\, P_t(k)$, computed
once when the operator is built rather than summed over frames at every
iteration.

**Non-Cartesian sampling** (`"kernel"`).  The normal operator of the transform
alone, $Q = \mathrm{NUFFT}^H W^H W\, \mathrm{NUFFT}$, is a convolution with the
point spread function of the weighted trajectory, evaluated exactly on a grid
doubled in each dimension ({doc}`non-cartesian`).[^fessler2005]  The full SENSE
normal operator is

$$
A^H A = \sum_c \overline{S_c}\; Q\; S_c ,
$$

which is not translation invariant: the sensitivities vary in space, so only
the transform's normal $Q$ is a convolution.  It is applied coil by coil as
multiplication by $S_c$, zero-padding to the doubled grid, an FFT,
multiplication by $\hat h$, an inverse FFT, cropping, and multiplication by
$\overline{S_c}$.  With a subspace basis, $\hat h$ becomes a kernel over pairs
of coefficients, as in the Cartesian case.

**Forward and adjoint** (`"applications"`).  bartorch builds no kernel for a
contraction over terms — the segments of an off-resonance correction, the
slices of a simultaneous multislice acquisition — for a slice phase, for a
wave pattern that varies along the readout, or when `toeplitz=False`; $A^H A$
is then applied as the forward operator followed by the adjoint.  `"transform"` names the case with no
k-space factor, where the transform's own normal operator is the whole of it.

## Encoding axes and batch axes

| Axis | Definition | Examples | Cost |
| --- | --- | --- | --- |
| Encoding axis | Indexed by the trajectory or the sampling pattern | Frames, echoes, cardiac phases | Joins the samples of one transform when the image does not vary along it; one transform per item when it does |
| Batch axis | Not indexed by the trajectory; items share the trajectory and are encoded independently | Slices, averages, repetitions | One transform plan for all items, applied to each |

Where a subspace basis contracts an encoding axis, the image carries
coefficients rather than frames, and every sample the trajectory indexes
belongs to one point set: the frames join the shots and the readout samples in
one transform.  Where the image itself varies along the axis — a dynamic
series of frames — each item has its own transform and normal kernel inside the
one operator, applied under the same coil loop, and `plan.items` reports their
number.  Under BART's own commands the non-uniform FFT substitution refuses an
image that varies along a trajectory axis, with the reason.

## References

[^pruessmann1999]: Pruessmann KP, Weiger M, Scheidegger MB, Boesiger P. SENSE: sensitivity encoding for fast MRI. *Magn Reson Med* 42(5):952–962 (1999). [doi:10.1002/(SICI)1522-2594(199911)42:5\<952::AID-MRM16\>3.0.CO;2-S](https://doi.org/10.1002/(SICI)1522-2594(199911)42:5%3C952::AID-MRM16%3E3.0.CO;2-S)

[^espirit]: Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS, Lustig M. ESPIRiT — an eigenvalue approach to autocalibrating parallel MRI: where SENSE meets GRAPPA. *Magn Reson Med* 71(3):990–1001 (2014). [doi:10.1002/mrm.24751](https://doi.org/10.1002/mrm.24751)

[^wave]: Bilgic B, Gagoski BA, Cauley SF, Fan AP, Polimeni JR, Grant PE, Wald LL, Setsompop K. Wave-CAIPI for highly accelerated 3D imaging. *Magn Reson Med* 73(6):2152–2162 (2015). [doi:10.1002/mrm.25347](https://doi.org/10.1002/mrm.25347)

[^tamir]: Tamir JI, Uecker M, Chen W, Lai P, Alley MT, Vasanawala SS, Lustig M. T2 shuffling: sharp, multicontrast, volumetric fast spin-echo imaging. *Magn Reson Med* 77(1):180–195 (2017). [doi:10.1002/mrm.26102](https://doi.org/10.1002/mrm.26102)

[^sutton]: Sutton BP, Noll DC, Fessler JA. Fast, iterative image reconstruction for MRI in the presence of field inhomogeneities. *IEEE Trans Med Imaging* 22(2):178–188 (2003). [doi:10.1109/TMI.2002.808360](https://doi.org/10.1109/TMI.2002.808360)

[^fessler2005]: Fessler JA, Lee S, Olafsson VT, Shi HR, Noll DC. Toeplitz-based iterative image reconstruction for MRI with correction for magnetic field inhomogeneity. *IEEE Trans Signal Process* 53(9):3393–3402 (2005). [doi:10.1109/TSP.2005.853152](https://doi.org/10.1109/TSP.2005.853152)
