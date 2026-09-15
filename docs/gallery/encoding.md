# Encoding operators

An encoding maps unknowns to acquired samples.  For Cartesian parallel imaging
it is $A = P F S$: sensitivities $S$, Fourier encoding $F$, sampling $P$.  A
reconstruction solves an inverse problem involving this model and a
regularizer.  An adjoint reverses the operations and conjugates complex
factors; it is not in general an inverse.

## Building blocks

| Component | Python | Role and convention |
| --- | --- | --- |
| Coil encoding, contraction | `linop.MultiplySum` | Multiply by a tensor and sum the axes missing from the output; conjugate in the adjoint.  Serves sensitivities and subspace contractions. |
| SENSE encoding | `linop.CartesianSense`, `linop.NoncartesianSense` | Sensitivities followed by an FFT or a NUFFT, applied `coil_batch` coils at a time.  A `basis` reads the image as a temporal subspace; `(sets, coils, *spatial)` sensitivities are summed over the sets, as ESPIRiT's second map and ENLIVE mean them. |
| Wave encoding | `linop.WaveSense` | The hybrid-space model below, as one BART operator; takes sensitivities as maps or as kernels, and a `basis` for Wave-Shuffling. |
| Cartesian FFT | `linop.FFT`; `bartorch.fft`, `bartorch.ifft` | The operator is centred and unitary; the function needs `unitary=True` for that scaling. |
| Phase, weights, sampling | `linop.Diagonal` | Complex pointwise multiplication, broadcast over singleton axes; the adjoint uses the conjugate.  A sampling mask is one of these. |
| Non-Cartesian transform | `linop.NUFFT`; `bartorch.nufft`, `tools.traj` | Trajectories in grid units, computed by FINUFFT or cuFINUFFT.  Density weights and a temporal basis belong to the operator; its Toeplitz normal should be checked against the explicit forward-adjoint pair. |
| Custom encoding | `linop.Callback`, or a `LinearOperator` subclass | Supply a forward and an adjoint, and optionally a cheaper normal.  Callbacks see views of BART's buffers and must not modify their inputs. |
| Operator algebra | `A @ B`, `A + B`, `A.to_nonlinear()` | The rightmost operator runs first; domains, codomains and devices must match. |
| Signal models | `nlop.FromTorch`, `nlop.Callback` | Nonlinear models with derivative and adjoint; evaluate the model at the current parameters before using its derivative. |
| Solvers | `optim.CG`; `optim.FISTA`, `optim.ADMM` and the others with `prox` terms; `optim.IRGNM` | BART's iterations: conjugate gradients, regularized least squares, and Gauss-Newton for parameter fitting. |
| Reconstructions | `tools.pics`, `tools.nlinv`, `tools.moba`, `tools.wave`, `tools.wshfl` | Whole BART applications, with command-specific layouts and options. |

See {doc}`/api/linop` and the
{doc}`Cartesian example </auto_examples/02_encoding/plot_01_cartesian>`.  For
weighted least squares, apply a factor $W$ to both model and data,
$\|W(Ax-y)\|^2$; for statistical weights $w$, $W=\sqrt{w}$.  Multiplying the
data alone changes the problem, and density compensation used for an
illustrative backprojection is not a noise model.

## Coil preparation

{func}`bartorch.tools.whiten` estimates a noise transform from noise-only data;
{func}`bartorch.tools.cc` and {func}`bartorch.tools.ccapply` estimate and apply
a coil compression.  Transform calibration and imaging data consistently, and
calibrate maps in the resulting coil space.  {func}`bartorch.tools.ecalib`
computes ESPIRiT maps; `caldir` and `walsh` are other calibrations.
{func}`bartorch.rss` combines magnitudes for display and discards image phase.

The [ESPIRiT paper (Uecker et al., 2014)](https://doi.org/10.1002/mrm.24751)
explains why calibration can yield several sets of maps.  Keeping one is a
modelling choice; phase gauges and coil-space normalization matter when
comparing maps.  See the
{doc}`coil preparation example </auto_examples/01_tools/plot_02_coil_preparation>`.

## Encodings written as compositions

An encoding with an element-wise factor on either side of it, summed over
terms, is one operator rather than a sum of them.  Write the terms with `@`
and `+`; nothing is built until something needs the operator, and what is
built is one encoding whose contraction is those terms.  `A.plan` says which
happened -- `contraction=segments(n)` where the terms were folded in,
`chained(n)` where the sum stands.

```python
A = None
for term in range(len(b)):
    built = linop.Diagonal(b[term], E.oshape) @ E @ linop.Diagonal(c[term], E.ishape)
    A = built if A is None else A + built
```

| Model | `c_l` (image side) | `b_l` (k-space side) |
| --- | --- | --- |
| Off-resonance by time segmentation | spatial weights of the fit | the segment's sample weights |
| Multishot with a known phase | the shot's phase | the samples that shot took |
| Echo phase with a subspace basis | the frame's phase | the frame, picked out of the samples |

{func}`bartorch.linop.FieldCorrected` is the first of these with the fit done
for you; the others are the composition and nothing more.  Each costs one
transform per term inside the coil loop, which is what an image-side factor
that varies along the frames costs.

Simultaneous multislice is the same shape with the slices as sets of maps:
`c_l` picks slice `l` whole and `b_l` is its phase in k-space.  The slices are
summed on the far side of the transform, so it runs once per slice, and
`plan.contraction` is `slices`.

Picking a slice whole is the only image-side factor that may differ between
sets.  The sensitivities contract the sets before the image factor is reached,
so any other weight along them leaves the sum standing and the plan says
`chained`.

## Beyond Cartesian and radial encoding

These are mathematical decompositions for planning applications, not further
Python classes.  The literature and implementation evidence are in
{doc}`research`.  Start with the runnable
{doc}`known-phase shot model </auto_examples/02_encoding/plot_03_epi_shots>` and
{doc}`synthetic wave and subspace model </auto_examples/02_encoding/plot_04_wave_subspace>`.

**EPI.**  A simplified multi-shot model is $y_s=P_s F S D_s x$, with shot phase
$D_s$.  Known phases are diagonal operators with per-shot sampling.  Real EPI
also needs readout polarity handling, Nyquist-ghost correction and possibly
off-resonance encoding $\exp(-i2\pi\Delta f(r)t_j)$ at each sample, which a
static image phase map cannot replace.  A paper using BART for ESPIRiT alone is
no evidence that its EPI correction or solver is available here.

**Wave encoding.**  A hybrid-space model is $A=P F_{yz} W F_x R S$, with
readout padding $R$ and the wave modulation $W(k_x,y,z)$, whose phase comes
from the gradient waveforms, spatial coordinates and timing.  BART's
`src/wave.c` applies coil encoding, readout resizing, readout FFT, diagonal
wave modulation, transverse FFT and sampling in that order.  `tools.wave`
wraps it; `tools.wavepsf` generates a 2-D hybrid-space response.  Its options
mix cm, microseconds, seconds, G/cm and G/cm/s, so check the reference before
converting scanner units.  A full 3-D response and measured-gradient
calibration need further preparation.

**Shuffling and Wave-Shuffling.**  Model an echo series as
$x_t=\sum_k\Phi_{tk}\alpha_k$; the encoding acts on these echo-dependent images
and samples the acquired echo and phase-encode ordering.  Wave-Shuffling adds
the wave modulation and extended readout field of view: a time-resolved inverse
problem, not a wave FFT applied to a static reconstruction.
{func}`bartorch.tools.wshfl` takes maps, wave response, temporal basis,
reordering and acquired-data table.  There is no dedicated wave or EPI operator
class.

For each new model, check units, axis order, the complex adjoint identity,
independent forward predictions and reconstruction residuals before moving to
measured data.
