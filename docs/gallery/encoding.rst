Tour of encoding operators and utilities
========================================

An encoding model maps unknowns to acquired samples. For Cartesian parallel
imaging, write :math:`A = P F S`: sensitivities :math:`S`, Fourier encoding
:math:`F`, sampling :math:`P`. Reconstruction solves an inverse problem involving
this model and a regularizer. An adjoint reverses the operations and conjugates
complex factors; it is not generally an inverse.

Available building blocks
-------------------------

.. list-table::
   :header-rows: 1
   :widths: 22 35 43

   * - Component
     - Current Python surface
     - Role and important convention
   * - Coil encoding / contraction
     - ``linop.MultiplySum``
     - Multiply by a tensor, sum axes missing from the output; conjugate in
       the adjoint. Supports sensitivity and subspace contractions.
   * - Cartesian FFT
     - ``linop.FFT``; ``tools.fft/ifft``
     - Operator is centered/unitary; tool needs ``unitary=True`` for that scale.
   * - Sampling
     - ``linop.Sampling``
     - Mask on a full grid, with broadcast-compatible singleton axes.
   * - Phase / weights
     - ``linop.Diagonal``
     - Complex pointwise multiplication; the adjoint uses the conjugate.
   * - Non-Cartesian transform
     - ``linop.NUFFT``; ``tools.traj/nufft``
     - Grid-unit trajectories. FINUFFT/cuFINUFFT backend; weights and temporal
       basis can be carried by the NUFFT. Toeplitz normals are approximations
       whose agreement with an explicit forward/adjoint pair should be checked.
   * - Custom encoding
     - ``linop.Callback``
     - Supply forward and adjoint; optionally a cheaper normal. Callbacks see
       working-buffer views and should not modify their inputs.
   * - Operator algebra
     - ``A @ B``, ``A + B``, ``A.to_nonlinear()``
     - Rightmost operator runs first; match domain/codomain shapes and devices.
   * - Signal models
     - ``nlop.FromTorch``, ``nlop.Callback``
     - Nonlinear model with derivative/adjoint callbacks; evaluate the forward
       model at the current parameters before using its local derivative.
   * - Solvers
     - ``linop.LinearOperator.lstsq``; ``nlop.NonlinearOperator.irgnm``
     - Conjugate gradients for quadratic least squares; regularized Gauss-Newton
       for parameter fitting. Broader proximal solvers are not yet exposed here.
   * - Prepackaged reconstructions
     - ``tools.pics/nlinv/moba/wave/wshfl``
     - Whole BART applications, with command-specific layouts and options.
       A generated wrapper is not a dedicated composable encoding class.

See :doc:`/api/generated/operators` and the executable
:doc:`Cartesian example </auto_examples/02_encoding/plot_01_cartesian>`.
For weighted least squares, apply a factor :math:`W` to both model and data:
:math:`\|W(Ax-y)\|^2`. If the objective uses statistical weights :math:`w`, then
:math:`W=\sqrt{w}`. Multiplying data alone changes the problem. Density compensation
used for an illustrative backprojection is not automatically a noise model.

Coil preparation
----------------

:func:`bartorch.tools.whiten` estimates a noise transform from noise-only data;
:func:`bartorch.tools.cc` and :func:`bartorch.tools.ccapply` estimate and apply a
coil compression basis. Transform calibration and imaging data consistently,
and calibrate maps in the resulting coil space. :func:`bartorch.tools.ecalib`
provides ESPIRiT maps; ``caldir`` and ``walsh`` offer other calibration routes.
``rss`` combines magnitudes for display but does not preserve image phase.

The `ESPIRiT paper (Uecker et al., 2014)
<https://doi.org/10.1002/mrm.24751>`_ explains why calibration can yield multiple
map sets. Keeping one map is a modeling choice; phase gauges and coil-space
normalization matter when comparing maps. See the
:doc:`coil preparation example </auto_examples/01_tools/plot_02_coil_preparation>`.

Beyond standard Cartesian and radial encoding
---------------------------------------------

The following are mathematical decompositions for planning apps, not additional
Python classes. The literature and implementation evidence are in
:doc:`research`. Start with the runnable
:doc:`known-phase shot model </auto_examples/02_encoding/plot_03_epi_shots>` and
:doc:`synthetic wave/subspace model </auto_examples/02_encoding/plot_04_wave_subspace>`.

**EPI.** A simplified multi-shot model is
:math:`y_s=P_s F S D_s x`, with shot phase :math:`D_s`. Known phases can be
represented by diagonal operators and per-shot sampling. Real EPI also requires
readout polarity handling, Nyquist-ghost correction and possibly off-resonance
encoding :math:`\exp(-i2\pi\Delta f(r)t_j)` at each sample. A static image phase
map cannot replace that time-dependent term. BART use for ESPIRiT alone is not
evidence that a paper's EPI correction or solver is packaged in bartorch.

**Wave encoding.** A hybrid-space model is
:math:`A=P F_{yz} W F_x R S`, with readout padding :math:`R` and the wave
modulation :math:`W(k_x,y,z)`. Its phase comes from the applied gradient
waveforms, spatial coordinates and timing. The local ``bart/src/wave.c`` uses
this sequence of coil encoding, readout resizing, readout FFT, diagonal wave
modulation, transverse FFT and sampling. ``tools.wave`` is available as a
command wrapper; ``wavepsf`` generates a 2-D hybrid-space response. Its options
mix cm, microseconds, seconds, Gauss/cm and Gauss/cm/s: inspect the API before
converting scanner units. A full 3-D response and measured-gradient calibration
require additional preparation.

**Shuffling and Wave-Shuffling.** Model an echo series as
:math:`x_t=\sum_k\Phi_{tk}\alpha_k`. Encoding acts on these echo-dependent images
and samples the acquired echo/phase-encode ordering. Wave-Shuffling adds the
wave modulation and extended readout FOV; the result is a time-resolved inverse
problem, not a wave FFT applied to a static reconstruction. The
:func:`bartorch.tools.wshfl` wrapper accepts maps, wave response, temporal basis,
reordering and acquired-data table. No dedicated ``linop.Wave`` or
``linop.EPI`` class exists in this checkout.

For each new model, check units, axis order, complex adjoint identity, independent
forward predictions, and reconstruction residuals before moving to measured data.
