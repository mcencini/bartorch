# Research recipes: EPI, wave encoding, shuffling

These are literature-backed implementation plans, not executed reproductions.
Sources were reviewed on 8 September 2026. A paper may use BART for a single
calibration step; that is distinguished below from implementing its reconstruction
in BART. The Python entry points were checked against the local wrapper and C
sources. No measured datasets are downloaded by the documentation build.

## EPI: phase and shot structure

[Hu et al., Motion-robust reconstruction of multishot diffusion-weighted images without phase estimation through locally low-rank regularization (2019)](https://doi.org/10.1002/mrm.27488) formulates a local low-rank constraint across
shot images. This is a useful route for a shot-resolved reconstruction example;
a generic SENSE solve with a known phase map does not reproduce that algorithm.
See the [open manuscript](https://pmc.ncbi.nlm.nih.gov/articles/PMC6289606/).

[Improving robustness of 3D multi-shot EPI by structured low-rank reconstruction of segmented CAIPI sampling for fMRI at 7T](https://pmc.ncbi.nlm.nih.gov/articles/PMC10933751/) explicitly reports BART
ESPIRiT calibration, MATLAB reconstruction and reference-scan Nyquist-ghost
correction. This is evidence for BART's role in calibration, not for a ready-made
BART implementation of the whole method.

A bartorch reproduction should start with a synthetic known-phase multi-shot
model, then add measured data with documented polarity, timing, shot assignment,
phase correction and coil maps. Compare corrected and uncorrected reconstructions,
report residuals by shot and measure remaining ghosts. Implement and validate the
paper's structured or locally low-rank prior separately. `pics` has regularizer
options, but their dimensions and splitting must match the selected formulation.

[Liao et al., Highly Accelerated EPI with Wave Encoding and Multi-shot Simultaneous Multi-Slice Imaging](https://arxiv.org/abs/2106.01918) is a useful
extension: it combines wave encoding with EPI and SMS. It is a method reference
here, not evidence that `tools.wave` handles EPI readout polarity or SMS
shot-phase corrections without additional code.

## Wave-CAIPI and Wave-CS

[Bilgic et al., Wave-CAIPI for highly accelerated 3D imaging (2015)](https://doi.org/10.1002/mrm.25347) supplies the wave-encoding model.
The [authors' Wave-CAIPI software page](https://www.martinos.org/~berkin/wave_caipi.html) states that BART/ESPIRiT is
used for sensitivity estimation. The original method reference alone should
not be described as a complete BART reconstruction implementation.

There is also a direct BART teaching route: the
[ISMRM 2016 Wave-CS workshop](https://gitlab.tugraz.at/ibi/mrirecon/tutorials/bart-workshop/tree/master/ismrm2016/wave-cs)
contains Wave-CAIPI and retrospective Wave-CS material, as indexed by the
[BART tutorial catalogue](https://mrirecon.codeberg.page/tutorials.html).
This is an exception to the predominance of conventional Cartesian and
non-Cartesian examples in the teaching material.

The reproduction path is: obtain the workshop's documented data and license;
load maps, sampling and wave response; preserve the extended readout FOV;
translate file order into tensor order; run `tools.wave` with matching options;
then compare a Python-composed forward/adjoint pair against an independent tiny
hybrid-space calculation. Compare Cartesian and wave sampling at equal acquired
sample counts. Report calibration error and FOV padding alongside image errors;
a synthetic phase diagonal alone is not a calibrated Wave-CAIPI acquisition.

## Wave-Shuffling

[Iyer et al., Wave-encoding and Shuffling Enables Rapid Time Resolved Structural Imaging](https://arxiv.org/abs/2103.15881) (2021 preprint, revised 2022) combines
wave encoding with temporal subspace reconstruction for FSE and MPRAGE. The
[manuscript's implementation and data statement](https://arxiv.org/html/2103.15881v1) explicitly identifies BART as the
reconstruction implementation and links the
[authors' reproduction code](https://github.com/sidward/wave-shuffling) and
[archived materials](https://doi.org/10.5281/zenodo.4603207).

The method builds on [Tamir et al., T2 Shuffling: Sharp, multicontrast, volumetric fast spin-echo imaging](https://doi.org/10.1002/mrm.26102) and its
[demonstration code](https://jtamir.github.io/t2shuffling-support/).

The current `tools.wshfl(maps, wave, phi, reorder, table, ...)` wrapper supplies
an entry point. From the pinned `external/bart/src/wshfl.c` help, the principal layouts
are below (other BART dimensions are singleton):

| Array | BART dimension order | Reversed Python order |
| --- | --- | --- |
| Sensitivity maps | `(sx, sy, sz, nc, md)` | `(md, nc, sz, sy, sx)` |
| Wave response | `(wx, sy, sz)` | `(sz, sy, wx)` |
| Temporal basis | `(1, 1, 1, 1, 1, tf, tk)` | `(tk, tf, 1, 1, 1, 1, 1)` |
| Reordering | `(n, 3)` | `(3, n)` |
| Acquired table | `(wx, nc, n)` | `(n, nc, wx)` |

Here `wx` is the extended readout length, `nc` coil count, `md` map count,
`tf` echo-train length, `tk` basis rank, and `n` acquired readout count.
Verify these layouts and the coordinate indexing against the pinned source and
reproduction data before running; the reordering array is not a generic mask.

Port the authors' preprocessing, basis construction and sampling order before
changing regularization. Reproduce one published figure, then assess coefficient
and synthesized-echo residuals, basis truncation, peak memory and runtime. Record
the BART revision, data archive version, CPU/GPU configuration and solver options.
This reproduction remains future gallery work; the current gallery does not
claim to reproduce the paper's acceleration or image quality.

## Model-based and learning extensions

The executable gallery already demonstrates a small analytic signal model and
Gauss-Newton fit. The March 2021 BART webinar provides a route to subspace and
quantitative mapping data (see {doc}`learning_resources`).

[Blumenthal et al., Deep, Deep Learning with BART](https://arxiv.org/abs/2202.14005) describes BART's learning machinery. It should
not be conflated with the separate PyTorch/DeepInverse integration here.
[Cho et al., Wave-Encoded Model-based Deep Learning for Highly Accelerated Imaging with Joint Reconstruction](https://arxiv.org/abs/2202.02814) is a
further method reference for a future wave/learning application, after fixed
wave encoding and its derivatives have independent validation.
