"""
===========================
Radial SENSE reconstruction
===========================

An undersampled golden-angle radial acquisition reconstructed by regularized
least squares, with the non-Cartesian SENSE operator

.. math::

   A = W \\, \\mathrm{NUFFT} \\, S.

The sensitivities are estimated from the radial data itself, and the same
reconstruction is run twice: once through :func:`bartorch.tools.pics` and once
through the operator and a solver, the route an encoding BART has no
application for would take.

The measured data here are simulated by the same transform the reconstruction
inverts, so the experiment reports what undersampling and noise cost, not what
a model error costs.

The phantom and the coil sensitivities are built as in
:doc:`../01-basics/01-from-kspace-to-image`; the cell that does it is hidden on
this page and present in the script this page can be downloaded as.
"""

# %%

# sphinx_gallery_start_ignore
import matplotlib.pyplot as plt
from cmap import Colormap
from matplotlib.colors import ListedColormap

# Fuderer et al. (Magn Reson Med 2025) recommend one perceptually uniform
# colormap per relaxation parameter, so that a T1 map is never read as a T2 map.
LIPARI = Colormap("crameri:lipari").to_matplotlib()
NAVIA = Colormap("crameri:navia").to_matplotlib()
# Phase is cyclic, so the colormap has to be: -pi and +pi are the same colour.
# mygbm, turned so that zero phase is yellow and +/-pi is blue.
MYGBM = Colormap("colorcet:CET_C2").to_matplotlib().reversed()
PHASE = ListedColormap(MYGBM([((step + 60) % 256) / 255 for step in range(256)]))

# Colormap, window and unit per parameter.  Both relaxation windows stop short
# of cerebrospinal fluid, so that white and grey matter -- 500 against 833 ms
# in T1, 70 against 83 ms in T2 -- take up most of the scale and CSF saturates.
STYLE = {
    "T1": (LIPARI, (0.0, 1200.0), "$T_1$ [ms]"),
    "T2": (NAVIA, (0.0, 120.0), "$T_2$ [ms]"),
}

plt.rcParams.update(
    {
        "figure.dpi": 110,
        "savefig.dpi": 110,
        "font.size": 11,
        "axes.titlesize": 11,
        "figure.constrained_layout.use": True,
    }
)

PAGE_WIDTH = 8.0  # inches, the width of the documentation column


def panels(rows, columns, height=1.0):
    """A grid of square image panels filling the documentation column."""
    side = PAGE_WIDTH / columns
    figure, axes = plt.subplots(
        rows, columns, squeeze=False, figsize=(PAGE_WIDTH, rows * side * height + 0.4)
    )
    for axis in axes.ravel():
        axis.set_xticks([])
        axis.set_yticks([])
    return figure, axes


def show(axis, values, title=None, vmax=None, cmap="gray", vmin=0.0):
    """One panel, of a magnitude by default."""
    values = values.detach().abs().cpu().numpy() if hasattr(values, "detach") else values
    handle = axis.imshow(values, cmap=cmap, vmin=vmin, vmax=vmax)
    if title is not None:
        axis.set_title(title)
    return handle


def parameter(axis, values, name, title=None):
    """One relaxation map in the colormap and window its parameter is read in."""
    cmap, limits, _ = STYLE[name]
    return show(axis, values, title, vmax=limits[1], cmap=cmap, vmin=limits[0])


def domain(axis, values, title=None):
    """A complex map the way a coil sensitivity is read: phase in colour,
    magnitude in brightness, so an unsupported corner reads as background
    rather than as a phase."""
    values = values.detach().cpu()
    colours = PHASE((values.angle() / (2 * np.pi) + 0.5).numpy())[..., :3]
    magnitude = values.abs().numpy()
    magnitude = magnitude / max(float(magnitude.max()), 1e-12)
    axis.imshow(colours * magnitude[..., None])
    if title is not None:
        axis.set_title(title)


def scalebar(figure, axes, handle=None, label=None, name=None):
    """One colorbar for a group of panels, so none gives up width to its own."""
    if name is not None:
        cmap, limits, label = STYLE[name]
        handle = plt.cm.ScalarMappable(plt.Normalize(*limits), cmap)
    bar = figure.colorbar(handle, ax=axes, fraction=0.046, label=label)
    return bar


def phase_bar(figure, axes):
    """The colour-to-phase key for the panels beside it."""
    bar = figure.colorbar(
        plt.cm.ScalarMappable(plt.Normalize(-np.pi, np.pi), PHASE),
        ax=axes,
        fraction=0.046,
        ticks=[-np.pi, 0.0, np.pi],
    )
    bar.ax.set_yticklabels(["$-\\pi$", "0", "$\\pi$"])
    bar.set_label("phase [rad]")


def scaled(estimate, reference):
    """``estimate`` scaled to ``reference`` in the least-squares sense."""
    a, b = estimate.abs().double(), reference.abs().double()
    return (float((a * b).sum() / (a * a).sum()) * a).float()


# sphinx_gallery_end_ignore
import csv
import time
from pathlib import Path

import brainweb_dl
import numpy as np
import torch
from brainweb_dl import get_mri

import bartorch
import bartorch.tools as bt
from bartorch import linop, optim, priors

SIZE = 192
COILS = 8
SPOKES = 64  # against pi/2 * SIZE = 302 for a trajectory that is not undersampled

# sphinx_gallery_start_ignore
# The phantom, the relaxation maps behind it and the coil sensitivities, built
# as :doc:`/auto_examples/01-basics/01-from-kspace-to-image` builds them.
SLICE = 90  # axial, through the lateral ventricles
TISSUES = (1, 2, 3, 4, 5, 6, 8)  # everything the table gives relaxation times
MARGIN = 0.25  # what the field of view leaves around the head

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
entries = list(csv.DictReader(table.open()))
tissue_t1 = np.array([float(row["T1 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_t2 = np.array([float(row["T2 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_pd = np.array([float(row["PD (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]

# BrainWeb's volume is indexed (inferior-superior, posterior-anterior,
# left-right), so its first axis selects an axial slice; an image is drawn from
# its first row down, so flipping puts anterior at the top.
fractions = np.flipud(get_mri(sub_id=0, contrast="fuzzy")[SLICE])[..., list(TISSUES)].copy()

# A square field of view around the head, with a margin for the aliasing of an
# undersampled acquisition to fold into.
occupied = np.nonzero(fractions.sum(-1) > 0.5)
middle = [int((axis.min() + axis.max()) / 2) for axis in occupied]
half = int(round((1 + MARGIN) * max(axis.max() - axis.min() for axis in occupied) / 2))
source = tuple(
    slice(max(0, c - half), min(n, c + half)) for c, n in zip(middle, fractions.shape[:2])
)
box = np.zeros((2 * half, 2 * half, fractions.shape[-1]), dtype=np.float32)
box[tuple(slice(s.start - (c - half), s.stop - (c - half)) for s, c in zip(source, middle))] = (
    fractions[source]
)
memberships = torch.nn.functional.interpolate(
    torch.as_tensor(box).permute(2, 0, 1)[None],
    size=(SIZE, SIZE),
    mode="bilinear",
    align_corners=False,
)[0]

# Where each class sits in ``memberships``, by the name the table gives it.
CLASS = {entries[label]["Tissue"]: index for index, label in enumerate(TISSUES)}

weights = memberships * torch.as_tensor(tissue_pd)[:, None, None]
share = weights.sum(0).clamp(min=1e-6)
T1 = (weights * torch.as_tensor(tissue_t1)[:, None, None]).sum(0) / share
T2 = (weights * torch.as_tensor(tissue_t2)[:, None, None]).sum(0) / share
proton_density = weights.sum(0) / weights.sum(0).max()

# A T1-weighted spin echo, at a repetition time of 600 ms and an echo time of
# 12 ms.
signal = (
    proton_density
    * (1 - torch.exp(-600.0 / T1.clamp(min=1e-3)))
    * torch.exp(-12.0 / T2.clamp(min=1e-3))
)
signal = torch.where(T1 > 0, signal, torch.zeros(()))
signal = signal / signal.max()

# A smooth quadratic phase, so that nothing depends on the image being real.
grid_y, grid_x = torch.meshgrid(
    torch.linspace(-1.0, 1.0, SIZE), torch.linspace(-1.0, 1.0, SIZE), indexing="ij"
)
image = (signal * torch.exp(0.8j * (grid_x**2 - 0.5 * grid_y**2))).to(torch.complex64)
# sphinx_gallery_end_ignore

# sphinx_gallery_start_ignore
# BART's analytical head coil on the image grid, normalized so that the
# combination of the coil images is the image itself.
sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)
# sphinx_gallery_end_ignore

# %%
#
# Acquisition
# -----------
#
# Sixty-four golden-angle spokes across a 192 matrix, which is a fifth of the
# number radial sampling would need. The encoding operator that simulates the
# measurement is the one the reconstruction will use:
# :class:`bartorch.linop.NoncartesianSense` maps an image to the samples of
# every channel along the trajectory.

trajectory = bt.traj(readout=SIZE, spokes=SPOKES, radial=True, golden=True)

E = linop.NoncartesianSense(sensitivities, (SIZE, SIZE), traj=trajectory)
measured = bt.noise(E(image), n=1e-6, s=7)

print(f"{E.ishape} -> {E.oshape}")
print(E.plan)

# %%
#
# The operator's samples are ``(coils, shots, samples)``. BART's applications
# carry the k-space in its own layout, ``(coils, shots, samples, 1)``, whose
# trailing axis is the readout dimension a Cartesian acquisition would use, so
# an application is given ``measured[..., None]``.
#
# Sensitivity calibration
# -----------------------
#
# ESPIRiT reads its calibration matrix from a Cartesian neighbourhood, so on
# non-Cartesian data it needs the centre of k-space gridded first, and at this
# undersampling the gridded centre is already aliased.
# :func:`bartorch.tools.ncalib` instead estimates the sensitivities from the
# samples as they were measured, by nonlinear inversion at low resolution.
#
# ``N=True`` divides the estimated maps by their own root sum of squares. What
# a SENSE fit recovers is the image :math:`x` for which :math:`Sx` explains the
# data, so maps whose root sum of squares varies across the field of view leave
# its reciprocal in the image as a smooth shading. ESPIRiT normalizes its maps
# by construction; a nonlinear inversion does not, and the flag is where that
# is asked for.

maps = bt.ncalib(measured[..., None], t=trajectory, N=True)

# %%
#
# Gridding
# --------
#
# The reconstruction to beat is the density-compensated adjoint: weight each
# sample by its distance from the centre of k-space, map the samples onto the
# grid, and combine the channels by the root sum of squares. It inverts
# nothing, so the undersampling shows up in it as the streaks the point spread
# function of a radial trajectory predicts.

weights = torch.linalg.norm(trajectory.real[..., :2], dim=-1, keepdim=True)
weights = weights.clamp(min=0.25).to(torch.complex64)

channels = bartorch.nufft_adjoint(measured[..., None] * weights, trajectory, (SIZE, SIZE))
gridded = bartorch.rss(channels[:, 0], axes=(0,))

# %%
#
# Reconstruction
# --------------
#
# Total variation is the regularizer a piecewise-smooth image and a streaking
# artefact separate best under: the streaks are not piecewise constant, and the
# anatomy largely is. ADMM is the algorithm ``pics`` selects for it.

term = priors.TotalVariation(axes=(-1, -2), weight=0.001)

start = time.perf_counter()
reconstruction = bt.pics(
    measured[..., None], maps, traj=trajectory, regularizers=term, solver="admm", maxiter=50
)
print(f"pics: {time.perf_counter() - start:.2f} s")

# %%
#
# The same solve through the operator. What the application does around its
# iteration is the data scaling, which off a grid is estimated from the spread
# of the adjoint reconstruction and therefore needs the operator;
# :func:`bartorch.optim.data_scaling` takes it. The encoding is the operator
# built above, now over the estimated sensitivities rather than the true ones.

A = linop.NoncartesianSense(maps[:, 0], (SIZE, SIZE), traj=trajectory)
data = measured / optim.data_scaling(measured[..., None], A=A)

start = time.perf_counter()
assembled = optim.ADMM(term, maxiter=50)(data, A)
print(f"operator and solver: {time.perf_counter() - start:.2f} s")

print(f"identical to pics: {torch.equal(assembled.squeeze(), reconstruction.squeeze())}")

# %%
#
# The normal operator
# -------------------
#
# Each iteration applies :math:`A^H A`, which the operator computes by default
# as a convolution with a point spread function rather than as a transform each
# way. ``toeplitz=False`` asks for the transform pair instead. The two normal
# operators differ by the tolerance the transform is planned to, and fifty
# iterations carry that difference into the reconstructions.

start = time.perf_counter()
pair = optim.ADMM(term, maxiter=50)(
    data, linop.NoncartesianSense(maps[:, 0], (SIZE, SIZE), traj=trajectory, toeplitz=False)
)
print(f"without the Toeplitz normal: {time.perf_counter() - start:.2f} s")
print(f"relative difference {float((pair - assembled).abs().max() / assembled.abs().max()):.1e}")

# %%
#
# The two normal operators are the same convolution computed two ways, to the
# tolerance the transform is planned to, and fifty iterations carry that
# difference into the images. What separates them is the cost: the convolution
# is one multiplication on a doubled grid, the pair is two transforms over
# every sample of every channel.

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(2, 3, height=1.12)
peak = float(image.abs().max())
for axis, values, title in (
    (axes[0, 0], image, "phantom"),
    (axes[0, 1], scaled(gridded, image), f"gridding, {SPOKES} spokes"),
    (axes[0, 2], scaled(reconstruction, image), "total variation"),
):
    show(axis, values, title, vmax=peak)
axes[1, 0].axis("off")
for axis, values, title in (
    (axes[1, 1], scaled(gridded, image), "gridding"),
    (axes[1, 2], scaled(reconstruction, image), "total variation"),
):
    show(axis, (values - image.abs()).abs(), f"|error|, {title}", vmax=0.25 * peak)
plt.show()
# sphinx_gallery_end_ignore

# %%

for name, estimate in (("gridding", gridded), ("total variation", reconstruction)):
    print(f"{name:>16}  NRMSE {bartorch.nrmse(image.abs(), estimate.abs(), scaled=True):.3f}")

# %%
#
# The error maps are at a quarter of the image scale. Gridding leaves the
# streaks spread over the whole field of view; the regularized fit leaves its
# error at the tissue boundaries, where the piecewise-constant model the total
# variation penalty prefers is least accurate. Neither recovers the frequencies
# outside the disc the radial trajectory samples.
