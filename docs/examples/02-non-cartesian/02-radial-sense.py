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


def show(axis, values, title=None, vmax=None, cmap="gray"):
    values = values.detach().abs().cpu().numpy() if hasattr(values, "detach") else values
    handle = axis.imshow(values, cmap=cmap, vmin=0.0, vmax=vmax)
    if title is not None:
        axis.set_title(title)
    return handle


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
SLICE = 90

# sphinx_gallery_start_ignore
table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
proton_density = np.array(
    [float(row["PD (ms)"]) for row in csv.DictReader(table.open())], dtype=np.float32
)
fractions = get_mri(sub_id=0, contrast="fuzzy")[:, :, SLICE]
fractions = np.flip(fractions.transpose(1, 0, 2), 0).copy()
magnitude = torch.nn.functional.interpolate(
    torch.as_tensor(fractions @ proton_density)[None, None],
    size=(SIZE, SIZE),
    mode="bilinear",
    align_corners=False,
)[0, 0]
magnitude = magnitude / magnitude.max()
grid_y, grid_x = torch.meshgrid(
    torch.linspace(-1.0, 1.0, SIZE), torch.linspace(-1.0, 1.0, SIZE), indexing="ij"
)
image = (magnitude * torch.exp(0.8j * (grid_x**2 - 0.5 * grid_y**2))).to(torch.complex64)

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

maps = bt.ncalib(measured[..., None], t=trajectory)

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

term = priors.TotalVariation(axes=(-1, -2), weight=0.005)

start = time.perf_counter()
reconstruction = bt.pics(
    measured[..., None], maps, traj=trajectory, regularizers=term, solver="admm", maxiter=40
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
assembled = optim.ADMM(term, maxiter=40)(data, A)
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
# operators differ by the tolerance the transform is planned to, and forty
# iterations carry that difference into the reconstructions.

start = time.perf_counter()
pair = optim.ADMM(term, maxiter=40)(
    data, linop.NoncartesianSense(maps[:, 0], (SIZE, SIZE), traj=trajectory, toeplitz=False)
)
print(f"without the Toeplitz normal: {time.perf_counter() - start:.2f} s")
print(f"relative difference {float((pair - assembled).abs().max() / assembled.abs().max()):.1e}")

# %%
#
# The two normal operators are the same convolution computed two ways, to the
# tolerance the transform is planned to, and forty iterations carry that
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
