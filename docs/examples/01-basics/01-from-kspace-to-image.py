"""
=====================
From k-space to image
=====================

Reconstruction of an undersampled Cartesian acquisition, from the measured
k-space to a coil-combined image.

The acquisition is simulated from a BrainWeb tissue segmentation and the eight
channels of BART's head coil model, sampled at a third of the Nyquist rate
along the phase-encode direction. The reconstruction that follows is the one a
scanner pipeline performs: channel compression, sensitivity calibration by
ESPIRiT, and a regularized least-squares fit of the SENSE model

.. math::

   y = P F S x + \\varepsilon,

with :math:`S` the coil sensitivities, :math:`F` the Fourier transform and
:math:`P` the sampling operator. :doc:`../../explanation/encoding` states the
model and :doc:`../../explanation/inverse-problems` the estimator.

Shapes here are C order, so a Cartesian k-space is ``(coils, z, y, x)`` and an
axis argument indexes that shape; see :doc:`../../guides/user/conventions`.
"""

# %%
# The phantom is BrainWeb subject 0, reached through ``brainweb-dl``:
# ``get_mri`` returns fuzzy tissue memberships rather than labels, and the
# package ships the table of relaxation times and proton densities that goes
# with them.

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
from pathlib import Path

import brainweb_dl
import numpy as np
import torch
from brainweb_dl import get_mri

import bartorch
import bartorch.tools as bt
from bartorch import priors

# %%
#
# Phantom and coils
# -----------------
#
# One axial slice at a 192 matrix. The proton densities of the tissue classes,
# weighted by their memberships, give an image whose partially occupied voxels
# lie between the pure ones. A smooth quadratic phase stands in for the
# transmit and off-resonance phase of a real object, so that nothing below
# depends on the image being real.

SIZE = 192
COILS = 8
SLICE = 90  # axial, through the lateral ventricles

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
proton_density = np.array(
    [float(row["PD (ms)"]) for row in csv.DictReader(table.open())], dtype=np.float32
)

fractions = get_mri(sub_id=0, contrast="fuzzy")[:, :, SLICE]
# The volume is (left-right, posterior-anterior, inferior-superior); the
# transpose puts the anterior-posterior axis along the rows of the figures.
fractions = np.flip(fractions.transpose(1, 0, 2), 0).copy()

magnitude = torch.nn.functional.interpolate(
    torch.as_tensor(fractions @ proton_density)[None, None],
    size=(SIZE, SIZE),
    mode="bilinear",
    align_corners=False,
)[0, 0]
magnitude = magnitude / magnitude.max()

y, x = torch.meshgrid(
    torch.linspace(-1.0, 1.0, SIZE), torch.linspace(-1.0, 1.0, SIZE), indexing="ij"
)
phase = 0.8 * (x**2 - 0.5 * y**2)
image = (magnitude * torch.exp(1j * phase)).to(torch.complex64)

# %%
#
# The sensitivities are BART's analytical head coil, evaluated on the image
# grid that :func:`bartorch.tools.grid` describes. Dividing by the root sum of
# squares over the channels makes the combination of the coil images the image
# itself, which is what lets a reconstruction be compared against it.

grid = bt.grid(D=(SIZE, SIZE, 1))
sensitivities = bt.coils(t=grid, n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)

coil_images = sensitivities * image
kspace = bartorch.fft(coil_images, axes=(-2, -1), unitary=True)
kspace = bt.noise(kspace, n=2e-5, s=42)

# %%
#
# Sampling
# --------
#
# The readout is fully sampled and the phase encodes are drawn at random from a
# variable density, with a 24-line calibration region at the centre kept in
# full. ESPIRiT reads its calibration matrix from that region, so an
# acquisition that omitted it would need a separate calibration scan.

ACCELERATION = 3
CALIBRATION = 24

encodes = torch.arange(SIZE) - SIZE // 2
density = (1.0 + 2.0 * encodes.abs() / SIZE) ** -3.0
centre = (encodes.abs() < CALIBRATION // 2).to(torch.float32)
drawn = torch.multinomial(
    density * (1.0 - centre),
    SIZE // ACCELERATION - CALIBRATION,
    replacement=False,
    generator=torch.Generator().manual_seed(11),
)
lines = centre.clone()
lines[drawn] = 1.0

# A pattern broadcasts over one channel's samples, so a column of it
# undersamples the phase-encode axis for every channel.
pattern = lines.reshape(SIZE, 1).to(torch.complex64)
measured = kspace[:, None] * pattern

print(f"{float(lines.mean()):.0%} of the phase encodes acquired")

# %%
#
# Channel compression
# -------------------
#
# Eight channels carry less independent information than eight images: the
# sensitivities overlap, and the singular value spectrum of the calibration
# matrix falls off. :func:`bartorch.tools.cc` returns the matrix that projects
# the channels onto their leading singular vectors, and
# :func:`bartorch.tools.ccapply` applies it. Everything downstream --
# calibration, the encoding operator, every iteration -- then costs four
# channels rather than eight.

VIRTUAL = 4

matrix = bt.cc(measured, p=VIRTUAL, M=True, r=CALIBRATION)
compressed = bt.ccapply(measured, matrix, p=VIRTUAL)

# %%
#
# Sensitivity calibration
# -----------------------
#
# ESPIRiT estimates the sensitivities as the leading eigenvector, per voxel, of
# an operator built from the calibration region. ``crop`` discards the voxels
# whose eigenvalue falls below it, which is what keeps the maps from being
# extrapolated into the background.

maps = bt.ecalib(compressed, maps=1, calib_size=CALIBRATION, crop=0.8)

# %%
#
# Reconstruction
# --------------
#
# :func:`bartorch.tools.pics` solves the regularized least-squares problem. A
# Tikhonov weight alone gives the conjugate-gradient SENSE reconstruction; an
# :math:`\ell_1` penalty on the wavelet coefficients is the compressed-sensing
# reconstruction of the same data, solved by FISTA. Both are compared against
# the root sum of squares of the zero-filled channel images, which inverts
# nothing.

channel_images = bartorch.ifft(compressed[:, 0], axes=(-2, -1), unitary=True)
gridded = bartorch.rss(channel_images, axes=(0,))

sense = bt.pics(compressed, maps, l2=0.01, maxiter=30)
wavelet = bt.pics(
    compressed,
    maps,
    regularizers=priors.Wavelet((-1, -2), 0.002),
    solver="fista",
    maxiter=60,
)

# %%
#
# The sensitivities ESPIRiT estimates and the ones the acquisition was
# simulated with differ by a phase that varies from voxel to voxel, so the
# reconstructed image does too, and the comparison is between magnitudes.
# :func:`bartorch.nrmse` with ``scaled=True`` divides out the one degree of
# freedom a SENSE reconstruction leaves undetermined, the global scale.

for name, estimate in (
    ("root sum of squares", gridded),
    ("SENSE", sense),
    ("wavelet", wavelet),
):
    error = bartorch.nrmse(image.abs(), estimate.abs(), scaled=True)
    print(f"{name:>20}  NRMSE {error:.3f}  SSIM {bartorch.ssim(image.abs(), estimate.abs()):.3f}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(1, 4)
peak = float(image.abs().max())
show(axes[0, 0], image, "phantom", vmax=peak)
show(axes[0, 1], scaled(gridded, image), "root sum of squares", vmax=peak)
show(axes[0, 2], scaled(sense, image), "SENSE", vmax=peak)
show(axes[0, 3], scaled(wavelet, image), "wavelet", vmax=peak)
figure.suptitle(f"{ACCELERATION}x undersampled, {VIRTUAL} virtual channels")

figure, axes = panels(1, 4)
for axis, channel in zip(axes.ravel(), range(4)):
    show(axis, sensitivities[channel], f"channel {channel}", vmax=1.0)
figure.suptitle("coil sensitivities, magnitude")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The root sum of squares carries the aliasing the missing phase encodes
# produce, since it inverts nothing. Both fits invert the sampling operator
# and remove it, and the wavelet penalty reaches the lower error of the two on
# this phantom. How much lower depends on its weight, which is chosen here and
# not estimated: a larger one removes more noise and more texture with it.
#
# The same reconstruction written as an encoding operator and a solver, rather
# than as a call to a BART application, is the subject of
# :doc:`02-operators-and-solvers`.
