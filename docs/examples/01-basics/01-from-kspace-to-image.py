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
# Phantom
# -------
#
# BrainWeb publishes a segmentation rather than an image: one membership map
# per tissue class, which a table of relaxation times and proton densities
# turns into whatever contrast the experiment would have produced. The volume
# ``brainweb-dl`` returns is indexed ``(inferior-superior, posterior-anterior,
# left-right)``, so its first axis selects an axial slice, and an image is
# drawn from its first row down, so flipping it puts anterior at the top.

SIZE = 192
COILS = 8
SLICE = 90  # axial, through the lateral ventricles
TISSUES = (1, 2, 3, 4, 5, 6, 8)  # everything the table gives relaxation times

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
entries = list(csv.DictReader(table.open()))
tissue_t1 = np.array([float(row["T1 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_t2 = np.array([float(row["T2 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_pd = np.array([float(row["PD (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]

fractions = np.flipud(get_mri(sub_id=0, contrast="fuzzy")[SLICE])[..., list(TISSUES)].copy()

# %%
#
# The slice is cropped to a square field of view around the head and resampled
# to the matrix reconstructed here. The crop leaves a margin, as a real field
# of view does: a head that filled it would have nowhere for the aliasing of an
# undersampled acquisition to fold into.

MARGIN = 0.25

# sphinx_gallery_start_ignore
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
# sphinx_gallery_end_ignore

# %%
#
# A membership-weighted average of the table gives :math:`T_1`, :math:`T_2` and
# the proton density at every voxel, and the spin-echo signal
#
# .. math::
#
#    S = \rho \, \left(1 - e^{-T_R/T_1}\right) e^{-T_E/T_2}
#
# turns those into the image the experiment measures. At a short repetition
# time and a short echo time the contrast is :math:`T_1`-weighted: white matter
# bright, cerebrospinal fluid dark, subcutaneous fat brightest of all.

TR, TE = 600.0, 12.0  # ms

weights = memberships * torch.as_tensor(tissue_pd)[:, None, None]
share = weights.sum(0).clamp(min=1e-6)
T1 = (weights * torch.as_tensor(tissue_t1)[:, None, None]).sum(0) / share
T2 = (weights * torch.as_tensor(tissue_t2)[:, None, None]).sum(0) / share
proton_density = weights.sum(0) / weights.sum(0).max()

signal = (
    proton_density * (1 - torch.exp(-TR / T1.clamp(min=1e-3))) * torch.exp(-TE / T2.clamp(min=1e-3))
)
signal = torch.where(T1 > 0, signal, torch.zeros(()))
signal = signal / signal.max()

# A smooth quadratic phase stands in for the transmit and off-resonance phase
# of a real object, so that nothing below depends on the image being real.
grid_y, grid_x = torch.meshgrid(
    torch.linspace(-1.0, 1.0, SIZE), torch.linspace(-1.0, 1.0, SIZE), indexing="ij"
)
image = (signal * torch.exp(0.8j * (grid_x**2 - 0.5 * grid_y**2))).to(torch.complex64)

# %%
#
# Coils
# -----
#
# The sensitivities are BART's analytical head coil, evaluated on the image
# grid that :func:`bartorch.tools.grid` describes. Dividing by the root sum of
# squares over the channels makes the combination of the coil images the image
# itself, so a reconstruction can be compared against it directly.

sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)

coil_images = sensitivities * image
kspace = bartorch.fft(coil_images, axes=(-2, -1), unitary=True)
kspace = bt.noise(kspace, n=1e-5, s=42)

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(1, 3)
peak = float(image.abs().max())
show(axes[0, 0], image, "$T_1$-weighted phantom", vmax=peak)
parameter(axes[0, 1], T1, "T1", "$T_1$")
scalebar(figure, axes[0, 1], name="T1")
parameter(axes[0, 2], T2, "T2", "$T_2$")
scalebar(figure, axes[0, 2], name="T2")

figure, axes = panels(1, 4)
for column in range(4):
    domain(axes[0, column], sensitivities[column], f"channel {column}")
    axes[0, column].set_xticks([])
    axes[0, column].set_yticks([])
phase_bar(figure, axes[0, 3])
figure.suptitle("coil sensitivities: colour is phase, brightness is magnitude")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The relaxation maps are drawn with the perceptually uniform colormaps the
# relaxometry community settled on -- lipari for :math:`T_1`, navia for
# :math:`T_2` -- so that the two are never read as each other, and with a
# window that stops short of cerebrospinal fluid, which is far enough from the
# rest to take the whole scale. The sensitivities are complex, and are drawn
# the way a sensitivity is read: a cyclic colormap for the phase, brightness
# for the magnitude.
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
profile = (1.0 + 2.0 * encodes.abs() / SIZE) ** -3.0
centre = (encodes.abs() < CALIBRATION // 2).to(torch.float32)
drawn = torch.multinomial(
    profile * (1.0 - centre),
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
# calibration, the encoding operator, every iteration -- then costs six
# channels rather than eight.

VIRTUAL = 6

matrix = bt.cc(measured, p=VIRTUAL, M=True, r=CALIBRATION)
compressed = bt.ccapply(measured, matrix, p=VIRTUAL)

# %%
#
# Sensitivity calibration
# -----------------------
#
# ESPIRiT estimates the sensitivities as the leading eigenvector, per voxel, of
# an operator built from the calibration region. ``crop`` discards the voxels
# whose eigenvalue falls below it, and so keeps the maps from being
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

sense = bt.pics(compressed, maps, l2=0.001, maxiter=60)
wavelet = bt.pics(
    compressed,
    maps,
    regularizers=priors.Wavelet((-1, -2), 0.002),
    solver="fista",
    maxiter=100,
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
    similarity = bartorch.ssim(image.abs(), scaled(estimate, image))
    print(f"{name:>20}  NRMSE {error:.3f}  SSIM {similarity:.3f}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(2, 4, height=1.1)
for axis, values, title in (
    (axes[0, 0], image, "phantom"),
    (axes[0, 1], scaled(gridded, image), "root sum of squares"),
    (axes[0, 2], scaled(sense, image), "SENSE"),
    (axes[0, 3], scaled(wavelet, image), "wavelet"),
):
    show(axis, values, title, vmax=peak)
axes[1, 0].axis("off")
for axis, values in (
    (axes[1, 1], scaled(gridded, image)),
    (axes[1, 2], scaled(sense, image)),
    (axes[1, 3], scaled(wavelet, image)),
):
    show(axis, (values - image.abs()).abs(), vmax=0.2 * peak)
axes[1, 1].set_ylabel("|error|, x5")
figure.suptitle(f"{ACCELERATION}x undersampled, {VIRTUAL} virtual channels")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The root sum of squares carries the aliasing the missing phase encodes
# produce, since it inverts nothing. The Tikhonov fit inverts the sampling
# operator but has no reason to prefer one image among those that fit the data
# equally well, and a variable-density random pattern leaves many: what it
# leaves behind is the incoherent residue of that choice. The wavelet penalty
# is that reason, and removes it.
#
# How much it removes depends on its weight, which is chosen here and not
# estimated: a larger one removes more noise and more texture with it.
#
# The same reconstruction written as an encoding operator and a solver, rather
# than as a call to a BART application, is the subject of
# :doc:`02-operators-and-solvers`.
