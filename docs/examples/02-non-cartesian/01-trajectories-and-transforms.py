"""
=============================
Trajectories and transforms
=============================

The non-Cartesian interfaces: the trajectories
:func:`bartorch.tools.traj` generates, the non-uniform Fourier transform along
one, the density compensation an adjoint reconstruction needs, and the point
spread function the normal operator convolves with.

Every non-Cartesian transform in bartorch is computed by FINUFFT [#finufft]_,
which evaluates

.. math::

   y_j = \\frac{1}{\\sqrt{N}} \\sum_{m} x_m \\,
   \\exp\\!\\Big(-2\\pi i \\sum_d \\frac{k_{j,d}\\, m_d}{n_d}\\Big)

to a requested tolerance, with the sum over the :math:`N` voxels :math:`m`
of an image of :math:`n_d` voxels along dimension :math:`d`, and
:math:`k_j` in grid units.  The spreading kernel and the deapodization are
FINUFFT's, sized from the tolerance: :doc:`../../explanation/non-cartesian`
states the conventions and the accuracy.

The phantom is built as in :doc:`../01-basics/01-from-kspace-to-image`; the
cell that does it is hidden on this page and present in the script this page can
be downloaded as.
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
from bartorch import linop

SIZE = 128

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

# %%
#
# Trajectories
# ------------
#
# A trajectory is ``(*encoding, shots, samples, 3)`` in grid units: the
# coordinates of every sample, in units of the k-space cell of the image it
# encodes, so a readout of ``SIZE`` samples runs from :math:`-n/2` to
# :math:`n/2` for an image of :math:`n` voxels along the readout. The third
# component is :math:`k_z`, zero throughout for a two-dimensional trajectory,
# and whether it is used determines whether the transform is two- or
# three-dimensional.
#
# Successive spokes are separated either by :math:`\pi` over their number,
# which tiles k-space uniformly for one frame, or by the golden angle, which
# tiles it approximately uniformly for *any* number of consecutive spokes
# [#winkelmann]_. Only
# the second lets an acquisition be cut into frames after it was measured, as
# :doc:`../03-applications/01-dynamic-golden-angle` does.

SPOKES = 201  # pi/2 * SIZE, the number at which radial sampling is not undersampled

uniform = bt.traj(readout=SIZE, spokes=SPOKES, radial=True)
golden = bt.traj(readout=SIZE, spokes=SPOKES, radial=True, golden=True)

print(f"{tuple(golden.shape)}: {SPOKES} shots of {SIZE} samples")

# %%

# sphinx_gallery_start_ignore
figure, axes = plt.subplots(1, 2, figsize=(PAGE_WIDTH, PAGE_WIDTH / 2 + 0.4))
for axis, arms, title in (
    (axes[0], uniform, "uniform"),
    (axes[1], golden, "golden angle"),
):
    for spoke in range(0, 24):
        line = arms[spoke].real
        axis.plot(line[:, 0], line[:, 1], lw=0.5, color="0.2")
    axis.set_aspect("equal")
    axis.set_title(f"{title}, first 24 spokes")
    axis.set_xlabel("$k_x$ [grid units]")
axes[0].set_ylabel("$k_y$ [grid units]")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The transform
# -------------
#
# :func:`bartorch.nufft` samples an image along a trajectory and
# :func:`bartorch.nufft_adjoint` maps samples back onto a grid. The samples an
# acquisition would measure are the transform of the image; below they are
# checked against the sum that defines them, evaluated in double precision over
# one spoke, which is a reference outside BART and outside FINUFFT.

samples = bartorch.nufft(image, golden)
print(f"samples {tuple(samples.shape)}")

spoke = golden[0].real.to(torch.float64)
axis_y, axis_x = torch.meshgrid(
    torch.arange(SIZE) - SIZE // 2, torch.arange(SIZE) - SIZE // 2, indexing="ij"
)
phase = (
    -2j
    * np.pi
    / SIZE
    * (
        spoke[:, 0:1] * axis_x.reshape(1, -1).to(torch.float64)
        + spoke[:, 1:2] * axis_y.reshape(1, -1).to(torch.float64)
    )
)
explicit = torch.exp(phase) @ image.to(torch.complex128).reshape(-1, 1) / SIZE

difference = float((explicit - samples[0]).abs().max() / explicit.abs().max())
print(f"largest relative difference from the explicit sum: {difference:.1e}")

# %%
#
# The transform is planned to a tolerance rather than computed exactly, and the
# difference above is within the tolerance it was planned with: a thousandth by
# default, on a grid a quarter larger than the image. The default is chosen for
# reconstruction, where the transform's error is intended to stay small beside
# the effect of noise and undersampling; :class:`bartorch.linop.NUFFT` takes
# ``oversampling`` and ``width`` where more accuracy is needed.
#
# Density compensation
# --------------------
#
# The adjoint is not the inverse. Every spoke passes through the centre of
# k-space, so the radial sampling density falls as :math:`1/\lvert k \rvert`
# and the adjoint overweights low frequencies. The weight that compensates for
# it is the inverse sampling density [#pipe]_, which for radial sampling is
# proportional to the distance from the centre.

radius = torch.linalg.norm(golden.real[..., :2], dim=-1, keepdim=True)
weights = radius.clamp(min=0.25).to(torch.complex64)

plain = bartorch.nufft_adjoint(samples, golden, image_shape=(SIZE, SIZE))
compensated = bartorch.nufft_adjoint(samples * weights, golden, image_shape=(SIZE, SIZE))

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(1, 3)
show(axes[0, 0], image, "image", vmax=float(image.abs().max()))
show(axes[0, 1], plain / plain.abs().max(), "adjoint", vmax=1.0)
show(axes[0, 2], compensated / compensated.abs().max(), "density compensated", vmax=1.0)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The uncompensated adjoint is the image convolved with the point spread
# function, the inverse Fourier transform of the sampling density; the density
# is concentrated at the centre of k-space, so the result is blurred. The
# compensated adjoint resolves the tissue boundaries. Neither recovers the
# k-space the trajectory does not reach: a radial acquisition samples a disc,
# so the frequencies in the corners of the Cartesian grid are missing whatever
# the weights are.
#
# The normal operator
# -------------------
#
# :class:`bartorch.linop.NUFFT` is the transform as an operator, and carries the
# weights and a subspace basis where there are any, because its normal operator
# :math:`A^H A` is built over both. That normal is a convolution with a point
# spread function on a doubled grid rather than a transform each way
# [#fessler2005]_, which is what a solver applies once per iteration.

A = linop.NUFFT(golden, image_shape=(SIZE, SIZE))

repeats = 5
start = time.perf_counter()
for _ in range(repeats):
    toeplitz = A.gram()(image)
convolution = (time.perf_counter() - start) / repeats

start = time.perf_counter()
for _ in range(repeats):
    pair = A.H(A(image))
transforms = (time.perf_counter() - start) / repeats

print(f"A^H A as a convolution   {1e3 * convolution:6.1f} ms")
print(f"A^H A as two transforms  {1e3 * transforms:6.1f} ms")
print(f"relative difference      {float((toeplitz - pair).abs().max() / pair.abs().max()):.1e}")

# %%
#
# The two agree to a small multiple of the transform's tolerance.
#
# :func:`bartorch.tools.psf` computes that function on its own. Its extent is
# the aliasing the trajectory produces: for a fully sampled radial trajectory
# it is a central peak with a low, broad skirt, and undersampling raises the
# skirt into the streaks a radial reconstruction is known for.

fully_sampled = bt.psf(bt.traj(readout=SIZE, spokes=SPOKES, radial=True, golden=True))
undersampled = bt.psf(bt.traj(readout=SIZE, spokes=SPOKES // 8, radial=True, golden=True))

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(1, 2)
for axis, values, title in (
    (axes[0, 0], fully_sampled, f"{SPOKES} spokes"),
    (axes[0, 1], undersampled, f"{SPOKES // 8} spokes"),
):
    values = values.abs()
    handle = axis.imshow(
        (values / values.max()).log10().clamp(min=-4).cpu().numpy(), cmap="magma", vmin=-4, vmax=0
    )
    axis.set_title(title)
figure.colorbar(handle, ax=axes[0, 1], label="$\\log_{10}$ magnitude", fraction=0.046)
figure.suptitle("point spread function, normalized")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# A reconstruction that uses all of this -- the transform, the weights, the
# sensitivities and the normal operator -- is
# :doc:`02-radial-sense`.

# %%
#
# References
# ----------
#
# .. [#finufft] Barnett AH, Magland J, af Klinteberg L. A parallel nonuniform fast
#    Fourier transform library based on an "exponential of semicircle" kernel.
#    *SIAM J Sci Comput* 41(5):C479-C504 (2019).
#    https://doi.org/10.1137/18M120885X
#
# .. [#winkelmann] Winkelmann S, Schaeffter T, Koehler T, Eggers H, Doessel O. An optimal
#    radial profile order based on the Golden Ratio for time-resolved MRI.
#    *IEEE Trans Med Imaging* 26(1):68-76 (2007).
#    https://doi.org/10.1109/TMI.2006.885337
#
# .. [#pipe] Pipe JG, Menon P. Sampling density compensation in MRI: rationale and an
#    iterative numerical solution. *Magn Reson Med* 41(1):179-186 (1999).
#    https://doi.org/10.1002/(SICI)1522-2594(199901)41:1%3C179::AID-MRM25%3E3.0.CO;2-V
#
# .. [#fessler2005] Fessler JA, Lee S, Olafsson VT, Shi HR, Noll DC. Toeplitz-based iterative
#    image reconstruction for MRI with correction for magnetic field
#    inhomogeneity. *IEEE Trans Signal Process* 53(9):3393-3402 (2005).
#    https://doi.org/10.1109/TSP.2005.853152
