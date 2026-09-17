"""
=============================
Trajectories and transforms
=============================

The non-Cartesian interfaces: the trajectories
:func:`bartorch.tools.traj` generates, the non-uniform Fourier transform along
one, the density compensation an adjoint reconstruction needs, and the point
spread function the normal operator convolves with.

Every non-Cartesian transform in bartorch is computed by FINUFFT, which
evaluates

.. math::

   y_j = \\frac{1}{\\sqrt{N}} \\sum_{r} x_r \\, e^{-2\\pi i\\, k_j \\cdot r}

to a requested tolerance, with the sum over the :math:`N` voxels of the image.
There is no gridding kernel to choose and no deapodization to match:
:doc:`../../explanation/non-cartesian` states what the substitution covers and
what it costs.

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
image = (magnitude / magnitude.max()).to(torch.complex64)
# sphinx_gallery_end_ignore

# %%
#
# Trajectories
# ------------
#
# A trajectory is ``(*encoding, shots, samples, 3)`` in grid units: the
# coordinates of every sample, in units of the k-space cell of the image it
# encodes, so a readout of ``SIZE`` samples runs from :math:`-N/2` to
# :math:`N/2`. The third component is :math:`k_z`, zero throughout for a
# two-dimensional trajectory, and whether it is used decides whether the
# transform is two- or three-dimensional.
#
# Successive spokes are separated either by :math:`\pi` over their number,
# which tiles k-space uniformly for one frame, or by the golden angle, which
# tiles it approximately uniformly for *any* number of consecutive spokes. Only
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
# default, on a grid a quarter larger than the image. A reconstruction is limited by its data rather
# than by its transform, so spending less on the transform is usually the right
# trade; :class:`bartorch.linop.NUFFT` takes ``oversampling`` and ``width``
# where it is not.
#
# Density compensation
# --------------------
#
# The adjoint is not the inverse. A radial trajectory samples the centre of
# k-space once per spoke and its periphery once per spoke per ring, so summing
# the samples onto the grid weights low frequencies by the number of spokes.
# The weight that undoes it is the inverse sampling density, which for radial
# sampling is the distance from the centre.

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
# The uncompensated adjoint is the image convolved with the sampling density,
# which is concentrated at the centre of k-space and therefore low-pass. The
# compensated one resolves the tissue boundaries, and what it cannot recover is
# the k-space the trajectory never reaches: a radial acquisition samples a disc,
# so the frequencies in the corners of the Cartesian grid are missing whatever
# the weights are.
#
# The normal operator
# -------------------
#
# :class:`bartorch.linop.NUFFT` is the transform as an operator, and carries the
# weights and a subspace basis where there are any, because its normal operator
# :math:`A^H A` is built over both. That normal is a convolution with a point
# spread function on a doubled grid rather than a transform each way, which is
# what a solver applies once per iteration.

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
# The two agree to a small multiple of the transform's tolerance, and the
# difference closes as the tolerance is tightened -- the evidence that the
# point spread function is the right one rather than nearly so.
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
