"""
====================
Nonlinear inversion
====================

Estimating the image and the coil sensitivities together, from undersampled
data whose fully sampled central region is too small for a separate
calibration.

ESPIRiT [#espirit]_ estimates the sensitivities from a fully sampled region at
the centre of k-space, and a linear reconstruction then uses them as known.
Where the acquisition provides no such region, the sensitivities are unknowns
like the image,
and the forward model

.. math::

   y_c = P F (S_c \\cdot x)

is bilinear rather than linear: it is a product of two unknowns. Nonlinear
inversion (``nlinv``) [#nlinv]_ solves it by iteratively regularized
Gauss-Newton [#bakushinsky]_, and
the smoothness of the sensitivities, which constrains the factorization,
enters as a weighting inside the model rather than as a penalty beside it.

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
from pathlib import Path

import brainweb_dl
import numpy as np
import torch
from brainweb_dl import get_mri

import bartorch
import bartorch.tools as bt
from bartorch import nlop

SIZE = 128
COILS = 8
ACCELERATION = 3
CALIBRATION = 6  # lines at the centre, far fewer than ESPIRiT needs

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

# sphinx_gallery_start_ignore
# The acquisition of :doc:`/auto_examples/01-basics/01-from-kspace-to-image`:
# the k-space of the coil images, and a variable-density random set of phase
# encodes with a fully sampled centre.
kspace = bt.noise(bartorch.fft(sensitivities * image, axes=(-2, -1), unitary=True), n=1e-5, s=42)

encodes = torch.arange(SIZE) - SIZE // 2
centre = (encodes.abs() < CALIBRATION // 2).to(torch.float32)
drawn = torch.multinomial(
    (1.0 + 2.0 * encodes.abs() / SIZE) ** -3.0 * (1.0 - centre),
    SIZE // ACCELERATION - CALIBRATION,
    replacement=False,
    generator=torch.Generator().manual_seed(11),
)
lines = centre.clone()
lines[drawn] = 1.0
# sphinx_gallery_end_ignore

pattern = lines.reshape(SIZE, 1).to(torch.complex64)
measured = kspace[:, None] * pattern

print(f"{float(lines.mean()):.0%} of the phase encodes, {CALIBRATION} of them at the centre")

# %%
#
# Six central lines locate the centre of k-space. With ESPIRiT's default
# kernel of six points, a calibration region of six lines leaves a single
# kernel position along the phase-encoding axis, too few rows for the
# calibration matrix of :func:`bartorch.tools.ecalib`.
#
# The application
# ---------------
#
# :func:`bartorch.tools.nlinv` takes the k-space and returns the image and,
# when asked, the sensitivities it estimated along the way. Its iteration count
# is Gauss-Newton steps rather than linear iterations, and it is a
# regularization parameter rather than a convergence threshold: the
# regularization weight is halved after every step, so stopping early leaves a
# smoother image and running longer eventually lets the noise in. Eight steps
# is BART's default; twelve are used here.

STEPS = 12

reconstruction, estimated = bt.nlinv(measured, maxiter=STEPS, return_sensitivities=True)

print(f"NRMSE {bt.nrmse(image.abs(), reconstruction.abs(), scaled=True):.3f}")

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(1, 3)
peak = float(image.abs().max())
show(axes[0, 0], image, "phantom", vmax=peak)
show(axes[0, 1], reconstruction, "nlinv", vmax=None)
show(axes[0, 2], bartorch.rss(estimated[:, 0], axes=(0,)), "root sum of squares of the maps")

figure, axes = panels(2, 4)
# Each map on its own scale: the estimate is determined only up to the scale
# the image takes the reciprocal of.
for column in range(4):
    domain(axes[0, column], sensitivities[column], f"channel {column}")
    domain(axes[1, column], estimated[column, 0])
for row, label in enumerate(("simulated", "estimated")):
    axes[row, 0].set_ylabel(label)
    for axis in axes[row]:
        axis.set_xticks([])
        axis.set_yticks([])
phase_bar(figure, axes)
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# The estimated sensitivities are smooth by construction rather than by
# agreement with the data: the coil unknown is
# not the sensitivity map but its k-space representation
# :math:`\hat{s}`, and the map follows as
# :math:`S = \mathcal{F}^{-1}[(1 + a|k|^2)^{-b/2} \hat{s}]`. A step in the
# unknown is therefore a smooth change in the map by construction, and the
# joint problem needs no separate penalty on the coils. The pair is determined
# only up to a common factor: multiplying every map by a nonzero function
# :math:`\gamma(r)` and dividing the image by it leaves the data unchanged
# (:doc:`../../explanation/nonlinear`). The smoothness weighting restricts
# :math:`\gamma` to smooth functions, which is why the two rows above are drawn
# on their own scales and why a nonlinear inversion is reported after
# normalizing by the root sum of squares of the maps. Outside the object
# neither factor is determined at all -- their product is zero for any pair --
# so what is drawn there follows from the initialization and the weighting.
#
# The model and the solver
# ------------------------
#
# :class:`bartorch.nlop.NonlinearSense` is that forward model as a nonlinear
# operator with two inputs, and :class:`bartorch.nlop.IRGNM` is the
# Gauss-Newton loop over it. Each step linearizes the model at the current
# point :math:`x_k` and solves
#
# .. math::
#
#    \min_x \, \| DF_{x_k} (x - x_k) - (y - F(x_k)) \|^2
#    + \alpha_k \| x - x_{\mathrm{ref}} \|^2,
#
# with :math:`x_{\mathrm{ref}}` zero unless one is given and :math:`\alpha_k`
# halved after every step, so the first steps are heavily regularized and the
# later ones are not.

model = nlop.NonlinearSense(
    (COILS, 1, SIZE, SIZE), pattern=lines.reshape(1, SIZE, 1).to(torch.complex64)
)
print(f"inputs {model.ishapes} -> output {model.oshapes}")

# %%
#
# ``nlinv`` scales the data by ``100 / ||y||`` before it starts, which fixes
# the meaning of :math:`\alpha`, and takes its conjugate gradients to a hundred
# iterations or a tolerance of a tenth. Given the same three settings the loop
# written here is the application.

data = model.prepare(measured * (100.0 / float(torch.linalg.vector_norm(measured))))
fitted, coefficients = nlop.IRGNM(iterations=STEPS, cg_maxiter=100, cg_tol=0.1)(data, model)

maps = model.coils(coefficients)
combined = fitted.squeeze() * bartorch.rss(maps[:, 0], axes=(0,))

difference = float(
    (fitted.squeeze() - bt.nlinv(measured, maxiter=STEPS, normalize=False)).abs().max()
)
print(f"largest difference from nlinv: {difference / float(fitted.abs().max()):.1e}")
print(f"NRMSE {bt.nrmse(image.abs(), combined.abs(), scaled=True):.3f}")

# %%
#
# The two agree to single-precision round-off rather than to the last bit,
# because the scaling above is computed here and inside the application by
# different expressions.
#
# What the operator layer adds is everything around the step. The linearized
# problem can go to a solver from :mod:`bartorch.optim` instead of the
# conjugate gradients inside the library (``inner=optim.CG()`` is the same
# method written out, and a regularized solver makes the step a regularized
# one), the loop can be unrolled as :class:`bartorch.nlop.IRGNMBlock`, and a
# Gauss-Newton step is differentiable with respect to the data, the iterate,
# the regularization centre and :math:`\alpha`
# (:doc:`../../explanation/differentiation`).
#
# Reconstructing parameter maps rather than an image, by putting a signal model
# in front of the same encoding, is :doc:`02-quantitative-models`.

# %%
#
# References
# ----------
#
# .. [#espirit] Uecker M, Lai P, Murphy MJ, Virtue P, Elad M, Pauly JM, Vasanawala SS,
#    Lustig M. ESPIRiT -- an eigenvalue approach to autocalibrating parallel
#    MRI: where SENSE meets GRAPPA. *Magn Reson Med* 71(3):990-1001 (2014).
#    https://doi.org/10.1002/mrm.24751
#
# .. [#nlinv] Uecker M, Hohage T, Block KT, Frahm J. Image reconstruction by regularized
#    nonlinear inversion -- joint estimation of coil sensitivities and image
#    content. *Magn Reson Med* 60(3):674-682 (2008).
#    https://doi.org/10.1002/mrm.21691
#
# .. [#bakushinsky] Bakushinsky AB, Kokurin MY. *Iterative Methods for Approximate Solution of
#    Inverse Problems.* Springer (2004).
#    https://doi.org/10.1007/978-1-4020-3122-9
