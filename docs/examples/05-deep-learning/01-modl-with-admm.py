r"""
====================
MoDL, on BART's ADMM
====================

An unrolled network for undersampled Cartesian SENSE: a convolutional denoiser
in the proximal step of BART's alternating-direction iteration, trained end to
end against fully sampled images.

MoDL [#modl]_ writes a reconstruction as an alternation between a learned denoiser and a
data-consistency step, and trains the denoiser through it. As published the
alternation is half-quadratic splitting, which is

.. math::

   z^{k} &= D_w(x^{k}) \\
   x^{k+1} &= \arg\min_x \; \|A x - y\|_2^2 + \lambda \|x - z^{k}\|_2^2,

the second line a conjugate-gradient solve of :math:`(A^H A + \lambda) x = A^H
y + \lambda z^{k}`. The alternating direction method of multipliers
[#boyd]_ is the same splitting with a dual variable :math:`u` carried along:

.. math::

   x^{k+1} &= \arg\min_x \; \|A x - y\|_2^2 + \rho \|x - z^{k} + u^{k}\|_2^2 \\
   z^{k+1} &= D_w(x^{k+1} + u^{k}) \\
   u^{k+1} &= u^{k} + x^{k+1} - z^{k+1},

MoDL is therefore this iteration with :math:`u` fixed at zero. The dual
variable accumulates the mismatch between the data-consistent and the denoised
iterate, so that a fixed point of the iteration solves the constrained problem
rather than the penalized one.

The iteration itself is BART's. :class:`bartorch.optim.ADMMBlock` implements
``admm.c``'s step, including the conjugate-gradient x-update that MoDL's own
implementation writes out, and :class:`bartorch.learning.Unrolled` applies it
repeatedly. The network supplies the proximal step, through
:class:`bartorch.priors.ImplicitPrior`, and :math:`\rho`, which is a
:class:`torch.nn.Parameter`.

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
    """One panel, of a magnitude."""
    values = values.detach().abs().cpu().numpy() if hasattr(values, "detach") else values
    handle = axis.imshow(values, cmap=cmap, vmin=0.0, vmax=vmax)
    if title is not None:
        axis.set_title(title)
    return handle


# sphinx_gallery_end_ignore
import csv
from pathlib import Path

import brainweb_dl
import lightning
import numpy as np
import torch
import torchio
from brainweb_dl import get_mri
from monai.metrics import PSNRMetric, SSIMMetric
from torch.utils.data import DataLoader

import bartorch
import bartorch.tools as bt
from bartorch import learning, linop, optim, priors

SIZE = 128
COILS = 8
SLICES = 32  # axial slices taken from the volume
ITERATIONS = 5  # unrolled steps, MoDL's K
EPOCHS = 15

torch.manual_seed(0)

# %%
#
# Images
# ------
#
# Axial slices of one BrainWeb subject, each converted into a
# :math:`T_1`-weighted spin-echo image as in
# :doc:`../01-basics/01-from-kspace-to-image` and given a smooth phase, so that
# no step below depends on the image being real. The slices are split into
# training and validation sets by position rather than at random, so that a
# validation slice is not adjacent to a training slice.
#
# One subject, twenty-four slices and a single sampling pattern constitute a
# phantom. The weights obtained below are not expected to generalize, and the
# page demonstrates the construction rather than a trained model.

# sphinx_gallery_start_ignore
TISSUES = (1, 2, 3, 4, 5, 6, 8)  # everything the table gives relaxation times
MARGIN = 0.25  # what the field of view leaves around the head
FIRST = 60  # the first axial slice taken, above the skull base
STEP = 3  # slices apart, so that neighbours are not near-duplicates
TR, TE = 600.0, 12.0  # ms

table = Path(brainweb_dl.__file__).parent / "data" / "brainweb1_tissues.csv"
entries = list(csv.DictReader(table.open()))
tissue_t1 = np.array([float(row["T1 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_t2 = np.array([float(row["T2 (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]
tissue_pd = np.array([float(row["PD (ms)"]) for row in entries], dtype=np.float32)[list(TISSUES)]

volume = get_mri(sub_id=0, contrast="fuzzy")

grid_y, grid_x = torch.meshgrid(
    torch.linspace(-1.0, 1.0, SIZE), torch.linspace(-1.0, 1.0, SIZE), indexing="ij"
)
phase = torch.exp(0.8j * (grid_x**2 - 0.5 * grid_y**2))


def _slice_image(index):
    """One axial slice as the complex image a spin-echo acquisition would measure."""
    fractions = np.flipud(volume[index])[..., list(TISSUES)].copy()
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

    weights = memberships * torch.as_tensor(tissue_pd)[:, None, None]
    share = weights.sum(0).clamp(min=1e-6)
    T1 = (weights * torch.as_tensor(tissue_t1)[:, None, None]).sum(0) / share
    T2 = (weights * torch.as_tensor(tissue_t2)[:, None, None]).sum(0) / share
    density = weights.sum(0) / weights.sum(0).max().clamp(min=1e-6)

    recovery = 1 - torch.exp(-TR / T1.clamp(min=1e-3))
    signal = density * recovery * torch.exp(-TE / T2.clamp(min=1e-3))
    signal = torch.where(T1 > 0, signal, torch.zeros(()))
    return ((signal / signal.max().clamp(min=1e-6)) * phase).to(torch.complex64)


images = [_slice_image(FIRST + STEP * k) for k in range(SLICES)]
# sphinx_gallery_end_ignore
train_images = images[:24]
valid_images = images[24:]

print(f"{len(train_images)} slices to train on, {len(valid_images)} to validate on")

# %%
#
# Acquisition
# -----------
#
# Eight channels of BART's analytical head coil, and a variable-density random
# undersampling of the phase encodes with a fully sampled k-space centre: the
# one-dimensional Cartesian sampling MoDL is posed over. The pattern is the
# same for every slice, so a single operator serves the whole dataset. Where
# the sampling varies between items, an operator is constructed per item: its
# sensitivities and pattern are not batch axes of it.

ACCELERATION = 4
CENTRE = 8  # phase encodes always acquired

sensitivities = bt.coils(t=bt.grid(D=(SIZE, SIZE, 1)), n=COILS)[:, 0]
sensitivities = sensitivities / bartorch.rss(sensitivities, axes=(0,), keepdim=True)

density = torch.exp(-0.5 * ((torch.arange(SIZE) - SIZE / 2) / (SIZE / 6)) ** 2)
density = density / density.sum() * (SIZE / ACCELERATION)
lines = torch.rand(SIZE, generator=torch.Generator().manual_seed(1)) < density
lines[SIZE // 2 - CENTRE // 2 : SIZE // 2 + CENTRE // 2] = True
pattern = lines.to(torch.complex64)[:, None].expand(SIZE, SIZE).contiguous()

A = linop.CartesianSense(sensitivities, (SIZE, SIZE), pattern=pattern)

print(f"{int(lines.sum())} of {SIZE} phase encodes, {SIZE / int(lines.sum()):.1f}-fold")
print(f"A: {A.ishape} -> {A.oshape}")

# %%
#
# The measured k-space of a slice is :math:`A x` with additive complex
# Gaussian noise. An operator is constructed for a single image, so a batch is
# applied item by item; the iteration blocks in :mod:`bartorch.optim` do the
# same internally, and the network below therefore accepts a batch where the
# operator does not.

NOISE = 0.005


def measure(images, generator=None):
    """Simulate k-space for each image, with the adjoint reconstruction to start from."""
    x = torch.stack(list(images))
    y = torch.stack([A(item) for item in x])
    y = y + NOISE * torch.randn(y.shape, dtype=torch.complex64, generator=generator)
    return x, y, torch.stack([A.H(item) for item in y])


# %%
#
# Dataset
# -------
#
# ``torchio`` holds the images and augments them. Its ``ScalarImage`` requires
# a real tensor of shape ``(channels, width, height, depth)``;
# :func:`bartorch.learning.as_real` and :func:`~bartorch.learning.as_complex`
# convert between that layout and a complex image, the real and imaginary parts
# becoming the two channels and a slice a volume one voxel deep.
#
# Augmentation is the reason to prefer it to a plain list. A transform is drawn
# per subject and applied to every image of that subject, so an image and
# anything that must remain registered with it -- coil sensitivities, parameter
# maps -- are transformed consistently. Here each subject holds one image, and
# the transform is a flip and a rotation of at most eight degrees, which
# preserve the tissue statistics the denoiser is trained on while varying the
# anatomy.

augmentation = torchio.Compose(
    [
        torchio.RandomFlip(axes=(0,), flip_probability=0.5),
        torchio.RandomAffine(scales=0, degrees=(0, 0, 0, 0, -8, 8), translation=0),
    ]
)


def subjects(images):
    return [
        torchio.Subject(image=torchio.ScalarImage(tensor=learning.as_real(image)[..., None]))
        for image in images
    ]


def collate(batch):
    """Collate subjects into images, simulated k-space and adjoint reconstructions."""
    return measure(learning.as_complex(subject["image"][torchio.DATA][..., 0]) for subject in batch)


train_loader = DataLoader(
    torchio.SubjectsDataset(subjects(train_images), transform=augmentation),
    batch_size=2,
    shuffle=True,
    collate_fn=collate,
)
valid_loader = DataLoader(
    torchio.SubjectsDataset(subjects(valid_images)), batch_size=2, collate_fn=collate
)

# %%
#
# Network
# -------
#
# Four objects:
#
# * ``deepinv``'s ``DnCNN`` [#dncnn]_, a residual convolutional denoiser of the family
#   MoDL's own five-layer network belongs to. It is an ``nn.Module`` operating
#   on real images.
# * :class:`bartorch.learning.Denoiser`, which converts between that layout and
#   a complex image: two channels for the real and imaginary parts, the batch
#   axes folded, and each image scaled to unit peak modulus around the call.
# * :class:`bartorch.priors.ImplicitPrior`, which presents the result as a
#   regularization term.
# * :class:`bartorch.learning.Unrolled`, which applies the ADMM step
#   ``ITERATIONS`` times. A single block is shared by every iteration, the
#   weight sharing MoDL specifies, and ``rho`` is made differentiable, MoDL's
#   learned :math:`\lambda`.
#
# ``alpha=1.0`` disables BART's over-relaxation, so that the step is the
# iteration written above; ``cg_maxiter`` is the x-update budget, MoDL's ten.

from deepinv.models import DnCNN


def modl():
    """Construct an unrolled network and return it with the block it shares."""
    network = DnCNN(in_channels=2, out_channels=2, depth=5, pretrained=None)
    denoiser = learning.Denoiser(network, channels=2)
    block = optim.ADMMBlock(priors.ImplicitPrior(denoiser), rho=0.05, alpha=1.0, cg_maxiter=10)
    block.rho.requires_grad_()
    return learning.Unrolled(block, iterations=ITERATIONS), block


model, block = modl()
learned = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"{learned} learned values, shared by all {ITERATIONS} iterations")
print(f"rho starts at {float(block.rho.detach()):.3f}")

# %%
#
# Training
# --------
#
# ``lightning`` runs the loop. The module is the ordinary supervised one: a
# forward pass, a loss against the fully sampled image, and metrics from
# ``monai``. The loss is taken on a tensor and requires nothing of the
# reconstruction that produced it.

psnr = PSNRMetric(max_val=1.0)
ssim = SSIMMetric(spatial_dims=2, data_range=1.0)


class Reconstruction(lightning.LightningModule):
    """Supervised training of an unrolled network against fully sampled images."""

    def __init__(self, model, lr=1e-3):
        super().__init__()
        self.model = model
        self.lr = lr

    def forward(self, y, start):
        return self.model(y, A, x0=start)

    def training_step(self, batch, index):
        x, y, start = batch
        loss = (self(y, start) - x).abs().square().mean()
        self.log("loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, index):
        x, y, start = batch
        out = self(y, start).abs()[:, None]
        self.log("psnr", psnr(out, x.abs()[:, None]).mean(), prog_bar=True)
        self.log("ssim", ssim(out, x.abs()[:, None]).mean(), prog_bar=True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


trainer = lightning.Trainer(
    max_epochs=EPOCHS,
    accelerator="cpu",
    logger=False,
    enable_checkpointing=False,
    enable_model_summary=False,
    gradient_clip_val=1.0,
)
trainer.fit(Reconstruction(model), train_loader, valid_loader)

print(f"rho ended at {float(block.rho.detach()):.3f}")

# %%
#
# Results
# -------
#
# Three reconstructions of the same k-space serve as references: the adjoint,
# which the network is started from; a conjugate-gradient SENSE fit, which
# minimizes the data term alone; and the same ADMM iteration with a wavelet
# penalty in place of the denoiser, run for fifty iterations rather than
# five.

torch.manual_seed(7)
truth, kspace, adjoint = measure(valid_images)

with torch.no_grad():
    learnedrecon = model(kspace, A, x0=adjoint)

cg = torch.stack([optim.cg(item, A, maxiter=30) for item in kspace])
wavelet = torch.stack(
    [
        optim.admm(item, A, priors.Wavelet(axes=(-1, -2), weight=0.002), maxiter=50, rho=0.1)
        for item in kspace
    ]
)


def quality(estimate):
    """PSNR and SSIM of a batch of magnitudes against the reference magnitudes."""
    a, b = estimate.abs()[:, None], truth.abs()[:, None]
    return float(psnr(a, b).mean()), float(ssim(a, b).mean())


rows = {
    "adjoint": adjoint,
    "CG SENSE": cg,
    "ADMM, wavelet": wavelet,
    f"MoDL, K={ITERATIONS}": learnedrecon,
}
for name, estimate in rows.items():
    made = quality(estimate)
    print(f"{name:>16}   PSNR {made[0]:5.2f} dB   SSIM {made[1]:.3f}")

# %%
#
# The table is not a comparison of methods. Fifteen epochs over twenty-four
# slices of one subject, set against a wavelet penalty of fifty iterations with
# a manually chosen weight, supports no conclusion about either on measured
# data; the observation available here is that five learned iterations reach
# the range of fifty hand-specified ones. A quantitative comparison would
# require many subjects, validation on subjects excluded from training, and a
# fixed reconstruction time.

# %%

# sphinx_gallery_start_ignore
figure, axes = panels(2, 5)
for row in range(2):
    top = float(truth[row].abs().max())
    show(axes[row, 0], truth[row], "truth" if 0 == row else None, vmax=top)
    for column, (name, estimate) in enumerate(rows.items(), start=1):
        show(axes[row, column], estimate[row], name if 0 == row else None, vmax=top)
figure.suptitle(f"two validation slices, {SIZE / int(lines.sum()):.1f}-fold undersampled")
plt.show()
# sphinx_gallery_end_ignore

# %%
#
# Differentiating a deeper stack
# ------------------------------
#
# Five iterations of a two-dimensional encoding record a graph of modest size.
# Ten iterations of a three-dimensional non-Cartesian encoding do not, and the
# memory is dominated by the denoiser's activations: the backward pass of an
# operator is a further application of that operator and stores nothing growing
# with the iteration count, whereas a convolutional network stores every
# activation, once per iteration.
#
# :class:`~bartorch.learning.Unrolled` provides two alternatives, neither of
# which alters the value the network computes:
#
# * ``detach=True`` starts each iteration from a detached state, so that the
#   graph spans one iteration. With a loss on each image yielded by
#   :meth:`~bartorch.learning.Unrolled.steps`, this is greedy per-iteration
#   training, whose memory is independent of the iteration count.
# * ``checkpoint=True`` retains the states between iterations and recomputes
#   the interior of a step during the backward pass. The gradient is the
#   end-to-end one and each block is applied twice.
#
# Pretraining the denoiser in isolation, then greedy per-iteration training,
# then end-to-end fine-tuning with checkpointing, is the staged schedule
# reported for a fully three-dimensional unrolled reconstruction [#urman]_.

greedy, greedy_block = modl()
greedy.detach = True

optimizer = torch.optim.Adam(greedy.parameters(), lr=1e-3)
x, y, start = measure(train_images[:2], torch.Generator().manual_seed(3))

for step in range(3):
    optimizer.zero_grad()
    # One loss per iteration, each propagating only into the step that produced it.
    for image in greedy.steps(y, A, x0=start):
        (image - x).abs().square().mean().backward()
    optimizer.step()

print(f"greedy: rho {float(greedy_block.rho.detach()):.3f}")

# %%
#
# Checkpointing yields the same gradient as recording the whole stack,
# establishing that the choice is one of memory and not of model. The gradient
# shown is that of ``rho``, which propagates through every iteration and
# through the conjugate-gradient solve of each x-update.

x, y, start = measure(valid_images[:1])
made = []

for recompute in (False, True):
    stack = learning.Unrolled(block, iterations=ITERATIONS, checkpoint=recompute)
    for parameter in stack.parameters():
        parameter.grad = None
    (stack(y, A, x0=start) - x).abs().square().mean().backward()
    made.append(float(block.rho.grad))

print(f"rho's gradient: {made[0]:.6g} recorded, {made[1]:.6g} recomputed")

# %%
#
# A third alternative is not to unroll. :class:`bartorch.optim.FixedPoint`
# drives the block to its fixed point and differentiates there by solving the
# adjoint fixed-point equation, so that its memory is that of a single step
# irrespective of the iteration count. This is a deep-equilibrium model [#deq]_, of
# which the stack above is the truncated form.

# %%
#
# References
# ----------
#
# .. [#modl] Aggarwal HK, Mani MP, Jacob M. MoDL: model-based deep learning
#    architecture for inverse problems. *IEEE Trans Med Imaging* 38(2):394-405
#    (2019). https://doi.org/10.1109/TMI.2018.2865356
#
# .. [#boyd] Boyd S, Parikh N, Chu E, Peleato B, Eckstein J. Distributed optimization
#    and statistical learning via the alternating direction method of
#    multipliers. *Found Trends Mach Learn* 3(1):1-122 (2011).
#    https://doi.org/10.1561/2200000016
#
# .. [#dncnn] Zhang K, Zuo W, Chen Y, Meng D, Zhang L. Beyond a Gaussian denoiser:
#    residual learning of deep CNN for image denoising. *IEEE Trans Image
#    Process* 26(7):3142-3155 (2017). https://doi.org/10.1109/TIP.2017.2662206
#
# .. [#urman] Urman Y, Nishimura M, Abraham DR, Cao X, Setsompop K. Fully 3D unrolled
#    magnetic resonance fingerprinting reconstruction via staged pretraining and
#    implicit gridding. *Magn Reson Med* 96(5):2516-2529 (2026).
#    https://doi.org/10.1002/mrm.70500
#
# .. [#deq] Bai S, Kolter JZ, Koltun V. Deep equilibrium models. *Advances in Neural
#    Information Processing Systems* 32:688-699 (2019).
#    https://arxiv.org/abs/1909.01377
