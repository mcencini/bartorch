r"""
====================
MoDL, on BART's ADMM
====================

An unrolled network for undersampled Cartesian SENSE: a convolutional denoiser
in the proximal step of BART's alternating-direction iteration, trained end to
end against fully sampled images.

MoDL writes a reconstruction as an alternation between a learned denoiser and a
data-consistency step, and trains the denoiser through it. As published the
alternation is half-quadratic splitting, which is

.. math::

   z^{k} &= D_w(x^{k}) \\
   x^{k+1} &= \arg\min_x \; \|A x - y\|_2^2 + \lambda \|x - z^{k}\|_2^2,

the second line a conjugate-gradient solve of :math:`(A^H A + \lambda) x = A^H
y + \lambda z^{k}`. The alternating direction method of multipliers is the
same splitting with a dual variable :math:`u` carried along:

.. math::

   x^{k+1} &= \arg\min_x \; \|A x - y\|_2^2 + \rho \|x - z^{k} + u^{k}\|_2^2 \\
   z^{k+1} &= D_w(x^{k+1} + u^{k}) \\
   u^{k+1} &= u^{k} + x^{k+1} - z^{k+1},

so MoDL is this iteration with :math:`u` held at zero. The dual accumulates the
mismatch between the data-consistent iterate and the denoised one, which is what
makes the fixed point of the iteration a solution of the constrained problem
rather than of the penalized one: the denoiser's strength stops being something
:math:`\rho` has to be balanced against at every step.

Nothing of that iteration is written here. :class:`bartorch.optim.ADMMBlock` is
``admm.c``'s step, x-update included -- BART's ``cg_xupdate`` is the conjugate
gradients MoDL's own implementation writes out -- and
:class:`bartorch.learning.Unrolled` is the loop over it. What the network
contributes is the proximal step, through
:class:`bartorch.priors.ImplicitPrior`, and :math:`\rho`, which is a
:class:`torch.nn.Parameter` like any weight.

Aggarwal HK, Mani MP, Jacob M. *MoDL: model-based deep learning architecture
for inverse problems.* IEEE Trans Med Imaging 38(2):394-405 (2019).

Boyd S, Parikh N, Chu E, Peleato B, Eckstein J. *Distributed optimization and
statistical learning via the alternating direction method of multipliers.*
Found Trends Mach Learn 3(1):1-122 (2011).
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
# The images
# ----------
#
# A stack of axial slices of one BrainWeb subject, each turned into a
# :math:`T_1`-weighted spin-echo image the way
# :doc:`../01-basics/01-from-kspace-to-image` turns one, and given a smooth
# phase so that nothing here depends on the image being real. The slices are
# split into a training set and a validation set by position rather than at
# random, so that a validation slice is not the neighbour of a training one.
#
# This is a demonstration of the assembly, not of a trained network: one
# subject, one sampling pattern and twenty-four slices are a phantom, and the
# weights that come out of it mean nothing beyond this page.

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
    """One axial slice as the complex image a spin-echo experiment would measure."""
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
# The acquisition
# ---------------
#
# Eight channels of BART's analytical head coil, and a variable-density random
# undersampling of the phase encodes with the centre of k-space kept -- the
# one-dimensional Cartesian mask MoDL is posed over. The pattern is fixed for
# the whole dataset, so one operator serves every slice; where the sampling
# varies per item the operator does too, and it is built per item, because its
# data is not a batch axis of it.

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
# The measured k-space of a slice is :math:`A x` with complex Gaussian noise
# added. An operator is built for one image and not for a batch of them, so a
# batch is applied item by item; the iteration blocks in
# :mod:`bartorch.optim` do the same thing internally, which is why the network
# below takes a batch and the operator does not.

NOISE = 0.005


def measure(images, generator=None):
    """The k-space of each image, and the adjoint reconstruction a network starts from."""
    x = torch.stack(list(images))
    y = torch.stack([A(item) for item in x])
    y = y + NOISE * torch.randn(y.shape, dtype=torch.complex64, generator=generator)
    return x, y, torch.stack([A.H(item) for item in y])


# %%
#
# The dataset
# -----------
#
# ``torchio`` carries the images and augments them. Its ``ScalarImage`` holds a
# real tensor of ``(channels, width, height, depth)``, which
# :func:`bartorch.learning.as_real` and :func:`~bartorch.learning.as_complex`
# convert to and from: the real and imaginary parts become the two channels,
# and a slice is a volume one voxel deep.
#
# The augmentation is the reason to use it rather than a list. A transform is
# drawn per subject and applied to every image in that subject, so an image and
# anything that has to stay registered to it -- its coil sensitivities, its
# parameter maps -- move together; here there is one image per subject, and the
# transform is a flip and a small rotation, which keep the tissue statistics a
# denoiser learns while changing the anatomy it sees.

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
    """A batch of subjects as the images, the k-space and the adjoint reconstruction."""
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
# The network
# -----------
#
# Four objects, each of which is one thing:
#
# * ``deepinv``'s ``DnCNN`` is the denoiser MoDL's own five-layer residual
#   network belongs to the family of. It is an ``nn.Module`` taking real
#   images, and needs no adapter of its own.
# * :class:`bartorch.learning.Denoiser` is the layout between that network and
#   an image here: two channels for the real and imaginary parts, the batch
#   folded, and each image scaled to unit peak modulus around the call.
# * :class:`bartorch.priors.ImplicitPrior` puts it where a regularizer goes.
# * :class:`bartorch.learning.Unrolled` applies the ADMM step
#   ``ITERATIONS`` times. One block is shared by every iteration, which is the
#   weight sharing MoDL means, and ``rho`` is asked for a gradient, which is
#   MoDL's learned :math:`\lambda`.
#
# ``alpha=1.0`` turns off BART's over-relaxation, so the step is the iteration
# written above; ``cg_maxiter`` is the budget of the x-update, MoDL's ten.

from deepinv.models import DnCNN


def modl():
    """A fresh unrolled network, and the block whose weights it shares."""
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
# ``monai``. Nothing about the reconstruction reaches into it -- by the time
# the loss is taken, the output is a tensor.

psnr = PSNRMetric(max_val=1.0)
ssim = SSIMMetric(spatial_dims=2, data_range=1.0)


class Reconstruction(lightning.LightningModule):
    """The unrolled network, trained against fully sampled images."""

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
# What it reconstructs
# --------------------
#
# Against three reconstructions of the same k-space that learn nothing: the
# adjoint, which is what the network starts from; a conjugate-gradient SENSE
# fit, which is the data term alone; and the same ADMM iteration with a
# wavelet penalty in place of the denoiser, run to fifty iterations rather than
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
    """PSNR and SSIM of a batch of magnitudes against the truth's."""
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
# What the table is not is a comparison of methods. Fifteen epochs over
# twenty-four slices of one subject, against a wavelet penalty with fifty
# iterations and a weight chosen by hand, says nothing about either on real
# data -- five learned iterations landing in the same range as fifty
# hand-written ones is the whole of what it shows. A network trained to be
# believed is trained on many subjects, validated on subjects it never saw, and
# compared at a fixed reconstruction time.

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
# Training a stack that does not fit
# ----------------------------------
#
# Five iterations of a two-dimensional encoding record a graph that fits
# anywhere. A three-dimensional non-Cartesian encoding with ten of them does
# not, and the memory is the denoiser's activations: an operator's backward
# pass is another application of the operator and stores nothing that grows
# with the iteration count, while a convolutional network's stores every
# activation it made, once per iteration.
#
# :class:`~bartorch.learning.Unrolled` takes both ways around that, and neither
# changes what the network computes:
#
# * ``detach=True`` starts each iteration from a detached state, so the graph
#   spans one iteration. With a loss on each of
#   :meth:`~bartorch.learning.Unrolled.steps` this is greedy per-iteration
#   training, whose memory does not depend on the count at all.
# * ``checkpoint=True`` keeps the states between iterations and recomputes a
#   step's interior in the backward pass. The gradient is the end-to-end one,
#   to the bit; each block is applied twice.
#
# Pretraining the denoiser on its own, then training greedily, then fine-tuning
# the whole stack with checkpointing, is the staged schedule a fully
# three-dimensional unrolled reconstruction is trained with.
#
# Urman Y, Nishimura M, Abraham DR, Cao X, Setsompop K. *Fully 3D unrolled
# magnetic resonance fingerprinting reconstruction via staged pretraining and
# implicit gridding.* Magn Reson Med 96(5):2516-2529 (2026).

greedy, greedy_block = modl()
greedy.detach = True

optimizer = torch.optim.Adam(greedy.parameters(), lr=1e-3)
x, y, start = measure(train_images[:2], torch.Generator().manual_seed(3))

for step in range(3):
    optimizer.zero_grad()
    # One loss per iteration, each reaching only the step that made it.
    for image in greedy.steps(y, A, x0=start):
        (image - x).abs().square().mean().backward()
    optimizer.step()

print(f"greedy: rho {float(greedy_block.rho.detach()):.3f}")

# %%
#
# Checkpointing answers with the same gradient as recording the whole stack,
# which is what says it is a memory decision and not a modelling one. Here it
# is the gradient of ``rho``, which reaches through every iteration and through
# each x-update's conjugate-gradient solve.

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
# The third route is not to unroll at all.
# :class:`bartorch.optim.FixedPoint` drives the block to its fixed point and
# differentiates there by solving the adjoint fixed-point equation, so its
# memory is one step's whatever the iteration count -- a deep-equilibrium
# model, of which this stack is the truncated version.
