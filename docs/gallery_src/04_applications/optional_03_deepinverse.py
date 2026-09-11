"""
DeepInverse physics and a tiny learned reconstruction
=====================================================

Wrap a fixed BART encoding in DeepInverse and train a small residual network on
synthetic examples. Requires DeepInverse and the base CPU installation. No
pretrained weights are downloaded. This demonstrates gradient plumbing, not
clinical reconstruction quality or a reproduction of MoDL.

:meth:`~bartorch.linop.LinearOperator.to_deepinv` is the whole adapter: an
operator already differentiates in torch, with the adjoint as its backward
pass, and what the wrapper adds is DeepInverse's batch axis and ``A_dagger``
as BART's conjugate gradients. See the `DeepInverse custom physics tutorial
<https://deepinv.org/auto_examples/basics/demo_custom_physics.html>`_ for the
forward/adjoint interface it expects.
"""

import deepinv as dinv
import matplotlib.pyplot as plt
import torch
from torch import nn

import bartorch
from bartorch import linop

bartorch.set_num_threads(1)
torch.set_num_threads(1)
torch.manual_seed(7)


n = 16
shape = (1, n, n)
mask = torch.zeros(shape, dtype=torch.complex64)
mask[:, ::2, :] = 1
mask[:, n // 2 - 2 : n // 2 + 2, :] = 1
A = linop.Sampling(mask, shape) @ linop.FFT(shape, axes=(-2, -1))
physics = A.to_deepinv()
assert isinstance(physics, dinv.physics.LinearPhysics)

# %%
# Check the adapter against an independent torch-only Fourier expression,
# including a directional finite difference of the real-valued loss. BART uses
# complex64, so use a float32-appropriate finite-difference step.
probe = torch.randn(2, *shape, dtype=torch.complex64, requires_grad=True)
measurements = physics.A(probe)
reference = (
    torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(probe, dim=(-2, -1)), norm="ortho"), dim=(-2, -1)
    )
    * mask
)
torch.testing.assert_close(measurements, reference, atol=2e-6, rtol=2e-5)
loss = measurements.abs().square().sum() / 2
(gradient,) = torch.autograd.grad(loss, probe)
(reference_gradient,) = torch.autograd.grad(reference.abs().square().sum() / 2, probe)
torch.testing.assert_close(gradient, reference_gradient, atol=2e-6, rtol=2e-5)
direction = torch.randn_like(probe)
eps = 1e-3
with torch.no_grad():
    plus = physics.A(probe + eps * direction).abs().square().sum() / 2
    minus = physics.A(probe - eps * direction).abs().square().sum() / 2
finite_difference = (plus - minus) / (2 * eps)
predicted = torch.vdot(gradient.flatten(), direction.flatten()).real
torch.testing.assert_close(finite_difference, predicted, atol=0.05, rtol=0.01)

# Also validate the adjoint application's backward path, used in data consistency.
dual = torch.randn_like(measurements, requires_grad=True)
(dual_gradient,) = torch.autograd.grad(physics.A_adjoint(dual).abs().square().sum() / 2, dual)
expected = physics.A(physics.A_adjoint(dual.detach()))
torch.testing.assert_close(dual_gradient, expected, atol=2e-6, rtol=2e-5)

# %%
# Real/imaginary channels let a standard real-valued CNN process complex images.
# Use a distinct synthetic validation set. This tiny experiment makes no claim
# about generalization to scanner data.
grid = torch.linspace(-1, 1, n)
yy, xx = torch.meshgrid(grid, grid, indexing="ij")


def phantoms(count):
    centers = 0.8 * (torch.rand(count, 2, 1, 1) - 0.5)
    width = 8 + 8 * torch.rand(count, 1, 1)
    magnitude = torch.exp(-width * ((xx - centers[:, 0]) ** 2 + (yy - centers[:, 1]) ** 2))
    phase = torch.exp(1j * (xx + 0.3 * yy))
    return (magnitude * phase)[:, None].to(torch.complex64)


def as_channels(x):
    return torch.cat([x.real, x.imag], dim=1)


def as_complex(x):
    return torch.complex(x[:, :1], x[:, 1:])


network = nn.Sequential(nn.Conv2d(2, 8, 3, padding=1), nn.ReLU(), nn.Conv2d(8, 2, 3, padding=1))
nn.init.zeros_(network[-1].weight)
nn.init.zeros_(network[-1].bias)
optimizer = torch.optim.Adam(network.parameters(), lr=0.01)
train, validation = phantoms(8), phantoms(2)
y_train = physics.A(train)
initial = physics.A_adjoint(y_train)
history = []
for _ in range(30):
    optimizer.zero_grad()
    proposed = initial + as_complex(network(as_channels(initial)))
    # One explicit data-consistency step. Its gradient uses both adapter paths.
    estimate = proposed - 0.5 * physics.A_adjoint(physics.A(proposed) - y_train)
    objective = (estimate - train).abs().square().mean()
    objective.backward()
    optimizer.step()
    history.append(objective.item())
assert all(torch.isfinite(torch.tensor(history)))
assert history[-1] < history[0]
print(f"Training MSE: {history[0]:.4g} -> {history[-1]:.4g}")

with torch.no_grad():
    y_val = physics.A(validation)
    zero_filled = physics.A_adjoint(y_val)
    proposed = zero_filled + as_complex(network(as_channels(zero_filled)))
    learned = proposed - 0.5 * physics.A_adjoint(physics.A(proposed) - y_val)
    print("Validation MSE:", (learned - validation).abs().square().mean().item())

# %%
# This adapter keeps encoding parameters fixed and defines first-order gradients
# only. It does not differentiate trajectories, maps, or an iterative BART solve.
# Rebuild operators to change devices; this example's native handles are CPU-only.
fig, axes = plt.subplots(1, 4, figsize=(12, 3), layout="constrained")
for ax, value, title in zip(
    axes[:3],
    [validation[0, 0], zero_filled[0, 0], learned[0, 0]],
    ["Validation truth", "Zero-filled", "Learned + data consistency"],
):
    ax.imshow(value.abs().numpy(), cmap="gray", vmin=0, vmax=1)
    ax.set_title(title)
    ax.axis("off")
axes[3].semilogy(history)
axes[3].set(xlabel="Training step", ylabel="Complex MSE")
plt.show()
