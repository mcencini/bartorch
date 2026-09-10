"""
A Cartesian encoding and its adjoint
====================================

Compose sensitivity multiplication, a centered unitary FFT, and sampling.
Check the complex adjoint identity and compare the forward result to PyTorch's
FFT. Requires the base CPU installation.
"""

import matplotlib.pyplot as plt
import torch

import bartorch
from bartorch import linop

bartorch.set_num_threads(1)
torch.manual_seed(0)
n, ncoils = 24, 4
grid = torch.linspace(-1, 1, n)
y, x = torch.meshgrid(grid, grid, indexing="ij")
maps = torch.stack([torch.exp(1j * (j * x + (3 - j) * y)) for j in range(ncoils)])
maps = maps / ncoils**0.5
image_shape, coil_shape = (1, n, n), (ncoils, n, n)
S = linop.MultiplySum(maps, image_shape, coil_shape)
F = linop.FFT(coil_shape, axes=(-2, -1))
mask = torch.zeros(1, n, n, dtype=torch.complex64)
mask[:, ::2, :] = 1
P = linop.Sampling(mask, coil_shape)
A = P @ F @ S
image = torch.exp(-5 * (x.square() + y.square())).to(torch.complex64)[None]
measurements = A(image)

reference = (
    torch.fft.fftshift(
        torch.fft.fft2(torch.fft.ifftshift(maps * image, dim=(-2, -1)), norm="ortho"),
        dim=(-2, -1),
    )
    * mask
)
torch.testing.assert_close(measurements, reference, atol=2e-6, rtol=2e-5)
probe = torch.randn(*coil_shape, dtype=torch.complex64)
lhs = torch.vdot(measurements.flatten(), probe.flatten())
rhs = torch.vdot(image.flatten(), A.adjoint(probe).flatten())
torch.testing.assert_close(lhs, rhs, atol=2e-5, rtol=2e-5)
print(f"Complex adjoint error: {abs(lhs - rhs).item():.2e}")

# %%
# The adjoint is a backprojection, not generally the inverse. Sampling removes
# information, so an iterative solve and appropriate prior may be needed.
fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
for ax, data, title in zip(
    axes,
    [image[0].abs(), mask[0].real, A.adjoint(measurements)[0].abs()],
    ["Image magnitude", "Sampling mask", "Adjoint magnitude"],
):
    ax.imshow(data.numpy(), cmap="gray")
    ax.set_title(title)
    ax.axis("off")
plt.show()
