"""
Radial encoding and a direct Fourier reference
==============================================

Generate a radial trajectory and evaluate a NUFFT. Requires ``bartorch[finufft]``
and the base plotting dependencies. Explicitly select optional examples to run
this page. No density compensation is used: an adjoint is not an inverse.
"""

import matplotlib.pyplot as plt
import torch

import bartorch
import bartorch.tools as bt
from bartorch.ops import LinearOperator

bartorch.set_num_threads(1)
bartorch.finufft.configure(tolerance=1e-6, upsampling=2.0)
n = 16
trajectory = bt.traj(x=n, y=12, r=True)
image = bt.phantom([n, n]).reshape(1, n, n)
A = LinearOperator.nufft(trajectory, image.shape, toeplitz=False)
samples = A(image)

# %%
# BART's trajectory is (spokes, samples, 3) in grid units. Compare with the
# defining Fourier sum on this tiny problem, using a negative exponent and
# unitary 2-D scaling 1/n. This reference is too expensive for a full scan.
grid = torch.arange(n, dtype=torch.float64) - n // 2
y, x = torch.meshgrid(grid, grid, indexing="ij")
coords = trajectory.real.to(torch.float64)
phase = torch.exp(-2j * torch.pi * (
    coords[..., 0, None, None] * x + coords[..., 1, None, None] * y
) / n)
reference = (phase * image[0]).sum(dim=(-2, -1)) / n
error = (samples.reshape(reference.shape) - reference).norm() / reference.norm()
assert error < 2e-4
print(f"Relative NUFFT error against direct sum: {error.item():.2e}")

fig, axes = plt.subplots(1, 2, figsize=(7, 3), layout="constrained")
axes[0].plot(coords[..., 0].flatten(), coords[..., 1].flatten(), ".", markersize=2)
axes[0].set(xlabel="kx (grid units)", ylabel="ky (grid units)", aspect="equal")
axes[1].imshow(A.adjoint(samples).abs().squeeze().numpy(), cmap="gray")
axes[1].set_title("Unweighted adjoint magnitude")
axes[1].axis("off")
plt.show()
