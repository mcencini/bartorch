"""
Wave encoding and temporal subspaces
====================================

Assemble a synthetic hybrid-space wave model with readout padding and temporal
basis expansion. Requires the base CPU installation. The phase response is
illustrative, not calibrated to scanner gradients. Data are fully sampled to
isolate the operator algebra; this is not an accelerated Wave-Shuffling paper
reproduction. See :doc:`/gallery/research` for the BART ``wave``/``wshfl`` route.
"""

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as functional

import bartorch
import bartorch.tools as bt
from bartorch.ops import LinearOperator

bartorch.set_num_threads(1)
torch.manual_seed(2)
n, wx, echoes, rank = 8, 16, 4, 2
grid = torch.linspace(-1, 1, n)
zz, yy, xx = torch.meshgrid(grid, grid, grid, indexing="ij")
# Orthonormal columns keep the fully sampled coefficient inverse well posed.
basis, _ = torch.linalg.qr(torch.stack([torch.ones(echoes), torch.linspace(-1, 1, echoes)], 1))
basis = basis.to(torch.complex64)
image_shape = (1, rank, 1, n, n, n)
echo_shape = (echoes, 1, 1, n, n, n)
extended_shape = (echoes, 1, 1, n, n, wx)
Phi = LinearOperator.multiply_sum(basis.reshape(echoes, rank, 1, 1, 1, 1), image_shape, echo_shape)
padding = (wx - n) // 2
R = LinearOperator.from_callbacks(
    extended_shape,
    echo_shape,
    forward=lambda x: functional.pad(x, (padding, padding)),
    adjoint=lambda y: y[..., padding : padding + n].contiguous(),
)
Fx = LinearOperator.fft(extended_shape, axes=-1)
kybrid = torch.linspace(-torch.pi, torch.pi, wx)
wave_phase = 1.5 * yy[..., :1] * torch.sin(kybrid) + 1.5 * zz[..., :1] * torch.cos(kybrid)
response = torch.exp(1j * wave_phase).reshape(1, 1, 1, n, n, wx)
W = LinearOperator.diagonal(response, extended_shape)
Fyz = LinearOperator.fft(extended_shape, axes=(-3, -2))
A = Fyz @ W @ Fx @ R @ Phi

# %%
# A single coil is used to keep this example small. Multi-coil sensitivity
# multiplication goes after basis expansion and before readout padding.
phantom = bt.phantom([n, n]).reshape(1, n, n) * torch.exp(-2 * grid[:, None, None] ** 2)
coefficients = torch.stack([phantom, 0.3 * phantom * xx]).reshape(image_shape)


def fftc(x, axes):
    return torch.fft.fftshift(
        torch.fft.fftn(torch.fft.ifftshift(x, dim=axes), dim=axes, norm="ortho"), dim=axes
    )


series = torch.einsum("tk,kzyx->tzyx", basis, coefficients.reshape(rank, n, n, n))
reference = fftc(
    response * fftc(functional.pad(series[:, None, None], (padding, padding)), (-1,)),
    (-3, -2),
)
torch.testing.assert_close(A(coefficients), reference, atol=3e-6, rtol=3e-5)
probe = torch.randn(*extended_shape, dtype=torch.complex64)
torch.testing.assert_close(
    torch.vdot(reference.flatten(), probe.flatten()),
    torch.vdot(coefficients.flatten(), A.adjoint(probe).flatten()),
    atol=3e-5,
    rtol=3e-5,
)
lambda_ = 0.01
estimate = A.lstsq(reference, lambda_=lambda_, maxiter=30, tol=1e-7)
# Basis expansion, padding, unit-magnitude modulation and unitary FFTs are
# isometries here, so the ridge solution has this closed form.
torch.testing.assert_close(estimate, coefficients / (1 + lambda_), atol=3e-5, rtol=3e-4)
print("Wave/subspace forward, adjoint and ridge checks passed.")

# %%
# Actual Wave-Shuffling uses an acquired readout table and echo-dependent
# ordering, a calibrated wave response and suitable regularization. Those
# additions change the inverse problem and its conditioning.
fig, axes = plt.subplots(1, 3, figsize=(9, 3), layout="constrained")
for ax, value, title in zip(
    axes,
    [series[0, n // 2].abs(), wave_phase[n // 2], estimate.reshape(rank, n, n, n)[0, n // 2].abs()],
    ["Synthetic echo 1 magnitude", "Hybrid-space phase (rad)", "Recovered coefficient 1"],
):
    im = ax.imshow(value.numpy(), cmap="viridis")
    ax.set_title(title)
    ax.axis("off")
    fig.colorbar(im, ax=ax, shrink=0.7)
plt.show()
