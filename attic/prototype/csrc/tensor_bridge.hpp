/*
 * tensor_bridge.hpp — Axis-reversal helpers for torch::Tensor ↔ BART CFL.
 *
 * Design note
 * -----------
 * BART stores arrays in Fortran (column-major) order with shape
 * [d_0, d_1, …, d_{N-1}] where d_0 is the fastest-varying axis ("read").
 *
 * bartorch exposes arrays in C-order with the axes *reversed*:
 * Python shape = [d_{N-1}, …, d_1, d_0].
 *
 * Because a C-order tensor with shape (a, b, c) and a Fortran-order
 * array with shape (c, b, a) share the *identical* byte layout, no copy
 * is needed at the boundary — only the dim vector is reversed.
 *
 * This header contains utility functions used by bartorch_ext.cpp.
 * The actual BART dispatch happens in bartorch_ext.cpp via BART's own
 * internal headers (misc/memcfl.h, etc.).
 */

#pragma once

#include <torch/extension.h>
#include <vector>
#include <string>

namespace bartorch {

// ---------------------------------------------------------------------------
// Stride helpers
// ---------------------------------------------------------------------------

/**
 * Compute column-major (Fortran) strides for the given shape.
 *
 * ``strides[0] = 1``, ``strides[i] = strides[i-1] * dims[i-1]``.
 */
inline std::vector<int64_t> fortran_strides(const std::vector<int64_t>& dims)
{
    std::vector<int64_t> strides(dims.size());
    int64_t s = 1;
    for (size_t i = 0; i < dims.size(); ++i) {
        strides[i] = s;
        s *= dims[i];
    }
    return strides;
}

/**
 * Return true iff ``t`` is complex64 with Fortran (column-major) strides.
 */
inline bool is_bart_compatible(const torch::Tensor& t)
{
    if (t.dtype() != torch::kComplexFloat) return false;
    auto dims     = t.sizes().vec();
    auto expected = fortran_strides(dims);
    auto actual   = t.strides().vec();
    return expected == actual;
}

// ---------------------------------------------------------------------------
// Allocation
// ---------------------------------------------------------------------------

/**
 * Allocate an uninitialised complex64 tensor with Fortran strides on
 * the requested device.
 *
 * The result is the canonical "output slot" used when pre-allocating before
 * calling bart_command() (e.g. for CUDA paths where BART writes directly).
 */
inline torch::Tensor make_bart_tensor(
    const std::vector<int64_t>& dims,
    torch::Device device = torch::kCPU)
{
    auto opts = torch::TensorOptions()
                    .dtype(torch::kComplexFloat)
                    .device(device);
    // Allocate with reversed dims in C-order, then as_strided back.
    std::vector<int64_t> rdims(dims.rbegin(), dims.rend());
    auto storage = torch::empty(rdims, opts).contiguous();
    return storage.as_strided(dims, fortran_strides(dims));
}

} // namespace bartorch
