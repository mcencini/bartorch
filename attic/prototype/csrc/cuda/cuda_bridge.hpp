/*
 * cuda/cuda_bridge.hpp — Zero-copy bridge for CUDA (GPU) BartTensors.
 *
 * When BART is compiled with USE_CUDA=ON and PyTorch is built with CUDA
 * support, GPU tensors can be passed to BART's CUDA kernels without any
 * host↔device transfer.
 *
 * The key insight: BART's vptr system (src/num/vptr.c) detects whether a
 * pointer lives on device (via cudaPointerGetAttributes) and routes all
 * arithmetic to GPU kernels automatically.  We simply register the CUDA
 * tensor's device pointer the same way we register CPU pointers.
 *
 * Constraints
 * -----------
 * - The CUDA tensor must be on the same GPU device BART was initialised on
 *   (cuda_init() called with that device).
 * - dtype must be complex64 (= cuComplex), Fortran strides.
 * - The tensor must be contiguous in device memory (not a strided view of a
 *   larger allocation, unless BART only reads/writes within the valid region).
 *
 * Functions
 * ---------
 *   is_cuda_bart_compatible(t)  — check device + dtype + strides
 *   register_cuda_input(name,t) — register_mem_cfl_non_managed with dev ptr
 *   sync_after_bart(t)          — cudaDeviceSynchronize() barrier
 */

#pragma once

#ifdef USE_CUDA

#include <torch/extension.h>
#include <string>

extern "C" {
#include "misc/memcfl.h"
#include "num/gpuops.h"
}

#include "../tensor_bridge.hpp"

namespace bartorch {
namespace cuda {

inline bool is_cuda_bart_compatible(const torch::Tensor& t)
{
    return t.is_cuda() && is_bart_compatible(t);
}

/**
 * Register a CUDA tensor's device pointer in BART's in-memory CFL registry.
 * BART will dispatch all operations on this name to GPU kernels automatically
 * (via its vptr device-detection mechanism).
 */
inline void register_cuda_input(const std::string& name, const torch::Tensor& t)
{
    TORCH_CHECK(is_cuda_bart_compatible(t),
                "register_cuda_input: tensor must be CUDA complex64 with "
                "Fortran strides");

    auto dims_i64 = t.sizes().vec();
    std::vector<long> dims(dims_i64.begin(), dims_i64.end());

    memcfl_register(
        name.c_str(),
        static_cast<int>(dims.size()),
        dims.data(),
        // device pointer — BART vptr will recognise this as GPU memory
        reinterpret_cast<std::complex<float>*>(t.data_ptr()),
        /* managed = */ false);
}

inline void sync_after_bart()
{
    // Ensure all BART CUDA kernels have completed before we return the tensor
    // to Python.  Equivalent to torch.cuda.synchronize() but at C level.
    cuda_sync_device();
}

} // namespace cuda
} // namespace bartorch

#endif // USE_CUDA
