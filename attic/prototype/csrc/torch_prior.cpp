/*
 * torch_prior.cpp — C++ side of the PyTorch denoiser prior for BART PICS.
 *
 * This file contains ONLY:
 *   • The C++ prior registry (std::unordered_map keyed by name)
 *   • The C-linkage denoiser callback `cpp_prior_apply` and its cleanup
 *     `cpp_prior_cleanup` — these bridge torch_prior_nlop.c ↔ PyTorch/GIL
 *   • pybind11 bindings: register_torch_prior / unregister_torch_prior
 *
 * All `complex float` usage and BART nlop types live exclusively in
 * torch_prior_nlop.c (compiled as C99).  The interface between the two
 * translation units is the plain-C API in torch_prior_nlop.h, which uses
 * only `float*` (interleaved [re,im,...]) and opaque `void*` — no C99 or
 * GCC-specific types, so this file builds cleanly on GCC, Clang, and MSVC.
 *
 * Strategy (identical to finufft_grid.cpp for grid2/grid2H):
 *
 *   By linking with
 *     Linux : -Wl,--wrap,nlop_tf_create
 *     macOS : -Wl,-wrap,_nlop_tf_create
 *   every call to nlop_tf_create() inside the linked binary is redirected to
 *   __wrap_nlop_tf_create() in torch_prior_nlop.c.  When the path argument
 *   starts with "bartorch://", the function looks up the registry populated
 *   by register_torch_prior() and returns a custom BART nlop whose
 *   forward/adjoint steps invoke the Python denoiser via cpp_prior_apply.
 *
 * Usage from Python (via pics() in _commands.py):
 *
 *   1. Python calls _ext.register_torch_prior(name, fn, bart_img_dims).
 *   2. Python appends R='TF:{bartorch://name}:lambda' to the BART pics flags.
 *   3. BART's optreg.c TENFL case calls nlop_tf_create("bartorch://name").
 *   4. __wrap_nlop_tf_create (torch_prior_nlop.c) intercepts, builds the nlop.
 *   5. BART's prox_nlgrad_create wraps the nlop in a proximal operator.
 *   6. Each ADMM/IST/FISTA iteration calls the proximal operator, which
 *      calls nlop_apply on our nlop, which calls cpp_prior_apply, which
 *      acquires the GIL and invokes the Python denoiser.
 *
 * Denoiser convention (pure PnP, no sigma argument):
 *
 *   fn : torch.Tensor (complex64, shape [numel]) → torch.Tensor (same shape)
 *
 * nlop semantics (grad_nlop=false, one gradient step):
 *
 *   forward(x)  → ‖x − D(x)‖² / 2   (caches residual)
 *   adjoint(s)  → s[0] · (x − D(x))  (returns scaled residual)
 *
 *   prox_nlgrad_create(nlop, steps=1, stepsize=1.0, lambda, false):
 *     prox(z, mu) = z − mu·lambda·(z − D(z))
 *   With lambda=1, mu=1 (ADMM rho=1):  prox(z) = D(z)  ✓
 */

#include <Python.h>

#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

/* Shared C/C++ interface — no complex float, no GCC-specific types. */
#include "torch_prior_nlop.h"

namespace py = pybind11;

/* ==========================================================================
 * C++ prior entry
 * ==========================================================================
 *
 * Heap-allocated; ownership is shared between the registry and the nlop:
 *   • While pics() runs: the nlop holds a reference via tp_cleanup_t.
 *   • After pics(): bartorch_unregister_torch_prior() cleans it up via the
 *     registry's tp_registry_remove(), which calls cpp_prior_cleanup().
 *
 * Because the same entry can be referenced by both the registry and a live
 * nlop, we use a simple reference count (no shared_ptr to avoid C++ ABI
 * issues at the C boundary).
 */
struct TorchPriorEntry {
    PyObject*         fn_obj;  /* Python callable — owned reference     */
    std::vector<long> dims;    /* BART Fortran-order image dims (len 16) */
};

/* ==========================================================================
 * C-linkage callbacks invoked by torch_prior_nlop.c
 * ==========================================================================
 *
 * cpp_prior_apply — called on every nlop forward pass.
 *   ctx      : TorchPriorEntry*
 *   numel    : number of complex samples
 *   out      : OUTPUT — D(x), 2*numel interleaved floats
 *   in       : INPUT  — image x, 2*numel interleaved floats
 */
extern "C" void cpp_prior_apply(void* ctx, long numel,
                                float* out, const float* in)
{
    auto* entry = static_cast<TorchPriorEntry*>(ctx);

    /* Wrap the input buffer as a complex64 tensor — zero copy. */
    const auto opts = torch::TensorOptions().dtype(torch::kComplexFloat);
    auto x_t = torch::from_blob(const_cast<float*>(in), {numel}, opts);

    /* Call the Python denoiser under the GIL. */
    torch::Tensor dx_t;
    {
        py::gil_scoped_acquire gil;
        py::object fn = py::reinterpret_borrow<py::object>(entry->fn_obj);
        dx_t = fn(x_t).cast<torch::Tensor>();
    }

    /* Ensure the result is contiguous complex64 on CPU. */
    dx_t = dx_t.contiguous().to(torch::kComplexFloat).cpu();

    /* Copy D(x) into the output buffer as interleaved floats. */
    const float* dx_ptr =
        reinterpret_cast<const float*>(dx_t.data_ptr<c10::complex<float>>());
    std::memcpy(out, dx_ptr, static_cast<std::size_t>(2 * numel) * sizeof(float));
}

/*
 * cpp_prior_cleanup — called by torch_prior_nlop.c when the entry is freed
 * (either from tp_registry_remove or from torch_prior_del).
 */
extern "C" void cpp_prior_cleanup(void* ctx)
{
    auto* entry = static_cast<TorchPriorEntry*>(ctx);
    {
        py::gil_scoped_acquire gil;
        Py_DECREF(entry->fn_obj);
    }
    delete entry;
}

/* ==========================================================================
 * pybind11 bindings — called from _commands.py before / after pics()
 * ========================================================================== */

/*
 * Register a Python denoiser callable with the given name and image dims.
 * Must be called before the BART pics command runs.
 */
void bartorch_register_torch_prior(const std::string& name,
                                   py::object        fn,
                                   std::vector<long> dims)
{
    auto* entry = new TorchPriorEntry{ fn.ptr(), std::move(dims) };
    Py_INCREF(entry->fn_obj);

    tp_registry_insert(name.c_str(),
                       cpp_prior_apply,
                       cpp_prior_cleanup,
                       entry,
                       static_cast<int>(entry->dims.size()),
                       entry->dims.data());
}

/*
 * Remove the named prior from the registry.  The cleanup callback is
 * invoked by tp_registry_remove(), releasing the Python ref and freeing
 * the TorchPriorEntry.
 */
void bartorch_unregister_torch_prior(const std::string& name)
{
    tp_registry_remove(name.c_str());
}

