/*
 * torch_prior_nlop.h — C/C++ shared interface for the BART nlop that backs
 * a PyTorch denoiser prior.
 *
 * This header deliberately contains NO `complex float` types so it can be
 * safely included from both C99 sources (torch_prior_nlop.c) and standard
 * C++ sources (torch_prior.cpp).  The complex buffers are exposed only as
 * `float*` (interleaved [re0,im0,re1,im1,...]) at the C/C++ boundary.
 *
 * This design is portable: it works with GCC, Clang, MSVC, and any other
 * compiler that supports standard C and C++.  No GCC-specific extensions
 * (_Complex, __auto_type, __typeof__) appear in this header.
 */

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

/* Forward declaration — avoids including the C99-only nlops/nlop.h here. */
struct nlop_s;

/* --------------------------------------------------------------------------
 * C-compatible callback types used across the C / C++ boundary.
 * --------------------------------------------------------------------------
 *
 * tp_apply_t — denoiser application:
 *   ctx       : opaque pointer set by the C++ caller (TorchPriorEntry*)
 *   numel     : number of complex samples in the image
 *   out       : OUTPUT — D(x) written as 2*numel interleaved floats
 *   in        : INPUT  — image x as 2*numel interleaved floats
 *
 * tp_cleanup_t — called once when the nlop (or registry entry) is freed:
 *   ctx       : same opaque pointer; must release Python ref and free ctx.
 */
typedef void (*tp_apply_t)(void* ctx, long numel,
                            float* out, const float* in);
typedef void (*tp_cleanup_t)(void* ctx);

/* --------------------------------------------------------------------------
 * nlop factory — implemented in torch_prior_nlop.c
 * --------------------------------------------------------------------------
 * Creates a BART nlop_s* whose forward/adjoint steps call `apply`.
 * `cleanup` is invoked exactly once when the nlop is freed (nlop_free).
 *
 *   N    : number of dimensions (must equal DIMS = 16)
 *   dims : array of N longs giving the BART Fortran-order image dimensions
 */
const struct nlop_s* nlop_torch_prior_create(
    tp_apply_t   apply,
    tp_cleanup_t cleanup,
    void*        ctx,
    int          N,
    const long*  dims);

/* --------------------------------------------------------------------------
 * Prior registry — thin fixed-size table implemented in torch_prior_nlop.c
 * --------------------------------------------------------------------------
 * Called from the C++ side (torch_prior.cpp) before / after pics().
 *
 * tp_registry_insert : register a named prior entry.  If `name` already
 *   exists the old entry is cleaned up and replaced.
 * tp_registry_remove : remove (and clean up) the named entry.
 */
void tp_registry_insert(const char*  name,
                        tp_apply_t   apply,
                        tp_cleanup_t cleanup,
                        void*        ctx,
                        int          N,
                        const long*  dims);

void tp_registry_remove(const char* name);

/* --------------------------------------------------------------------------
 * --wrap nlop_tf_create intercept — implemented in torch_prior_nlop.c
 * --------------------------------------------------------------------------
 * The linker replaces every call to nlop_tf_create() with this function.
 * Paths that start with "bartorch://" are handled by looking up the name
 * in the registry and calling nlop_torch_prior_create().  All other paths
 * fall through to __real_nlop_tf_create() — BART's original TF back-end.
 */
const struct nlop_s* __wrap_nlop_tf_create(const char* path);

/* Original BART symbol (renamed by the linker --wrap mechanism). */
extern const struct nlop_s* __real_nlop_tf_create(const char* path);

#ifdef __cplusplus
}
#endif
