/*
 * torch_prior_nlop.c — Pure C99 implementation of the BART nlop that backs
 * a PyTorch denoiser prior.
 *
 * All `complex float` usage is confined to this file so that torch_prior.cpp
 * (compiled as C++) never sees the C99 `complex` keyword.  The C/C++
 * boundary is through torch_prior_nlop.h which uses only plain `float*`.
 *
 * Build: compiled as C (not C++) — CMakeLists.txt adds this file to the
 * extension source list; CMake automatically selects the C compiler for
 * .c files.
 */

#include "torch_prior_nlop.h"

#include <stdlib.h>
#include <string.h>

/* BART headers — safe to include in C99 mode. */
#include "misc/types.h"    /* DEF_TYPEID, SET_TYPEID, CAST_DOWN, CAST_UP, PTR_ALLOC, PTR_PASS */
#include "misc/misc.h"     /* xmalloc, xfree, error */
#include "misc/mri.h"      /* DIMS */
#include "num/multind.h"   /* md_calc_size, md_clear, md_alloc, md_free */
#include "nlops/nlop.h"    /* nlop_data_t, nlop_create, nlop_s */

/* CFL_SIZE = sizeof(complex float).  BART defines this in individual .c files
 * (not in any shared header).  Define it here the same way. */
#ifndef CFL_SIZE
#  define CFL_SIZE sizeof(complex float)
#endif

/* ==========================================================================
 * Internal nlop data structure
 * ==========================================================================
 *
 * `super` MUST be the first member — CAST_DOWN / CAST_UP rely on identical
 * base-pointer arithmetic (i.e. &d->super == (nlop_data_t*)d).
 */

struct torch_prior_nlop_s {
    nlop_data_t  super;
    tp_apply_t   apply;        /* C-callable denoiser (from C++ via .h)    */
    tp_cleanup_t cleanup;      /* called once in destructor                 */
    void*        ctx;          /* opaque; typically TorchPriorEntry* in C++ */
    int          N;            /* number of BART dims (DIMS = 16)           */
    long         dims[DIMS];   /* BART Fortran-order image dimensions       */
    complex float* res;        /* cached residual x − D(x), allocated lazily */
};

DEF_TYPEID(torch_prior_nlop_s);

/* --------------------------------------------------------------------------
 * Forward: image → scalar
 *
 * 1. Call apply(ctx, numel, tmp_denoised, src)  →  D(x) in tmp_denoised
 * 2. residual[i] = src[i] - tmp_denoised[i]     →  x − D(x)
 * 3. dst[0] = ||residual||² / 2                 →  debug scalar
 * -------------------------------------------------------------------------- */
static void torch_prior_fwd(const nlop_data_t* _data,
                             complex float*       dst,
                             const complex float* src)
{
    struct torch_prior_nlop_s* d = CAST_DOWN(torch_prior_nlop_s, _data);
    long numel = md_calc_size(d->N, d->dims);

    /* Allocate residual buffer on first use. */
    if (!d->res)
        d->res = (complex float*)md_alloc(d->N, d->dims, CFL_SIZE);

    /* Let the C++ side call the Python denoiser:
     * it writes D(x) into d->res (treated as a float[2*numel] buffer). */
    d->apply(d->ctx, numel, (float*)d->res, (const float*)src);

    /* Compute residual in-place: res = x - D(x). */
    float*       resp = (float*)d->res;
    const float* srcp = (const float*)src;
    for (long i = 0; i < 2 * numel; ++i)
        resp[i] = srcp[i] - resp[i];

    /* Scalar output = ||residual||² / 2 (used by BART for debug logging). */
    double acc = 0.0;
    for (long i = 0; i < 2 * numel; ++i)
        acc += (double)resp[i] * resp[i];
    dst[0] = (float)(acc * 0.5);
}

/* --------------------------------------------------------------------------
 * Forward derivative (linearised, domain → codomain).
 *
 * Not used by prox_nlgrad with grad_nlop=false; return zero.
 * -------------------------------------------------------------------------- */
static void torch_prior_der(const nlop_data_t* _data,
                             int o, int i,
                             complex float*       dst,
                             const complex float* src)
{
    (void)_data; (void)o; (void)i; (void)src;
    dst[0] = 0.f;
}

/* --------------------------------------------------------------------------
 * Adjoint (codomain → domain): returns scale * cached residual x − D(x).
 *
 * In prox_nlgrad, src[0] = {1.0}, so dst = residual = x − D(x).
 * -------------------------------------------------------------------------- */
static void torch_prior_adj(const nlop_data_t* _data,
                             int o, int ii,
                             complex float*       dst,
                             const complex float* src)
{
    (void)o; (void)ii;
    const struct torch_prior_nlop_s* d = CAST_DOWN(torch_prior_nlop_s, _data);
    long numel = md_calc_size(d->N, d->dims);

    if (!d->res) {
        md_clear(d->N, d->dims, dst, CFL_SIZE);
        return;
    }

    complex float scale = src[0];
    for (long j = 0; j < numel; ++j)
        dst[j] = scale * d->res[j];
}

/* --------------------------------------------------------------------------
 * Destructor
 * -------------------------------------------------------------------------- */
static void torch_prior_del(const nlop_data_t* _data)
{
    struct torch_prior_nlop_s* d = CAST_DOWN(torch_prior_nlop_s, _data);
    if (d->cleanup)
        d->cleanup(d->ctx);   /* releases Python ref, frees TorchPriorEntry */
    md_free(d->res);
    xfree(d);
}

/* --------------------------------------------------------------------------
 * Factory
 * -------------------------------------------------------------------------- */
const struct nlop_s* nlop_torch_prior_create(
    tp_apply_t   apply,
    tp_cleanup_t cleanup,
    void*        ctx,
    int          N,
    const long*  dims)
{
    PTR_ALLOC(struct torch_prior_nlop_s, data);
    SET_TYPEID(torch_prior_nlop_s, data);

    data->apply   = apply;
    data->cleanup = cleanup;
    data->ctx     = ctx;
    data->N       = (N <= DIMS) ? N : DIMS;
    data->res     = NULL;

    for (int i = 0; i < DIMS; ++i)
        data->dims[i] = (i < data->N) ? dims[i] : 1L;

    /* Codomain: scalar (1 complex float). */
    const long scalar_dims[1] = { 1L };

    return nlop_create(
        /*OD*/ 1,           scalar_dims,
        /*ID*/ DIMS,        data->dims,
        CAST_UP(PTR_PASS(data)),
        torch_prior_fwd,
        torch_prior_der,
        torch_prior_adj,
        /*normal*/   NULL,
        /*norm_inv*/ NULL,
        torch_prior_del);
}

/* ==========================================================================
 * Prior registry — fixed-size table, no heap allocation, no C++ STL.
 * ==========================================================================
 *
 * Maximum 64 concurrent priors (more than enough for any realistic workflow).
 */

#define TP_REGISTRY_MAX 64

struct tp_entry {
    char         name[256];
    tp_apply_t   apply;
    tp_cleanup_t cleanup;
    void*        ctx;
    int          N;
    long         dims[DIMS];
};

static struct tp_entry g_registry[TP_REGISTRY_MAX];
static int             g_nentries = 0;

void tp_registry_insert(const char*  name,
                        tp_apply_t   apply,
                        tp_cleanup_t cleanup,
                        void*        ctx,
                        int          N,
                        const long*  dims)
{
    /* Overwrite existing entry if the name is already present. */
    for (int k = 0; k < g_nentries; ++k) {
        if (strcmp(g_registry[k].name, name) == 0) {
            /* Clean up the old entry first. */
            if (g_registry[k].cleanup)
                g_registry[k].cleanup(g_registry[k].ctx);
            g_registry[k].apply   = apply;
            g_registry[k].cleanup = cleanup;
            g_registry[k].ctx     = ctx;
            g_registry[k].N       = N;
            for (int i = 0; i < DIMS; ++i)
                g_registry[k].dims[i] = (i < N) ? dims[i] : 1L;
            return;
        }
    }

    if (g_nentries < TP_REGISTRY_MAX) {
        struct tp_entry* e = &g_registry[g_nentries++];
        strncpy(e->name, name, sizeof(e->name) - 1);
        e->name[sizeof(e->name) - 1] = '\0';
        e->apply   = apply;
        e->cleanup = cleanup;
        e->ctx     = ctx;
        e->N       = N;
        for (int i = 0; i < DIMS; ++i)
            e->dims[i] = (i < N) ? dims[i] : 1L;
    } else {
        error("bartorch: torch_prior registry full (max %d entries)\n",
              TP_REGISTRY_MAX);
    }
}

void tp_registry_remove(const char* name)
{
    for (int k = 0; k < g_nentries; ++k) {
        if (strcmp(g_registry[k].name, name) == 0) {
            if (g_registry[k].cleanup)
                g_registry[k].cleanup(g_registry[k].ctx);
            /* Swap-and-pop to keep the table compact. */
            g_registry[k] = g_registry[--g_nentries];
            return;
        }
    }
}

/* ==========================================================================
 * --wrap nlop_tf_create intercept
 * ==========================================================================
 *
 * The linker replaces every call to nlop_tf_create() with this function.
 * Paths starting with "bartorch://" are resolved via the registry above.
 * All other paths fall through to the original BART TF implementation
 * (__real_nlop_tf_create), which will fail unless BART was compiled with
 * TensorFlow support — matching the original behaviour.
 */

const struct nlop_s* __wrap_nlop_tf_create(const char* path)
{
    static const char   kPrefix[]   = "bartorch://";
    static const size_t kPrefixLen  = sizeof(kPrefix) - 1;

    if (strncmp(path, kPrefix, kPrefixLen) == 0) {
        const char* name = path + kPrefixLen;
        for (int k = 0; k < g_nentries; ++k) {
            if (strcmp(g_registry[k].name, name) == 0) {
                struct tp_entry* e = &g_registry[k];
                return nlop_torch_prior_create(e->apply, e->cleanup, e->ctx,
                                               e->N, e->dims);
            }
        }
        error("bartorch: torch_prior '%s' not found in registry — "
              "was bt.pics() called with torch_prior= set?\n", name);
    }

    /* Fall through to BART's real TF implementation. */
    return __real_nlop_tf_create(path);
}
