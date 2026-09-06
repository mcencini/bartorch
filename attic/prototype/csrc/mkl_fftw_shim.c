/*
 * mkl_fftw_shim.c — intercept BART's fftwf_* calls and route them through MKL DFTI.
 *
 * MKL ships a partial FFTW3 compatibility layer in libmkl_rt, but its
 * fftwf_plan_guru64_dft implementation rejects any call with l > 1 howmany
 * dimensions.  BART uses l = D − k (up to 15 for DIMS=16 arrays), so every
 * FFT-based BART command silently fails with the MKL FFTW3 compat wrapper.
 *
 * This shim intercepts fftwf_plan_guru64_dft / fftwf_execute_dft /
 * fftwf_destroy_plan (and the no-op wisdom + thread helpers) via the GNU
 * linker --wrap mechanism and re-implements them on top of MKL's native DFTI
 * interface, which has no such restriction.
 *
 * Link with:
 *   -Wl,--wrap,fftwf_plan_guru64_dft
 *   -Wl,--wrap,fftwf_execute_dft
 *   -Wl,--wrap,fftwf_destroy_plan
 *   -Wl,--wrap,fftwf_export_wisdom_to_filename
 *   -Wl,--wrap,fftwf_import_wisdom_from_filename
 *   -Wl,--wrap,fftwf_init_threads
 *   -Wl,--wrap,fftwf_plan_with_nthreads
 *   -Wl,--wrap,fftwf_cleanup_threads
 *
 * MKL DFTI handles threading natively; no per-plan thread count is needed.
 *
 * Contiguous-buffer strategy
 * --------------------------
 * MKL DFTI silently produces wrong results when given non-standard stride
 * patterns (e.g. ecalib's compute_imgcov uses stride-permuted FFTs with
 * output strides like {132, 1584} instead of contiguous {1, 12}).
 *
 * To ensure correctness for all stride patterns:
 * - The DFTI descriptor ALWAYS uses contiguous strides.
 * - At execute time, we gather non-contiguous input into a contiguous temp
 *   buffer, execute DFTI on contiguous buffers, then scatter the contiguous
 *   output back to the non-contiguous destination.
 * - For the common case (both input and output are contiguous), the
 *   gather/scatter is skipped and DFTI operates directly on the user buffers.
 *
 * In-place transforms (in == out):
 * BART's fftmod calls fftwf_execute_dft(plan, ptr, ptr) with in==out.
 * MKL DFTI with DFTI_NOT_INPLACE forbids aliased buffers.  We always
 * use a temp input buffer when in==out (via gather or memcpy).
 *
 * Batched transforms (multi-dimensional howmany):
 * BART can pass multiple howmany dimensions with n > 1.  We handle this
 * by iterating over ALL howmany index combinations, computing the correct
 * input/output offsets from the per-dimension strides.
 */

#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <stdio.h>
#include <assert.h>
#include <complex.h>

#include <mkl_dfti.h>

/* ─── FFTW3 type stubs (we don't link against real FFTW3) ──────────────── */

typedef struct {
    MKL_LONG          n;
    MKL_LONG          is;
    MKL_LONG          os;
} _bt_iodim64;

/* Howmany dimension descriptor */
typedef struct {
    MKL_LONG n;   /* size */
    MKL_LONG is;  /* input stride */
    MKL_LONG os;  /* output stride */
} _bt_hmdim;

/* Internal plan struct that wraps a DFTI descriptor */
typedef struct {
    DFTI_DESCRIPTOR_HANDLE  desc;
    int                     sign;         /* FFTW_FORWARD=-1, FFTW_BACKWARD=+1 */
    MKL_LONG                ntrans;       /* total number of transforms */
    int                     l;            /* number of howmany dimensions */
    _bt_hmdim              *hmdims;       /* [l] howmany dimensions */
    int                     k;            /* number of transform dimensions */
    MKL_LONG               *lengths;      /* [k] transform dim sizes */
    MKL_LONG               *istrides;     /* [k] input element strides */
    MKL_LONG               *ostrides;     /* [k] output element strides */
    size_t                  nel;          /* product of transform dim sizes */
    int                     need_gather;  /* input strides are non-contiguous */
    int                     need_scatter; /* output strides are non-contiguous */
} _bt_plan;

/* ─── Helpers ──────────────────────────────────────────────────────────── */

static inline void _bt_check(MKL_LONG status, const char *where)
{
    if (status != DFTI_NO_ERROR) {
        fprintf(stderr, "bartorch mkl_fftw_shim: DFTI error in %s: %s\n",
                where, DftiErrorMessage(status));
        abort();
    }
}

/*
 * Check if strides form a contiguous innermost-first layout:
 *   strides[0] == 1
 *   strides[j] == product(lengths[0..j-1])  for j > 0
 */
static int _is_contiguous(int k, const MKL_LONG *lengths, const MKL_LONG *strides)
{
    MKL_LONG expected = 1;
    for (int i = 0; i < k; i++) {
        if (strides[i] != expected) return 0;
        expected *= lengths[i];
    }
    return 1;
}

/*
 * Gather from strided layout to contiguous buffer.
 * For each element in contiguous order, compute the strided offset.
 */
static void _gather(float _Complex *dst, const float _Complex *src,
                    int k, const MKL_LONG *lengths, const MKL_LONG *strides,
                    size_t nel)
{
    for (size_t idx = 0; idx < nel; idx++) {
        size_t rem = idx;
        MKL_LONG offset = 0;
        for (int d = 0; d < k; d++) {
            MKL_LONG coord = (MKL_LONG)(rem % (size_t)lengths[d]);
            rem /= (size_t)lengths[d];
            offset += coord * strides[d];
        }
        dst[idx] = src[offset];
    }
}

/*
 * Scatter from contiguous buffer to strided layout.
 */
static void _scatter(float _Complex *dst, const float _Complex *src,
                     int k, const MKL_LONG *lengths, const MKL_LONG *strides,
                     size_t nel)
{
    for (size_t idx = 0; idx < nel; idx++) {
        size_t rem = idx;
        MKL_LONG offset = 0;
        for (int d = 0; d < k; d++) {
            MKL_LONG coord = (MKL_LONG)(rem % (size_t)lengths[d]);
            rem /= (size_t)lengths[d];
            offset += coord * strides[d];
        }
        dst[offset] = src[idx];
    }
}

/*
 * Compute the input and output base offsets for transform index `tr`.
 * The howmany dimensions form a multi-index: for flat index `tr`,
 * decompose into (j0, j1, ..., j_{l-1}) and sum j_d * stride_d.
 */
static void _compute_offsets(MKL_LONG tr, int l, const _bt_hmdim *hm,
                             MKL_LONG *in_off, MKL_LONG *out_off)
{
    MKL_LONG ioff = 0, ooff = 0;
    MKL_LONG rem = tr;
    for (int d = 0; d < l; d++) {
        MKL_LONG coord = rem % hm[d].n;
        rem /= hm[d].n;
        ioff += coord * hm[d].is;
        ooff += coord * hm[d].os;
    }
    *in_off  = ioff;
    *out_off = ooff;
}

/* ─── Main plan creation shim ──────────────────────────────────────────── */

void *__wrap_fftwf_plan_guru64_dft(
    int                    k,
    const _bt_iodim64     *dims,
    int                    l,
    const _bt_iodim64     *hmdims,
    float _Complex        *in,
    float _Complex        *out,
    int                    sign,
    unsigned               flags)
{
    (void)in; (void)out; (void)flags;

    if (k <= 0) return NULL;

    /* ── Compute total number of transforms ────────────────────────────── */
    MKL_LONG ntrans = 1;
    for (int i = 0; i < l; i++)
        ntrans *= (MKL_LONG)hmdims[i].n;

    /* ── Store howmany dimensions ──────────────────────────────────────── */
    _bt_hmdim *hm = NULL;
    if (l > 0) {
        hm = (_bt_hmdim *)malloc((size_t)l * sizeof(_bt_hmdim));
        if (!hm) return NULL;
        for (int i = 0; i < l; i++) {
            hm[i].n  = (MKL_LONG)hmdims[i].n;
            hm[i].is = (MKL_LONG)hmdims[i].is;
            hm[i].os = (MKL_LONG)hmdims[i].os;
        }
    }

    /* ── Extract dim sizes and strides from FFTW dims ──────────────────── */
    MKL_LONG *lengths  = (MKL_LONG *)malloc((size_t)k * sizeof(MKL_LONG));
    MKL_LONG *istrides = (MKL_LONG *)malloc((size_t)k * sizeof(MKL_LONG));
    MKL_LONG *ostrides = (MKL_LONG *)malloc((size_t)k * sizeof(MKL_LONG));

    if (!lengths || !istrides || !ostrides) {
        free(lengths); free(istrides); free(ostrides); free(hm);
        return NULL;
    }

    size_t nel = 1;
    for (int i = 0; i < k; i++) {
        lengths[i]  = (MKL_LONG)dims[i].n;
        istrides[i] = (MKL_LONG)dims[i].is;
        ostrides[i] = (MKL_LONG)dims[i].os;
        nel *= (size_t)dims[i].n;
    }

    int need_gather  = !_is_contiguous(k, lengths, istrides);
    int need_scatter = !_is_contiguous(k, lengths, ostrides);

    /* ── Build DFTI descriptor with CONTIGUOUS strides ─────────────────── */
    MKL_LONG *contig_strides = (MKL_LONG *)malloc((size_t)(k + 1) * sizeof(MKL_LONG));
    if (!contig_strides) {
        free(lengths); free(istrides); free(ostrides); free(hm);
        return NULL;
    }
    contig_strides[0] = 0;  /* first-element offset */
    MKL_LONG stride = 1;
    for (int i = 0; i < k; i++) {
        contig_strides[i + 1] = stride;
        stride *= lengths[i];
    }

    DFTI_DESCRIPTOR_HANDLE desc = NULL;
    MKL_LONG status;

    if (k == 1) {
        status = DftiCreateDescriptor(&desc, DFTI_SINGLE, DFTI_COMPLEX,
                                      (MKL_LONG)1, lengths[0]);
    } else {
        status = DftiCreateDescriptor(&desc, DFTI_SINGLE, DFTI_COMPLEX,
                                      (MKL_LONG)k, lengths);
    }

    if (status != DFTI_NO_ERROR) {
        free(lengths); free(istrides); free(ostrides);
        free(contig_strides); free(hm);
        return NULL;
    }

    /* Set contiguous strides for both input and output */
    status = DftiSetValue(desc, DFTI_INPUT_STRIDES, contig_strides);
    if (status != DFTI_NO_ERROR) {
        free(lengths); free(istrides); free(ostrides);
        free(contig_strides); free(hm);
        DftiFreeDescriptor(&desc);
        return NULL;
    }
    status = DftiSetValue(desc, DFTI_OUTPUT_STRIDES, contig_strides);
    free(contig_strides);
    if (status != DFTI_NO_ERROR) {
        free(lengths); free(istrides); free(ostrides); free(hm);
        DftiFreeDescriptor(&desc);
        return NULL;
    }

    /* Always out-of-place */
    status = DftiSetValue(desc, DFTI_PLACEMENT, DFTI_NOT_INPLACE);
    if (status != DFTI_NO_ERROR) {
        free(lengths); free(istrides); free(ostrides); free(hm);
        DftiFreeDescriptor(&desc);
        return NULL;
    }

    /* Commit */
    status = DftiCommitDescriptor(desc);
    if (status != DFTI_NO_ERROR) {
        free(lengths); free(istrides); free(ostrides); free(hm);
        DftiFreeDescriptor(&desc);
        return NULL;
    }

    /* ── Wrap in our plan struct ────────────────────────────────────────── */
    _bt_plan *plan = (_bt_plan *)malloc(sizeof(_bt_plan));
    if (!plan) {
        free(lengths); free(istrides); free(ostrides); free(hm);
        DftiFreeDescriptor(&desc);
        return NULL;
    }
    plan->desc         = desc;
    plan->sign         = sign;
    plan->ntrans       = ntrans;
    plan->l            = l;
    plan->hmdims       = hm;
    plan->k            = k;
    plan->lengths      = lengths;
    plan->istrides     = istrides;
    plan->ostrides     = ostrides;
    plan->nel          = nel;
    plan->need_gather  = need_gather;
    plan->need_scatter = need_scatter;
    return (void *)plan;
}

/* ─── Execute shim ─────────────────────────────────────────────────────── */

void __wrap_fftwf_execute_dft(void *_plan, float _Complex *in, float _Complex *out)
{
    _bt_plan *plan = (_bt_plan *)_plan;
    assert(plan && plan->desc);

    size_t nel = plan->nel;
    int inplace = (in == out);

    /*
     * Determine if we need temp buffers:
     * - temp_in:  needed if input strides are non-contiguous OR in-place
     * - temp_out: needed if output strides are non-contiguous
     */
    int use_temp_in  = plan->need_gather || inplace;
    int use_temp_out = plan->need_scatter;

    /*
     * In-place hazard: when in == out and the output strides differ from
     * the input strides, scattering the output of sub-transform i can
     * overwrite input data needed by sub-transform j (j > i).
     *
     * Fix: when in == out, copy the ENTIRE input buffer upfront so all
     * subsequent gathers read from the safe copy.  We compute the total
     * span of all transforms (max offset over all howmany combinations
     * plus the per-transform span).
     */
    float _Complex *in_safe = in;  /* points to safe (unmodified) input */
    float _Complex *in_copy = NULL;
    if (inplace && plan->ntrans > 0) {
        /* Compute total input span covering all sub-transforms */
        MKL_LONG max_hm_off = 0;
        for (int d = 0; d < plan->l; d++) {
            if (plan->hmdims[d].n > 1)
                max_hm_off += (plan->hmdims[d].n - 1) * plan->hmdims[d].is;
        }
        MKL_LONG max_xf_off = 0;
        for (int d = 0; d < plan->k; d++)
            max_xf_off += (plan->lengths[d] - 1) * plan->istrides[d];

        size_t total_span = (size_t)(max_hm_off + max_xf_off) + 1;
        in_copy = (float _Complex *)malloc(total_span * sizeof(float _Complex));
        if (!in_copy) {
            fprintf(stderr, "bartorch mkl_fftw_shim: alloc failed (in_copy, %zu)\n", total_span);
            abort();
        }
        memcpy(in_copy, in, total_span * sizeof(float _Complex));
        in_safe = in_copy;
    }

    float _Complex *temp_in  = NULL;
    float _Complex *temp_out = NULL;

    if (use_temp_in) {
        temp_in = (float _Complex *)malloc(nel * sizeof(float _Complex));
        if (!temp_in) {
            fprintf(stderr, "bartorch mkl_fftw_shim: alloc failed (temp_in, %zu)\n", nel);
            abort();
        }
    }
    if (use_temp_out) {
        temp_out = (float _Complex *)malloc(nel * sizeof(float _Complex));
        if (!temp_out) {
            free(temp_in); free(in_copy);
            fprintf(stderr, "bartorch mkl_fftw_shim: alloc failed (temp_out, %zu)\n", nel);
            abort();
        }
    }

    MKL_LONG status = DFTI_NO_ERROR;

    for (MKL_LONG tr = 0; tr < plan->ntrans; tr++) {
        /* Compute offsets for this sub-transform using all howmany dims */
        MKL_LONG in_off = 0, out_off = 0;
        _compute_offsets(tr, plan->l, plan->hmdims, &in_off, &out_off);

        float _Complex *in_tr  = in_safe + in_off;
        float _Complex *out_tr = out     + out_off;

        float _Complex *dfti_in;
        float _Complex *dfti_out;

        /* Prepare DFTI input (must be contiguous) */
        if (use_temp_in) {
            if (plan->need_gather || inplace) {
                _gather(temp_in, in_tr, plan->k, plan->lengths,
                        plan->istrides, nel);
            } else {
                memcpy(temp_in, in_tr, nel * sizeof(float _Complex));
            }
            dfti_in = temp_in;
        } else {
            dfti_in = in_tr;
        }

        /* Prepare DFTI output (must be contiguous) */
        if (use_temp_out) {
            dfti_out = temp_out;
        } else {
            dfti_out = out_tr;
        }

        /* Execute single transform */
        if (plan->sign == -1)
            status = DftiComputeForward(plan->desc, (void *)dfti_in, (void *)dfti_out);
        else
            status = DftiComputeBackward(plan->desc, (void *)dfti_in, (void *)dfti_out);
        if (status != DFTI_NO_ERROR) break;

        /* Scatter output if needed */
        if (use_temp_out) {
            _scatter(out_tr, temp_out, plan->k, plan->lengths,
                     plan->ostrides, nel);
        }
    }

    free(temp_in);
    free(temp_out);
    free(in_copy);

    _bt_check(status, "fftwf_execute_dft");
}

/* ─── Destroy shim ─────────────────────────────────────────────────────── */

void __wrap_fftwf_destroy_plan(void *_plan)
{
    if (!_plan) return;
    _bt_plan *plan = (_bt_plan *)_plan;
    DftiFreeDescriptor(&plan->desc);
    free(plan->lengths);
    free(plan->istrides);
    free(plan->ostrides);
    free(plan->hmdims);
    free(plan);
}

/* ─── No-op stubs for wisdom and thread helpers ────────────────────────── */

int __wrap_fftwf_export_wisdom_to_filename(const char *filename)
{
    (void)filename;
    return 1; /* FFTW_SUCCESS */
}

int __wrap_fftwf_import_wisdom_from_filename(const char *filename)
{
    (void)filename;
    return 0; /* FFTW_FAILURE — no wisdom, BART ignores this */
}

int __wrap_fftwf_init_threads(void)
{
    return 1; /* success */
}

void __wrap_fftwf_plan_with_nthreads(int n)
{
    (void)n; /* MKL DFTI uses its own threading */
}

void __wrap_fftwf_cleanup_threads(void)
{
    /* no-op */
}
