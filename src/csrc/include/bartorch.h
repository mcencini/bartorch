/*
 * bartorch C ABI.
 *
 * This is the only interface the host language sees.  It is plain C:
 * no complex types, no variable-length arrays, no GNU extensions, so it
 * can be consumed by ctypes, by an MSVC-built extension or by any C++
 * translation unit.  Every array is described by a data pointer and a
 * BART-order (Fortran, first index fastest) dimension vector; the host
 * owns every buffer it registers and every buffer the allocator callback
 * hands out.
 */
#ifndef BARTORCH_H
#define BARTORCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#define BARTORCH_API __attribute__((visibility("default")))

/* Number of dimensions BART carries for every array. */
#define BARTORCH_DIMS 16

/* Log levels, matching BART's enum debug_levels. */
enum bartorch_log_level {
	BARTORCH_LOG_ERROR = 0,
	BARTORCH_LOG_WARN = 1,
	BARTORCH_LOG_INFO = 2,
	BARTORCH_LOG_DEBUG1 = 3,
	BARTORCH_LOG_DEBUG2 = 4,
	BARTORCH_LOG_DEBUG3 = 5,
	BARTORCH_LOG_DEBUG4 = 6,
	BARTORCH_LOG_TRACE = 7,
};

/*
 * Allocate a complex-float array of the given BART dimensions and return
 * its data pointer.  The host keeps the buffer alive until the matching
 * free callback receives the same pointer.  Returning NULL aborts the
 * running command with an error.
 */
typedef void* (*bartorch_alloc_fn)(void* ctx, int D, const long* dims);
typedef void (*bartorch_free_fn)(void* ctx, void* data);

/* Receives every BART log line at or below the current debug level. */
typedef void (*bartorch_log_fn)(void* ctx, int level, const char* func,
		const char* file, int line, const char* msg);

BARTORCH_API const char* bartorch_bart_version(void);
BARTORCH_API const char* bartorch_build_info(void);

BARTORCH_API void bartorch_set_allocator(bartorch_alloc_fn alloc, bartorch_free_fn free_, void* ctx);
BARTORCH_API void bartorch_set_log_handler(bartorch_log_fn fn, void* ctx);
BARTORCH_API void bartorch_set_debug_level(int level);
BARTORCH_API int bartorch_get_debug_level(void);
BARTORCH_API void bartorch_set_num_threads(int n);

/*
 * In-memory array registry.  A name ending in ".mem" given to any BART
 * command resolves here instead of on disk.  Registered arrays are never
 * freed by BART; arrays BART creates itself come from the allocator
 * callback and are released through the free callback on unlink.
 */
BARTORCH_API int bartorch_register(const char* name, int D, const long* dims, void* data);
BARTORCH_API int bartorch_exists(const char* name);
BARTORCH_API int bartorch_lookup(const char* name, int D, long* dims, void** data);
BARTORCH_API int bartorch_unlink(const char* name);
BARTORCH_API int bartorch_unlink_all(void);

/*
 * Run one BART command in-process.  argv[0] is the tool name.  Text the
 * tool prints to standard output lands in `out`; the last error message
 * lands in `err`.  Returns the tool's exit code.
 */
BARTORCH_API int bartorch_command(int argc, const char* const* argv,
		char* out, size_t outlen, char* err, size_t errlen);

/*
 * BLAS and LAPACK backend.  Each entry is a Fortran-ABI routine
 * ("cgemm_", "cgesdd_", ...): every argument by pointer, character
 * arguments followed by hidden size_t lengths.  An entry may be the
 * address of a symbol exported by a library already in the process or a
 * callback implemented by the host.  Passing NULL restores the built-in
 * reference implementation where one exists.
 */
BARTORCH_API int bartorch_backend_set(const char* symbol, void* fn);
BARTORCH_API int bartorch_backend_count(void);
BARTORCH_API const char* bartorch_backend_name(int index);
BARTORCH_API int bartorch_backend_has_fallback(int index);

/*
 * FFT.  BART plans through the FFTW guru interface, which is served by MKL
 * where the process has it and by a compiled-in transform otherwise.  MKL is
 * reached through DFTI, its own interface: the FFTW one it also publishes
 * refuses more than one loop dimension, and BART passes one per dimension it
 * is not transforming.  Give this the DFTI entry points, from the same
 * library the BLAS table was filled from.
 */
/*
 * SENSE.  A coil-by-coil loop inside the operator keeps a slab of coils
 * resident rather than the whole bank, and shrinks the doubled grid the
 * Toeplitz normal convolves on by the same factor.  The slab is how many
 * coils at once: larger is faster and larger, zero leaves BART its own
 * operator over every coil at once.
 */
/* Keep the function a Toeplitz normal convolves with off the card, and bring
 * it over one set of frequencies at a time.  On by default: it costs BART's
 * low-memory normal, which walks the sets rather than convolving them at
 * once. */
BARTORCH_API void bartorch_nufft_set_stream_psf(int enable);
BARTORCH_API int bartorch_nufft_stream_psf(void);

/* Keep only the places the samples reach of the function a Toeplitz normal
 * convolves with.  On by default, wherever there is a pattern to say where
 * they reached. */
BARTORCH_API void bartorch_nufft_set_compress_psf(int enable);
BARTORCH_API int bartorch_nufft_compress_psf(void);

/* Bring the set of frequencies that will be wanted next over while the card
 * convolves the one it has.  It costs a second slot on the card, which is one
 * set of frequencies, and page-locks the function on the host; what it buys is
 * the crossing, which is most of what is left in a normal once the function is
 * compressed. */
BARTORCH_API void bartorch_nufft_set_overlap_psf(int enable);
BARTORCH_API int bartorch_nufft_overlap_psf(void);

/* Let the device's transform pair go at the first normal: with a Toeplitz
 * function built a normal reads neither the plans nor the points, and a
 * transform asked for afterwards plans again.  On by default. */
/* Whether a real upper-triangular contraction runs in bartorch's kernel or in
 * BART's, which is kept to be held against. */
BARTORCH_API void bartorch_nufft_set_contraction_kernel(int enable);
BARTORCH_API void bartorch_nufft_set_release_transforms(int enable);
BARTORCH_API int bartorch_nufft_release_transforms(void);
BARTORCH_API void bartorch_nufft_set_fft_callbacks(int enable);
BARTORCH_API int bartorch_nufft_fft_callbacks(void);
BARTORCH_API void bartorch_nufft_set_paired(int enable);
BARTORCH_API int bartorch_nufft_paired(void);
BARTORCH_API int bartorch_nufft_paired_built(void);
BARTORCH_API void bartorch_nufft_set_bf16(int enable);
BARTORCH_API int bartorch_nufft_bf16(void);

/* Host memory a copy engine can read directly, so an asynchronous copy out of
 * it is one.  Ordinary memory when it is not asked for, or where there is no
 * card: page-locking is not free, and only a crossing that overlaps something
 * repays it. */
BARTORCH_API void* bartorch_host_alloc(long size, int pinned);
BARTORCH_API void bartorch_host_free(void* ptr);

/* A stream of its own for bringing a function over, so the set that will be
 * wanted next crosses while the card convolves the one it has.  Two slots:
 * `copy` fills one once the card has released it, `wait` holds BART's stream
 * until it has arrived, `release` says BART is done reading it. */
BARTORCH_API int bartorch_cuda_stage_open(void** stage);
BARTORCH_API void bartorch_cuda_stage_close(void* stage);
BARTORCH_API int bartorch_cuda_stage_copy(void* stage, int slot, void* dst, const void* src, long size);
BARTORCH_API int bartorch_cuda_stage_wait(void* stage, int slot);
BARTORCH_API int bartorch_cuda_stage_release(void* stage, int slot);

/* A copy between the card and pageable host memory through two page-locked
 * buffers, so it runs near the bus's rate even into pages never touched. */
BARTORCH_API int bartorch_cuda_copy_pageable(void* dst, const void* src, long size);
BARTORCH_API void* bartorch_host_prefault_begin(void* ptr, long size);
BARTORCH_API void bartorch_host_prefault_end(void* handle);

/* Page-lock, and release, host memory that already exists. */
BARTORCH_API int bartorch_cuda_host_register(void* ptr, long size);
BARTORCH_API void bartorch_cuda_host_unregister(void* ptr);

BARTORCH_API void bartorch_sense_set_coil_batch(int coils);
BARTORCH_API int bartorch_sense_coil_batch(void);

/* Apply the sensitivity inside the transform, rather than making a coil image
 * to multiply it into and another for the answer to land in.  On by default,
 * wherever the transform reads and writes a coefficient at a time. */
BARTORCH_API void bartorch_sense_set_fold_maps(int enable);
BARTORCH_API int bartorch_sense_fold_maps(void);
/* Operators built since the last reset: 0 with the coil loop, 1 as BART's
 * own chain because the arrangement could not be sliced. */
BARTORCH_API long bartorch_sense_counter(int which);
BARTORCH_API void bartorch_sense_reset_counters(void);

BARTORCH_API int bartorch_fft_set(const char* symbol, void* fn);
BARTORCH_API int bartorch_fft_usable(void);
/* Plans built since the last reset: 0 by MKL, 1 by the built-in transform. */
BARTORCH_API long bartorch_fft_counter(int which);
BARTORCH_API void bartorch_fft_reset_counters(void);

/*
 * Operators.  A handle wraps one BART linear or nonlinear operator.  Both
 * kinds can be built from host callbacks, in which case the host owns the
 * memory behind the callbacks' pointers for the duration of each call, and
 * both can be chained with each other and solved against.  Dimension
 * vectors are BART order; every callback receives raw complex-float buffers
 * sized by the dimensions the operator was created with and returns 0 on
 * success.
 */
typedef struct bartorch_linop_s bartorch_linop;
typedef struct bartorch_prox_s bartorch_prox;
typedef struct bartorch_nlop_s bartorch_nlop;
typedef int (*bartorch_apply_fn)(void* ctx, void* dst, const void* src);
typedef void (*bartorch_release_fn)(void* ctx);

BARTORCH_API bartorch_linop* bartorch_linop_callback(int ON, const long* odims, int IN, const long* idims,
		bartorch_apply_fn forward, bartorch_apply_fn adjoint, bartorch_apply_fn normal,
		void* ctx, bartorch_release_fn release);
BARTORCH_API bartorch_linop* bartorch_linop_fft(int N, const long* dims, unsigned long flags, int inverse, int centered);
BARTORCH_API bartorch_linop* bartorch_linop_cdiag(int N, const long* dims, unsigned long flags, const void* diag);
BARTORCH_API bartorch_linop* bartorch_linop_fmac(int N, const long* odims, const long* idims, const long* tdims, const void* tensor);
BARTORCH_API bartorch_linop* bartorch_linop_sampling(const long* dims, const long* pat_dims, const void* pattern);
BARTORCH_API bartorch_linop* bartorch_linop_nufft(int N, const long* ksp_dims, const long* cim_dims, const long* traj_dims,
		const void* traj, const long* wgh_dims, const void* weights,
		const long* bas_dims, const void* basis, int toeplitz, float os, float width);
/* SENSE, over sensitivities held as maps or as k-space kernels, walking the
 * coils a slab at a time.  A NULL trajectory makes the Cartesian operator,
 * and `modulated` asks that one for BART's own sample convention -- a scale
 * and a modulation folded into the sensitivities, the plain transform after
 * them -- rather than the centred transform.  Off the grid there is only one
 * convention and `modulated` is refused. */
BARTORCH_API bartorch_linop* bartorch_linop_sense(const long* max_dims, const long* ksp_dims,
		const long* sens_dims, const void* sens, int kernels,
		const long* traj_dims, const void* traj,
		const long* wgh_dims, const void* weights,
		const long* bas_dims, const void* basis, int toeplitz, int modulated);
/* The coil multiply on its own, over sensitivities held as maps or as kernels,
 * walking the coils a slab at a time.  For an encoding whose transform is not
 * a Fourier transform and so cannot be a SENSE operator. */
BARTORCH_API bartorch_linop* bartorch_linop_coils(const long* max_dims, const long* sens_dims,
		const void* sens, int kernels);
/* `a`, answering `normal` when it is asked for A^H A, rather than the adjoint
 * chained onto the forward. */
BARTORCH_API bartorch_linop* bartorch_linop_with_normal(const bartorch_linop* a, const bartorch_linop* normal);
BARTORCH_API bartorch_linop* bartorch_linop_chain(const bartorch_linop* a, const bartorch_linop* b);
BARTORCH_API bartorch_linop* bartorch_linop_plus(const bartorch_linop* a, const bartorch_linop* b);
BARTORCH_API bartorch_linop* bartorch_linop_adjoint_op(const bartorch_linop* a);
BARTORCH_API bartorch_linop* bartorch_linop_stack_cod(int n, const bartorch_linop** ops, int stack_dim);
BARTORCH_API bartorch_linop* bartorch_linop_stack(int cod_dim, int dom_dim, const bartorch_linop* a, const bartorch_linop* b);
BARTORCH_API bartorch_linop* bartorch_linop_normal_op(const bartorch_linop* a);
BARTORCH_API bartorch_linop* bartorch_linop_scale(int N, const long* dims, float re, float im);
BARTORCH_API bartorch_linop* bartorch_linop_zconj(int N, const long* dims);
BARTORCH_API bartorch_linop* bartorch_linop_identity(int N, const long* dims);
BARTORCH_API bartorch_linop* bartorch_linop_null(int NO, const long* odims, int NI, const long* idims);
BARTORCH_API double bartorch_linop_maxeigen(const bartorch_linop* a);
BARTORCH_API bartorch_linop* bartorch_linop_zreal(int N, const long* dims);
BARTORCH_API bartorch_linop* bartorch_linop_rdiag(int N, const long* dims, unsigned long flags, const void* diag);
BARTORCH_API bartorch_linop* bartorch_linop_matrix(int N, const long* odims, const long* idims, const long* mdims, const void* matrix);
BARTORCH_API bartorch_linop* bartorch_linop_conv(int N, unsigned long flags, int ctype, int cmode, const long* odims, const long* idims, const long* kdims, const void* kernel);
BARTORCH_API bartorch_linop* bartorch_linop_grad(int N, const long* dims, int d, unsigned long flags);
BARTORCH_API bartorch_linop* bartorch_linop_sum(int N, const long* dims, unsigned long flags);
BARTORCH_API bartorch_linop* bartorch_linop_scaled_sum(int N, const long* dims, unsigned long flags);
BARTORCH_API bartorch_linop* bartorch_linop_avg(int N, const long* dims, unsigned long flags);
BARTORCH_API bartorch_linop* bartorch_linop_repmat(int N, const long* odims, unsigned long flags);
BARTORCH_API bartorch_linop* bartorch_linop_flip(int N, const long* dims, unsigned long flags);
BARTORCH_API bartorch_linop* bartorch_linop_hankel(int N, const long* dims, int dim, int window_dim, int window);
BARTORCH_API bartorch_linop* bartorch_linop_reshape(int NO, const long* odims, int NI, const long* idims);
BARTORCH_API bartorch_linop* bartorch_linop_resize(int N, const long* odims, const long* idims);
BARTORCH_API bartorch_linop* bartorch_linop_extract(int N, const long* pos, const long* odims, const long* idims);
BARTORCH_API bartorch_linop* bartorch_linop_transpose(int N, int a, int b, const long* dims);
BARTORCH_API bartorch_linop* bartorch_linop_permute(int N, const int* order, const long* idims);
BARTORCH_API bartorch_linop* bartorch_linop_shift(int N, const long* dims, int dim, long shift, int pad);
BARTORCH_API bartorch_linop* bartorch_linop_padding(int N, const long* dims, int pad, const long* before, const long* after);
BARTORCH_API int bartorch_linop_has_pseudo_inv(const bartorch_linop* h);
BARTORCH_API int bartorch_linop_pseudo_inv(const bartorch_linop* h, float lambda, void* dst, const void* src);
BARTORCH_API int bartorch_linop_domain(const bartorch_linop* h, int N, long* dims);
BARTORCH_API int bartorch_linop_codomain(const bartorch_linop* h, int N, long* dims);
BARTORCH_API int bartorch_linop_forward(const bartorch_linop* h, void* dst, const void* src);
BARTORCH_API int bartorch_linop_adjoint(const bartorch_linop* h, void* dst, const void* src);
BARTORCH_API int bartorch_linop_normal(const bartorch_linop* h, void* dst, const void* src);
BARTORCH_API void bartorch_linop_free(bartorch_linop* h);

/*
 * The solve `pics` runs, assembled from here.
 *
 * `pics` turns its arguments into proximal operators, an algorithm and an
 * encoding, and hands the three to `lsqr2`.  This does the same with the same
 * BART functions -- `opt_reg_configure`, `italgo_config`, `lsqr2` -- so that
 * an operator built by the host and solved through here is the tool's own
 * computation rather than a second one that resembles it.
 *
 * A regularization term is named rather than spelled: `reg_kinds[i]` is the
 * letter `pics -R` uses for it and the arrays beside it are what that term's
 * specification carries -- the axes it works over, the axes it joins, its
 * weight, and the count an NIHT term takes.  They fill the table BART's own
 * parser would have filled, so what `opt_reg_configure` builds from them is
 * what it builds for the tool.
 * `algorithm` is one of "cg", "ist", "fista", "admm", "pridu", "niht",
 * "eulermaruyama", or NULL to let BART choose as it does for the tool.
 * A negative `step` or fista parameter leaves BART its own default, which for
 * the proximal-gradient iterations is the 0.95 `pics` settles on.  `cclambda`
 * is the weight in the normal equations, which is `pics -q`; the regularizers'
 * own weights are theirs, and `pics -r` is a term (`-R Q`) rather than a knob.
 * `sigma_tau_ratio` balances the primal and dual steps of PRIDU, and is the
 * scaling the caller divided the data by: `pics` sets it from the scaling it
 * estimated for itself, so an assembled reconstruction that scales its own
 * data has to say by how much.
 * `cg_tol` is the tolerance of conjugate gradients, which `italgo_config`
 * leaves at BART's default of zero; the other iterations ignore it.
 * `admm_dynamic_rho`, `admm_dynamic_tau`, `admm_relative_norm` and
 * `admm_fast` are the rest of `struct admm_conf`: Boyd's penalty adaptation
 * and Wohlberg's residual balancing, the residuals it balances taken relative
 * to their scalings, and the mode that skips computing them at all.  What
 * `italgo_config` does not take -- the over-relaxation, `mu`, `tau_max` and
 * the two tolerances -- is out of reach from here and reachable only from the
 * iteration written in Python.
 * `iterations`, when given, is filled with the steps the algorithm took --
 * every one of BART's calls `iter_monitor` once a step, so counting those
 * counts them.  For conjugate gradients that is the number an
 * alternating-direction solver budgets by, and there is no other way to see
 * it from outside.
 *
 * Returns 0, or a code `bartorch_solve_error` turns into a sentence.
 */
/* The largest eigenvalue of the operator a step is divided by (`pics -e`).
 *
 * A power iteration from a random start, so it draws on BART's own generator:
 * a loop written outside the library has to ask for it here, at the point in
 * the sequence the library would have asked, or the draws that follow it --
 * a wavelet term's cycle spinning, say -- are different ones.
 *
 * `A` and `cclambda` are the encoding and the quadratic weight, which
 * together are the operator `lsqr` builds.  `proxes` are terms whose
 * transforms are added to it, which is what the primal-dual iteration
 * estimates over and the proximal ones do not.
 *
 * Returns 0 and writes `out`, or a negative code.
 */
BARTORCH_API int bartorch_maxeigen(const bartorch_linop* A, float cclambda,
		int nprox, const bartorch_prox* const* proxes,
		int iterations, double* out);

BARTORCH_API int bartorch_solve(const bartorch_linop* A,
		const char* algorithm,
		const char* const* reg_kinds, const long* reg_xflags, const long* reg_jflags,
		const float* reg_lambda, const int* reg_k,
		const bartorch_prox* const* reg_ops, int n_reg,
		float cclambda, int maxiter, float step, int eigen, int hogwild,
		float admm_rho, int admm_maxitercg, float cg_tol,
		int admm_dynamic_rho, int admm_dynamic_tau, int admm_relative_norm, int admm_fast,
		float fista_p, float fista_q, float fista_r,
		float sigma_tau_ratio, int adaptive_step,
		int warmstart,
		void* x, const void* y, long* iterations);
BARTORCH_API const char* bartorch_solve_error(int code);

/*
 * The scaling `pics` estimates for a non-Cartesian encoding: the spread of
 * |A^H y| read off its own order statistics.  BART has no tool for this one --
 * `estscaling` is the Cartesian branch -- and `pics` does it around the solve
 * rather than inside it, so the host does it around the solve too.
 *
 * `image` is `size` complex floats; it is copied, because BART's own estimate
 * sorts what it is given.  `p` is a percentile in (0, 1], or negative for the
 * rule `pics` uses.
 */
BARTORCH_API float bartorch_scaling_norm(long size, const void* image, float rescale,
		int compat, float p);

/*
 * One regularization term, built once and held.  What BART makes of a term is
 * a proximal operator and, for most of them, a transform to apply it through,
 * and the two belong together: `bartorch_solve` is handed these rather than a
 * description to build from, so a term built once is a term reused.
 *
 * `kind` is the letter `pics -R` uses, `xflags` and `jflags` the two bitmasks
 * that term's specification carries.  `img_dims` is a BART-order dimension
 * vector of BARTORCH_DIMS entries.  A term that extends the optimisation
 * variable -- TGV and the infimal convolutions -- is declined, because what it
 * adds is counted across the whole set.
 *
 * `shift_mode` is what `pics` passes `opt_reg_configure`: 0 for no shifting,
 * 1 for the random cycle spinning the tool does unless `-n`, 2 for its fully
 * overlapping blocks (`-N`).  A wavelet threshold's shifts come from a
 * generator of its own, which `bartorch_solve` rewinds before each solve, so a
 * term that is reused is a term freshly built as far as the answer goes.
 */
BARTORCH_API int bartorch_prox_create(const char* kind, long xflags, long jflags,
		float lambda, int k, int llr_blk, const char* wavelet, int shift_mode,
		const long* img_dims, bartorch_prox** out);
/* The shape a term's proximal operator works on -- the image's, or the
 * codomain of the transform the term applies first.  Returns the rank. */
BARTORCH_API int bartorch_prox_domain(const bartorch_prox* h, int N, long* dims);
/* prox_{gamma f}(src) into dst, over that shape. */
BARTORCH_API int bartorch_prox_apply(const bartorch_prox* h, float gamma, void* dst, const void* src);
/* The transform applied in place, for the one whose rank an operator here
 * cannot hold.  `mode` is 0 forward, 1 adjoint, 2 normal. */
BARTORCH_API int bartorch_prox_transform_apply(const bartorch_prox* h, int mode, void* dst, const void* src);
/* The transform the term applies before its proximal operator; the identity
 * for a term that carries its own.  The handle is the caller's to free. */
BARTORCH_API bartorch_linop* bartorch_prox_transform(const bartorch_prox* h);
/* Whether that transform is the identity, which is the question
 * `iter2_chambolle_pock` asks of the first term before deciding whether it
 * is a dual or the primal proximal step.  Returns 1, 0, or a negative code.
 */
BARTORCH_API int bartorch_prox_transform_is_identity(const bartorch_prox* h);
/* Put a term's own random generator back where a fresh term would have it.
 * A wavelet threshold spins its transform by a random shift drawn from a
 * generator seeded at one when the operator is made; the tool builds a fresh
 * operator per run, and a term kept across solves is rewound instead.
 * `bartorch_solve` does this itself.  Returns 0, or a negative code.
 */
BARTORCH_API int bartorch_prox_rewind(const bartorch_prox* h);
BARTORCH_API void bartorch_prox_free(bartorch_prox* h);

BARTORCH_API bartorch_nlop* bartorch_nlop_callback(int ON, const long* odims, int IN, const long* idims,
		bartorch_apply_fn forward, bartorch_apply_fn derivative, bartorch_apply_fn adjoint,
		void* ctx, bartorch_release_fn release);
BARTORCH_API bartorch_nlop* bartorch_nlop_from_linop(const bartorch_linop* lin);
BARTORCH_API bartorch_nlop* bartorch_nlop_chain(const bartorch_nlop* a, const bartorch_nlop* b);
BARTORCH_API int bartorch_nlop_domain(const bartorch_nlop* h, int N, long* dims);
BARTORCH_API int bartorch_nlop_codomain(const bartorch_nlop* h, int N, long* dims);
BARTORCH_API int bartorch_nlop_apply(const bartorch_nlop* h, void* dst, const void* src);
BARTORCH_API int bartorch_nlop_derivative(const bartorch_nlop* h, void* dst, const void* src);
BARTORCH_API int bartorch_nlop_adjoint(const bartorch_nlop* h, void* dst, const void* src);
/* How many arguments the operator takes, and the shape of each.
 *
 * BART's `nlop_s` is many inputs to many outputs -- the model `nlinv` inverts
 * has two inputs, the image and the coil profiles -- and these are what make
 * that arity reachable.  Arguments are counted BART's way: outputs first,
 * then inputs, which is the order `bartorch_nlop_apply_generic` reads them.
 * The domain and codomain calls return the rank, or a negative code.
 */
BARTORCH_API int bartorch_nlop_inputs(const bartorch_nlop* h);
BARTORCH_API int bartorch_nlop_outputs(const bartorch_nlop* h);
BARTORCH_API int bartorch_nlop_input_domain(const bartorch_nlop* h, int i, int N, long* dims);
BARTORCH_API int bartorch_nlop_output_codomain(const bartorch_nlop* h, int o, int N, long* dims);
/* Apply an operator of any arity; `args` is outputs then inputs.  Fixes the
 * point every derivative is taken at, as the one-argument apply does. */
BARTORCH_API int bartorch_nlop_apply_generic(const bartorch_nlop* h, int nargs, void** args);
/* The derivative of one output by one input, as a linear operator, at
 * wherever the last application left the point.  `nlop_get_derivative` hands
 * back a `linop_s`, so the whole linear surface applies to it -- which is how
 * `noir/recon2.c` builds the inner problem of a Gauss-Newton step. */
BARTORCH_API bartorch_linop* bartorch_nlop_derivative_linop(const bartorch_nlop* h, int o, int i);

/* The algebra of `nlops/chain.h`: one output into one input, two operators
 * side by side, an output tied back to an input, two inputs made one, and the
 * reorderings that make those usable.  Each returns a handle the caller owns,
 * or NULL. */
BARTORCH_API bartorch_nlop* bartorch_nlop_chain2(const bartorch_nlop* a, int o, const bartorch_nlop* b, int i);
BARTORCH_API bartorch_nlop* bartorch_nlop_combine(const bartorch_nlop* a, const bartorch_nlop* b);
BARTORCH_API bartorch_nlop* bartorch_nlop_link(const bartorch_nlop* x, int oo, int ii);
BARTORCH_API bartorch_nlop* bartorch_nlop_dup(const bartorch_nlop* x, int a, int b);
BARTORCH_API bartorch_nlop* bartorch_nlop_stack_inputs(const bartorch_nlop* x, int a, int b, int dim);
BARTORCH_API bartorch_nlop* bartorch_nlop_stack_outputs(const bartorch_nlop* x, int a, int b, int dim);
/* `outputs` non-zero permutes the outputs, zero the inputs. */
BARTORCH_API bartorch_nlop* bartorch_nlop_permute(const bartorch_nlop* x, int outputs, int n, const int* perm);
BARTORCH_API bartorch_nlop* bartorch_nlop_del_out(const bartorch_nlop* x, int o);

/* The basic nonlinear operators: the tensor product and the elementwise maps.
 *
 * `tenmul` is the pointwise product of two inputs, broadcast over the axes
 * where one of them is one -- the model `nlinv` inverts is an image times
 * coil profiles -- and is what makes the algebra above worth having.  The
 * rest take one input and return one output of the same shape; `eps`, where
 * it appears, picks BART's regularised variant when it is positive.
 *
 * `zphsr` is not here: BART builds it out of `zabs` and `zdiv` and finishes
 * with `nlop_dup(x, 0, 0)`, which trips its own `a < b` assertion, so the
 * constructor cannot be called at all.  The same operator is built out of
 * the two pieces on the Python side, where the indices are right.
 */
BARTORCH_API bartorch_nlop* bartorch_nlop_tenmul(int N, const long* odims, const long* idims1, const long* idims2);
BARTORCH_API bartorch_nlop* bartorch_nlop_zdiv(int N, const long* dims, float eps);
BARTORCH_API bartorch_nlop* bartorch_nlop_zaxpbz(int N, const long* dims, float a, float b);
BARTORCH_API bartorch_nlop* bartorch_nlop_zexp(int N, const long* dims);
BARTORCH_API bartorch_nlop* bartorch_nlop_zlog(int N, const long* dims);
BARTORCH_API bartorch_nlop* bartorch_nlop_zinv(int N, const long* dims, float eps);
BARTORCH_API bartorch_nlop* bartorch_nlop_zsqrt(int N, const long* dims);
BARTORCH_API bartorch_nlop* bartorch_nlop_zspow(int N, const long* dims, float re, float im);
BARTORCH_API bartorch_nlop* bartorch_nlop_zsadd(int N, const long* dims, float re, float im);
BARTORCH_API bartorch_nlop* bartorch_nlop_zabs(int N, const long* dims);
BARTORCH_API bartorch_nlop* bartorch_nlop_smo_abs(int N, const long* dims, float eps);
BARTORCH_API bartorch_nlop* bartorch_nlop_zrss(int N, const long* dims, unsigned long flags, float eps);
BARTORCH_API bartorch_nlop* bartorch_nlop_zss(int N, const long* dims, unsigned long flags);
/* An operator of no inputs, and pinning one input of an operator to a value. */
BARTORCH_API bartorch_nlop* bartorch_nlop_const(int N, const long* dims, const void* val);
BARTORCH_API bartorch_nlop* bartorch_nlop_set_input_const(const bartorch_nlop* a, int i, int N, const long* dims, const void* val);

BARTORCH_API void bartorch_nlop_free(bartorch_nlop* h);

/* Iteratively regularised Gauss-Newton: x starts at its initial value and returns the solution. */
BARTORCH_API int bartorch_irgnm(const bartorch_nlop* F, int iter, float alpha, float alpha_min, float redu,
		int cgiter, float cgtol, void* x, const void* y, const void* xref);

/*
 * CUDA.  Every entry point exists in both builds; without CUDA compiled in,
 * bartorch_cuda_built() returns 0 and the rest report failure.
 *
 * BART recognises device memory by asking the driver about the pointer, so a
 * tensor the host allocated on a device needs no registration, and an array
 * BART creates comes from the host's allocator on the selected device.
 * bartorch_cuda_enable() must be called before a command that is to run on a
 * device, and with -1 to go back to the host.
 *
 * The stream functions take a cudaStream_t as a void*.  wait_for_stream holds
 * BART's streams until the work already queued on the caller's stream has
 * run; signal_stream holds the caller's stream until BART's work has.  A
 * caller that does both around a command never synchronises the device.
 */
BARTORCH_API int bartorch_cuda_built(void);
BARTORCH_API int bartorch_cuda_device_count(void);
BARTORCH_API int bartorch_cuda_enable(int device);
BARTORCH_API int bartorch_cuda_device(void);
BARTORCH_API int bartorch_cuda_set_streams(int n);
BARTORCH_API int bartorch_cuda_get_streams(void);
BARTORCH_API int bartorch_cuda_use_memcache(int enable);
/* Hand every stream's cache of freed device blocks back to the driver. */
BARTORCH_API void bartorch_cuda_memcache_clear_all(void);
BARTORCH_API int bartorch_cuda_wait_for_stream(void* stream);
BARTORCH_API int bartorch_cuda_signal_stream(void* stream);
BARTORCH_API long bartorch_cuda_free_memory(void);

/*
 * FINUFFT under BART's own NUFFT operator, which is what makes nufft, pics,
 * nlinv and moba compute their transform with it.  The seam is nufft_create
 * rather than the gridding kernel: FINUFFT does the spreading, the FFT and
 * the deapodisation together, so none of the three has to agree with BART's,
 * only the sign and the scaling.
 *
 * Give this the entry points of an installed FINUFFT and the byte layout of
 * its options struct, which the host reads from the same package, then turn
 * it on.  A subspace basis, weights that do not lie along k-space, and a
 * trajectory that changes across frames fall back to BART's own operator,
 * and bartorch_nufft_decline_reason says which.
 */
BARTORCH_API int bartorch_finufft_set(const char* symbol, void* fn);
/* `device` picks the table: FINUFFT's on the host, cuFINUFFT's on a card.
 * `device_field` is the byte offset of the one option this sets -- the thread
 * count on the host, the device number on a card. */
BARTORCH_API int bartorch_finufft_layout(int device, int opts_size, int device_field, int upsampling_field, int spreadonly_field);
BARTORCH_API void bartorch_finufft_set_tolerance(double eps);
BARTORCH_API double bartorch_finufft_tolerance(void);
BARTORCH_API void bartorch_finufft_set_upsampling(double upsampling);
BARTORCH_API double bartorch_finufft_upsampling(void);
/* Threads a transform on the host takes; zero leaves the count to FINUFFT.
 * bartorch_set_num_threads sets this too. */
BARTORCH_API void bartorch_finufft_set_threads(int n);
BARTORCH_API int bartorch_finufft_threads(void);
BARTORCH_API void bartorch_finufft_use_in_tools(int enable);
/* Whether a transform can be served where the data is: 0 host, 1 device. */
BARTORCH_API int bartorch_finufft_usable_on(int device);
BARTORCH_API int bartorch_finufft_usable(void);
/* Plans made and not yet destroyed: zero once every operator, point spread
 * function and mask that asked for one has been freed. */
BARTORCH_API long bartorch_finufft_live_plans(void);
BARTORCH_API const char* bartorch_last_error(void);
BARTORCH_API void bartorch_clear_error(void);
BARTORCH_API int bartorch_nufft_decline_reason(void);
BARTORCH_API const char* bartorch_nufft_decline_text(void);
BARTORCH_API void bartorch_nufft_allow_fallback(int enable);
BARTORCH_API int bartorch_nufft_fallback_allowed(void);
/* Operators built since the last reset: 0 by FINUFFT, 1 by BART. */
BARTORCH_API long bartorch_nufft_counter(int which);
BARTORCH_API void bartorch_nufft_reset_counters(void);

/*
 * A^H A for a non-Cartesian encoding is a convolution, so a solve applies it
 * as one multiply against a point spread function rather than a forward and
 * an adjoint transform.  That is BART's own Toeplitz embedding, which the
 * substituted operator borrows for its normal while FINUFFT keeps the pair;
 * `pics --no-toeplitz` and `nufft -t` are what decide whether there is one.
 *
 * Normal operators since the last reset: 0 answered by a point spread
 * function, 1 by the transform pair.
 */
BARTORCH_API long bartorch_toeplitz_counter(int which);
BARTORCH_API void bartorch_toeplitz_reset_counters(void);

/* Whether a pointer is device memory; always false without CUDA. */
BARTORCH_API int bartorch_on_device(const void* ptr);

#ifdef __cplusplus
}
#endif

#endif
