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
typedef struct bartorch_noir_s bartorch_noir;
typedef int (*bartorch_apply_fn)(void* ctx, void* dst, const void* src);
/* A many-argument apply: `args` holds the output buffers and then the input
 * ones, which is the order `nlop_generic_create` passes them in. */
typedef int (*bartorch_generic_apply_fn)(void* ctx, int N, void** args);
/* One derivative, or its adjoint, for the pair of arguments `(o, i)`. */
typedef int (*bartorch_pair_apply_fn)(void* ctx, int o, int i, void* dst, const void* src);
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
/* An operator built for one block, applied to `n` consecutive blocks of arrays
 * of `idims` and `odims`: how an operator in the torch layout, whose batches
 * are slowest in memory, is built on BART's dimensions without a copy.  The
 * block operator is referenced, so the caller still frees it. */
BARTORCH_API bartorch_linop* bartorch_linop_blocks(const bartorch_linop* block, int N,
		const long* odims, const long* idims, long n);
/* `op` over the same memory described by `idims` and `odims`, with nothing
 * copied.  The operator is referenced, so the caller still frees it. */
BARTORCH_API bartorch_linop* bartorch_linop_reshaped(const bartorch_linop* op, int N,
		const long* odims, const long* idims);
/*
 * The MRI encoding form.  Every encoding this library builds reduces to
 *
 *	y[c, t, k] = sum_a O[a, t](k) . T_t( I[c, a, t](r) . x[a](r) )(k)
 *
 * and one executor runs all of them: the transform T, the image-side factor
 * I, the k-space factor O, and the contraction over a.  What a caller
 * composes in Python is matched against this form and lowered into one of
 * these records; the executor is parameterised by the record rather than
 * written once per encoding.
 */
enum bartorch_encoding_transform {

	/* The coil multiply with no transform after it. */
	BARTORCH_ENCODING_NONE = 0,
	/* A Fourier transform on the grid the image lies on. */
	BARTORCH_ENCODING_FFT = 1,
	/* A NUFFT over a trajectory. */
	BARTORCH_ENCODING_NUFFT = 2,
	/* Readout transform, point spread function, phase-encode transform. */
	BARTORCH_ENCODING_WAVE = 3,
};

struct bartorch_encoding {

	/* Which transform, from bartorch_encoding_transform. */
	int transform;

	/* The whole operator's dimensions -- spatial axes, coils, sets of
	 * maps, coefficients -- and one coil's samples with the coefficients
	 * a basis contracts still on them. */
	const long* max_dims;
	const long* ksp_dims;

	/* The image-side factor: coil sensitivities as maps, or as the
	 * k-space kernels they band-limit to, inflated a slab at a time. */
	const long* sens_dims;
	const void* sens;
	int kernels;

	/* The k-space factors.  Each pointer may be NULL, and its dimensions
	 * are then not read. */
	const long* pat_dims;
	const void* pattern;
	const long* bas_dims;
	const void* basis;
	const long* wgh_dims;
	const void* weights;

	/* A NUFFT's trajectory, in grid units. */
	const long* traj_dims;
	const void* traj;
	/* A stack whose kz lies on the image's own z grid, decoupled: the
	 * trajectory is one position's in-plane shots, kz zero, the transform
	 * over x and y has z as a batch of it, and the coil images are
	 * transformed along z around it. */
	int stacked;

	/* A table of the phase encodes that were sampled, instead of a dense
	 * pattern: `frames` x `shots` places of `components` long indices
	 * each -- (y) for a 2D image, (z, y) for a 3D one, -1 for padding --
	 * with the whole readout along each.  `kspace_readout` says the
	 * samples are in k-space along the readout rather than transformed
	 * back along it.  NULL `positions` is dense samples. */
	long frames;
	long shots;
	int components;
	const long* positions;
	int kspace_readout;

	/* A wave: the oversampled readout the coil images are zero-filled to,
	 * the point spread function over it, and whether its two transforms
	 * are centred and unitary rather than BART's own. */
	long readout;
	const void* psf;
	int centred;

	/* The contraction over `segments` terms, sum_l diag(sample_l) E
	 * diag(image_l): off-resonance by time segmentation.  Each weight is
	 * laid out on its own dimensions, the terms contiguous one after
	 * another.  Zero segments is no contraction. */
	long segments;
	const long* segment_sample_dims;
	const void* segment_sample;
	const long* segment_image_dims;
	const void* segment_image;

	/* The BART dimension a batch the sensitivities vary along lies on, or
	 * -1 for none.  The torch layout puts such a batch above the coils, and
	 * it is inside this operator rather than around it because one bank
	 * cannot serve every item of it: independent slices, each with their own
	 * maps, over one trajectory. */
	int batch_dim;

	/* A k-space factor that differs between sets of maps, applied after
	 * the transform, with the sets summed over after it: the slice phase
	 * of a simultaneous-multislice acquisition.  The sets then survive the
	 * sensitivities rather than being contracted by them, so the transform
	 * runs once per set -- which is what the sum being on the far side of
	 * it costs.  NULL leaves the sets where the sensitivities sum them. */
	const long* slice_dims;
	const void* slice;

	/* The closed-form normal rather than the two applications. */
	int toeplitz;
	/* BART's own sample convention rather than the centred one; a grid
	 * transform's to answer, and refused off one. */
	int modulated;
	/* Coils in a slab, and whether the sensitivities are applied inside
	 * the transform of the normal.  Zero coils leaves BART its own
	 * operator over every coil at once. */
	int coil_batch;
	int fold_maps;
};

/* The encoding this form describes, as one BART operator. */
BARTORCH_API bartorch_linop* bartorch_linop_encoding(const struct bartorch_encoding* form);
/* What the slab executor has built and run since the last reset, so that a
 * test can say which path was taken rather than infer it from timing.  The
 * first three are the counts bartorch_sense_counter reports, over the same
 * storage; bartorch_encoding_reset_counters and bartorch_sense_reset_counters
 * both clear all of them. */
enum bartorch_encoding_count {

	/* Forms built into the slab loop, and forms the loop could not take
	 * and which BART's plain chain of operators answers instead. */
	BARTORCH_ENCODING_BUILT = 0,
	BARTORCH_ENCODING_CHAINED = 1,
	/* Normals applied with the sensitivities inside the transform. */
	BARTORCH_ENCODING_FOLDED = 2,
	/* Applications of the slab loop. */
	BARTORCH_ENCODING_FORWARD = 3,
	BARTORCH_ENCODING_ADJOINT = 4,
	BARTORCH_ENCODING_NORMAL = 5,
	/* Forms built with a contraction over segments. */
	BARTORCH_ENCODING_SEGMENTED = 6,
	/* Forms built as a stack decoupled along z. */
	BARTORCH_ENCODING_STACKED = 7,
};
BARTORCH_API long bartorch_encoding_counter(int which);
BARTORCH_API void bartorch_encoding_reset_counters(void);
/* Normals of a Cartesian encoding applied through cuFFT's callbacks since the
 * library was loaded; the rest were applied as BART's chain of operators. */
BARTORCH_API long bartorch_grid_fused(void);
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
 *
 * Three terms are not one proximal operator on the image.  Total generalized
 * variation and the two infimal convolutions add unknowns -- BART calls them
 * supporting variables -- and split into several penalties at offsets into
 * the enlarged vector, and `opt_reg_configure` works those offsets out across
 * the whole set at once, which is why they cannot be built one at a time.
 * When one of them is in the set, this configures the set itself rather than
 * taking `reg_ops`, chains `linop_extract_create` onto the encoding so the
 * model still sees an image, solves over the longer vector, and hands back
 * the image part of it.  That is what `pics.c` does, and the terms that are
 * built here are built fresh per solve, as the tool builds them.
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
		/* Left preconditioning, chained onto the normal operator and the
		 * adjoint by `lsqr2_create`.  NULL for none, which is what `pics`
		 * passes and what every solve did before.  `conjgrad` has no
		 * preconditioner of its own; this is the only place one enters. */
		const bartorch_linop* precond,
		/* The Euler-Maruyama sampler's own preconditioner, which is a
		 * different thing: `eulermaruyama_precond` solves `(M^H M + diag) o = x`
		 * with `conjgrad` every step, and `pics` has no flag for it.  A
		 * `diag` of zero leaves the plain sampler. */
		const bartorch_linop* em_precond, float em_precond_diag, float em_precond_tol,
		int em_precond_maxiter,
		/* What `opt_reg_configure` needs when the set has to be built here:
		 * one block size, one wavelet family and one shift mode for the whole
		 * of it, as `pics` has one `-b` and one `-w`; and the two pairs
		 * `pics --alpha` and `pics --gamma` set, two floats each or NULL for
		 * BART's own.  Ignored otherwise. */
		int llr_blk, const char* wavelet, int shift_mode,
		const float* alpha, const float* gamma,
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

/* The same, for an operator of many arguments.
 *
 * `nlop_generic_create`.  The shapes arrive flat, one argument's dimension
 * vector after another's: `OO` outputs of rank `ON`, then `II` inputs of rank
 * `IN`.  The forward callback is handed every buffer at once -- the outputs
 * first and then the inputs, which is BART's order -- and the derivative and
 * its adjoint are handed the pair `(o, i)` they are being asked for.
 *
 * What this is for is a Python function with more than one argument standing
 * inside a BART graph: a denoiser whose weights are an *input* rather than
 * something it closed over, so that a gradient reaches them. */
BARTORCH_API bartorch_nlop* bartorch_nlop_callback_generic(int OO, int ON, const long* odims,
		int II, int IN, const long* idims,
		bartorch_generic_apply_fn forward,
		bartorch_pair_apply_fn derivative,
		bartorch_pair_apply_fn adjoint,
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
/* The same operator with one argument's shape written at a different rank.
 *
 * `nlop_chain2` and `nlop_link` compare iovecs, and an iovec carries its rank:
 * BART builds each of its own operators at whatever rank it needs -- the
 * Gauss-Newton step's state is two long -- while an operator defined through
 * the callbacks above is built at DIMS.  So two arguments of the same shape
 * can still refuse to meet.  Padding a shape with ones is not a change to it,
 * and this is how that is said. */
BARTORCH_API bartorch_nlop* bartorch_nlop_reshape_in(const bartorch_nlop* a, int i, int N, const long* dims);
BARTORCH_API bartorch_nlop* bartorch_nlop_reshape_out(const bartorch_nlop* a, int o, int N, const long* dims);

BARTORCH_API bartorch_nlop* bartorch_nlop_chain2(const bartorch_nlop* a, int o, const bartorch_nlop* b, int i);
BARTORCH_API bartorch_nlop* bartorch_nlop_combine(const bartorch_nlop* a, const bartorch_nlop* b);
BARTORCH_API bartorch_nlop* bartorch_nlop_link(const bartorch_nlop* x, int oo, int ii);
BARTORCH_API bartorch_nlop* bartorch_nlop_dup(const bartorch_nlop* x, int a, int b);
BARTORCH_API bartorch_nlop* bartorch_nlop_stack_inputs(const bartorch_nlop* x, int a, int b, int dim);
BARTORCH_API bartorch_nlop* bartorch_nlop_stack_outputs(const bartorch_nlop* x, int a, int b, int dim);
/* `outputs` non-zero permutes the outputs, zero the inputs. */
BARTORCH_API bartorch_nlop* bartorch_nlop_permute(const bartorch_nlop* x, int outputs, int n, const int* perm);
BARTORCH_API bartorch_nlop* bartorch_nlop_del_out(const bartorch_nlop* x, int o);
/* Every input reshaped into one flat vector, and every output into another,
 * which is how a many-unknown model reaches a solver that knows one vector.
 * `inputs_only` leaves the outputs as they are. */
BARTORCH_API bartorch_nlop* bartorch_nlop_flatten(const bartorch_nlop* x, int inputs_only);

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

/* The nonlinear SENSE model `nlinv` inverts, from `noir/model2.c`.
 *
 *	kspace = A[ (mask * image) * ifftuc(weights * ksens) ]
 *
 * The model takes the image and the coil coefficients, in that order, and
 * returns data of whatever shape `bartorch_noir_data` takes: off the grid the
 * model is asymmetric and returns gridded coil images, so the measurement has
 * to be gridded with that operator's adjoint first; on the grid it is the
 * identity and the model returns k-space.
 *
 * The coils are unknown as k-space coefficients; `bartorch_noir_coils` is the
 * Sobolev weighting and transform that turn a fitted set of them into
 * sensitivities.  A NULL data pointer beside a dimension vector means the
 * quantity is not used; `pat_dims` is required on the grid and `trj_dims` off
 * it.
 *
 * `bartorch_noir_dims` reports the shapes BART settled on: 0 k-space, 1 coil
 * images, 2 image, 3 coils, 4 coils as the product takes them, 5 pattern,
 * 6 trajectory.  The shape of the coil coefficients is the model's second
 * input, which the arity queries above report.
 */
BARTORCH_API bartorch_noir* bartorch_noir_create(int N,
		const long* ksp_dims, const long* cim_dims, const long* img_dims,
		const long* kco_dims, const long* col_dims,
		const long* pat_dims, const void* pattern,
		const long* trj_dims, const void* traj,
		const long* wgh_dims, const void* weights,
		const long* bas_dims, const void* basis,
		const long* msk_dims, const void* mask,
		int noncart, int optimized, int toeplitz,
		unsigned long fft_flags, unsigned long wght_flags,
		int rvc, int sos, float a, float b, float c,
		float oversampling_coils, int ret_os_coils);
BARTORCH_API bartorch_nlop* bartorch_noir_model(const bartorch_noir* h);
BARTORCH_API bartorch_linop* bartorch_noir_coils(const bartorch_noir* h);
BARTORCH_API bartorch_linop* bartorch_noir_image(const bartorch_noir* h);
BARTORCH_API bartorch_linop* bartorch_noir_data(const bartorch_noir* h);
BARTORCH_API bartorch_linop* bartorch_noir_transform(const bartorch_noir* h);
BARTORCH_API int bartorch_noir_dims(const bartorch_noir* h, int which, int N, long* dims);
BARTORCH_API void bartorch_noir_free(bartorch_noir* h);

/* The same model, built for a network rather than for a solve: `noir2_net_s`
 * from `noir/model_net.c`, which is what `nlinvnet` reconstructs with.
 *
 * What it adds is a Gauss-Newton step that is itself an `nlop`.  BART builds
 * it out of `nlop`s throughout -- the forward model, its derivative *as a
 * function of the linearisation point*, the adjoint, and `norm_inv`'s
 * implicitly differentiated inverse of the normal operator -- so the step
 * differentiates with respect to the data, the iterate, the regularisation
 * centre and the weight, second-order terms included.  That is the cell an
 * unrolled NLINV is made of, and it is the library's own rather than a
 * reconstruction of it here.
 *
 * The batch is the model's: `batch` copies of it are built and stacked, and
 * `batch_flag` says which of BART's axes are already a batch.
 *
 * `basis` and `mask` are copied; the trajectory and the sampling pattern are
 * not held by the model at all -- they are inputs of the operators below,
 * because a network is handed them per call.
 *
 * The operators:
 *
 *	step		(y, xn, x0, alpha)  -> x		one Gauss-Newton step
 *	iterations	(y, xn, x0, alpha)  -> x		`iterations` of them,
 *							 alpha decaying by `redu`
 *							 towards `alpha_min`
 *	adjoint		(kspace, pattern)   -> y		the data as the step
 *							 takes it; off the grid
 *							 (trajectory, pattern)
 *	decompose	x -> (image, sensitivities)	with the model's transforms
 *	split		x -> (image, coefficients)	without them
 *	join		(image, coefficients) -> x
 *
 * What shape each of them takes is read off the operator itself, with the
 * arity queries above; there is no second place here that says so.
 */
typedef struct bartorch_noir_net_s bartorch_noir_net;

BARTORCH_API bartorch_noir_net* bartorch_noir_net_create(int N,
		const long* ksp_dims, const long* cim_dims,
		const long* img_dims, const long* col_dims,
		const long* trj_dims, const long* wgh_dims,
		const long* bas_dims, const void* basis,
		const long* msk_dims, const void* mask,
		unsigned long batch_flag, int batch,
		unsigned long fft_flags, unsigned long wght_flags,
		int rvc, int sos, float a, float b, float c, int toeplitz);
BARTORCH_API bartorch_nlop* bartorch_noir_net_step(const bartorch_noir_net* h,
		int cgiter, float cgtol, float l2lambda);
BARTORCH_API bartorch_nlop* bartorch_noir_net_iterations(const bartorch_noir_net* h,
		int cgiter, float cgtol, float l2lambda,
		int iterations, float redu, float alpha_min);
BARTORCH_API bartorch_nlop* bartorch_noir_net_adjoint(const bartorch_noir_net* h);
BARTORCH_API bartorch_nlop* bartorch_noir_net_decompose(const bartorch_noir_net* h);
BARTORCH_API bartorch_nlop* bartorch_noir_net_split(const bartorch_noir_net* h);
BARTORCH_API bartorch_nlop* bartorch_noir_net_join(const bartorch_noir_net* h);
BARTORCH_API void bartorch_noir_net_free(bartorch_noir_net* h);

/* Iteratively regularised Gauss-Newton: x starts at its initial value and returns the solution. */
BARTORCH_API int bartorch_irgnm(const bartorch_nlop* F, int iter, float alpha, float alpha_min, float redu,
		int cgiter, float cgtol, void* x, const void* y, const void* xref);
/* The second form: the extra derivative application that lets the inner
 * problem go to a generic solver.  A NULL solver, which is what this passes,
 * is BART's own conjugate gradients; the Python loop that takes any of the
 * proximal solvers instead is held against this one. */
BARTORCH_API int bartorch_irgnm2(const bartorch_nlop* F, int iter, float alpha, float alpha_min,
		float alpha_min0, float redu, int cgiter, float cgtol,
		void* x, const void* y, const void* xref);

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
