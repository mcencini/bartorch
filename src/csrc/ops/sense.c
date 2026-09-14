/*
 * The SENSE operators, with the coil loop inside them.
 *
 * BART builds these as two operators chained: the sensitivities as one
 * multiply-accumulate over every coil at once, then the transform over every
 * coil at once.  Everything between them is the size of the coil images, and
 * so is everything the Toeplitz normal allocates behind a non-Cartesian
 * transform -- the doubled grid it convolves on is `coils x 2^d x image`, one
 * block.  On a three-dimensional problem that is what decides whether a
 * reconstruction fits on a card at all.
 *
 * The coils are independent until the sum that ends the adjoint, so these
 * walk them a slab at a time: the sensitivities of that slab are applied, the
 * transform is asked for that slab, and the slab is summed into the answer.
 * What is resident is a slab rather than a bank, and the transform is built
 * for a slab, so its own working grid shrinks with it.
 *
 * The slab is a whole coil or several, because a transform of one coil is not
 * a transform of eight at an eighth of the cost: FINUFFT batches its plan
 * across transforms and a slab of one gives that up.  What the slab costs in
 * throughput and saves in memory is the whole trade, and it is a setting.
 *
 * A pattern or a basis laid out along the coils, or sensitivities that do not
 * carry the coils the images do, is not something this arrangement can serve;
 * those go back to BART's own chain, which the rename leaves reachable.
 */
#include <complex.h>
#include <math.h>
#include <stdbool.h>
#include <stdlib.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/types.h"

#include "num/flpmath.h"
#include "num/fft.h"
#include "num/multind.h"
#include "num/iovec.h"

#include "linops/fmac.h"
#include "linops/linop.h"
#include "linops/someops.h"

#include "num/gpuops.h"

#include "noncart/nufft.h"

#include "sense/model.h"

#include "include/bartorch.h"

/* Provided by nufft_finufft.c, which holds the function a normal convolves
 * with.  A streamed function crosses once for each set that is used, so the
 * sets belong outside the coils rather than inside them. */
extern int bartorch_nufft_cosets(const struct linop_s* op);
extern void bartorch_nufft_coset_begin(const struct linop_s* op, const void* ref);
extern void bartorch_nufft_coset_use(const struct linop_s* op, int i);
extern void bartorch_nufft_coset_normal(const struct linop_s* op, complex float* dst, const complex float* src, int last);
extern int bartorch_nufft_coset_folds(const struct linop_s* op);
extern void bartorch_nufft_coset_normal_sense(const struct linop_s* op,
		complex float* dst, const complex float* src,
		const long map_strs[], const complex float* map, int last);
extern void bartorch_nufft_coset_end(const struct linop_s* op);

/* Provided by grid.c: a Cartesian transform's normal, run through cuFFT's
 * callbacks with the sensitivity handed in, where the card allows it. */
extern int bartorch_grid_folds(const struct linop_s* op, const void* ref);
extern int bartorch_grid_folds_samples(const struct linop_s* op, const void* ref);
extern void bartorch_grid_forward_sense(const struct linop_s* op, complex float* dst, const complex float* src,
		const long map_strs[DIMS], const complex float* map);
extern void bartorch_grid_adjoint_sense(const struct linop_s* op, complex float* dst, const complex float* src,
		const long map_strs[DIMS], const complex float* map);
extern void bartorch_grid_normal_sense(const struct linop_s* op, complex float* dst, const complex float* src,
		const long map_strs[DIMS], const complex float* map);

#ifdef USE_CUDA
/* csrc/kernels.cu: a volume times BART's inverse fftmod along its first three axes. */
extern void bartorch_cuda_modulate(const long dims[3], long rest, const long grid[3], const long off[3],
		float scale, complex float* x);
#endif

extern struct linop_s* bart_sense_init(unsigned long shared_img_flags, const long max_dims[DIMS],
		unsigned long sens_flags, const complex float* sens);

extern const struct linop_s* bart_sense_nc_init(const long max_dims[DIMS], const long map_dims[DIMS], const complex float* maps,
		const long ksp_dims[DIMS],
		const long traj_dims[DIMS], const complex float* traj, const struct nufft_conf_s* conf,
		const long wgs_dims[DIMS], const complex float* weights,
		const long basis_dims[DIMS], const complex float* basis,
		const struct linop_s** fft_opp, unsigned long shared_img_dims);

/* How many coils a slab holds.  Zero leaves the operators as BART builds
 * them, every coil at once, which is the fastest and the largest. */
static int coil_batch = 1;

/* Whether the sensitivity is applied inside the transform rather than by
 * making a coil image to multiply it into and another for the answer. */
static int fold_maps = 1;

void bartorch_sense_set_fold_maps(int enable)
{
	fold_maps = (0 != enable);
}

int bartorch_sense_fold_maps(void)
{
	return fold_maps;
}

/* Operators built since the last reset: with the coil loop, and as BART's own
 * chain because this could not serve them. */
enum { SN_FUSED, SN_CHAINED, SN_FOLDED };
static long sense_counters[3];

void bartorch_sense_set_coil_batch(int coils)
{
	coil_batch = (coils > 0) ? coils : 0;
}

int bartorch_sense_coil_batch(void)
{
	return coil_batch;
}

long bartorch_sense_counter(int which)
{
	return sense_counters[((SN_FUSED == which) || (SN_FOLDED == which)) ? which : SN_CHAINED];
}

void bartorch_sense_reset_counters(void)
{
	sense_counters[SN_FUSED] = 0;
	sense_counters[SN_CHAINED] = 0;
	sense_counters[SN_FOLDED] = 0;
}

struct sense_s {

	linop_data_t super;

	long batch;		/* coils in a slab */
	bool fold;		/* apply the maps inside the transform of the normal */
	long coils;		/* coils in all */

	/* One slab: the dimensions the sensitivities contract over, the coil
	 * images they produce, and what the transform answers. */
	long slab_dims[DIMS];
	long cim_dims[DIMS];
	long out_dims[DIMS];
	long img_dims[DIMS];

	long map_dims[DIMS];
	long slab_map_strs[DIMS];	/* a slab of maps, densely */
	long cim_strs[DIMS];
	long out_strs[DIMS];
	long img_strs[DIMS];
	long map_strs[DIMS];

	/* The whole of what the caller sees, and the stride that steps a slab
	 * along the coil axis of it and of the sensitivities. */
	long full_out_dims[DIMS];
	long full_out_strs[DIMS];
	long out_slab_offset;
	long map_slab_offset;

	/* Where the coils lie in the samples.  BART's tools keep them on
	 * COIL_DIM, so a slab is a stride along it; the torch layout puts them
	 * slowest, so the whole is one coil's samples, contiguous, one block
	 * after another. */
	bool coils_slowest;
	long block_dims[DIMS];
	long block_strs[DIMS];
	long block_size;

	/* The sensitivities, and whether they are ours to free: the Cartesian
	 * operator scales and modulates a copy, the non-Cartesian one reads
	 * the caller's array in place. */
	const complex float* maps;
	complex float* owned;

	/* Or the sensitivities as k-space kernels: the centre of their
	 * spectrum, which is all a smooth map carries.  A slab is inflated
	 * into a buffer of its own at the top of each iteration and nothing
	 * the size of the whole bank is ever resident.  Padding a cropped
	 * unitary spectrum back to the grid it was taken on needs no scaling,
	 * so what comes back is the map band-limited and nothing else. */
	const complex float* kernels;
	long kern_dims[DIMS];
	long kern_strs[DIMS];
	long kern_slab_offset;

	/* The transform for one slab: a Fourier transform on a grid, a NUFFT
	 * off one. */
	const struct linop_s* slab;
};

static DEF_TYPEID(sense_s);

/* Where a slab starts, as an index into an array laid out over every coil.
 * Strides are in bytes and this indexes complex floats. */
static long slab_at(long stride, long coil)
{
	return coil * stride / (long)CFL_SIZE;
}

/* Whether a slab has to be brought to where the arithmetic is.
 *
 * A bank left on the host is a bank the card never holds: the loop already
 * reads one slab at a time, so one slab at a time is all that has to cross. */
static bool staged(const struct sense_s* d, const void* ref)
{
	return (NULL == d->kernels)
		&& (0 == bartorch_on_device(d->maps))
		&& (0 != bartorch_on_device(ref));
}

/* Whether a slab has to be put together at all, rather than read where it
 * lies: kernels have to be inflated, a bank on the host brought over. */
static bool assembled(const struct sense_s* d, const void* ref)
{
	return (NULL != d->kernels) || staged(d, ref);
}

/* A slab's worth of sensitivities, made or fetched into `into`.
 *
 * Kernels are laid out, padded back on to the image grid and transformed,
 * which is the map they were taken from with everything above the kernel's
 * own band removed.  A bank on the host is copied across as it stands. */
static void fetch_slab(const struct sense_s* d, long coil, complex float* into)
{
	long mdims[DIMS];
	md_copy_dims(DIMS, mdims, d->map_dims);
	mdims[COIL_DIM] = d->batch;

	if (NULL == d->kernels) {

		md_copy2(DIMS, mdims, d->slab_map_strs, into, d->map_strs,
				d->maps + slab_at(d->map_slab_offset, coil), CFL_SIZE);
		return;
	}

	long kdims[DIMS];
	md_copy_dims(DIMS, kdims, d->kern_dims);
	kdims[COIL_DIM] = d->batch;

	long kstrs[DIMS];
	md_calc_strides(DIMS, kstrs, kdims, CFL_SIZE);

	complex float* k = md_alloc_sameplace(DIMS, kdims, CFL_SIZE, into);

	md_copy2(DIMS, kdims, kstrs, k, d->kern_strs,
			d->kernels + slab_at(d->kern_slab_offset, coil), CFL_SIZE);

	/* A kernel is a few samples across, so most of what a transform of the
	 * padded grid computes is transforms of zeros.  Taken an axis at a time,
	 * each axis is padded only when it is transformed: along the first only
	 * the lines through the kernel are transformed, along the second the
	 * planes through it, and only the third is the whole grid.  A centred
	 * unitary transform is one transform per axis whichever way it is taken,
	 * so this is the same map. */
	long sdims[DIMS];
	md_copy_dims(DIMS, sdims, kdims);

	/* Each axis's centred unitary transform is a modulation, a transform, the
	 * modulation again and a scale, and all of it but the transform is
	 * diagonal.  On a card the diagonal parts are taken out of the loop: the
	 * modulations before the transforms go on the kernel, a few samples
	 * across, and the ones after go on with the scales in one pass over the
	 * map at the end -- where the last axis alone would otherwise make three
	 * passes over the whole grid. */
	bool modulated = false;
	long grid[3];
	long off[3];
	float scale = 1.f;

#ifdef USE_CUDA
	modulated = cuda_ondevice(into) && (md_calc_size(3, mdims) < (1L << 31));

	for (int a = 0; a < 3; a++) {

		bool fft = MD_IS_SET(FFT_FLAGS, a) && (1 < mdims[a]);

		grid[a] = fft ? mdims[a] : 1;
		off[a] = labs(mdims[a] / 2 - kdims[a] / 2);

		if (fft)
			scale /= sqrtf((float)mdims[a]);

		modulated = modulated && (kdims[a] <= mdims[a]);
	}

	if (modulated)
		bartorch_cuda_modulate(kdims, md_calc_size(DIMS - 3, kdims + 3), grid, off, 1.f, k);
#endif

	complex float* cur = k;

	for (int a = 0; a < 3; a++) {

		long ndims[DIMS];
		md_copy_dims(DIMS, ndims, sdims);
		ndims[a] = mdims[a];

		complex float* next = (2 == a) ? into : md_alloc_sameplace(DIMS, ndims, CFL_SIZE, into);

		md_resize_center(DIMS, ndims, next, sdims, cur, CFL_SIZE);
		md_free(cur);

		if (MD_IS_SET(FFT_FLAGS, a) && (1 < ndims[a])) {

			if (modulated)
				ifft(DIMS, ndims, MD_BIT(a), next, next);
			else
				ifftuc(DIMS, ndims, MD_BIT(a), next, next);
		}

		cur = next;
		md_copy_dims(DIMS, sdims, ndims);
	}

	assert(md_check_equal_dims(DIMS, sdims, mdims, ~0UL));

#ifdef USE_CUDA
	if (modulated) {

		long zero[3] = { 0, 0, 0 };

		bartorch_cuda_modulate(mdims, md_calc_size(DIMS - 3, mdims + 3), grid, zero, scale, into);
	}
#else
	(void)grid;
	(void)off;
	(void)scale;
#endif
}

/* Somewhere to put a slab, when one has to be put together. */
static complex float* slab_buffer(const struct sense_s* d, const void* ref)
{
	long mdims[DIMS];
	md_copy_dims(DIMS, mdims, d->map_dims);
	mdims[COIL_DIM] = d->batch;

	return md_alloc_sameplace(DIMS, mdims, CFL_SIZE, ref);
}

/* Streams, where there are any.  BART hands every `md_` call the stream of
 * the OpenMP thread that issued it, so what decides whether a fetch and the
 * arithmetic can run at once is how many streams BART was asked for. */
#ifdef USE_CUDA
static int stream_count(void) { return cuda_set_stream_level(); }
static void stream_wait(void) { cuda_sync_stream(); }
#else
static int stream_count(void) { return 1; }
static void stream_wait(void) { }
#endif

/* What one slab does, whichever way the operator is being applied. */
typedef void (*slab_fn)(const struct sense_s* d, long coil, const complex float* map,
		const long* mstrs, bool last, void* ctx);

/* Slab buffers kept from one walk over the coils to the next.
 *
 * Every set of frequencies walks the same coils, so a walk that starts where
 * the one before ended finds that slab already put together.  Taken in
 * alternate directions, each set after the first skips one slab -- the one
 * that would otherwise be put together with nothing to overlap it. */
struct slab_walk {

	complex float* buf[2];
	long holds[2];		/* the first coil each buffer holds, or -1 */
	bool reverse;
};

/* Walk the coils a slab at a time, putting the next one together while the
 * current one is worked on.
 *
 * A slab that has to be assembled is the only thing there is to overlap: a
 * region of two threads puts the fetch on one stream and the arithmetic on
 * another, and the card does both at once.  The thread that fetches waits on
 * its own stream before the region closes, so what the next turn reads is
 * there; the region's own barrier does the rest.  Two buffers are enough
 * because a turn reads one while the other is being filled.
 *
 * With the sensitivities already beside the arithmetic there is nothing to
 * hide, and the loop reads them where they lie. */
static void drive_slabs(const struct sense_s* d, const void* ref, slab_fn fn, void* ctx, struct slab_walk* walk)
{
	if (!assembled(d, ref)) {

		for (long c = 0; c < d->coils; c += d->batch)
			fn(d, c, d->maps + slab_at(d->map_slab_offset, c), d->map_strs, c + d->batch >= d->coils, ctx);

		return;
	}

	struct slab_walk own = { { NULL, NULL }, { -1, -1 }, false };
	struct slab_walk* w = (NULL != walk) ? walk : &own;

	if (NULL == w->buf[0])
		w->buf[0] = slab_buffer(d, ref);

	const long* mstrs = d->slab_map_strs;

	bool overlap = (1 < stream_count()) && (d->batch < d->coils);

	if (overlap && (NULL == w->buf[1]))
		w->buf[1] = slab_buffer(d, ref);

	overlap = overlap && (NULL != w->buf[1]);

	long slabs = (d->coils + d->batch - 1) / d->batch;
	long order[slabs];

	for (long t = 0; t < slabs; t++)
		order[t] = (w->reverse ? slabs - 1 - t : t) * d->batch;

	/* The buffer the first slab is in, if the walk before left it there. */
	int b = (order[0] == w->holds[0]) ? 0 : (overlap && (order[0] == w->holds[1])) ? 1 : -1;

	if (-1 == b) {

		b = 0;
		fetch_slab(d, order[0], w->buf[0]);
		w->holds[0] = order[0];
	}

	for (long t = 0; t < slabs; t++) {

		long c = order[t];
		bool last = (t + 1 == slabs);

		if (!overlap) {

			if (w->holds[0] != c) {

				fetch_slab(d, c, w->buf[0]);
				w->holds[0] = c;
			}

			fn(d, c, w->buf[0], mstrs, last, ctx);
			continue;
		}

		long next = last ? -1 : order[t + 1];

		/* Armed here rather than once: BART forgets which level owns
		 * the streams as soon as anything asks for one from below it,
		 * and the fetch of the first slab does exactly that. */
		(void)stream_count();

#ifdef _OPENMP
#pragma omp parallel num_threads(2)
		{
			if (0 == omp_get_thread_num()) {

				fn(d, c, w->buf[b], mstrs, last, ctx);

				/* The barrier below waits for the host, not the
				 * card, and the next turn fills the buffer this
				 * one is still reading: what has been asked of
				 * the card has to have happened before then. */
				stream_wait();

			} else if (0 <= next) {

				fetch_slab(d, next, w->buf[b ^ 1]);
				stream_wait();
			}
		}
#else
		/* No second thread to fetch on, so the fetch follows the
		 * arithmetic rather than running beside it.  The buffers still
		 * alternate, so the walk is the same walk; only the overlap it
		 * was arranged for is gone. */
		fn(d, c, w->buf[b], mstrs, last, ctx);
		stream_wait();

		if (0 <= next) {

			fetch_slab(d, next, w->buf[b ^ 1]);
			stream_wait();
		}
#endif

		if (0 <= next) {

			w->holds[b ^ 1] = next;
			b ^= 1;
		}
	}

	w->reverse = !w->reverse;

	if (NULL == walk) {

		md_free(own.buf[1]);
		md_free(own.buf[0]);
	}
}

/* An operand where the arithmetic is.
 *
 * The arithmetic is on the card whenever one is in use, whatever the caller
 * hands over: a solver that keeps its vectors on the host sees an operator
 * that takes and returns host arrays, and between two applications the card
 * holds the operator and nothing of the solver's.  An image crosses whole,
 * once each way; the samples cross a slab at a time, which the loop does
 * already.  Where there is no card, or the operand is on it, it is used
 * where it lies. */
static bool crosses(const void* ptr)
{
	return !bartorch_on_device(ptr) && (0 <= bartorch_cuda_device());
}

static complex float* onto_card(const long dims[DIMS], const complex float* ptr, bool filled)
{
#ifdef USE_CUDA
	if (crosses(ptr)) {

		complex float* on = md_alloc_gpu(DIMS, dims, CFL_SIZE);

		if (filled && (0 != bartorch_cuda_copy_pageable(on, ptr, md_calc_size(DIMS, dims) * (long)CFL_SIZE)))
			md_copy(DIMS, dims, on, ptr, CFL_SIZE);

		return on;
	}
#else
	(void)dims; (void)filled;
#endif
	return (complex float*)ptr;
}

static void off_card(const long dims[DIMS], complex float* ptr, complex float* on, bool filled)
{
	if (on == ptr)
		return;

	if (filled && (0 != bartorch_cuda_copy_pageable(ptr, on, md_calc_size(DIMS, dims) * (long)CFL_SIZE)))
		md_copy(DIMS, dims, ptr, on, CFL_SIZE);

	md_free(on);
}

/* Each way of applying the operator, as what it does to one slab. */
struct slab_ctx {

	complex float* dst;
	const complex float* src;
	complex float* cim;
	complex float* out;	/* the samples a slab answers, forward and adjoint */
	complex float* nrm;	/* what the normal convolves, per slab */
};

/* A slab's samples into their place in the whole, and back out of it. */
static void put_samples(const struct sense_s* d, long coil, complex float* dst, const complex float* out)
{
	if (!d->coils_slowest) {

		md_copy2(DIMS, d->out_dims, d->full_out_strs, dst + slab_at(d->out_slab_offset, coil),
				d->out_strs, out, CFL_SIZE);
		return;
	}

	/* The slab's coils one after another, each into its block of the whole. */
	long coil_step = d->out_strs[COIL_DIM] / (long)CFL_SIZE;

	for (long b = 0; b < d->batch; b++)
		md_copy2(DIMS, d->block_dims, d->block_strs, dst + (coil + b) * d->block_size,
				d->out_strs, out + b * coil_step, CFL_SIZE);
}

static void take_samples(const struct sense_s* d, long coil, complex float* out, const complex float* src)
{
	if (!d->coils_slowest) {

		md_copy2(DIMS, d->out_dims, d->out_strs, out,
				d->full_out_strs, src + slab_at(d->out_slab_offset, coil), CFL_SIZE);
		return;
	}

	long coil_step = d->out_strs[COIL_DIM] / (long)CFL_SIZE;

	for (long b = 0; b < d->batch; b++)
		md_copy2(DIMS, d->block_dims, d->out_strs, out + b * coil_step,
				d->block_strs, src + (coil + b) * d->block_size, CFL_SIZE);
}

static void forward_slab(const struct sense_s* d, long coil, const complex float* map,
		const long* mstrs, bool last, void* _c)
{
	(void)last;
	struct slab_ctx* c = _c;

	md_ztenmul2(DIMS, d->slab_dims, d->cim_strs, c->cim, d->img_strs, c->src, mstrs, map);

	linop_forward(d->slab, DIMS, d->out_dims, c->out, DIMS, d->cim_dims, c->cim);

	put_samples(d, coil, c->dst, c->out);
}

static void adjoint_slab(const struct sense_s* d, long coil, const complex float* map,
		const long* mstrs, bool last, void* _c)
{
	(void)last;
	struct slab_ctx* c = _c;

	take_samples(d, coil, c->out, c->src);

	linop_adjoint(d->slab, DIMS, d->cim_dims, c->cim, DIMS, d->out_dims, c->out);

	md_zfmacc2(DIMS, d->slab_dims, d->img_strs, c->dst, d->cim_strs, c->cim, mstrs, map);
}

/* The same for a sampled-only Cartesian transform on a card: the sensitivity
 * goes on as a coefficient is read into the transform and comes off as the
 * adjoint writes into the image, so no coil image is made. */
static void forward_slab_gridded(const struct sense_s* d, long coil, const complex float* map,
		const long* mstrs, bool last, void* _c)
{
	(void)last;
	struct slab_ctx* c = _c;

	bartorch_grid_forward_sense(d->slab, c->out, c->src, mstrs, map);

	put_samples(d, coil, c->dst, c->out);
}

static void adjoint_slab_gridded(const struct sense_s* d, long coil, const complex float* map,
		const long* mstrs, bool last, void* _c)
{
	(void)last;
	struct slab_ctx* c = _c;

	take_samples(d, coil, c->out, c->src);

	bartorch_grid_adjoint_sense(d->slab, c->dst, c->out, mstrs, map);
}

static void normal_slab(const struct sense_s* d, long coil, const complex float* map,
		const long* mstrs, bool last, void* _c)
{
	(void)coil;
	(void)last;
	struct slab_ctx* c = _c;

	md_ztenmul2(DIMS, d->slab_dims, d->cim_strs, c->cim, d->img_strs, c->src, mstrs, map);

	linop_normal_unchecked(d->slab, c->nrm, c->cim);

	md_zfmacc2(DIMS, d->slab_dims, d->img_strs, c->dst, d->cim_strs, c->nrm, mstrs, map);
}

/* One coil against the set that is loaded.
 *
 * The set is already where the transform will look for it, so what this costs
 * is the coil: its sensitivities on, the convolution, and its sensitivities
 * off into the answer.  Clearing what the convolution accumulates into is a
 * pass over the coil images, which is what walking the sets outside costs --
 * against the whole function crossing again, which is what it saves. */
static void normal_slab_coset(const struct sense_s* d, long coil, const complex float* map,
		const long* mstrs, bool last, void* _c)
{
	(void)coil;
	struct slab_ctx* c = _c;

	md_ztenmul2(DIMS, d->slab_dims, d->cim_strs, c->cim, d->img_strs, c->src, mstrs, map);

	md_clear(DIMS, d->cim_dims, c->nrm, CFL_SIZE);
	bartorch_nufft_coset_normal(d->slab, c->nrm, c->cim, last);

	md_zfmacc2(DIMS, d->slab_dims, d->img_strs, c->dst, d->cim_strs, c->nrm, mstrs, map);
}

/* The same, with the sensitivity folded into the transform.
 *
 * What the transform does with it is multiply a coefficient by the map as it
 * reads it and by the map's conjugate as it writes it, so neither the coil
 * image the map would have been multiplied into nor the one the answer would
 * have landed in is ever made.  At 256^3 over four coefficients each of those
 * is half a gigabyte. */
static void normal_slab_folded(const struct sense_s* d, long coil, const complex float* map,
		const long* mstrs, bool last, void* _c)
{
	(void)coil;
	struct slab_ctx* c = _c;

	bartorch_nufft_coset_normal_sense(d->slab, c->dst, c->src, mstrs, map, last);
}

/* The same for a Cartesian transform: the sensitivity goes on as a
 * coefficient is read into the transform and comes off as it is written
 * into the answer, so here too no coil image is made. */
static void normal_slab_gridded(const struct sense_s* d, long coil, const complex float* map,
		const long* mstrs, bool last, void* _c)
{
	(void)coil;
	(void)last;
	struct slab_ctx* c = _c;

	bartorch_grid_normal_sense(d->slab, c->dst, c->src, mstrs, map);
}

static void sense_forward(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(sense_s, _d);

	complex float* src_on = onto_card(d->img_dims, src, true);

	/* A sampled-only Cartesian transform folds the sensitivity into its
	 * transforms, where they run through cuFFT on the card. */
	bool gridded = d->fold && (1 == d->slab_dims[MAPS_DIM])
			&& (0 != bartorch_grid_folds_samples(d->slab, src_on));

	struct slab_ctx c = {

		.dst = dst, .src = src_on,
		.cim = gridded ? NULL : md_alloc_sameplace(DIMS, d->cim_dims, CFL_SIZE, src_on),
		.out = md_alloc_sameplace(DIMS, d->out_dims, CFL_SIZE, src_on),
	};

	drive_slabs(d, src_on, gridded ? forward_slab_gridded : forward_slab, &c, NULL);

	md_free(c.out);

	if (NULL != c.cim)
		md_free(c.cim);

	off_card(d->img_dims, (complex float*)src, src_on, false);

#ifdef USE_CUDA
	if (crosses(src))
		bartorch_cuda_memcache_clear_all();
#endif
}

static void sense_adjoint(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(sense_s, _d);

	complex float* dst_on = onto_card(d->img_dims, dst, false);

	/* The caller's pages are faulted in while the card works (cuda.c). */
	void* faulting = (dst_on != dst) ? bartorch_host_prefault_begin(dst, md_calc_size(DIMS, d->img_dims) * (long)CFL_SIZE) : NULL;

	bool gridded = d->fold && (1 == d->slab_dims[MAPS_DIM])
			&& (0 != bartorch_grid_folds_samples(d->slab, dst_on));

	struct slab_ctx c = {

		.dst = dst_on, .src = src,
		.cim = gridded ? NULL : md_alloc_sameplace(DIMS, d->cim_dims, CFL_SIZE, dst_on),
		.out = md_alloc_sameplace(DIMS, d->out_dims, CFL_SIZE, dst_on),
	};

	md_clear(DIMS, d->img_dims, dst_on, CFL_SIZE);

	drive_slabs(d, dst_on, gridded ? adjoint_slab_gridded : adjoint_slab, &c, NULL);

	md_free(c.out);

	if (NULL != c.cim)
		md_free(c.cim);

	bartorch_host_prefault_end(faulting);
	off_card(d->img_dims, dst, dst_on, true);

#ifdef USE_CUDA
	if (crosses(dst))
		bartorch_cuda_memcache_clear_all();
#endif
}

/* A^H A, which is where the memory goes: a non-Cartesian transform answers
 * its normal as a convolution over the doubled grid, and asking for it a slab
 * at a time is what keeps that grid off the card. */
static void sense_normal(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(sense_s, _d);

	/* A function held off the card crosses once for each set that is used.
	 * With the coils outside and the sets inside, every coil brings the
	 * whole of it over again; with the sets outside it crosses once for the
	 * application.  On eight coils that is eight times less over the bus. */
	int cosets = bartorch_nufft_cosets(d->slab);

	/* Folded, the two coil images are not needed at all.  It takes a
	 * transform that reads and writes a coefficient at a time, and one map
	 * per coil rather than a set of them to contract. */
	bool folds = (0 != cosets) && (0 != bartorch_nufft_coset_folds(d->slab))
			&& (1 == d->slab_dims[MAPS_DIM]) && d->fold;

	if (folds)
#pragma omp atomic
		sense_counters[SN_FOLDED]++;

	complex float* src_on = onto_card(d->img_dims, src, true);
	complex float* dst_on = onto_card(d->img_dims, dst, false);

	/* A Cartesian transform folds the sensitivity in the same way, where its
	 * normal runs through cuFFT on the card the image is on. */
	bool gridded = !folds && d->fold && (1 == d->slab_dims[MAPS_DIM])
			&& (0 != bartorch_grid_folds(d->slab, dst_on));

	/* The caller's pages are faulted in while the card works (cuda.c). */
	void* faulting = (dst_on != dst) ? bartorch_host_prefault_begin(dst, md_calc_size(DIMS, d->img_dims) * (long)CFL_SIZE) : NULL;

	struct slab_ctx c = {

		.dst = dst_on, .src = src_on,
		.cim = (folds || gridded) ? NULL : md_alloc_sameplace(DIMS, d->cim_dims, CFL_SIZE, dst_on),
		.nrm = (folds || gridded) ? NULL : md_alloc_sameplace(DIMS, d->cim_dims, CFL_SIZE, dst_on),
	};

	md_clear(DIMS, d->img_dims, dst_on, CFL_SIZE);

	if (gridded) {

		drive_slabs(d, dst_on, normal_slab_gridded, &c, NULL);

	} else if (0 == cosets) {

		drive_slabs(d, dst_on, normal_slab, &c, NULL);

	} else {

		/* Each set walks the coils the other way round from the one
		 * before, so it starts on the slab the last one ended on. */
		struct slab_walk walk = { { NULL, NULL }, { -1, -1 }, false };

		bartorch_nufft_coset_begin(d->slab, dst_on);

		for (int i = 0; i < cosets; i++) {

			bartorch_nufft_coset_use(d->slab, i);
			drive_slabs(d, dst_on, folds ? normal_slab_folded : normal_slab_coset, &c, &walk);
		}

		bartorch_nufft_coset_end(d->slab);

		md_free(walk.buf[1]);
		md_free(walk.buf[0]);
	}

	if (NULL != c.nrm)
		md_free(c.nrm);

	if (NULL != c.cim)
		md_free(c.cim);

	off_card(d->img_dims, (complex float*)src, src_on, false);
	bartorch_host_prefault_end(faulting);
	off_card(d->img_dims, dst, dst_on, true);

	/* BART keeps every block it frees for the next application, which
	 * serves the hundreds of transform workspaces an application asks for
	 * -- and leaves the card holding, between applications, whatever the
	 * last one freed.  For a caller whose arrays are on the host the card
	 * holds the operator and nothing else, so the cache is handed back;
	 * the forward and the adjoint do the same. */
#ifdef USE_CUDA
	if (crosses(src))
		bartorch_cuda_memcache_clear_all();
#endif
}

static void sense_del(const linop_data_t* _d)
{
	const auto d = CAST_DOWN(sense_s, _d);

	linop_free(d->slab);
	md_free(d->owned);
	xfree(d);
}

/* Whether the coils can be walked a slab at a time.
 *
 * They can when nothing else is laid out along them: a pattern or a basis
 * that varies across coils would have to be sliced with them, and the
 * sensitivities have to carry the coils the images do. */
static bool sliceable(const long max_dims[DIMS], const long map_dims[DIMS], const long out_dims[DIMS],
		unsigned long shared_img_flags)
{
	long coils = max_dims[COIL_DIM];

	if ((0 == coil_batch) || (coils < 2))
		return false;

	if ((map_dims[COIL_DIM] != coils) || (out_dims[COIL_DIM] != coils))
		return false;

	/* An image axis shared across coils is the one thing the loop cannot
	 * take apart, because a slab would no longer own its output. */
	if (MD_IS_SET(shared_img_flags, COIL_DIM))
		return false;

	return true;
}

/* How many coils a slab actually holds.
 *
 * The loop walks the bank in steps of the slab, and the transform is built for
 * exactly that many coils: a last slab with fewer of them reads sensitivities
 * that are not there and writes its answer past the end of the samples.  So
 * the slab is the largest divisor of the coil count that is no larger than the
 * one asked for -- the one asked for whenever it divides the coils, and one
 * when nothing else does.
 */
static long slab_size(long coils, long want)
{
	for (long n = MIN(want, coils); n > 1; n--)
		if (0 == coils % n)
			return n;

	return 1;
}

/* The parts of the operator that do not depend on which transform it is. */
static struct sense_s* sense_slabs(const long max_dims[DIMS], const long map_dims[DIMS],
		const long out_dims[DIMS], unsigned long shared_img_flags)
{
	PTR_ALLOC(struct sense_s, d);
	SET_TYPEID(sense_s, d);

	d->coils = max_dims[COIL_DIM];
	d->batch = slab_size(d->coils, (long)coil_batch);
	d->fold = (0 != fold_maps);
	d->maps = NULL;
	d->owned = NULL;
	d->kernels = NULL;
	d->slab = NULL;
	d->kern_slab_offset = 0;

	md_copy_dims(DIMS, d->map_dims, map_dims);

	md_copy_dims(DIMS, d->slab_dims, max_dims);
	d->slab_dims[COIL_DIM] = d->batch;

	md_select_dims(DIMS, ~MAPS_FLAG, d->cim_dims, d->slab_dims);
	md_select_dims(DIMS, ~COIL_FLAG & ~shared_img_flags, d->img_dims, max_dims);

	md_calc_strides(DIMS, d->cim_strs, d->cim_dims, CFL_SIZE);
	md_calc_strides(DIMS, d->img_strs, d->img_dims, CFL_SIZE);

	/* The sensitivities and the output are read and written in place, so a
	 * slab steps along the strides of the whole. */
	md_calc_strides(DIMS, d->map_strs, map_dims, CFL_SIZE);
	d->map_slab_offset = d->map_strs[COIL_DIM];

	long slab_map_dims[DIMS];
	md_copy_dims(DIMS, slab_map_dims, map_dims);
	slab_map_dims[COIL_DIM] = d->batch;
	md_calc_strides(DIMS, d->slab_map_strs, slab_map_dims, CFL_SIZE);

	return PTR_PASS(d);
}

/* What a slab answers with is the transform's to say, not this: a subspace
 * basis contracts its coefficients away on the k-space side, so the samples
 * that come back are not the dimensions the operator was asked for.  The
 * whole is that shape with every coil in it: on COIL_DIM for BART's tools,
 * slowest for the torch layout. */
static void sense_output_from(struct sense_s* d, bool coils_slowest)
{
	auto cod = linop_codomain(d->slab);

	md_copy_dims(DIMS, d->out_dims, cod->dims);
	md_calc_strides(DIMS, d->out_strs, d->out_dims, CFL_SIZE);

	d->coils_slowest = coils_slowest;

	if (!coils_slowest) {

		md_copy_dims(DIMS, d->full_out_dims, d->out_dims);
		d->full_out_dims[COIL_DIM] = d->coils;
		md_calc_strides(DIMS, d->full_out_strs, d->full_out_dims, CFL_SIZE);
		d->out_slab_offset = d->full_out_strs[COIL_DIM];
		return;
	}

	md_copy_dims(DIMS, d->block_dims, d->out_dims);
	d->block_dims[COIL_DIM] = 1;
	md_calc_strides(DIMS, d->block_strs, d->block_dims, CFL_SIZE);
	d->block_size = md_calc_size(DIMS, d->block_dims);

	/* The whole is the blocks one coil after another, so the coils go on the
	 * first axis past everything a block has: BART's own coil axis where a
	 * block has nothing beyond it, a later one where encoding axes lie there. */
	int last = DIMS - 1;

	while ((last > 0) && (1 == d->block_dims[last]))
		last--;

	int coil_axis = (last < COIL_DIM) ? COIL_DIM : last + 1;

	if (coil_axis >= DIMS)
		error("bartorch: the samples leave no axis for the coils\n");

	md_copy_dims(DIMS, d->full_out_dims, d->block_dims);
	d->full_out_dims[coil_axis] = d->coils;
}

static struct linop_s* sense_operator(struct sense_s* d)
{
	debug_printf(DP_DEBUG1, "SENSE over %ld coils, %ld at a time\n", d->coils, d->batch);

#pragma omp atomic
	sense_counters[SN_FUSED]++;

	return linop_create(DIMS, d->full_out_dims, DIMS, d->img_dims, CAST_UP(d),
			sense_forward, sense_adjoint, sense_normal, NULL, sense_del);
}

static void chained(void)
{
#pragma omp atomic
	sense_counters[SN_CHAINED]++;
}

/* y = F S x, on a grid. */
struct linop_s* sense_init(unsigned long shared_img_flags, const long max_dims[DIMS],
		unsigned long sens_flags, const complex float* sens)
{
	long map_dims[DIMS];
	long ksp_dims[DIMS];

	md_select_dims(DIMS, sens_flags, map_dims, max_dims);
	md_select_dims(DIMS, ~MAPS_FLAG, ksp_dims, max_dims);

	if (!sliceable(max_dims, map_dims, ksp_dims, shared_img_flags)) {

		chained();
		return bart_sense_init(shared_img_flags, max_dims, sens_flags, sens);
	}

	struct sense_s* d = sense_slabs(max_dims, map_dims, ksp_dims, shared_img_flags);

	/* The scaling and the modulation `maps_create` folds into the
	 * sensitivities, kept here because the loop reads them many times. */
	d->owned = md_alloc_sameplace(DIMS, map_dims, CFL_SIZE, sens);
	fftscale(DIMS, map_dims, FFT_FLAGS, d->owned, sens);
	fftmod(DIMS, map_dims, FFT_FLAGS, d->owned, d->owned);
	d->maps = d->owned;

	long slab_ksp_dims[DIMS];
	md_copy_dims(DIMS, slab_ksp_dims, ksp_dims);
	slab_ksp_dims[COIL_DIM] = d->batch;

	d->slab = linop_fft_create(DIMS, slab_ksp_dims, FFT_FLAGS);
	sense_output_from(d, false);

	return sense_operator(d);
}

/* y = A S x, off one. */
const struct linop_s* sense_nc_init(const long max_dims[DIMS], const long map_dims[DIMS], const complex float* maps,
		const long ksp_dims[DIMS],
		const long traj_dims[DIMS], const complex float* traj, const struct nufft_conf_s* _conf,
		const long wgs_dims[DIMS], const complex float* weights,
		const long basis_dims[DIMS], const complex float* basis,
		const struct linop_s** fft_opp, unsigned long shared_img_dims)
{
	long ksp_dims2[DIMS];
	md_copy_dims(DIMS, ksp_dims2, ksp_dims);
	ksp_dims2[COEFF_DIM] = max_dims[COEFF_DIM];

	bool sliced = sliceable(max_dims, map_dims, ksp_dims2, shared_img_dims)
		&& ((NULL == weights) || (1 == wgs_dims[COIL_DIM]))
		&& ((NULL == basis) || (1 == basis_dims[COIL_DIM]));

	if (!sliced) {

		chained();
		return bart_sense_nc_init(max_dims, map_dims, maps, ksp_dims, traj_dims, traj, _conf,
				wgs_dims, weights, basis_dims, basis, fft_opp, shared_img_dims);
	}

	struct sense_s* d = sense_slabs(max_dims, map_dims, ksp_dims2, shared_img_dims);

	long slab_ksp_dims[DIMS];
	md_copy_dims(DIMS, slab_ksp_dims, ksp_dims2);
	slab_ksp_dims[COIL_DIM] = d->batch;

	d->maps = maps;
	d->slab = nufft_create2(DIMS, slab_ksp_dims, d->cim_dims, traj_dims, traj,
			(weights ? wgs_dims : NULL), weights,
			(basis ? basis_dims : NULL), basis, *_conf);

	sense_output_from(d, false);

	/* The caller reads the point spread function off this and imports one
	 * into it; a slab's transform carries the same one, because a point
	 * spread function has no coil axis. */
	if (NULL != fft_opp)
		*fft_opp = linop_clone(d->slab);

	return sense_operator(d);
}


/* Hold the sensitivities the way the caller has them: as maps the loop reads
 * where they lie, or as the kernels it inflates a slab at a time. */
static void sense_hold(struct sense_s* d, const long sens_dims[DIMS], const complex float* sens, int kernels)
{
	if (0 == kernels) {

		d->maps = sens;
		return;
	}

	d->kernels = sens;
	md_copy_dims(DIMS, d->kern_dims, sens_dims);
	md_calc_strides(DIMS, d->kern_strs, sens_dims, CFL_SIZE);
	d->kern_slab_offset = d->kern_strs[COIL_DIM];

	d->map_slab_offset = 0;
}

/* What a caller is told when the bank is held as kernels and the loop that
 * inflates them is not going to run. */
static void kernels_need_the_loop(void)
{
	error("bartorch: sensitivities held as kernels are what the coil loop is "
		"for, and this arrangement cannot be sliced into one; inflate "
		"them with bartorch.kernels_to_maps first\n");
}

/* A SENSE operator built here rather than by BART, from sensitivities held
 * either way.
 *
 * This is what a caller reaches when the bank itself is what will not fit:
 * kernels are a few kilobytes a coil against an image apiece, and the loop
 * inflates only the slab it is about to use.  BART's own tools hand over
 * dense maps and get the same operator over them.
 */
const struct linop_s* bartorch_sense_operator(const long max_dims[DIMS], const long sens_dims[DIMS],
		const complex float* sens, int kernels, const long ksp_dims[DIMS],
		const long traj_dims[DIMS], const complex float* traj,
		const long wgh_dims[DIMS], const complex float* weights,
		const long bas_dims[DIMS], const complex float* basis,
		const struct nufft_conf_s* conf, int modulated)
{
	if ((NULL != traj) && (0 != modulated))
		error("bartorch: the modulated convention is the grid's; off it there is only one\n");

	if ((NULL == traj) && ((NULL != weights) || (NULL != basis)))
		error("bartorch: weights and a basis belong to a non-Cartesian transform\n");

	long map_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS | COIL_FLAG | MAPS_FLAG, map_dims, max_dims);

	/* The transform takes the coefficients the image carries and a basis
	 * contracts them into the frames the samples have, so what it is asked
	 * for carries both -- as `sense_nc_init` asks for it. */
	long ksp_dims2[DIMS];
	md_copy_dims(DIMS, ksp_dims2, ksp_dims);
	ksp_dims2[COEFF_DIM] = max_dims[COEFF_DIM];

	/* The same condition the tools' own entry points apply before they slice.
	 * Without it a slab of no coils -- which is what `set_coil_batch(0)`
	 * asks for, and what it means by leaving BART its own operator -- walks
	 * the bank in steps of nothing. */
	bool sliced = sliceable(max_dims, map_dims, ksp_dims2, 0UL)
		&& ((NULL == weights) || (1 == wgh_dims[COIL_DIM]))
		&& ((NULL == basis) || (1 == bas_dims[COIL_DIM]));

	if (!sliced) {

		if (0 != kernels)
			kernels_need_the_loop();

		chained();

		if (NULL == traj) {

			if (0 != modulated) {

				/* The flags `pics` gives it, so that what comes
				 * back is the operator the tool builds and not
				 * one like it -- which is what a caller asking
				 * for this convention is after. */
				unsigned long map_flags = FFT_FLAGS | SENS_FLAGS
					| md_nontriv_dims(DIMS, sens_dims);

				return bart_sense_init(0UL, max_dims, map_flags, sens);
			}

			/* Centred, as every slab of this operator is.  BART's
			 * own chain is the other convention, so it cannot
			 * stand in here: the coils and the transform are put
			 * together directly instead. */
			long img_dims[DIMS];
			md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);

			long cim_dims[DIMS];
			md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, max_dims);

			return linop_chain_FF(linop_fmac_dims_create(DIMS, cim_dims, img_dims, sens_dims, sens),
					linop_fftc_create(DIMS, cim_dims, FFT_FLAGS));
		}

		return bart_sense_nc_init(max_dims, map_dims, sens, ksp_dims, traj_dims, traj, conf,
				wgh_dims, weights, bas_dims, basis, NULL, 0UL);
	}

	struct sense_s* d = sense_slabs(max_dims, map_dims, ksp_dims2, 0UL);
	sense_hold(d, sens_dims, sens, kernels);

	long slab_ksp_dims[DIMS];
	md_copy_dims(DIMS, slab_ksp_dims, ksp_dims2);
	slab_ksp_dims[COIL_DIM] = d->batch;

	/* Centred, which is what `bartorch.tools.fft` is and so what a caller
	 * who chains this against one will expect.  BART's own SENSE operator
	 * puts the same centring somewhere else -- a scale and a modulation
	 * folded into the sensitivities, and the plain transform after them --
	 * which leaves the samples modulated, and is what `pics` works in.
	 * That convention is reached by asking for it, not by the slab. */
	if (NULL == traj) {

		if (0 == modulated) {

			d->slab = linop_fftc_create(DIMS, slab_ksp_dims, FFT_FLAGS);

		} else {

			if (0 != kernels)
				error("bartorch: the modulated convention folds a scale and a "
					"modulation into the sensitivities, which is done on the "
					"whole grid and so cannot be done to a kernel; ask for the "
					"centred convention, or inflate the kernels first\n");

			if (!md_check_equal_dims(DIMS, map_dims, sens_dims, ~0UL))
				error("bartorch: the modulation folded into the sensitivities is "
					"the grid's, so this convention needs a bank on the grid "
					"rather than one broadcast onto it\n");

			d->owned = md_alloc_sameplace(DIMS, map_dims, CFL_SIZE, sens);
			fftscale(DIMS, map_dims, FFT_FLAGS, d->owned, sens);
			fftmod(DIMS, map_dims, FFT_FLAGS, d->owned, d->owned);
			d->maps = d->owned;

			d->slab = linop_fft_create(DIMS, slab_ksp_dims, FFT_FLAGS);
		}

	} else
		d->slab = nufft_create2(DIMS, slab_ksp_dims, d->cim_dims, traj_dims, traj,
				(weights ? wgh_dims : NULL), weights,
				(basis ? bas_dims : NULL), basis, *conf);

	sense_output_from(d, true);

	return sense_operator(d);
}

/* Provided by grid.c: a Cartesian slab's transform, pattern and basis, with
 * the normal that transforms only the axes the pattern varies along. */
extern const struct linop_s* grid_transform_create(const long cim_dims[DIMS],
		const long pat_dims[DIMS], const complex float* pattern,
		const long bas_dims[DIMS], const complex float* basis, int toeplitz);

/* The Cartesian SENSE encoding with its pattern and subspace basis inside the
 * coil loop.
 *
 * Chained in Python, the pattern and the basis are operators of their own,
 * which run where their arrays are: for a caller whose arrays are on the host
 * the whole k-space crosses back for them, and the normal crosses it twice.
 * Here they are part of the transform a slab carries, so what crosses is the
 * image, once each way. */
const struct linop_s* bartorch_cartesian_operator(const long max_dims[DIMS], const long sens_dims[DIMS],
		const complex float* sens, int kernels,
		const long pat_dims[DIMS], const complex float* pattern,
		const long bas_dims[DIMS], const complex float* basis, int toeplitz)
{
	long map_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS | COIL_FLAG | MAPS_FLAG, map_dims, max_dims);

	long cim_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, max_dims);

	if (!sliceable(max_dims, map_dims, cim_dims, 0UL)) {

		if (0 != kernels)
			kernels_need_the_loop();

		chained();

		long img_dims[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);

		return linop_chain_FF(linop_fmac_dims_create(DIMS, cim_dims, img_dims, sens_dims, sens),
				grid_transform_create(cim_dims, pat_dims, pattern, bas_dims, basis, toeplitz));
	}

	struct sense_s* d = sense_slabs(max_dims, map_dims, cim_dims, 0UL);
	sense_hold(d, sens_dims, sens, kernels);

	d->slab = grid_transform_create(d->cim_dims, pat_dims, pattern, bas_dims, basis, toeplitz);

	sense_output_from(d, true);

	return sense_operator(d);
}

/* Provided by grid.c: the slab transform over sampled-only k-space. */
extern const struct linop_s* grid_sampled_create(const long cim_dims[DIMS], long T, long S, int components,
		const long* positions, const long bas_dims[DIMS], const complex float* basis,
		int kspace_readout, int toeplitz);

/* The Cartesian SENSE encoding over sampled-only k-space: a table of phase
 * encodes per frame, with the whole readout along each, instead of k-space
 * over the whole plane.  On a card the forward and the adjoint fold the
 * sensitivity into the same cuFFT transforms the normal runs. */
const struct linop_s* bartorch_cartesian_sampled_operator(const long max_dims[DIMS], const long sens_dims[DIMS],
		const complex float* sens, int kernels, long frames, long shots, int components, const long* positions,
		const long bas_dims[DIMS], const complex float* basis, int kspace_readout, int toeplitz)
{
	long map_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS | COIL_FLAG | MAPS_FLAG, map_dims, max_dims);

	long cim_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, max_dims);

	if (!sliceable(max_dims, map_dims, cim_dims, 0UL)) {

		if (0 != kernels)
			kernels_need_the_loop();

		chained();

		long img_dims[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);

		return linop_chain_FF(linop_fmac_dims_create(DIMS, cim_dims, img_dims, sens_dims, sens),
				grid_sampled_create(cim_dims, frames, shots, components, positions, bas_dims, basis,
					kspace_readout, toeplitz));
	}

	struct sense_s* d = sense_slabs(max_dims, map_dims, cim_dims, 0UL);
	sense_hold(d, sens_dims, sens, kernels);

	d->slab = grid_sampled_create(d->cim_dims, frames, shots, components, positions, bas_dims, basis,
			kspace_readout, toeplitz);

	sense_output_from(d, true);

	return sense_operator(d);
}

/* Provided by grid.c: the wave slab transform, dense or over sampled-only
 * k-space. */
extern const struct linop_s* wave_transform_create(const long dom_dims[DIMS], long wx, const complex float* psf,
		int centred, const long pat_dims[DIMS], const complex float* pattern,
		const long bas_dims[DIMS], const complex float* basis, int toeplitz);
extern const struct linop_s* wave_sampled_create(const long dom_dims[DIMS], long wx, const complex float* psf,
		int centred, long T, long S, int components, const long* positions,
		const long bas_dims[DIMS], const complex float* basis, int toeplitz);

static const struct linop_s* wave_slab(const long cim_dims[DIMS], long wx, const complex float* psf, int centred,
		const long pat_dims[DIMS], const complex float* pattern,
		long frames, long shots, int components, const long* positions,
		const long bas_dims[DIMS], const complex float* basis, int toeplitz)
{
	if (NULL != positions)
		return wave_sampled_create(cim_dims, wx, psf, centred, frames, shots, components, positions,
				bas_dims, basis, toeplitz);

	return wave_transform_create(cim_dims, wx, psf, centred, pat_dims, pattern, bas_dims, basis, toeplitz);
}

/* The wave encoding with everything after the coils in the coil loop: the
 * zero-fill, the readout transform, the point spread function and the
 * phase-encode transform on the card a slab at a time, and on a card the
 * normal -- and a table's forward and adjoint -- through cuFFT's callbacks.
 * `positions` NULL is dense samples, with `pattern` if any. */
const struct linop_s* bartorch_wave_operator(const long max_dims[DIMS], const long sens_dims[DIMS],
		const complex float* sens, int kernels, long wx, const complex float* psf, int centred,
		const long pat_dims[DIMS], const complex float* pattern,
		long frames, long shots, int components, const long* positions,
		const long bas_dims[DIMS], const complex float* basis, int toeplitz)
{
	long map_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS | COIL_FLAG | MAPS_FLAG, map_dims, max_dims);

	long cim_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, max_dims);

	if (!sliceable(max_dims, map_dims, cim_dims, 0UL)) {

		if (0 != kernels)
			kernels_need_the_loop();

		chained();

		long img_dims[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);

		return linop_chain_FF(linop_fmac_dims_create(DIMS, cim_dims, img_dims, sens_dims, sens),
				wave_slab(cim_dims, wx, psf, centred, pat_dims, pattern, frames, shots, components, positions,
					bas_dims, basis, toeplitz));
	}

	struct sense_s* d = sense_slabs(max_dims, map_dims, cim_dims, 0UL);
	sense_hold(d, sens_dims, sens, kernels);

	d->slab = wave_slab(d->cim_dims, wx, psf, centred, pat_dims, pattern, frames, shots, components, positions,
			bas_dims, basis, toeplitz);

	sense_output_from(d, true);

	return sense_operator(d);
}

/* The coil multiply on its own: the same slab loop with nothing after it.
 *
 * What a caller wants when the transform beside the coils is not a Fourier
 * transform.  The wave encoding's is a chain of four -- a resize, a readout
 * transform, the point spread diagonal and the phase-encode transforms -- and
 * chaining that onto BART's own `fmac` would hold the whole bank; chaining it
 * onto this one holds a slab of it.
 *
 * Without slicing this is `linop_fmac` and nothing else, which is the operator
 * `linop.MultiplySum` builds, so the two answer alike and a test can say so.
 * With slicing it is that operator with the bank read, or inflated, a slab at
 * a time.
 */
const struct linop_s* bartorch_coils_operator(const long max_dims[DIMS], const long sens_dims[DIMS],
		const complex float* sens, int kernels)
{
	long map_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS | COIL_FLAG | MAPS_FLAG, map_dims, max_dims);

	/* The coil images the multiply answers with: every axis the operator
	 * has except the one the sets of maps are summed over. */
	long cim_dims[DIMS];
	md_select_dims(DIMS, ~MAPS_FLAG, cim_dims, max_dims);

	if (!sliceable(max_dims, map_dims, cim_dims, 0UL)) {

		if (0 != kernels)
			kernels_need_the_loop();

		chained();

		long img_dims[DIMS];
		md_select_dims(DIMS, ~COIL_FLAG, img_dims, max_dims);

		return linop_fmac_dims_create(DIMS, cim_dims, img_dims, map_dims, sens);
	}

	struct sense_s* d = sense_slabs(max_dims, map_dims, cim_dims, 0UL);
	sense_hold(d, sens_dims, sens, kernels);

	/* Nothing after the multiply, so a slab's transform is the identity and
	 * what it answers with is the slab of coil images itself. */
	d->slab = linop_identity_create(DIMS, d->cim_dims);

	sense_output_from(d, true);

	return sense_operator(d);
}
