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
extern void bartorch_nufft_coset_normal(const struct linop_s* op, complex float* dst, const complex float* src);
extern int bartorch_nufft_coset_folds(const struct linop_s* op);
extern void bartorch_nufft_coset_normal_sense(const struct linop_s* op,
		complex float* dst, const complex float* src,
		const long map_strs[], const complex float* map);
extern void bartorch_nufft_coset_end(const struct linop_s* op);

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

	md_resize_center(DIMS, mdims, into, kdims, k, CFL_SIZE);
	md_free(k);

	ifftuc(DIMS, mdims, FFT_FLAGS, into, into);
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
		const long* mstrs, void* ctx);

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
static void drive_slabs(const struct sense_s* d, const void* ref, slab_fn fn, void* ctx)
{
	if (!assembled(d, ref)) {

		for (long c = 0; c < d->coils; c += d->batch)
			fn(d, c, d->maps + slab_at(d->map_slab_offset, c), d->map_strs, ctx);

		return;
	}

	complex float* slab[2] = { slab_buffer(d, ref), NULL };
	const long* mstrs = d->slab_map_strs;

	bool overlap = (1 < stream_count()) && (d->batch < d->coils);

	if (overlap) {

		slab[1] = slab_buffer(d, ref);
		overlap = (NULL != slab[1]);
	}

	fetch_slab(d, 0, slab[0]);

	if (!overlap) {

		fn(d, 0, slab[0], mstrs, ctx);

		for (long c = d->batch; c < d->coils; c += d->batch) {

			fetch_slab(d, c, slab[0]);
			fn(d, c, slab[0], mstrs, ctx);
		}

	} else {

		int turn = 0;

		for (long c = 0; c < d->coils; c += d->batch, turn++) {

			long next = c + d->batch;

			/* Armed here rather than once: BART forgets which
			 * level owns the streams as soon as anything asks for
			 * one from below it, and the fetch of the first slab
			 * does exactly that. */
			(void)stream_count();

#pragma omp parallel num_threads(2)
			{
				if (0 == omp_get_thread_num()) {

					fn(d, c, slab[turn % 2], mstrs, ctx);

					/* The barrier below waits for the host,
					 * not the card, and the next turn fills
					 * the buffer this one is still reading:
					 * what has been asked of the card has
					 * to have happened before then. */
					stream_wait();

				} else if (next < d->coils) {

					fetch_slab(d, next, slab[(turn + 1) % 2]);
					stream_wait();
				}
			}
		}
	}

	md_free(slab[1]);
	md_free(slab[0]);
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

		if (filled)
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

	if (filled)
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

static void forward_slab(const struct sense_s* d, long coil, const complex float* map,
		const long* mstrs, void* _c)
{
	struct slab_ctx* c = _c;

	md_ztenmul2(DIMS, d->slab_dims, d->cim_strs, c->cim, d->img_strs, c->src, mstrs, map);

	linop_forward(d->slab, DIMS, d->out_dims, c->out, DIMS, d->cim_dims, c->cim);

	md_copy2(DIMS, d->out_dims, d->full_out_strs, c->dst + slab_at(d->out_slab_offset, coil),
			d->out_strs, c->out, CFL_SIZE);
}

static void adjoint_slab(const struct sense_s* d, long coil, const complex float* map,
		const long* mstrs, void* _c)
{
	struct slab_ctx* c = _c;

	md_copy2(DIMS, d->out_dims, d->out_strs, c->out,
			d->full_out_strs, c->src + slab_at(d->out_slab_offset, coil), CFL_SIZE);

	linop_adjoint(d->slab, DIMS, d->cim_dims, c->cim, DIMS, d->out_dims, c->out);

	md_zfmacc2(DIMS, d->slab_dims, d->img_strs, c->dst, d->cim_strs, c->cim, mstrs, map);
}

static void normal_slab(const struct sense_s* d, long coil, const complex float* map,
		const long* mstrs, void* _c)
{
	(void)coil;
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
		const long* mstrs, void* _c)
{
	(void)coil;
	struct slab_ctx* c = _c;

	md_ztenmul2(DIMS, d->slab_dims, d->cim_strs, c->cim, d->img_strs, c->src, mstrs, map);

	md_clear(DIMS, d->cim_dims, c->nrm, CFL_SIZE);
	bartorch_nufft_coset_normal(d->slab, c->nrm, c->cim);

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
		const long* mstrs, void* _c)
{
	(void)coil;
	struct slab_ctx* c = _c;

	bartorch_nufft_coset_normal_sense(d->slab, c->dst, c->src, mstrs, map);
}

static void sense_forward(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(sense_s, _d);

	complex float* src_on = onto_card(d->img_dims, src, true);

	struct slab_ctx c = {

		.dst = dst, .src = src_on,
		.cim = md_alloc_sameplace(DIMS, d->cim_dims, CFL_SIZE, src_on),
		.out = md_alloc_sameplace(DIMS, d->out_dims, CFL_SIZE, src_on),
	};

	drive_slabs(d, src_on, forward_slab, &c);

	md_free(c.out);
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

	struct slab_ctx c = {

		.dst = dst_on, .src = src,
		.cim = md_alloc_sameplace(DIMS, d->cim_dims, CFL_SIZE, dst_on),
		.out = md_alloc_sameplace(DIMS, d->out_dims, CFL_SIZE, dst_on),
	};

	md_clear(DIMS, d->img_dims, dst_on, CFL_SIZE);

	drive_slabs(d, dst_on, adjoint_slab, &c);

	md_free(c.out);
	md_free(c.cim);

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
			&& (1 == d->slab_dims[MAPS_DIM]) && fold_maps;

	if (folds)
#pragma omp atomic
		sense_counters[SN_FOLDED]++;

	complex float* src_on = onto_card(d->img_dims, src, true);
	complex float* dst_on = onto_card(d->img_dims, dst, false);

	struct slab_ctx c = {

		.dst = dst_on, .src = src_on,
		.cim = folds ? NULL : md_alloc_sameplace(DIMS, d->cim_dims, CFL_SIZE, dst_on),
		.nrm = folds ? NULL : md_alloc_sameplace(DIMS, d->cim_dims, CFL_SIZE, dst_on),
	};

	md_clear(DIMS, d->img_dims, dst_on, CFL_SIZE);

	if (0 == cosets) {

		drive_slabs(d, dst_on, normal_slab, &c);

	} else {

		bartorch_nufft_coset_begin(d->slab, dst_on);

		for (int i = 0; i < cosets; i++) {

			bartorch_nufft_coset_use(d->slab, i);
			drive_slabs(d, dst_on, folds ? normal_slab_folded : normal_slab_coset, &c);
		}

		bartorch_nufft_coset_end(d->slab);
	}

	if (NULL != c.nrm)
		md_free(c.nrm);

	if (NULL != c.cim)
		md_free(c.cim);

	off_card(d->img_dims, (complex float*)src, src_on, false);
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

/* The parts of the operator that do not depend on which transform it is. */
static struct sense_s* sense_slabs(const long max_dims[DIMS], const long map_dims[DIMS],
		const long out_dims[DIMS], unsigned long shared_img_flags)
{
	PTR_ALLOC(struct sense_s, d);
	SET_TYPEID(sense_s, d);

	d->coils = max_dims[COIL_DIM];
	d->batch = MIN((long)coil_batch, d->coils);
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
 * whole is that shape with every coil in it. */
static void sense_output_from(struct sense_s* d)
{
	auto cod = linop_codomain(d->slab);

	md_copy_dims(DIMS, d->out_dims, cod->dims);
	md_calc_strides(DIMS, d->out_strs, d->out_dims, CFL_SIZE);

	md_copy_dims(DIMS, d->full_out_dims, d->out_dims);
	d->full_out_dims[COIL_DIM] = d->coils;
	md_calc_strides(DIMS, d->full_out_strs, d->full_out_dims, CFL_SIZE);

	d->out_slab_offset = d->full_out_strs[COIL_DIM];
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
	sense_output_from(d);

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

	sense_output_from(d);

	/* The caller reads the point spread function off this and imports one
	 * into it; a slab's transform carries the same one, because a point
	 * spread function has no coil axis. */
	if (NULL != fft_opp)
		*fft_opp = linop_clone(d->slab);

	return sense_operator(d);
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
		const struct nufft_conf_s* conf)
{
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

	struct sense_s* d = sense_slabs(max_dims, map_dims, ksp_dims2, 0UL);

	if (0 != kernels) {

		d->kernels = sens;
		md_copy_dims(DIMS, d->kern_dims, sens_dims);
		md_calc_strides(DIMS, d->kern_strs, sens_dims, CFL_SIZE);
		d->kern_slab_offset = d->kern_strs[COIL_DIM];

		d->map_slab_offset = 0;

	} else {

		d->maps = sens;
	}

	long slab_ksp_dims[DIMS];
	md_copy_dims(DIMS, slab_ksp_dims, ksp_dims2);
	slab_ksp_dims[COIL_DIM] = d->batch;

	/* Centred, which is what `bartorch.tools.fft` is and so what a caller
	 * who chains this against one will expect; BART's own SENSE operator
	 * folds the same centring into the sensitivities instead. */
	if (NULL == traj)
		d->slab = linop_fftc_create(DIMS, slab_ksp_dims, FFT_FLAGS);
	else
		d->slab = nufft_create2(DIMS, slab_ksp_dims, d->cim_dims, traj_dims, traj,
				(weights ? wgh_dims : NULL), weights,
				(basis ? bas_dims : NULL), basis, *conf);

	sense_output_from(d);

	return sense_operator(d);
}
