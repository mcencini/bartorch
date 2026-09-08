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

#include "noncart/nufft.h"

#include "sense/model.h"

#include "include/bartorch.h"

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

/* Operators built since the last reset: with the coil loop, and as BART's own
 * chain because this could not serve them. */
enum { SN_FUSED, SN_CHAINED };
static long sense_counters[2];

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
	return sense_counters[(SN_FUSED == which) ? SN_FUSED : SN_CHAINED];
}

void bartorch_sense_reset_counters(void)
{
	sense_counters[SN_FUSED] = 0;
	sense_counters[SN_CHAINED] = 0;
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

/* The sensitivities this slab needs.
 *
 * Held densely they are already there and are read where they lie.  Held as
 * kernels they are made here: the slab's kernels are laid out, padded back on
 * to the image grid and transformed, which is the map they were taken from
 * with everything above the kernel's own band removed. */
static const complex float* slab_sens(const struct sense_s* d, long coil, complex float* scratch)
{
	if (NULL == d->kernels)
		return d->maps + slab_at(d->map_slab_offset, coil);

	long kdims[DIMS];
	md_copy_dims(DIMS, kdims, d->kern_dims);
	kdims[COIL_DIM] = d->batch;

	long mdims[DIMS];
	md_copy_dims(DIMS, mdims, d->map_dims);
	mdims[COIL_DIM] = d->batch;

	long kstrs[DIMS];
	md_calc_strides(DIMS, kstrs, kdims, CFL_SIZE);

	complex float* k = md_alloc_sameplace(DIMS, kdims, CFL_SIZE, scratch);

	md_copy2(DIMS, kdims, kstrs, k, d->kern_strs,
			d->kernels + slab_at(d->kern_slab_offset, coil), CFL_SIZE);

	md_resize_center(DIMS, mdims, scratch, kdims, k, CFL_SIZE);
	md_free(k);

	ifftuc(DIMS, mdims, FFT_FLAGS, scratch, scratch);

	return scratch;
}

/* A slab's worth of sensitivities, when they have to be made. */
static complex float* sens_scratch(const struct sense_s* d, const void* ref)
{
	if (NULL == d->kernels)
		return NULL;

	long mdims[DIMS];
	md_copy_dims(DIMS, mdims, d->map_dims);
	mdims[COIL_DIM] = d->batch;

	return md_alloc_sameplace(DIMS, mdims, CFL_SIZE, ref);
}

static void sense_forward(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(sense_s, _d);

	complex float* cim = md_alloc_sameplace(DIMS, d->cim_dims, CFL_SIZE, dst);
	complex float* out = md_alloc_sameplace(DIMS, d->out_dims, CFL_SIZE, dst);
	complex float* sens = sens_scratch(d, dst);

	for (long c = 0; c < d->coils; c += d->batch) {

		md_ztenmul2(DIMS, d->slab_dims, d->cim_strs, cim,
				d->img_strs, src,
				d->map_strs, slab_sens(d, c, sens));

		linop_forward(d->slab, DIMS, d->out_dims, out, DIMS, d->cim_dims, cim);

		md_copy2(DIMS, d->out_dims, d->full_out_strs, dst + slab_at(d->out_slab_offset, c),
				d->out_strs, out, CFL_SIZE);
	}

	md_free(sens);
	md_free(out);
	md_free(cim);
}

static void sense_adjoint(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(sense_s, _d);

	complex float* cim = md_alloc_sameplace(DIMS, d->cim_dims, CFL_SIZE, dst);
	complex float* out = md_alloc_sameplace(DIMS, d->out_dims, CFL_SIZE, dst);
	complex float* sens = sens_scratch(d, dst);

	md_clear(DIMS, d->img_dims, dst, CFL_SIZE);

	for (long c = 0; c < d->coils; c += d->batch) {

		md_copy2(DIMS, d->out_dims, d->out_strs, out,
				d->full_out_strs, src + slab_at(d->out_slab_offset, c), CFL_SIZE);

		linop_adjoint(d->slab, DIMS, d->cim_dims, cim, DIMS, d->out_dims, out);

		md_zfmacc2(DIMS, d->slab_dims, d->img_strs, dst,
				d->cim_strs, cim,
				d->map_strs, slab_sens(d, c, sens));
	}

	md_free(sens);
	md_free(out);
	md_free(cim);
}

/* A^H A, which is where the memory goes: a non-Cartesian transform answers
 * its normal as a convolution over the doubled grid, and asking for it a slab
 * at a time is what keeps that grid off the card. */
static void sense_normal(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(sense_s, _d);

	complex float* cim = md_alloc_sameplace(DIMS, d->cim_dims, CFL_SIZE, dst);
	complex float* nrm = md_alloc_sameplace(DIMS, d->cim_dims, CFL_SIZE, dst);
	complex float* sens = sens_scratch(d, dst);

	md_clear(DIMS, d->img_dims, dst, CFL_SIZE);

	for (long c = 0; c < d->coils; c += d->batch) {

		const complex float* map = slab_sens(d, c, sens);

		md_ztenmul2(DIMS, d->slab_dims, d->cim_strs, cim, d->img_strs, src, d->map_strs, map);

		linop_normal_unchecked(d->slab, nrm, cim);

		md_zfmacc2(DIMS, d->slab_dims, d->img_strs, dst, d->cim_strs, nrm, d->map_strs, map);
	}

	md_free(sens);
	md_free(nrm);
	md_free(cim);
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
		const long traj_dims[DIMS], const complex float* traj, const struct nufft_conf_s* conf)
{
	long map_dims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS | COIL_FLAG | MAPS_FLAG, map_dims, max_dims);

	struct sense_s* d = sense_slabs(max_dims, map_dims, ksp_dims, 0UL);

	if (0 != kernels) {

		d->kernels = sens;
		md_copy_dims(DIMS, d->kern_dims, sens_dims);
		md_calc_strides(DIMS, d->kern_strs, sens_dims, CFL_SIZE);
		d->kern_slab_offset = d->kern_strs[COIL_DIM];

		/* What a slab of inflated maps looks like, which is what the
		 * contraction reads rather than a window on to a whole bank. */
		long slab_map_dims[DIMS];
		md_copy_dims(DIMS, slab_map_dims, map_dims);
		slab_map_dims[COIL_DIM] = d->batch;
		md_calc_strides(DIMS, d->map_strs, slab_map_dims, CFL_SIZE);
		d->map_slab_offset = 0;

	} else {

		d->maps = sens;
	}

	long slab_ksp_dims[DIMS];
	md_copy_dims(DIMS, slab_ksp_dims, ksp_dims);
	slab_ksp_dims[COIL_DIM] = d->batch;

	/* Centred, which is what `bartorch.tools.fft` is and so what a caller
	 * who chains this against one will expect; BART's own SENSE operator
	 * folds the same centring into the sensitivities instead. */
	if (NULL == traj)
		d->slab = linop_fftc_create(DIMS, slab_ksp_dims, FFT_FLAGS);
	else
		d->slab = nufft_create2(DIMS, slab_ksp_dims, d->cim_dims, traj_dims, traj,
				NULL, NULL, NULL, NULL, *conf);

	sense_output_from(d);

	return sense_operator(d);
}
