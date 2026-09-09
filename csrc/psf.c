/*
 * The point spread function a Toeplitz normal convolves with.
 *
 * A^H A is a convolution, and BART builds the function it convolves with by
 * taking the adjoint NUFFT of ones over a doubled trajectory.  That adjoint
 * is a gridding, and `nufft.c` reaches it through its own `nufft_create2`,
 * which the rename sends to BART's operator along with everything else in
 * that file -- so a PSF stayed BART's however the transform around it was
 * served.
 *
 * These three carry the same signatures under the original names and differ
 * from BART's in one line: the transform is whatever `nufft_create2` answers,
 * which is FINUFFT's on the host, cuFINUFFT's on a card, and BART's own when
 * the substitution is off.  Everything around it -- the squared weights and
 * basis, the doubled grid, the shifts, the decomposition -- is what BART does,
 * done the same way, because a PSF that differs from BART's is a normal
 * operator that is not the adjoint of the transform beside it.
 *
 * `nlinv`, `moba`, `rtnlinv`, `noir/model2` and the `psf` tool call these
 * directly.  What `nufft.c` computes for its own Toeplitz embedding is
 * reached from inside that file and does not come here.
 */
#include <complex.h>
#include <math.h>
#include <stdbool.h>

#include "misc/misc.h"
#include "misc/debug.h"
#include "misc/mri.h"
#include "misc/version.h"

#include "num/fft.h"
#include "num/flpmath.h"
#include "num/multind.h"
#include "num/compress.h"
#include "num/shuffle.h"
#include "num/triagmat.h"
#include "num/vptr.h"

#include "linops/linop.h"

#include "noncart/nufft.h"

/* The weights the normal carries are the transform's, squared. */
static complex float* square_weights(int N, const long wgh_dims[N], const complex float* weights)
{
	if (NULL == weights)
		return NULL;

	complex float* sqr = md_alloc_sameplace(N, wgh_dims, CFL_SIZE, weights);
	md_zmulc(N, wgh_dims, sqr, weights, weights);

	return sqr;
}

/* A subspace basis enters the normal as its own Gram matrix, laid out along
 * the coefficient axis the transform contracts. */
static complex float* square_basis(bool upper_triag, int N, long sqr_bas_dims[N],
		const long bas_dims[N], const complex float* basis, const long ksp_dims[N])
{
	if (NULL == basis) {

		md_singleton_dims(N, sqr_bas_dims);
		return NULL;
	}

	assert(1 == bas_dims[7]);

	long bas_dimsT[N];

	md_transpose_dims(N, 6, 7, bas_dimsT, bas_dims);
	md_max_dims(N, ~0UL, sqr_bas_dims, bas_dims, bas_dimsT);
	sqr_bas_dims[5] = ksp_dims[5];

	complex float* sqr = md_alloc_sameplace(N, sqr_bas_dims, CFL_SIZE, basis);
	md_ztenmulc(N, sqr_bas_dims, sqr, bas_dims, basis, bas_dimsT, basis);

	sqr_bas_dims[6] *= sqr_bas_dims[6];
	sqr_bas_dims[7] = 1;

	if (use_compat_to_version("v0.7.00"))
		md_zsmul(N, sqr_bas_dims, sqr, sqr, (double)bas_dims[6]);

	if (upper_triag) {

		long tri_dims[N];
		complex float* tri = hermite_to_uppertriag(6, 6, 6, N, tri_dims, sqr_bas_dims, sqr);

		md_free(sqr);
		sqr = tri;

		md_copy_dims(N, sqr_bas_dims, tri_dims);
	}

	return sqr;
}

/* The transform the PSF is the adjoint of: no Toeplitz of its own, or it
 * would ask for a point spread function to compute one. */
static struct nufft_conf_s psf_conf(bool periodic, bool lowmem, bool vptr)
{
	struct nufft_conf_s conf = nufft_conf_defaults;

	/* Nobody asked for a grid or a width here: BART's defaults are what this
	 * struct carries, and taking them for a request would pin the transform
	 * at a kernel of six on a grid twice over whatever the caller configured. */
	conf.os = 0.;
	conf.width = 0.;

	conf.periodic = periodic;
	conf.toeplitz = false;
	conf.lowmem = lowmem;

	conf.precomp_linphase = vptr || use_compat_to_version("v0.8.00");
	conf.precomp_roll = vptr || use_compat_to_version("v0.8.00");
	conf.precomp_fftmod = vptr || use_compat_to_version("v0.8.00");

	return conf;
}

/* The adjoint transform of ones, which is what a point spread function is. */
static complex float* psf_int(int N, const long img_dims[N], const long trj_dims[N], const complex float* traj,
		const long bas_dims[N], const complex float* basis,
		const long wgh_dims[N], const complex float* weights,
		bool periodic, bool lowmem, bool upper_triag)
{
	long ksp_dims[N];
	md_select_dims(N, ~MD_BIT(0), ksp_dims, trj_dims);

	if (NULL != weights)
		md_max_dims(N, ~0UL, ksp_dims, ksp_dims, wgh_dims);

	long sqr_bas_dims[N];

	complex float* sqr_basis = square_basis(upper_triag, N, sqr_bas_dims, bas_dims, basis, ksp_dims);
	complex float* sqr_weights = square_weights(N, wgh_dims, weights);

	long img_dims2[N];
	md_copy_dims(N, img_dims2, img_dims);

	if (upper_triag) {

		assert(1 == img_dims2[5]);

	} else if (NULL != sqr_basis) {

		img_dims2[6] *= img_dims2[6];
		img_dims2[5] = 1;
	}

	complex float* psf = md_alloc_sameplace(N, img_dims, CFL_SIZE, traj);

	complex float* ones = md_alloc_sameplace(N, ksp_dims, CFL_SIZE, traj);
	md_zfill(N, ksp_dims, ones, 1.);

	struct nufft_conf_s conf = psf_conf(periodic, lowmem, is_vptr(traj));

	struct linop_s* op = nufft_create2(N, ksp_dims, img_dims2, trj_dims, traj,
			wgh_dims, sqr_weights, sqr_bas_dims, sqr_basis, conf);

	op = linop_reshape_in_F(op, N, img_dims);

	md_free(sqr_weights);
	md_free(sqr_basis);

	linop_adjoint(op, N, img_dims, psf, N, ksp_dims, ones);
	linop_free(op);

	md_free(ones);

	return psf;
}

complex float* compute_psf(int N, const long img_dims[N], const long trj_dims[N], const complex float* traj,
		const long bas_dims[N], const complex float* basis,
		const long wgh_dims[N], const complex float* weights,
		bool periodic, bool lowmem)
{
	return psf_int(N, img_dims, trj_dims, traj, bas_dims, basis, wgh_dims, weights,
			periodic, lowmem, false);
}

/* On the grid twice over, which is where a convolution the size of the image
 * has room to be one. */
complex float* compute_psf2(int N, const long psf_dims[N + 1], unsigned long flags, const long trj_dims[N + 1], const complex float* traj,
		const long bas_dims[N + 1], const complex float* basis, const long wgh_dims[N + 1], const complex float* weights,
		bool periodic, bool lowmem, bool upper_triag)
{
	int ND = N + 1;

	long img_dims[ND];
	md_select_dims(ND, ~MD_BIT(N + 0), img_dims, psf_dims);

	long img2_dims[ND];
	md_copy_dims(ND, img2_dims, img_dims);

	for (int i = 0; i < N; i++)
		if (MD_IS_SET(flags, i))
			img2_dims[i] = (1 == img_dims[i]) ? 1 : (2 * img_dims[i]);

	complex float* traj2 = md_alloc_sameplace(ND, trj_dims, CFL_SIZE, traj);
	md_zsmul(ND, trj_dims, traj2, traj, 2.);

	complex float* psft = psf_int(ND, img2_dims, trj_dims, traj2, bas_dims, basis,
			wgh_dims, weights, periodic, lowmem, upper_triag);

	md_free(traj2);

	fftuc(ND, img2_dims, flags, psft, psft);

	complex float* psf = md_alloc_sameplace(ND, psf_dims, CFL_SIZE, traj);

	long factors[N];

	for (int i = 0; i < N; i++)
		factors[i] = ((img_dims[i] > 1) && (MD_IS_SET(flags, i))) ? 2 : 1;

	md_decompose(N + 0, factors, psf_dims, psf, img2_dims, psft, CFL_SIZE);

	md_free(psft);

	return psf;
}

static void psf_factors(int N, unsigned long flags, long factors[N], const long dims[N])
{
	flags = flags & md_nontriv_dims(N, dims);

	for (int i = 0; i < N; i++)
		factors[i] = (MD_IS_SET(flags, i)) ? 2 : 1;
}

/* The shift of one set of frequencies, as `nufft.c` computes it.  Shared
 * because the mask a compressed function keeps is gridded at the same shifts
 * the function itself was decomposed at. */
void bartorch_psf_shift(int NS, float shift[NS], int N, const long factors[N], int idx)
{
	assert(NS <= N);

	for (int i = 0; i < NS; i++) {

		shift[i] = -(float)(idx % factors[i]) / factors[i];
		idx /= factors[i];
	}

	assert(0 == idx);

	for (int i = NS; i < N; i++)
		assert(1 == factors[i]);
}

/* The same function, taken one set of frequencies at a time.
 *
 * The even and the odd frequencies of the doubled grid are independent, so
 * computing them separately never holds the doubled grid whole, which is what
 * makes a three-dimensional point spread function fit. */
/* Where a compressed function is going, when one is asked for: the places the
 * samples reach, and the shape it takes once only those are kept. */
struct psf_packing {

	const long* com_dims;
	const long* idx;
	const long* com_psf_dims;	/* the whole compressed function */
	const long* com_psf_dims3;	/* one set of frequencies of it */
};

static complex float* psf_decomposed(bool to_host, const struct psf_packing* pack,
		int N, const long psf_dims[N + 1], unsigned long flags, const long trj_dims[N + 1], const complex float* traj,
		const long bas_dims[N + 1], const complex float* basis, const long wgh_dims[N + 1], const complex float* weights,
		bool periodic, bool lowmem, bool upper_triag)
{
	int ND = N + 1;

	long ksp_dims[ND];
	md_select_dims(ND, ~MD_BIT(0), ksp_dims, trj_dims);
	ksp_dims[N] = psf_dims[N];

	if (NULL != weights)
		md_max_dims(ND, ~0UL, ksp_dims, ksp_dims, wgh_dims);

	long sqr_bas_dims[ND];

	complex float* sqr_basis = square_basis(upper_triag, ND, sqr_bas_dims, bas_dims, basis, ksp_dims);
	complex float* sqr_weights = square_weights(ND, wgh_dims, weights);

	long psf_dims2[ND];
	md_copy_dims(ND, psf_dims2, psf_dims);

	if (upper_triag) {

		assert(1 == psf_dims2[5]);

	} else if (NULL != sqr_basis) {

		psf_dims2[6] *= psf_dims2[6];
		psf_dims2[5] = 1;
	}

	struct nufft_conf_s conf = psf_conf(periodic, lowmem, is_vptr(traj));

	long trj_dims2[ND];
	md_copy_dims(ND, trj_dims2, trj_dims);
	trj_dims2[N] = psf_dims2[N];

	long factors[ND];
	psf_factors(ND, flags, factors, psf_dims);

	complex float tp[trj_dims2[N]][trj_dims2[0]];

	for (int k = 0; k < trj_dims2[N]; k++) {

		float shift[3];
		bartorch_psf_shift(3, shift, ND, factors, k);

		for (int j = 0; j < trj_dims2[0]; j++)
			tp[k][j] = (1 != psf_dims2[j] ? 0.5 * psf_dims2[j] : 0.) + shift[j];
	}

	long sdims[ND];
	md_select_dims(ND, MD_BIT(0) | MD_BIT(N), sdims, trj_dims2);

	complex float* tshift = md_alloc_sameplace(ND, sdims, CFL_SIZE, traj);
	md_copy(ND, sdims, tshift, &(tp[0][0]), CFL_SIZE);

	complex float* traj2 = md_alloc_sameplace(ND, trj_dims2, CFL_SIZE, traj);
	md_zadd2(ND, trj_dims2, MD_STRIDES(ND, trj_dims2, CFL_SIZE), traj2,
			MD_STRIDES(ND, trj_dims, CFL_SIZE), traj,
			MD_STRIDES(ND, sdims, CFL_SIZE), tshift);
	md_free(tshift);

	/* One transform per set of frequencies, stacked, which is what BART does
	 * for `lowmem` and what this does always: each carries its own shifted
	 * trajectory and writes its own image, and a transform whose points and
	 * whose image both vary along an axis is not one plan.  Stacking is the
	 * same operator either way and holds one set at a time. */
	long ksp_dims2[ND];
	long psf_dims3[ND];
	long trj_dims3[ND];

	md_select_dims(ND, ~MD_BIT(N), ksp_dims2, ksp_dims);
	md_select_dims(ND, ~MD_BIT(N), psf_dims3, psf_dims2);
	md_select_dims(ND, ~MD_BIT(N), trj_dims3, trj_dims2);

	(void)lowmem;

	/* The kernel the decomposition transforms: a cosine per doubled axis,
	 * with the half-sample shift an odd length needs. */
/* The kernel one set of frequencies transforms against: a cosine per doubled
 * axis, with the half-sample shift an odd length needs.  Built for the set
 * that is about to be used rather than for all of them at once, which is what
 * keeps the samples of every set off the card together. */
	/* `to_host` keeps the function where the card is not: each set is made
	 * on the card and copied out, so what is resident is one set rather
	 * than the whole of it. */
	const long* whole_dims = (NULL != pack) ? pack->com_psf_dims : psf_dims;

	complex float* psf = to_host ? md_alloc(ND, whole_dims, CFL_SIZE)
				     : md_alloc_sameplace(ND, whole_dims, CFL_SIZE, traj);

	/* One set of frequencies at a time.
	 *
	 * Stacking them into one operator and answering them together is the
	 * same arithmetic, and puts every set's transform on the card at once
	 * along with the whole function; taking them in turn holds one set's
	 * transform and writes into the one place the function lives.  That is
	 * what makes computing the sets separately cost less than computing
	 * them together rather than more. */
	long psf_coset = md_calc_size(ND, (NULL != pack) ? pack->com_psf_dims3 : psf_dims3);
	long ksp_coset = md_calc_size(ND, ksp_dims2);
	long trj_coset = md_calc_size(ND, trj_dims3);

	(void)ksp_coset;

	/* One transform for every set of frequencies, pointed at each in turn.
	 *
	 * The sets differ only in where their samples sit, so one plan serves
	 * them all -- and making a plan is the largest allocation a build does,
	 * more than the function it produces.  `nufft_update_traj` is what
	 * points it somewhere else without making it again. */
	/* And one coefficient of the function at a time.
	 *
	 * A subspace function is a matrix at every frequency, and answering the
	 * whole matrix at once means an image for every entry of it live
	 * together: at 256^3 with four coefficients that is ten images where
	 * one would do.  Each entry is its own gridding of the samples weighted
	 * by its own pair of basis coefficients, so the transform is built for
	 * one and pointed at each pair in turn -- the same retargeting the sets
	 * use, since a basis is something `nufft_update_traj` replaces. */
	long one_dims[ND];
	long one_bas_dims[ND];

	md_copy_dims(ND, one_dims, psf_dims3);
	md_copy_dims(ND, one_bas_dims, sqr_bas_dims);

	long pairs = psf_dims3[COEFF_DIM];

	one_dims[COEFF_DIM] = 1;
	one_bas_dims[COEFF_DIM] = 1;

	long pair_stride = md_calc_size(ND, one_dims);
	long basis_stride = (NULL == sqr_basis) ? 0 : md_calc_size(ND, one_bas_dims);

	struct linop_s* op = nufft_create2(ND, ksp_dims2, one_dims, trj_dims3, traj2,
			wgh_dims, sqr_weights, one_bas_dims, sqr_basis, conf);

	for (int i = 0; i < trj_dims2[N]; i++) {

		const complex float* traj_i = traj2 + i * trj_coset;

		complex float* kern = md_alloc_sameplace(ND, ksp_dims2, CFL_SIZE, traj);
		md_zfill(ND, ksp_dims2, kern, 1. / sqrt(md_calc_size(3, psf_dims)));

		for (int j = 0; j < 3; j++) {

			if (1 == psf_dims[j])
				continue;

			complex float* tkern = md_alloc_sameplace(ND, ksp_dims2, CFL_SIZE, traj);
			md_copy2(ND, ksp_dims2, MD_STRIDES(ND, ksp_dims2, CFL_SIZE), tkern,
					MD_STRIDES(ND, trj_dims3, CFL_SIZE), traj_i + j, CFL_SIZE);
			md_zsmul(ND, ksp_dims2, tkern, tkern, M_PI);
			md_zcos(ND, ksp_dims2, tkern, tkern);
			md_zmul(ND, ksp_dims2, kern, kern, tkern);
			md_free(tkern);

			if (0 == psf_dims[j] % 2)
				continue;

			tkern = md_alloc_sameplace(ND, ksp_dims2, CFL_SIZE, traj);
			md_copy2(ND, ksp_dims2, MD_STRIDES(ND, ksp_dims2, CFL_SIZE), tkern,
					MD_STRIDES(ND, trj_dims3, CFL_SIZE), traj_i + j, CFL_SIZE);
			md_zsmul(ND, ksp_dims2, tkern, tkern, 2.i * M_PI * (psf_dims[j] / 2 - psf_dims[j] / 2.) / psf_dims[j]);
			md_zexp(ND, ksp_dims2, tkern, tkern);
			md_zmul(ND, ksp_dims2, kern, kern, tkern);
			md_free(tkern);
		}

		/* What is whole at any moment is one entry of the matrix over
		 * one set of frequencies: one image, gridded, transformed, and
		 * -- where a compressed function was asked for -- reduced to the
		 * places the samples reach before the next one is made. */
		long out_pair_stride = (NULL != pack) ? md_calc_size(ND, pack->com_psf_dims3) / pairs
						     : pair_stride;

		for (long q = 0; q < pairs; q++) {

			nufft_update_traj(op, ND, trj_dims3, traj_i, wgh_dims, sqr_weights,
					one_bas_dims, (NULL == sqr_basis) ? NULL : sqr_basis + q * basis_stride);

			complex float* one = md_alloc_sameplace(ND, one_dims, CFL_SIZE, traj);

			linop_adjoint_unchecked(op, one, kern);
			fft(ND, one_dims, conf.flags, one, one);

			complex float* out = psf + i * psf_coset + q * out_pair_stride;

			if (NULL != pack) {

				long one_com[ND];
				md_copy_dims(ND, one_com, pack->com_psf_dims3);
				one_com[COEFF_DIM] = 1;

				complex float* packed = md_alloc_sameplace(ND, one_com, CFL_SIZE, traj);
				md_compress(ND, one_com, packed, one_dims, one, pack->com_dims, pack->idx, CFL_SIZE);
				md_free(one);

				md_copy(ND, one_com, out, packed, CFL_SIZE);
				md_free(packed);

			} else {

				md_copy(ND, one_dims, out, one, CFL_SIZE);
				md_free(one);
			}
		}

		md_free(kern);
	}

	linop_free(op);



	md_free(sqr_weights);
	md_free(sqr_basis);
	md_free(traj2);

	return psf;
}

complex float* compute_psf2_decomposed(int N, const long psf_dims[N + 1], unsigned long flags, const long trj_dims[N + 1], const complex float* traj,
		const long bas_dims[N + 1], const complex float* basis, const long wgh_dims[N + 1], const complex float* weights,
		bool periodic, bool lowmem, bool upper_triag)
{
	return psf_decomposed(false, NULL, N, psf_dims, flags, trj_dims, traj, bas_dims, basis,
			wgh_dims, weights, periodic, lowmem, upper_triag);
}

/* The same function, left where the card is not: neither it nor any set of
 * frequencies but the one being made is ever resident. */
complex float* bartorch_psf_to_host(int N, const long psf_dims[N + 1], unsigned long flags, const long trj_dims[N + 1], const complex float* traj,
		const long bas_dims[N + 1], const complex float* basis, const long wgh_dims[N + 1], const complex float* weights,
		bool periodic, bool lowmem, bool upper_triag,
		const long com_dims[N + 1], const long* idx,
		const long com_psf_dims[N + 1], const long com_psf_dims3[N + 1])
{
	struct psf_packing pack = { com_dims, idx, com_psf_dims, com_psf_dims3 };

	return psf_decomposed(true, (NULL != idx) ? &pack : NULL, N, psf_dims, flags, trj_dims, traj,
			bas_dims, basis, wgh_dims, weights, periodic, lowmem, upper_triag);
}
