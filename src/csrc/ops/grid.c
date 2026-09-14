/*
 * The transform a Cartesian SENSE slab carries, with a normal of its own.
 *
 * Forward it is the centred unitary transform, a subspace basis contracting
 * the coefficients into the frames that were acquired, and the pattern that
 * keeps the samples taken -- the chain `grecon/model.c` builds, applied to a
 * slab of coils where the arithmetic is rather than to all of them on the
 * host.
 *
 * The normal is where the two differ.  Along an axis the pattern does not
 * vary, keeping the samples commutes with the transform, and the transform
 * and its inverse cancel: the readout of a Cartesian acquisition, and every
 * axis a pattern of phase encodes leaves alone.  So the normal transforms only
 * the axes the pattern varies along, and between the transforms it applies
 * one kernel,
 *
 *     K[k', k] = sum_t |P[t]|^2 conj(B[k', t]) B[k, t],
 *
 * over those axes alone: coefficients by coefficients at each place, however
 * many frames there are.  Without a basis it is the pattern's square; without
 * a pattern it is the basis's Gram, and nothing is transformed at all.
 */
#include <complex.h>
#include <stdbool.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"

#include "num/multind.h"
#include "num/flpmath.h"

#include "linops/fmac.h"
#include "linops/linop.h"
#include "linops/someops.h"

#include "sense/model.h"

/* The spatial axes along which the pattern has more than one value. */
static unsigned long varying(const long pat_dims[DIMS])
{
	unsigned long flags = 0UL;

	for (int a = 0; a < 3; a++)
		if (1 < pat_dims[a])
			flags |= MD_BIT(a);

	return flags;
}

/* The kernel above, with k' on the TE axis and k on the COEFF axis -- the one
 * arrangement a single multiply-accumulate contracts -- and the pattern's
 * spatial axes, and one everywhere else.  Computed on the host: it is
 * coefficients squared over a plane of phase encodes. */
static complex float* grid_kernel(long kdims[DIMS], const long pat_dims[DIMS], const complex float* pattern,
		const long bas_dims[DIMS], const complex float* basis)
{
	long R = (NULL == basis) ? 1 : bas_dims[COEFF_DIM];
	long T = (NULL == basis) ? 1 : bas_dims[TE_DIM];
	long F = (NULL == pattern) ? 1 : pat_dims[TE_DIM];

	md_singleton_dims(DIMS, kdims);

	if (NULL != pattern)
		for (int a = 0; a < 3; a++)
			kdims[a] = pat_dims[a];

	kdims[TE_DIM] = R;
	kdims[COEFF_DIM] = R;

	long places = md_calc_size(3, kdims);

	complex float* P = NULL;

	if (NULL != pattern) {

		P = md_alloc(DIMS, pat_dims, CFL_SIZE);
		md_copy(DIMS, pat_dims, P, pattern, CFL_SIZE);
	}

	complex float* B = NULL;

	if (NULL != basis) {

		B = md_alloc(DIMS, bas_dims, CFL_SIZE);
		md_copy(DIMS, bas_dims, B, basis, CFL_SIZE);
	}

	/* conj(B[k', t]) B[k, t], once per frame. */
	complex float* G = xmalloc((size_t)(T * R * R) * sizeof(complex float));

	for (long t = 0; t < T; t++)
		for (long kp = 0; kp < R; kp++)
			for (long k = 0; k < R; k++)
				G[t * R * R + kp * R + k] = (NULL == B) ? 1.f : conjf(B[t + T * kp]) * B[t + T * k];

	complex float* K = md_alloc(DIMS, kdims, CFL_SIZE);

#pragma omp parallel for
	for (long p = 0; p < places; p++) {

		for (long kp = 0; kp < R; kp++)
			for (long k = 0; k < R; k++) {

				complex double acc = 0.;

				for (long t = 0; t < T; t++) {

					double w = 1.;

					if (NULL != P) {

						/* A pattern that is the same for every frame
						 * has one of them. */
						complex float v = P[p + places * ((1 == F) ? 0 : t)];
						w = (double)(crealf(v) * crealf(v) + cimagf(v) * cimagf(v));
					}

					acc += w * G[t * R * R + kp * R + k];
				}

				K[p + places * (kp + R * k)] = (complex float)acc;
			}
	}

	xfree(G);
	md_free(B);
	md_free(P);

	return K;
}

/* The slab transform over coil images of `cim_dims`, or over every coil when
 * the loop does not run.  `pattern` and `basis` may each be NULL. */
const struct linop_s* grid_transform_create(const long cim_dims[DIMS],
		const long pat_dims[DIMS], const complex float* pattern,
		const long bas_dims[DIMS], const complex float* basis)
{
	unsigned long fft_flags = FFT_FLAGS & md_nontriv_dims(DIMS, cim_dims);

	if (NULL != pattern) {

		/* The kernel is laid out over the spatial axes and the frames and
		 * nothing else, so a pattern that differs between coils or sets
		 * of maps is not one this can collapse. */
		if (0UL != (md_nontriv_dims(DIMS, pat_dims) & ~(FFT_FLAGS | TE_FLAG)))
			error("bartorch: a Cartesian pattern varies along the spatial axes and the frames only\n");

		for (int a = 0; a < 3; a++)
			if ((1 != pat_dims[a]) && (pat_dims[a] != cim_dims[a]))
				error("bartorch: the pattern is %ld along axis %d, where the images are %ld\n",
						pat_dims[a], a, cim_dims[a]);

		long frames = (NULL == basis) ? 1 : bas_dims[TE_DIM];

		if ((1 != pat_dims[TE_DIM]) && (frames != pat_dims[TE_DIM]))
			error("bartorch: the pattern has %ld frames and the basis %ld\n", pat_dims[TE_DIM], frames);
	}

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, cim_dims);

	struct linop_s* fwd = linop_fftc_create(DIMS, cim_dims, fft_flags);

	long R = 1;

	if (NULL != basis) {

		R = bas_dims[COEFF_DIM];

		if (R != cim_dims[COEFF_DIM])
			error("bartorch: the basis has %ld coefficients and the image %ld\n", R, cim_dims[COEFF_DIM]);

		out_dims[COEFF_DIM] = 1;
		out_dims[TE_DIM] = bas_dims[TE_DIM];

		fwd = linop_chain_FF(fwd, linop_fmac_dims_create(DIMS, out_dims, cim_dims, bas_dims, basis));
	}

	if (NULL != pattern)
		fwd = linop_chain_FF(fwd, linop_sampling_create(out_dims, pat_dims, pattern));

	long kdims[DIMS];
	complex float* K = grid_kernel(kdims, pat_dims, pattern, bas_dims, basis);

	struct linop_s* core = NULL;

	if (NULL == basis) {

		core = linop_cdiag_create(DIMS, cim_dims, md_nontriv_dims(DIMS, kdims), K);

	} else {

		/* The coefficients go out on TE, where the multiply-accumulate
		 * can put them, and a reshape reads them back on COEFF; the two
		 * layouts are the same memory. */
		long mixed_dims[DIMS];
		md_copy_dims(DIMS, mixed_dims, cim_dims);
		mixed_dims[COEFF_DIM] = 1;
		mixed_dims[TE_DIM] = R;

		core = linop_chain_FF(linop_fmac_dims_create(DIMS, mixed_dims, cim_dims, kdims, K),
				linop_reshape_create(DIMS, cim_dims, DIMS, mixed_dims));
	}

	md_free(K);

	unsigned long nrm_flags = (NULL == pattern) ? 0UL : (varying(pat_dims) & fft_flags);

	struct linop_s* nrm = core;

	if (0UL != nrm_flags)
		nrm = linop_chain_FF(linop_chain_FF(linop_fftc_create(DIMS, cim_dims, nrm_flags), core),
				linop_ifftc_create(DIMS, cim_dims, nrm_flags));

	debug_printf(DP_DEBUG1, "Cartesian slab: %ld coefficients, normal transforms axes %lx of %lx\n",
			R, nrm_flags, fft_flags);

	struct linop_s* op = linop_from_ops(fwd->forward, fwd->adjoint, nrm->forward, NULL);

	linop_free(nrm);
	linop_free(fwd);

	return op;
}
