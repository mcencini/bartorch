/*
 * lib_shim.c — Thin C wrappers over BART's library API for use from C++.
 *
 * Provides simple C-linkage functions so that lib_ops.cpp can call BART
 * library routines without dealing with C99/GNU extensions (_Complex float,
 * VLAs, TYPEID) that are problematic in C++ translation units.
 *
 * All "complex data" pointers are typed as _Complex float* here (C) but are
 * declared void* in the corresponding extern "C" blocks in lib_ops.cpp.  On
 * all supported platforms void* and _Complex float* share identical size and
 * calling convention, making this ABI-compatible.
 */

#include <stdbool.h>
#include <string.h>

#include "misc/mri.h"       /* DIMS, COIL_DIM, MAPS_DIM, COEFF_DIM */
#include "num/iovec.h"      /* iovec_s, linop_domain/codomain */
#include "num/init.h"       /* num_init */
#include "linops/linop.h"   /* linop_s, linop_*_unchecked, linop_free */
#include "grecon/model.h"   /* pics_model, pics_config */
#include "noncart/nufft.h"  /* nufft_conf_s, nufft_conf_defaults */
#include "iter/iter.h"      /* iter_conjgrad, iter_conjgrad_defaults, iter_conf */
#include "iter/lsqr.h"      /* lsqr, lsqr_conf, lsqr_defaults */

/* ------------------------------------------------------------------
 * Shape queries
 * ------------------------------------------------------------------ */

void bartorch_linop_get_domain_dims(const struct linop_s *op, long dims[DIMS])
{
	const struct iovec_s *iov = linop_domain(op);
	int n = (iov->N < DIMS) ? iov->N : DIMS;

	for (int i = 0; i < n; i++)
		dims[i] = iov->dims[i];
	for (int i = n; i < DIMS; i++)
		dims[i] = 1L;
}

void bartorch_linop_get_codomain_dims(const struct linop_s *op, long dims[DIMS])
{
	const struct iovec_s *iov = linop_codomain(op);
	int n = (iov->N < DIMS) ? iov->N : DIMS;

	for (int i = 0; i < n; i++)
		dims[i] = iov->dims[i];
	for (int i = n; i < DIMS; i++)
		dims[i] = 1L;
}

/* ------------------------------------------------------------------
 * Operator creation
 * ------------------------------------------------------------------ */

/*
 * Create a SENSE/NUFFT encoding operator via pics_model().
 *
 * Caller computes all DIMS-length Fortran-order dim arrays.
 * Optional inputs (traj, basis, pattern) are passed as NULL when absent,
 * and the corresponding dim arrays are ignored.
 *
 * img_dims  : image dimensions (spatial + ESPIRiT maps + subspace coefficients)
 * ksp_dims  : k-space dimensions (spatial/readout + coils)
 * map_dims  : sensitivity map dimensions (spatial + coils + ESPIRiT maps)
 * maps      : sensitivity map data (always required)
 * pat_dims  : sampling pattern dimensions (always required; use all-1s for full k-space)
 * pattern   : sampling mask data (always required; use all-ones for full k-space)
 * traj_dims : trajectory dimensions (NULL → Cartesian reconstruction)
 * traj      : trajectory data (NULL → Cartesian)
 * bas_dims  : subspace basis dimensions (NULL → no subspace projection)
 * basis     : subspace basis data (NULL → no subspace)
 * use_cuda  : non-zero to enable GPU (requires CUDA build); when non-zero AND
 *             traj is present, also enables GPU gridding (gpu_gridding=true).
 *
 * Returns NULL on failure.
 */
const struct linop_s *bartorch_create_encoding_op(
	const long img_dims[DIMS],
	const long ksp_dims[DIMS],
	const long map_dims[DIMS], const _Complex float *maps,
	const long pat_dims[DIMS], const _Complex float *pattern,
	const long traj_dims[DIMS], const _Complex float *traj,
	const long bas_dims[DIMS], const _Complex float *basis,
	int use_cuda)
{
	num_init();

	/* Use default NUFFT config for non-Cartesian; NULL is invalid for traj!=NULL. */
	struct nufft_conf_s nuconf = nufft_conf_defaults;

	struct pics_config conf = { 0 };
	conf.gpu = (bool)use_cuda;
	/* For non-Cartesian + CUDA: also enable gpu_gridding so BART moves the
	 * trajectory to GPU and wraps the NUFFT via linop_gpu_wrapper. */
	conf.gpu_gridding = (bool)use_cuda && (NULL != traj);
	conf.nuconf = (NULL != traj) ? &nuconf : NULL;

	return pics_model(&conf,
		img_dims, ksp_dims,
		traj ? traj_dims : NULL, traj,
		basis ? bas_dims : NULL, basis,
		map_dims, maps,
		pattern ? pat_dims : NULL, pattern,
		NULL, NULL,  /* no motion */
		NULL);       /* don't return nufft_op handle */
}

/* ------------------------------------------------------------------
 * Operator application
 * ------------------------------------------------------------------ */

void bartorch_linop_forward(const struct linop_s *op, void *dst, const void *src)
{
	linop_forward_unchecked(op, (_Complex float *)dst, (const _Complex float *)src);
}

void bartorch_linop_adjoint(const struct linop_s *op, void *dst, const void *src)
{
	linop_adjoint_unchecked(op, (_Complex float *)dst, (const _Complex float *)src);
}

void bartorch_linop_normal(const struct linop_s *op, void *dst, const void *src)
{
	linop_normal_unchecked(op, (_Complex float *)dst, (const _Complex float *)src);
}

/* ------------------------------------------------------------------
 * CG solver via lsqr + iter_conjgrad
 * ------------------------------------------------------------------ */

/*
 * Solve the regularised normal equation (A^H A + lambda*I) x = A^H y
 * using conjugate gradients entirely in C.
 *
 * op       : the encoding linop (domain = image space, codomain = k-space)
 * maxiter  : maximum CG iterations
 * lambda   : L2 regularisation weight (Tikhonov)
 * tol      : convergence tolerance
 * x        : solution buffer (domain, DIMS-length array, in/out — initialised
 *             to zero before call; lsqr warm-starts if lsqr_conf.warmstart=true)
 * y        : measured data (codomain, DIMS-length array, const)
 * x_dims   : BART Fortran-order dims for x (length DIMS)
 * y_dims   : BART Fortran-order dims for y (length DIMS)
 *
 * lsqr with include_adjoint=true internally computes b = A^H y and
 * then solves (A^H A + lambda*I) x = b by CG.
 */
void bartorch_conjgrad_solve(
	const struct linop_s *op,
	int maxiter, float lambda, float tol,
	const long x_dims[DIMS], _Complex float *x,
	const long y_dims[DIMS], const _Complex float *y)
{
	struct lsqr_conf lconf = lsqr_defaults;
	lconf.lambda = lambda;

	struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;
	cgconf.maxiter = maxiter;
	cgconf.tol = tol;

	lsqr(DIMS, &lconf, iter_conjgrad, (iter_conf *)&cgconf,
		op, NULL,
		x_dims, x,
		y_dims, y,
		NULL);
}

/* ------------------------------------------------------------------
 * Lifecycle
 * ------------------------------------------------------------------ */

void bartorch_linop_free(const struct linop_s *op)
{
	linop_free(op);
}
