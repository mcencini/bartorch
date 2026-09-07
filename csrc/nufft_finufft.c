/*
 * BART's NUFFT operator, computed by FINUFFT.
 *
 * The seam is `nufft_create` rather than the gridding kernel underneath it:
 * FINUFFT does the spreading, the FFT and the deapodisation together, so none
 * of the three has to agree with BART's, only the sign and the scaling, which
 * tests pin against a discrete Fourier sum.
 *
 * `nufft.c` is compiled with its entry points renamed, so BART's own operator
 * is still there as `bart_nufft_*` and serves whatever this declines: a
 * subspace basis, weights that do not lie along k-space, a trajectory that
 * varies across frames, or no FINUFFT at all.
 */
#include <complex.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/types.h"

#include "linops/linop.h"

#include "num/flpmath.h"
#include "num/multind.h"

#include "noncart/nufft.h"

#include "include/bartorch.h"

extern struct linop_s* bart_nufft_create2(int N, const long ksp_dims[N], const long cim_dims[N], const long traj_dims[N], const complex float* traj, const long wgh_dims[N], const complex float* weights, const long bas_dims[N], const complex float* basis, struct nufft_conf_s conf);
extern int bart_nufft_get_psf_dims(const struct linop_s* nufft, int N, long psf_dims[N]);
extern void bart_nufft_get_psf(const struct linop_s* nufft, int N, const long psf_dims[N], complex float* psf);
extern void bart_nufft_get_psf2(const struct linop_s* nufft, int N, const long psf_dims[N], const long psf_strs[N], complex float* psf);
extern void bart_nufft_update_psf(const struct linop_s* nufft, int ND, const long psf_dims[ND], const complex float* psf);
extern void bart_nufft_update_psf2(const struct linop_s* nufft, int ND, const long psf_dims[ND], const long psf_strs[ND], const complex float* psf);
extern void bart_nufft_update_traj(const struct linop_s* nufft, int N, const long trj_dims[N], const complex float* traj, const long wgh_dims[N], const complex float* weights, const long bas_dims[N], const complex float* basis);
extern const struct operator_s* bart_nufft_precond_create(const struct linop_s* nufft_op);

/* Provided by finufft.c, which owns the FINUFFT entry points. */
extern int bartorch_finufft_plan(int device, int type, int dim, const int64_t n_modes[3],
		int ntrans, int isign, double eps, void** plan);
extern int bartorch_finufft_setpts(void* plan, long M, float* x, float* y, float* z);
extern int bartorch_finufft_exec(void* plan, complex float* c, complex float* f);
extern void bartorch_finufft_free(void* plan);
extern double bartorch_finufft_tolerance(void);

/* ------------------------------------------------------------------------ */

/* Normal operators built since the last reset, so a test can say that a
 * solve ran on a point spread function rather than on the transform pair. */
enum { TP_PSF, TP_PAIR };
static long toeplitz_counters[2];

long bartorch_toeplitz_counter(int which)
{
	return ((0 == which) || (1 == which)) ? toeplitz_counters[which] : -1;
}

void bartorch_toeplitz_reset_counters(void)
{
	toeplitz_counters[TP_PSF] = 0;
	toeplitz_counters[TP_PAIR] = 0;
}

struct nufft_fi_s {

	linop_data_t super;

	void* forward_plan;
	void* adjoint_plan;
	/* A plan holds the points by pointer, so they outlive setpts. */
	float* coord[3];
	pthread_mutex_t lock;

	/* BART's own operator over the same trajectory, held for the point
	 * spread function its normal applies.  NULL when the caller asked for
	 * no Toeplitz embedding, and the pair answers the normal instead. */
	const struct linop_s* toeplitz;

	long samples;
	long batch;
	long image_elements;
	float scale;

	/* The weights BART would multiply the transform by, NULL when there are
	 * none; ksp_dims and their strides say how they broadcast onto k-space. */
	int N;
	complex float* weights;
	long* cim_dims;
	long* ksp_dims;
	long* ksp_strs;
	long* wgh_strs;
};

static DEF_TYPEID(nufft_fi_s);

/* Whether an operator is one of these, so the entry points that read BART's
 * internals can tell one from one of BART's. */
static bool is_ours(const struct linop_s* op)
{
	return NULL != CAST_MAYBE(nufft_fi_s, linop_get_data(op));
}

static void nufft_fi_forward(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	struct nufft_fi_s* d = CAST_DOWN(nufft_fi_s, _d);

	pthread_mutex_lock(&d->lock);
	int ret = bartorch_finufft_exec(d->forward_plan, dst, (complex float*)src);
	pthread_mutex_unlock(&d->lock);

	if (0 != ret)
		error("bartorch: FINUFFT forward transform failed\n");

	md_zsmul(d->N, d->ksp_dims, dst, dst, d->scale);

	if (NULL != d->weights)
		md_zmul2(d->N, d->ksp_dims, d->ksp_strs, dst, d->ksp_strs, dst, d->wgh_strs, d->weights);
}

static void nufft_fi_adjoint(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	struct nufft_fi_s* d = CAST_DOWN(nufft_fi_s, _d);

	complex float* weighted = NULL;

	if (NULL != d->weights) {

		weighted = md_alloc_sameplace(d->N, d->ksp_dims, CFL_SIZE, dst);
		md_zmulc2(d->N, d->ksp_dims, d->ksp_strs, weighted, d->ksp_strs, src, d->wgh_strs, d->weights);
		src = weighted;
	}

	pthread_mutex_lock(&d->lock);
	int ret = bartorch_finufft_exec(d->adjoint_plan, (complex float*)src, dst);
	pthread_mutex_unlock(&d->lock);

	md_free(weighted);

	if (0 != ret)
		error("bartorch: FINUFFT adjoint transform failed\n");

	md_zsmul(d->N, d->cim_dims, dst, dst, d->scale);
}

static void nufft_fi_normal(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	struct nufft_fi_s* d = CAST_DOWN(nufft_fi_s, _d);

	linop_normal_unchecked(d->toeplitz, dst, src);
}

static void nufft_fi_del(const linop_data_t* _d)
{
	struct nufft_fi_s* d = CAST_DOWN(nufft_fi_s, _d);

	bartorch_finufft_free(d->forward_plan);
	bartorch_finufft_free(d->adjoint_plan);

	if (NULL != d->toeplitz)
		linop_free(d->toeplitz);

	for (int i = 0; i < 3; i++)
		md_free(d->coord[i]);

	md_free(d->weights);
	xfree(d->cim_dims);
	xfree(d->ksp_dims);
	xfree(d->ksp_strs);
	xfree(d->wgh_strs);

	pthread_mutex_destroy(&d->lock);
	xfree(d);
}

/* ------------------------------------------------------------------------ */

static int decline_reason;

int bartorch_nufft_decline_reason(void)
{
	return decline_reason;
}

/* Which operator each nufft_create call returned, so a test can say that a
 * tool ran on FINUFFT rather than that FINUFFT was merely available. */
enum { CNT_FI, CNT_BART };
static long counters[2];

long bartorch_nufft_counter(int which)
{
	return ((0 == which) || (1 == which)) ? counters[which] : -1;
}

void bartorch_nufft_reset_counters(void)
{
	counters[CNT_FI] = 0;
	counters[CNT_BART] = 0;
}

static void count(int which)
{
#pragma omp atomic
	counters[which]++;
}

#define DECLINE(code) do { decline_reason = (code); return NULL; } while (0)

/* BART's own operator over the same trajectory, for its normal alone.
 *
 * A^H A is a convolution, so BART answers it with one multiply against a
 * point spread function rather than a forward and an adjoint transform; that
 * is `nufft.c`'s work and there is no reason to do it twice.  What is left to
 * FINUFFT is the transform pair, which is what a solve spends the rest of its
 * time in.
 *
 * The caller decides: `conf.toeplitz` is what `pics --no-toeplitz` and
 * `nufft -t` set, and it carries the memory the function costs.
 */
static const struct linop_s* toeplitz_for(int N, const long ksp_dims[N], const long cim_dims[N],
		const long traj_dims[N], const complex float* traj,
		const long wgh_dims[N], const complex float* weights, struct nufft_conf_s conf)
{
	if (!conf.toeplitz) {

#pragma omp atomic
		toeplitz_counters[TP_PAIR]++;

		return NULL;
	}

	const struct linop_s* op = bart_nufft_create2(N, ksp_dims, cim_dims, traj_dims, traj,
			wgh_dims, weights, NULL, NULL, conf);

#pragma omp atomic
	toeplitz_counters[TP_PSF]++;

	return op;
}

static struct linop_s* try_create(int N, const long ksp_dims[N], const long cim_dims[N],
		const long traj_dims[N], const complex float* traj,
		const long wgh_dims[N], const complex float* weights,
		const complex float* basis, struct nufft_conf_s conf)
{
	int device = bartorch_on_device(traj) ? 1 : 0;

	if (!bartorch_finufft_usable_on(device))
		DECLINE(device ? 3 : 1);

	if (NULL != basis)
		DECLINE(14);

	if ((N < 4) || (NULL == traj))
		DECLINE(2);

	if (3 != traj_dims[0])
		DECLINE(4);

	if (1 != ksp_dims[0])
		DECLINE(5);

	if ((0UL != conf.flags) && (7UL != conf.flags))
		DECLINE(6);

	int dim = 0;
	int axis[3];
	int64_t n_modes[3];

	for (int i = 0; i < 3; i++) {

		if (cim_dims[i] > 1) {

			axis[dim] = i;
			n_modes[dim] = cim_dims[i];
			dim++;
		}
	}

	if (0 == dim)
		DECLINE(7);

	long samples = ksp_dims[1] * ksp_dims[2];

	if (samples != md_calc_size(N - 1, traj_dims + 1))
		DECLINE(8);

	/* Frames beyond the coils are more batches, and each one has to see the
	 * same trajectory: a plan holds the points it was given. */
	long batch = 1;

	for (int i = 3; i < N; i++) {

		if (cim_dims[i] != ksp_dims[i])
			DECLINE(9);

		batch *= cim_dims[i];
	}

	if ((batch < 1) || (batch > 1024))
		DECLINE(10);

	/* BART multiplies the transform by the weights on the way out and by
	 * their conjugate on the way back, so they broadcast onto k-space. */
	if (NULL != weights)
		for (int i = 0; i < N; i++)
			if ((1 != wgh_dims[i]) && (wgh_dims[i] != ksp_dims[i]))
				DECLINE(15);

	double eps = bartorch_finufft_tolerance();
	void* forward_plan = NULL;
	void* adjoint_plan = NULL;

	if (0 != bartorch_finufft_plan(device, 2, dim, n_modes, (int)batch, -1, eps, &forward_plan))
		DECLINE(11);

	if (0 != bartorch_finufft_plan(device, 1, dim, n_modes, (int)batch, +1, eps, &adjoint_plan)) {

		bartorch_finufft_free(forward_plan);
		DECLINE(12);
	}

	/* BART's trajectory counts samples of the image grid and FINUFFT takes
	 * the same position in radians.  Taking one component and rescaling it
	 * with BART's own operations is what makes this the same code on a card:
	 * the arrays come out where the trajectory is. */
	long one_dims[N];
	long one_strs[N];
	long trj_strs[N];

	md_select_dims(N, ~1UL, one_dims, traj_dims);
	md_calc_strides(N, one_strs, one_dims, CFL_SIZE);
	md_calc_strides(N, trj_strs, traj_dims, CFL_SIZE);

	complex float* component = md_alloc_sameplace(N, one_dims, CFL_SIZE, traj);
	float* coord[3] = { NULL, NULL, NULL };

	for (int i = 0; i < dim; i++) {

		md_copy2(N, one_dims, one_strs, component, trj_strs, traj + axis[i], CFL_SIZE);

		coord[i] = md_alloc_sameplace(N, one_dims, FL_SIZE, traj);
		md_real(N, one_dims, coord[i], component);
		md_smul(N, one_dims, coord[i], coord[i], (float)(2. * M_PI / (double)cim_dims[axis[i]]));
	}

	md_free(component);

	int ret = bartorch_finufft_setpts(forward_plan, samples, coord[0], coord[1], coord[2]);

	if (0 == ret)
		ret = bartorch_finufft_setpts(adjoint_plan, samples, coord[0], coord[1], coord[2]);

	if (0 != ret) {

		bartorch_finufft_free(forward_plan);
		bartorch_finufft_free(adjoint_plan);

		for (int i = 0; i < 3; i++)
			md_free(coord[i]);

		DECLINE(13);
	}

	long image_elements = 1;

	for (int i = 0; i < dim; i++)
		image_elements *= (long)n_modes[i];

	PTR_ALLOC(struct nufft_fi_s, d);
	SET_TYPEID(nufft_fi_s, d);

	d->forward_plan = forward_plan;
	d->adjoint_plan = adjoint_plan;

	for (int i = 0; i < 3; i++)
		d->coord[i] = coord[i];

	pthread_mutex_init(&d->lock, NULL);
	d->samples = samples;
	d->batch = batch;
	d->image_elements = image_elements;
	d->scale = (float)(1. / sqrt((double)image_elements));

	d->N = N;
	d->weights = NULL;
	d->cim_dims = xmalloc((size_t)N * sizeof(long));
	d->ksp_dims = xmalloc((size_t)N * sizeof(long));
	d->ksp_strs = xmalloc((size_t)N * sizeof(long));
	d->wgh_strs = xmalloc((size_t)N * sizeof(long));

	md_copy_dims(N, d->cim_dims, cim_dims);
	md_copy_dims(N, d->ksp_dims, ksp_dims);
	md_calc_strides(N, d->ksp_strs, ksp_dims, CFL_SIZE);

	if (NULL != weights) {

		d->weights = md_alloc_sameplace(N, wgh_dims, CFL_SIZE, weights);
		md_copy(N, wgh_dims, d->weights, weights, CFL_SIZE);
		md_calc_strides(N, d->wgh_strs, wgh_dims, CFL_SIZE);
	}

	d->toeplitz = toeplitz_for(N, ksp_dims, cim_dims, traj_dims, traj, wgh_dims, weights, conf);

	/* PTR_PASS hands the data over and clears the pointer, so what the
	 * operator is built with is read out first. */
	lop_fun_t normal = (NULL != d->toeplitz) ? nufft_fi_normal : NULL;

	struct linop_s* op = linop_create(N, ksp_dims, N, cim_dims, CAST_UP(PTR_PASS(d)),
			nufft_fi_forward, nufft_fi_adjoint, normal, NULL, nufft_fi_del);

	decline_reason = 0;
	count(CNT_FI);
	return op;
}

struct linop_s* nufft_create2(int N, const long ksp_dims[N], const long cim_dims[N],
		const long traj_dims[N], const complex float* traj,
		const long wgh_dims[N], const complex float* weights,
		const long bas_dims[N], const complex float* basis, struct nufft_conf_s conf)
{
	struct linop_s* op = try_create(N, ksp_dims, cim_dims, traj_dims, traj, wgh_dims, weights, basis, conf);

	if (NULL != op)
		return op;

	count(CNT_BART);

	return bart_nufft_create2(N, ksp_dims, cim_dims, traj_dims, traj, wgh_dims, weights, bas_dims, basis, conf);
}

struct linop_s* nufft_create(int N, const long ksp_dims[N], const long cim_dims[N],
		const long traj_dims[N], const complex float* traj,
		const complex float* weights, struct nufft_conf_s conf)
{
	long wgh_dims[N];
	md_select_dims(N, ~MD_BIT(0), wgh_dims, traj_dims);

	return nufft_create2(N, ksp_dims, cim_dims, traj_dims, traj, wgh_dims, weights, NULL, NULL, conf);
}

/* The rest read BART's own operator internals, so they are only safe on one
 * of BART's; on one of these they say so rather than read the wrong struct. */

static void refuse(const char* what)
{
	error("bartorch: %s needs BART's own NUFFT operator, and this one is FINUFFT's.\n"
	      "Turn the substitution off with bartorch.finufft.use_in_tools(False).\n", what);
}

int nufft_get_psf_dims(const struct linop_s* nufft, int N, long psf_dims[N])
{
	if (is_ours(nufft))
		refuse("reading a point spread function");

	return bart_nufft_get_psf_dims(nufft, N, psf_dims);
}

void nufft_get_psf(const struct linop_s* nufft, int N, const long psf_dims[N], complex float* psf)
{
	if (is_ours(nufft))
		refuse("reading a point spread function");

	bart_nufft_get_psf(nufft, N, psf_dims, psf);
}

void nufft_get_psf2(const struct linop_s* nufft, int N, const long psf_dims[N], const long psf_strs[N], complex float* psf)
{
	if (is_ours(nufft))
		refuse("reading a point spread function");

	bart_nufft_get_psf2(nufft, N, psf_dims, psf_strs, psf);
}

void nufft_update_psf(const struct linop_s* nufft, int ND, const long psf_dims[ND], const complex float* psf)
{
	if (is_ours(nufft))
		refuse("supplying a point spread function");

	bart_nufft_update_psf(nufft, ND, psf_dims, psf);
}

void nufft_update_psf2(const struct linop_s* nufft, int ND, const long psf_dims[ND], const long psf_strs[ND], const complex float* psf)
{
	if (is_ours(nufft))
		refuse("supplying a point spread function");

	bart_nufft_update_psf2(nufft, ND, psf_dims, psf_strs, psf);
}

void nufft_update_traj(const struct linop_s* nufft, int N, const long trj_dims[N], const complex float* traj, const long wgh_dims[N], const complex float* weights, const long bas_dims[N], const complex float* basis)
{
	if (is_ours(nufft))
		refuse("changing a trajectory in place");

	bart_nufft_update_traj(nufft, N, trj_dims, traj, wgh_dims, weights, bas_dims, basis);
}

const struct operator_s* nufft_precond_create(const struct linop_s* nufft_op)
{
	if (is_ours(nufft_op))
		refuse("building a preconditioner");

	return bart_nufft_precond_create(nufft_op);
}
