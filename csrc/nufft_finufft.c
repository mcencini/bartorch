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
#include "num/init.h"
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

/* One side of the bus: the pair of plans FINUFFT holds there, the coordinates
 * they were given, and the weights the transform is multiplied by.
 *
 * A solve applies the same operator to whichever memory the iteration hands
 * it -- `pics` takes its first adjoint from the k-space it mapped and then
 * iterates on device vectors -- and a plan belongs to the library that made
 * it, so a side is built the first time one is asked for. */
struct fi_side {

	void* forward_plan;
	void* adjoint_plan;
	/* A plan holds the points by pointer, so they outlive setpts. */
	float* coord[3];
	complex float* weights;
	complex float* basis;

	/* Transforms one execute carries, and how many executes that leaves.
	 *
	 * On a card the images a batch writes are already the largest thing
	 * resident, and FINUFFT wants a working set beside them that grows with
	 * the count.  Asking for one transform at a time keeps that set flat and
	 * is no slower: forty-eight 192^3 adjoints take 6.9 s against 8.1 s, and
	 * the closer the images come to filling the card the wider that gap
	 * gets.  The host has the memory, and threads a batch across its
	 * transforms rather than inside one, so there it keeps them.
	 */
	int ntrans;
	long executes;
};

struct nufft_fi_s {

	linop_data_t super;

	struct fi_side side[2];		/* [0] the host, [1] a device */
	pthread_mutex_t lock;

	/* What a side is built from, on the host: the trajectory in radians,
	 * one array per transformed axis, the weights and the basis. */
	float* radians[3];
	complex float* host_weights;
	complex float* host_basis;

	/* BART's own operator over the same trajectory, held for the point
	 * spread function its normal applies.  NULL when the caller asked for
	 * no Toeplitz embedding, and the pair answers the normal instead. */
	const struct linop_s* toeplitz;

	int dim;
	int64_t n_modes[3];
	double eps;

	long samples;
	long batch;
	long image_elements;
	float scale;

	/* out_dims is k-space as the caller sees it -- what BART calls out_dims,
	 * frames present and coefficients contracted away -- and the strides say
	 * how the weights broadcast onto it.
	 *
	 * `grd_dims` is what the transform pair works in.  Without a basis it is
	 * k-space; with one it carries the coefficients k-space does not, and
	 * the basis contracts them away on the way out and spreads them on the
	 * way back, which is what `nufft.c` does either side of its gridder.
	 *
	 * `trf_strs` lays `grd_dims` out the way FINUFFT executes: a transform's
	 * samples together, the batch stepping over them.  BART's own order is
	 * that already unless a sample axis sits above a batch axis -- frames do,
	 * when the trajectory varies across them -- and `needs_tmp` says whether
	 * a buffer in that layout has to stand between.
	 */
	int N;
	bool needs_tmp;
	long* cim_dims;
	long* out_dims;
	long* out_strs;
	long* grd_dims;
	long* trf_strs;
	long* wgh_dims;
	long* wgh_strs;
	long* bas_dims;
	long* bas_strs;
};

static DEF_TYPEID(nufft_fi_s);

/* Memory on one side of the bus.  A device only exists in a CUDA build, and
 * `device` is never set without one: `bartorch_on_device` says no, and
 * `bart_use_gpu` stays false. */
static void* alloc_on(int device, int N, const long dims[N], size_t size)
{
#ifdef USE_CUDA
	if (device)
		return md_alloc_gpu(N, dims, size);
#else
	(void)device;
#endif
	return md_alloc(N, dims, size);
}

static void side_free(struct nufft_fi_s* d, struct fi_side* s)
{
	bartorch_finufft_free(s->forward_plan);
	bartorch_finufft_free(s->adjoint_plan);

	s->forward_plan = NULL;
	s->adjoint_plan = NULL;

	for (int i = 0; i < d->dim; i++) {

		md_free(s->coord[i]);
		s->coord[i] = NULL;
	}

	md_free(s->weights);
	s->weights = NULL;

	md_free(s->basis);
	s->basis = NULL;
}

/* Plans on `which`, over a copy of the coordinates and the weights there. */
static int side_build(struct nufft_fi_s* d, int which)
{
	struct fi_side* s = &d->side[which];

	s->ntrans = which ? 1 : (int)d->batch;
	s->executes = which ? d->batch : 1;

	if (0 != bartorch_finufft_plan(which, 2, d->dim, d->n_modes, s->ntrans, -1, d->eps, &s->forward_plan))
		return 11;

	if (0 != bartorch_finufft_plan(which, 1, d->dim, d->n_modes, s->ntrans, +1, d->eps, &s->adjoint_plan)) {

		side_free(d, s);
		return 12;
	}

	long one[1] = { d->samples };

	for (int i = 0; i < d->dim; i++) {

		s->coord[i] = alloc_on(which, 1, one, FL_SIZE);
		md_copy(1, one, s->coord[i], d->radians[i], FL_SIZE);
	}

	if (NULL != d->host_weights) {

		s->weights = alloc_on(which, d->N, d->wgh_dims, CFL_SIZE);
		md_copy(d->N, d->wgh_dims, s->weights, d->host_weights, CFL_SIZE);
	}

	if (NULL != d->host_basis) {

		s->basis = alloc_on(which, d->N, d->bas_dims, CFL_SIZE);
		md_copy(d->N, d->bas_dims, s->basis, d->host_basis, CFL_SIZE);
	}

	if (   (0 != bartorch_finufft_setpts(s->forward_plan, d->samples, s->coord[0], s->coord[1], s->coord[2]))
	    || (0 != bartorch_finufft_setpts(s->adjoint_plan, d->samples, s->coord[0], s->coord[1], s->coord[2]))) {

		side_free(d, s);
		return 13;
	}

	return 0;
}

/* The side `ptr` is on, built if this is the first transform there.
 *
 * A side is built once and never rebuilt, so what this returns stays good
 * after the lock is dropped.  error() leaves by a longjmp, which is why it is
 * called with the lock released. */
static const struct fi_side* side_for(struct nufft_fi_s* d, const void* ptr)
{
	int which = bartorch_on_device(ptr) ? 1 : 0;

	pthread_mutex_lock(&d->lock);

	int ret = (NULL == d->side[which].forward_plan) ? side_build(d, which) : 0;

	pthread_mutex_unlock(&d->lock);

	if (0 != ret)
		error("bartorch: FINUFFT would not plan the transform on the %s\n",
				which ? "device" : "host");

	return &d->side[which];
}

/* The buffer the transform pair works in, or NULL when it works in place.
 *
 * One is needed when the basis has to be applied, and when BART's own layout
 * does not already put a transform's samples together. */
static complex float* transform_buffer(const struct nufft_fi_s* d, const void* ref)
{
	if (!d->needs_tmp)
		return NULL;

	long dims[1] = { d->samples * d->batch };

	return md_alloc_sameplace(1, dims, CFL_SIZE, ref);
}

/* Whether an operator is one of these, so the entry points that read BART's
 * internals can tell one from one of BART's. */
static bool is_ours(const struct linop_s* op)
{
	return NULL != CAST_MAYBE(nufft_fi_s, linop_get_data(op));
}

static void nufft_fi_forward(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	struct nufft_fi_s* d = CAST_DOWN(nufft_fi_s, _d);

	const struct fi_side* s = side_for(d, dst);

	complex float* tmp = transform_buffer(d, dst);
	complex float* out = (NULL != tmp) ? tmp : dst;

	pthread_mutex_lock(&d->lock);

	int ret = 0;

	for (long i = 0; (0 == ret) && (i < s->executes); i++)
		ret = bartorch_finufft_exec(s->forward_plan,
				out + i * s->ntrans * d->samples,
				(complex float*)src + i * s->ntrans * d->image_elements);

	pthread_mutex_unlock(&d->lock);

	if (0 != ret) {

		md_free(tmp);
		error("bartorch: FINUFFT forward transform failed\n");
	}

	/* The coefficients the transform produced are contracted away here, the
	 * frames left as they are; without a basis this only reads the buffer
	 * back into BART's own layout. */
	if (NULL != tmp) {

		if (NULL != s->basis)
			md_ztenmul2(d->N, d->grd_dims, d->out_strs, dst, d->trf_strs, tmp, d->bas_strs, s->basis);
		else
			md_copy2(d->N, d->grd_dims, d->out_strs, dst, d->trf_strs, tmp, CFL_SIZE);

		md_free(tmp);
	}

	md_zsmul(d->N, d->out_dims, dst, dst, d->scale);

	if (NULL != s->weights)
		md_zmul2(d->N, d->out_dims, d->out_strs, dst, d->out_strs, dst, d->wgh_strs, s->weights);
}

static void nufft_fi_adjoint(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	struct nufft_fi_s* d = CAST_DOWN(nufft_fi_s, _d);

	const struct fi_side* s = side_for(d, dst);

	complex float* weighted = NULL;

	if (NULL != s->weights) {

		weighted = md_alloc_sameplace(d->N, d->out_dims, CFL_SIZE, dst);
		md_zmulc2(d->N, d->out_dims, d->out_strs, weighted, d->out_strs, src, d->wgh_strs, s->weights);
		src = weighted;
	}

	/* Spread the samples back over the coefficients the images carry, into
	 * the layout the transform executes in; without a basis this only lays
	 * the samples out that way. */
	complex float* tmp = transform_buffer(d, dst);

	if (NULL != tmp) {

		if (NULL != s->basis)
			md_ztenmulc2(d->N, d->grd_dims, d->trf_strs, tmp, d->out_strs, src, d->bas_strs, s->basis);
		else
			md_copy2(d->N, d->grd_dims, d->trf_strs, tmp, d->out_strs, src, CFL_SIZE);

		src = tmp;

		/* The weighted copy has been read into the buffer, and the
		 * transform is the point where memory is tightest. */
		md_free(weighted);
		weighted = NULL;
	}

	pthread_mutex_lock(&d->lock);

	int ret = 0;

	for (long i = 0; (0 == ret) && (i < s->executes); i++)
		ret = bartorch_finufft_exec(s->adjoint_plan,
				(complex float*)src + i * s->ntrans * d->samples,
				dst + i * s->ntrans * d->image_elements);

	pthread_mutex_unlock(&d->lock);

	md_free(tmp);
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

	side_free(d, &d->side[0]);
	side_free(d, &d->side[1]);

	if (NULL != d->toeplitz)
		linop_free(d->toeplitz);

	for (int i = 0; i < 3; i++)
		md_free(d->radians[i]);

	md_free(d->host_weights);
	md_free(d->host_basis);
	xfree(d->cim_dims);
	xfree(d->out_dims);
	xfree(d->out_strs);
	xfree(d->grd_dims);
	xfree(d->trf_strs);
	xfree(d->wgh_dims);
	xfree(d->wgh_strs);
	xfree(d->bas_dims);
	xfree(d->bas_strs);

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
		const long wgh_dims[N], const complex float* weights,
		const long bas_dims[N], const complex float* basis, struct nufft_conf_s conf)
{
	if (!conf.toeplitz) {

#pragma omp atomic
		toeplitz_counters[TP_PAIR]++;

		return NULL;
	}

	const struct linop_s* op = bart_nufft_create2(N, ksp_dims, cim_dims, traj_dims, traj,
			wgh_dims, weights, (NULL != basis) ? bas_dims : NULL, basis, conf);

#pragma omp atomic
	toeplitz_counters[TP_PSF]++;

	return op;
}

static struct linop_s* try_create(int N, const long ksp_dims[N], const long cim_dims[N],
		const long traj_dims[N], const complex float* traj,
		const long wgh_dims[N], const complex float* weights,
		const long bas_dims[N], const complex float* basis, struct nufft_conf_s conf)
{
	/* Which sides the operator may be applied on.  BART on a card applies
	 * one to either: `pics` takes its first adjoint from the k-space it
	 * mapped and iterates on device vectors, and `nufft -g` wraps the
	 * operator so that its arguments arrive on the device.  Both sides have
	 * to be servable before the substitution takes the operator at all. */
	int device = (bart_use_gpu || bartorch_on_device(traj)) ? 1 : 0;

	if (!bartorch_finufft_usable_on(0))
		DECLINE(1);

	if (device && !bartorch_finufft_usable_on(1))
		DECLINE(3);

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

	/* Two spaces.  `out_dims` is k-space as the caller sees it, which is what
	 * BART hands back from the operator: frames along TE, coefficients
	 * contracted away.  `grd_dims` is what the transform pair works in, with
	 * the coefficients present.  Without a basis they are the same thing.
	 *
	 * A caller may pass either one in: `nufft` gives k-space with a single
	 * coefficient and `pics` gives it with all of them, which is why BART
	 * fills the coefficient axis in rather than reading it. */
	long out_dims[N];
	long grd_dims[N];

	md_copy_dims(N, out_dims, ksp_dims);
	md_copy_dims(N, grd_dims, ksp_dims);

	if (NULL != basis) {

		if (1 != md_calc_size(5, bas_dims))
			DECLINE(14);

		if (cim_dims[6] != bas_dims[6])
			DECLINE(14);

		if ((1 != ksp_dims[6]) && (ksp_dims[6] != bas_dims[6]))
			DECLINE(14);

		grd_dims[6] = bas_dims[6];
		out_dims[5] = bas_dims[5];
		out_dims[6] = 1;
	}

	/* Every axis the trajectory indexes is a sample of one transform; the
	 * rest are separate transforms.  A frame is a sample axis when the
	 * trajectory varies across frames, which is what lets one plan over the
	 * whole raveled trajectory serve every coefficient and every coil. */
	long samples = md_calc_size(N - 1, traj_dims + 1);
	long batch = 1;

	for (int i = 1; i < N; i++) {

		if (1 < traj_dims[i]) {

			if (grd_dims[i] != traj_dims[i])
				DECLINE(8);

			/* An image that varies along a sample axis would need one
			 * transform per frame, not one plan over all of them. */
			if ((3 <= i) && (1 != cim_dims[i]))
				DECLINE(16);

			continue;
		}

		if (3 > i) {

			if (1 != grd_dims[i])
				DECLINE(8);

			continue;
		}

		if (cim_dims[i] != grd_dims[i])
			DECLINE(9);

		batch *= grd_dims[i];
	}

	/* FINUFFT batches the transforms it is asked for against one point set
	 * internally, a slice at a time, so what bounds this is the int its plan
	 * takes rather than memory: the arrays belong to BART and exist either
	 * way. */
	if ((batch < 1) || (batch > INT32_MAX))
		DECLINE(10);

	/* BART multiplies the transform by the weights on the way out and by
	 * their conjugate on the way back, so they broadcast onto k-space. */
	if (NULL != weights)
		for (int i = 0; i < N; i++)
			if ((1 != wgh_dims[i]) && (wgh_dims[i] != out_dims[i]))
				DECLINE(15);

	double eps = bartorch_finufft_tolerance();

	/* BART's trajectory counts samples of the image grid and FINUFFT takes
	 * the same position in radians.  One component is taken and rescaled
	 * with BART's own operations, on the host, and a side copies it to
	 * wherever its plans are; md_copy2 crosses the bus if the trajectory is
	 * on the other side. */
	long one_dims[N];
	long one_strs[N];
	long trj_strs[N];

	md_select_dims(N, ~1UL, one_dims, traj_dims);
	md_calc_strides(N, one_strs, one_dims, CFL_SIZE);
	md_calc_strides(N, trj_strs, traj_dims, CFL_SIZE);

	complex float* component = md_alloc(N, one_dims, CFL_SIZE);
	float* radians[3] = { NULL, NULL, NULL };

	for (int i = 0; i < dim; i++) {

		md_copy2(N, one_dims, one_strs, component, trj_strs, traj + axis[i], CFL_SIZE);

		radians[i] = md_alloc(N, one_dims, FL_SIZE);
		md_real(N, one_dims, radians[i], component);
		md_smul(N, one_dims, radians[i], radians[i], (float)(2. * M_PI / (double)cim_dims[axis[i]]));
	}

	md_free(component);

	long image_elements = 1;

	for (int i = 0; i < dim; i++)
		image_elements *= (long)n_modes[i];

	PTR_ALLOC(struct nufft_fi_s, d);
	SET_TYPEID(nufft_fi_s, d);

	memset(&d->side, 0, sizeof d->side);
	d->toeplitz = NULL;

	for (int i = 0; i < 3; i++)
		d->radians[i] = radians[i];

	pthread_mutex_init(&d->lock, NULL);
	d->dim = dim;

	for (int i = 0; i < 3; i++)
		d->n_modes[i] = n_modes[i];

	d->eps = eps;
	d->samples = samples;
	d->batch = batch;
	d->image_elements = image_elements;
	d->scale = (float)(1. / sqrt((double)image_elements));

	d->N = N;
	d->host_weights = NULL;
	d->host_basis = NULL;
	d->cim_dims = xmalloc((size_t)N * sizeof(long));
	d->out_dims = xmalloc((size_t)N * sizeof(long));
	d->out_strs = xmalloc((size_t)N * sizeof(long));
	d->grd_dims = xmalloc((size_t)N * sizeof(long));
	d->trf_strs = xmalloc((size_t)N * sizeof(long));
	d->wgh_dims = xmalloc((size_t)N * sizeof(long));
	d->wgh_strs = xmalloc((size_t)N * sizeof(long));
	d->bas_dims = xmalloc((size_t)N * sizeof(long));
	d->bas_strs = xmalloc((size_t)N * sizeof(long));

	md_copy_dims(N, d->cim_dims, cim_dims);
	md_copy_dims(N, d->out_dims, out_dims);
	md_calc_strides(N, d->out_strs, out_dims, CFL_SIZE);
	md_copy_dims(N, d->grd_dims, grd_dims);
	md_singleton_dims(N, d->wgh_dims);
	md_singleton_strides(N, d->wgh_strs);
	md_singleton_dims(N, d->bas_dims);
	md_singleton_strides(N, d->bas_strs);

	/* FINUFFT executes on a transform's samples together with the batch
	 * stepping over them, so the sample axes take the fastest strides and
	 * the rest follow.  BART's own order is already that unless a sample
	 * axis sits above a batch axis. */
	long stride = 1;

	for (int i = 1; i < N; i++)
		if (1 < traj_dims[i]) {

			d->trf_strs[i] = (1 == grd_dims[i]) ? 0 : stride * (long)CFL_SIZE;
			stride *= grd_dims[i];
		}

	for (int i = 0; i < N; i++)
		if ((0 == i) || (1 >= traj_dims[i])) {

			d->trf_strs[i] = (1 == grd_dims[i]) ? 0 : stride * (long)CFL_SIZE;
			stride *= grd_dims[i];
		}

	long grd_strs[N];
	md_calc_strides(N, grd_strs, grd_dims, CFL_SIZE);

	d->needs_tmp = (NULL != basis) || (0 != memcmp(d->trf_strs, grd_strs, (size_t)N * sizeof(long)));

	if (NULL != basis) {

		md_copy_dims(N, d->bas_dims, bas_dims);
		md_calc_strides(N, d->bas_strs, bas_dims, CFL_SIZE);
		d->host_basis = md_alloc(N, bas_dims, CFL_SIZE);
		md_copy(N, bas_dims, d->host_basis, basis, CFL_SIZE);
	}

	if (NULL != weights) {

		md_copy_dims(N, d->wgh_dims, wgh_dims);
		d->host_weights = md_alloc(N, wgh_dims, CFL_SIZE);
		md_copy(N, wgh_dims, d->host_weights, weights, CFL_SIZE);
		md_calc_strides(N, d->wgh_strs, wgh_dims, CFL_SIZE);
	}

	/* The side the operator is most likely to be asked for first, so that a
	 * plan FINUFFT will not make is a decline here rather than an error in
	 * the middle of a solve. */
	int ret = side_build(d, device);

	if (0 != ret) {

		nufft_fi_del(CAST_UP(PTR_PASS(d)));
		DECLINE(ret);
	}

	d->toeplitz = toeplitz_for(N, ksp_dims, cim_dims, traj_dims, traj, wgh_dims, weights,
			bas_dims, basis, conf);

	/* PTR_PASS hands the data over and clears the pointer, so what the
	 * operator is built with is read out first. */
	lop_fun_t normal = (NULL != d->toeplitz) ? nufft_fi_normal : NULL;

	struct linop_s* op = linop_create(N, out_dims, N, cim_dims, CAST_UP(PTR_PASS(d)),
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
	struct linop_s* op = try_create(N, ksp_dims, cim_dims, traj_dims, traj, wgh_dims, weights, bas_dims, basis, conf);

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
