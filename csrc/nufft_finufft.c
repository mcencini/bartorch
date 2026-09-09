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

#ifdef _OPENMP
#include <omp.h>
#endif
#include <float.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/types.h"

#include "linops/linop.h"

#include "num/flpmath.h"
#include "num/init.h"
#include "num/multind.h"
#include "num/shuffle.h"

#include "noncart/nufft.h"
#include "noncart/nufft_priv.h"

#include "noncart/grid.h"

#include "num/compress.h"
#include "num/multiplace.h"
#include "num/triagmat.h"
#include "num/gpuops.h"

#include "include/bartorch.h"

extern struct linop_s* bart_nufft_create2(int N, const long ksp_dims[N], const long cim_dims[N], const long traj_dims[N], const complex float* traj, const long wgh_dims[N], const complex float* weights, const long bas_dims[N], const complex float* basis, struct nufft_conf_s conf);
extern int bart_nufft_get_psf_dims(const struct linop_s* nufft, int N, long psf_dims[N]);
extern void bart_nufft_get_psf(const struct linop_s* nufft, int N, const long psf_dims[N], complex float* psf);
extern void bart_nufft_get_psf2(const struct linop_s* nufft, int N, const long psf_dims[N], const long psf_strs[N], complex float* psf);
extern void bart_nufft_update_psf(const struct linop_s* nufft, int ND, const long psf_dims[ND], const complex float* psf);
extern void bart_nufft_update_psf2(const struct linop_s* nufft, int ND, const long psf_dims[ND], const long psf_strs[ND], const complex float* psf);
extern void bart_nufft_update_traj(const struct linop_s* nufft, int N, const long trj_dims[N], const complex float* traj, const long wgh_dims[N], const complex float* weights, const long bas_dims[N], const complex float* basis);
extern const struct operator_s* bart_nufft_precond_create(const struct linop_s* nufft_op);

/* Provided by psf.c, which computes the function this convolves with. */
extern void bartorch_psf_shift(int NS, float shift[NS], int N, const long factors[N], int idx);
extern complex float* bartorch_psf_to_host(int N, const long psf_dims[N + 1], unsigned long flags,
		const long trj_dims[N + 1], const complex float* traj,
		const long bas_dims[N + 1], const complex float* basis,
		const long wgh_dims[N + 1], const complex float* weights,
		bool periodic, bool lowmem, bool upper_triag);

/* Provided by finufft.c, which owns the FINUFFT entry points. */
extern int bartorch_finufft_plan(int device, int type, int dim, const int64_t n_modes[3],
		int ntrans, int isign, double eps, double upsampling, int spread_only, void** plan);
extern int bartorch_finufft_setpts(void* plan, long M, float* x, float* y, float* z);
extern int bartorch_finufft_exec(void* plan, complex float* c, complex float* f);
extern void bartorch_finufft_free(void* plan);
extern double bartorch_finufft_tolerance(void);
extern double bartorch_finufft_upsampling(void);

/* ------------------------------------------------------------------------ */

/* Normal operators built since the last reset, so a test can say that a
 * solve ran on a point spread function rather than on the transform pair. */
enum { TP_PSF, TP_PAIR };
static long toeplitz_counters[2];

/* Building a point spread function needs a transform of its own, and that
 * transform is nobody's normal: it is asked for one adjoint and freed.  While
 * one is being made, an operator built underneath does not count as having
 * answered a normal either way. */
static _Thread_local int making_psf;

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
	int axis[3];
	int64_t n_modes[3];
	double eps;
	double upsampling;

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
	long* trj_dims;
	long* ksp_dims;

	/* What the normal is built from, for a trajectory that arrives after the
	 * operator: a point spread function is over one, so there is none to
	 * convolve with until there is a trajectory to make it from. */
	struct nufft_conf_s conf;
	long* bas_strs;

	/* The function the normal convolves with, kept where the card is not.
	 *
	 * `toeplitz_mult_lowmem` reads it as one array and takes the set of
	 * frequencies it wants out of it, so BART brings the whole function
	 * over the first time a normal is applied on a card -- and for a
	 * subspace problem that function is coefficients by sets by image,
	 * which is the one thing in a three-dimensional reconstruction that
	 * does not fit.  So the loop over sets is driven from here instead:
	 * BART is left believing it has one, and the one it has is swapped for
	 * each in turn.  The arithmetic is still its own.
	 */
	struct nufft_data* toeplitz_data;
	complex float* psf_host;
	complex float* linphase_host;
	void* psf_slot;			/* where a set lands, made once */
	complex float* linphase_slot;
	long psf_coset;			/* elements in one set of the function */
	long linphase_coset;
	size_t psf_size;		/* a real function is stored as floats */
	int cosets;
};

static DEF_TYPEID(nufft_fi_s);

/* BART's trajectory counts samples of the image grid and FINUFFT takes the
 * same position in radians.  One component is taken and rescaled with BART's
 * own operations, on the host, and a side copies it to wherever its plans
 * are; md_copy2 crosses the bus if the trajectory is on the other side.
 *
 * A trajectory can arrive after the operator: `nlinv` and the network models
 * build theirs against dimensions alone and fill it in with
 * `nufft_update_traj` once there is one. */
static void install_traj(struct nufft_fi_s* d, const long traj_dims[], const complex float* traj)
{
	int N = d->N;

	long one_dims[N];
	long one_strs[N];
	long trj_strs[N];

	md_select_dims(N, ~1UL, one_dims, traj_dims);
	md_calc_strides(N, one_strs, one_dims, CFL_SIZE);
	md_calc_strides(N, trj_strs, traj_dims, CFL_SIZE);

	complex float* component = md_alloc(N, one_dims, CFL_SIZE);

	for (int i = 0; i < d->dim; i++) {

		md_copy2(N, one_dims, one_strs, component, trj_strs, traj + d->axis[i], CFL_SIZE);

		md_free(d->radians[i]);
		d->radians[i] = md_alloc(N, one_dims, FL_SIZE);
		md_real(N, one_dims, d->radians[i], component);
		md_smul(N, one_dims, d->radians[i], d->radians[i],
				(float)(2. * M_PI / (double)d->cim_dims[d->axis[i]]));
	}

	md_free(component);
}

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

	if (NULL == d->radians[0])
		return 18;

	s->ntrans = which ? 1 : (int)d->batch;
	s->executes = which ? d->batch : 1;

	if (0 != bartorch_finufft_plan(which, 2, d->dim, d->n_modes, s->ntrans, -1, d->eps, d->upsampling, 0, &s->forward_plan))
		return 11;

	if (0 != bartorch_finufft_plan(which, 1, d->dim, d->n_modes, s->ntrans, +1, d->eps, d->upsampling, 0, &s->adjoint_plan)) {

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

/* Streams, where there are any.  BART hands every `md_` call the stream of
 * the OpenMP thread that issued it, and forgets which level owns them as soon
 * as anything asks for one from below, so this is armed just before a region
 * rather than once. */
#ifdef USE_CUDA
static int stream_count(void) { return cuda_set_stream_level(); }
static void stream_wait(void) { cuda_sync_stream(); }
#else
static int stream_count(void) { return 1; }
static void stream_wait(void) { }
#endif

/* Whether the function is kept off the card and brought over a set of
 * frequencies at a time.
 *
 * On, because the function is the largest thing a three-dimensional subspace
 * reconstruction holds and this is what decides whether one fits: on 96^3 with
 * eight coils it is 190 MB against 364, and against 576 for a function built
 * whole.  With the fused multiply below it costs nothing in time.
 *
 * It applies only where there is a card to keep the function off; on the host
 * it would be copies to no purpose, so the trajectory's side decides. */
static int stream_psf_enabled = 1;

void bartorch_nufft_set_stream_psf(int enable)
{
	stream_psf_enabled = (0 != enable);
}

int bartorch_nufft_stream_psf(void)
{
	return stream_psf_enabled;
}

/* Where a set of frequencies lands, made once and written into thereafter.
 *
 * `multiplace_move_wrapper` leaves the array where it is handed and does not
 * free it, so these stay ours: what happens per set is one copy across rather
 * than an allocation, a copy and a free.  There are two of each because the
 * next set is fetched while this one is used. */
static void open_slots(struct nufft_fi_s* d, const void* ref)
{
	struct nufft_data* t = d->toeplitz_data;
	int ND = t->N + 1;

	long psf_dims[ND];
	md_copy_dims(ND, psf_dims, t->psf_dims);
	psf_dims[t->N] = 1;

	long lph_dims[ND];
	md_copy_dims(ND, lph_dims, t->lph_dims);
	lph_dims[t->N] = 1;

	/* One slot is enough.  The copy that fills it for the next turn is
	 * issued on the stream the convolution is already on, and a stream
	 * keeps what it was given in order, so it cannot overtake the read it
	 * follows. */
	d->psf_slot = md_alloc_sameplace(ND, psf_dims, d->psf_size, ref);
	d->linphase_slot = md_alloc_sameplace(ND, lph_dims, CFL_SIZE, ref);
}

/* Bring one set of frequencies over, into the slot given. */
static void fetch_coset(struct nufft_fi_s* d, int i)
{
	struct nufft_data* t = d->toeplitz_data;
	int ND = t->N + 1;

	long psf_dims[ND];
	md_copy_dims(ND, psf_dims, t->psf_dims);
	psf_dims[t->N] = 1;

	long lph_dims[ND];
	md_copy_dims(ND, lph_dims, t->lph_dims);
	lph_dims[t->N] = 1;

	md_copy(ND, psf_dims, d->psf_slot,
			(const char*)d->psf_host + (size_t)i * (size_t)d->psf_coset * d->psf_size,
			d->psf_size);

	md_copy(ND, lph_dims, d->linphase_slot,
			d->linphase_host + (long)i * d->linphase_coset, CFL_SIZE);
}

/* Point BART at the slot a set has landed in. */
static void use_coset(struct nufft_fi_s* d)
{
	struct nufft_data* t = d->toeplitz_data;
	int ND = t->N + 1;

	long psf_dims[ND];
	md_copy_dims(ND, psf_dims, t->psf_dims);
	psf_dims[t->N] = 1;

	long lph_dims[ND];
	md_copy_dims(ND, lph_dims, t->lph_dims);
	lph_dims[t->N] = 1;

	if (NULL != t->psf)
		multiplace_free(t->psf);

	t->psf = multiplace_move_wrapper(ND, psf_dims, d->psf_size, d->psf_slot);

	if (NULL != t->linphase)
		multiplace_free(t->linphase);

	t->linphase = multiplace_move_wrapper(ND, lph_dims, CFL_SIZE, d->linphase_slot);
}

/* One set of frequencies, convolved and added to what is there.
 *
 * This is `toeplitz_mult_lowmem` without the two things that cost a pass over
 * the coil images: it accumulates into the answer rather than clearing it, so
 * the caller needs no second image to add up, and it multiplies the function
 * in place where the shape allows.  Every number is BART's -- the same
 * `md_zmul2`, the same transform, the same contraction against the upper
 * triangle -- and the arrangements it does not cover go back to BART's own.
 */
static bool fused_coset(struct nufft_fi_s* d, complex float* dst, const complex float* src)
{
	struct nufft_data* t = d->toeplitz_data;

	/* A compressed function is scattered and gathered around the multiply,
	 * which is BART's to do. */
	if (NULL != t->compress)
		return false;

	const complex float* linphase = multiplace_read(t->linphase, src);
	const void* psf = multiplace_read(t->psf, src);

	if ((NULL == linphase) || (NULL == psf))
		return false;

	complex float* grid = md_alloc_sameplace(t->N, t->cim_dims, CFL_SIZE, dst);

	md_zmul2(t->N, t->cim_dims, t->cim_strs, grid, t->cim_strs, src, t->img_strs, linphase);

	linop_forward(t->cfft_op, t->N, t->cim_dims, grid, t->N, t->cim_dims, grid);

	/* Without a basis the function is a diagonal and multiplies in place;
	 * with one it is a matrix at every frequency and the contraction needs
	 * somewhere to land. */
	bool contracts = !md_check_equal_dims(t->N, t->cim_dims, t->ciT_dims, ~0UL);

	if (!contracts) {

		if (t->conf.real)
			md_mul2(t->N, MD_REAL_DIMS(t->N, t->cim_dims),
					MD_REAL_STRS(t->N, t->cim_strs, FL_SIZE), (float*)grid,
					MD_REAL_STRS(t->N, t->cim_strs, FL_SIZE), (float*)grid,
					MD_REAL_STRS(t->N, t->psf_strs, 0), (const float*)psf);
		else
			md_zmul2(t->N, t->cim_dims, t->cim_strs, grid, t->cim_strs, grid, t->psf_strs, psf);

	} else {

		long max_dims[t->N];
		md_max_dims(t->N, ~0UL, max_dims, t->ciT_dims, t->cim_dims);

		long ciT_strs[t->N];
		md_calc_strides(t->N, ciT_strs, t->ciT_dims, CFL_SIZE);

		long cim_strs[t->N];
		md_calc_strides(t->N, cim_strs, t->cim_dims, CFL_SIZE);

		complex float* out = md_alloc_sameplace(t->N, t->ciT_dims, CFL_SIZE, dst);

		if (t->conf.real) {

			if (t->conf.upper_triag)
				md_tenmul_upper_triag2(6, 7, t->N + 1, MD_REAL_DIMS(t->N, max_dims),
						MD_REAL_STRS(t->N, ciT_strs, FL_SIZE), (float*)out,
						MD_REAL_STRS(t->N, cim_strs, FL_SIZE), (float*)grid,
						MD_REAL_DIMS(t->N, t->psf_dims),
						MD_REAL_STRS(t->N, t->psf_strs, 0), (const float*)psf);
			else
				md_tenmul2(t->N + 1, MD_REAL_DIMS(t->N, max_dims),
						MD_REAL_STRS(t->N, ciT_strs, FL_SIZE), (float*)out,
						MD_REAL_STRS(t->N, cim_strs, FL_SIZE), (float*)grid,
						MD_REAL_STRS(t->N, t->psf_strs, 0), (const float*)psf);

		} else {

			if (t->conf.upper_triag)
				md_ztenmul_upper_triag(5, 6, t->N, t->ciT_dims, out, t->cim_dims, grid, t->psf_dims, psf);
			else
				md_ztenmul(t->N, t->ciT_dims, out, t->cim_dims, grid, t->psf_dims, psf);
		}

		md_free(grid);
		grid = out;
	}

	linop_adjoint(t->cfft_op, t->N, t->cim_dims, grid, t->N, t->cim_dims, grid);

	/* Into the answer, not over it: this is what the second image was for. */
	md_zfmacc2(t->N, t->cim_dims, t->cim_strs, dst, t->cim_strs, grid, t->img_strs, linphase);

	md_free(grid);

	return true;
}

static void nufft_fi_normal(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	struct nufft_fi_s* d = CAST_DOWN(nufft_fi_s, _d);

	if (NULL == d->toeplitz)
		error("bartorch: the normal was asked for before the trajectory arrived\n");

	if (NULL == d->psf_host) {

		linop_normal_unchecked(d->toeplitz, dst, src);
		return;
	}

	struct nufft_data* t = d->toeplitz_data;

	/* BART clears its output and accumulates the sets into it; driving one
	 * at a time means accumulating them here instead.  Swapping what the
	 * operator points at is not something two applications can do at once. */
	complex float* part = NULL;

	pthread_mutex_lock(&d->lock);

	if (NULL == d->psf_slot)
		open_slots(d, dst);

	md_clear(t->N, t->cim_dims, dst, CFL_SIZE);

	/* The sets are brought over one at a time, and the fetch is not put on
	 * a stream of its own.
	 *
	 * It could be: the copy and the convolution would then run at once, as
	 * they do where a slab of sensitivities is fetched.  But what is being
	 * overlapped there is a transform on the card, and what is being
	 * overlapped here is BART's own arithmetic, which threads.  Putting it
	 * inside a region of two makes that threading nested, and nested it
	 * does not happen -- measured at 96^3 with eight coils, a region of two
	 * costs 6.48 s against 2.75 s and saves 8 MB of 210.  So the copy is
	 * issued on the stream the arithmetic is already on, where it is
	 * asynchronous and the next turn waits for no more than it must.
	 */
	use_coset(d);

	for (int i = 0; i < d->cosets; i++) {

		fetch_coset(d, i);

		if (fused_coset(d, dst, src))
			continue;

		if (NULL == part)
			part = md_alloc_sameplace(t->N, t->cim_dims, CFL_SIZE, dst);

		linop_normal_unchecked(d->toeplitz, part, src);
		md_zadd(t->N, t->cim_dims, dst, dst, part);
	}

	pthread_mutex_unlock(&d->lock);

	md_free(part);
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
	md_free(d->psf_host);
	md_free(d->linphase_host);

	md_free(d->psf_slot);
	md_free(d->linphase_slot);

	xfree(d->cim_dims);
	xfree(d->out_dims);
	xfree(d->out_strs);
	xfree(d->grd_dims);
	xfree(d->trf_strs);
	xfree(d->wgh_dims);
	xfree(d->wgh_strs);
	xfree(d->bas_dims);
	xfree(d->trj_dims);
	xfree(d->ksp_dims);
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

/* Why the last operator was not FINUFFT's.  The words live here rather than
 * in the host so that the two cannot drift apart. */
const char* bartorch_nufft_decline_text(void)
{
	switch (decline_reason) {

	case 0: return "";
	case 1: return "FINUFFT is not in use, and installing itself did not work";
	case 2: return "k-space carries fewer than four axes";
	case 3: return "cuFINUFFT is not in use and BART is on a device";
	case 4: return "the trajectory does not carry three components";
	case 5: return "k-space is not a single line of samples per readout";
	case 6: return "the transform is over axes other than the spatial three";
	case 7: return "the image has no spatial extent";
	case 8: return "the trajectory and k-space disagree on the samples";
	case 9: return "k-space and the coil images disagree beyond the spatial axes";
	case 10: return "there are more frames than one plan can batch";
	case 11: return "FINUFFT would not plan the forward transform";
	case 12: return "FINUFFT would not plan the adjoint transform";
	case 13: return "FINUFFT would not take the trajectory";
	case 14: return "the subspace basis does not lie along frames and coefficients";
	case 15: return "the weights do not lie along k-space";
	case 16: return "the images vary across frames as well as the trajectory";
	case 17: return "the kernel width asked for has no tolerance that would give it";
	case 18: return "a transform was asked for before the trajectory arrived";
	}

	return "of a reason this build does not name";
}

/* Whether BART's own operator may answer what FINUFFT will not.
 *
 * Off, a transform FINUFFT cannot serve is an error rather than a quieter
 * reconstruction: a caller who asked for FINUFFT gets it or gets told why
 * not, instead of BART's gridder standing in unannounced. */
static bool allow_fallback;

void bartorch_nufft_allow_fallback(int enable)
{
	allow_fallback = (0 != enable);
}

int bartorch_nufft_fallback_allowed(void)
{
	return allow_fallback ? 1 : 0;
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

/* BART's own conf: the oversampling it would have had if we had not taken
 * zero to mean that nobody asked for one. */
static struct nufft_conf_s barts_conf(struct nufft_conf_s conf)
{
	if (0. == conf.os)
		conf.os = nufft_conf_defaults.os;

	if (0.f == conf.width)
		conf.width = nufft_conf_defaults.width;

	return conf;
}

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
/* What FINUFFT's kernel spans, asked of the library rather than worked out.
 *
 * It sizes the kernel from the tolerance and the upsampling by a formula of
 * its own, and a copy of that formula here would be a copy that goes stale
 * quietly: it lives in FINUFFT's `src/common/kernel.cpp`, the only thing
 * exported for it is a C++ symbol over an internal struct, and cuFINUFFT
 * exports nothing at all.  So the library is asked instead.  Spreading one
 * sample with nothing after it -- no transform, no deapodisation -- puts the
 * kernel on the grid, and what came back is as wide as the kernel is.  The
 * two libraries are asked separately, because nothing says they must agree.
 */
static int fi_measure_width(int device, int dim, double eps, double upsampling)
{
	enum { PROBE = 32 };	/* wider than any kernel FINUFFT will use */

	int64_t n_modes[3] = { PROBE, PROBE, PROBE };
	long grid_dims[3] = { 1, 1, 1 };
	long one[1] = { 1 };

	for (int i = 0; i < dim; i++)
		grid_dims[i] = PROBE;

	void* plan = NULL;

	if (0 != bartorch_finufft_plan(device, 1, dim, n_modes, 1, +1, eps, upsampling, 1, &plan))
		return -1;

	float* coord[3] = { NULL, NULL, NULL };
	complex float* sample = alloc_on(device, 1, one, CFL_SIZE);
	complex float* grid = alloc_on(device, 3, grid_dims, CFL_SIZE);

	for (int i = 0; i < dim; i++) {

		coord[i] = alloc_on(device, 1, one, FL_SIZE);
		md_clear(1, one, coord[i], FL_SIZE);
	}

	md_zfill(1, one, sample, 1.);
	md_clear(3, grid_dims, grid, CFL_SIZE);

	int ret = bartorch_finufft_setpts(plan, 1, coord[0], coord[1], coord[2]);

	if (0 == ret)
		ret = bartorch_finufft_exec(plan, sample, grid);

	bartorch_finufft_free(plan);
	md_free(sample);

	for (int i = 0; i < 3; i++)
		md_free(coord[i]);

	if (0 != ret) {

		md_free(grid);
		return -1;
	}

	complex float* host = md_alloc(3, grid_dims, CFL_SIZE);
	md_copy(3, grid_dims, host, grid, CFL_SIZE);
	md_free(grid);

	/* Along one axis through the middle, where the sample was put. */
	long stride = 1;
	long centre = 0;

	for (int i = 1; i < dim; i++) {

		stride *= PROBE;
		centre += (PROBE / 2) * stride;
	}

	int width = 0;

	for (int i = 0; i < PROBE; i++)
		if (0. != cabsf(host[centre + i]))
			width++;

	md_free(host);

	return (0 == width) ? -1 : width;
}

/* The tolerance that buys a width, by asking for widths until one of them is
 * it.  The width falls as the tolerance rises, so this is a bisection; the
 * formula FINUFFT publishes is a good enough starting bracket to make it a
 * short one, and is never the answer. */
static double fi_tolerance_for(int device, int dim, double width, double upsampling)
{
	double tolfac = 0.18;

	for (int i = 1; i < dim; i++)
		tolfac *= 1.4;

	double guess = tolfac * exp(-(width - 1.) * M_PI * sqrt(1. - 1. / upsampling));

	if (width == (double)fi_measure_width(device, dim, guess, upsampling))
		return guess;

	double lo = (double)FLT_EPSILON;	/* the widest kernel it will make */
	double hi = 0.5;			/* the narrowest */

	for (int i = 0; i < 40; i++) {

		double mid = sqrt(lo * hi);
		int got = fi_measure_width(device, dim, mid, upsampling);

		if (got < 0)
			return guess;

		if (got > width)
			lo = mid;
		else if (got < width)
			hi = mid;
		else
			return mid;
	}

	return guess;
}

/* The mask a compressed point spread function keeps: which grid points the
 * samples reach.
 *
 * BART finds them by spreading the sampling pattern with its own kernel, and
 * that is the wrong footprint for a function spread with another: what the
 * mask covers has to be where the function has signal, and the function is
 * FINUFFT's now.  So the pattern is spread with FINUFFT's kernel instead, at
 * the tolerance and grid the transforms were planned with.
 *
 * `spreadinterponly` is the spreading with nothing after it -- no transform,
 * no deapodisation -- so what comes back is the kernel's own reach, on
 * whichever grid it is asked for.  It is asked for the one the mask lives on,
 * a set of frequencies at a time: the doubled grid decomposes into as many
 * copies of the image, each carrying the samples shifted by its own fraction
 * of a cell, and a point any of them reaches is a point the mask keeps.  That
 * is also what keeps the doubled grid from ever being allocated, which is the
 * whole reason the decomposition is there.
 */
static int spread_mask(struct nufft_data* data, const complex float* traj, long* max_idx)
{
	int N = data->N;
	int ND = N + 1;

	const complex float* pattern = multiplace_read(data->weights, traj);

	if (NULL == pattern)
		error("bartorch: a compressed point spread function needs a pattern\n");

	int device = bartorch_on_device(traj) ? 1 : 0;

	int dim = 0;
	int axis[3];
	int64_t n_modes[3];

	for (int i = 0; i < 3; i++) {

		if (!MD_IS_SET(data->flags, i) || (1 == data->img_dims[i]))
			continue;

		axis[dim] = i;
		n_modes[dim] = data->img_dims[i];
		dim++;
	}

	if (0 == dim)
		return -1;

	long factors[N];

	for (int i = 0; i < N; i++)
		factors[i] = ((data->img_dims[i] > 1) && MD_IS_SET(data->flags, i)) ? 2 : 1;

	long sets = md_calc_size(N, factors);

	long one_dims[ND];
	long one_strs[ND];
	long trj_strs[ND];
	long wgh_strs[ND];

	md_select_dims(ND, ~1UL, one_dims, data->trj_dims);
	md_calc_strides(ND, one_strs, one_dims, CFL_SIZE);
	md_calc_strides(ND, trj_strs, data->trj_dims, CFL_SIZE);
	md_calc_strides(ND, wgh_strs, data->wgh_dims, CFL_SIZE);

	long samples = md_calc_size(ND, one_dims);

	/* One component of the trajectory per transformed axis, in grid samples,
	 * before the shift of a set turns it into that set's coordinates. */
	complex float* component = alloc_on(device, ND, one_dims, CFL_SIZE);
	float* grid_samples[3] = { NULL, NULL, NULL };

	for (int i = 0; i < dim; i++) {

		md_copy2(ND, one_dims, one_strs, component, trj_strs, traj + axis[i], CFL_SIZE);

		grid_samples[i] = alloc_on(device, ND, one_dims, FL_SIZE);
		md_real(ND, one_dims, grid_samples[i], component);
	}

	md_free(component);

	complex float* samples_in = alloc_on(device, ND, one_dims, CFL_SIZE);
	md_copy2(ND, one_dims, one_strs, samples_in, wgh_strs, pattern, CFL_SIZE);

	complex float* reach = alloc_on(device, ND, data->com_dims, CFL_SIZE);
	complex float* mask = alloc_on(device, ND, data->com_dims, CFL_SIZE);
	md_clear(ND, data->com_dims, mask, CFL_SIZE);

	float* coord[3] = { NULL, NULL, NULL };

	for (int i = 0; i < dim; i++)
		coord[i] = alloc_on(device, ND, one_dims, FL_SIZE);

	long com_strs[ND];
	md_calc_strides(ND, com_strs, data->com_dims, CFL_SIZE);

	/* The width to spread the mask with.
	 *
	 * A kernel of ns cells on a grid oversampled by sigma covers ns/sigma
	 * cells of the grid underneath it, and that is the footprint the mask
	 * has to have -- BART says the same thing as a width of K/2 at os 1 for
	 * a transform of width K at os 2.  FINUFFT will not be asked for an
	 * upsampling of one, and takes no width, so the width is asked for as
	 * the tolerance that buys it.
	 */
	/* The kernel the function was spread with, which is the one the mask has
	 * to cover.  `compute_psf2` asks for it the way any transform here does,
	 * so it is the configured one and not the operator's: a caller's `-o` or
	 * `-w` reaches the transform pair, and BART's own point spread function
	 * ignores them too. */
	double upsampling = bartorch_finufft_upsampling();

	if (upsampling <= 1.)
		upsampling = 2.;

	int spread = fi_measure_width(device, dim, bartorch_finufft_tolerance(), upsampling);

	if (spread < 0)
		return -1;

	double width = ceil((double)spread / upsampling);

	if (width < 2.)
		width = 2.;

	double mask_eps = fi_tolerance_for(device, dim, width, upsampling);

	debug_printf(DP_DEBUG2, "PSF mask spread at width %g, tolerance %g\n", width, mask_eps);

	int ret = 0;

	for (long set = 0; (0 == ret) && (set < sets); set++) {

		float shift[3];
		bartorch_psf_shift(3, shift, N, factors, (int)set);

		for (int i = 0; i < dim; i++) {

			int a = axis[i];
			double scale = 2. * M_PI / (double)data->img_dims[a];

			/* The half sample an odd length carries, as `nufft.c` adds it. */
			float odd = (float)((data->img_dims[a] / 2.0 - data->img_dims[a] / 2));

			md_smul(ND, one_dims, coord[i], grid_samples[i], (float)scale);
			md_sadd(ND, one_dims, coord[i], coord[i], (float)((shift[a] + odd) * scale));
		}

		md_clear(ND, data->com_dims, reach, CFL_SIZE);

		void* plan = NULL;

		ret = bartorch_finufft_plan(device, 1, dim, n_modes, 1, +1,
				mask_eps, upsampling, 1, &plan);

		if (0 == ret)
			ret = bartorch_finufft_setpts(plan, samples, coord[0], coord[1], coord[2]);

		if (0 == ret)
			ret = bartorch_finufft_exec(plan, samples_in, reach);

		bartorch_finufft_free(plan);

		if (0 != ret)
			break;

		md_zabs(ND, data->com_dims, reach, reach);
		md_zmax(ND, data->com_dims, mask, mask, reach);
	}

	md_free(samples_in);
	md_free(reach);

	for (int i = 0; i < 3; i++) {

		md_free(grid_samples[i]);
		md_free(coord[i]);
	}

	if (0 != ret) {

		md_free(mask);
		return -1;
	}

	complex float* mask_cpu = md_alloc(ND, data->com_dims, CFL_SIZE);
	md_copy(ND, data->com_dims, mask_cpu, mask, CFL_SIZE);
	md_free(mask);

	long* idx = md_alloc(ND, data->com_dims, sizeof(long));
	*max_idx = md_compress_mask_to_index(ND, data->com_dims, idx, mask_cpu);
	md_free(mask_cpu);

	multiplace_free(data->compress);
	data->compress = multiplace_move_F(ND, data->com_dims, sizeof(long), idx);

	debug_printf(DP_DEBUG1, "Compressing PSF to %.0f%%\n",
			100. * *max_idx / md_calc_size(ND, data->com_dims));

	return 0;
}

/* The function BART's Toeplitz normal convolves with, computed here and
 * stored the way the operator wants it.
 *
 * `nufft.c` would compute one for itself, with its own gridder, from inside
 * the file where the rename cannot reach it -- `conf.nopsf` is what stops it,
 * and is what `pics --psf_import` uses to bring one in from outside.  What is
 * left is to make the function and store it, which is the block this mirrors:
 * the same `md_real` for a real one and the same `md_compress` for a
 * compressed one, over dimensions the operator worked out for itself rather
 * than any derived again here.
 */
/* Whether a basis leaves the function real.
 *
 * `U^H diag(m) U` is real when `U` is, and a basis that is real once turned
 * through a single angle is as good: the angle appears as `conj(e^{it}) e^{it}`
 * and cancels.  A basis is small enough to ask about on the host, and the
 * angle is half the argument of the sum of its squares -- which is `e^{2it}`
 * times something real when there is one angle to find. */
static bool basis_is_real(int N, const long bas_dims[N], const complex float* basis)
{
	if (NULL == basis)
		return true;

	long size = md_calc_size(N, bas_dims);

	complex float* host = md_alloc(N, bas_dims, CFL_SIZE);
	md_copy(N, bas_dims, host, basis, CFL_SIZE);

	complex float squares = 0.;
	double energy = 0.;

	for (long i = 0; i < size; i++) {

		squares += host[i] * host[i];
		energy += (double)crealf(host[i]) * crealf(host[i]) + (double)cimagf(host[i]) * cimagf(host[i]);
	}

	complex float turn = (0. == cabsf(squares)) ? 1. : conjf(csqrtf(squares / cabsf(squares)));

	double left = 0.;

	for (long i = 0; i < size; i++) {

		float part = cimagf(host[i] * turn);
		left += (double)part * part;
	}

	md_free(host);

	bool real = (0. == energy) || (left <= 1.e-12 * energy);

	debug_printf(DP_DEBUG1, "Basis is %sreal, %g of it left after one turn\n",
			real ? "" : "not ", (0. == energy) ? 0. : sqrt(left / energy));

	return real;
}

static void install_psf(struct nufft_data* data, const complex float* traj, complex float** to_host)
{
	int N = data->N;
	int ND = N + 1;

	const complex float* weights = multiplace_read(data->weights, traj);
	const complex float* basis = multiplace_read(data->basis, traj);

	/* A function with nothing in its imaginary part is stored as floats,
	 * which halves it; the basis is what says whether there is anything
	 * there.  It is asked before the function is built, because whether it
	 * comes out real decides where it is built. */
	bool store_real = data->conf.real || basis_is_real(ND, data->bas_dims, basis);

	/* Streamed, the function is made a set of frequencies at a time and
	 * kept on the host: neither it nor any set but the one being made is
	 * ever resident.  A compressed function is not served that way.
	 *
	 */
	bool stream = (NULL != to_host) && stream_psf_enabled
		&& (0 != bartorch_on_device(traj)) && !data->conf.compress_psf;


	complex float* psf = stream
		? bartorch_psf_to_host(N, data->psf_dims, data->flags, data->trj_dims, traj,
				data->bas_dims, basis, data->wgh_dims, weights,
				true, data->conf.lowmem, data->conf.upper_triag)
		: (data->conf.decomposed_psf ? compute_psf2_decomposed : compute_psf2)(N,
				data->psf_dims, data->flags, data->trj_dims, traj,
				data->bas_dims, basis, data->wgh_dims, weights,
				true /* as nufft.c asks for it */, data->conf.lowmem, data->conf.upper_triag);

	long max_idx = 0;

	if (data->conf.compress_psf && !stream) {

		md_select_dims(ND, FFT_FLAGS, data->com_dims, data->img_dims);

		if (0 != spread_mask(data, traj, &max_idx))
			error("bartorch: FINUFFT would not spread the pattern for a compressed function\n");
	}

	multiplace_free(data->psf);

	if (store_real) {

		float* psf_real = md_alloc_sameplace(ND, data->psf_dims, FL_SIZE, psf);
		md_real(ND, data->psf_dims, psf_real, psf);
		md_free(psf);
		psf = (complex float*)psf_real;

		md_calc_strides(ND, data->psf_strs, data->psf_dims, FL_SIZE);

		/* What is stored is what the multiply has to read. */
		data->conf.real = true;
	}

	if (stream) {

		/* The sets are put where BART looks for them one at a time, so
		 * nothing is installed here -- but what was freed above has to
		 * stop being pointed at, or an arrangement that turns out not to
		 * be streamable after all reads it. */
		data->psf = NULL;
		*to_host = psf;
		return;
	}

	data->psf = multiplace_move_F(ND, data->psf_dims, store_real ? FL_SIZE : CFL_SIZE, psf);

	if (NULL != data->compress) {

		size_t size = data->conf.real ? FL_SIZE : CFL_SIZE;

		long com_psf_dims[ND];
		md_compress_dims(ND, com_psf_dims, data->psf_dims, data->com_dims, max_idx);

		complex float* com_psf = md_alloc_sameplace(ND, com_psf_dims, size, traj);
		md_compress(ND, com_psf_dims, com_psf, data->psf_dims, multiplace_read(data->psf, traj),
				data->com_dims, multiplace_read(data->compress, traj), size);

		multiplace_free(data->psf);

		md_copy_dims(ND, data->psf_dims, com_psf_dims);
		md_calc_strides(ND, data->psf_strs, data->psf_dims, size);
		data->psf = multiplace_move_F(ND, data->psf_dims, size, com_psf);
	}
}

/* BART's own normal, over a function computed here.
 *
 * A^H A is a convolution, so it is one multiply against a function rather than
 * a transform each way, and `nufft.c` has the machinery for that: the
 * oversampled grid, the linear phases, the decomposition, and the three ways
 * of storing the function that make it fit.  All of that is kept.  What is not
 * is the gridding it would do to build the function, which `nopsf` turns off
 * and `install_psf` replaces.
 *
 * The caller decides whether there is one at all: `conf.toeplitz` is what
 * `pics --no-toeplitz` and `nufft -t` set, and it carries the memory the
 * function costs.
 */
/* Take the function off the card.
 *
 * What is left behind is BART's operator believing it has a single set of
 * frequencies, so one call to its normal does one of them; the loop that walks
 * them is the caller's.  Only the plainest arrangement is taken. */
static void stream_psf(struct nufft_fi_s* d)
{
	if (!stream_psf_enabled || (NULL == d->toeplitz) || (NULL == d->psf_host))
		return;

	struct nufft_data* t = CAST_DOWN(nufft_data, linop_get_data_nested(d->toeplitz));

	if (NULL == t->linphase) {

		md_free(d->psf_host);
		d->psf_host = NULL;
		return;
	}

	int ND = t->N + 1;
	int cosets = (int)t->lph_dims[t->N];

	if ((cosets < 2) || (t->psf_dims[t->N] != cosets)) {

		md_free(d->psf_host);
		d->psf_host = NULL;
		return;
	}

	const complex float* lph = multiplace_read(t->linphase, d->radians[0]);

	d->linphase_host = md_alloc(ND, t->lph_dims, CFL_SIZE);
	md_copy(ND, t->lph_dims, d->linphase_host, lph, CFL_SIZE);

	d->psf_coset = md_calc_size(t->N, t->psf_dims);
	d->linphase_coset = md_calc_size(t->N, t->lph_dims);
	d->psf_size = t->conf.real ? FL_SIZE : CFL_SIZE;
	d->cosets = cosets;
	d->toeplitz_data = t;

	/* One set is all BART is told it has, so its own loop runs once. */
	t->lph_dims[t->N] = 1;

	debug_printf(DP_DEBUG1, "Streaming the function: %d sets of %ld\n", cosets, d->psf_coset);
}

static const struct linop_s* toeplitz_for(int N, const long ksp_dims[N], const long cim_dims[N],
		const long traj_dims[N], const complex float* traj,
		const long wgh_dims[N], const complex float* weights,
		const long bas_dims[N], const complex float* basis, struct nufft_conf_s conf,
		complex float** to_host)
{
	if (!conf.toeplitz) {

		if (0 == making_psf)
#pragma omp atomic
			toeplitz_counters[TP_PAIR]++;

		return NULL;
	}

	struct nufft_conf_s barts = barts_conf(conf);
	barts.nopsf = true;

	/* The oversampling of two is the grid the embedding needs, and BART
	 * builds its machinery for that one alone: anything else sends
	 * `nufft_create2` down its chained path, which has no normal to borrow
	 * and is not even the same data underneath.  The kernel's upsampling is
	 * FINUFFT's and has nothing to do with it, so it does not come here.
	 *
	 * The width does not come here either.  Nothing of BART's kernel is
	 * evaluated -- the normal is a convolution, and `toeplitz_mult` reads
	 * neither the roll-off nor the gridder -- so leaving it at BART's own
	 * keeps the Kaiser-Bessel table, which is one table for the process,
	 * from being asked for a second beta. */
	barts.os = nufft_conf_defaults.os;
	barts.width = nufft_conf_defaults.width;

	/* A subspace function is a Gram matrix at every frequency, so it is
	 * Hermitian and its upper triangle is the whole of it.  Storing that
	 * is exact, and a quarter of a rank-eight problem's peak. */
	if (NULL != basis)
		barts.upper_triag = true;

	/* Streaming needs the normal that walks the sets rather than the one
	 * that convolves them at once, and a linear phase it can be handed one
	 * of: without a precomputed phase the shift is worked out from the set
	 * BART thinks it is on, which is always the first once it is told it
	 * has one. */
	/* Only where the function will actually be streamed: asking for the
	 * low-memory normal where it buys nothing would walk the sets for no
	 * reason. */
	if (stream_psf_enabled && (0 != bartorch_on_device(traj))) {

		barts.lowmem = true;
		barts.precomp_linphase = true;
	}


	const struct linop_s* op = bart_nufft_create2(N, ksp_dims, cim_dims, traj_dims, traj,
			wgh_dims, weights, (NULL != basis) ? bas_dims : NULL, basis, barts);

	struct nufft_data* data = CAST_DOWN(nufft_data, linop_get_data_nested(op));

	making_psf++;
	install_psf(data, traj, to_host);
	making_psf--;

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

	if (N < 4)
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

	/* BART's `-o` and FINUFFT's upsampfac are the same number: how far past
	 * the image the transform is computed on.  BART's own default is two, so
	 * anything else was asked for on purpose and is carried across; two
	 * itself leaves the choice to whatever `enable` was told, because a
	 * quarter over costs a third of the memory for a wider kernel and that
	 * is the cheaper half of the trade here. */
	double upsampling = (0. == conf.os) ? bartorch_finufft_upsampling() : conf.os;

	/* BART's `-w` and FINUFFT's ns are the same count of grid points, and
	 * FINUFFT has no field to be told one: it sizes ns from the tolerance,
	 *
	 *     ns = ceil( ln(tolfac / tol) / (pi sqrt(1 - 1/sigma)) + 1 )
	 *
	 * with tolfac = 0.18 * 1.4^(dim-1) for a type 1 or 2, from FINUFFT's
	 * src/common/kernel.cpp.  Inverting it for the width asked for is what
	 * carries `-w` across.  BART's own default is six, which leaves the
	 * tolerance as the caller set it. */
	if (0.f != conf.width) {

		/* A width is a count of grid points at a given upsampling, so it
		 * says nothing until one is fixed; the textbook factor is what it
		 * is read against. */
		if (0. == upsampling)
			upsampling = 2.;

		eps = fi_tolerance_for(device, dim, conf.width, upsampling);

		/* Past a certain width the tolerance it stands for is below what a
		 * single-precision transform can reach, and FINUFFT refuses one it
		 * cannot honour.  The widest kernel it will use is the answer. */
		if (eps < FLT_EPSILON)
			eps = FLT_EPSILON;

		if (eps >= 1.)
			DECLINE(17);
	}

	long image_elements = 1;

	for (int i = 0; i < dim; i++)
		image_elements *= (long)n_modes[i];

	PTR_ALLOC(struct nufft_fi_s, d);
	SET_TYPEID(nufft_fi_s, d);

	memset(&d->side, 0, sizeof d->side);
	d->toeplitz = NULL;
	d->toeplitz_data = NULL;
	d->psf_host = NULL;
	d->linphase_host = NULL;
	d->cosets = 0;

	d->psf_slot = NULL;
	d->linphase_slot = NULL;

	for (int i = 0; i < 3; i++)
		d->radians[i] = NULL;

	pthread_mutex_init(&d->lock, NULL);
	d->dim = dim;

	for (int i = 0; i < 3; i++) {

		d->axis[i] = axis[i];
		d->n_modes[i] = n_modes[i];
	}

	d->eps = eps;
	d->upsampling = upsampling;
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
	d->trj_dims = xmalloc((size_t)N * sizeof(long));
	d->ksp_dims = xmalloc((size_t)N * sizeof(long));
	md_copy_dims(N, d->trj_dims, traj_dims);
	md_copy_dims(N, d->ksp_dims, ksp_dims);
	d->conf = conf;

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

	if (NULL != wgh_dims) {

		md_copy_dims(N, d->wgh_dims, wgh_dims);
		md_calc_strides(N, d->wgh_strs, wgh_dims, CFL_SIZE);
	}

	if (NULL != weights) {

		d->host_weights = md_alloc(N, d->wgh_dims, CFL_SIZE);
		md_copy(N, d->wgh_dims, d->host_weights, weights, CFL_SIZE);
	}

	/* An operator built against dimensions alone waits for
	 * `nufft_update_traj`; one built over a trajectory plans the side it is
	 * most likely to be asked for first, so that a plan FINUFFT will not
	 * make is a decline here rather than an error in the middle of a solve. */
	if (NULL != traj) {

		install_traj(d, traj_dims, traj);

		int ret = side_build(d, device);

		if (0 != ret) {

			nufft_fi_del(CAST_UP(PTR_PASS(d)));
			DECLINE(ret);
		}
	}

	if (NULL != traj)
		d->toeplitz = toeplitz_for(N, ksp_dims, cim_dims, traj_dims, traj,
				wgh_dims, weights, bas_dims, basis, conf, &d->psf_host);
		stream_psf(d);

	/* PTR_PASS hands the data over and clears the pointer, so what the
	 * operator is built with is read out first. */
	lop_fun_t normal = conf.toeplitz ? nufft_fi_normal : NULL;

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

	/* Nothing reaches BART's own gridder without having been sent there.
	 * The substitution being switched off is a reason like any other: a
	 * caller who never asked for it would otherwise get an answer an order
	 * further from the transform, several times slower, silently. */
	if (!allow_fallback)
		error("bartorch: FINUFFT cannot serve this NUFFT: %s.\n",
		      bartorch_nufft_decline_text());

	count(CNT_BART);

	return bart_nufft_create2(N, ksp_dims, cim_dims, traj_dims, traj, wgh_dims, weights, bas_dims, basis, barts_conf(conf));
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

/* `nlinv`, `moba` and the network models build their NUFFT against dimensions
 * alone and hand the trajectory over here, once per frame of a run.  The
 * plans belong to it, so both sides go and are made again on the next
 * transform, over whatever else arrived with it. */
void nufft_update_traj(const struct linop_s* nufft, int N, const long trj_dims[N], const complex float* traj, const long wgh_dims[N], const complex float* weights, const long bas_dims[N], const complex float* basis)
{
	if (!is_ours(nufft)) {

		bart_nufft_update_traj(nufft, N, trj_dims, traj, wgh_dims, weights, bas_dims, basis);
		return;
	}

	struct nufft_fi_s* d = CAST_DOWN(nufft_fi_s, linop_get_data(nufft));

	if (N != d->N)
		error("bartorch: a trajectory of %d axes for an operator of %d\n", N, d->N);

	if (d->samples != md_calc_size(N - 1, trj_dims + 1))
		error("bartorch: a trajectory of %ld samples for an operator of %ld\n",
				md_calc_size(N - 1, trj_dims + 1), d->samples);

	if ((NULL != weights) && !md_check_equal_dims(N, wgh_dims, d->wgh_dims, ~0UL))
		error("bartorch: weights of a shape the operator was not built for\n");

	if ((NULL != basis) && !md_check_equal_dims(N, bas_dims, d->bas_dims, ~0UL))
		error("bartorch: a basis of a shape the operator was not built for\n");

	pthread_mutex_lock(&d->lock);

	side_free(d, &d->side[0]);
	side_free(d, &d->side[1]);

	install_traj(d, trj_dims, traj);

	md_free(d->host_weights);
	d->host_weights = NULL;

	if (NULL != weights) {

		d->host_weights = md_alloc(N, d->wgh_dims, CFL_SIZE);
		md_copy(N, d->wgh_dims, d->host_weights, weights, CFL_SIZE);
	}

	if (NULL != basis) {

		if (NULL == d->host_basis)
			d->host_basis = md_alloc(N, d->bas_dims, CFL_SIZE);

		md_copy(N, d->bas_dims, d->host_basis, basis, CFL_SIZE);
	}

	pthread_mutex_unlock(&d->lock);

	/* The normal convolves with a point spread function over this
	 * trajectory, so it is made again from the one that just arrived --
	 * `nlinv` and the network models get here once per frame of a run. */
	if (d->conf.toeplitz) {

		if (NULL != d->toeplitz)
			linop_free(d->toeplitz);

		md_free(d->psf_host);
		md_free(d->linphase_host);
		d->psf_host = NULL;
		d->linphase_host = NULL;

		md_free(d->psf_slot);
		md_free(d->linphase_slot);
		d->psf_slot = NULL;
		d->linphase_slot = NULL;

		d->toeplitz = toeplitz_for(N, d->ksp_dims, d->cim_dims, trj_dims, traj,
				(NULL != weights) ? wgh_dims : d->wgh_dims, weights,
				(NULL != basis) ? bas_dims : d->bas_dims, basis, d->conf, &d->psf_host);
		stream_psf(d);
	}
}

const struct operator_s* nufft_precond_create(const struct linop_s* nufft_op)
{
	if (is_ours(nufft_op))
		refuse("building a preconditioner");

	return bart_nufft_precond_create(nufft_op);
}
