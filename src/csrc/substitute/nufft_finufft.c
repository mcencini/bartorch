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
#include "linops/someops.h"

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

/* csrc/fft_callbacks.cu: a volume's transform pair with the passes around it
 * run inside it. */
struct bartorch_cb_fft;

/* csrc/paired.cu: a coil against a pair of sets, in cuFFTDx kernels. */
struct bartorch_paired;

#ifdef USE_CUDA
#include "noncart/gpu_grid.h"

/* csrc/kernels.cu: the set's phase and the coil's sensitivity in one pass. */
extern void bartorch_cuda_phase_map_in(int N, const long dims[], const float shift[3], float scale,
		complex float* dst, const complex float* src, const complex float* map);
extern void bartorch_cuda_phase_map_out(int N, const long dims[], const float shift[3], float scale,
		complex float* dst, const complex float* src, const complex float* map);
extern void bartorch_cuda_gather(long V, const unsigned int* mask, const int* prefix, complex float* dst, const complex float* src);
extern int bartorch_cuda_contract_upper_real(long L, int R, complex float* bank, const float* mat);
extern int bartorch_cuda_contract_upper_real_bf16(long L, int R, complex float* bank, const void* mat);
extern void bartorch_cuda_scatter(long V, const unsigned int* mask, const int* prefix, complex float* dst, const complex float* src);

extern struct bartorch_cb_fft* bartorch_cb_fft_create(const long dims[3]);
extern void bartorch_cb_fft_free(struct bartorch_cb_fft* p);
extern void bartorch_cb_fft_forward(struct bartorch_cb_fft* p, int N, const long dims[],
		const float shift[3], float scale, const unsigned int* mask, const int* prefix,
		complex float* bank, complex float* volume, const complex float* src, const complex float* map);
extern void bartorch_cb_fft_inverse(struct bartorch_cb_fft* p, int N, const long dims[],
		const float shift[3], float scale, const unsigned int* mask, const int* prefix,
		complex float* dst, complex float* volume, const complex float* bank, const complex float* map);
#endif

#ifdef BARTORCH_PAIRED
extern struct bartorch_paired* bartorch_paired_create(const long dims[3], int coeffs, int sets, const float (*shifts)[3]);
extern void bartorch_paired_free(struct bartorch_paired* p);
extern void bartorch_paired_in(const struct bartorch_paired* p, int k,
		const complex float* src, const complex float* map, complex float* scratch);
extern void bartorch_paired_fused(const struct bartorch_paired* p, int k, complex float* scratch,
		const void* psf0, const void* psf1, int bf16, const unsigned int* mask, const int* prefix, long L);
extern void bartorch_paired_back(const struct bartorch_paired* p, int k,
		complex float* dst, const complex float* map, complex float* scratch);
#endif

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
		bool periodic, bool lowmem, bool upper_triag,
		const long com_dims[N + 1], const long* idx,
		const long com_psf_dims[N + 1], const long com_psf_dims3[N + 1], int real);

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
 * solve ran on a point spread function rather than on the transform pair;
 * the functions among them that were compressed; the sets convolved with
 * the passes run inside cuFFT's transforms; the functions stored real; the
 * pairs of sets convolved together by the pair kernels; and the functions
 * kept in bfloat16. */
enum { TP_PSF, TP_PAIR, TP_COMPRESSED, TP_CALLBACKS, TP_REAL, TP_PAIRED, TP_BF16, TP_COUNTERS };
static long toeplitz_counters[TP_COUNTERS];

/* Building a point spread function needs a transform of its own, and that
 * transform is nobody's normal: it is asked for one adjoint and freed.  While
 * one is being made, an operator built underneath does not count as having
 * answered a normal either way. */
static _Thread_local int making_psf;

long bartorch_toeplitz_counter(int which)
{
	return ((0 <= which) && (which < TP_COUNTERS)) ? toeplitz_counters[which] : -1;
}

void bartorch_toeplitz_reset_counters(void)
{
	for (int i = 0; i < TP_COUNTERS; i++)
		toeplitz_counters[i] = 0;
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
	void* psf_slot[2];		/* where a set lands */
	void* stage;			/* the stream a set crosses on */
	int slot;			/* the one BART is pointed at */
	int slot_set[2];		/* the set each slot holds or is receiving */
	int slot_pending;		/* a slot the card has not yet been held for, or -1 */
	int coset;			/* the set in it */
	bool psf_registered;		/* the host copy is page-locked */
	long psf_coset;			/* elements in one set of the function */
	size_t psf_size;		/* a real function is stored as floats */
	int cosets;
	int unit;			/* sets in a slot: one, or the two of a pair */
	int units;			/* units in the function */
	int set;			/* the set the per-set convolution is at */

	/* The places the samples reach, on the card, as coset.cuh reads them:
	 * a bit per grid point and a count per word of bits. */
	unsigned int* kept_mask;
	int* kept_prefix;

	/* The transform of one volume, which is what a compressed function
	 * works a coefficient at a time against.  BART's own is over every
	 * coil and coefficient at once. */
	struct linop_s* vol_fft;

	/* The same transform with the passes around it run inside it, made the
	 * first time a set is convolved; NULL where cuFFT cannot link them in,
	 * and then `vol_fft` and the passes serve. */
	struct bartorch_cb_fft* cb_fft;
	bool cb_tried;

	/* The pair kernels, where the grid, the rank and the function allow them:
	 * then a slot holds the two sets that differ only along x, and a coil is
	 * convolved against both at once. */
	struct bartorch_paired* paired;
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

/* The same plans, pointed at a trajectory that has changed.
 *
 * Making a plan is the largest allocation this operator does -- more than the
 * function a normal ends up convolving with -- and a trajectory that arrives
 * later has the same number of samples on the same grid, which is what a plan
 * is made for.  So the points are replaced and the plan is kept, which is what
 * `setpts` is for. */
static int side_retarget(struct nufft_fi_s* d, int which)
{
	struct fi_side* s = &d->side[which];

	if (NULL == s->forward_plan)
		return 0;

	long one[1] = { d->samples };

	for (int i = 0; i < d->dim; i++)
		md_copy(1, one, s->coord[i], d->radians[i], FL_SIZE);

	if ((NULL != s->weights) && (NULL != d->host_weights))
		md_copy(d->N, d->wgh_dims, s->weights, d->host_weights, CFL_SIZE);

	if ((NULL != s->basis) && (NULL != d->host_basis))
		md_copy(d->N, d->bas_dims, s->basis, d->host_basis, CFL_SIZE);

	if (   (0 != bartorch_finufft_setpts(s->forward_plan, d->samples, s->coord[0], s->coord[1], s->coord[2]))
	    || (0 != bartorch_finufft_setpts(s->adjoint_plan, d->samples, s->coord[0], s->coord[1], s->coord[2]))) {

		side_free(d, s);
		return 13;
	}

	return 0;
}

/* The side `ptr` is on, built if it has not been or has been let go.
 *
 * What this returns stays good after the lock is dropped unless a normal of
 * the same operator runs meanwhile and lets it go.  error() leaves by a
 * longjmp, which is why it is called with the lock released. */
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

/* Whether only the places the samples reach are kept. */
static int compress_psf_enabled = 1;

/* Whether a set crosses while the one before it is convolved. */
static int overlap_psf_enabled = 0;

void bartorch_nufft_set_overlap_psf(int enable)
{
	overlap_psf_enabled = (0 != enable);
}

int bartorch_nufft_overlap_psf(void)
{
	return overlap_psf_enabled;
}

/* Whether a real, upper-triangular contraction runs in the kernel of our own
 * or in BART's -- the second is there to be held against. */
static int contraction_kernel = 1;

void bartorch_nufft_set_contraction_kernel(int enable)
{
	contraction_kernel = (0 != enable);
}

/* Whether the device's transform pair is let go at the first normal. */
static int release_transforms_enabled = 1;

void bartorch_nufft_set_release_transforms(int enable)
{
	release_transforms_enabled = (0 != enable);
}

int bartorch_nufft_release_transforms(void)
{
	return release_transforms_enabled;
}

/* Let the device's transform pair go.
 *
 * With a Toeplitz function built, a normal is a convolution: it reads neither
 * the plans nor the points they were set on.  A solve applies the adjoint once
 * to form its right-hand side and then only normals, so the pair would stay
 * resident for every iteration with nothing reading it.  The host keeps the
 * trajectory the pair was built from, so a transform asked for afterwards
 * plans again.  Called with the lock held. */
/* Let the host copy of the function go.  A crossing may still be reading it --
 * the next application's first set is started as the last one ends -- so the
 * stream it crosses on is closed, which waits for it, before anything is
 * freed. */
static void psf_host_free(struct nufft_fi_s* d)
{
	if (NULL == d->psf_host)
		return;

	bartorch_cuda_stage_close(d->stage);
	d->stage = NULL;

	if (d->psf_registered)
		bartorch_cuda_host_unregister(d->psf_host);

	d->psf_registered = false;

	bartorch_host_free(d->psf_host);
	d->psf_host = NULL;
}

static void release_device_side(struct nufft_fi_s* d)
{
	if (!release_transforms_enabled || (NULL == d->toeplitz) || (NULL == d->side[1].forward_plan))
		return;

	side_free(d, &d->side[1]);

	/* cuFINUFFT allocates from the stream-ordered pool, which keeps what is
	 * freed until the device next synchronises: without this the plans are
	 * gone and their memory is not, and the peak stays where it was. */
#ifdef USE_CUDA
	cuda_sync_device();
#endif
}

void bartorch_nufft_set_compress_psf(int enable)
{
	compress_psf_enabled = (0 != enable);
}

int bartorch_nufft_compress_psf(void)
{
	return compress_psf_enabled;
}

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
	psf_dims[t->N] = d->unit;

	/* A set crosses on a stream of its own, ordered against BART's by
	 * events, so it can run behind the convolution: into the one slot as
	 * soon as the set before it has been read for the last time, or -- with
	 * the overlap -- into a second slot while the set before it is
	 * convolved.  Where there is no such stream it is an ordinary copy. */
	if (0 != bartorch_cuda_stage_open(&d->stage))
		d->stage = NULL;

	int slots = ((NULL != d->stage) && overlap_psf_enabled) ? 2 : 1;

	for (int i = 0; i < slots; i++)
		d->psf_slot[i] = md_alloc_sameplace(ND, psf_dims, d->psf_size, ref);

	d->slot = 0;
	d->slot_set[0] = -1;
	d->slot_set[1] = -1;
}

static void use_coset(struct nufft_fi_s* d, int s);

/* Start a unit's crossing into a slot: its sets lie side by side. */
static void issue_coset(struct nufft_fi_s* d, int i, int slot)
{
	struct nufft_data* t = d->toeplitz_data;
	int ND = t->N + 1;

	long psf_bytes = d->psf_coset * (long)d->psf_size * d->unit;

	const char* psf_src = (const char*)d->psf_host + (size_t)i * (size_t)psf_bytes;

	if (NULL != d->stage) {

		bartorch_cuda_stage_copy(d->stage, slot, d->psf_slot[slot], psf_src, psf_bytes);
		return;
	}

	long psf_dims[ND];
	md_copy_dims(ND, psf_dims, t->psf_dims);
	psf_dims[t->N] = d->unit;

	md_copy(ND, psf_dims, d->psf_slot[slot], psf_src, d->psf_size);
}

/* Bring one set of frequencies over, into the slot given. */
static void fetch_coset(struct nufft_fi_s* d, int i)
{
	d->coset = i;
	d->set = i * d->unit;

	if (NULL == d->stage) {

		issue_coset(d, i, 0);
		d->slot = 0;
		use_coset(d, 0);
		return;
	}

	int slots = (NULL != d->psf_slot[1]) ? 2 : 1;
	int cur = i % slots;

	/* A set nobody has started is started now. */
	if (d->slot_set[cur] != i) {

		issue_coset(d, i, cur);
		d->slot_set[cur] = i;
	}

	/* With two slots the next set starts as this one does, into the slot
	 * the set before this one released.  With one it starts once this
	 * one's function has been read for the last time: `slot_read`.  A slot
	 * is only overwritten once the card has said it is done with it. */
	if ((2 == slots) && (i + 1 < d->units)) {

		issue_coset(d, i + 1, cur ^ 1);
		d->slot_set[cur ^ 1] = i + 1;
	}

	/* The card is held for it where it is first read (`slot_ready`). */
	d->slot_pending = cur;

	d->slot = cur;

	use_coset(d, 0);
}

/* Point BART at set `s` of the unit in the slot. */
static void use_coset(struct nufft_fi_s* d, int s)
{
	struct nufft_data* t = d->toeplitz_data;
	int ND = t->N + 1;

	long psf_dims[ND];
	md_copy_dims(ND, psf_dims, t->psf_dims);
	psf_dims[t->N] = 1;

	if (NULL != t->psf)
		multiplace_free(t->psf);

	char* slot = (char*)d->psf_slot[d->slot] + (size_t)s * (size_t)d->psf_coset * d->psf_size;

	t->psf = multiplace_move_wrapper(ND, psf_dims, d->psf_size, slot);
}

/* The shift of the set in the slot: half a cell back along each axis the
 * set's index has a bit for, which is how the function's sets were made. */
static void coset_shift(const struct nufft_fi_s* d, int set, float shift[3])
{
	const struct nufft_data* t = d->toeplitz_data;

	long factors[3];

	for (int i = 0; i < 3; i++)
		factors[i] = ((t->img_dims[i] > 1) && MD_IS_SET(t->flags, i)) ? 2 : 1;

	bartorch_psf_shift(3, shift, 3, factors, set);
}

/* The set's linear phase, put on as the image is read and, conjugated and
 * accumulated, taken off as it is written -- computed where it is applied
 * rather than read from a volume of it.  It is BART's own kernel, over the
 * shift, centring and scale its precomputed phases carry, so what it applies
 * is what they would have. */
static void apply_phase(const struct nufft_data* t, const long dims[], const float shift[3],
		complex float* dst, const complex float* src, bool out)
{
	float scale = 1.f / sqrtf((float)md_calc_size(3, t->img_dims));

#ifdef USE_CUDA
	cuda_apply_linphases_3D(t->N, dims, shift, dst, src, out, out, true, scale);
#else
	(void)dims; (void)dst; (void)src; (void)out; (void)scale;
	error("bartorch: a streamed set is convolved on a card\n");
#endif
}

/* The same, with the coil's sensitivity on as well: one pass over the volume
 * rather than one for the phase and one for the map.  The map is one coil's,
 * laid out as the volume is. */
static void apply_phase_map(const struct nufft_data* t, const float shift[3],
		complex float* dst, const complex float* src, const complex float* map, bool out)
{
	if (NULL == map) {

		apply_phase(t, t->img_dims, shift, dst, src, out);
		return;
	}

	float scale = 1.f / sqrtf((float)md_calc_size(3, t->img_dims));

#ifdef USE_CUDA
	if (out)
		bartorch_cuda_phase_map_out(t->N, t->img_dims, shift, scale, dst, src, map);
	else
		bartorch_cuda_phase_map_in(t->N, t->img_dims, shift, scale, dst, src, map);
#else
	(void)dst; (void)src; (void)scale;
	error("bartorch: a streamed set is convolved on a card\n");
#endif
}

/* How many gathered locations are contracted at once.
 *
 * The coefficients at a location meet only each other, so the bank need not be
 * contracted whole -- and whole, the answer stands beside the question, each
 * of them the gathered spectrum's size.  A chunk is contracted into a buffer
 * that holds a chunk and copied back over itself, which leaves a chunk beside
 * the bank instead of a second bank.  Large enough that a kernel over it has
 * work: a quarter of a million locations is a million entries at rank four.
 */
enum { CONTRACT_CHUNK = 1 << 18 };

static void contract_bank(struct nufft_data* t, const void* psf,
		const long bank_dims[], const long ciT_dims[], complex float* bank)
{
	int N = t->N;

#ifdef USE_CUDA
	/* In place and in one pass, where the function is real and kept as its
	 * upper triangle: what follows is the general contraction, over a chunk
	 * buffer that has to be cleared and copied back. */
	if (contraction_kernel && t->conf.real && t->conf.upper_triag && cuda_ondevice(bank)) {

		long L = bank_dims[0];
		int R = (int)(md_calc_size(N, bank_dims) / L);

		if ((md_calc_size(N, ciT_dims) == md_calc_size(N, bank_dims))
		    && (0 == bartorch_cuda_contract_upper_real(L, R, bank, psf)))
			return;
	}
#endif

	long locations = bank_dims[0];
	long chunk = MIN(locations, (long)CONTRACT_CHUNK);

	long bank_strs[N];
	long ciT_strs[N];

	md_calc_strides(N, bank_strs, bank_dims, CFL_SIZE);
	md_calc_strides(N, ciT_strs, ciT_dims, CFL_SIZE);

	long in_dims[N];
	long out_dims[N];

	md_copy_dims(N, in_dims, bank_dims);
	md_copy_dims(N, out_dims, ciT_dims);

	in_dims[0] = chunk;
	out_dims[0] = chunk;

	complex float* out = md_alloc_sameplace(N, out_dims, CFL_SIZE, bank);

	for (long start = 0; start < locations; start += chunk) {

		long here = MIN(chunk, locations - start);

		in_dims[0] = here;
		out_dims[0] = here;

		long out_strs[N];
		md_calc_strides(N, out_strs, out_dims, CFL_SIZE);

		long max_dims[N];
		md_max_dims(N, ~0UL, max_dims, out_dims, in_dims);

		complex float* in = bank + start;
		const void* mat = (const char*)psf + (size_t)start * (size_t)t->psf_strs[0];

		if (t->conf.real) {

			if (t->conf.upper_triag)
				md_tenmul_upper_triag2(6, 7, N + 1, MD_REAL_DIMS(N, max_dims),
						MD_REAL_STRS(N, out_strs, FL_SIZE), (float*)out,
						MD_REAL_STRS(N, bank_strs, FL_SIZE), (const float*)in,
						MD_REAL_DIMS(N, t->psf_dims),
						MD_REAL_STRS(N, t->psf_strs, 0), (const float*)mat);
			else
				md_tenmul2(N + 1, MD_REAL_DIMS(N, max_dims),
						MD_REAL_STRS(N, out_strs, FL_SIZE), (float*)out,
						MD_REAL_STRS(N, bank_strs, FL_SIZE), (const float*)in,
						MD_REAL_STRS(N, t->psf_strs, 0), (const float*)mat);

		} else {

			if (t->conf.upper_triag)
				md_ztenmul_upper_triag2(5, 6, N, max_dims,
						out_strs, out, bank_strs, in,
						t->psf_dims, t->psf_strs, mat);
			else
				md_ztenmul2(N, max_dims, out_strs, out, bank_strs, in,
						t->psf_strs, mat);
		}

		/* Back over the question it answered. */
		md_copy2(N, out_dims, ciT_strs, bank + start, out_strs, out, CFL_SIZE);
	}

	md_free(out);
}

/* The function against the spectrum, wherever the spectrum lies.
 *
 * Without a basis the function is a diagonal and multiplies in place; with one
 * it is a matrix at every frequency and the contraction needs somewhere to
 * land, so what comes back may not be what went in. */
static complex float* multiply_transfer(struct nufft_data* t, const void* psf,
		const long cim_dims[], const long ciT_dims[], complex float* grid)
{
	int N = t->N;

	if (md_check_equal_dims(N, cim_dims, ciT_dims, ~0UL)) {

		long cim_strs[N];
		md_calc_strides(N, cim_strs, cim_dims, CFL_SIZE);

		if (t->conf.real)
			md_mul2(N, MD_REAL_DIMS(N, cim_dims),
					MD_REAL_STRS(N, cim_strs, FL_SIZE), (float*)grid,
					MD_REAL_STRS(N, cim_strs, FL_SIZE), (float*)grid,
					MD_REAL_STRS(N, t->psf_strs, 0), (const float*)psf);
		else
			md_zmul2(N, cim_dims, cim_strs, grid, cim_strs, grid, t->psf_strs, psf);

		return grid;
	}

	long max_dims[N];
	md_max_dims(N, ~0UL, max_dims, ciT_dims, cim_dims);

	long ciT_strs[N];
	md_calc_strides(N, ciT_strs, ciT_dims, CFL_SIZE);

	long cim_strs[N];
	md_calc_strides(N, cim_strs, cim_dims, CFL_SIZE);

	complex float* out = md_alloc_sameplace(N, ciT_dims, CFL_SIZE, grid);

	if (t->conf.real) {

		if (t->conf.upper_triag)
			md_tenmul_upper_triag2(6, 7, N + 1, MD_REAL_DIMS(N, max_dims),
					MD_REAL_STRS(N, ciT_strs, FL_SIZE), (float*)out,
					MD_REAL_STRS(N, cim_strs, FL_SIZE), (float*)grid,
					MD_REAL_DIMS(N, t->psf_dims),
					MD_REAL_STRS(N, t->psf_strs, 0), (const float*)psf);
		else
			md_tenmul2(N + 1, MD_REAL_DIMS(N, max_dims),
					MD_REAL_STRS(N, ciT_strs, FL_SIZE), (float*)out,
					MD_REAL_STRS(N, cim_strs, FL_SIZE), (float*)grid,
					MD_REAL_STRS(N, t->psf_strs, 0), (const float*)psf);

	} else {

		if (t->conf.upper_triag)
			md_ztenmul_upper_triag(5, 6, N, ciT_dims, out, cim_dims, grid, t->psf_dims, psf);
		else
			md_ztenmul(N, ciT_dims, out, cim_dims, grid, t->psf_dims, psf);
	}

	md_free(grid);

	return out;
}

/* The set's function has been read for the last time.
 *
 * With one slot this is when the next set can start to cross: the card still
 * has this coil's scatter, inverse transforms and accumulation ahead of it,
 * none of which reads the function, and the crossing runs behind them.  After
 * the last set the first is started again, for the next application. */
static void slot_read(struct nufft_fi_s* d, bool last)
{
	if (!last || (NULL == d->stage) || (NULL != d->psf_slot[1]))
		return;

	bartorch_cuda_stage_release(d->stage, 0);

	int next = (d->coset + 1) % d->units;

	issue_coset(d, next, 0);
	d->slot_set[0] = next;
}

/* Whether sets are convolved in pairs where the pair kernels allow it.  Read
 * when a function is streamed, so it applies to operators built afterwards. */
static int paired_enabled = 1;

void bartorch_nufft_set_paired(int enable)
{
	paired_enabled = (0 != enable);
}

int bartorch_nufft_paired(void)
{
	return paired_enabled;
}

/* Whether a function whose sets are paired is kept in bfloat16: half the host
 * copy, half of what crosses and of the slot it lands in, at a rounding of
 * 2^-9 of each value where floats keep 2^-24.  bfloat16 keeps the exponent
 * of a float, so no value is clipped.  Read when a function is streamed. */
static int bf16_enabled = 1;

void bartorch_nufft_set_bf16(int enable)
{
	bf16_enabled = (0 != enable);
}

int bartorch_nufft_bf16(void)
{
	return bf16_enabled;
}

/* A float as bfloat16, rounded to nearest even. */
static uint16_t to_bf16(float f)
{
	uint32_t u;
	memcpy(&u, &f, sizeof u);

	if ((u & 0x7fffffffu) > 0x7f800000u)
		return (uint16_t)((u >> 16) | 0x40u);

	u += 0x7fffu + ((u >> 16) & 1u);

	return (uint16_t)(u >> 16);
}

/* Whether this library has pair kernels at all. */
int bartorch_nufft_paired_built(void)
{
#ifdef BARTORCH_PAIRED
	return 1;
#else
	return 0;
#endif
}

/* Whether the passes around a volume's transform run inside it, as cuFFT
 * callbacks, where cuFFT can link them in. */
static int fft_callbacks_enabled = 1;

void bartorch_nufft_set_fft_callbacks(int enable)
{
	fft_callbacks_enabled = (0 != enable);
}

int bartorch_nufft_fft_callbacks(void)
{
	return fft_callbacks_enabled;
}

/* The operator's transform pair with the passes inside it, where it can have
 * one: every axis of the volume longer than one is among the first three,
 * and transformed. */
static struct bartorch_cb_fft* callbacks_for(struct nufft_fi_s* d)
{
	if (!fft_callbacks_enabled)
		return NULL;

	if (!d->cb_tried) {

		d->cb_tried = true;

#ifdef USE_CUDA
		const struct nufft_data* t = d->toeplitz_data;
		unsigned long flags = t->flags | t->conf.cfft;
		bool covered = true;

		for (int i = 0; i < t->N; i++)
			if (1 < t->img_dims[i])
				covered = covered && (i < 3) && MD_IS_SET(flags, i);

		if (covered)
			d->cb_fft = bartorch_cb_fft_create(t->img_dims);
#endif
	}

	return d->cb_fft;
}

/* Hold the card until the set in the slot has arrived.
 *
 * As late as it can be: a set's first transforms do not read the function,
 * so they run while its last part is still crossing, and only the first
 * multiplication by it waits.  At 256^3 a set takes 30 ms to cross and the
 * work that can hide it after the last read of the set before is 22 ms of
 * inverse transforms; the forward transforms of the next set's first coil
 * are what cover the rest.  The wait goes on the stream of whoever
 * multiplies. */
static void slot_ready(struct nufft_fi_s* d)
{
	if ((NULL == d->stage) || (-1 == d->slot_pending))
		return;

	bartorch_cuda_stage_wait(d->stage, d->slot_pending);
	d->slot_pending = -1;
}

/* Down to the places the samples reach, and back up with zeros elsewhere. */
static void gather(const struct nufft_fi_s* d, long grid, complex float* dst, const complex float* src)
{
#ifdef USE_CUDA
	bartorch_cuda_gather(grid, d->kept_mask, d->kept_prefix, dst, src);
#else
	(void)d; (void)grid; (void)dst; (void)src;
	error("bartorch: a streamed set is convolved on a card\n");
#endif
}

static void scatter(const struct nufft_fi_s* d, long grid, complex float* dst, const complex float* src)
{
#ifdef USE_CUDA
	bartorch_cuda_scatter(grid, d->kept_mask, d->kept_prefix, dst, src);
#else
	(void)d; (void)grid; (void)dst; (void)src;
	error("bartorch: a streamed set is convolved on a card\n");
#endif
}

/* A coefficient into the gathered spectrum: the phase and the sensitivity on,
 * the transform, the gather. */
static void coefficient_in(struct nufft_fi_s* d, struct bartorch_cb_fft* cb, const float shift[3],
		complex float* bank, complex float* volume, const complex float* src, const complex float* map)
{
	struct nufft_data* t = d->toeplitz_data;
	int N = t->N;
	long grid = md_calc_size(3, t->img_dims);

#ifdef USE_CUDA
	if (NULL != cb) {

		bartorch_cb_fft_forward(cb, N, t->img_dims, shift, 1.f / sqrtf((float)grid),
				d->kept_mask, d->kept_prefix, bank, volume, src, map);
		return;
	}
#else
	(void)cb;
#endif

	apply_phase_map(t, shift, volume, src, map, false);

	linop_forward(d->vol_fft, N, t->img_dims, volume, N, t->img_dims, volume);

	gather(d, grid, bank, volume);
}

/* And out of it: the scatter, the inverse, the conjugates of both, and the
 * sum into `dst`. */
static void coefficient_out(struct nufft_fi_s* d, struct bartorch_cb_fft* cb, const float shift[3],
		complex float* dst, complex float* volume, const complex float* bank, const complex float* map)
{
	struct nufft_data* t = d->toeplitz_data;
	int N = t->N;
	long grid = md_calc_size(3, t->img_dims);

#ifdef USE_CUDA
	if (NULL != cb) {

		bartorch_cb_fft_inverse(cb, N, t->img_dims, shift, 1.f / sqrtf((float)grid),
				d->kept_mask, d->kept_prefix, dst, volume, bank, map);
		return;
	}
#else
	(void)cb;
#endif

	scatter(d, grid, volume, bank);

	linop_adjoint(d->vol_fft, N, t->img_dims, volume, N, t->img_dims, volume);

	apply_phase_map(t, shift, dst, volume, map, true);
}

/* The set convolved with `src`, added to `dst`, against a compressed function.
 *
 * A compressed function has values only where the samples reach, so what
 * multiplies it is the spectrum gathered down to those places -- and gathering
 * is what makes it worth keeping the spectrum nowhere else.  One volume is
 * transformed and gathered at a time, so what is resident is one volume and
 * the gathered coefficients of one coil, rather than every coil and
 * coefficient at full size.  Coils are independent until the sum that ends
 * them, so they are taken one at a time; coefficients are not, because the
 * function contracts them, so a coil's are gathered before any is multiplied.
 *
 * Around each transform are the passes that put the phase and the sensitivity
 * on and gather, and that scatter and take them off; where cuFFT can run them
 * inside the transforms, they run there.
 */
static void packed_coset(struct nufft_fi_s* d, complex float* dst, const complex float* src,
		const void* psf, const long map_strs[], const complex float* map, bool last)
{
	struct nufft_data* t = d->toeplitz_data;
	int N = t->N;

	float shift[3];
	coset_shift(d, d->set, shift);

	struct bartorch_cb_fft* cb = callbacks_for(d);

	if ((NULL == cb) && (NULL == d->vol_fft))
		d->vol_fft = linop_fft_create(N, t->img_dims, t->flags | t->conf.cfft);

	/* The spectrum of one coil, gathered: the places the samples reach on
	 * the first axis, the coefficients on the axis the function contracts. */
	long bank_dims[N];
	long ciT_bank_dims[N];
	long one_bank_dims[N];

	md_select_dims(N, ~MD_BIT(3), bank_dims, t->cim_dims);
	md_select_dims(N, ~MD_BIT(3), ciT_bank_dims, t->ciT_dims);

	md_copy_dims(3, bank_dims, t->psf_dims);
	md_copy_dims(3, ciT_bank_dims, t->psf_dims);

	md_select_dims(N, MD_BIT(0), one_bank_dims, bank_dims);

	long locations = bank_dims[0];
	long coils = t->cim_dims[3];
	long coeffs = md_calc_size(N, bank_dims) / locations;
	long out_coeffs = md_calc_size(N, ciT_bank_dims) / locations;

	/* Where a coil's coefficient sits in what is read and written.  Counted
	 * here rather than read off `cim_strs`, which carries a zero for every
	 * axis of one and so cannot be walked.
	 *
	 * With the sensitivity folded in, what is read and written is the image
	 * rather than a coil image: it has no coil axis, every coil accumulates
	 * into the same one, and the map is what tells them apart. */
	long vol = md_calc_size(3, t->cim_dims);

	long coil_step = (NULL == map) ? vol : 0;
	long rest_step = (NULL == map) ? vol * t->cim_dims[3] : vol;

	long map_coil_step = (NULL == map) ? 0 : map_strs[3] / (long)CFL_SIZE;

	complex float* volume = md_alloc_sameplace(N, t->img_dims, CFL_SIZE, dst);

	for (long c = 0; c < coils; c++) {

		const complex float* m = (NULL == map) ? NULL : map + c * map_coil_step;

		complex float* bank = md_alloc_sameplace(N, bank_dims, CFL_SIZE, dst);

		for (long r = 0; r < coeffs; r++)
			coefficient_in(d, cb, shift, bank + r * locations, volume, src + c * coil_step + r * rest_step, m);

		slot_ready(d);

#ifdef USE_CUDA
		if (2 == d->psf_size)
			bartorch_cuda_contract_upper_real_bf16(locations, (int)coeffs, bank, psf);
		else
#endif
		if (md_check_equal_dims(N, bank_dims, ciT_bank_dims, ~0UL))
			bank = multiply_transfer(t, psf, bank_dims, ciT_bank_dims, bank);
		else
			contract_bank(t, psf, bank_dims, ciT_bank_dims, bank);

		slot_read(d, last && (c == coils - 1));

		for (long r = 0; r < out_coeffs; r++)
			coefficient_out(d, cb, shift, dst + c * coil_step + r * rest_step, volume, bank + r * locations, m);

		md_free(bank);
	}

	md_free(volume);

	if (NULL != cb)
		toeplitz_counters[TP_CALLBACKS]++;
}

/* The two sets in the slot, convolved with `src` and added to `dst`.
 *
 * With the kernels a coil goes against both sets at once: its passes along z
 * and y serve both, and the pass along x multiplies by each set's function in
 * turn.  The kernels read a coil's coefficients a volume apart, which a folded
 * sensitivity or a single coil gives; anything else goes a set at a time. */
static void paired_unit(struct nufft_fi_s* d, complex float* dst, const complex float* src,
		const long map_strs[], const complex float* map, bool last)
{
	struct nufft_data* t = d->toeplitz_data;

	long coils = t->cim_dims[3];

	if ((NULL == map) && (1 != coils)) {

		for (int s = 0; s < 2; s++) {

			d->set = 2 * d->coset + s;
			use_coset(d, s);

			const void* psf = multiplace_read(t->psf, src);

			packed_coset(d, dst, src, psf, map_strs, map, last && (1 == s));
		}

		return;
	}

#ifdef BARTORCH_PAIRED
	long vol = md_calc_size(3, t->cim_dims);
	long coeffs = md_calc_size(t->N, t->cim_dims) / md_calc_size(4, t->cim_dims);
	long map_coil_step = (NULL == map) ? 0 : map_strs[3] / (long)CFL_SIZE;

	long sdims[1] = { coeffs * vol };
	complex float* scratch = md_alloc_sameplace(1, sdims, CFL_SIZE, dst);

	const char* psf0 = d->psf_slot[d->slot];
	const char* psf1 = psf0 + (size_t)d->psf_coset * d->psf_size;

	for (long c = 0; c < coils; c++) {

		const complex float* m = (NULL == map) ? NULL : map + c * map_coil_step;

		bartorch_paired_in(d->paired, d->coset, src, m, scratch);

		slot_ready(d);

		bartorch_paired_fused(d->paired, d->coset, scratch, psf0, psf1, 2 == d->psf_size,
				d->kept_mask, d->kept_prefix, t->psf_dims[0]);

		/* The pair has been read for the last time once the last coil is past
		 * its pass along x: the next pair can cross while this one's passes
		 * back run. */
		slot_read(d, last && (c == coils - 1));

		bartorch_paired_back(d->paired, d->coset, dst, m, scratch);
	}

	md_free(scratch);

	toeplitz_counters[TP_PAIRED]++;
#else
	(void)dst; (void)src; (void)map_strs; (void)map; (void)last;
	error("bartorch: sets in pairs without the pair kernels\n");
#endif
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
static bool fused_coset(struct nufft_fi_s* d, complex float* dst, const complex float* src,
		const long map_strs[], const complex float* map, bool last)
{
	if (2 == d->unit) {

		paired_unit(d, dst, src, map_strs, map, last);
		return true;
	}

	struct nufft_data* t = d->toeplitz_data;

	const void* psf = multiplace_read(t->psf, src);

	if (NULL == psf)
		return false;

	if (NULL != d->kept_mask) {

		packed_coset(d, dst, src, psf, map_strs, map, last);
		return true;
	}

	/* Only the gathered arrangement folds a sensitivity in; the rest is the
	 * coil image the caller made, so there is nothing to fold. */
	if (NULL != map)
		return false;

	float shift[3];
	coset_shift(d, d->set, shift);

	complex float* grid = md_alloc_sameplace(t->N, t->cim_dims, CFL_SIZE, dst);

	apply_phase(t, t->cim_dims, shift, grid, src, false);

	linop_forward(t->cfft_op, t->N, t->cim_dims, grid, t->N, t->cim_dims, grid);

	slot_ready(d);

	grid = multiply_transfer(t, psf, t->cim_dims, t->ciT_dims, grid);

	slot_read(d, last);

	linop_adjoint(t->cfft_op, t->N, t->cim_dims, grid, t->N, t->cim_dims, grid);

	/* Into the answer, not over it: this is what the second image was for. */
	apply_phase(t, t->cim_dims, shift, dst, grid, true);

	md_free(grid);

	return true;
}

/* Driving the sets from outside.
 *
 * The function crosses once for each set that is used, so what decides the
 * traffic is how often a set is used: with the coils outside and the sets
 * inside, every coil brings the whole function over again.  These let the
 * caller put the sets outside instead -- one set, then every coil against it
 * -- so the function crosses once for an application rather than once for
 * each coil.  On eight coils that is eight times less over the bus.
 *
 * The operator holds one slot, so the sets have to be walked one at a time and
 * nothing else may swap what it points at meanwhile: `begin` takes the lock
 * and `end` gives it back.
 */
int bartorch_nufft_cosets(const struct linop_s* op)
{
	if (!is_ours(op))
		return 0;

	struct nufft_fi_s* d = CAST_DOWN(nufft_fi_s, linop_get_data(op));

	return (NULL == d->psf_host) ? 0 : d->units;
}

static void coset_begin(struct nufft_fi_s* d, const void* ref)
{
	pthread_mutex_lock(&d->lock);

	release_device_side(d);

	if (NULL == d->psf_slot[0])
		open_slots(d, ref);
}

/* The set that is loaded, convolved with `src` and added to `dst`. */
static void coset_normal(struct nufft_fi_s* d, complex float* dst, const complex float* src,
		const long map_strs[], const complex float* map, bool last)
{
	if (!fused_coset(d, dst, src, map_strs, map, last))
		error("bartorch: a streamed set with no function in its slot\n");

	if (NULL != d->stage)
		bartorch_cuda_stage_release(d->stage, d->slot);
}

void bartorch_nufft_coset_begin(const struct linop_s* op, const void* ref)
{
	coset_begin(CAST_DOWN(nufft_fi_s, linop_get_data(op)), ref);
}

void bartorch_nufft_coset_use(const struct linop_s* op, int i)
{
	fetch_coset(CAST_DOWN(nufft_fi_s, linop_get_data(op)), i);
}

void bartorch_nufft_coset_normal(const struct linop_s* op, complex float* dst, const complex float* src, int last)
{
	coset_normal(CAST_DOWN(nufft_fi_s, linop_get_data(op)), dst, src, NULL, NULL, (0 != last));
}

/* Whether a set can be convolved with the sensitivity folded in.
 *
 * Only the gathered arrangement can: it reads a coefficient of one coil at a
 * time and writes one at a time, so the map goes on as a coefficient is read
 * and comes off as it is written.  The arrangements that work over a whole
 * coil image at once have nowhere to put it. */
int bartorch_nufft_coset_folds(const struct linop_s* op)
{
	struct nufft_fi_s* d = CAST_DOWN(nufft_fi_s, linop_get_data(op));

	if ((NULL == d->psf_host) || (NULL == d->toeplitz_data))
		return 0;

	return (NULL != d->kept_mask) ? 1 : 0;
}

/* The set convolved with a coil's image, added to the caller's image, with the
 * sensitivity applied on the way in and taken off on the way out.
 *
 * Otherwise the caller holds two coil images -- one to multiply the map into
 * and one for the answer to land in -- and at 256^3 over four coefficients
 * each of those is half a gigabyte.  Folded in here neither is made. */
void bartorch_nufft_coset_normal_sense(const struct linop_s* op,
		complex float* dst, const complex float* src,
		const long map_strs[], const complex float* map, int last)
{
	coset_normal(CAST_DOWN(nufft_fi_s, linop_get_data(op)), dst, src, map_strs, map, (0 != last));
}

void bartorch_nufft_coset_end(const struct linop_s* op)
{
	pthread_mutex_unlock(&CAST_DOWN(nufft_fi_s, linop_get_data(op))->lock);
}

static void nufft_fi_normal(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	struct nufft_fi_s* d = CAST_DOWN(nufft_fi_s, _d);

	if (NULL == d->toeplitz)
		error("bartorch: the normal was asked for before the trajectory arrived\n");

	if (NULL == d->psf_host) {

		pthread_mutex_lock(&d->lock);
		release_device_side(d);
		pthread_mutex_unlock(&d->lock);

		linop_normal_unchecked(d->toeplitz, dst, src);
		return;
	}

	/* Nobody put the sets outside, so they are walked here, each read by a
	 * single convolution. */
	struct nufft_data* t = d->toeplitz_data;

	coset_begin(d, dst);

	md_clear(t->N, t->cim_dims, dst, CFL_SIZE);

	for (int i = 0; i < d->units; i++) {

		fetch_coset(d, i);
		coset_normal(d, dst, src, NULL, NULL, true);
	}

	pthread_mutex_unlock(&d->lock);
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
	psf_host_free(d);

	bartorch_cuda_stage_close(d->stage);

	for (int i = 0; i < 2; i++)
		md_free(d->psf_slot[i]);

	md_free(d->kept_mask);
	md_free(d->kept_prefix);

	if (NULL != d->vol_fft)
		linop_free(d->vol_fft);

#ifdef USE_CUDA
	bartorch_cb_fft_free(d->cb_fft);
#endif
#ifdef BARTORCH_PAIRED
	bartorch_paired_free(d->paired);
#endif

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
	case 14: return "the basis lies along something other than coefficients, frames, shots and samples";
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


	int device = bartorch_on_device(traj) ? 1 : 0;

	int transformed = 0;

	for (int i = 0; i < 3; i++)
		if (MD_IS_SET(data->flags, i) && (1 < data->img_dims[i]))
			transformed++;

	if (0 == transformed)
		return -1;

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

	int spread = fi_measure_width(device, transformed, bartorch_finufft_tolerance(), upsampling);

	if (spread < 0)
		return -1;

	double width = ceil((double)spread / upsampling);

	if (width < 2.)
		width = 2.;

	/* FINUFFT spreads the mask on the grid itself, and refuses an axis shorter
	 * than twice the kernel.  Along such an axis the mask is kept whole: the
	 * kernel reaches most or all of it anyway, and a mask that keeps more
	 * than it needs costs compression and never accuracy. */
	int dim = 0;
	int axis[3];
	int64_t n_modes[3];
	unsigned long whole = 0UL;

	for (int i = 0; i < 3; i++) {

		if (!MD_IS_SET(data->flags, i) || (1 == data->img_dims[i]))
			continue;

		if ((double)data->img_dims[i] < 2. * width) {

			whole = MD_SET(whole, i);
			continue;
		}

		axis[dim] = i;
		n_modes[dim] = data->img_dims[i];
		dim++;
	}

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
	/* Without a pattern every sample counts. */
	if (NULL == pattern)
		md_zfill(ND, one_dims, samples_in, 1.);
	else
		md_copy2(ND, one_dims, one_strs, samples_in, wgh_strs, pattern, CFL_SIZE);

	long spread_dims[ND];
	md_select_dims(ND, ~whole, spread_dims, data->com_dims);

	complex float* reach = alloc_on(device, ND, spread_dims, CFL_SIZE);
	complex float* mask = alloc_on(device, ND, spread_dims, CFL_SIZE);
	md_clear(ND, spread_dims, mask, CFL_SIZE);

	float* coord[3] = { NULL, NULL, NULL };

	for (int i = 0; i < dim; i++)
		coord[i] = alloc_on(device, ND, one_dims, FL_SIZE);

	double mask_eps = (0 == dim) ? 0. : fi_tolerance_for(device, dim, width, upsampling);

	debug_printf(DP_DEBUG2, "PSF mask spread at width %g, tolerance %g\n", width, mask_eps);

	int ret = 0;

	/* Kept whole along every axis: every point is reached. */
	if (0 == dim)
		md_zfill(ND, spread_dims, mask, 1.);

	for (long set = 0; (0 < dim) && (0 == ret) && (set < sets); set++) {

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

		md_clear(ND, spread_dims, reach, CFL_SIZE);

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

		md_zabs(ND, spread_dims, reach, reach);
		md_zmax(ND, spread_dims, mask, mask, reach);
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
	md_copy2(ND, data->com_dims, MD_STRIDES(ND, data->com_dims, CFL_SIZE), mask_cpu,
			MD_STRIDES(ND, spread_dims, CFL_SIZE), mask, CFL_SIZE);
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

	/* What the turn leaves is held against single precision: a basis computed
	 * in floats -- an SVD of a dictionary, say -- keeps imaginary parts at that
	 * level when it is real, and dropping them changes the function by about
	 * as little. */
	const double noise = 100. * FLT_EPSILON;

	bool real = (0. == energy) || (left <= noise * noise * energy);

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

	if (store_real)
		toeplitz_counters[TP_REAL]++;

	/* Streamed, the function is made an entry at a time and kept on the
	 * host: neither it nor any entry but the one being made is ever
	 * resident. */
	bool stream = (NULL != to_host) && stream_psf_enabled
		&& (0 != bartorch_on_device(traj));

	/* The places the samples reach.  Worked out before the function is
	 * built, because an entry is compressed as it is made. */
	long max_idx = 0;

	if (data->conf.compress_psf) {

		md_select_dims(ND, FFT_FLAGS, data->com_dims, data->img_dims);

		if (0 != spread_mask(data, traj, &max_idx))
			error("bartorch: FINUFFT would not spread the pattern for a compressed function\n");

		/* Kept only where it is worth keeping.
		 *
		 * What compression gives back is the part of the function the
		 * samples never reached: a fraction 1 - f of every entry the
		 * function holds at a frequency, which is one for a scalar
		 * function and the triangle's worth for a subspace one.  What
		 * it costs is an index over the grid, one long a point, whether
		 * it gives back anything or not.  So it earns its place when
		 *
		 *	(1 - f) * entries * element > sizeof(long)
		 *
		 * A scalar function never clears it -- one real volume a set is
		 * four bytes a point against the index's eight -- and a
		 * subspace one clears it easily: ten entries at rank four need
		 * only a fifth of the grid to go unreached.  A three-
		 * dimensional radial trajectory leaves the corners of the cube
		 * outside its ball, which is a fraction 1 - pi/6 of it before
		 * the spreading kernel's width is added back. */
		long grid = md_calc_size(ND, data->com_dims);
		long entries = md_calc_size(N, data->psf_dims) / md_calc_size(3, data->psf_dims);
		size_t element = store_real ? FL_SIZE : CFL_SIZE;

		double dropped = 1. - (double)max_idx / (double)grid;

		if (dropped * (double)entries * (double)element <= (double)sizeof(long)) {

			debug_printf(DP_DEBUG1, "Not compressing: %.0f%% of the grid is reached\n",
					100. * max_idx / grid);

			multiplace_free(data->compress);
			data->compress = NULL;
			data->conf.compress_psf = false;
			max_idx = 0;
		}
	}

	if (NULL != data->compress)
#pragma omp atomic
		toeplitz_counters[TP_COMPRESSED]++;

	long com_psf_dims[ND];
	long com_psf_dims3[ND];
	const long* idx = NULL;

	if (stream && (NULL != data->compress)) {

		md_compress_dims(ND, com_psf_dims, data->psf_dims, data->com_dims, max_idx);
		md_select_dims(ND, ~MD_BIT(N), com_psf_dims3, com_psf_dims);
		idx = multiplace_read(data->compress, traj);
	}

	complex float* psf = stream
		? bartorch_psf_to_host(N, data->psf_dims, data->flags, data->trj_dims, traj,
				data->bas_dims, basis, data->wgh_dims, weights,
				true, data->conf.lowmem, data->conf.upper_triag,
				data->com_dims, idx, com_psf_dims, com_psf_dims3, store_real ? 1 : 0)
		: (data->conf.decomposed_psf ? compute_psf2_decomposed : compute_psf2)(N,
				data->psf_dims, data->flags, data->trj_dims, traj,
				data->bas_dims, basis, data->wgh_dims, weights,
				true /* as nufft.c asks for it */, data->conf.lowmem, data->conf.upper_triag);

	multiplace_free(data->psf);

	/* Built compressed, the function came back compressed, and that is the
	 * shape everything after this works in. */
	bool packed = (NULL != idx);

	if (packed) {

		md_copy_dims(ND, data->psf_dims, com_psf_dims);
		md_calc_strides(ND, data->psf_strs, data->psf_dims, CFL_SIZE);
	}

	if (store_real) {

		/* A streamed function was made real as it was built, an entry at
		 * a time; a resident one is made real here, whole. */
		if (!stream) {

			float* psf_real = md_alloc_sameplace(ND, data->psf_dims, FL_SIZE, psf);
			md_real(ND, data->psf_dims, psf_real, psf);
			md_free(psf);
			psf = (complex float*)psf_real;
		}

		md_calc_strides(ND, data->psf_strs, data->psf_dims, FL_SIZE);

		/* What is stored is what the multiply has to read. */
		data->conf.real = true;
	}

	/* Only the places the samples reach are kept.  A resident function is
	 * reduced here, once it is whole; a streamed one was reduced an entry
	 * at a time as it was built, so that neither it nor its compressed copy
	 * is ever whole at once. */
	if (!packed && (NULL != data->compress)) {

		size_t size = store_real ? FL_SIZE : CFL_SIZE;

		long com_psf_dims[ND];
		md_compress_dims(ND, com_psf_dims, data->psf_dims, data->com_dims, max_idx);

		complex float* com_psf = md_alloc_sameplace(ND, com_psf_dims, size, traj);

		md_compress(ND, com_psf_dims, com_psf, data->psf_dims, psf,
				data->com_dims, multiplace_read(data->compress, com_psf), size);

		md_free(psf);
		psf = com_psf;

		md_copy_dims(ND, data->psf_dims, com_psf_dims);
		md_calc_strides(ND, data->psf_strs, data->psf_dims, size);
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

	int cosets = (int)t->lph_dims[t->N];

	if ((cosets < 2) || (t->psf_dims[t->N] != cosets)) {

		psf_host_free(d);
		d->psf_host = NULL;
		return;
	}

	d->psf_coset = md_calc_size(t->N, t->psf_dims);
	d->psf_size = t->conf.real ? FL_SIZE : CFL_SIZE;
	d->cosets = cosets;
	d->toeplitz_data = t;

	/* A compressed function is gathered against through a mask of the
	 * places the samples reach, made once from the map they were found
	 * with; the map, one long per grid point, is let go. */
	if (NULL != t->compress) {

		int ND = t->N + 1;

		long grid = md_calc_size(ND, t->com_dims);
		const long* map = multiplace_read(t->compress, d->radians[0]);

		long wdims[1] = { (grid + 31) / 32 };

		unsigned int* mask = xmalloc((size_t)wdims[0] * sizeof(unsigned int));
		int* prefix = xmalloc((size_t)wdims[0] * sizeof(int));

		long n = 0;

		for (long w = 0; w < wdims[0]; w++) {

			unsigned int bits = 0;

			for (long b = 0; (b < 32) && (w * 32 + b < grid); b++)
				if (0 <= map[w * 32 + b])
					bits |= 1u << b;

			mask[w] = bits;
			prefix[w] = (int)n;
			n += __builtin_popcount(bits);
		}

#ifdef USE_CUDA
		d->kept_mask = md_alloc_gpu(1, wdims, sizeof(unsigned int));
		d->kept_prefix = md_alloc_gpu(1, wdims, sizeof(int));
#else
		d->kept_mask = md_alloc(1, wdims, sizeof(unsigned int));
		d->kept_prefix = md_alloc(1, wdims, sizeof(int));
#endif
		md_copy(1, wdims, d->kept_mask, mask, sizeof(unsigned int));
		md_copy(1, wdims, d->kept_prefix, prefix, sizeof(int));

		xfree(mask);
		xfree(prefix);

		multiplace_free(t->compress);
		t->compress = NULL;
	}

	/* In pairs, where the kernels allow it: a compressed, real function kept
	 * as its upper triangle, over eight sets.  The kernels decide the rest --
	 * the grid and the number of coefficients. */
	d->unit = 1;

#ifdef BARTORCH_PAIRED
	if (paired_enabled && (NULL != d->kept_mask) && t->conf.real && t->conf.upper_triag && (8 == cosets)) {

		if (NULL == d->paired) {

			float shifts[8][3];

			for (int i = 0; i < 8; i++)
				coset_shift(d, i, shifts[i]);

			long coeffs = md_calc_size(t->N, t->cim_dims) / md_calc_size(4, t->cim_dims);

			d->paired = bartorch_paired_create(t->img_dims, (int)coeffs, 8, (const float (*)[3])shifts);
		}

		if (NULL != d->paired)
			d->unit = 2;
	}
#endif

	d->units = cosets / d->unit;

	/* Paired, the function is kept in bfloat16, converted in place and the
	 * copy shrunk to what it holds: each value lands on the first half of
	 * where it was read, and the values still to be read lie beyond it.  A
	 * set convolved on its own inside a pair reads it through the
	 * contraction kernel, so that kernel is a condition. */
	if ((2 == d->unit) && bf16_enabled && contraction_kernel) {

		long n = (long)cosets * d->psf_coset;
		const float* in = (const float*)d->psf_host;
		uint16_t* out = (uint16_t*)d->psf_host;

		for (long i = 0; i < n; i++)
			out[i] = to_bf16(in[i]);

		void* shrunk = realloc(d->psf_host, (size_t)n * sizeof(uint16_t));

		if (NULL != shrunk)
			d->psf_host = shrunk;

		d->psf_size = sizeof(uint16_t);
		toeplitz_counters[TP_BF16]++;
	}

	/* Page-locked, so a set's crossing runs behind the convolution rather
	 * than holding up the host that issued it. */
	d->psf_registered = (0 == bartorch_cuda_host_register(d->psf_host,
				(long)cosets * d->psf_coset * (long)d->psf_size));

	/* One set is all BART is told it has, so its own loop runs once. */
	t->lph_dims[t->N] = 1;

	debug_printf(DP_DEBUG1, "Streaming the function: %d sets of %ld, %d to a slot\n", cosets, d->psf_coset, d->unit);
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

	/* The doubled grid a set of frequencies decomposes is what a
	 * three-dimensional function cannot afford: computed whole it is 2^d
	 * images at once for every entry of the matrix, and computed a set at a
	 * time it is never allocated at all.  The sets are the same either way,
	 * and they are what the function is brought over in. */
	barts.decomposed_psf = true;

	/* Keeping only where the samples reach: the function loses the part
	 * that lies outside them, and what crosses the bus for every set loses
	 * it too.  Whether that pays is decided when the function is built.
	 * The places are a map of the grid the function is over, so a volume
	 * batched along an axis between that grid and the coefficients -- a
	 * stack's z, laid out where the sets would be -- is not read through one. */
	bool transformed = true;

	for (int i = 3; i < 6; i++)
		if ((1 < cim_dims[i]) && (COIL_DIM != i))
			transformed = false;

	if (compress_psf_enabled && transformed)
		barts.compress_psf = true;

	/* Streaming needs the normal that walks the sets rather than the one
	 * that convolves them at once, and only where the function will
	 * actually be streamed: asking for it where it buys nothing would walk
	 * the sets for no reason.  The sets' phases are not precomputed --
	 * that is a volume of phase per set, made on the host -- because the
	 * streamed convolution computes each where it applies it. */
	/* Whether a card is in use is not where the trajectory lies: an
	 * operator built for a card from inputs on the host streams just the
	 * same, from a copy of the trajectory made there for the build. */
	bool on_card = (0 != bartorch_on_device(traj)) || (0 <= bartorch_cuda_device());

	if (stream_psf_enabled && on_card) {

		barts.lowmem = true;
		barts.precomp_linphase = false;
	}

	/* BART's own operator, whose normal is borrowed, takes a basis along the
	 * frames and the coefficients alone.  A basis along the shots and the
	 * samples -- the segments of a time-segmented off-resonance -- is handed
	 * to it as one sample's worth along the coefficients, which is all its
	 * normal and the layout of its function read of a basis, and put back in
	 * its place before the function is built from it. */
	bool along_samples = (NULL != basis) && ((1 != bas_dims[1]) || (1 != bas_dims[2]));

	long stand_dims[N];
	complex float* stand = NULL;

	if (along_samples) {

		md_select_dims(N, ~(MD_BIT(1) | MD_BIT(2)), stand_dims, bas_dims);

		long pos[N];
		md_set_dims(N, pos, 0);

		complex float* whole = md_alloc(N, bas_dims, CFL_SIZE);
		md_copy(N, bas_dims, whole, basis, CFL_SIZE);

		stand = md_alloc(N, stand_dims, CFL_SIZE);
		md_copy_block(N, pos, stand_dims, stand, bas_dims, whole, CFL_SIZE);

		md_free(whole);
	}

	const struct linop_s* op = bart_nufft_create2(N, ksp_dims, cim_dims, traj_dims, traj,
			wgh_dims, weights, (NULL == basis) ? NULL : (along_samples ? stand_dims : bas_dims),
			along_samples ? stand : basis, barts);

	struct nufft_data* data = CAST_DOWN(nufft_data, linop_get_data_nested(op));

	if (along_samples) {

		md_free(stand);

		md_copy_dims(N, data->bas_dims, bas_dims);
		data->bas_dims[N] = 1;
		md_calc_strides(N + 1, data->bas_strs, data->bas_dims, CFL_SIZE);

		multiplace_free(data->basis);
		data->basis = multiplace_move(N + 1, data->bas_dims, CFL_SIZE, basis);
	}

	making_psf++;
	/* The function is built where the arithmetic will be.  BART's own
	 * operator keeps its copy of the trajectory where the caller's is,
	 * which for a caller on the host means the card never holds it. */
	const complex float* traj_on = traj;
	complex float* traj_copy = NULL;

#ifdef USE_CUDA
	if (on_card && (0 == bartorch_on_device(traj))) {

		traj_copy = md_alloc_gpu(N, traj_dims, CFL_SIZE);
		md_copy(N, traj_dims, traj_copy, traj, CFL_SIZE);
		traj_on = traj_copy;
	}
#endif

	install_psf(data, traj_on, to_host);

	if (NULL != traj_copy)
		md_free(traj_copy);

	/* Building the function allocates a transform of its own, the pattern
	 * spread for the mask and the trajectory shifted for a set, and BART's
	 * cache keeps every block of it once it is freed.  Handed back here,
	 * what a solve holds is what it uses. */
#ifdef USE_CUDA
	if (on_card)
		bartorch_cuda_memcache_clear_all();
#endif

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

		/* A basis lies along the coefficients and along any of what a
		 * sample is indexed by -- its frame, as a subspace has it, or its
		 * shot and its place along the readout, as the segments of a
		 * time-segmented off-resonance have them -- and along nothing
		 * else.  The contraction and the point spread function read it by
		 * its strides either way. */
		if ((1 != bas_dims[0]) || (1 != bas_dims[3]) || (1 != bas_dims[4]))
			DECLINE(14);

		for (int i = 7; i < N; i++)
			if (1 != bas_dims[i])
				DECLINE(14);

		if (((1 != bas_dims[1]) && (bas_dims[1] != ksp_dims[1]))
				|| ((1 != bas_dims[2]) && (bas_dims[2] != ksp_dims[2])))
			DECLINE(14);

		if (cim_dims[6] != bas_dims[6])
			DECLINE(14);

		if ((1 != ksp_dims[6]) && (ksp_dims[6] != bas_dims[6]))
			DECLINE(14);

		grd_dims[6] = bas_dims[6];

		if (1 != bas_dims[5])
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
	d->cosets = 0;
	d->vol_fft = NULL;

	d->stage = NULL;
	d->slot = 0;
	d->slot_set[0] = -1;
	d->slot_set[1] = -1;
	d->slot_pending = -1;
	d->coset = 0;
	d->psf_registered = false;
	d->kept_mask = NULL;
	d->kept_prefix = NULL;
	d->cb_fft = NULL;
	d->cb_tried = false;
	d->paired = NULL;
	d->unit = 1;
	d->units = 0;
	d->set = 0;

	for (int i = 0; i < 2; i++)
		d->psf_slot[i] = NULL;

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

	/* The plans are kept and pointed at the new samples. */
	side_retarget(d, 0);
	side_retarget(d, 1);

	pthread_mutex_unlock(&d->lock);

	/* The normal convolves with a point spread function over this
	 * trajectory, so it is made again from the one that just arrived --
	 * `nlinv` and the network models get here once per frame of a run. */
	if (d->conf.toeplitz) {

		if (NULL != d->toeplitz)
			linop_free(d->toeplitz);

		psf_host_free(d);
		d->psf_host = NULL;

		bartorch_cuda_stage_close(d->stage);
		d->stage = NULL;
		d->slot = 0;
		d->slot_pending = -1;

		md_free(d->kept_mask);
		md_free(d->kept_prefix);

		d->kept_mask = NULL;
		d->kept_prefix = NULL;

		for (int i = 0; i < 2; i++) {

			md_free(d->psf_slot[i]);
			d->psf_slot[i] = NULL;
		}

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
