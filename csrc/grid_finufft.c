/*
 * BART's gridder, backed by FINUFFT.
 *
 * BART grids in `grid` and interpolates in `gridH`, and its NUFFT is those
 * two either side of an oversampled FFT, with a deapodisation that undoes the
 * gridding kernel.  Replacing the kernel therefore means replacing the
 * deapodisation with it, or the transform is silently wrong; the two are
 * kept together here.
 *
 * `grid.c` is compiled with its five entry points renamed, so the originals
 * are still available under `bart_kb_*` and serve whenever this cannot: a
 * strided array, an oversampling FINUFFT has no kernel for, or no FINUFFT at
 * all.
 *
 * The deapodisation is not derived from FINUFFT's kernel parameters but
 * measured from its spreader: one unit sample is spread onto an otherwise
 * empty line, and the transform of what lands there is what the image has to
 * be divided by.  That stays correct whatever kernel a FINUFFT release picks
 * for a tolerance.
 */
#include <complex.h>
#include <math.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/nested.h"

#include "num/flpmath.h"
#include "num/multind.h"

#include "noncart/grid.h"

#include "include/bartorch.h"
#include "backend.h"

#ifndef CFL_SIZE
#define CFL_SIZE sizeof(complex float)
#endif

/* The originals, renamed where grid.c is compiled. */
extern void bart_kb_grid2(const struct grid_conf_s* conf, int D, const long trj_dims[D], const complex float* traj, const long grid_dims[D], complex float* grid, const long ksp_dims[D], const complex float* src);
extern void bart_kb_grid2H(const struct grid_conf_s* conf, int D, const long trj_dims[D], const complex float* traj, const long ksp_dims[D], complex float* dst, const long grid_dims[D], const complex float* grid);
extern void bart_kb_rolloff_correction(float os, float width, float beta, const long dimensions[3], complex float* dst);
extern void bart_kb_apply_rolloff_correction2(float os, float width, float beta, int N, const long dims[N], const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src);
extern void bart_kb_apply_rolloff_correction(float os, float width, float beta, int N, const long dimensions[N], complex float* dst, const complex float* src);

/* ------------------------------------------------------------------------ */
/* The FINUFFT entry points, filled in by the host from the installed wheel. */
/* ------------------------------------------------------------------------ */

typedef void* finufft_plan_t;

typedef int (*fi_makeplan_t)(int type, int dim, const int64_t* n_modes, int iflag, int ntrans, float eps, finufft_plan_t* plan, void* opts);
typedef int (*fi_setpts_t)(finufft_plan_t plan, int64_t M, float* x, float* y, float* z, int64_t N, float* s, float* t, float* u);
typedef int (*fi_execute_t)(finufft_plan_t plan, complex float* c, complex float* f);
typedef int (*fi_destroy_t)(finufft_plan_t plan);
typedef void (*fi_default_opts_t)(void* opts);

static struct {

	fi_makeplan_t makeplan;
	fi_setpts_t setpts;
	fi_execute_t execute;
	fi_destroy_t destroy;
	fi_default_opts_t default_opts;

	int opts_size;
	int off_modeord;
	int off_spreadinterponly;
	int off_upsampfac;
	int off_nthreads;

	int enabled;
	double tolerance;

	/* The ratio the fine grid was gridded at, which the deapodisation needs
	 * and its own argument does not carry: rolloff_correction's `os` says
	 * whether it was handed image dimensions or fine ones. */
	double grid_os;

} fi = { .enabled = 1, .tolerance = 1.e-6, .grid_os = 2. };

static pthread_mutex_t fi_lock = PTHREAD_MUTEX_INITIALIZER;

/* Which path each call took, so a test can tell a fallback from a failure. */
static long counters[4];

long bartorch_finufft_counter(int which)
{
	return ((which >= 0) && (which < 4)) ? counters[which] : -1;
}

void bartorch_finufft_reset_counters(void)
{
	for (int i = 0; i < 4; i++)
		counters[i] = 0;
}

enum { CNT_GRID_FI, CNT_GRID_KB, CNT_ROLL_FI, CNT_ROLL_KB };

/* Why the last call could not be served, for a test to report. */
static int last_reject;

int bartorch_finufft_last_reject(void)
{
	return last_reject;
}

#define REJECT(code) do { last_reject = (code); return false; } while (0)

static void count(int which)
{
#pragma omp atomic
	counters[which]++;
}

int bartorch_finufft_set(const char* symbol, void* fn)
{
	if (0 == strcmp(symbol, "finufftf_makeplan")) fi.makeplan = (fi_makeplan_t)fn;
	else if (0 == strcmp(symbol, "finufftf_setpts")) fi.setpts = (fi_setpts_t)fn;
	else if (0 == strcmp(symbol, "finufftf_execute")) fi.execute = (fi_execute_t)fn;
	else if (0 == strcmp(symbol, "finufftf_destroy")) fi.destroy = (fi_destroy_t)fn;
	else if (0 == strcmp(symbol, "finufftf_default_opts")) fi.default_opts = (fi_default_opts_t)fn;
	else return -1;

	return 0;
}

int bartorch_finufft_layout(int opts_size, int modeord, int spreadinterponly, int upsampfac, int nthreads)
{
	if ((opts_size < 16) || (opts_size > 4096))
		return -1;

	fi.opts_size = opts_size;
	fi.off_modeord = modeord;
	fi.off_spreadinterponly = spreadinterponly;
	fi.off_upsampfac = upsampfac;
	fi.off_nthreads = nthreads;
	return 0;
}

void bartorch_finufft_enable(int enable)
{
	fi.enabled = (0 != enable);

}

void bartorch_finufft_set_tolerance(double eps)
{
	if ((eps > 0.) && (eps < 1.))
		fi.tolerance = eps;
}

double bartorch_finufft_tolerance(void)
{
	return fi.tolerance;
}

static bool finufft_ready(void)
{
	return fi.enabled && (NULL != fi.makeplan) && (NULL != fi.setpts)
		&& (NULL != fi.execute) && (NULL != fi.destroy)
		&& (NULL != fi.default_opts) && (0 != fi.opts_size);
}

int bartorch_finufft_active(void)
{
	return finufft_ready() ? 1 : 0;
}

/* The ratio FINUFFT designs its kernel for.  This is not BART's `conf->os`,
 * which only says what units the trajectory is in: BART grids either an image
 * onto a grid twice its size with os = 2, or a trajectory already in fine-grid
 * units with os = 1.  Either way the fine grid is twice the image, which is
 * the ratio the kernel and its deapodisation are built at. */
#define BARTORCH_UPSAMPFAC 2.0

static bool supported_os(float os)
{
	return (fabsf(os - 2.0f) < 1.e-3f) || (fabsf(os - 1.0f) < 1.e-3f)
		|| (fabsf(os - 1.25f) < 1.e-3f);
}

static finufft_plan_t make_plan(int type, int dim, const int64_t n_modes[3], int ntrans)
{
	char opts[4096];

	fi.default_opts(opts);
	*(int*)(opts + fi.off_spreadinterponly) = 1;
	*(double*)(opts + fi.off_upsampfac) = BARTORCH_UPSAMPFAC;
	*(int*)(opts + fi.off_nthreads) = 0;

	finufft_plan_t plan = NULL;

	/* The sign is irrelevant to spreading, which is what this plan does. */
	if (0 != fi.makeplan(type, dim, n_modes, 1, ntrans, (float)fi.tolerance, &plan, opts))
		return NULL;

	return plan;
}

/* ------------------------------------------------------------------------ */
/* Deciding whether a call can be served                                     */
/* ------------------------------------------------------------------------ */

struct layout {

	int dim;
	int axis[3];		/* the BART axes that are not singleton */
	int64_t n_modes[3];
	long samples;		/* per slab */
	long channels;
	long batches;		/* everything beyond the first four dimensions */
	long grid_slab;		/* elements in one slab */
	long ksp_slab;
	long trj_slab;		/* zero when one trajectory serves every batch */
};

/* BART's arrays carry sixteen dimensions and the gridder works on the first
 * four, so the rest are slabs to be walked.  Only the layout FINUFFT reads
 * directly is served here; anything else is BART's own gridder's. */
static bool describe(const struct grid_conf_s* conf, int D, const long trj_dims[D],
		const long grid_dims[D], const long ksp_dims[D], struct layout* l)
{
	if (!finufft_ready())
		REJECT(1);

	if (!supported_os(conf->os))
		REJECT(2);

	if (D < 4)
		REJECT(3);

	if (3 != trj_dims[0])
		REJECT(4);

	if (1 != ksp_dims[0])
		REJECT(5);

	if (grid_dims[3] != ksp_dims[3])
		REJECT(6);

	l->dim = 0;

	for (int i = 0; i < 3; i++)
		if (grid_dims[i] > 1)
			l->axis[l->dim++] = i;

	if (0 == l->dim)
		REJECT(7);

	/* FINUFFT's first mode dimension is the one that varies fastest in the
	 * array, which is BART's first: both lay out x fastest. */
	for (int i = 0; i < l->dim; i++)
		l->n_modes[i] = grid_dims[l->axis[i]];

	if (getenv("BARTORCH_FINUFFT_FLIP")) {

		for (int i = 0; i < l->dim / 2; i++) {

			int64_t t = l->n_modes[i];
			l->n_modes[i] = l->n_modes[l->dim - 1 - i];
			l->n_modes[l->dim - 1 - i] = t;

			int a = l->axis[i];
			l->axis[i] = l->axis[l->dim - 1 - i];
			l->axis[l->dim - 1 - i] = a;
		}
	}

	l->samples = ksp_dims[1] * ksp_dims[2];
	l->channels = ksp_dims[3];
	l->grid_slab = md_calc_size(4, grid_dims);
	l->ksp_slab = md_calc_size(4, ksp_dims);

	long trj_slab = md_calc_size(4, trj_dims);

	l->batches = 1;

	bool traj_per_batch = false;

	for (int i = 4; i < D; i++) {

		if (1 == ksp_dims[i])
			continue;

		if (grid_dims[i] != ksp_dims[i])
			REJECT(8);

		if (trj_dims[i] == ksp_dims[i])
			traj_per_batch = true;
		else if (1 != trj_dims[i])
			REJECT(9);

		l->batches *= ksp_dims[i];
	}

	/* A trajectory that varies along one outer dimension but not another
	 * would need striding this does not do. */
	for (int i = 4; i < D; i++)
		if ((trj_dims[i] > 1) && (trj_dims[i] != ksp_dims[i]))
			REJECT(10);

	l->trj_slab = traj_per_batch ? trj_slab : 0;

	fi.grid_os = conf->os;
	last_reject = 0;
	return true;
}

/* BART puts a sample at os * (k + shift) + N/2 on the fine grid; FINUFFT puts
 * one at N/2 + N x / 2pi, so the coordinate is that position in radians. */
static void sample_coordinates(const struct grid_conf_s* conf, const struct layout* l,
		const long grid_dims[], const complex float* traj, float* buffer[3])
{
#pragma omp parallel for
	for (long s = 0; s < l->samples; s++) {

		for (int i = 0; i < l->dim; i++) {

			int a = l->axis[i];
			float k = crealf(traj[3 * s + a]) + conf->shift[a];
			buffer[i][s] = (float)(2. * M_PI * conf->os * k / (double)grid_dims[a]);
		}
	}
}

static long grid_elements(const struct layout* l)
{
	long n = 1;

	for (int i = 0; i < l->dim; i++)
		n *= (long)l->n_modes[i];

	return n;
}

/* One slab through FINUFFT, accumulated the way BART's gridder accumulates. */
static bool transform_slab(const struct grid_conf_s* conf, const struct layout* l, int type,
		const long grid_dims[], const complex float* traj,
		complex float* dst, const complex float* src, long dst_elements)
{
	pthread_mutex_lock(&fi_lock);
	finufft_plan_t plan = make_plan(type, l->dim, l->n_modes, (int)l->channels);
	pthread_mutex_unlock(&fi_lock);

	if (NULL == plan)
		return false;

	float* coord[3] = { NULL, NULL, NULL };

	for (int i = 0; i < l->dim; i++)
		coord[i] = xmalloc((size_t)l->samples * sizeof(float));

	sample_coordinates(conf, l, grid_dims, traj, coord);

	complex float* tmp = xmalloc((size_t)dst_elements * CFL_SIZE);

	int ret = fi.setpts(plan, l->samples, coord[0], coord[1], coord[2], 0, NULL, NULL, NULL);

	if (0 == ret)
		ret = (1 == type) ? fi.execute(plan, (complex float*)src, tmp)
				  : fi.execute(plan, tmp, (complex float*)src);

	pthread_mutex_lock(&fi_lock);
	fi.destroy(plan);
	pthread_mutex_unlock(&fi_lock);

	for (int i = 0; i < 3; i++)
		if (NULL != coord[i])
			xfree(coord[i]);

	if (0 == ret)
		for (long i = 0; i < dst_elements; i++)
			dst[i] += tmp[i];

	xfree(tmp);
	return (0 == ret);
}

void grid2(const struct grid_conf_s* conf, int D, const long trj_dims[D], const complex float* traj, const long grid_dims[D], complex float* grid, const long ksp_dims[D], const complex float* src)
{
	struct layout l;

	if (bartorch_on_device(traj) || !describe(conf, D, trj_dims, grid_dims, ksp_dims, &l)) {

		count(CNT_GRID_KB);
		return bart_kb_grid2(conf, D, trj_dims, traj, grid_dims, grid, ksp_dims, src);
	}

	long elements = grid_elements(&l) * l.channels;

	for (long b = 0; b < l.batches; b++) {

		if (!transform_slab(conf, &l, 1, grid_dims, traj + b * l.trj_slab,
				grid + b * l.grid_slab, src + b * l.ksp_slab, elements)) {

			count(CNT_GRID_KB);
			return bart_kb_grid2(conf, D, trj_dims, traj, grid_dims, grid, ksp_dims, src);
		}
	}

	if (NULL != getenv("BARTORCH_FINUFFT_DEBUG"))
		debug_printf(DP_WARN, "grid2:  os=%f grid=[%ld %ld %ld] ksp=[%ld %ld %ld %ld] batches=%ld\n",
				conf->os, grid_dims[0], grid_dims[1], grid_dims[2],
				ksp_dims[0], ksp_dims[1], ksp_dims[2], ksp_dims[3], l.batches);

	count(CNT_GRID_FI);
}

void grid2H(const struct grid_conf_s* conf, int D, const long trj_dims[D], const complex float* traj, const long ksp_dims[D], complex float* dst, const long grid_dims[D], const complex float* grid)
{
	struct layout l;

	if (bartorch_on_device(traj) || !describe(conf, D, trj_dims, grid_dims, ksp_dims, &l)) {

		count(CNT_GRID_KB);
		return bart_kb_grid2H(conf, D, trj_dims, traj, ksp_dims, dst, grid_dims, grid);
	}

	long elements = l.samples * l.channels;

	for (long b = 0; b < l.batches; b++) {

		if (!transform_slab(conf, &l, 2, grid_dims, traj + b * l.trj_slab,
				dst + b * l.ksp_slab, grid + b * l.grid_slab, elements)) {

			count(CNT_GRID_KB);
			return bart_kb_grid2H(conf, D, trj_dims, traj, ksp_dims, dst, grid_dims, grid);
		}
	}

	if (NULL != getenv("BARTORCH_FINUFFT_DEBUG"))
		debug_printf(DP_WARN, "grid2H: os=%f grid=[%ld %ld %ld] ksp=[%ld %ld %ld %ld] batches=%ld\n",
				conf->os, grid_dims[0], grid_dims[1], grid_dims[2],
				ksp_dims[0], ksp_dims[1], ksp_dims[2], ksp_dims[3], l.batches);

	count(CNT_GRID_FI);
}

/* ------------------------------------------------------------------------ */
/* Deapodisation, measured from the same spreader                            */
/* ------------------------------------------------------------------------ */

/* One unit sample spread onto an empty line of the fine grid, transformed:
 * the image has to be divided by this for the pair to be a transform. */
static bool kernel_transform(long image_size, float os, double* out)
{
	/* `os` says how the dimensions given relate to the fine grid, so their
	 * product is the fine grid whichever convention the caller used. */
	int64_t fine = (int64_t)lround((double)os * (double)image_size);

	pthread_mutex_lock(&fi_lock);
	finufft_plan_t plan = make_plan(1, 1, &fine, 1);
	pthread_mutex_unlock(&fi_lock);

	if (NULL == plan)
		return false;

	float x = 0.f;
	complex float one = 1.f;
	complex float* line = xmalloc((size_t)fine * CFL_SIZE);

	int ret = fi.setpts(plan, 1, &x, NULL, NULL, 0, NULL, NULL, NULL);

	if (0 == ret)
		ret = fi.execute(plan, &one, line);

	pthread_mutex_lock(&fi_lock);
	fi.destroy(plan);
	pthread_mutex_unlock(&fi_lock);

	if (0 != ret) {

		xfree(line);
		return false;
	}

	/* The image occupies the middle of the fine grid, so image index i sits
	 * at frequency (i - image_size/2) / fine. */
	for (long i = 0; i < image_size; i++) {

		double f = (double)(i - image_size / 2) / (double)fine;
		double acc = 0.;

		for (int64_t n = 0; n < fine; n++)
			acc += crealf(line[n]) * cos(2. * M_PI * f * (double)(n - fine / 2));

		out[i] = acc;
	}

	xfree(line);
	return true;
}

void rolloff_correction(float os, float width, float beta, const long dimensions[3], complex float* dst)
{
	if (!finufft_ready()) {

		count(CNT_ROLL_KB);
		return bart_kb_rolloff_correction(os, width, beta, dimensions, dst);
	}

	count(CNT_ROLL_FI);

	if (NULL != getenv("BARTORCH_FINUFFT_DEBUG"))
		debug_printf(DP_WARN, "rolloff: os=%f width=%f dims=[%ld %ld %ld]\n",
				os, width, dimensions[0], dimensions[1], dimensions[2]);

	double* axis[3] = { NULL, NULL, NULL };

	for (int i = 0; i < 3; i++) {

		if (1 == dimensions[i])
			continue;

		axis[i] = xmalloc((size_t)dimensions[i] * sizeof(double));

		if (!kernel_transform(dimensions[i], os, axis[i])) {

			for (int j = 0; j <= i; j++)
				if (NULL != axis[j])
					xfree(axis[j]);

			return bart_kb_rolloff_correction(os, width, beta, dimensions, dst);
		}
	}

#pragma omp parallel for collapse(3)
	for (long z = 0; z < dimensions[2]; z++)
		for (long y = 0; y < dimensions[1]; y++)
			for (long x = 0; x < dimensions[0]; x++) {

				double w = 1.;

				if (NULL != axis[0]) w *= axis[0][x];
				if (NULL != axis[1]) w *= axis[1][y];
				if (NULL != axis[2]) w *= axis[2][z];

				dst[x + dimensions[0] * (y + z * dimensions[1])] = (complex float)(1. / w);
			}

	for (int i = 0; i < 3; i++)
		if (NULL != axis[i])
			xfree(axis[i]);
}

void apply_rolloff_correction2(float os, float width, float beta, int N, const long dims[N], const long ostrs[N], complex float* dst, const long istrs[N], const complex float* src)
{
	if (!finufft_ready())
		return bart_kb_apply_rolloff_correction2(os, width, beta, N, dims, ostrs, dst, istrs, src);

	long rdims[N];
	md_select_dims(N, 7UL, rdims, dims);

	complex float* roll = md_alloc_sameplace(N, rdims, CFL_SIZE, dst);

	rolloff_correction(os, width, beta, rdims, roll);

	long rstrs[N];
	md_calc_strides(N, rstrs, rdims, CFL_SIZE);

	md_zmul2(N, dims, ostrs, dst, istrs, src, rstrs, roll);

	md_free(roll);
}

void apply_rolloff_correction(float os, float width, float beta, int N, const long dimensions[N], complex float* dst, const complex float* src)
{
	long strs[N];
	md_calc_strides(N, strs, dimensions, CFL_SIZE);

	apply_rolloff_correction2(os, width, beta, N, dimensions, strs, dst, strs, src);
}
