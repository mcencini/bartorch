/*
 * The FINUFFT library, as BART reaches it.
 *
 * The `finufft` wheel carries a compiled shared library with a plain C plan
 * API.  The host hands the entry points and the byte layout of the options
 * struct across the ABI, both read from that same package, so a release that
 * moves a field cannot be misread here and nothing is built or vendored.
 *
 * What is done with a plan belongs to nufft_finufft.c, which builds BART's
 * NUFFT operator out of a pair of them.
 */
#include <complex.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "include/bartorch.h"

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
	int off_nthreads;

	int use_in_tools;
	double tolerance;

} fi = { .use_in_tools = 0, .tolerance = 1.e-6 };

static pthread_mutex_t fi_lock = PTHREAD_MUTEX_INITIALIZER;

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

int bartorch_finufft_layout(int opts_size, int nthreads)
{
	if ((opts_size < 16) || (opts_size > 4096) || (nthreads < 0) || (nthreads + 4 > opts_size))
		return -1;

	fi.opts_size = opts_size;
	fi.off_nthreads = nthreads;
	return 0;
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

/* The entry points are there and the options layout is known. */
static bool symbols_ready(void)
{
	return (NULL != fi.makeplan) && (NULL != fi.setpts) && (NULL != fi.execute)
		&& (NULL != fi.destroy) && (NULL != fi.default_opts) && (0 != fi.opts_size);
}

int bartorch_finufft_usable(void)
{
	return (symbols_ready() && fi.use_in_tools) ? 1 : 0;
}

void bartorch_finufft_use_in_tools(int enable)
{
	fi.use_in_tools = (0 != enable);
}

int bartorch_finufft_plan(int type, int dim, const int64_t n_modes[3], int ntrans,
		int isign, double eps, void** plan)
{
	if (!symbols_ready())
		return -1;

	char opts[4096];

	pthread_mutex_lock(&fi_lock);

	fi.default_opts(opts);
	*(int*)(opts + fi.off_nthreads) = 0;

	finufft_plan_t p = NULL;
	int ret = fi.makeplan(type, dim, n_modes, isign, ntrans, (float)eps, &p, opts);

	pthread_mutex_unlock(&fi_lock);

	if (0 != ret)
		return -1;

	*plan = p;
	return 0;
}

/* A plan keeps the points by pointer rather than copying them, so the arrays
 * given here have to outlive it. */
int bartorch_finufft_setpts(void* plan, long M, float* x, float* y, float* z)
{
	return fi.setpts(plan, M, x, y, z, 0, NULL, NULL, NULL);
}

int bartorch_finufft_exec(void* plan, complex float* c, complex float* f)
{
	return fi.execute(plan, c, f);
}

void bartorch_finufft_free(void* plan)
{
	if (NULL == plan)
		return;

	pthread_mutex_lock(&fi_lock);
	fi.destroy(plan);
	pthread_mutex_unlock(&fi_lock);
}
