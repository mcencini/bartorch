/*
 * The FINUFFT libraries, as BART reaches them.
 *
 * The `finufft` and `cufinufft` wheels each carry a compiled shared library
 * with a plain C plan API.  The host hands the entry points and the byte
 * layout of the options struct across the ABI, both read from those same
 * packages, so a release that moves a field cannot be misread here and
 * nothing is built or vendored.
 *
 * The two libraries have the same shape and different types -- cuFINUFFT
 * takes the sample count as an int and carries a device number in its options
 * -- so they are held as two tables and picked by where the data is.
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

/* cuFINUFFT counts samples in an int and answers a device plan. */
typedef int (*cufi_setpts_t)(finufft_plan_t plan, int M, float* x, float* y, float* z, int N, float* s, float* t, float* u);

struct fi_table {

	fi_makeplan_t makeplan;
	fi_setpts_t setpts;
	cufi_setpts_t setpts_int;
	fi_execute_t execute;
	fi_destroy_t destroy;
	fi_default_opts_t default_opts;

	int opts_size;
	int off_device;		/* nthreads on the host, gpu_device_id on a device */
};

static struct {

	struct fi_table host;
	struct fi_table device;

	int use_in_tools;
	double tolerance;

} fi = { .use_in_tools = 0, .tolerance = 1.e-6 };

static pthread_mutex_t fi_lock = PTHREAD_MUTEX_INITIALIZER;

/* "finufftf_makeplan" fills the host table, "cufinufftf_makeplan" the device
 * one, and the two carry the same five entry points. */
int bartorch_finufft_set(const char* symbol, void* fn)
{
	bool cuda = (0 == strncmp(symbol, "cu", 2));
	struct fi_table* t = cuda ? &fi.device : &fi.host;
	const char* name = symbol + (cuda ? 2 : 0);

	if (0 == strcmp(name, "finufftf_makeplan")) t->makeplan = (fi_makeplan_t)fn;
	else if (0 == strcmp(name, "finufftf_setpts")) { t->setpts = cuda ? NULL : (fi_setpts_t)fn; t->setpts_int = cuda ? (cufi_setpts_t)fn : NULL; }
	else if (0 == strcmp(name, "finufftf_execute")) t->execute = (fi_execute_t)fn;
	else if (0 == strcmp(name, "finufftf_destroy")) t->destroy = (fi_destroy_t)fn;
	else if ((0 == strcmp(name, "finufftf_default_opts")) || (0 == strcmp(name, "finufft_default_opts"))) t->default_opts = (fi_default_opts_t)fn;
	else return -1;

	return 0;
}

/* `device_field` is the byte offset of the one field this sets: the thread
 * count on the host, the device number on a card. */
int bartorch_finufft_layout(int device, int opts_size, int device_field)
{
	struct fi_table* t = device ? &fi.device : &fi.host;

	if ((opts_size < 16) || (opts_size > 4096) || (device_field < 0) || (device_field + 4 > opts_size))
		return -1;

	t->opts_size = opts_size;
	t->off_device = device_field;
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
static bool table_ready(const struct fi_table* t)
{
	return (NULL != t->makeplan) && ((NULL != t->setpts) || (NULL != t->setpts_int))
		&& (NULL != t->execute) && (NULL != t->destroy) && (NULL != t->default_opts)
		&& (0 != t->opts_size);
}

int bartorch_finufft_usable_on(int device)
{
	const struct fi_table* t = device ? &fi.device : &fi.host;
	return (table_ready(t) && fi.use_in_tools) ? 1 : 0;
}

int bartorch_finufft_usable(void)
{
	return bartorch_finufft_usable_on(0);
}

void bartorch_finufft_use_in_tools(int enable)
{
	fi.use_in_tools = (0 != enable);
}

/* A plan carries which library made it, so the operator does not have to. */
struct bartorch_fi_plan {

	const struct fi_table* table;
	finufft_plan_t plan;
};

int bartorch_finufft_plan(int device, int type, int dim, const int64_t n_modes[3], int ntrans,
		int isign, double eps, void** plan)
{
	const struct fi_table* t = device ? &fi.device : &fi.host;

	if (!table_ready(t))
		return -1;

	char opts[4096];

	pthread_mutex_lock(&fi_lock);

	int which = device ? bartorch_cuda_device() : 0;

	t->default_opts(opts);
	*(int*)(opts + t->off_device) = (which > 0) ? which : 0;

	finufft_plan_t p = NULL;
	int ret = t->makeplan(type, dim, n_modes, isign, ntrans, (float)eps, &p, opts);

	pthread_mutex_unlock(&fi_lock);

	if (0 != ret)
		return -1;

	struct bartorch_fi_plan* held = malloc(sizeof *held);

	if (NULL == held) {

		t->destroy(p);
		return -1;
	}

	held->table = t;
	held->plan = p;
	*plan = held;
	return 0;
}

/* A plan keeps the points by pointer rather than copying them, so the arrays
 * given here have to outlive it. */
int bartorch_finufft_setpts(void* plan, long M, float* x, float* y, float* z)
{
	const struct bartorch_fi_plan* p = plan;

	if (NULL != p->table->setpts)
		return p->table->setpts(p->plan, M, x, y, z, 0, NULL, NULL, NULL);

	if (M > INT32_MAX)
		return -1;

	return p->table->setpts_int(p->plan, (int)M, x, y, z, 0, NULL, NULL, NULL);
}

int bartorch_finufft_exec(void* plan, complex float* c, complex float* f)
{
	const struct bartorch_fi_plan* p = plan;
	return p->table->execute(p->plan, c, f);
}

void bartorch_finufft_free(void* plan)
{
	if (NULL == plan)
		return;

	struct bartorch_fi_plan* p = plan;

	pthread_mutex_lock(&fi_lock);
	p->table->destroy(p->plan);
	pthread_mutex_unlock(&fi_lock);

	free(p);
}
