/*
 * The FINUFFT libraries, as BART reaches them.
 *
 * The `finufft` and `cufinufft` wheels each carry a compiled shared library
 * with a plain C plan API.  The host hands the entry points and the byte
 * layout of the options struct across the ABI, both read from those same
 * packages, so a release that moves a field cannot be misread here and
 * nothing is built or vendored.
 *
 * The two libraries have the same entry points and answer different memory --
 * cuFINUFFT spells its defaults without the precision suffix, and carries a
 * device number in its options where FINUFFT carries a thread count -- so
 * they are held as two tables, picked by where the data is.
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

struct fi_table {

	fi_makeplan_t makeplan;
	fi_setpts_t setpts;
	fi_execute_t execute;
	fi_destroy_t destroy;
	fi_default_opts_t default_opts;

	int opts_size;
	int off_device;		/* nthreads on the host, gpu_device_id on a device */
	int off_upsampling;	/* upsampfac, a double, which both spell alike */
	int off_spreadonly;	/* spreadinterponly on the host, gpu_ prefixed on a device */
};

static struct {

	struct fi_table host;
	struct fi_table device;

	int use_in_tools;
	double tolerance;
	double upsampling;
	int threads;

} fi = { .use_in_tools = 0, .tolerance = 1.e-3, .upsampling = 1.25, .threads = 0 };

static pthread_mutex_t fi_lock = PTHREAD_MUTEX_INITIALIZER;

/* "finufftf_makeplan" fills the host table, "cufinufftf_makeplan" the device
 * one, and the two carry the same five entry points. */
int bartorch_finufft_set(const char* symbol, void* fn)
{
	bool cuda = (0 == strncmp(symbol, "cu", 2));
	struct fi_table* t = cuda ? &fi.device : &fi.host;
	const char* name = symbol + (cuda ? 2 : 0);

	if (0 == strcmp(name, "finufftf_makeplan")) t->makeplan = (fi_makeplan_t)fn;
	else if (0 == strcmp(name, "finufftf_setpts")) t->setpts = (fi_setpts_t)fn;
	else if (0 == strcmp(name, "finufftf_execute")) t->execute = (fi_execute_t)fn;
	else if (0 == strcmp(name, "finufftf_destroy")) t->destroy = (fi_destroy_t)fn;
	else if ((0 == strcmp(name, "finufftf_default_opts")) || (0 == strcmp(name, "finufft_default_opts"))) t->default_opts = (fi_default_opts_t)fn;
	else return -1;

	return 0;
}

/* The byte offsets of the fields this sets: the thread count on the host and
 * the device number on a card, and the grid FINUFFT spreads onto. */
int bartorch_finufft_layout(int device, int opts_size, int device_field, int upsampling_field, int spreadonly_field)
{
	struct fi_table* t = device ? &fi.device : &fi.host;

	if ((opts_size < 16) || (opts_size > 4096) || (device_field < 0) || (device_field + 4 > opts_size))
		return -1;

	if ((spreadonly_field < 0) || (spreadonly_field + 4 > opts_size))
		return -1;

	if ((upsampling_field < 0) || (upsampling_field + 8 > opts_size))
		return -1;

	t->opts_size = opts_size;
	t->off_device = device_field;
	t->off_upsampling = upsampling_field;
	t->off_spreadonly = spreadonly_field;
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

/* How far past the image FINUFFT spreads before it transforms.
 *
 * Two is the textbook grid; a quarter over trades a smaller one for a wider
 * kernel, and that is the default here: it holds a quarter of the memory the
 * textbook grid does at the tolerance this asks for, and a reconstruction is
 * not made better by a transform an order more accurate than the data.  Zero
 * lets FINUFFT weigh it per problem instead. */
void bartorch_finufft_set_upsampling(double upsampling)
{
	if ((0. == upsampling) || ((upsampling > 1.) && (upsampling <= 4.)))
		fi.upsampling = upsampling;
}

double bartorch_finufft_upsampling(void)
{
	return fi.upsampling;
}

/* How many threads a transform on the host is given.
 *
 * Zero, the state this starts in, leaves the count to FINUFFT, which takes a
 * thread per physical core.  bartorch_set_num_threads sets this along with
 * BART's own, so one number covers the process; this is how that is undone
 * without setting BART to a count of its own.  A card has no say in it --
 * cuFINUFFT carries a device number where FINUFFT carries this. */
void bartorch_finufft_set_threads(int n)
{
	fi.threads = (n > 0) ? n : 0;
}

int bartorch_finufft_threads(void)
{
	return fi.threads;
}

/* The entry points are there and the options layout is known. */
static bool table_ready(const struct fi_table* t)
{
	return (NULL != t->makeplan) && (NULL != t->setpts) && (NULL != t->execute)
		&& (NULL != t->destroy) && (NULL != t->default_opts) && (0 != t->opts_size);
}

/* The cufinufft wheel can be installed beside a library built without CUDA,
 * and then nothing can ever be on a device for it to serve. */
int bartorch_finufft_usable_on(int device)
{
	if (device && !bartorch_cuda_built())
		return 0;

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

/* Plans made and not yet destroyed.  A plan belongs to whatever made it -- an
 * operator, a point spread function, the spreading a compressed one is masked
 * with -- and outlives none of them, so this is back at zero once the last of
 * them is gone.  That is a property worth testing, and one RSS cannot be read
 * for: FINUFFT's own multithreaded execute retains a kilobyte per thread per
 * call, which any measurement of the process would drown this in. */
static long fi_live_plans;

/* `spread_only` asks FINUFFT for the spreading alone -- no transform, no
 * deapodisation -- which puts the kernel's own footprint on the grid.  That is
 * what a compressed point spread function's mask is: which grid points the
 * samples reach. */
int bartorch_finufft_plan(int device, int type, int dim, const int64_t n_modes[3], int ntrans,
		int isign, double eps, double upsampling, int spread_only, void** plan)
{
	const struct fi_table* t = device ? &fi.device : &fi.host;

	if (!table_ready(t))
		return -1;

	char opts[4096];

	pthread_mutex_lock(&fi_lock);

	t->default_opts(opts);

	/* One offset, a different option on each side: the device to run on, or
	 * the number of threads to take. */
	if (device) {

		int which = bartorch_cuda_device();
		*(int*)(opts + t->off_device) = (which > 0) ? which : 0;

	} else {

		*(int*)(opts + t->off_device) = fi.threads;
	}

	if (0. != upsampling)
		*(double*)(opts + t->off_upsampling) = upsampling;

	if (0 != spread_only)
		*(int*)(opts + t->off_spreadonly) = 1;

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

#pragma omp atomic
	fi_live_plans++;

	return 0;
}

/* A plan keeps the points by pointer rather than copying them, so the arrays
 * given here have to outlive it. */
int bartorch_finufft_setpts(void* plan, long M, float* x, float* y, float* z)
{
	const struct bartorch_fi_plan* p = plan;

	return p->table->setpts(p->plan, M, x, y, z, 0, NULL, NULL, NULL);
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

#pragma omp atomic
	fi_live_plans--;
}

long bartorch_finufft_live_plans(void)
{
	return fi_live_plans;
}
