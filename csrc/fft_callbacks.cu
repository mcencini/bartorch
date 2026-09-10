/*
 * A volume's transform pair with the passes around it run inside it.
 *
 * Convolving a streamed set costs, per coil and coefficient, a forward and an
 * inverse transform of the volume and, around them, passes of its own: the
 * phase and the sensitivity on the way in, the gather after, the scatter
 * before the inverse, the conjugates and the sum after it.  Each reads or
 * writes the whole volume.  cuFFT runs callbacks where its kernels read their
 * input and write their output, so each of those becomes part of a read or a
 * write the transform makes anyway.
 *
 * The callbacks are LTO-IR (fft_callbacks_lto.cu, embedded at build time)
 * that cuFFT links in with nvJitLink when a plan is made, once for an
 * operator.  Where that cannot be done there is no plan, and the passes run
 * on their own.
 */
#include <dlfcn.h>
#include <stdbool.h>
#include <stddef.h>

#include <cuda_runtime_api.h>
#include <cufftXt.h>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/gpuops.h"

#include "coset.cuh"
#include "fft_callbacks_lto.h"

struct bartorch_cb_fft {

	cufftHandle plan[2];		/* forward, inverse */
	bool made[2];
	size_t work;			/* workspace the larger of the two needs */
	struct coset_info* info;	/* on the card, read by all four callbacks */
};

typedef cufftResult (*set_jit_t)(cufftHandle plan, const char* name, const void* fatbin,
		size_t size, cufftXtCallbackType type, void** info);

/* cuFFT's entry point for linking a callback in, looked up in the cuFFT this
 * library runs against rather than linked to, so that one without it leaves
 * the passes on their own instead of a library that does not load. */
static set_jit_t set_jit_callback(void)
{
	Dl_info where;

	if ((0 == dladdr((void*)&cufftCreate, &where)) || (NULL == where.dli_fname))
		return NULL;

	void* lib = dlopen(where.dli_fname, RTLD_NOW | RTLD_NOLOAD);

	if (NULL == lib)
		return NULL;

	set_jit_t fn = (set_jit_t)dlsym(lib, "__cufftXtSetJITCallback_12_7");

	dlclose(lib);

	return fn;
}

static void destroy(struct bartorch_cb_fft* p)
{
	for (int i = 0; i < 2; i++)
		if (p->made[i])
			cufftDestroy(p->plan[i]);

	if (NULL != p->info)
		cudaFree(p->info);

	xfree(p);
}

extern "C" void bartorch_cb_fft_free(struct bartorch_cb_fft* p)
{
	if (NULL != p)
		destroy(p);
}

/* The pair for a volume of `dims`, or NULL where cuFFT cannot link the
 * callbacks in. */
extern "C" struct bartorch_cb_fft* bartorch_cb_fft_create(const long dims[3])
{
	set_jit_t set_jit = set_jit_callback();

	if (NULL == set_jit) {

		debug_printf(DP_DEBUG1, "bartorch: this cuFFT links no callbacks in\n");
		return NULL;
	}

	/* cuFFT's first dimension is the slowest, BART's the fastest. */
	int n[3];
	int rank = 0;

	for (int i = 2; i >= 0; i--)
		if (1 < dims[i])
			n[rank++] = (int)dims[i];

	if (0 == rank)
		return NULL;

	struct bartorch_cb_fft* p = (struct bartorch_cb_fft*)xmalloc(sizeof *p);

	p->made[0] = false;
	p->made[1] = false;
	p->work = 0;
	p->info = NULL;

	const char* load[2] = { "bartorch_load_in", "bartorch_load_scatter" };
	const char* store[2] = { "bartorch_store_gather", "bartorch_store_out" };

	const void* fatbin = bartorch_fft_callbacks_lto;
	size_t size = sizeof bartorch_fft_callbacks_lto;

	cufftResult r = CUFFT_SUCCESS;

	if (cudaSuccess != cudaMalloc((void**)&p->info, sizeof(struct coset_info))) {

		p->info = NULL;
		r = CUFFT_ALLOC_FAILED;
	}

	for (int i = 0; (i < 2) && (CUFFT_SUCCESS == r); i++) {

		size_t work = 0;

		r = cufftCreate(&p->plan[i]);

		if (CUFFT_SUCCESS != r)
			break;

		p->made[i] = true;

		r = set_jit(p->plan[i], load[i], fatbin, size, CUFFT_CB_LD_COMPLEX, (void**)&p->info);

		if (CUFFT_SUCCESS == r)
			r = set_jit(p->plan[i], store[i], fatbin, size, CUFFT_CB_ST_COMPLEX, (void**)&p->info);

		if (CUFFT_SUCCESS == r)
			r = cufftSetAutoAllocation(p->plan[i], 0);

		if (CUFFT_SUCCESS == r)
			r = cufftMakePlanMany(p->plan[i], rank, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, 1, &work);

		if (work > p->work)
			p->work = work;
	}

	if (CUFFT_SUCCESS != r) {

		debug_printf(DP_DEBUG1, "bartorch: cuFFT linked no callbacks in (%d); the passes run on their own\n", (int)r);

		destroy(p);
		cudaGetLastError();

		return NULL;
	}

	return p;
}

static void check(cufftResult r)
{
	if (CUFFT_SUCCESS != r)
		error("bartorch: cuFFT error %d\n", (int)r);
}

__global__ static void kern_set_info(struct coset_info* dst, struct coset_info src)
{
	*dst = src;
}

static void run(struct bartorch_cb_fft* p, int which, const struct coset_info* info, void* in, void* out)
{
	cudaStream_t stream = cuda_get_stream();

	/* The callbacks read their arguments from the card.  A launch takes
	 * them there in its parameters, in order with the transform after it
	 * and without waiting on the transform before. */
	kern_set_info<<<1, 1, 0, stream>>>(p->info, *info);

	CUDA_KERNEL_ERROR;

	void* work = (0 < p->work) ? cuda_malloc((long)p->work) : NULL;

	cufftHandle plan = p->plan[which];

	check(cufftSetStream(plan, stream));

	if (NULL != work)
		check(cufftSetWorkArea(plan, work));

	check(cufftExecC2C(plan, (cufftComplex*)in, (cufftComplex*)out, (0 == which) ? CUFFT_FORWARD : CUFFT_INVERSE));

	if (NULL != work)
		cuda_free(work);
}

static struct coset_info info_for(int N, const long dims[], const float shift[3], float scale,
		const unsigned int* mask, const int* prefix, const _Complex float* map)
{
	struct coset_info c;

	c.phase = phase_setup(N, dims, shift, scale);
	c.map = (const cuFloatComplex*)map;
	c.dst = NULL;
	c.bank = NULL;
	c.mask = mask;
	c.prefix = prefix;

	return c;
}

/* The coefficient `src`, multiplied by the set's phase and the coil's
 * sensitivity, transformed, and gathered into `bank`.  `volume` is where the
 * transform works; `src` is left as it is. */
extern "C" void bartorch_cb_fft_forward(struct bartorch_cb_fft* p, int N, const long dims[],
		const float shift[3], float scale, const unsigned int* mask, const int* prefix,
		_Complex float* bank, _Complex float* volume, const _Complex float* src, const _Complex float* map)
{
	struct coset_info c = info_for(N, dims, shift, scale, mask, prefix, map);

	c.bank = (cuFloatComplex*)bank;

	run(p, 0, &c, (void*)src, volume);
}

/* The gathered spectrum `bank`, scattered, transformed back, multiplied by the
 * conjugates of the phase and the sensitivity, and added to `dst`. */
extern "C" void bartorch_cb_fft_inverse(struct bartorch_cb_fft* p, int N, const long dims[],
		const float shift[3], float scale, const unsigned int* mask, const int* prefix,
		_Complex float* dst, _Complex float* volume, const _Complex float* bank, const _Complex float* map)
{
	struct coset_info c = info_for(N, dims, shift, scale, mask, prefix, map);

	c.bank = (cuFloatComplex*)bank;
	c.dst = (cuFloatComplex*)dst;

	run(p, 1, &c, volume, volume);
}
