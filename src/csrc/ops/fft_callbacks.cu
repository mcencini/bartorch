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
#include <stdint.h>

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

	/* The callbacks index the volume in 32 bits. */
	if (dims[0] * dims[1] * dims[2] > (long)UINT32_MAX)
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


/* --- a Cartesian normal --------------------------------------------------
 *
 * The same pair of transforms with callbacks, for a Cartesian pattern: over
 * the axes the pattern varies along, batched over the rest, and reading and
 * writing the image in BART's layout through the index the callbacks recover
 * (grid.cuh). */

#include "grid.cuh"

struct bartorch_cb_grid {

	cufftHandle plan[2];		/* forward, inverse */
	bool made[2];
	size_t work;
	struct grid_info geom;		/* what does not change between transforms */
	struct grid_info* info;		/* on the card, read by all four callbacks */
};

static void destroy_grid(struct bartorch_cb_grid* p)
{
	for (int i = 0; i < 2; i++)
		if (p->made[i])
			cufftDestroy(p->plan[i]);

	if (NULL != p->info)
		cudaFree(p->info);

	xfree(p);
}

extern "C" void bartorch_cb_grid_free(struct bartorch_cb_grid* p)
{
	if (NULL != p)
		destroy_grid(p);
}

/* The pair for an image of spatial `dims`, transformed along `flags`, with
 * `kept` places of a plane in the gathered spectrum -- or NULL where cuFFT
 * cannot link the callbacks in.  `mask`, `prefix` and `mod` are on the card
 * and stay the caller's.  `unitary` scales each direction by one over the
 * square root of the plane; without it the forward is cuFFT's unnormalized
 * transform and the inverse its adjoint, as BART's uncentred `fft` is. */
extern "C" struct bartorch_cb_grid* bartorch_cb_grid_create(const long dims[3], unsigned long flags, long kept,
		const unsigned int* mask, const int* prefix, const _Complex float* mod[3], int unitary)
{
	set_jit_t set_jit = set_jit_callback();

	if (NULL == set_jit)
		return NULL;

	if (dims[0] * dims[1] * dims[2] > (long)UINT32_MAX)
		return NULL;

	struct grid_info g;

	/* cuFFT's first dimension is the slowest, BART's the fastest. */
	int n[3];
	int rank = 0;

	for (int i = 2; i >= 0; i--)
		if ((0 != ((flags >> i) & 1UL)) && (1 < dims[i]))
			n[rank++] = (int)dims[i];

	if (0 == rank)
		return NULL;

	unsigned int ps = 1;
	unsigned int bs = 1;
	unsigned int ms = 1;

	for (int a = 0; a < 3; a++) {

		bool t = (0 != ((flags >> a) & 1UL)) && (1 < dims[a]);

		g.n[a] = (unsigned int)dims[a];
		g.mstr[a] = ms;
		g.pstr[a] = t ? ps : 0;
		g.bstr[a] = (!t && (1 < dims[a])) ? bs : 0;
		g.mod[a] = t ? (const cuFloatComplex*)mod[a] : NULL;

		ms *= (unsigned int)dims[a];

		if (t)
			ps *= (unsigned int)dims[a];
		else
			bs *= (unsigned int)dims[a];
	}

	g.plane = ps;
	g.L = (unsigned int)kept;
	g.scale = (0 != unitary) ? (float)(1. / sqrt((double)ps)) : 1.f;
	g.map = NULL;
	g.src = NULL;
	g.dst = NULL;
	g.bank = NULL;
	g.mask = mask;
	g.prefix = prefix;

	struct bartorch_cb_grid* p = (struct bartorch_cb_grid*)xmalloc(sizeof *p);

	p->made[0] = false;
	p->made[1] = false;
	p->work = 0;
	p->geom = g;
	p->info = NULL;

	const char* load[2] = { "bartorch_grid_load_in", "bartorch_grid_load_scatter" };
	const char* store[2] = { "bartorch_grid_store_gather", "bartorch_grid_store_out" };

	const void* fatbin = bartorch_fft_callbacks_lto;
	size_t size = sizeof bartorch_fft_callbacks_lto;

	cufftResult r = CUFFT_SUCCESS;

	if (cudaSuccess != cudaMalloc((void**)&p->info, sizeof(struct grid_info))) {

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
			r = cufftMakePlanMany(p->plan[i], rank, n, NULL, 1, 0, NULL, 1, 0, CUFFT_C2C, (int)bs, &work);

		if (work > p->work)
			p->work = work;
	}

	if (CUFFT_SUCCESS != r) {

		debug_printf(DP_DEBUG1, "bartorch: cuFFT linked no Cartesian callbacks in (%d)\n", (int)r);

		destroy_grid(p);
		cudaGetLastError();

		return NULL;
	}

	return p;
}

__global__ static void kern_set_grid_info(struct grid_info* dst, struct grid_info src)
{
	*dst = src;
}

static void run_grid(struct bartorch_cb_grid* p, int which, const struct grid_info* g, void* volume)
{
	cudaStream_t stream = cuda_get_stream();

	kern_set_grid_info<<<1, 1, 0, stream>>>(p->info, *g);

	CUDA_KERNEL_ERROR;

	void* work = (0 < p->work) ? cuda_malloc((long)p->work) : NULL;

	cufftHandle plan = p->plan[which];

	check(cufftSetStream(plan, stream));

	if (NULL != work)
		check(cufftSetWorkArea(plan, work));

	/* In place on a buffer in cuFFT's own layout: what goes in is read
	 * through the load callback and what comes out is written through the
	 * store callback, so the buffer is only where the transform works. */
	check(cufftExecC2C(plan, (cufftComplex*)volume, (cufftComplex*)volume, (0 == which) ? CUFFT_FORWARD : CUFFT_INVERSE));

	if (NULL != work)
		cuda_free(work);
}

/* The coefficient `src`, multiplied by the sensitivity `map` (or none) and the
 * centring, transformed, and its kept places gathered into `bank`. */
extern "C" void bartorch_cb_grid_forward(struct bartorch_cb_grid* p, _Complex float* bank, _Complex float* volume,
		const _Complex float* src, const _Complex float* map)
{
	struct grid_info g = p->geom;

	g.src = (const cuFloatComplex*)src;
	g.map = (const cuFloatComplex*)map;
	g.bank = (cuFloatComplex*)bank;

	run_grid(p, 0, &g, volume);
}

/* `bank` scattered, transformed back, multiplied by the conjugates of the
 * centring and the sensitivity, and added to `dst`. */
extern "C" void bartorch_cb_grid_inverse(struct bartorch_cb_grid* p, _Complex float* dst, _Complex float* volume,
		const _Complex float* bank, const _Complex float* map)
{
	struct grid_info g = p->geom;

	g.dst = (cuFloatComplex*)dst;
	g.map = (const cuFloatComplex*)map;
	g.bank = (cuFloatComplex*)bank;

	run_grid(p, 1, &g, volume);
}
