/*
 * The Toeplitz normal of one coil against a pair of sets, in cuFFTDx kernels
 * compiled ahead of time.
 *
 * The function a normal convolves with is decomposed into eight sets of
 * frequencies, each the image under a half-cell phase along each axis, and the
 * phase along one axis passes through the transforms along the other two.  So
 * the two sets that differ only along x share their transforms along z and y,
 * forward and back, and only the transform along x is taken for each -- in one
 * kernel that also multiplies by both sets' functions and sums what comes
 * back.  For eight sets that is 20 passes over a coefficient's volume, where
 * transforming each set whole is 48.
 *
 * A pass along x reads its lines where they lie; the passes along y and z
 * stage a tile of lines side by side through shared memory, so every access to
 * the card's memory runs along x.  Kernels exist for the cubic grids listed in
 * BARTORCH_PAIRED_SIZES, four coefficients, and a real function kept as its
 * upper triangle and compressed; anything else is served as before.
 */
#include <math.h>
#include <stdbool.h>
#include <string.h>

#include <cuda_runtime_api.h>

#include <cufftdx.hpp>

#include "misc/debug.h"
#include "misc/misc.h"

#include "num/gpuops.h"

#include "paired_sizes.h"

using namespace cufftdx;

#ifndef BARTORCH_PAIRED_SM
#define BARTORCH_PAIRED_SM 890
#endif

namespace {

constexpr unsigned R = 4;

template <unsigned N, fft_direction D, unsigned F>
using FFT1 = decltype(Size<N>() + Precision<float>() + Type<fft_type::c2c>() + Direction<D>()
		+ FFTsPerBlock<F>() + SM<BARTORCH_PAIRED_SM>() + Block());

using cplx = typename FFT1<64, fft_direction::forward, 1>::value_type;

__device__ inline cplx cmul(cplx a, cplx b) { cplx c; c.x = a.x * b.x - a.y * b.y; c.y = a.x * b.y + a.y * b.x; return c; }
__device__ inline cplx cmulc(cplx a, cplx b) { cplx c; c.x = a.x * b.x + a.y * b.y; c.y = a.y * b.x - a.x * b.y; return c; }	/* a conj(b) */

constexpr size_t cmax(size_t a, size_t b) { return (a > b) ? a : b; }

template <unsigned N>
struct Shape {

	static constexpr unsigned tile = (N == 512) ? 8 : ((N % 16 == 0) ? 16 : 20);
	static constexpr unsigned pitch = tile + 1;

	using SF = FFT1<N, fft_direction::forward, tile>;
	using SI = FFT1<N, fft_direction::inverse, tile>;
	using XF = FFT1<N, fft_direction::forward, R>;
	using XI = FFT1<N, fft_direction::inverse, R>;

	static constexpr size_t smem_s = cmax(sizeof(cplx) * N * pitch, cmax(SF::shared_memory_size, SI::shared_memory_size));
	static constexpr size_t smem_x = cmax(sizeof(cplx) * R * N, cmax(XF::shared_memory_size, XI::shared_memory_size));
};

/* A pass along an axis whose elements are S apart, a tile of lines side by
 * side along x; `op` says where a line is read from and written to.  The
 * block's row (y for a pass along z, z for a pass along y) is blockIdx.y, and
 * a line's element index is the third argument of load and store. */
template <unsigned N, class FFT, class Op>
__global__ void __launch_bounds__(FFT::max_threads_per_block) strided(size_t S, size_t O, Op op)
{
	constexpr unsigned TILE = Shape<N>::tile;
	constexpr unsigned P = Shape<N>::pitch;

	extern __shared__ __align__(16) unsigned char smem[];
	cplx* tile = reinterpret_cast<cplx*>(smem);

	const size_t base = blockIdx.x * TILE + (size_t)blockIdx.y * O;
	const unsigned tid = threadIdx.x + threadIdx.y * FFT::block_dim.x;
	const unsigned nthreads = FFT::block_dim.x * FFT::block_dim.y;

	for (unsigned k = tid; k < N * TILE; k += nthreads) {

		unsigned c = k % TILE, j = k / TILE;
		tile[j * P + c] = op.load(base + (size_t)j * S + c, blockIdx.y, j);
	}

	__syncthreads();

	cplx r[FFT::storage_size];

#pragma unroll
	for (unsigned i = 0; i < FFT::elements_per_thread; i++) {

		unsigned e = threadIdx.x + i * FFT::stride;

		if (e < N)
			r[i] = tile[e * P + threadIdx.y];
	}

	__syncthreads();
	FFT().execute(r, smem);
	__syncthreads();

#pragma unroll
	for (unsigned i = 0; i < FFT::elements_per_thread; i++) {

		unsigned e = threadIdx.x + i * FFT::stride;

		if (e < N)
			tile[e * P + threadIdx.y] = r[i];
	}

	__syncthreads();

	for (unsigned k = tid; k < N * TILE; k += nthreads) {

		unsigned c = k % TILE, j = k / TILE;
		op.store(base + (size_t)j * S + c, blockIdx.y, j, tile[j * P + c]);
	}
}

struct Plain {

	cplx* d;
	__device__ cplx load(size_t g, unsigned, unsigned) const { return d[g]; }
	__device__ void store(size_t g, unsigned, unsigned, cplx v) const { d[g] = v; }
};

/* The pass along z in: the coefficient times the map and the pair's phase
 * along y and z (the row is y, the line runs along z). */
struct ZIn {

	const cplx* src;
	const cplx* map;
	const cplx* ty;
	const cplx* tz;
	cplx* B;

	__device__ cplx load(size_t g, unsigned y, unsigned z) const
	{
		cplx v = src[g];

		if (NULL != map)
			v = cmul(v, map[g]);

		return cmul(v, cmul(ty[y], tz[z]));
	}

	__device__ void store(size_t g, unsigned, unsigned, cplx v) const { B[g] = v; }
};

/* The pass along z out: the conjugates of both, the scale, and the sum into
 * the answer. */
struct ZOut {

	const cplx* B;
	const cplx* map;
	const cplx* ty;
	const cplx* tz;
	cplx* dst;
	float scale;

	__device__ cplx load(size_t g, unsigned, unsigned) const { return B[g]; }

	__device__ void store(size_t g, unsigned y, unsigned z, cplx v) const
	{
		cplx w = cmulc(v, cmul(ty[y], tz[z]));

		if (NULL != map)
			w = cmulc(w, map[g]);

		cplx d = dst[g];
		d.x += scale * w.x;
		d.y += scale * w.y;
		dst[g] = d;
	}
};

/* What the pass along x reads. */
struct XArgs {

	cplx* B[R];
	const float* psf[2];
	const unsigned* mask;
	const int* prefix;
	const cplx* tx;			/* 2 x N: each set's phase along x */
	long L;
};

/* The pass along x of both sets of the pair: per line of four coefficients,
 * each set's phase, the transform, the multiplication by the set's function at
 * the places the samples reach, the transform back, the conjugate phase, and
 * the sum of the two. */
template <unsigned N>
__global__ void __launch_bounds__(Shape<N>::XF::max_threads_per_block) fused(XArgs a)
{
	using XF = typename Shape<N>::XF;
	using XI = typename Shape<N>::XI;

	extern __shared__ __align__(16) unsigned char smem[];
	cplx* exch = reinterpret_cast<cplx*>(smem);
	__shared__ int any;

	const unsigned q = threadIdx.y;
	const size_t line = blockIdx.x;
	const unsigned tid = threadIdx.x + threadIdx.y * XF::block_dim.x;
	const unsigned nthreads = XF::block_dim.x * XF::block_dim.y;
	const size_t g0 = line * N;

	/* A line that reaches no kept place contributes nothing. */
	if (0 == tid)
		any = 0;

	__syncthreads();

	for (size_t w = (g0 >> 5) + tid; w <= ((g0 + N - 1) >> 5); w += nthreads)
		if (0 != a.mask[w])
			any = 1;

	__syncthreads();

	if (0 == any) {

		for (unsigned i = 0; i < XF::elements_per_thread; i++) {

			unsigned e = threadIdx.x + i * XF::stride;

			if (e < N) {

				cplx z;
				z.x = 0.f;
				z.y = 0.f;
				a.B[q][g0 + e] = z;
			}
		}

		return;
	}

	cplx in[XF::storage_size];
	cplx w[XF::storage_size];
	cplx acc[XF::storage_size];

#pragma unroll
	for (unsigned i = 0; i < XF::elements_per_thread; i++) {

		unsigned e = threadIdx.x + i * XF::stride;
		cplx z;
		z.x = 0.f;
		z.y = 0.f;
		in[i] = (e < N) ? a.B[q][g0 + e] : z;
		acc[i] = z;
	}

	for (int sx = 0; sx < 2; sx++) {

#pragma unroll
		for (unsigned i = 0; i < XF::elements_per_thread; i++) {

			unsigned e = threadIdx.x + i * XF::stride;
			w[i] = (e < N) ? cmul(in[i], a.tx[sx * N + e]) : in[i];
		}

		__syncthreads();
		XF().execute(w, smem);
		__syncthreads();

#pragma unroll
		for (unsigned i = 0; i < XF::elements_per_thread; i++) {

			unsigned e = threadIdx.x + i * XF::stride;

			if (e < N)
				exch[q * N + e] = w[i];
		}

		__syncthreads();

		for (unsigned e = tid; e < N; e += nthreads) {

			const size_t g = g0 + e;
			cplx v[R];
			cplx o[R];

#pragma unroll
			for (unsigned c = 0; c < R; c++) {

				v[c] = exch[c * N + e];
				o[c].x = 0.f;
				o[c].y = 0.f;
			}

			const unsigned word = a.mask[g >> 5];
			const unsigned bit = 1u << (g & 31);

			if (word & bit) {

				const long j = a.prefix[g >> 5] + __popc(word & (bit - 1));

#pragma unroll
				for (unsigned r = 0; r < R; r++)
#pragma unroll
					for (unsigned c = 0; c < R; c++) {

						const unsigned lo = (r < c) ? r : c;
						const unsigned hi = (r < c) ? c : r;
						const float m = a.psf[sx][(long)(lo + hi * (hi + 1) / 2) * a.L + j];

						o[r].x += m * v[c].x;
						o[r].y += m * v[c].y;
					}
			}

#pragma unroll
			for (unsigned r = 0; r < R; r++)
				exch[r * N + e] = o[r];
		}

		__syncthreads();

#pragma unroll
		for (unsigned i = 0; i < XF::elements_per_thread; i++) {

			unsigned e = threadIdx.x + i * XF::stride;

			if (e < N)
				w[i] = exch[q * N + e];
		}

		__syncthreads();
		XI().execute(w, smem);

#pragma unroll
		for (unsigned i = 0; i < XF::elements_per_thread; i++) {

			unsigned e = threadIdx.x + i * XF::stride;

			if (e < N) {

				cplx t = cmulc(w[i], a.tx[sx * N + e]);
				acc[i].x += t.x;
				acc[i].y += t.y;
			}
		}
	}

#pragma unroll
	for (unsigned i = 0; i < XF::elements_per_thread; i++) {

		unsigned e = threadIdx.x + i * XF::stride;

		if (e < N)
			a.B[q][g0 + e] = acc[i];
	}
}

/* One coil against one pair: what a call hands over. */
struct Call {

	cplx* dst;
	const cplx* src;
	const cplx* map;
	cplx* B;
	const float* psf[2];
	const unsigned* mask;
	const int* prefix;
	long L;
	const cplx* tab;		/* the pair's tables: x of each set, y, z */
	float scale;
};

template <unsigned N>
int prepare(void)
{
	using S = Shape<N>;

	int device = 0;
	int optin = 0;

	if ((cudaSuccess != cudaGetDevice(&device))
	    || (cudaSuccess != cudaDeviceGetAttribute(&optin, cudaDevAttrMaxSharedMemoryPerBlockOptin, device))
	    || ((size_t)optin < cmax(S::smem_s, S::smem_x))) {

		cudaGetLastError();
		return -1;
	}

	int s = (int)S::smem_s;
	int x = (int)S::smem_x;

	if (   (cudaSuccess != cudaFuncSetAttribute(strided<N, typename S::SF, ZIn>, cudaFuncAttributeMaxDynamicSharedMemorySize, s))
	    || (cudaSuccess != cudaFuncSetAttribute(strided<N, typename S::SF, Plain>, cudaFuncAttributeMaxDynamicSharedMemorySize, s))
	    || (cudaSuccess != cudaFuncSetAttribute(strided<N, typename S::SI, Plain>, cudaFuncAttributeMaxDynamicSharedMemorySize, s))
	    || (cudaSuccess != cudaFuncSetAttribute(strided<N, typename S::SI, ZOut>, cudaFuncAttributeMaxDynamicSharedMemorySize, s))
	    || (cudaSuccess != cudaFuncSetAttribute(fused<N>, cudaFuncAttributeMaxDynamicSharedMemorySize, x))) {

		cudaGetLastError();
		return -1;
	}

	return 0;
}

/* The passes along z and y in, for the four coefficients of a coil. */
template <unsigned N>
void run_in(const Call* c, cudaStream_t stream)
{
	using S = Shape<N>;

	const size_t V = (size_t)N * N * N;
	const dim3 grid(N / S::tile, N);
	const cplx* ty = c->tab + 2 * N;
	const cplx* tz = c->tab + 3 * N;

	for (unsigned r = 0; r < R; r++)
		strided<N, typename S::SF, ZIn><<<grid, S::SF::block_dim, S::smem_s, stream>>>((size_t)N * N, (size_t)N,
				ZIn{ c->src + r * V, c->map, ty, tz, c->B + r * V });

	for (unsigned r = 0; r < R; r++)
		strided<N, typename S::SF, Plain><<<grid, S::SF::block_dim, S::smem_s, stream>>>((size_t)N, (size_t)N * N,
				Plain{ c->B + r * V });
}

/* The pass along x of both sets, and the passes along y and z back out into
 * the answer. */
template <unsigned N>
void run_out(const Call* c, cudaStream_t stream)
{
	using S = Shape<N>;

	const size_t V = (size_t)N * N * N;
	const dim3 grid(N / S::tile, N);
	const cplx* ty = c->tab + 2 * N;
	const cplx* tz = c->tab + 3 * N;

	XArgs a;

	for (unsigned r = 0; r < R; r++)
		a.B[r] = c->B + r * V;

	a.psf[0] = c->psf[0];
	a.psf[1] = c->psf[1];
	a.mask = c->mask;
	a.prefix = c->prefix;
	a.tx = c->tab;
	a.L = c->L;

	fused<N><<<(unsigned)((size_t)N * N), S::XF::block_dim, S::smem_x, stream>>>(a);

	for (unsigned r = 0; r < R; r++)
		strided<N, typename S::SI, Plain><<<grid, S::SI::block_dim, S::smem_s, stream>>>((size_t)N, (size_t)N * N,
				Plain{ c->B + r * V });

	for (unsigned r = 0; r < R; r++)
		strided<N, typename S::SI, ZOut><<<grid, S::SI::block_dim, S::smem_s, stream>>>((size_t)N * N, (size_t)N,
				ZOut{ c->B + r * V, c->map, ty, tz, c->dst + r * V, c->scale });
}

struct Entry {

	unsigned n;
	int (*prepare)(void);
	void (*run_in)(const Call*, cudaStream_t);
	void (*run_out)(const Call*, cudaStream_t);
};

#define BARTORCH_PAIRED_ENTRY(n) { n, prepare<n>, run_in<n>, run_out<n> },

const Entry table[] = { BARTORCH_PAIRED_SIZES(BARTORCH_PAIRED_ENTRY) };

/* A set's phase along one axis, as `phase_setup` in coset.cuh builds it --
 * the shift, the centring, the fftmod folded in -- in double precision. */
void axis_phase(long d, float shift, cplx* out)
{
	double s = shift;

	if (1 < d)
		s += d / 2. - d / 2;

	double slope = 2. * M_PI * s / d;
	double offset = -slope * d / 2.;

	long centre = d / 2;
	double half = (double)centre / d;

	slope += 2. * M_PI * half;
	offset -= 2. * M_PI * half * centre / 2.;

	for (long i = 0; i < d; i++) {

		double v = offset + i * slope;
		out[i].x = (float)cos(v);
		out[i].y = (float)sin(v);
	}
}

} // namespace

struct bartorch_paired {

	const Entry* entry;
	int pairs;
	cplx* tables;			/* on the card: per pair, x of each set, y, z */
	float scale;
};

/* The pair kernels for a grid of `dims`, with the sets' shifts as
 * `bartorch_psf_shift` gives them (set i and i + 1, i even, differ only along
 * x); NULL where there are none or the card cannot run them. */
extern "C" struct bartorch_paired* bartorch_paired_create(const long dims[3], int coeffs, int sets, const float (*shifts)[3])
{
	if ((R != (unsigned)coeffs) || (8 != sets) || (dims[0] != dims[1]) || (dims[0] != dims[2]))
		return NULL;

	const Entry* entry = NULL;

	for (const Entry& e : table)
		if ((long)e.n == dims[0])
			entry = &e;

	if ((NULL == entry) || (0 != entry->prepare()))
		return NULL;

	for (int i = 0; i < sets; i += 2)
		if ((shifts[i][1] != shifts[i + 1][1]) || (shifts[i][2] != shifts[i + 1][2]))
			return NULL;

	const long n = dims[0];
	const int pairs = sets / 2;

	cplx* host = (cplx*)xmalloc(sizeof(cplx) * 4 * n * pairs);

	for (int k = 0; k < pairs; k++) {

		cplx* t = host + 4 * n * k;

		axis_phase(n, shifts[2 * k][0], t);
		axis_phase(n, shifts[2 * k + 1][0], t + n);
		axis_phase(n, shifts[2 * k][1], t + 2 * n);
		axis_phase(n, shifts[2 * k][2], t + 3 * n);
	}

	struct bartorch_paired* p = (struct bartorch_paired*)xmalloc(sizeof *p);

	p->entry = entry;
	p->pairs = pairs;
	p->scale = (float)(1. / ((double)n * n * n));

	if (cudaSuccess != cudaMalloc((void**)&p->tables, sizeof(cplx) * 4 * n * pairs)) {

		cudaGetLastError();
		xfree(host);
		xfree(p);
		return NULL;
	}

	cudaMemcpy(p->tables, host, sizeof(cplx) * 4 * n * pairs, cudaMemcpyHostToDevice);
	xfree(host);

	debug_printf(DP_DEBUG1, "bartorch: paired kernels for %ld^3\n", n);

	return p;
}

extern "C" void bartorch_paired_free(struct bartorch_paired* p)
{
	if (NULL == p)
		return;

	cudaFree(p->tables);
	xfree(p);
}

/* One coil against pair `k`, in two halves: `in` takes the coil's four
 * coefficients `src` -- times the sensitivity `map`, or NULL -- through the
 * passes along z and y into `scratch`; `out` takes them through the pass
 * along x against the two sets' functions `psf0`, `psf1` (the upper triangle,
 * compressed to the places `mask` and `prefix` keep, `L` of them) and back out,
 * adding to `dst`.  Only `out` reads the functions, so the card needs to be
 * held for them only between the two. */
extern "C" void bartorch_paired_in(const struct bartorch_paired* p, int k,
		const _Complex float* src, const _Complex float* map, _Complex float* scratch)
{
	Call c;

	memset(&c, 0, sizeof c);
	c.src = (const cplx*)src;
	c.map = (const cplx*)map;
	c.B = (cplx*)scratch;
	c.tab = p->tables + 4 * p->entry->n * k;

	p->entry->run_in(&c, cuda_get_stream());

	CUDA_KERNEL_ERROR;
}

extern "C" void bartorch_paired_out(const struct bartorch_paired* p, int k,
		_Complex float* dst, const _Complex float* map, _Complex float* scratch,
		const float* psf0, const float* psf1, const unsigned int* mask, const int* prefix, long L)
{
	Call c;

	memset(&c, 0, sizeof c);
	c.dst = (cplx*)dst;
	c.map = (const cplx*)map;
	c.B = (cplx*)scratch;
	c.psf[0] = psf0;
	c.psf[1] = psf1;
	c.mask = mask;
	c.prefix = prefix;
	c.L = L;
	c.tab = p->tables + 4 * p->entry->n * k;
	c.scale = p->scale;

	p->entry->run_out(&c, cuda_get_stream());

	CUDA_KERNEL_ERROR;
}
