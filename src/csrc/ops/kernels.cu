/*
 * Kernels of the streamed convolution that BART has no single pass for.
 *
 * A coefficient goes into the transform multiplied by the set's linear phase
 * and by the coil's sensitivity, and comes out of it multiplied by the
 * conjugates of both and added to the answer.  BART applies the phase in one
 * pass and the sensitivity in another, and each pass over a 256^3 volume
 * reads and writes it whole.  These do both in one: three streams over the
 * volume on the way in, four on the way out.  Where cuFFT can run them
 * inside the transforms (fft_callbacks.cu) they run there, and these serve
 * where it cannot.
 */
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <math.h>
#include <stdbool.h>

#include "misc/misc.h"

#include "num/gpukrnls_misc.h"
#include "num/gpuops.h"

#include "coset.cuh"

#include <cuda_bf16.h>

__device__ static inline float widen(float v) { return v; }
__device__ static inline float widen(__nv_bfloat16 v) { return __bfloat162float(v); }

/* dst = src * map * phase */
__global__ static void kern_phase_map_in(struct phase_conf c, cuFloatComplex* dst,
		const cuFloatComplex* src, const cuFloatComplex* map)
{
	int startX = threadIdx.x + blockDim.x * blockIdx.x;
	int strideX = blockDim.x * gridDim.x;
	int startY = threadIdx.y + blockDim.y * blockIdx.y;
	int strideY = blockDim.y * gridDim.y;
	int startZ = threadIdx.z + blockDim.z * blockIdx.z;
	int strideZ = blockDim.z * gridDim.z;

	for (long z = startZ; z < c.dims[2]; z += strideZ)
		for (long y = startY; y < c.dims[1]; y += strideY)
			for (long x = startX; x < c.dims[0]; x += strideX) {

				long idx = x + c.dims[0] * (y + c.dims[1] * z);

				cuFloatComplex w = cuCmulf(map[idx], phase_at(c, x, y, z, false));

				for (long i = 0; i < c.batch; i++)
					dst[idx + i * c.tot] = cuCmulf(src[idx + i * c.tot], w);
			}
}

/* dst += src * conj(map) * conj(phase) */
__global__ static void kern_phase_map_out(struct phase_conf c, cuFloatComplex* dst,
		const cuFloatComplex* src, const cuFloatComplex* map)
{
	int startX = threadIdx.x + blockDim.x * blockIdx.x;
	int strideX = blockDim.x * gridDim.x;
	int startY = threadIdx.y + blockDim.y * blockIdx.y;
	int strideY = blockDim.y * gridDim.y;
	int startZ = threadIdx.z + blockDim.z * blockIdx.z;
	int strideZ = blockDim.z * gridDim.z;

	for (long z = startZ; z < c.dims[2]; z += strideZ)
		for (long y = startY; y < c.dims[1]; y += strideY)
			for (long x = startX; x < c.dims[0]; x += strideX) {

				long idx = x + c.dims[0] * (y + c.dims[1] * z);

				cuFloatComplex w = cuCmulf(cuConjf(map[idx]), phase_at(c, x, y, z, true));

				for (long i = 0; i < c.batch; i++)
					dst[idx + i * c.tot] = cuCaddf(dst[idx + i * c.tot], cuCmulf(src[idx + i * c.tot], w));
			}
}

extern "C" void bartorch_cuda_phase_map_in(int N, const long dims[], const float shift[3], float scale,
		_Complex float* dst, const _Complex float* src, const _Complex float* map)
{
	struct phase_conf c = phase_setup(N, dims, shift, scale);

	const void* func = (const void*)kern_phase_map_in;

	kern_phase_map_in<<<getGridSize3(c.dims, func), getBlockSize3(c.dims, func), 0, cuda_get_stream()>>>(
			c, (cuFloatComplex*)dst, (const cuFloatComplex*)src, (const cuFloatComplex*)map);

	CUDA_KERNEL_ERROR;
}

extern "C" void bartorch_cuda_phase_map_out(int N, const long dims[], const float shift[3], float scale,
		_Complex float* dst, const _Complex float* src, const _Complex float* map)
{
	struct phase_conf c = phase_setup(N, dims, shift, scale);

	const void* func = (const void*)kern_phase_map_out;

	kern_phase_map_out<<<getGridSize3(c.dims, func), getBlockSize3(c.dims, func), 0, cuda_get_stream()>>>(
			c, (cuFloatComplex*)dst, (const cuFloatComplex*)src, (const cuFloatComplex*)map);

	CUDA_KERNEL_ERROR;
}

/* Gather and scatter over the places the samples reach, found through the
 * mask and counts of coset.cuh.  A scatter writes zeros everywhere else, which
 * is what the transform that follows it needs to see there. */
__global__ static void kern_gather(long V, const unsigned int* mask, const int* prefix,
		cuFloatComplex* dst, const cuFloatComplex* src)
{
	long start = threadIdx.x + (long)blockDim.x * blockIdx.x;
	long stride = (long)blockDim.x * gridDim.x;

	for (long i = start; i < V; i += stride) {

		long j = kept_at(mask, prefix, i);

		if (0 <= j)
			dst[j] = src[i];
	}
}

__global__ static void kern_scatter(long V, const unsigned int* mask, const int* prefix,
		cuFloatComplex* dst, const cuFloatComplex* src)
{
	long start = threadIdx.x + (long)blockDim.x * blockIdx.x;
	long stride = (long)blockDim.x * gridDim.x;

	for (long i = start; i < V; i += stride) {

		long j = kept_at(mask, prefix, i);

		dst[i] = (0 <= j) ? src[j] : make_cuFloatComplex(0.f, 0.f);
	}
}

static dim3 grid_for(long n)
{
	long blocks = (n + 255) / 256;

	return dim3((unsigned int)((blocks < 65535) ? blocks : 65535));
}

extern "C" void bartorch_cuda_gather(long V, const unsigned int* mask, const int* prefix,
		_Complex float* dst, const _Complex float* src)
{
	kern_gather<<<grid_for(V), 256, 0, cuda_get_stream()>>>(V, mask, prefix, (cuFloatComplex*)dst, (const cuFloatComplex*)src);

	CUDA_KERNEL_ERROR;
}

extern "C" void bartorch_cuda_scatter(long V, const unsigned int* mask, const int* prefix,
		_Complex float* dst, const _Complex float* src)
{
	kern_scatter<<<grid_for(V), 256, 0, cuda_get_stream()>>>(V, mask, prefix, (cuFloatComplex*)dst, (const cuFloatComplex*)src);

	CUDA_KERNEL_ERROR;
}

/* BART's inverse fftmod at index `j` of an axis `n` long: exp(-2 pi i r), with
 * r the fractional part of (j - c/2) c / n and c = n/2.  Kept in integers --
 * (2j - c) c modulo 2n -- so that the quarter turns BART writes out exactly
 * come out exact here too. */
__device__ static inline cuFloatComplex ifftmod_at(int n, int j)
{
	if (n <= 1)
		return make_cuFloatComplex(1.f, 0.f);

	/* On an axis a multiple of four long every factor is a sign: (-1)^j,
	 * negated once more where the axis is not a multiple of eight. */
	if (0 == n % 4)
		return make_cuFloatComplex((((j & 1) ? -1.f : 1.f) * ((0 == n % 8) ? 1.f : -1.f)), 0.f);

	int c = n / 2;
	int num = ((2 * j - c) * c) % (2 * n);

	if (num < 0)
		num += 2 * n;

	float si;
	float co;
	sincospif(-(float)num / (float)n, &si, &co);

	return make_cuFloatComplex(co, si);
}

/* `x` times `scale` and, along each of the first three axes, the inverse
 * fftmod of an axis `grid[a]` long at the array's own index plus `off[a]`:
 * what the centred transforms of a padded kernel put on around their
 * transforms.  Every axis after the third is a batch the same factor
 * multiplies. */
__global__ static void kern_modulate(unsigned int d0, unsigned int d1, unsigned int d2, long rest,
		int g0, int g1, int g2, int o0, int o1, int o2, float scale, cuFloatComplex* x)
{
	unsigned int vol = d0 * d1 * d2;
	unsigned int start = threadIdx.x + blockDim.x * blockIdx.x;
	unsigned int stride = blockDim.x * gridDim.x;

	for (unsigned int i = start; i < vol; i += stride) {

		unsigned int bc = i / d0;

		cuFloatComplex f = cuCmulf(ifftmod_at(g0, (int)(i - bc * d0) + o0),
				cuCmulf(ifftmod_at(g1, (int)(bc % d1) + o1), ifftmod_at(g2, (int)(bc / d1) + o2)));

		f = make_cuFloatComplex(scale * f.x, scale * f.y);

		for (long r = 0; r < rest; r++)
			x[i + r * vol] = cuCmulf(x[i + r * vol], f);
	}
}

extern "C" void bartorch_cuda_modulate(const long dims[3], long rest, const long grid[3], const long off[3],
		float scale, _Complex float* x)
{
	long vol = dims[0] * dims[1] * dims[2];

	kern_modulate<<<grid_for(vol), 256, 0, cuda_get_stream()>>>((unsigned int)dims[0], (unsigned int)dims[1],
			(unsigned int)dims[2], rest, (int)grid[0], (int)grid[1], (int)grid[2],
			(int)off[0], (int)off[1], (int)off[2], scale, (cuFloatComplex*)x);

	CUDA_KERNEL_ERROR;
}

/* A coil's gathered coefficients contracted, in place, against a real
 * function kept as the upper triangle of a symmetric matrix.
 *
 * At a kept location the coefficients meet only each other: they are read,
 * multiplied by the matrix, and written back over themselves, so there is no
 * second bank and nothing to clear or copy back.  Entry (i, j), i <= j, of the
 * matrix at location l is `mat[(i + j (j + 1) / 2) L + l]`, the order
 * `hermite_to_uppertriag` lays the function's entries out in. */
enum { CONTRACT_MAX = 16 };

template <typename P>
__global__ static void kern_contract_upper_real(long L, int R, cuFloatComplex* bank, const P* mat)
{
	long start = threadIdx.x + (long)blockDim.x * blockIdx.x;
	long stride = (long)blockDim.x * gridDim.x;

	for (long l = start; l < L; l += stride) {

		cuFloatComplex in[CONTRACT_MAX];

		for (int c = 0; c < R; c++)
			in[c] = bank[c * L + l];

		for (int r = 0; r < R; r++) {

			float re = 0.f;
			float im = 0.f;

			for (int c = 0; c < R; c++) {

				int lo = (r < c) ? r : c;
				int hi = (r < c) ? c : r;
				float m = widen(mat[(long)(lo + hi * (hi + 1) / 2) * L + l]);

				re += m * in[c].x;
				im += m * in[c].y;
			}

			bank[r * L + l] = make_cuFloatComplex(re, im);
		}
	}
}

extern "C" int bartorch_cuda_contract_upper_real(long L, int R, _Complex float* bank, const float* mat)
{
	if ((R < 1) || (R > CONTRACT_MAX))
		return -1;

	kern_contract_upper_real<float><<<grid_for(L), 256, 0, cuda_get_stream()>>>(L, R, (cuFloatComplex*)bank, mat);

	CUDA_KERNEL_ERROR;

	return 0;
}

/* The same, against a function kept in bfloat16. */
extern "C" int bartorch_cuda_contract_upper_real_bf16(long L, int R, _Complex float* bank, const void* mat)
{
	if ((R < 1) || (R > CONTRACT_MAX))
		return -1;

	kern_contract_upper_real<__nv_bfloat16><<<grid_for(L), 256, 0, cuda_get_stream()>>>(L, R, (cuFloatComplex*)bank, (const __nv_bfloat16*)mat);

	CUDA_KERNEL_ERROR;

	return 0;
}


/* A coil's gathered spectrum under a Cartesian normal, laid out with the
 * coefficients slowest, the batch next and the kept places fastest,
 * multiplied at each place by that place's coefficients-by-coefficients
 * kernel and written back over itself.  Entry (r, c) of the kernel at place l
 * is `K[(l R + r) R + c]`: the kernel varies over the transformed plane only,
 * so every batch reads the same one. */
__global__ static void kern_contract_grid(long L, long B, int R, cuFloatComplex* bank, const cuFloatComplex* K)
{
	long start = threadIdx.x + (long)blockDim.x * blockIdx.x;
	long stride = (long)blockDim.x * gridDim.x;
	long n = L * B;

	for (long i = start; i < n; i += stride) {

		long l = i % L;

		cuFloatComplex in[CONTRACT_MAX];

		for (int c = 0; c < R; c++)
			in[c] = bank[c * n + i];

		for (int r = 0; r < R; r++) {

			cuFloatComplex acc = make_cuFloatComplex(0.f, 0.f);

			for (int c = 0; c < R; c++)
				acc = cuCaddf(acc, cuCmulf(K[(l * R + r) * R + c], in[c]));

			bank[r * n + i] = acc;
		}
	}
}

extern "C" int bartorch_cuda_contract_grid(long L, long B, int R, _Complex float* bank, const _Complex float* K)
{
	if ((R < 1) || (R > CONTRACT_MAX))
		return -1;

	kern_contract_grid<<<grid_for(L * B), 256, 0, cuda_get_stream()>>>(L, B, R, (cuFloatComplex*)bank, (const cuFloatComplex*)K);

	CUDA_KERNEL_ERROR;

	return 0;
}
