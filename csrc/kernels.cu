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

/* A coil's gathered coefficients contracted, in place, against a real
 * function kept as the upper triangle of a symmetric matrix.
 *
 * At a kept location the coefficients meet only each other: they are read,
 * multiplied by the matrix, and written back over themselves, so there is no
 * second bank and nothing to clear or copy back.  Entry (i, j), i <= j, of the
 * matrix at location l is `mat[(i + j (j + 1) / 2) L + l]`, the order
 * `hermite_to_uppertriag` lays the function's entries out in. */
enum { CONTRACT_MAX = 16 };

__global__ static void kern_contract_upper_real(long L, int R, cuFloatComplex* bank, const float* mat)
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
				float m = mat[(long)(lo + hi * (hi + 1) / 2) * L + l];

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

	kern_contract_upper_real<<<grid_for(L), 256, 0, cuda_get_stream()>>>(L, R, (cuFloatComplex*)bank, mat);

	CUDA_KERNEL_ERROR;

	return 0;
}
