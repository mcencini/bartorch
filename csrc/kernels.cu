/*
 * Kernels of the streamed convolution that BART has no single pass for.
 *
 * A coefficient goes into the transform multiplied by the set's linear phase
 * and by the coil's sensitivity, and comes out of it multiplied by the
 * conjugates of both and added to the answer.  BART applies the phase in one
 * pass and the sensitivity in another, and each pass over a 256^3 volume
 * reads and writes it whole.  These do both in one: three streams over the
 * volume on the way in, four on the way out.
 *
 * The phase is the one `cuda_apply_linphases_3D` computes -- the same shift,
 * the same centring, the same fftmod folded in and the same scale -- so a
 * coefficient that goes through here goes through what BART's precomputed
 * phases would have applied.
 */
#include <cuda_runtime_api.h>
#include <cuComplex.h>
#include <math.h>
#include <stdbool.h>

#include "misc/misc.h"

#include "num/gpukrnls_misc.h"
#include "num/gpuops.h"

struct phase_conf {

	long dims[3];
	long tot;
	long batch;
	float shifts[3];
	float cn;
	float scale;
};

static struct phase_conf phase_setup(int N, const long dims[], const float shift[3], float scale)
{
	struct phase_conf c;

	c.cn = 0.f;
	c.tot = 1;
	c.scale = scale;

	for (int n = 0; n < 3; n++) {

		float s = shift[n];

		if (1 < dims[n])
			s += (float)(dims[n] / 2. - dims[n] / 2);

		c.shifts[n] = 2. * M_PI * s / (float)dims[n];
		c.cn -= c.shifts[n] * (float)dims[n] / 2.f;

		c.dims[n] = dims[n];
		c.tot *= dims[n];

		long centre = dims[n] / 2;
		double half = (double)centre / (double)dims[n];

		c.shifts[n] += 2. * M_PI * half;
		c.cn -= 2. * M_PI * half * (double)centre / 2.;
	}

	c.batch = 1;

	for (int n = 3; n < N; n++)
		c.batch *= dims[n];

	return c;
}

__device__ static inline cuFloatComplex phase_at(const struct phase_conf& c, long x, long y, long z, bool conj)
{
	float val = c.cn + x * c.shifts[0] + y * c.shifts[1] + z * c.shifts[2];

	if (conj)
		val = -val;

	float si;
	float co;
	sincosf(val, &si, &co);

	return make_cuFloatComplex(c.scale * co, c.scale * si);
}

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

/* Gather and scatter over the places the samples reach.
 *
 * `kept` lists the grid positions that are kept, in grid order, one int per
 * kept point -- against one long per grid point for the map BART gathers
 * with, which a gather reads whole.  A scatter puts zeros everywhere else,
 * which is what the transform that follows it needs to see there.
 */
__global__ static void kern_gather(long L, const int* kept, cuFloatComplex* dst, const cuFloatComplex* src)
{
	long start = threadIdx.x + (long)blockDim.x * blockIdx.x;
	long stride = (long)blockDim.x * gridDim.x;

	for (long j = start; j < L; j += stride)
		dst[j] = src[kept[j]];
}

__global__ static void kern_scatter(long L, const int* kept, cuFloatComplex* dst, const cuFloatComplex* src)
{
	long start = threadIdx.x + (long)blockDim.x * blockIdx.x;
	long stride = (long)blockDim.x * gridDim.x;

	for (long j = start; j < L; j += stride)
		dst[kept[j]] = src[j];
}

static dim3 grid_for(long n)
{
	long blocks = (n + 255) / 256;

	return dim3((unsigned int)((blocks < 65535) ? blocks : 65535));
}

extern "C" void bartorch_cuda_gather(long L, const int* kept, _Complex float* dst, const _Complex float* src)
{
	kern_gather<<<grid_for(L), 256, 0, cuda_get_stream()>>>(L, kept, (cuFloatComplex*)dst, (const cuFloatComplex*)src);

	CUDA_KERNEL_ERROR;
}

extern "C" void bartorch_cuda_scatter(long V, long L, const int* kept, _Complex float* dst, const _Complex float* src)
{
	CUDA_ERROR(cudaMemsetAsync(dst, 0, (size_t)V * sizeof(cuFloatComplex), cuda_get_stream()));

	kern_scatter<<<grid_for(L), 256, 0, cuda_get_stream()>>>(L, kept, (cuFloatComplex*)dst, (const cuFloatComplex*)src);

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
