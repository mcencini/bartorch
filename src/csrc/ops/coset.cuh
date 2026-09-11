/*
 * The arithmetic of a streamed set, shared by the passes in kernels.cu and by
 * the callbacks cuFFT runs inside its transforms (fft_callbacks_lto.cu), so
 * both apply the same numbers.
 *
 * The phase is the one `cuda_apply_linphases_3D` computes -- the same shift,
 * the same centring, the same fftmod folded in and the same scale -- so a
 * coefficient that goes through either goes through what BART's precomputed
 * phases would have applied.
 *
 * A kept point's place in the gathered spectrum is read off two arrays: a bit
 * per grid point, set where the samples reach, and for each 32-bit word of
 * bits the number of kept points before it.  That is two bits a grid point,
 * where a list of positions costs thirty-two a kept point.
 */
#ifndef BARTORCH_COSET_CUH
#define BARTORCH_COSET_CUH

#include <cuComplex.h>
#include <math.h>
#include <stdbool.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

struct phase_conf {

	long dims[3];
	long tot;
	long batch;
	float shifts[3];
	float cn;
	float scale;
};

static inline struct phase_conf phase_setup(int N, const long dims[], const float shift[3], float scale)
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

/* Where grid point `i` sits in the gathered spectrum, or -1 where it is not kept. */
__device__ static inline long kept_at(const unsigned int* mask, const int* prefix, long i)
{
	unsigned int word = mask[i >> 5];
	unsigned int bit = 1u << (i & 31);

	if (0 == (word & bit))
		return -1;

	return (long)prefix[i >> 5] + __popc(word & (bit - 1));
}

/* What the callbacks of one transform read: a set's phase, a coil's
 * sensitivity, a coefficient's gathered spectrum and, for the inverse, the
 * image it adds into. */
struct coset_info {

	struct phase_conf phase;
	const cuFloatComplex* map;	/* NULL where there is no sensitivity */
	cuFloatComplex* dst;
	cuFloatComplex* bank;
	const unsigned int* mask;
	const int* prefix;
};

#endif
