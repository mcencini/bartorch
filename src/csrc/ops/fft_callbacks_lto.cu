/*
 * The passes around a volume's transform, as callbacks cuFFT runs while the
 * transform reads and writes.
 *
 * The forward transform reads a coefficient multiplied by the set's phase and
 * the coil's sensitivity, and writes only the places the samples reach, into
 * the gathered spectrum.  The inverse reads the gathered spectrum, with zeros
 * everywhere else, and adds what it writes, multiplied by the conjugates of
 * the phase and the sensitivity, into the answer.
 *
 * This file is compiled to LTO-IR alone and embedded in the library; cuFFT
 * links it into its kernels when a plan is made (fft_callbacks.cu).  cuFFT
 * declares a callback with C++ linkage, so these have it too.
 */
#include <cufftXt.h>

#include "coset.cuh"

/* The phase at offset `off`, found in 32-bit arithmetic: a 64-bit division per
 * point costs the inverse transform about 1.1 ms of 6.6 at 256^3, and a volume
 * has callbacks only if it has fewer than 2^32 points (fft_callbacks.cu). */
__device__ static inline cuFloatComplex phase_of(const struct coset_info* c, unsigned long long off, bool conj)
{
	unsigned int i = (unsigned int)off;
	unsigned int nx = (unsigned int)c->phase.dims[0];
	unsigned int ny = (unsigned int)c->phase.dims[1];
	unsigned int yz = i / nx;

	return phase_at(c->phase, (long)(i - yz * nx), (long)(yz % ny), (long)(yz / ny), conj);
}

__device__ cufftComplex bartorch_load_in(void* in, unsigned long long off, void* info, void* shared)
{
	const struct coset_info* c = (const struct coset_info*)info;

	cuFloatComplex w = phase_of(c, off, false);

	if (NULL != c->map)
		w = cuCmulf(c->map[off], w);

	return cuCmulf(((const cufftComplex*)in)[off], w);
}

__device__ void bartorch_store_gather(void* out, unsigned long long off, cufftComplex val, void* info, void* shared)
{
	const struct coset_info* c = (const struct coset_info*)info;

	long j = kept_at(c->mask, c->prefix, (long)off);

	if (0 <= j)
		c->bank[j] = val;
}

__device__ cufftComplex bartorch_load_scatter(void* in, unsigned long long off, void* info, void* shared)
{
	const struct coset_info* c = (const struct coset_info*)info;

	long j = kept_at(c->mask, c->prefix, (long)off);

	return (0 <= j) ? c->bank[j] : make_cuFloatComplex(0.f, 0.f);
}

__device__ void bartorch_store_out(void* out, unsigned long long off, cufftComplex val, void* info, void* shared)
{
	const struct coset_info* c = (const struct coset_info*)info;

	cuFloatComplex w = phase_of(c, off, true);

	if (NULL != c->map)
		w = cuCmulf(cuConjf(c->map[off]), w);

	c->dst[off] = cuCaddf(c->dst[off], cuCmulf(val, w));
}
