/*
 * What the callbacks of a Cartesian normal read (fft_callbacks_lto.cu).
 *
 * cuFFT transforms the axes the pattern varies along and batches over the
 * rest, laying its buffer out with the batch slowest.  The image is laid out
 * the way BART lays it out, with the readout fastest.  So an offset into
 * cuFFT's buffer is taken apart into a place in the transformed plane and a
 * batch, and those into coordinates, which give the image's index -- and
 * nothing is copied into cuFFT's layout or out of it.
 */
#ifndef BARTORCH_GRID_CUH
#define BARTORCH_GRID_CUH

#include <cuComplex.h>

#include "coset.cuh"

struct grid_info {

	unsigned int plane;		/* places in one transformed plane */
	unsigned int L;			/* the places of a plane the pattern keeps */
	unsigned int n[3];		/* the image's spatial axes */
	unsigned int pstr[3];		/* an axis's stride in the plane, 0 if it is not transformed */
	unsigned int bstr[3];		/* an axis's stride in the batch, 0 if it is transformed or one */
	unsigned int mstr[3];		/* an axis's stride in the image */
	float scale;			/* the unitary scale of one direction */
	const cuFloatComplex* mod[3];	/* the centring of a transformed axis, NULL otherwise */

	const cuFloatComplex* map;	/* the coil's sensitivity, NULL where there is none */
	const cuFloatComplex* src;	/* the coefficient going in */
	cuFloatComplex* dst;		/* where the coefficient coming out is added */
	cuFloatComplex* bank;		/* the coefficient's gathered spectrum: batch x L */
	const unsigned int* mask;	/* a bit per place of the plane */
	const int* prefix;
};

/* The image index of cuFFT offset `off`, and the centring there. */
__device__ static inline unsigned int grid_index(const struct grid_info* g, unsigned int off, cuFloatComplex* mod)
{
	unsigned int batch = off / g->plane;
	unsigned int place = off - batch * g->plane;
	unsigned int idx = 0;

	cuFloatComplex m = make_cuFloatComplex(g->scale, 0.f);

	for (int a = 0; a < 3; a++) {

		unsigned int c = 0;

		if (0 != g->pstr[a]) {

			c = (place / g->pstr[a]) % g->n[a];
			m = cuCmulf(m, g->mod[a][c]);

		} else if (0 != g->bstr[a]) {

			c = (batch / g->bstr[a]) % g->n[a];
		}

		idx += c * g->mstr[a];
	}

	*mod = m;

	return idx;
}

/* Where cuFFT offset `off` sits in the gathered spectrum, or -1 where the
 * pattern does not keep its place. */
__device__ static inline long grid_kept(const struct grid_info* g, unsigned int off)
{
	unsigned int batch = off / g->plane;
	unsigned int place = off - batch * g->plane;

	long j = kept_at(g->mask, g->prefix, (long)place);

	return (0 > j) ? -1 : (long)batch * g->L + j;
}

#endif
