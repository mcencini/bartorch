/*
 * The transform a Cartesian SENSE slab carries, with a normal of its own.
 *
 * Forward it is the centred unitary transform, a subspace basis contracting
 * the coefficients into the frames that were acquired, and the pattern that
 * keeps the samples taken -- the chain `grecon/model.c` builds, applied to a
 * slab of coils where the arithmetic is rather than to all of them on the
 * host.
 *
 * The normal is where the two differ.  Along an axis the pattern does not
 * vary, keeping the samples commutes with the transform, and the transform
 * and its inverse cancel: the readout of a Cartesian acquisition, and every
 * axis a pattern of phase encodes leaves alone.  So the normal transforms only
 * the axes the pattern varies along, and between the transforms it applies
 * one kernel,
 *
 *     K[k', k] = sum_t |P[t]|^2 conj(B[k', t]) B[k, t],
 *
 * over those axes alone: coefficients by coefficients at each place, however
 * many frames there are.  Without a basis it is the pattern's square; without
 * a pattern it is the basis's Gram, and nothing is transformed at all.
 *
 * On a card the normal runs inside cuFFT: per coil and coefficient, one
 * transform batched over the axes the pattern leaves alone, whose read puts
 * the sensitivity and the centring on and whose write gathers the kept
 * places; the kernel over the gathered places; and one transform back, whose
 * read scatters them and whose write takes the conjugates off into the
 * answer (fft_callbacks.cu).  Nothing the size of the coil images is made.
 * Where cuFFT cannot link the callbacks in, or off a card, the normal is
 * BART's chain of the same operators.
 */
#include <complex.h>
#include <math.h>
#include <stdbool.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/types.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/fft.h"
#include "num/gpuops.h"

#include "linops/fmac.h"
#include "linops/linop.h"
#include "linops/someops.h"

#include "sense/model.h"

#include "include/bartorch.h"

#ifdef USE_CUDA
struct bartorch_cb_grid;
extern struct bartorch_cb_grid* bartorch_cb_grid_create(const long dims[3], unsigned long flags, long kept,
		const unsigned int* mask, const int* prefix, const complex float* mod[3]);
extern void bartorch_cb_grid_free(struct bartorch_cb_grid* p);
extern void bartorch_cb_grid_forward(struct bartorch_cb_grid* p, complex float* bank, complex float* volume,
		const complex float* src, const complex float* map);
extern void bartorch_cb_grid_inverse(struct bartorch_cb_grid* p, complex float* dst, complex float* volume,
		const complex float* bank, const complex float* map);
extern int bartorch_cuda_contract_grid(long L, long B, int R, complex float* bank, const complex float* K);
#endif

static long grid_fused_count = 0;

long bartorch_grid_fused(void)
{
	return grid_fused_count;
}

struct grid_s {

	linop_data_t super;

	long cim_dims[DIMS];
	long out_dims[DIMS];

	const struct linop_s* fwd;	/* transform, basis, pattern */
	const struct linop_s* nrm;	/* the normal as BART's chain */

	/* The normal through cuFFT's callbacks: the axes it transforms, the
	 * places of a plane the pattern keeps, and the kernel at each of them
	 * (NULL where it is one), on the host. */
	unsigned long flags;
	long R;
	long vol;
	long plane;
	long batch;
	long kept;
	long words;
	unsigned int* mask;
	int* prefix;
	complex float* kernel;
	complex float* mod[3];

	/* The same on the card, made the first time the normal is asked for
	 * there; `cb` stays NULL where cuFFT cannot link the callbacks in. */
	bool tried;
	struct bartorch_cb_grid* cb;
	unsigned int* dmask;
	int* dprefix;
	complex float* dkernel;
	complex float* dmod[3];
};

static DEF_TYPEID(grid_s);

/* The spatial axes along which the pattern has more than one value. */
static unsigned long varying(const long pat_dims[DIMS])
{
	unsigned long flags = 0UL;

	for (int a = 0; a < 3; a++)
		if (1 < pat_dims[a])
			flags |= MD_BIT(a);

	return flags;
}

/* The kernel above, with k' on the TE axis and k on the COEFF axis -- the one
 * arrangement a single multiply-accumulate contracts -- and the pattern's
 * spatial axes, and one everywhere else.  Computed on the host: it is
 * coefficients squared over a plane of phase encodes. */
static complex float* grid_kernel(long kdims[DIMS], const long pat_dims[DIMS], const complex float* pattern,
		const long bas_dims[DIMS], const complex float* basis)
{
	long R = (NULL == basis) ? 1 : bas_dims[COEFF_DIM];
	long T = (NULL == basis) ? 1 : bas_dims[TE_DIM];
	long F = (NULL == pattern) ? 1 : pat_dims[TE_DIM];

	md_singleton_dims(DIMS, kdims);

	if (NULL != pattern)
		for (int a = 0; a < 3; a++)
			kdims[a] = pat_dims[a];

	kdims[TE_DIM] = R;
	kdims[COEFF_DIM] = R;

	long places = md_calc_size(3, kdims);

	complex float* P = NULL;

	if (NULL != pattern) {

		P = md_alloc(DIMS, pat_dims, CFL_SIZE);
		md_copy(DIMS, pat_dims, P, pattern, CFL_SIZE);
	}

	complex float* B = NULL;

	if (NULL != basis) {

		B = md_alloc(DIMS, bas_dims, CFL_SIZE);
		md_copy(DIMS, bas_dims, B, basis, CFL_SIZE);
	}

	/* conj(B[k', t]) B[k, t], once per frame. */
	complex float* G = xmalloc((size_t)(T * R * R) * sizeof(complex float));

	for (long t = 0; t < T; t++)
		for (long kp = 0; kp < R; kp++)
			for (long k = 0; k < R; k++)
				G[t * R * R + kp * R + k] = (NULL == B) ? 1.f : conjf(B[t + T * kp]) * B[t + T * k];

	complex float* K = md_alloc(DIMS, kdims, CFL_SIZE);

#pragma omp parallel for
	for (long p = 0; p < places; p++) {

		for (long kp = 0; kp < R; kp++)
			for (long k = 0; k < R; k++) {

				complex double acc = 0.;

				for (long t = 0; t < T; t++) {

					double w = 1.;

					if (NULL != P) {

						/* A pattern that is the same for every frame
						 * has one of them. */
						complex float v = P[p + places * ((1 == F) ? 0 : t)];
						w = (double)(crealf(v) * crealf(v) + cimagf(v) * cimagf(v));
					}

					acc += w * G[t * R * R + kp * R + k];
				}

				K[p + places * (kp + R * k)] = (complex float)acc;
			}
	}

	xfree(G);
	md_free(B);
	md_free(P);

	return K;
}

/* What the callbacks need, on the host: the places of a plane some frame
 * samples, as a bit per place and a count before each word, and the kernel at
 * those places alone, `kernel[(l R + r) R + c]` -- or no kernel, where every
 * kept place's is one. */
static void grid_compress(struct grid_s* d, const long kdims[DIMS], const complex float* K,
		const long pat_dims[DIMS], const complex float* pattern)
{
	long places = md_calc_size(3, kdims);
	long R = d->R;
	long F = pat_dims[TE_DIM];

	complex float* P = md_alloc(DIMS, pat_dims, CFL_SIZE);
	md_copy(DIMS, pat_dims, P, pattern, CFL_SIZE);

	d->words = (places + 31) / 32;
	d->mask = xmalloc((size_t)d->words * sizeof(unsigned int));
	d->prefix = xmalloc((size_t)d->words * sizeof(int));

	long n = 0;

	for (long w = 0; w < d->words; w++) {

		unsigned int bits = 0;

		for (long b = 0; (b < 32) && (w * 32 + b < places); b++) {

			bool taken = false;

			for (long t = 0; (t < F) && !taken; t++)
				taken = (0. != cabsf(P[w * 32 + b + places * t]));

			if (taken)
				bits |= 1u << b;
		}

		d->mask[w] = bits;
		d->prefix[w] = (int)n;
		n += __builtin_popcount(bits);
	}

	md_free(P);

	d->kept = n;
	d->plane = places;

	complex float* kernel = xmalloc((size_t)MAX(1, n * R * R) * sizeof(complex float));
	bool ones = (1 == R);

	long l = 0;

	for (long p = 0; p < places; p++) {

		if (0 == (d->mask[p >> 5] & (1u << (p & 31))))
			continue;

		for (long r = 0; r < R; r++)
			for (long c = 0; c < R; c++) {

				complex float v = K[p + places * (r + R * c)];

				kernel[(l * R + r) * R + c] = v;
				ones = ones && (1.f == v);
			}

		l++;
	}

	if (ones) {

		xfree(kernel);
		kernel = NULL;
	}

	d->kernel = kernel;

	/* The centring of each transformed axis, as `fftmod` applies it. */
	for (int a = 0; a < 3; a++) {

		d->mod[a] = NULL;

		if (!MD_IS_SET(d->flags, a))
			continue;

		long ad[1] = { d->cim_dims[a] };

		d->mod[a] = md_alloc(1, ad, CFL_SIZE);
		md_zfill(1, ad, d->mod[a], 1.);
		fftmod(1, ad, 1UL, d->mod[a], d->mod[a]);
	}
}

/* Whether the normal can run through cuFFT's callbacks on the card `ref` is
 * on, making what it needs there the first time it is asked. */
static bool grid_ready(struct grid_s* d, const void* ref)
{
	if (0UL == d->flags)
		return false;

#ifdef USE_CUDA
	if (!cuda_ondevice(ref))
		return false;

	if (!d->tried) {

		d->tried = true;

		long wd[1] = { d->words };

		d->dmask = md_alloc_gpu(1, wd, sizeof(unsigned int));
		d->dprefix = md_alloc_gpu(1, wd, sizeof(int));
		md_copy(1, wd, d->dmask, d->mask, sizeof(unsigned int));
		md_copy(1, wd, d->dprefix, d->prefix, sizeof(int));

		if (NULL != d->kernel) {

			long kd[1] = { d->kept * d->R * d->R };

			d->dkernel = md_alloc_gpu(1, kd, CFL_SIZE);
			md_copy(1, kd, d->dkernel, d->kernel, CFL_SIZE);
		}

		for (int a = 0; a < 3; a++) {

			if (NULL == d->mod[a])
				continue;

			long ad[1] = { d->cim_dims[a] };

			d->dmod[a] = md_alloc_gpu(1, ad, CFL_SIZE);
			md_copy(1, ad, d->dmod[a], d->mod[a], CFL_SIZE);
		}

		d->cb = bartorch_cb_grid_create(d->cim_dims, d->flags, d->kept,
				d->dmask, d->dprefix, (const complex float**)d->dmod);

		debug_printf(DP_DEBUG1, "Cartesian normal: %s\n",
				(NULL != d->cb) ? "through cuFFT's callbacks" : "as BART's chain");
	}

	return (NULL != d->cb);
#else
	(void)ref;
	return false;
#endif
}

/* The normal through the callbacks, added to `dst`: `coils` coils, a coil
 * `coil_step` elements after the one before in `src` and `dst`, a coefficient
 * `coeff_step` after the one before, and a coil's sensitivity `map_step`
 * after the one before in `map` (or no map). */
static void grid_fused(struct grid_s* d, complex float* dst, const complex float* src,
		long coils, long coil_step, long coeff_step, const complex float* map, long map_step)
{
#ifdef USE_CUDA
	long vd[1] = { d->vol };
	long bd[1] = { d->R * d->batch * MAX(1, d->kept) };

	complex float* volume = md_alloc_gpu(1, vd, CFL_SIZE);
	complex float* bank = md_alloc_gpu(1, bd, CFL_SIZE);

	long per = d->batch * d->kept;

	for (long c = 0; c < coils; c++) {

		const complex float* m = (NULL == map) ? NULL : map + c * map_step;

		for (long r = 0; r < d->R; r++)
			bartorch_cb_grid_forward(d->cb, bank + r * per, volume, src + c * coil_step + r * coeff_step, m);

		if (NULL != d->dkernel)
			bartorch_cuda_contract_grid(d->kept, d->batch, (int)d->R, bank, d->dkernel);

		for (long r = 0; r < d->R; r++)
			bartorch_cb_grid_inverse(d->cb, dst + c * coil_step + r * coeff_step, volume, bank + r * per, m);
	}

	md_free(bank);
	md_free(volume);

#pragma omp atomic
	grid_fused_count++;
#else
	(void)d; (void)dst; (void)src; (void)coils; (void)coil_step; (void)coeff_step; (void)map; (void)map_step;
	error("bartorch: the Cartesian callbacks run on a card\n");
#endif
}

static void grid_forward(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(grid_s, _d);

	linop_forward(d->fwd, DIMS, d->out_dims, dst, DIMS, d->cim_dims, src);
}

static void grid_adjoint(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(grid_s, _d);

	linop_adjoint(d->fwd, DIMS, d->cim_dims, dst, DIMS, d->out_dims, src);
}

static void grid_normal(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(grid_s, _d);

	if (grid_ready(d, dst) && cuda_ondevice(src)) {

		long coils = d->cim_dims[COIL_DIM];

		md_clear(DIMS, d->cim_dims, dst, CFL_SIZE);
		grid_fused(d, dst, src, coils, d->vol, d->vol * coils, NULL, 0);
		return;
	}

	linop_forward(d->nrm, DIMS, d->cim_dims, dst, DIMS, d->cim_dims, src);
}

static void grid_del(const linop_data_t* _d)
{
	const auto d = CAST_DOWN(grid_s, _d);

#ifdef USE_CUDA
	bartorch_cb_grid_free(d->cb);
#endif
	md_free(d->dmask);
	md_free(d->dprefix);
	md_free(d->dkernel);

	for (int a = 0; a < 3; a++) {

		md_free(d->dmod[a]);
		md_free(d->mod[a]);
	}

	xfree(d->mask);
	xfree(d->prefix);
	xfree(d->kernel);

	linop_free(d->nrm);
	linop_free(d->fwd);

	xfree(d);
}

/* Whether `op` is a Cartesian transform whose normal the coil loop can hand
 * a sensitivity to, on the card `ref` is on. */
int bartorch_grid_folds(const struct linop_s* op, const void* ref)
{
	struct grid_s* d = CAST_MAYBE(grid_s, linop_get_data(op));

	return ((NULL != d) && grid_ready(d, ref)) ? 1 : 0;
}

/* The normal of a slab of coils, each with its sensitivity from `map`,
 * against the image `src`, added to the image `dst`. */
void bartorch_grid_normal_sense(const struct linop_s* op, complex float* dst, const complex float* src,
		const long map_strs[DIMS], const complex float* map)
{
	struct grid_s* d = CAST_DOWN(grid_s, linop_get_data(op));

	grid_fused(d, dst, src, d->cim_dims[COIL_DIM], 0, d->vol, map, map_strs[COIL_DIM] / (long)CFL_SIZE);
}

/* The slab transform over coil images of `cim_dims`, or over every coil when
 * the loop does not run.  `pattern` and `basis` may each be NULL.  Without
 * `toeplitz` the normal is the two applications, as BART derives it. */
const struct linop_s* grid_transform_create(const long cim_dims[DIMS],
		const long pat_dims[DIMS], const complex float* pattern,
		const long bas_dims[DIMS], const complex float* basis, int toeplitz)
{
	unsigned long fft_flags = FFT_FLAGS & md_nontriv_dims(DIMS, cim_dims);

	if (NULL != pattern) {

		/* The kernel is laid out over the spatial axes and the frames and
		 * nothing else, so a pattern that differs between coils or sets
		 * of maps is not one this can collapse. */
		if (0UL != (md_nontriv_dims(DIMS, pat_dims) & ~(FFT_FLAGS | TE_FLAG)))
			error("bartorch: a Cartesian pattern varies along the spatial axes and the frames only\n");

		for (int a = 0; a < 3; a++)
			if ((1 != pat_dims[a]) && (pat_dims[a] != cim_dims[a]))
				error("bartorch: the pattern is %ld along axis %d, where the images are %ld\n",
						pat_dims[a], a, cim_dims[a]);

		long frames = (NULL == basis) ? 1 : bas_dims[TE_DIM];

		if ((1 != pat_dims[TE_DIM]) && (frames != pat_dims[TE_DIM]))
			error("bartorch: the pattern has %ld frames and the basis %ld\n", pat_dims[TE_DIM], frames);
	}

	PTR_ALLOC(struct grid_s, d);
	SET_TYPEID(grid_s, d);

	md_copy_dims(DIMS, d->cim_dims, cim_dims);
	md_copy_dims(DIMS, d->out_dims, cim_dims);

	struct linop_s* fwd = linop_fftc_create(DIMS, cim_dims, fft_flags);

	long R = 1;

	if (NULL != basis) {

		R = bas_dims[COEFF_DIM];

		if (R != cim_dims[COEFF_DIM])
			error("bartorch: the basis has %ld coefficients and the image %ld\n", R, cim_dims[COEFF_DIM]);

		d->out_dims[COEFF_DIM] = 1;
		d->out_dims[TE_DIM] = bas_dims[TE_DIM];

		fwd = linop_chain_FF(fwd, linop_fmac_dims_create(DIMS, d->out_dims, cim_dims, bas_dims, basis));
	}

	if (NULL != pattern)
		fwd = linop_chain_FF(fwd, linop_sampling_create(d->out_dims, pat_dims, pattern));

	unsigned long flags = (NULL == pattern) ? 0UL : (varying(pat_dims) & fft_flags);

	/* cuFFT links no callbacks into a transform along a single axis -- its
	 * plan answers CUFFT_INTERNAL_ERROR -- so a pattern that varies along one
	 * axis is transformed along the largest axis it is flat along as well.
	 * Along that axis the transform and its inverse cancel, and so do the
	 * centring and its conjugate, so the normal is the same one; the pattern
	 * the kernel and the kept places are read from is repeated along it. */
	long kpat_dims[DIMS];
	complex float* repeated = NULL;

	if (NULL != pattern)
		md_copy_dims(DIMS, kpat_dims, pat_dims);

	if ((NULL != pattern) && (1 == __builtin_popcountl(flags))) {

		int extra = -1;

		for (int a = 0; a < 3; a++)
			if (MD_IS_SET(fft_flags & ~flags, a) && ((-1 == extra) || (cim_dims[a] > cim_dims[extra])))
				extra = a;

		if (-1 != extra) {

			kpat_dims[extra] = cim_dims[extra];

			long istrs[DIMS];
			md_calc_strides(DIMS, istrs, pat_dims, CFL_SIZE);
			istrs[extra] = 0;

			long ostrs[DIMS];
			md_calc_strides(DIMS, ostrs, kpat_dims, CFL_SIZE);

			complex float* host = md_alloc(DIMS, pat_dims, CFL_SIZE);
			md_copy(DIMS, pat_dims, host, pattern, CFL_SIZE);

			repeated = md_alloc(DIMS, kpat_dims, CFL_SIZE);
			md_copy2(DIMS, kpat_dims, ostrs, repeated, istrs, host, CFL_SIZE);

			md_free(host);

			flags |= MD_BIT(extra);
		}
	}

	const complex float* kpattern = (NULL != repeated) ? repeated : pattern;

	long kdims[DIMS];
	complex float* K = grid_kernel(kdims, kpat_dims, kpattern, bas_dims, basis);

	struct linop_s* core = NULL;

	if (NULL == basis) {

		core = linop_cdiag_create(DIMS, cim_dims, md_nontriv_dims(DIMS, kdims), K);

	} else {

		/* The coefficients go out on TE, where the multiply-accumulate
		 * can put them, and a reshape reads them back on COEFF; the two
		 * layouts are the same memory. */
		long mixed_dims[DIMS];
		md_copy_dims(DIMS, mixed_dims, cim_dims);
		mixed_dims[COEFF_DIM] = 1;
		mixed_dims[TE_DIM] = R;

		core = linop_chain_FF(linop_fmac_dims_create(DIMS, mixed_dims, cim_dims, kdims, K),
				linop_reshape_create(DIMS, cim_dims, DIMS, mixed_dims));
	}

	d->flags = flags;
	d->R = R;
	d->vol = md_calc_size(3, cim_dims);
	d->plane = 1;
	d->batch = d->vol;
	d->kept = 0;
	d->words = 0;
	d->mask = NULL;
	d->prefix = NULL;
	d->kernel = NULL;
	d->tried = false;
	d->cb = NULL;
	d->dmask = NULL;
	d->dprefix = NULL;
	d->dkernel = NULL;

	for (int a = 0; a < 3; a++) {

		d->mod[a] = NULL;
		d->dmod[a] = NULL;
	}

	if (0 == toeplitz)
		d->flags = 0UL;

	struct linop_s* nrm = core;

	if (0UL != d->flags) {

		nrm = linop_chain_FF(linop_chain_FF(linop_fftc_create(DIMS, cim_dims, d->flags), core),
				linop_ifftc_create(DIMS, cim_dims, d->flags));

		grid_compress(d, kdims, K, kpat_dims, kpattern);
		d->batch = d->vol / d->plane;
	}

	md_free(K);
	md_free(repeated);

	d->fwd = fwd;
	d->nrm = nrm;

	debug_printf(DP_DEBUG1, "Cartesian slab: %ld coefficients, normal transforms axes %lx of %lx, %ld of %ld places kept\n",
			R, d->flags, fft_flags, d->kept, d->plane);

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, d->out_dims);

	return linop_create(DIMS, out_dims, DIMS, cim_dims, CAST_UP(PTR_PASS(d)),
			grid_forward, grid_adjoint, (0 != toeplitz) ? grid_normal : NULL, NULL, grid_del);
}
