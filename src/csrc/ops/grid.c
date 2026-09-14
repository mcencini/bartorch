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
 *
 * The wave encoding is this grid over (oversampled readout, y, z): the coil
 * images are zero-filled along the readout, transformed along it and
 * multiplied by the point spread function, which the callbacks take where a
 * sensitivity would go.  That transform may be BART's uncentred one, as
 * `wave.c` uses, which is unnormalized with its adjoint as the inverse.
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
		const unsigned int* mask, const int* prefix, const complex float* mod[3], int unitary);
extern void bartorch_cuda_pad_map(long sx, long wx, long rest, long off,
		complex float* dst, const complex float* src, const complex float* map);
extern void bartorch_cuda_crop_mapc_add(long sx, long wx, long rest, long off,
		complex float* dst, const complex float* src, const complex float* map);
extern void bartorch_cb_grid_free(struct bartorch_cb_grid* p);
extern void bartorch_cb_grid_forward(struct bartorch_cb_grid* p, complex float* bank, complex float* volume,
		const complex float* src, const complex float* map);
extern void bartorch_cb_grid_inverse(struct bartorch_cb_grid* p, complex float* dst, complex float* volume,
		const complex float* bank, const complex float* map);
extern int bartorch_cuda_contract_grid(long L, long B, int R, complex float* bank, const complex float* K);

/* What the kernels between the gathered spectrum and a table of samples read
 * of the transform's layout (kernels.cu). */
struct bartorch_grid_axes {

	unsigned int pstr[3];
	unsigned int bstr[3];
	const complex float* mod[3];
};

extern void bartorch_cuda_bank_to_table(long E, long X, long S, int R, long L, long per,
		const int* entry_u, const int* u_coord, const unsigned int* mask, const int* prefix,
		const struct bartorch_grid_axes* ax, const complex float* B, const complex float* bank, complex float* table);
extern void bartorch_cuda_table_to_bank(long U, long X, long S, int R, long L, long per,
		const int* csr_start, const int* csr, const int* u_coord, const unsigned int* mask, const int* prefix,
		const struct bartorch_grid_axes* ax, const complex float* B, complex float* bank, const complex float* table);
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

	/* Whether the normal is the closed form or the two applications. */
	bool toeplitz;

	/* Centred and unitary, as BART's fftc is; otherwise BART's fft,
	 * unnormalized with its adjoint as the inverse. */
	bool centred;

	/* A wave (wave_transform_create): the domain is coil images of
	 * `dom_dims`, zero-filled along the readout `wave_off` from its start to
	 * the `cim_dims` the grid works in, transformed along it and multiplied
	 * by `psf` (over the spatial axes of `cim_dims`).  Without a wave
	 * `dom_dims` is `cim_dims` and `img_vol` is `vol`. */
	bool wave;
	long dom_dims[DIMS];
	long img_vol;
	long wave_off;
	complex float* psf;
	complex float* dpsf;

	/* Sampled-only samples (grid_sampled_create): `E` = `T` frames x `S`
	 * shots, each a phase-encode place with the readout `X` along it, and
	 * `U` distinct places among them.  `entry_u` is an entry's place, -1 for
	 * padding; `u_coord` a place's (z, y); `csr` the entries of each place,
	 * from `csr_start`.  `bt` is the basis as B[t R + r], NULL without one.
	 * `kspace_readout` says the samples are in k-space along the readout. */
	bool sampled;
	bool kspace_readout;
	long T;
	long S;
	long X;
	long E;
	long U;
	int* entry_u;
	int* u_coord;
	int* csr_start;
	int* csr;
	complex float* bt;
	int* d_entry_u;
	int* d_u_coord;
	int* d_csr_start;
	int* d_csr;
	complex float* d_bt;
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

	/* The centring of each transformed axis, as `fftmod` applies it; ones for
	 * the uncentred transform. */
	for (int a = 0; a < 3; a++) {

		d->mod[a] = NULL;

		if (!MD_IS_SET(d->flags, a))
			continue;

		long ad[1] = { d->cim_dims[a] };

		d->mod[a] = md_alloc(1, ad, CFL_SIZE);
		md_zfill(1, ad, d->mod[a], 1.);

		if (d->centred)
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

		if (NULL != d->psf) {

			long pd[1] = { d->vol };

			d->dpsf = md_alloc_gpu(1, pd, CFL_SIZE);
			md_copy(1, pd, d->dpsf, d->psf, CFL_SIZE);
		}

		if (d->sampled) {

			long ed[1] = { MAX(1, d->E) };
			long ud[1] = { MAX(1, 2 * d->U) };
			long sd[1] = { d->U + 1 };
			long cd[1] = { MAX(1, d->csr_start[d->U]) };

			d->d_entry_u = md_alloc_gpu(1, ed, sizeof(int));
			d->d_u_coord = md_alloc_gpu(1, ud, sizeof(int));
			d->d_csr_start = md_alloc_gpu(1, sd, sizeof(int));
			d->d_csr = md_alloc_gpu(1, cd, sizeof(int));
			md_copy(1, ed, d->d_entry_u, d->entry_u, sizeof(int));
			md_copy(1, ud, d->d_u_coord, d->u_coord, sizeof(int));
			md_copy(1, sd, d->d_csr_start, d->csr_start, sizeof(int));
			md_copy(1, cd, d->d_csr, d->csr, sizeof(int));

			if (NULL != d->bt) {

				long bd[1] = { d->T * d->R };

				d->d_bt = md_alloc_gpu(1, bd, CFL_SIZE);
				md_copy(1, bd, d->d_bt, d->bt, CFL_SIZE);
			}
		}

		d->cb = bartorch_cb_grid_create(d->cim_dims, d->flags, d->kept,
				d->dmask, d->dprefix, (const complex float**)d->dmod, (int)d->centred);

		debug_printf(DP_DEBUG1, "Cartesian normal: %s\n",
				(NULL != d->cb) ? "through cuFFT's callbacks" : "as BART's chain");
	}

	return (NULL != d->cb);
#else
	(void)ref;
	return false;
#endif
}

/* BART's transform along `flags` in place, in the grid's convention, or its
 * adjoint: fftuc and ifftuc centred, fft and ifft otherwise. */
static void grid_ft(const struct grid_s* d, const long dims[DIMS], unsigned long flags, complex float* x, bool adjoint)
{
	if (0UL == flags)
		return;

	if (d->centred)
		(adjoint ? ifftuc : fftuc)(DIMS, dims, flags, x, x);
	else
		(adjoint ? ifft : fft)(DIMS, dims, flags, x, x);
}

#ifdef USE_CUDA
/* One coefficient, times the sensitivity `map`, into its gathered spectrum:
 * through the callbacks, or for a wave through the front first, the point
 * spread function being the callbacks' map. */
static void grid_in(struct grid_s* d, complex float* bank, complex float* volume, complex float* hybrid,
		const complex float* src, const complex float* map)
{
	if (!d->wave) {

		bartorch_cb_grid_forward(d->cb, bank, volume, src, map);
		return;
	}

	long wx = d->cim_dims[READ_DIM];

	long hdims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, hdims, d->cim_dims);

	bartorch_cuda_pad_map(d->dom_dims[READ_DIM], wx, d->vol / wx, d->wave_off, hybrid, src, map);
	grid_ft(d, hdims, READ_FLAG, hybrid, false);
	bartorch_cb_grid_forward(d->cb, bank, volume, hybrid, d->dpsf);
}

/* The gathered spectrum back, times the conjugates, added to `dst`. */
static void grid_out(struct grid_s* d, complex float* dst, complex float* volume, complex float* hybrid,
		const complex float* bank, const complex float* map)
{
	if (!d->wave) {

		bartorch_cb_grid_inverse(d->cb, dst, volume, bank, map);
		return;
	}

	long wx = d->cim_dims[READ_DIM];

	long hdims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, hdims, d->cim_dims);

	md_clear(DIMS, hdims, hybrid, CFL_SIZE);
	bartorch_cb_grid_inverse(d->cb, hybrid, volume, bank, d->dpsf);
	grid_ft(d, hdims, READ_FLAG, hybrid, true);
	bartorch_cuda_crop_mapc_add(d->dom_dims[READ_DIM], wx, d->vol / wx, d->wave_off, dst, hybrid, map);
}
#endif

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
	complex float* hybrid = d->wave ? md_alloc_gpu(1, vd, CFL_SIZE) : NULL;

	long per = d->batch * d->kept;

	for (long c = 0; c < coils; c++) {

		const complex float* m = (NULL == map) ? NULL : map + c * map_step;

		for (long r = 0; r < d->R; r++)
			grid_in(d, bank + r * per, volume, hybrid, src + c * coil_step + r * coeff_step, m);

		if (NULL != d->dkernel)
			bartorch_cuda_contract_grid(d->kept, d->batch, (int)d->R, bank, d->dkernel);

		for (long r = 0; r < d->R; r++)
			grid_out(d, dst + c * coil_step + r * coeff_step, volume, hybrid, bank + r * per, m);
	}

	md_free(bank);
	md_free(volume);

	if (NULL != hybrid)
		md_free(hybrid);

#pragma omp atomic
	grid_fused_count++;
#else
	(void)d; (void)dst; (void)src; (void)coils; (void)coil_step; (void)coeff_step; (void)map; (void)map_step;
	error("bartorch: the Cartesian callbacks run on a card\n");
#endif
}

/* The axes a sampled table's spectrum is transformed along.  A wave's readout
 * has been transformed by its front already, so the grid never transforms it
 * again for a table -- where cuFFT's callbacks would (a plane of one axis
 * takes the readout along), the table is served on the host instead. */
static unsigned long sampled_flags(const struct grid_s* d)
{
	return d->wave ? (d->flags & ~READ_FLAG) : d->flags;
}

static bool sampled_fusable(const struct grid_s* d)
{
	return !(d->wave && MD_IS_SET(d->flags, READ_DIM));
}

/* On the host: coil images into the grid's domain -- for a wave zero-filled
 * along the readout, transformed along it and multiplied by the point spread
 * function -- and the adjoint back.  Without a wave both are copies. */
static void wave_psf_host(const struct grid_s* d, complex float* k, bool conj)
{
	long pdims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, pdims, d->cim_dims);

	long kstr[DIMS];
	md_calc_strides(DIMS, kstr, d->cim_dims, CFL_SIZE);

	long pstr[DIMS];
	md_calc_strides(DIMS, pstr, pdims, CFL_SIZE);

	for (int i = 0; i < DIMS; i++)
		if (1 == pdims[i])
			pstr[i] = 0;

	(conj ? md_zmulc2 : md_zmul2)(DIMS, d->cim_dims, kstr, k, kstr, k, pstr, d->psf);
}

static void wave_front_host(const struct grid_s* d, complex float* k, const complex float* src)
{
	if (!d->wave) {

		md_copy(DIMS, d->cim_dims, k, src, CFL_SIZE);
		return;
	}

	complex float* image = md_alloc(DIMS, d->dom_dims, CFL_SIZE);
	md_copy(DIMS, d->dom_dims, image, src, CFL_SIZE);
	md_resize_center(DIMS, d->cim_dims, k, d->dom_dims, image, CFL_SIZE);
	md_free(image);

	grid_ft(d, d->cim_dims, READ_FLAG, k, false);
	wave_psf_host(d, k, false);
}

static void wave_back_host(const struct grid_s* d, complex float* dst, complex float* k)
{
	if (!d->wave) {

		md_copy(DIMS, d->cim_dims, dst, k, CFL_SIZE);
		return;
	}

	wave_psf_host(d, k, true);
	grid_ft(d, d->cim_dims, READ_FLAG, k, true);

	complex float* image = md_alloc(DIMS, d->dom_dims, CFL_SIZE);
	md_resize_center(DIMS, d->dom_dims, image, d->cim_dims, k, CFL_SIZE);
	md_copy(DIMS, d->dom_dims, dst, image, CFL_SIZE);
	md_free(image);
}

/* The table's readout as the caller wants it.  The spectrum leaves the readout
 * as the image has it unless the plane took it along (a transform along one
 * axis alone takes a second), so the table is transformed along its samples
 * where the two differ: into k-space forward, out of it for the adjoint, and
 * the other way round for samples in image space along the readout. */
static void sampled_readout(const struct grid_s* d, const long dims[DIMS], complex float* table, bool forward)
{
	bool transformed = (0 != MD_IS_SET(sampled_flags(d), READ_DIM));

	if (d->kspace_readout == transformed)
		return;

	if (d->kspace_readout == forward)
		fftuc(DIMS, dims, PHS1_FLAG, table, table);
	else
		ifftuc(DIMS, dims, PHS1_FLAG, table, table);
}

/* The table of a spectrum `k` over the coil images: each entry's place and
 * readout, contracted with the basis for its frame. */
static void sampled_gather(const struct grid_s* d, complex float* out, const complex float* k)
{
	long cstr[DIMS];
	md_calc_strides(DIMS, cstr, d->cim_dims, 1);

	long ostr[DIMS];
	md_calc_strides(DIMS, ostr, d->out_dims, 1);

	long C = d->cim_dims[COIL_DIM];

#pragma omp parallel for collapse(2)
	for (long c = 0; c < C; c++) {
		for (long e = 0; e < d->E; e++) {

			long t = e / d->S;
			long u = d->entry_u[e];
			long base = (e - t * d->S) * ostr[PHS2_DIM] + c * ostr[COIL_DIM] + t * ostr[TE_DIM];

			for (long x = 0; x < d->X; x++) {

				complex float acc = 0.;

				if (0 <= u) {

					long at = x * cstr[READ_DIM] + d->u_coord[2 * u + 1] * cstr[PHS1_DIM]
						+ d->u_coord[2 * u] * cstr[PHS2_DIM] + c * cstr[COIL_DIM];

					for (long r = 0; r < d->R; r++)
						acc += ((NULL == d->bt) ? 1.f : d->bt[t * d->R + r]) * k[at + r * cstr[COEFF_DIM]];
				}

				out[base + x * ostr[PHS1_DIM]] = acc;
			}
		}
	}
}

/* Its adjoint: every place of the spectrum a table reaches, from the entries
 * of that place, and zeros everywhere else. */
static void sampled_scatter(const struct grid_s* d, complex float* k, const complex float* out)
{
	md_clear(DIMS, d->cim_dims, k, CFL_SIZE);

	long cstr[DIMS];
	md_calc_strides(DIMS, cstr, d->cim_dims, 1);

	long ostr[DIMS];
	md_calc_strides(DIMS, ostr, d->out_dims, 1);

	long C = d->cim_dims[COIL_DIM];

#pragma omp parallel for collapse(2)
	for (long c = 0; c < C; c++) {
		for (long u = 0; u < d->U; u++) {

			for (long x = 0; x < d->X; x++) {

				long at = x * cstr[READ_DIM] + d->u_coord[2 * u + 1] * cstr[PHS1_DIM]
					+ d->u_coord[2 * u] * cstr[PHS2_DIM] + c * cstr[COIL_DIM];

				for (long r = 0; r < d->R; r++) {

					complex float acc = 0.;

					for (long q = d->csr_start[u]; q < d->csr_start[u + 1]; q++) {

						long e = d->csr[q];
						long t = e / d->S;
						complex float v = out[x * ostr[PHS1_DIM] + (e - t * d->S) * ostr[PHS2_DIM]
							+ c * ostr[COIL_DIM] + t * ostr[TE_DIM]];

						acc += ((NULL == d->bt) ? 1.f : conjf(d->bt[t * d->R + r])) * v;
					}

					k[at + r * cstr[COEFF_DIM]] = acc;
				}
			}
		}
	}
}

/* On the host: the front, BART's transform over the axes the table's
 * spectrum takes, and the table read off it. */
static void sampled_host_forward(const struct grid_s* d, complex float* dst, const complex float* src)
{
	complex float* k = md_alloc(DIMS, d->cim_dims, CFL_SIZE);
	wave_front_host(d, k, src);
	grid_ft(d, d->cim_dims, sampled_flags(d), k, false);

	complex float* out = md_alloc(DIMS, d->out_dims, CFL_SIZE);
	sampled_gather(d, out, k);
	md_free(k);

	sampled_readout(d, d->out_dims, out, true);
	md_copy(DIMS, d->out_dims, dst, out, CFL_SIZE);
	md_free(out);
}

static void sampled_host_adjoint(const struct grid_s* d, complex float* dst, const complex float* src)
{
	complex float* out = md_alloc(DIMS, d->out_dims, CFL_SIZE);
	md_copy(DIMS, d->out_dims, out, src, CFL_SIZE);
	sampled_readout(d, d->out_dims, out, false);

	complex float* k = md_alloc(DIMS, d->cim_dims, CFL_SIZE);
	sampled_scatter(d, k, out);
	md_free(out);

	grid_ft(d, d->cim_dims, sampled_flags(d), k, true);
	wave_back_host(d, dst, k);
	md_free(k);
}

#ifdef USE_CUDA
/* The plane and batch strides as cuFFT's callbacks lay them out
 * (fft_callbacks.cu), with the centring on the card. */
static void grid_axes(const struct grid_s* d, struct bartorch_grid_axes* ax)
{
	unsigned int ps = 1;
	unsigned int bs = 1;

	for (int a = 0; a < 3; a++) {

		bool t = MD_IS_SET(d->flags, a) && (1 < d->cim_dims[a]);

		ax->pstr[a] = t ? ps : 0;
		ax->bstr[a] = (!t && (1 < d->cim_dims[a])) ? bs : 0;
		ax->mod[a] = t ? d->dmod[a] : NULL;

		if (t)
			ps *= (unsigned int)d->cim_dims[a];
		else
			bs *= (unsigned int)d->cim_dims[a];
	}
}

/* The table of `coils` coils, a coil `coil_step` elements after the one before
 * in `src`, a coefficient `coeff_step` after the one before, and a coil's
 * sensitivity `map_step` after the one before in `map` (or no map): per coil
 * and coefficient one transform whose read puts the sensitivity and the
 * centring on and whose write gathers the kept places, then the table read
 * off the gathered spectrum on the card. */
static void sampled_fused_forward(struct grid_s* d, complex float* dst, const complex float* src,
		long coils, long coil_step, long coeff_step, const complex float* map, long map_step)
{
	long vd[1] = { d->vol };
	long bd[1] = { d->R * d->batch * MAX(1, d->kept) };

	long td[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, td, d->out_dims);

	long tstr[DIMS];
	md_calc_strides(DIMS, tstr, td, CFL_SIZE);

	long ostr[DIMS];
	md_calc_strides(DIMS, ostr, d->out_dims, CFL_SIZE);

	complex float* volume = md_alloc_gpu(1, vd, CFL_SIZE);
	complex float* bank = md_alloc_gpu(1, bd, CFL_SIZE);
	complex float* hybrid = d->wave ? md_alloc_gpu(1, vd, CFL_SIZE) : NULL;
	complex float* table = md_alloc_gpu(DIMS, td, CFL_SIZE);

	struct bartorch_grid_axes ax;
	grid_axes(d, &ax);

	long per = d->batch * d->kept;

	for (long c = 0; c < coils; c++) {

		const complex float* m = (NULL == map) ? NULL : map + c * map_step;

		for (long r = 0; r < d->R; r++)
			grid_in(d, bank + r * per, volume, hybrid, src + c * coil_step + r * coeff_step, m);

		bartorch_cuda_bank_to_table(d->E, d->X, d->S, (int)d->R, d->kept, per, d->d_entry_u, d->d_u_coord,
				d->dmask, d->dprefix, &ax, d->d_bt, bank, table);

		sampled_readout(d, td, table, true);

		md_copy2(DIMS, td, ostr, dst + c * (ostr[COIL_DIM] / (long)CFL_SIZE), tstr, table, CFL_SIZE);
	}

	md_free(table);
	md_free(bank);
	md_free(volume);

	if (NULL != hybrid)
		md_free(hybrid);

#pragma omp atomic
	grid_fused_count++;
}

/* The adjoint, added to `dst`: per coil the table scattered into the gathered
 * spectrum on the card, and per coefficient one transform back whose read
 * scatters it and whose write takes the conjugates off into the image. */
static void sampled_fused_adjoint(struct grid_s* d, complex float* dst, const complex float* src,
		long coils, long coil_step, long coeff_step, const complex float* map, long map_step)
{
	long vd[1] = { d->vol };
	long bd[1] = { d->R * d->batch * MAX(1, d->kept) };

	long td[DIMS];
	md_select_dims(DIMS, ~COIL_FLAG, td, d->out_dims);

	long tstr[DIMS];
	md_calc_strides(DIMS, tstr, td, CFL_SIZE);

	long ostr[DIMS];
	md_calc_strides(DIMS, ostr, d->out_dims, CFL_SIZE);

	complex float* volume = md_alloc_gpu(1, vd, CFL_SIZE);
	complex float* bank = md_alloc_gpu(1, bd, CFL_SIZE);
	complex float* hybrid = d->wave ? md_alloc_gpu(1, vd, CFL_SIZE) : NULL;
	complex float* table = md_alloc_gpu(DIMS, td, CFL_SIZE);

	struct bartorch_grid_axes ax;
	grid_axes(d, &ax);

	long per = d->batch * d->kept;

	for (long c = 0; c < coils; c++) {

		const complex float* m = (NULL == map) ? NULL : map + c * map_step;

		md_copy2(DIMS, td, tstr, table, ostr, src + c * (ostr[COIL_DIM] / (long)CFL_SIZE), CFL_SIZE);

		sampled_readout(d, td, table, false);

		md_clear(1, bd, bank, CFL_SIZE);

		bartorch_cuda_table_to_bank(d->U, d->X, d->S, (int)d->R, d->kept, per, d->d_csr_start, d->d_csr,
				d->d_u_coord, d->dmask, d->dprefix, &ax, d->d_bt, bank, table);

		for (long r = 0; r < d->R; r++)
			grid_out(d, dst + c * coil_step + r * coeff_step, volume, hybrid, bank + r * per, m);
	}

	md_free(table);
	md_free(bank);
	md_free(volume);

	if (NULL != hybrid)
		md_free(hybrid);

#pragma omp atomic
	grid_fused_count++;
}
#endif

static void grid_forward(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(grid_s, _d);

	if (d->sampled) {

#ifdef USE_CUDA
		if (grid_ready(d, dst) && cuda_ondevice(src) && sampled_fusable(d)) {

			long coils = d->cim_dims[COIL_DIM];

			sampled_fused_forward(d, dst, src, coils, d->img_vol, d->img_vol * coils, NULL, 0);
			return;
		}
#endif
		sampled_host_forward(d, dst, src);
		return;
	}

	linop_forward(d->fwd, DIMS, d->out_dims, dst, DIMS, d->dom_dims, src);
}

static void grid_adjoint(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(grid_s, _d);

	if (d->sampled) {

#ifdef USE_CUDA
		if (grid_ready(d, dst) && cuda_ondevice(src) && sampled_fusable(d)) {

			long coils = d->cim_dims[COIL_DIM];

			md_clear(DIMS, d->dom_dims, dst, CFL_SIZE);
			sampled_fused_adjoint(d, dst, src, coils, d->img_vol, d->img_vol * coils, NULL, 0);
			return;
		}
#endif
		sampled_host_adjoint(d, dst, src);
		return;
	}

	linop_adjoint(d->fwd, DIMS, d->dom_dims, dst, DIMS, d->out_dims, src);
}

static void grid_normal(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const auto d = CAST_DOWN(grid_s, _d);

	if (grid_ready(d, dst) && cuda_ondevice(src)) {

		long coils = d->cim_dims[COIL_DIM];

		md_clear(DIMS, d->dom_dims, dst, CFL_SIZE);
		grid_fused(d, dst, src, coils, d->img_vol, d->img_vol * coils, NULL, 0);
		return;
	}

	linop_forward(d->nrm, DIMS, d->dom_dims, dst, DIMS, d->dom_dims, src);
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

	md_free(d->d_entry_u);
	md_free(d->d_u_coord);
	md_free(d->d_csr_start);
	md_free(d->d_csr);
	md_free(d->d_bt);
	md_free(d->dpsf);

	xfree(d->entry_u);
	xfree(d->u_coord);
	xfree(d->csr_start);
	xfree(d->csr);
	xfree(d->bt);
	md_free(d->psf);

	linop_free(d->nrm);
	linop_free(d->fwd);

	xfree(d);
}

/* Whether `op` is a Cartesian transform whose normal the coil loop can hand
 * a sensitivity to, on the card `ref` is on. */
int bartorch_grid_folds(const struct linop_s* op, const void* ref)
{
	struct grid_s* d = CAST_MAYBE(grid_s, linop_get_data(op));

	return ((NULL != d) && d->toeplitz && grid_ready(d, ref)) ? 1 : 0;
}

/* The normal of a slab of coils, each with its sensitivity from `map`,
 * against the image `src`, added to the image `dst`. */
void bartorch_grid_normal_sense(const struct linop_s* op, complex float* dst, const complex float* src,
		const long map_strs[DIMS], const complex float* map)
{
	struct grid_s* d = CAST_DOWN(grid_s, linop_get_data(op));

	grid_fused(d, dst, src, d->cim_dims[COIL_DIM], 0, d->img_vol, map, map_strs[COIL_DIM] / (long)CFL_SIZE);
}

/* Whether `op` is a sampled-only Cartesian transform whose forward and adjoint
 * the coil loop can hand a sensitivity to, on the card `ref` is on. */
int bartorch_grid_folds_samples(const struct linop_s* op, const void* ref)
{
	struct grid_s* d = CAST_MAYBE(grid_s, linop_get_data(op));

	return ((NULL != d) && d->sampled && sampled_fusable(d) && grid_ready(d, ref)) ? 1 : 0;
}

/* The samples of a slab of coils, each with its sensitivity from `map`, of the
 * image `src`, into the slab's samples `dst`. */
void bartorch_grid_forward_sense(const struct linop_s* op, complex float* dst, const complex float* src,
		const long map_strs[DIMS], const complex float* map)
{
	struct grid_s* d = CAST_DOWN(grid_s, linop_get_data(op));

#ifdef USE_CUDA
	sampled_fused_forward(d, dst, src, d->cim_dims[COIL_DIM], 0, d->img_vol, map, map_strs[COIL_DIM] / (long)CFL_SIZE);
#else
	(void)d; (void)dst; (void)src; (void)map_strs; (void)map;
	error("bartorch: the Cartesian callbacks run on a card\n");
#endif
}

/* The adjoint of the slab's samples `src`, added to the image `dst`. */
void bartorch_grid_adjoint_sense(const struct linop_s* op, complex float* dst, const complex float* src,
		const long map_strs[DIMS], const complex float* map)
{
	struct grid_s* d = CAST_DOWN(grid_s, linop_get_data(op));

#ifdef USE_CUDA
	sampled_fused_adjoint(d, dst, src, d->cim_dims[COIL_DIM], 0, d->img_vol, map, map_strs[COIL_DIM] / (long)CFL_SIZE);
#else
	(void)d; (void)dst; (void)src; (void)map_strs; (void)map;
	error("bartorch: the Cartesian callbacks run on a card\n");
#endif
}

/* What a Cartesian slab transform holds over coil images of `cim_dims`: the
 * dense forward, the normal as BART's chain, and what the callbacks need.
 * See grid_transform_create. */
/* BART's transform as a linop in the convention asked for, or its adjoint. */
static struct linop_s* grid_ft_create(const long dims[DIMS], unsigned long flags, bool centred, bool adjoint)
{
	if (centred)
		return adjoint ? linop_ifftc_create(DIMS, dims, flags) : linop_fftc_create(DIMS, dims, flags);

	return adjoint ? linop_ifft_create(DIMS, dims, flags) : linop_fft_create(DIMS, dims, flags);
}

static struct grid_s* grid_state(const long cim_dims[DIMS],
		const long pat_dims[DIMS], const complex float* pattern,
		const long bas_dims[DIMS], const complex float* basis, int toeplitz, bool centred)
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
	d->centred = centred;

	struct linop_s* fwd = grid_ft_create(cim_dims, fft_flags, centred, false);

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
	long promoted = 1;

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
			promoted = cim_dims[extra];
		}
	}

	const complex float* kpattern = (NULL != repeated) ? repeated : pattern;

	long kdims[DIMS];
	complex float* K = grid_kernel(kdims, kpat_dims, kpattern, bas_dims, basis);

	/* The uncentred transform is unnormalized, so along the axis taken in
	 * for cuFFT its adjoint and itself give that axis's length rather than
	 * cancelling; the kernel takes it back off. */
	if (!centred && (1 < promoted))
		md_zsmul(DIMS, kdims, K, K, 1. / (double)promoted);

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

		nrm = linop_chain_FF(linop_chain_FF(grid_ft_create(cim_dims, d->flags, centred, false), core),
				grid_ft_create(cim_dims, d->flags, centred, true));

		grid_compress(d, kdims, K, kpat_dims, kpattern);
		d->batch = d->vol / d->plane;
	}

	md_free(K);
	md_free(repeated);

	d->fwd = fwd;
	d->nrm = nrm;
	d->toeplitz = (0 != toeplitz);

	d->wave = false;
	md_copy_dims(DIMS, d->dom_dims, cim_dims);
	d->img_vol = d->vol;
	d->wave_off = 0;
	d->psf = NULL;
	d->dpsf = NULL;

	d->sampled = false;
	d->kspace_readout = true;
	d->T = 0;
	d->S = 0;
	d->X = 0;
	d->E = 0;
	d->U = 0;
	d->entry_u = NULL;
	d->u_coord = NULL;
	d->csr_start = NULL;
	d->csr = NULL;
	d->bt = NULL;
	d->d_entry_u = NULL;
	d->d_u_coord = NULL;
	d->d_csr_start = NULL;
	d->d_csr = NULL;
	d->d_bt = NULL;

	debug_printf(DP_DEBUG1, "Cartesian slab: %ld coefficients, normal transforms axes %lx of %lx, %ld of %ld places kept\n",
			R, d->flags, fft_flags, d->kept, d->plane);

	return PTR_PASS(d);
}

/* The slab transform over coil images of `cim_dims`, or over every coil when
 * the loop does not run.  `pattern` and `basis` may each be NULL.  Without
 * `toeplitz` the normal is the two applications, as BART derives it. */
const struct linop_s* grid_transform_create(const long cim_dims[DIMS],
		const long pat_dims[DIMS], const complex float* pattern,
		const long bas_dims[DIMS], const complex float* basis, int toeplitz)
{
	struct grid_s* d = grid_state(cim_dims, pat_dims, pattern, bas_dims, basis, toeplitz, true);

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, d->out_dims);

	return linop_create(DIMS, out_dims, DIMS, cim_dims, CAST_UP(d),
			grid_forward, grid_adjoint, d->toeplitz ? grid_normal : NULL, NULL, grid_del);
}

/* What the slab transform over sampled-only k-space holds: `T` frames of `S` shots, each
 * a phase-encode position of `components` indices -- (y) for a 2D image,
 * (z, y) for a 3D one, -1 for padding -- with the whole readout along it.
 * The samples are [1, readout, shots, coils, 1, frames]: in k-space along the
 * readout when `kspace_readout`, as the image has it otherwise.
 *
 * The transforms, the basis and the normal are the dense operator's, over the
 * pattern the positions stand for: the square root of how often each frame
 * samples a place, so a place sampled twice counts twice in the normal as it
 * does in the two applications.  Forward, the table is read off the gathered
 * spectrum and transformed along its readout where that is asked for; that
 * is a transform of the table, not of the volume. */
static struct grid_s* sampled_state(const long cim_dims[DIMS], long T, long S, int components,
		const long* positions, const long bas_dims[DIMS], const complex float* basis, bool centred)
{
	long ny = cim_dims[PHS1_DIM];
	long nz = cim_dims[PHS2_DIM];

	if (((1 < nz) ? 2 : 1) != components)
		error("bartorch: positions of %d indices for an image of %d phase-encode axes\n", components, (1 < nz) ? 2 : 1);

	if ((NULL == basis) ? (1 != T) : (bas_dims[TE_DIM] != T))
		error("bartorch: positions over %ld frames, and a basis of %ld\n", T, (NULL == basis) ? 1L : bas_dims[TE_DIM]);

	long plane = ny * nz;
	long E = T * S;

	int* place_u = xmalloc((size_t)plane * sizeof(int));
	float* counts = xmalloc((size_t)(T * plane) * sizeof(float));
	int* entry_u = xmalloc((size_t)MAX(1, E) * sizeof(int));

	for (long p = 0; p < plane; p++)
		place_u[p] = -1;

	for (long i = 0; i < T * plane; i++)
		counts[i] = 0.f;

	long U = 0;
	long valid = 0;

	for (long e = 0; e < E; e++) {

		long zc = (2 == components) ? positions[e * components] : 0;
		long yc = positions[e * components + components - 1];

		if ((-1 == zc) || (-1 == yc)) {

			entry_u[e] = -1;
			continue;
		}

		if ((zc < 0) || (zc >= nz) || (yc < 0) || (yc >= ny))
			error("bartorch: a phase encode lies outside the %ld x %ld plane\n", nz, ny);

		long p = yc + ny * zc;

		if (-1 == place_u[p])
			place_u[p] = (int)U++;

		entry_u[e] = place_u[p];
		counts[(e / S) * plane + p] += 1.f;
		valid++;
	}

	int* u_coord = xmalloc((size_t)MAX(1, 2 * U) * sizeof(int));
	int* csr_start = xmalloc((size_t)(U + 1) * sizeof(int));
	int* csr = xmalloc((size_t)MAX(1, valid) * sizeof(int));
	int* fill = xmalloc((size_t)MAX(1, U) * sizeof(int));

	for (long p = 0; p < plane; p++) {

		if (0 > place_u[p])
			continue;

		u_coord[2 * place_u[p]] = (int)(p / ny);
		u_coord[2 * place_u[p] + 1] = (int)(p % ny);
	}

	for (long u = 0; u <= U; u++)
		csr_start[u] = 0;

	for (long e = 0; e < E; e++)
		if (0 <= entry_u[e])
			csr_start[entry_u[e] + 1]++;

	for (long u = 0; u < U; u++) {

		csr_start[u + 1] += csr_start[u];
		fill[u] = csr_start[u];
	}

	for (long e = 0; e < E; e++)
		if (0 <= entry_u[e])
			csr[fill[entry_u[e]]++] = (int)e;

	xfree(fill);
	xfree(place_u);

	long pat_dims[DIMS];
	md_singleton_dims(DIMS, pat_dims);
	pat_dims[PHS1_DIM] = ny;
	pat_dims[PHS2_DIM] = nz;
	pat_dims[TE_DIM] = T;

	complex float* pattern = md_alloc(DIMS, pat_dims, CFL_SIZE);

	for (long i = 0; i < T * plane; i++)
		pattern[i] = sqrtf(counts[i]);

	xfree(counts);

	/* Built as for the closed-form normal whatever `toeplitz` says: the
	 * forward and the adjoint read the kept places it finds. */
	struct grid_s* d = grid_state(cim_dims, pat_dims, pattern, bas_dims, basis, 1, centred);

	/* The dense forward is not what this operator applies. */
	linop_free(d->fwd);
	d->fwd = NULL;

	md_free(pattern);

	d->sampled = true;
	d->T = T;
	d->S = S;
	d->X = cim_dims[READ_DIM];
	d->E = E;
	d->U = U;
	d->entry_u = entry_u;
	d->u_coord = u_coord;
	d->csr_start = csr_start;
	d->csr = csr;

	if (NULL != basis) {

		complex float* B = md_alloc(DIMS, bas_dims, CFL_SIZE);
		md_copy(DIMS, bas_dims, B, basis, CFL_SIZE);

		d->bt = xmalloc((size_t)(T * d->R) * sizeof(complex float));

		for (long t = 0; t < T; t++)
			for (long r = 0; r < d->R; r++)
				d->bt[t * d->R + r] = B[t + T * r];

		md_free(B);
	}

	md_select_dims(DIMS, COIL_FLAG, d->out_dims, cim_dims);
	d->out_dims[PHS1_DIM] = d->X;
	d->out_dims[PHS2_DIM] = S;
	d->out_dims[TE_DIM] = T;

	debug_printf(DP_DEBUG1, "Cartesian samples: %ld frames x %ld shots x %ld readout at %ld places\n",
			T, S, d->X, U);

	return d;
}

const struct linop_s* grid_sampled_create(const long cim_dims[DIMS], long T, long S, int components,
		const long* positions, const long bas_dims[DIMS], const complex float* basis,
		int kspace_readout, int toeplitz)
{
	struct grid_s* d = sampled_state(cim_dims, T, S, components, positions, bas_dims, basis, true);

	d->toeplitz = (0 != toeplitz);
	d->kspace_readout = (0 != kspace_readout);

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, d->out_dims);

	return linop_create(DIMS, out_dims, DIMS, cim_dims, CAST_UP(d),
			grid_forward, grid_adjoint, d->toeplitz ? grid_normal : NULL, NULL, grid_del);
}

/* A wave over the grid `d` holds: the domain, the offset of the zero-fill,
 * and the point spread function over the grid's spatial axes. */
static void wave_setup(struct grid_s* d, const long dom_dims[DIMS], const complex float* psf)
{
	long wx = d->cim_dims[READ_DIM];
	long sx = dom_dims[READ_DIM];

	d->wave = true;
	md_copy_dims(DIMS, d->dom_dims, dom_dims);
	d->img_vol = md_calc_size(3, dom_dims);
	d->wave_off = (wx / 2 > sx / 2) ? (wx / 2 - sx / 2) : (sx / 2 - wx / 2);

	long pdims[DIMS];
	md_select_dims(DIMS, FFT_FLAGS, pdims, d->cim_dims);

	d->psf = md_alloc(DIMS, pdims, CFL_SIZE);
	md_copy(DIMS, pdims, d->psf, psf, CFL_SIZE);
}

/* The front as BART's chain: the zero-fill, the transform along the readout,
 * the point spread function. */
static struct linop_s* wave_front(const struct grid_s* d)
{
	struct linop_s* R = linop_resize_center_create(DIMS, d->cim_dims, d->dom_dims);
	struct linop_s* F = grid_ft_create(d->cim_dims, READ_FLAG, d->centred, false);
	struct linop_s* W = linop_cdiag_create(DIMS, d->cim_dims, FFT_FLAGS, d->psf);

	return linop_chain_FF(linop_chain_FF(R, F), W);
}

/* The normal as BART's chain: the grid's, with the front on either side. */
static void wave_normal_chain(struct grid_s* d)
{
	struct linop_s* front = wave_front(d);
	struct linop_s* back = (struct linop_s*)linop_get_adjoint(front);

	d->nrm = linop_chain_FF(linop_chain_FF(linop_clone(front), (struct linop_s*)d->nrm), back);

	linop_free(front);
}

/* The wave slab transform over coil images of `dom_dims`: the front to a
 * readout of `wx`, the transform along the phase encodes, then the basis and
 * the pattern as the Cartesian grid has them; `psf` is over (wx, y, z).  The
 * closed-form normal puts its kernel between the phase-encode transforms, so
 * it takes a pattern the same all along the readout -- the readout is not
 * the transform's to cancel -- and any other normal is the two applications. */
const struct linop_s* wave_transform_create(const long dom_dims[DIMS], long wx, const complex float* psf, int centred,
		const long pat_dims[DIMS], const complex float* pattern,
		const long bas_dims[DIMS], const complex float* basis, int toeplitz)
{
	long cim_dims[DIMS];
	md_copy_dims(DIMS, cim_dims, dom_dims);
	cim_dims[READ_DIM] = wx;

	bool closed = (0 != toeplitz) && ((NULL == pattern) || (1 == pat_dims[READ_DIM]));

	struct grid_s* d = grid_state(cim_dims, pat_dims, pattern, bas_dims, basis, closed ? 1 : 0, 0 != centred);

	wave_setup(d, dom_dims, psf);

	unsigned long pe = FFT_FLAGS & ~READ_FLAG & md_nontriv_dims(DIMS, cim_dims);

	struct linop_s* fwd = linop_chain_FF(wave_front(d), grid_ft_create(cim_dims, pe, d->centred, false));

	if (NULL != basis)
		fwd = linop_chain_FF(fwd, linop_fmac_dims_create(DIMS, d->out_dims, cim_dims, bas_dims, basis));

	if (NULL != pattern)
		fwd = linop_chain_FF(fwd, linop_sampling_create(d->out_dims, pat_dims, pattern));

	linop_free(d->fwd);
	d->fwd = fwd;

	wave_normal_chain(d);

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, d->out_dims);

	return linop_create(DIMS, out_dims, DIMS, d->dom_dims, CAST_UP(d),
			grid_forward, grid_adjoint, d->toeplitz ? grid_normal : NULL, NULL, grid_del);
}

/* The same over sampled-only k-space: a table of phase encodes per frame with
 * the oversampled readout along each, as the front leaves it. */
const struct linop_s* wave_sampled_create(const long dom_dims[DIMS], long wx, const complex float* psf, int centred,
		long T, long S, int components, const long* positions,
		const long bas_dims[DIMS], const complex float* basis, int toeplitz)
{
	long cim_dims[DIMS];
	md_copy_dims(DIMS, cim_dims, dom_dims);
	cim_dims[READ_DIM] = wx;

	struct grid_s* d = sampled_state(cim_dims, T, S, components, positions, bas_dims, basis, 0 != centred);

	wave_setup(d, dom_dims, psf);

	d->toeplitz = (0 != toeplitz);
	d->kspace_readout = false;

	wave_normal_chain(d);

	long out_dims[DIMS];
	md_copy_dims(DIMS, out_dims, d->out_dims);

	return linop_create(DIMS, out_dims, DIMS, d->dom_dims, CAST_UP(d),
			grid_forward, grid_adjoint, d->toeplitz ? grid_normal : NULL, NULL, grid_del);
}
