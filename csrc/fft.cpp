/*
 * The FFTW guru interface BART plans with, executed by MKL where the process
 * has it and by pocketfft otherwise.
 *
 * A guru plan is a set of transformed dimensions plus a set of "howmany"
 * dimensions to loop over, each with its own input and output stride in
 * elements.  pocketfft takes exactly that description as one strided
 * N-dimensional array with a list of axes to transform, so a plan is the
 * translated description and nothing is precomputed.  That is what every
 * plan carries, and what answers whatever MKL will not.
 *
 * MKL is reached through DFTI, its own interface, rather than through the
 * FFTW one it also publishes: that one refuses more than a single loop
 * dimension, and BART passes one per dimension it is not transforming --
 * fifteen of them, every time.  DFTI takes the transformed axes and one loop
 * axis, so the loop axes left over are walked here.  Most of the fifteen are
 * of length one and drop out; what a reconstruction actually leaves is one
 * axis or none.
 *
 * The entry points are handed across the ABI by the host, from the same
 * library BART's BLAS comes from.  Nothing is linked against MKL, and a
 * description DFTI will not take, or a machine with no MKL at all, is
 * pocketfft's.
 */
#include <atomic>
#include <complex>
#include <cstring>
#include <mutex>
#include <thread>
#include <vector>

#include "pocketfft_hdronly.h"

#include "compat/fftw3.h"
#include "backend.h"
#include "include/bartorch.h"

namespace {

/* DFTI, as much of it as this needs.  MKL_LONG is `long` on the platforms
 * that have MKL at all, and the two calls that take a variable argument are
 * declared that way so the ABI is the one MKL was compiled with. */
typedef long mkl_long;

enum {
	DFTI_COMPLEX = 32,
	DFTI_NUMBER_OF_TRANSFORMS = 7,
	DFTI_PLACEMENT = 11,
	DFTI_INPUT_STRIDES = 12,
	DFTI_OUTPUT_STRIDES = 13,
	DFTI_INPUT_DISTANCE = 14,
	DFTI_OUTPUT_DISTANCE = 15,
	DFTI_THREAD_LIMIT = 27,
	DFTI_INPLACE = 43,
	DFTI_NOT_INPLACE = 44,
};

struct dfti_table {

	mkl_long (*create)(void**, int domain, mkl_long dim, mkl_long* lengths);
	mkl_long (*set)(void*, int param, ...);
	mkl_long (*commit)(void*);
	mkl_long (*forward)(void*, void* in, ...);
	mkl_long (*backward)(void*, void* in, ...);
	mkl_long (*release)(void**);
};

dfti_table dfti;

bool dfti_ready()
{
	return (NULL != dfti.create) && (NULL != dfti.set) && (NULL != dfti.commit)
		&& (NULL != dfti.forward) && (NULL != dfti.backward) && (NULL != dfti.release);
}

std::atomic<long> g_planned[2];	/* by MKL, by pocketfft */

/* One loop dimension, in elements. */
struct loop {
	long n, is, os;
};

struct plan_s {

	/* pocketfft's description of the whole thing, in bytes. */
	pocketfft::shape_t shape;
	pocketfft::stride_t stride_in;
	pocketfft::stride_t stride_out;
	pocketfft::shape_t axes;
	bool forward;

	/* What DFTI was given, when it took it: the transformed axes and at
	 * most one loop axis.  `outer` is what is left to walk. */
	int rank = 0;
	std::vector<mkl_long> lengths;
	std::vector<mkl_long> strides_in;	/* displacement first, then one per length */
	std::vector<mkl_long> strides_out;
	loop batch = { 0, 0, 0 };
	std::vector<loop> outer;

	/* Committed on first use, one for each placement BART executes with.
	 * A descriptor is thread-safe only up to the number of user threads it
	 * was committed for, and BART executes one plan at a time, so the lock
	 * costs nothing and holds whatever MKL's version does. */
	std::mutex lock;
	void* desc[2] = { NULL, NULL };
	bool refused[2] = { false, false };
};

std::atomic<int> g_threads{0};

int threads()
{
	int n = g_threads.load();

	if (n > 0)
		return n;

	unsigned hw = std::thread::hardware_concurrency();
	return (hw > 0) ? (int)hw : 1;
}

/* Whether DFTI can express this plan at all: something to transform, and no
 * loop axis whose stride it would have to guess. */
bool dfti_describable(const plan_s& p)
{
	return dfti_ready() && (p.rank > 0);
}

/* The descriptor for one placement, committed once and kept.  A refusal is
 * remembered so a plan MKL will not take is asked once and no more. */
void* dfti_descriptor(plan_s& p, int inplace)
{
	if ((NULL != p.desc[inplace]) || p.refused[inplace])
		return p.desc[inplace];

	void* desc = NULL;

	if (0 != dfti.create(&desc, DFTI_COMPLEX, p.rank, const_cast<mkl_long*>(p.lengths.data())))
		desc = NULL;

	mkl_long bad = (NULL == desc) ? 1 : 0;

	if (0 == bad) {

		if (0 != p.batch.n)
			bad |= dfti.set(desc, DFTI_NUMBER_OF_TRANSFORMS, (mkl_long)p.batch.n)
				| dfti.set(desc, DFTI_INPUT_DISTANCE, (mkl_long)p.batch.is)
				| dfti.set(desc, DFTI_OUTPUT_DISTANCE, (mkl_long)p.batch.os);

		bad |= dfti.set(desc, DFTI_PLACEMENT, (mkl_long)(inplace ? DFTI_INPLACE : DFTI_NOT_INPLACE));
		bad |= dfti.set(desc, DFTI_INPUT_STRIDES, const_cast<mkl_long*>(p.strides_in.data()));
		bad |= dfti.set(desc, DFTI_OUTPUT_STRIDES, const_cast<mkl_long*>(p.strides_out.data()));

		/* One thread, whatever BART was given.  MKL's transform is fast
		 * enough serially to beat a threaded pocketfft, and a thread team
		 * of its own is not: it is a second OpenMP runtime spinning
		 * against BART's own, which costs more inside a tool than the
		 * transform gains.  BART's parallelism sits above this. */
		dfti.set(desc, DFTI_THREAD_LIMIT, (mkl_long)1);

		bad |= dfti.commit(desc);
	}

	if (0 != bad) {

		if (NULL != desc)
			dfti.release(&desc);

		p.refused[inplace] = true;
		return NULL;
	}

	p.desc[inplace] = desc;
	return desc;
}

/* The loop axes DFTI did not take, walked here. */
void dfti_walk(const plan_s& p, void* desc, size_t level, const std::complex<float>* in, std::complex<float>* out)
{
	if (level == p.outer.size()) {

		if (p.forward)
			dfti.forward(desc, const_cast<std::complex<float>*>(in), out);
		else
			dfti.backward(desc, const_cast<std::complex<float>*>(in), out);

		return;
	}

	const loop& d = p.outer[level];

	for (long i = 0; i < d.n; i++)
		dfti_walk(p, desc, level + 1, in + i * d.is, out + i * d.os);
}

void strided_copy(const plan_s& p, const std::complex<float>* in, std::complex<float>* out)
{
	size_t nd = p.shape.size();
	std::vector<size_t> idx(nd, 0);
	size_t total = 1;

	for (size_t d : p.shape)
		total *= d;

	for (size_t c = 0; c < total; c++) {

		ptrdiff_t oi = 0, oo = 0;

		for (size_t d = 0; d < nd; d++) {

			oi += (ptrdiff_t)idx[d] * p.stride_in[d];
			oo += (ptrdiff_t)idx[d] * p.stride_out[d];
		}

		*(std::complex<float>*)((char*)out + oo) = *(const std::complex<float>*)((const char*)in + oi);

		for (size_t d = 0; d < nd; d++) {

			if (++idx[d] < p.shape[d])
				break;

			idx[d] = 0;
		}
	}
}

} // namespace

extern "C" {

void bartorch_fft_set_num_threads(int n)
{
	g_threads.store(n);
}

int bartorch_fft_get_num_threads(void)
{
	return threads();
}

int bartorch_fft_set(const char* symbol, void* fn)
{
	if (0 == strcmp(symbol, "DftiCreateDescriptor_s_md")) { dfti.create = (mkl_long (*)(void**, int, mkl_long, mkl_long*))fn; return 0; }
	if (0 == strcmp(symbol, "DftiSetValue")) { dfti.set = (mkl_long (*)(void*, int, ...))fn; return 0; }
	if (0 == strcmp(symbol, "DftiCommitDescriptor")) { dfti.commit = (mkl_long (*)(void*))fn; return 0; }
	if (0 == strcmp(symbol, "DftiComputeForward")) { dfti.forward = (mkl_long (*)(void*, void*, ...))fn; return 0; }
	if (0 == strcmp(symbol, "DftiComputeBackward")) { dfti.backward = (mkl_long (*)(void*, void*, ...))fn; return 0; }
	if (0 == strcmp(symbol, "DftiFreeDescriptor")) { dfti.release = (mkl_long (*)(void**))fn; return 0; }

	return -1;
}

int bartorch_fft_usable(void)
{
	return dfti_ready() ? 1 : 0;
}

long bartorch_fft_counter(int which)
{
	return g_planned[(0 == which) ? 0 : 1].load();
}

void bartorch_fft_reset_counters(void)
{
	g_planned[0].store(0);
	g_planned[1].store(0);
}

fftwf_plan fftwf_plan_guru64_dft(int rank, const fftwf_iodim64* dims, int howmany_rank, const fftwf_iodim64* howmany_dims, bartorch_cfloat*, bartorch_cfloat*, int sign, unsigned)
{
	auto* p = new plan_s;
	const ptrdiff_t es = (ptrdiff_t)sizeof(std::complex<float>);

	for (int i = 0; i < rank; i++) {

		p->axes.push_back(p->shape.size());
		p->shape.push_back((size_t)dims[i].n);
		p->stride_in.push_back(dims[i].is * es);
		p->stride_out.push_back(dims[i].os * es);
	}

	for (int i = 0; i < howmany_rank; i++) {

		p->shape.push_back((size_t)howmany_dims[i].n);
		p->stride_in.push_back(howmany_dims[i].is * es);
		p->stride_out.push_back(howmany_dims[i].os * es);
	}

	if (p->shape.empty()) {

		p->shape.push_back(1);
		p->stride_in.push_back(es);
		p->stride_out.push_back(es);
	}

	p->forward = (sign < 0);

	/* The same description for DFTI: strides in elements, the displacement
	 * first, and the longest loop axis handed over as its own count so the
	 * fewest are left to walk.  A loop axis of length one is no loop. */
	p->rank = rank;
	p->strides_in.push_back(0);
	p->strides_out.push_back(0);

	for (int i = 0; i < rank; i++) {

		p->lengths.push_back(dims[i].n);
		p->strides_in.push_back(dims[i].is);
		p->strides_out.push_back(dims[i].os);
	}

	int longest = -1;

	for (int i = 0; i < howmany_rank; i++)
		if ((howmany_dims[i].n > 1) && ((longest < 0) || (howmany_dims[i].n > howmany_dims[longest].n)))
			longest = i;

	if (longest >= 0)
		p->batch = { howmany_dims[longest].n, howmany_dims[longest].is, howmany_dims[longest].os };

	for (int i = 0; i < howmany_rank; i++)
		if ((i != longest) && (howmany_dims[i].n > 1))
			p->outer.push_back({ howmany_dims[i].n, howmany_dims[i].is, howmany_dims[i].os });

	g_planned[dfti_describable(*p) ? 0 : 1]++;

	return p;
}

void fftwf_execute_dft(const fftwf_plan plan, bartorch_cfloat* in, bartorch_cfloat* out)
{
	auto* p = static_cast<plan_s*>(plan);
	auto* cin = static_cast<const std::complex<float>*>(in);
	auto* cout = static_cast<std::complex<float>*>(out);

	if (p->axes.empty()) {

		if (cin != cout)
			strided_copy(*p, cin, cout);

		return;
	}

	if (dfti_describable(*p)) {

		std::lock_guard<std::mutex> held(p->lock);

		/* In place and out of place are different descriptors, and which
		 * one a plan is executed with is the caller's to decide each
		 * time: BART plans once for a shape and reuses it. */
		void* desc = dfti_descriptor(*p, (cin == cout) ? 1 : 0);

		if (NULL != desc) {

			dfti_walk(*p, desc, 0, cin, cout);
			return;
		}
	}

	pocketfft::c2c<float>(p->shape, p->stride_in, p->stride_out, p->axes, p->forward, cin, cout, 1.f, (size_t)threads());
}

void fftwf_destroy_plan(fftwf_plan plan)
{
	auto* p = static_cast<plan_s*>(plan);

	for (int i = 0; i < 2; i++)
		if (NULL != p->desc[i])
			dfti.release(&p->desc[i]);

	delete p;
}

int fftwf_export_wisdom_to_filename(const char*) { return 1; }
int fftwf_import_wisdom_from_filename(const char*) { return 0; }
int fftwf_init_threads(void) { return 1; }
void fftwf_plan_with_nthreads(int n) { g_threads.store(n); }
void fftwf_cleanup_threads(void) { }

}
