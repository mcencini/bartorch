/*
 * The FFTW guru interface BART plans with, executed by pocketfft.
 *
 * A guru plan is a set of transformed dimensions plus a set of "howmany"
 * dimensions to loop over, each with its own input and output stride in
 * elements.  pocketfft takes exactly that description as one strided
 * N-dimensional array with a list of axes to transform, so a plan is the
 * translated description and nothing is precomputed.
 */
#include <atomic>
#include <complex>
#include <cstring>
#include <thread>
#include <vector>

#include "pocketfft_hdronly.h"

#include "compat/fftw3.h"
#include "backend.h"

namespace {

struct plan_s {
	pocketfft::shape_t shape;
	pocketfft::stride_t stride_in;
	pocketfft::stride_t stride_out;
	pocketfft::shape_t axes;
	bool forward;
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
	return p;
}

void fftwf_execute_dft(const fftwf_plan plan, bartorch_cfloat* in, bartorch_cfloat* out)
{
	const auto* p = static_cast<const plan_s*>(plan);
	auto* cin = static_cast<const std::complex<float>*>(in);
	auto* cout = static_cast<std::complex<float>*>(out);

	if (p->axes.empty()) {

		if (cin != cout)
			strided_copy(*p, cin, cout);

		return;
	}

	pocketfft::c2c<float>(p->shape, p->stride_in, p->stride_out, p->axes, p->forward, cin, cout, 1.f, (size_t)threads());
}

void fftwf_destroy_plan(fftwf_plan plan)
{
	delete static_cast<plan_s*>(plan);
}

int fftwf_export_wisdom_to_filename(const char*) { return 1; }
int fftwf_import_wisdom_from_filename(const char*) { return 0; }
int fftwf_init_threads(void) { return 1; }
void fftwf_plan_with_nthreads(int n) { g_threads.store(n); }
void fftwf_cleanup_threads(void) { }

}
