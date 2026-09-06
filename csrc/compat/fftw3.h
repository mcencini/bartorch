/* The FFTW3 single-precision surface BART's num/fft_plan.c uses, served by
 * fftw_pocketfft.cpp. */
#ifndef BARTORCH_FFTW3_H
#define BARTORCH_FFTW3_H

#include <stddef.h>
#include <stdio.h>

#ifdef __cplusplus
typedef void bartorch_cfloat;
extern "C" {
#else
#include <complex.h>
typedef float _Complex bartorch_cfloat;
#endif

typedef struct { ptrdiff_t n; ptrdiff_t is; ptrdiff_t os; } fftwf_iodim64;
typedef void* fftwf_plan;

#define FFTW_FORWARD (-1)
#define FFTW_BACKWARD (+1)
#define FFTW_MEASURE (0U)
#define FFTW_ESTIMATE (1U << 6)

fftwf_plan fftwf_plan_guru64_dft(int rank, const fftwf_iodim64* dims, int howmany_rank, const fftwf_iodim64* howmany_dims, bartorch_cfloat* in, bartorch_cfloat* out, int sign, unsigned flags);
void fftwf_execute_dft(const fftwf_plan p, bartorch_cfloat* in, bartorch_cfloat* out);
void fftwf_destroy_plan(fftwf_plan p);
int fftwf_export_wisdom_to_filename(const char* filename);
int fftwf_import_wisdom_from_filename(const char* filename);
int fftwf_init_threads(void);
void fftwf_plan_with_nthreads(int n);
void fftwf_cleanup_threads(void);

#ifdef __cplusplus
}
#endif

#endif
