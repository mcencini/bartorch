/*
 * Reference BLAS, Fortran ABI, column-major.  These serve BART when no
 * optimised routine has been installed through bartorch_backend_set.
 */
#include <complex.h>
#include <stdbool.h>

#include "backend.h"

typedef float _Complex cf;

static bool is_trans(char t) { return ('T' == t) || ('t' == t) || ('C' == t) || ('c' == t); }
static bool is_conj(char t) { return ('C' == t) || ('c' == t); }

void bartorch_ref_sgemm(const char* ta, const char* tb, const int* m_, const int* n_, const int* k_, const float* alpha, const float* a, const int* lda_, const float* b, const int* ldb_, const float* beta, float* c, const int* ldc_)
{
	int m = *m_, n = *n_, k = *k_, lda = *lda_, ldb = *ldb_, ldc = *ldc_;
	bool tra = is_trans(*ta), trb = is_trans(*tb);

#pragma omp parallel for
	for (int j = 0; j < n; j++) {

		for (int i = 0; i < m; i++)
			c[i + j * ldc] *= *beta;

		for (int l = 0; l < k; l++) {

			float t = *alpha * (trb ? b[j + l * ldb] : b[l + j * ldb]);

			for (int i = 0; i < m; i++)
				c[i + j * ldc] += t * (tra ? a[l + i * lda] : a[i + l * lda]);
		}
	}
}

void bartorch_ref_cgemm(const char* ta, const char* tb, const int* m_, const int* n_, const int* k_, const float* alpha_, const float* a_, const int* lda_, const float* b_, const int* ldb_, const float* beta_, float* c_, const int* ldc_)
{
	int m = *m_, n = *n_, k = *k_, lda = *lda_, ldb = *ldb_, ldc = *ldc_;
	const cf* a = (const cf*)a_;
	const cf* b = (const cf*)b_;
	cf* c = (cf*)c_;
	cf alpha = *(const cf*)alpha_;
	cf beta = *(const cf*)beta_;
	bool tra = is_trans(*ta), cja = is_conj(*ta);
	bool trb = is_trans(*tb), cjb = is_conj(*tb);

#pragma omp parallel for
	for (int j = 0; j < n; j++) {

		for (int i = 0; i < m; i++)
			c[i + j * ldc] *= beta;

		for (int l = 0; l < k; l++) {

			cf bv = trb ? b[j + l * ldb] : b[l + j * ldb];
			cf t = alpha * (cjb ? conjf(bv) : bv);

			if (tra) {

				for (int i = 0; i < m; i++) {

					cf av = a[l + i * lda];
					c[i + j * ldc] += t * (cja ? conjf(av) : av);
				}

			} else {

				for (int i = 0; i < m; i++)
					c[i + j * ldc] += t * a[i + l * lda];
			}
		}
	}
}

void bartorch_ref_sgemv(const char* trans, const int* m_, const int* n_, const float* alpha, const float* a, const int* lda_, const float* x, const int* incx_, const float* beta, float* y, const int* incy_)
{
	int m = *m_, n = *n_, lda = *lda_, incx = *incx_, incy = *incy_;
	bool tr = is_trans(*trans);
	int leny = tr ? n : m;
	int lenx = tr ? m : n;

	for (int i = 0; i < leny; i++)
		y[i * incy] *= *beta;

	for (int j = 0; j < lenx; j++) {

		float t = *alpha * x[j * incx];

		for (int i = 0; i < leny; i++)
			y[i * incy] += t * (tr ? a[j + i * lda] : a[i + j * lda]);
	}
}

void bartorch_ref_cgemv(const char* trans, const int* m_, const int* n_, const float* alpha_, const float* a_, const int* lda_, const float* x_, const int* incx_, const float* beta_, float* y_, const int* incy_)
{
	int m = *m_, n = *n_, lda = *lda_, incx = *incx_, incy = *incy_;
	const cf* a = (const cf*)a_;
	const cf* x = (const cf*)x_;
	cf* y = (cf*)y_;
	cf alpha = *(const cf*)alpha_;
	cf beta = *(const cf*)beta_;
	bool tr = is_trans(*trans), cj = is_conj(*trans);
	int leny = tr ? n : m;
	int lenx = tr ? m : n;

	for (int i = 0; i < leny; i++)
		y[i * incy] *= beta;

	for (int j = 0; j < lenx; j++) {

		cf t = alpha * x[j * incx];

		for (int i = 0; i < leny; i++) {

			cf av = tr ? a[j + i * lda] : a[i + j * lda];
			y[i * incy] += t * (cj ? conjf(av) : av);
		}
	}
}

void bartorch_ref_sger(const int* m_, const int* n_, const float* alpha, const float* x, const int* incx_, const float* y, const int* incy_, float* a, const int* lda_)
{
	int m = *m_, n = *n_, lda = *lda_, incx = *incx_, incy = *incy_;

	for (int j = 0; j < n; j++) {

		float t = *alpha * y[j * incy];

		for (int i = 0; i < m; i++)
			a[i + j * lda] += t * x[i * incx];
	}
}

void bartorch_ref_cgeru(const int* m_, const int* n_, const float* alpha_, const float* x_, const int* incx_, const float* y_, const int* incy_, float* a_, const int* lda_)
{
	int m = *m_, n = *n_, lda = *lda_, incx = *incx_, incy = *incy_;
	const cf* x = (const cf*)x_;
	const cf* y = (const cf*)y_;
	cf* a = (cf*)a_;
	cf alpha = *(const cf*)alpha_;

	for (int j = 0; j < n; j++) {

		cf t = alpha * y[j * incy];

		for (int i = 0; i < m; i++)
			a[i + j * lda] += t * x[i * incx];
	}
}

void bartorch_ref_saxpy(const int* n_, const float* alpha, const float* x, const int* incx_, float* y, const int* incy_)
{
	int n = *n_, incx = *incx_, incy = *incy_;

	for (int i = 0; i < n; i++)
		y[i * incy] += *alpha * x[i * incx];
}

void bartorch_ref_caxpy(const int* n_, const float* alpha_, const float* x_, const int* incx_, float* y_, const int* incy_)
{
	int n = *n_, incx = *incx_, incy = *incy_;
	const cf* x = (const cf*)x_;
	cf* y = (cf*)y_;
	cf alpha = *(const cf*)alpha_;

	for (int i = 0; i < n; i++)
		y[i * incy] += alpha * x[i * incx];
}

void bartorch_ref_sscal(const int* n_, const float* alpha, float* x, const int* incx_)
{
	int n = *n_, incx = *incx_;

	for (int i = 0; i < n; i++)
		x[i * incx] *= *alpha;
}

void bartorch_ref_cscal(const int* n_, const float* alpha_, float* x_, const int* incx_)
{
	int n = *n_, incx = *incx_;
	cf* x = (cf*)x_;
	cf alpha = *(const cf*)alpha_;

	for (int i = 0; i < n; i++)
		x[i * incx] *= alpha;
}

/* C := alpha op(A) op(A)^T + beta C, with C symmetric (not Hermitian);
 * only the selected triangle is referenced. */
void bartorch_ref_csyrk(const char* uplo, const char* trans, const int* n_, const int* k_, const float* alpha_, const float* a_, const int* lda_, const float* beta_, float* c_, const int* ldc_)
{
	int n = *n_, k = *k_, lda = *lda_, ldc = *ldc_;
	const cf* a = (const cf*)a_;
	cf* c = (cf*)c_;
	cf alpha = *(const cf*)alpha_;
	cf beta = *(const cf*)beta_;
	bool tr = is_trans(*trans);
	bool upper = ('U' == *uplo) || ('u' == *uplo);

	for (int j = 0; j < n; j++) {

		int i0 = upper ? 0 : j;
		int i1 = upper ? j + 1 : n;

		for (int i = i0; i < i1; i++) {

			cf s = 0.f;

			for (int l = 0; l < k; l++)
				s += (tr ? a[l + i * lda] : a[i + l * lda]) * (tr ? a[l + j * lda] : a[j + l * lda]);

			c[i + j * ldc] = alpha * s + beta * c[i + j * ldc];
		}
	}
}
