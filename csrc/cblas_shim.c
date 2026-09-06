/*
 * CBLAS as BART calls it, forwarded to the Fortran-ABI backend table.
 * BART passes column-major layout only; a row-major call is a bug in the
 * caller and is rejected loudly.
 */
#include <complex.h>
#include <stdlib.h>

#include "compat/cblas.h"
#include "backend.h"

static void check_layout(CBLAS_LAYOUT layout)
{
	if (CblasColMajor != layout) {

		bartorch_record_error("cblas: row-major layout is not supported");
		abort();
	}
}

static char trans_char(CBLAS_TRANSPOSE t)
{
	return (CblasNoTrans == t) ? 'N' : ((CblasTrans == t) ? 'T' : 'C');
}

void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc)
{
	check_layout(layout);
	char cta = trans_char(ta), ctb = trans_char(tb);
	BT_FN(BT_sgemm, bt_fn15)(&cta, &ctb, &m, &n, &k, &alpha, a, &lda, b, &ldb, &beta, c, &ldc, BT_LEN(1), BT_LEN(1));
}

void cblas_cgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m, int n, int k, const void* alpha, const void* a, int lda, const void* b, int ldb, const void* beta, void* c, int ldc)
{
	check_layout(layout);
	char cta = trans_char(ta), ctb = trans_char(tb);
	BT_FN(BT_cgemm, bt_fn15)(&cta, &ctb, &m, &n, &k, alpha, a, &lda, b, &ldb, beta, c, &ldc, BT_LEN(1), BT_LEN(1));
}

void cblas_sgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int m, int n, float alpha, const float* a, int lda, const float* x, int incx, float beta, float* y, int incy)
{
	check_layout(layout);
	char ct = trans_char(trans);
	BT_FN(BT_sgemv, bt_fn12)(&ct, &m, &n, &alpha, a, &lda, x, &incx, &beta, y, &incy, BT_LEN(1));
}

void cblas_cgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int m, int n, const void* alpha, const void* a, int lda, const void* x, int incx, const void* beta, void* y, int incy)
{
	check_layout(layout);
	char ct = trans_char(trans);
	BT_FN(BT_cgemv, bt_fn12)(&ct, &m, &n, alpha, a, &lda, x, &incx, beta, y, &incy, BT_LEN(1));
}

void cblas_sger(CBLAS_LAYOUT layout, int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda)
{
	check_layout(layout);
	BT_FN(BT_sger, bt_fn9)(&m, &n, &alpha, x, &incx, y, &incy, a, &lda);
}

void cblas_cgeru(CBLAS_LAYOUT layout, int m, int n, const void* alpha, const void* x, int incx, const void* y, int incy, void* a, int lda)
{
	check_layout(layout);
	BT_FN(BT_cgeru, bt_fn9)(&m, &n, alpha, x, &incx, y, &incy, a, &lda);
}

void cblas_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy)
{
	BT_FN(BT_saxpy, bt_fn6)(&n, &alpha, x, &incx, y, &incy);
}

void cblas_caxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy)
{
	BT_FN(BT_caxpy, bt_fn6)(&n, alpha, x, &incx, y, &incy);
}

void cblas_sscal(int n, float alpha, float* x, int incx)
{
	BT_FN(BT_sscal, bt_fn4)(&n, &alpha, x, &incx);
}

void cblas_cscal(int n, const void* alpha, void* x, int incx)
{
	BT_FN(BT_cscal, bt_fn4)(&n, alpha, x, &incx);
}

/* Dot products stay in C: the Fortran ABI for a complex return value
 * differs between compilers, so they are not table entries. */
float cblas_sdot(int n, const float* x, int incx, const float* y, int incy)
{
	double s = 0.;

	for (int i = 0; i < n; i++)
		s += (double)x[i * incx] * (double)y[i * incy];

	return (float)s;
}

void cblas_cdotu_sub(int n, const void* x_, int incx, const void* y_, int incy, void* result)
{
	const float _Complex* x = x_;
	const float _Complex* y = y_;
	double _Complex s = 0.;

	for (int i = 0; i < n; i++)
		s += (double _Complex)x[i * incx] * (double _Complex)y[i * incy];

	*(float _Complex*)result = (float _Complex)s;
}

void cblas_csyrk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, const void* alpha, const void* a, int lda, const void* beta, void* c, int ldc)
{
	check_layout(layout);
	char cu = (CblasUpper == uplo) ? 'U' : 'L';
	char ct = trans_char(trans);
	BT_FN(BT_csyrk, bt_fn12)(&cu, &ct, &n, &k, alpha, a, &lda, beta, c, &ldc, BT_LEN(1), BT_LEN(1));
}
