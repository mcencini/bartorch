/* The CBLAS surface BART's num/blas.c uses, served by cblas_shim.c. */
#ifndef BARTORCH_CBLAS_H
#define BARTORCH_CBLAS_H

#ifdef __cplusplus
extern "C" {
#endif

typedef enum { CblasRowMajor = 101, CblasColMajor = 102 } CBLAS_LAYOUT;
typedef enum { CblasNoTrans = 111, CblasTrans = 112, CblasConjTrans = 113 } CBLAS_TRANSPOSE;
typedef enum { CblasUpper = 121, CblasLower = 122 } CBLAS_UPLO;
typedef CBLAS_LAYOUT CBLAS_ORDER;

void cblas_sgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m, int n, int k, float alpha, const float* a, int lda, const float* b, int ldb, float beta, float* c, int ldc);
void cblas_cgemm(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE ta, CBLAS_TRANSPOSE tb, int m, int n, int k, const void* alpha, const void* a, int lda, const void* b, int ldb, const void* beta, void* c, int ldc);
void cblas_sgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int m, int n, float alpha, const float* a, int lda, const float* x, int incx, float beta, float* y, int incy);
void cblas_cgemv(CBLAS_LAYOUT layout, CBLAS_TRANSPOSE trans, int m, int n, const void* alpha, const void* a, int lda, const void* x, int incx, const void* beta, void* y, int incy);
void cblas_sger(CBLAS_LAYOUT layout, int m, int n, float alpha, const float* x, int incx, const float* y, int incy, float* a, int lda);
void cblas_cgeru(CBLAS_LAYOUT layout, int m, int n, const void* alpha, const void* x, int incx, const void* y, int incy, void* a, int lda);
void cblas_saxpy(int n, float alpha, const float* x, int incx, float* y, int incy);
void cblas_caxpy(int n, const void* alpha, const void* x, int incx, void* y, int incy);
void cblas_sscal(int n, float alpha, float* x, int incx);
void cblas_cscal(int n, const void* alpha, void* x, int incx);
float cblas_sdot(int n, const float* x, int incx, const float* y, int incy);
void cblas_cdotu_sub(int n, const void* x, int incx, const void* y, int incy, void* result);
void cblas_csyrk(CBLAS_LAYOUT layout, CBLAS_UPLO uplo, CBLAS_TRANSPOSE trans, int n, int k, const void* alpha, const void* a, int lda, const void* beta, void* c, int ldc);

#ifdef __cplusplus
}
#endif

#endif
