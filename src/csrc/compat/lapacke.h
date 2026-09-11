/* The LAPACKE surface BART's num/lapack.c uses, served by lapacke_shim.c.
 * Column-major only: that is the only layout BART passes. */
#ifndef BARTORCH_LAPACKE_H
#define BARTORCH_LAPACKE_H

#include <complex.h>

#ifdef __cplusplus
extern "C" {
#endif

typedef int lapack_int;
typedef int lapack_logical;
typedef float _Complex lapack_complex_float;
typedef double _Complex lapack_complex_double;

#define LAPACK_ROW_MAJOR 101
#define LAPACK_COL_MAJOR 102

typedef lapack_logical (*LAPACK_C_SELECT1)(const lapack_complex_float*);
typedef lapack_logical (*LAPACK_Z_SELECT1)(const lapack_complex_double*);

lapack_int LAPACKE_cheev(int layout, char jobz, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda, float* w);
lapack_int LAPACKE_chegv(int layout, lapack_int itype, char jobz, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, float* w);
lapack_int LAPACKE_cgesdd(int layout, char jobz, lapack_int m, lapack_int n, lapack_complex_float* a, lapack_int lda, float* s, lapack_complex_float* u, lapack_int ldu, lapack_complex_float* vt, lapack_int ldvt);
lapack_int LAPACKE_cgesvd(int layout, char jobu, char jobvt, lapack_int m, lapack_int n, lapack_complex_float* a, lapack_int lda, float* s, lapack_complex_float* u, lapack_int ldu, lapack_complex_float* vt, lapack_int ldvt, float* superb);
lapack_int LAPACKE_cgeqrf(int layout, lapack_int m, lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_complex_float* tau);
lapack_int LAPACKE_cungqr(int layout, lapack_int m, lapack_int n, lapack_int k, lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau);
lapack_int LAPACKE_zheev(int layout, char jobz, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda, double* w);
lapack_int LAPACKE_zgesdd(int layout, char jobz, lapack_int m, lapack_int n, lapack_complex_double* a, lapack_int lda, double* s, lapack_complex_double* u, lapack_int ldu, lapack_complex_double* vt, lapack_int ldvt);
lapack_int LAPACKE_cpotrf(int layout, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda);
lapack_int LAPACKE_ctrtri(int layout, char uplo, char diag, lapack_int n, lapack_complex_float* a, lapack_int lda);
lapack_int LAPACKE_ctrtrs(int layout, char uplo, char trans, char diag, lapack_int n, lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb);
lapack_int LAPACKE_cgetrf(int layout, lapack_int m, lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_int* ipiv);
lapack_int LAPACKE_cgetri(int layout, lapack_int n, lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv);
lapack_int LAPACKE_sgetrf(int layout, lapack_int m, lapack_int n, float* a, lapack_int lda, lapack_int* ipiv);
lapack_int LAPACKE_sgetri(int layout, lapack_int n, float* a, lapack_int lda, const lapack_int* ipiv);
lapack_int LAPACKE_cgees(int layout, char jobvs, char sort, LAPACK_C_SELECT1 select, lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_int* sdim, lapack_complex_float* w, lapack_complex_float* vs, lapack_int ldvs);
lapack_int LAPACKE_zgees(int layout, char jobvs, char sort, LAPACK_Z_SELECT1 select, lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_int* sdim, lapack_complex_double* w, lapack_complex_double* vs, lapack_int ldvs);
lapack_int LAPACKE_ctrsyl(int layout, char trana, char tranb, lapack_int isgn, lapack_int m, lapack_int n, const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* c, lapack_int ldc, float* scale);
lapack_int LAPACKE_sgesv(int layout, lapack_int n, lapack_int nrhs, float* a, lapack_int lda, lapack_int* ipiv, float* b, lapack_int ldb);

#ifdef __cplusplus
}
#endif

#endif
