/* Internal: the Fortran-ABI function table behind BART's BLAS and LAPACK. */
#ifndef BARTORCH_BACKEND_H
#define BARTORCH_BACKEND_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

enum bartorch_fn_id {
	BT_sgemm, BT_cgemm, BT_sgemv, BT_cgemv, BT_sger, BT_cgeru,
	BT_saxpy, BT_caxpy, BT_sscal, BT_cscal, BT_csyrk,
	BT_cheev, BT_chegv, BT_cgesdd, BT_cgesvd, BT_cgeqrf, BT_cungqr,
	BT_zheev, BT_zgesdd, BT_cpotrf, BT_ctrtri, BT_ctrtrs,
	BT_cgetrf, BT_cgetri, BT_sgetrf, BT_sgetri,
	BT_cgees, BT_zgees, BT_ctrsyl, BT_sgesv,
	BT_FN_COUNT
};

extern void* bartorch_fn_table[BT_FN_COUNT];
extern const char* const bartorch_fn_names[BT_FN_COUNT];

/* Every Fortran argument is a pointer; hidden string lengths are passed
 * as pointer-sized integers, which the callee is free to ignore. */
#define P const void*
typedef void (*bt_fn4)(P, P, P, P);
typedef void (*bt_fn5)(P, P, P, P, P);
typedef void (*bt_fn6)(P, P, P, P, P, P);
typedef void (*bt_fn7)(P, P, P, P, P, P, P);
typedef void (*bt_fn8)(P, P, P, P, P, P, P, P);
typedef void (*bt_fn9)(P, P, P, P, P, P, P, P, P);
typedef void (*bt_fn10)(P, P, P, P, P, P, P, P, P, P);
typedef void (*bt_fn11)(P, P, P, P, P, P, P, P, P, P, P);
typedef void (*bt_fn12)(P, P, P, P, P, P, P, P, P, P, P, P);
typedef void (*bt_fn13)(P, P, P, P, P, P, P, P, P, P, P, P, P);
typedef void (*bt_fn14)(P, P, P, P, P, P, P, P, P, P, P, P, P, P);
typedef void (*bt_fn15)(P, P, P, P, P, P, P, P, P, P, P, P, P, P, P);
typedef void (*bt_fn16)(P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P);
typedef void (*bt_fn17)(P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P, P);
#undef P

#define BT_LEN(n) ((const void*)(size_t)(n))
#define BT_FN(id, type) ((type)bartorch_fn_table[id])

/* Reference implementations, Fortran ABI, in ref_blas.c. */
void bartorch_ref_sgemm(const char* ta, const char* tb, const int* m, const int* n, const int* k, const float* alpha, const float* a, const int* lda, const float* b, const int* ldb, const float* beta, float* c, const int* ldc);
void bartorch_ref_cgemm(const char* ta, const char* tb, const int* m, const int* n, const int* k, const float* alpha, const float* a, const int* lda, const float* b, const int* ldb, const float* beta, float* c, const int* ldc);
void bartorch_ref_sgemv(const char* trans, const int* m, const int* n, const float* alpha, const float* a, const int* lda, const float* x, const int* incx, const float* beta, float* y, const int* incy);
void bartorch_ref_cgemv(const char* trans, const int* m, const int* n, const float* alpha, const float* a, const int* lda, const float* x, const int* incx, const float* beta, float* y, const int* incy);
void bartorch_ref_sger(const int* m, const int* n, const float* alpha, const float* x, const int* incx, const float* y, const int* incy, float* a, const int* lda);
void bartorch_ref_cgeru(const int* m, const int* n, const float* alpha, const float* x, const int* incx, const float* y, const int* incy, float* a, const int* lda);
void bartorch_ref_saxpy(const int* n, const float* alpha, const float* x, const int* incx, float* y, const int* incy);
void bartorch_ref_caxpy(const int* n, const float* alpha, const float* x, const int* incx, float* y, const int* incy);
void bartorch_ref_sscal(const int* n, const float* alpha, float* x, const int* incx);
void bartorch_ref_cscal(const int* n, const float* alpha, float* x, const int* incx);
void bartorch_ref_csyrk(const char* uplo, const char* trans, const int* n, const int* k, const float* alpha, const float* a, const int* lda, const float* beta, float* c, const int* ldc);

/* Threads for the FFT backend, in fft.cpp. */
void bartorch_fft_set_num_threads(int n);
int bartorch_fft_get_num_threads(void);

/* Error text accumulated for the running command, in api.c. */
void bartorch_record_error(const char* msg);

#ifdef __cplusplus
}
#endif

#endif
