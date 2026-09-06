/*
 * LAPACKE as BART calls it, forwarded to Fortran-ABI routines in the
 * backend table.  Workspace sizes are obtained by the standard lwork = -1
 * query, so an installed routine may be a vendor library or a host
 * callback that answers the query with any positive size.
 */
#include <complex.h>
#include <stdio.h>
#include <stdlib.h>

#include "compat/lapacke.h"
#include "backend.h"

#define MAX(a, b) (((a) > (b)) ? (a) : (b))
#define MIN(a, b) (((a) < (b)) ? (a) : (b))

static lapack_int missing(int id)
{
	char msg[128];
	snprintf(msg, sizeof(msg), "no LAPACK backend installed for %s", bartorch_fn_names[id]);
	bartorch_record_error(msg);
	return -1;
}

static int check_layout(int layout)
{
	if (LAPACK_COL_MAJOR != layout) {

		bartorch_record_error("lapacke: row-major layout is not supported");
		return -1;
	}

	return 0;
}

lapack_int LAPACKE_cheev(int layout, char jobz, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda, float* w)
{
	if (check_layout(layout)) return -1;
	bt_fn12 fn = BT_FN(BT_cheev, bt_fn12);
	if (NULL == fn) return missing(BT_cheev);

	lapack_int info = 0, lwork = -1;
	lapack_complex_float query = 0;
	float* rwork = malloc(sizeof(float) * (size_t)MAX(1, 3 * n - 2));

	fn(&jobz, &uplo, &n, a, &lda, w, &query, &lwork, rwork, &info, BT_LEN(1), BT_LEN(1));

	if (0 == info) {

		lwork = MAX(1, (lapack_int)crealf(query));
		lapack_complex_float* work = malloc(sizeof(*work) * (size_t)lwork);
		fn(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info, BT_LEN(1), BT_LEN(1));
		free(work);
	}

	free(rwork);
	return info;
}

lapack_int LAPACKE_zheev(int layout, char jobz, char uplo, lapack_int n, lapack_complex_double* a, lapack_int lda, double* w)
{
	if (check_layout(layout)) return -1;
	bt_fn12 fn = BT_FN(BT_zheev, bt_fn12);
	if (NULL == fn) return missing(BT_zheev);

	lapack_int info = 0, lwork = -1;
	lapack_complex_double query = 0;
	double* rwork = malloc(sizeof(double) * (size_t)MAX(1, 3 * n - 2));

	fn(&jobz, &uplo, &n, a, &lda, w, &query, &lwork, rwork, &info, BT_LEN(1), BT_LEN(1));

	if (0 == info) {

		lwork = MAX(1, (lapack_int)creal(query));
		lapack_complex_double* work = malloc(sizeof(*work) * (size_t)lwork);
		fn(&jobz, &uplo, &n, a, &lda, w, work, &lwork, rwork, &info, BT_LEN(1), BT_LEN(1));
		free(work);
	}

	free(rwork);
	return info;
}

lapack_int LAPACKE_chegv(int layout, lapack_int itype, char jobz, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb, float* w)
{
	if (check_layout(layout)) return -1;
	bt_fn15 fn = BT_FN(BT_chegv, bt_fn15);
	if (NULL == fn) return missing(BT_chegv);

	lapack_int info = 0, lwork = -1;
	lapack_complex_float query = 0;
	float* rwork = malloc(sizeof(float) * (size_t)MAX(1, 3 * n - 2));

	fn(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, &query, &lwork, rwork, &info, BT_LEN(1), BT_LEN(1));

	if (0 == info) {

		lwork = MAX(1, (lapack_int)crealf(query));
		lapack_complex_float* work = malloc(sizeof(*work) * (size_t)lwork);
		fn(&itype, &jobz, &uplo, &n, a, &lda, b, &ldb, w, work, &lwork, rwork, &info, BT_LEN(1), BT_LEN(1));
		free(work);
	}

	free(rwork);
	return info;
}

static size_t gesdd_rwork(lapack_int m, lapack_int n)
{
	lapack_int mn = MIN(m, n), mx = MAX(m, n);
	return (size_t)MAX(1, MAX(5 * mn * mn + 7 * mn, 2 * mx * mn + 2 * mn * mn + mn));
}

lapack_int LAPACKE_cgesdd(int layout, char jobz, lapack_int m, lapack_int n, lapack_complex_float* a, lapack_int lda, float* s, lapack_complex_float* u, lapack_int ldu, lapack_complex_float* vt, lapack_int ldvt)
{
	if (check_layout(layout)) return -1;
	bt_fn16 fn = BT_FN(BT_cgesdd, bt_fn16);
	if (NULL == fn) return missing(BT_cgesdd);

	lapack_int info = 0, lwork = -1;
	lapack_complex_float query = 0;
	float* rwork = malloc(sizeof(float) * gesdd_rwork(m, n));
	lapack_int* iwork = malloc(sizeof(lapack_int) * (size_t)MAX(1, 8 * MIN(m, n)));

	fn(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &query, &lwork, rwork, iwork, &info, BT_LEN(1));

	if (0 == info) {

		lwork = MAX(1, (lapack_int)crealf(query));
		lapack_complex_float* work = malloc(sizeof(*work) * (size_t)lwork);
		fn(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info, BT_LEN(1));
		free(work);
	}

	free(iwork);
	free(rwork);
	return info;
}

lapack_int LAPACKE_zgesdd(int layout, char jobz, lapack_int m, lapack_int n, lapack_complex_double* a, lapack_int lda, double* s, lapack_complex_double* u, lapack_int ldu, lapack_complex_double* vt, lapack_int ldvt)
{
	if (check_layout(layout)) return -1;
	bt_fn16 fn = BT_FN(BT_zgesdd, bt_fn16);
	if (NULL == fn) return missing(BT_zgesdd);

	lapack_int info = 0, lwork = -1;
	lapack_complex_double query = 0;
	double* rwork = malloc(sizeof(double) * gesdd_rwork(m, n));
	lapack_int* iwork = malloc(sizeof(lapack_int) * (size_t)MAX(1, 8 * MIN(m, n)));

	fn(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &query, &lwork, rwork, iwork, &info, BT_LEN(1));

	if (0 == info) {

		lwork = MAX(1, (lapack_int)creal(query));
		lapack_complex_double* work = malloc(sizeof(*work) * (size_t)lwork);
		fn(&jobz, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, iwork, &info, BT_LEN(1));
		free(work);
	}

	free(iwork);
	free(rwork);
	return info;
}

lapack_int LAPACKE_cgesvd(int layout, char jobu, char jobvt, lapack_int m, lapack_int n, lapack_complex_float* a, lapack_int lda, float* s, lapack_complex_float* u, lapack_int ldu, lapack_complex_float* vt, lapack_int ldvt, float* superb)
{
	if (check_layout(layout)) return -1;
	bt_fn17 fn = BT_FN(BT_cgesvd, bt_fn17);
	if (NULL == fn) return missing(BT_cgesvd);

	lapack_int info = 0, lwork = -1;
	lapack_int mn = MIN(m, n);
	lapack_complex_float query = 0;
	float* rwork = malloc(sizeof(float) * (size_t)MAX(1, 5 * mn));

	fn(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, &query, &lwork, rwork, &info, BT_LEN(1), BT_LEN(1));

	if (0 == info) {

		lwork = MAX(1, (lapack_int)crealf(query));
		lapack_complex_float* work = malloc(sizeof(*work) * (size_t)lwork);
		fn(&jobu, &jobvt, &m, &n, a, &lda, s, u, &ldu, vt, &ldvt, work, &lwork, rwork, &info, BT_LEN(1), BT_LEN(1));
		free(work);
	}

	if (NULL != superb)
		for (lapack_int i = 0; i + 1 < mn; i++)
			superb[i] = rwork[i];

	free(rwork);
	return info;
}

lapack_int LAPACKE_cgeqrf(int layout, lapack_int m, lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_complex_float* tau)
{
	if (check_layout(layout)) return -1;
	bt_fn8 fn = BT_FN(BT_cgeqrf, bt_fn8);
	if (NULL == fn) return missing(BT_cgeqrf);

	lapack_int info = 0, lwork = -1;
	lapack_complex_float query = 0;

	fn(&m, &n, a, &lda, tau, &query, &lwork, &info);

	if (0 == info) {

		lwork = MAX(1, (lapack_int)crealf(query));
		lapack_complex_float* work = malloc(sizeof(*work) * (size_t)lwork);
		fn(&m, &n, a, &lda, tau, work, &lwork, &info);
		free(work);
	}

	return info;
}

lapack_int LAPACKE_cungqr(int layout, lapack_int m, lapack_int n, lapack_int k, lapack_complex_float* a, lapack_int lda, const lapack_complex_float* tau)
{
	if (check_layout(layout)) return -1;
	bt_fn9 fn = BT_FN(BT_cungqr, bt_fn9);
	if (NULL == fn) return missing(BT_cungqr);

	lapack_int info = 0, lwork = -1;
	lapack_complex_float query = 0;

	fn(&m, &n, &k, a, &lda, tau, &query, &lwork, &info);

	if (0 == info) {

		lwork = MAX(1, (lapack_int)crealf(query));
		lapack_complex_float* work = malloc(sizeof(*work) * (size_t)lwork);
		fn(&m, &n, &k, a, &lda, tau, work, &lwork, &info);
		free(work);
	}

	return info;
}

lapack_int LAPACKE_cpotrf(int layout, char uplo, lapack_int n, lapack_complex_float* a, lapack_int lda)
{
	if (check_layout(layout)) return -1;
	bt_fn6 fn = BT_FN(BT_cpotrf, bt_fn6);
	if (NULL == fn) return missing(BT_cpotrf);

	lapack_int info = 0;
	fn(&uplo, &n, a, &lda, &info, BT_LEN(1));
	return info;
}

lapack_int LAPACKE_ctrtri(int layout, char uplo, char diag, lapack_int n, lapack_complex_float* a, lapack_int lda)
{
	if (check_layout(layout)) return -1;
	bt_fn8 fn = BT_FN(BT_ctrtri, bt_fn8);
	if (NULL == fn) return missing(BT_ctrtri);

	lapack_int info = 0;
	fn(&uplo, &diag, &n, a, &lda, &info, BT_LEN(1), BT_LEN(1));
	return info;
}

lapack_int LAPACKE_ctrtrs(int layout, char uplo, char trans, char diag, lapack_int n, lapack_int nrhs, const lapack_complex_float* a, lapack_int lda, lapack_complex_float* b, lapack_int ldb)
{
	if (check_layout(layout)) return -1;
	bt_fn13 fn = BT_FN(BT_ctrtrs, bt_fn13);
	if (NULL == fn) return missing(BT_ctrtrs);

	lapack_int info = 0;
	fn(&uplo, &trans, &diag, &n, &nrhs, a, &lda, b, &ldb, &info, BT_LEN(1), BT_LEN(1), BT_LEN(1));
	return info;
}

lapack_int LAPACKE_cgetrf(int layout, lapack_int m, lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_int* ipiv)
{
	if (check_layout(layout)) return -1;
	bt_fn6 fn = BT_FN(BT_cgetrf, bt_fn6);
	if (NULL == fn) return missing(BT_cgetrf);

	lapack_int info = 0;
	fn(&m, &n, a, &lda, ipiv, &info);
	return info;
}

lapack_int LAPACKE_sgetrf(int layout, lapack_int m, lapack_int n, float* a, lapack_int lda, lapack_int* ipiv)
{
	if (check_layout(layout)) return -1;
	bt_fn6 fn = BT_FN(BT_sgetrf, bt_fn6);
	if (NULL == fn) return missing(BT_sgetrf);

	lapack_int info = 0;
	fn(&m, &n, a, &lda, ipiv, &info);
	return info;
}

lapack_int LAPACKE_cgetri(int layout, lapack_int n, lapack_complex_float* a, lapack_int lda, const lapack_int* ipiv)
{
	if (check_layout(layout)) return -1;
	bt_fn7 fn = BT_FN(BT_cgetri, bt_fn7);
	if (NULL == fn) return missing(BT_cgetri);

	lapack_int info = 0, lwork = -1;
	lapack_complex_float query = 0;

	fn(&n, a, &lda, ipiv, &query, &lwork, &info);

	if (0 == info) {

		lwork = MAX(1, (lapack_int)crealf(query));
		lapack_complex_float* work = malloc(sizeof(*work) * (size_t)lwork);
		fn(&n, a, &lda, ipiv, work, &lwork, &info);
		free(work);
	}

	return info;
}

lapack_int LAPACKE_sgetri(int layout, lapack_int n, float* a, lapack_int lda, const lapack_int* ipiv)
{
	if (check_layout(layout)) return -1;
	bt_fn7 fn = BT_FN(BT_sgetri, bt_fn7);
	if (NULL == fn) return missing(BT_sgetri);

	lapack_int info = 0, lwork = -1;
	float query = 0;

	fn(&n, a, &lda, ipiv, &query, &lwork, &info);

	if (0 == info) {

		lwork = MAX(1, (lapack_int)query);
		float* work = malloc(sizeof(*work) * (size_t)lwork);
		fn(&n, a, &lda, ipiv, work, &lwork, &info);
		free(work);
	}

	return info;
}

lapack_int LAPACKE_cgees(int layout, char jobvs, char sort, LAPACK_C_SELECT1 select, lapack_int n, lapack_complex_float* a, lapack_int lda, lapack_int* sdim, lapack_complex_float* w, lapack_complex_float* vs, lapack_int ldvs)
{
	if (check_layout(layout)) return -1;
	bt_fn17 fn = BT_FN(BT_cgees, bt_fn17);
	if (NULL == fn) return missing(BT_cgees);

	lapack_int info = 0, lwork = -1;
	lapack_complex_float query = 0;
	float* rwork = malloc(sizeof(float) * (size_t)MAX(1, n));
	lapack_logical* bwork = malloc(sizeof(lapack_logical) * (size_t)MAX(1, n));

	fn(&jobvs, &sort, (const void*)select, &n, a, &lda, sdim, w, vs, &ldvs, &query, &lwork, rwork, bwork, &info, BT_LEN(1), BT_LEN(1));

	if (0 == info) {

		lwork = MAX(1, (lapack_int)crealf(query));
		lapack_complex_float* work = malloc(sizeof(*work) * (size_t)lwork);
		fn(&jobvs, &sort, (const void*)select, &n, a, &lda, sdim, w, vs, &ldvs, work, &lwork, rwork, bwork, &info, BT_LEN(1), BT_LEN(1));
		free(work);
	}

	free(bwork);
	free(rwork);
	return info;
}

lapack_int LAPACKE_zgees(int layout, char jobvs, char sort, LAPACK_Z_SELECT1 select, lapack_int n, lapack_complex_double* a, lapack_int lda, lapack_int* sdim, lapack_complex_double* w, lapack_complex_double* vs, lapack_int ldvs)
{
	if (check_layout(layout)) return -1;
	bt_fn17 fn = BT_FN(BT_zgees, bt_fn17);
	if (NULL == fn) return missing(BT_zgees);

	lapack_int info = 0, lwork = -1;
	lapack_complex_double query = 0;
	double* rwork = malloc(sizeof(double) * (size_t)MAX(1, n));
	lapack_logical* bwork = malloc(sizeof(lapack_logical) * (size_t)MAX(1, n));

	fn(&jobvs, &sort, (const void*)select, &n, a, &lda, sdim, w, vs, &ldvs, &query, &lwork, rwork, bwork, &info, BT_LEN(1), BT_LEN(1));

	if (0 == info) {

		lwork = MAX(1, (lapack_int)creal(query));
		lapack_complex_double* work = malloc(sizeof(*work) * (size_t)lwork);
		fn(&jobvs, &sort, (const void*)select, &n, a, &lda, sdim, w, vs, &ldvs, work, &lwork, rwork, bwork, &info, BT_LEN(1), BT_LEN(1));
		free(work);
	}

	free(bwork);
	free(rwork);
	return info;
}

lapack_int LAPACKE_ctrsyl(int layout, char trana, char tranb, lapack_int isgn, lapack_int m, lapack_int n, const lapack_complex_float* a, lapack_int lda, const lapack_complex_float* b, lapack_int ldb, lapack_complex_float* c, lapack_int ldc, float* scale)
{
	if (check_layout(layout)) return -1;
	bt_fn15 fn = BT_FN(BT_ctrsyl, bt_fn15);
	if (NULL == fn) return missing(BT_ctrsyl);

	lapack_int info = 0;
	fn(&trana, &tranb, &isgn, &m, &n, a, &lda, b, &ldb, c, &ldc, scale, &info, BT_LEN(1), BT_LEN(1));
	return info;
}

lapack_int LAPACKE_sgesv(int layout, lapack_int n, lapack_int nrhs, float* a, lapack_int lda, lapack_int* ipiv, float* b, lapack_int ldb)
{
	if (check_layout(layout)) return -1;
	bt_fn8 fn = BT_FN(BT_sgesv, bt_fn8);
	if (NULL == fn) return missing(BT_sgesv);

	lapack_int info = 0;
	fn(&n, &nrhs, a, &lda, ipiv, b, &ldb, &info);
	return info;
}
