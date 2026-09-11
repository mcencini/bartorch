#include <string.h>

#include "backend.h"
#include "include/bartorch.h"

const char* const bartorch_fn_names[BT_FN_COUNT] = {
	[BT_sgemm] = "sgemm_", [BT_cgemm] = "cgemm_",
	[BT_sgemv] = "sgemv_", [BT_cgemv] = "cgemv_",
	[BT_sger] = "sger_", [BT_cgeru] = "cgeru_",
	[BT_saxpy] = "saxpy_", [BT_caxpy] = "caxpy_",
	[BT_sscal] = "sscal_", [BT_cscal] = "cscal_",
	[BT_csyrk] = "csyrk_",
	[BT_cheev] = "cheev_", [BT_chegv] = "chegv_",
	[BT_cgesdd] = "cgesdd_", [BT_cgesvd] = "cgesvd_",
	[BT_cgeqrf] = "cgeqrf_", [BT_cungqr] = "cungqr_",
	[BT_zheev] = "zheev_", [BT_zgesdd] = "zgesdd_",
	[BT_cpotrf] = "cpotrf_", [BT_ctrtri] = "ctrtri_", [BT_ctrtrs] = "ctrtrs_",
	[BT_cgetrf] = "cgetrf_", [BT_cgetri] = "cgetri_",
	[BT_sgetrf] = "sgetrf_", [BT_sgetri] = "sgetri_",
	[BT_cgees] = "cgees_", [BT_zgees] = "zgees_",
	[BT_ctrsyl] = "ctrsyl_", [BT_sgesv] = "sgesv_",
};

static void* const fallbacks[BT_FN_COUNT] = {
	[BT_sgemm] = (void*)bartorch_ref_sgemm, [BT_cgemm] = (void*)bartorch_ref_cgemm,
	[BT_sgemv] = (void*)bartorch_ref_sgemv, [BT_cgemv] = (void*)bartorch_ref_cgemv,
	[BT_sger] = (void*)bartorch_ref_sger, [BT_cgeru] = (void*)bartorch_ref_cgeru,
	[BT_saxpy] = (void*)bartorch_ref_saxpy, [BT_caxpy] = (void*)bartorch_ref_caxpy,
	[BT_sscal] = (void*)bartorch_ref_sscal, [BT_cscal] = (void*)bartorch_ref_cscal,
	[BT_csyrk] = (void*)bartorch_ref_csyrk,
};

void* bartorch_fn_table[BT_FN_COUNT] = {
	[BT_sgemm] = (void*)bartorch_ref_sgemm, [BT_cgemm] = (void*)bartorch_ref_cgemm,
	[BT_sgemv] = (void*)bartorch_ref_sgemv, [BT_cgemv] = (void*)bartorch_ref_cgemv,
	[BT_sger] = (void*)bartorch_ref_sger, [BT_cgeru] = (void*)bartorch_ref_cgeru,
	[BT_saxpy] = (void*)bartorch_ref_saxpy, [BT_caxpy] = (void*)bartorch_ref_caxpy,
	[BT_sscal] = (void*)bartorch_ref_sscal, [BT_cscal] = (void*)bartorch_ref_cscal,
	[BT_csyrk] = (void*)bartorch_ref_csyrk,
};

static int find(const char* symbol)
{
	for (int i = 0; i < BT_FN_COUNT; i++)
		if (0 == strcmp(symbol, bartorch_fn_names[i]))
			return i;

	return -1;
}

int bartorch_backend_set(const char* symbol, void* fn)
{
	int i = find(symbol);

	if (i < 0)
		return -1;

	bartorch_fn_table[i] = (NULL != fn) ? fn : fallbacks[i];
	return 0;
}

int bartorch_backend_count(void)
{
	return BT_FN_COUNT;
}

const char* bartorch_backend_name(int index)
{
	return ((0 <= index) && (index < BT_FN_COUNT)) ? bartorch_fn_names[index] : NULL;
}

int bartorch_backend_has_fallback(int index)
{
	return ((0 <= index) && (index < BT_FN_COUNT)) ? (NULL != fallbacks[index]) : 0;
}
