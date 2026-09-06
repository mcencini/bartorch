/*
 * bartorch C ABI.
 *
 * This is the only interface the host language sees.  It is plain C:
 * no complex types, no variable-length arrays, no GNU extensions, so it
 * can be consumed by ctypes, by an MSVC-built extension or by any C++
 * translation unit.  Every array is described by a data pointer and a
 * BART-order (Fortran, first index fastest) dimension vector; the host
 * owns every buffer it registers and every buffer the allocator callback
 * hands out.
 */
#ifndef BARTORCH_H
#define BARTORCH_H

#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
#define BARTORCH_API __declspec(dllexport)
#else
#define BARTORCH_API __attribute__((visibility("default")))
#endif

/* Number of dimensions BART carries for every array. */
#define BARTORCH_DIMS 16

/* Log levels, matching BART's enum debug_levels. */
enum bartorch_log_level {
	BARTORCH_LOG_ERROR = 0,
	BARTORCH_LOG_WARN = 1,
	BARTORCH_LOG_INFO = 2,
	BARTORCH_LOG_DEBUG1 = 3,
	BARTORCH_LOG_DEBUG2 = 4,
	BARTORCH_LOG_DEBUG3 = 5,
	BARTORCH_LOG_DEBUG4 = 6,
	BARTORCH_LOG_TRACE = 7,
};

/*
 * Allocate a complex-float array of the given BART dimensions and return
 * its data pointer.  The host keeps the buffer alive until the matching
 * free callback receives the same pointer.  Returning NULL aborts the
 * running command with an error.
 */
typedef void* (*bartorch_alloc_fn)(void* ctx, int D, const long* dims);
typedef void (*bartorch_free_fn)(void* ctx, void* data);

/* Receives every BART log line at or below the current debug level. */
typedef void (*bartorch_log_fn)(void* ctx, int level, const char* func,
		const char* file, int line, const char* msg);

BARTORCH_API const char* bartorch_bart_version(void);
BARTORCH_API const char* bartorch_build_info(void);

BARTORCH_API void bartorch_set_allocator(bartorch_alloc_fn alloc, bartorch_free_fn free_, void* ctx);
BARTORCH_API void bartorch_set_log_handler(bartorch_log_fn fn, void* ctx);
BARTORCH_API void bartorch_set_debug_level(int level);
BARTORCH_API int bartorch_get_debug_level(void);
BARTORCH_API void bartorch_set_num_threads(int n);

/*
 * In-memory array registry.  A name ending in ".mem" given to any BART
 * command resolves here instead of on disk.  Registered arrays are never
 * freed by BART; arrays BART creates itself come from the allocator
 * callback and are released through the free callback on unlink.
 */
BARTORCH_API int bartorch_register(const char* name, int D, const long* dims, void* data);
BARTORCH_API int bartorch_exists(const char* name);
BARTORCH_API int bartorch_lookup(const char* name, int D, long* dims, void** data);
BARTORCH_API int bartorch_unlink(const char* name);
BARTORCH_API int bartorch_unlink_all(void);

/*
 * Run one BART command in-process.  argv[0] is the tool name.  Text the
 * tool prints to standard output lands in `out`; the last error message
 * lands in `err`.  Returns the tool's exit code.
 */
BARTORCH_API int bartorch_command(int argc, const char* const* argv,
		char* out, size_t outlen, char* err, size_t errlen);

/*
 * BLAS and LAPACK backend.  Each entry is a Fortran-ABI routine
 * ("cgemm_", "cgesdd_", ...): every argument by pointer, character
 * arguments followed by hidden size_t lengths.  An entry may be the
 * address of a symbol exported by a library already in the process or a
 * callback implemented by the host.  Passing NULL restores the built-in
 * reference implementation where one exists.
 */
BARTORCH_API int bartorch_backend_set(const char* symbol, void* fn);
BARTORCH_API int bartorch_backend_count(void);
BARTORCH_API const char* bartorch_backend_name(int index);
BARTORCH_API int bartorch_backend_has_fallback(int index);

/*
 * Operators.  A handle wraps one BART linear or nonlinear operator.  Both
 * kinds can be built from host callbacks, in which case the host owns the
 * memory behind the callbacks' pointers for the duration of each call, and
 * both can be chained with each other and solved against.  Dimension
 * vectors are BART order; every callback receives raw complex-float buffers
 * sized by the dimensions the operator was created with and returns 0 on
 * success.
 */
typedef struct bartorch_linop_s bartorch_linop;
typedef struct bartorch_nlop_s bartorch_nlop;
typedef int (*bartorch_apply_fn)(void* ctx, void* dst, const void* src);
typedef void (*bartorch_release_fn)(void* ctx);

BARTORCH_API bartorch_linop* bartorch_linop_callback(int ON, const long* odims, int IN, const long* idims,
		bartorch_apply_fn forward, bartorch_apply_fn adjoint, bartorch_apply_fn normal,
		void* ctx, bartorch_release_fn release);
BARTORCH_API bartorch_linop* bartorch_linop_fft(int N, const long* dims, unsigned long flags, int inverse, int centered);
BARTORCH_API bartorch_linop* bartorch_linop_cdiag(int N, const long* dims, unsigned long flags, const void* diag);
BARTORCH_API bartorch_linop* bartorch_linop_fmac(int N, const long* odims, const long* idims, const long* tdims, const void* tensor);
BARTORCH_API bartorch_linop* bartorch_linop_sampling(const long* dims, const long* pat_dims, const void* pattern);
BARTORCH_API bartorch_linop* bartorch_linop_nufft(int N, const long* ksp_dims, const long* cim_dims, const long* traj_dims,
		const void* traj, int toeplitz, float os, float width);
BARTORCH_API bartorch_linop* bartorch_linop_chain(const bartorch_linop* a, const bartorch_linop* b);
BARTORCH_API bartorch_linop* bartorch_linop_plus(const bartorch_linop* a, const bartorch_linop* b);
BARTORCH_API int bartorch_linop_domain(const bartorch_linop* h, int N, long* dims);
BARTORCH_API int bartorch_linop_codomain(const bartorch_linop* h, int N, long* dims);
BARTORCH_API int bartorch_linop_forward(const bartorch_linop* h, void* dst, const void* src);
BARTORCH_API int bartorch_linop_adjoint(const bartorch_linop* h, void* dst, const void* src);
BARTORCH_API int bartorch_linop_normal(const bartorch_linop* h, void* dst, const void* src);
BARTORCH_API void bartorch_linop_free(bartorch_linop* h);

/* x = argmin ||A x - y||^2 + lambda ||x||^2 by conjugate gradients on the normal equations. */
BARTORCH_API int bartorch_lsqr(const bartorch_linop* A, int maxiter, float lambda, float tol, int warmstart,
		void* x, const void* y);

BARTORCH_API bartorch_nlop* bartorch_nlop_callback(int ON, const long* odims, int IN, const long* idims,
		bartorch_apply_fn forward, bartorch_apply_fn derivative, bartorch_apply_fn adjoint,
		void* ctx, bartorch_release_fn release);
BARTORCH_API bartorch_nlop* bartorch_nlop_from_linop(const bartorch_linop* lin);
BARTORCH_API bartorch_nlop* bartorch_nlop_chain(const bartorch_nlop* a, const bartorch_nlop* b);
BARTORCH_API int bartorch_nlop_domain(const bartorch_nlop* h, int N, long* dims);
BARTORCH_API int bartorch_nlop_codomain(const bartorch_nlop* h, int N, long* dims);
BARTORCH_API int bartorch_nlop_apply(const bartorch_nlop* h, void* dst, const void* src);
BARTORCH_API int bartorch_nlop_derivative(const bartorch_nlop* h, void* dst, const void* src);
BARTORCH_API int bartorch_nlop_adjoint(const bartorch_nlop* h, void* dst, const void* src);
BARTORCH_API void bartorch_nlop_free(bartorch_nlop* h);

/* Iteratively regularised Gauss-Newton: x starts at its initial value and returns the solution. */
BARTORCH_API int bartorch_irgnm(const bartorch_nlop* F, int iter, float alpha, float alpha_min, float redu,
		int cgiter, float cgtol, void* x, const void* y, const void* xref);

#ifdef __cplusplus
}
#endif

#endif
