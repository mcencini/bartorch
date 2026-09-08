/*
 * Operators and solvers behind the C ABI.
 *
 * A host-defined operator enters BART as a linop or nlop whose callbacks
 * call back into the host; BART's own operators (Fourier, coil multiply,
 * sampling, NUFFT) are exposed as handles of the same kind, so either can
 * be chained with the other and handed to BART's least-squares and
 * Gauss-Newton solvers.  Every entry point runs under BART's error catcher,
 * so a failure inside BART returns an error code instead of ending the
 * process.
 */
#include <complex.h>
#include <stdbool.h>
#include <stdlib.h>
#include <string.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/types.h"

#include "num/iovec.h"
#include "num/multind.h"

#include "linops/fmac.h"
#include "linops/linop.h"
#include "linops/someops.h"

#include "sense/model.h"
#include "noncart/nufft.h"

#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/nlop.h"

#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/italgos.h"
#include "iter/lsqr.h"

#include "include/bartorch.h"
#include "backend.h"

struct bartorch_linop_s { const struct linop_s* op; };
struct bartorch_nlop_s { const struct nlop_s* op; };


/* --- guarded execution ---------------------------------------------------- */

struct guard_call {

	int (*fn)(void*);
	void* arg;
};

static int guard_shim(int argc, char* argv[argc])
{
	(void)argc;
	struct guard_call* c = (struct guard_call*)argv[0];
	return c->fn(c->arg);
}

/* Run fn under BART's error catcher; an error inside returns -1.
 *
 * The message is cleared first so that what the host reads afterwards belongs
 * to this call.  A nested call leaves the outer one's catcher and message in
 * place, which is what carries the reason out. */
static int guarded(int (*fn)(void*), void* arg)
{
	if (error_jumper.initialized)
		return fn(arg);

	bartorch_clear_error();

	struct guard_call c = { fn, arg };
	char* argv[1] = { (char*)&c };
	return error_catcher(guard_shim, 1, argv);
}

static bartorch_linop* wrap_linop(const struct linop_s* op)
{
	if (NULL == op)
		return NULL;

	bartorch_linop* h = xmalloc(sizeof(*h));
	h->op = op;
	return h;
}

static bartorch_nlop* wrap_nlop(const struct nlop_s* op)
{
	if (NULL == op)
		return NULL;

	bartorch_nlop* h = xmalloc(sizeof(*h));
	h->op = op;
	return h;
}

static void copy_dims(const struct iovec_s* iov, int N, long* dims)
{
	for (int i = 0; i < N; i++)
		dims[i] = (i < iov->N) ? iov->dims[i] : 1;
}


/* --- host-defined linear operator ------------------------------------------ */

struct cb_linop_data {

	linop_data_t super;
	bartorch_apply_fn forward;
	bartorch_apply_fn adjoint;
	bartorch_apply_fn normal;
	void* ctx;
	bartorch_release_fn release;
};

static DEF_TYPEID(cb_linop_data);

static void cb_lin_forward(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const struct cb_linop_data* d = CAST_DOWN(cb_linop_data, _d);

	if (0 != d->forward(d->ctx, dst, src))
		error("bartorch: forward callback failed\n");
}

static void cb_lin_adjoint(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const struct cb_linop_data* d = CAST_DOWN(cb_linop_data, _d);

	if (0 != d->adjoint(d->ctx, dst, src))
		error("bartorch: adjoint callback failed\n");
}

static void cb_lin_normal(const linop_data_t* _d, complex float* dst, const complex float* src)
{
	const struct cb_linop_data* d = CAST_DOWN(cb_linop_data, _d);

	if (0 != d->normal(d->ctx, dst, src))
		error("bartorch: normal callback failed\n");
}

static void cb_lin_del(const linop_data_t* _d)
{
	const struct cb_linop_data* d = CAST_DOWN(cb_linop_data, _d);

	if (NULL != d->release)
		d->release(d->ctx);

	xfree(d);
}

struct linop_callback_args {

	int ON; const long* odims; int IN; const long* idims;
	bartorch_apply_fn forward; bartorch_apply_fn adjoint; bartorch_apply_fn normal;
	void* ctx; bartorch_release_fn release;
	bartorch_linop* result;
};

static int linop_callback_worker(void* p)
{
	struct linop_callback_args* a = p;

	PTR_ALLOC(struct cb_linop_data, d);
	SET_TYPEID(cb_linop_data, d);
	d->forward = a->forward;
	d->adjoint = a->adjoint;
	d->normal = a->normal;
	d->ctx = a->ctx;
	d->release = a->release;

	a->result = wrap_linop(linop_create(a->ON, a->odims, a->IN, a->idims, CAST_UP(PTR_PASS(d)),
			cb_lin_forward, cb_lin_adjoint, (NULL != a->normal) ? cb_lin_normal : NULL, NULL, cb_lin_del));
	return 0;
}

bartorch_linop* bartorch_linop_callback(int ON, const long* odims, int IN, const long* idims,
		bartorch_apply_fn forward, bartorch_apply_fn adjoint, bartorch_apply_fn normal,
		void* ctx, bartorch_release_fn release)
{
	struct linop_callback_args a = { ON, odims, IN, idims, forward, adjoint, normal, ctx, release, NULL };
	return (0 == guarded(linop_callback_worker, &a)) ? a.result : NULL;
}


/* --- BART's own linear operators ------------------------------------------- */

struct linop_fft_args { int N; const long* dims; unsigned long flags; int inverse; int centered; bartorch_linop* result; };

static int linop_fft_worker(void* p)
{
	struct linop_fft_args* a = p;
	const struct linop_s* op;

	if (a->centered)
		op = a->inverse ? linop_ifftc_create(a->N, a->dims, a->flags) : linop_fftc_create(a->N, a->dims, a->flags);
	else
		op = a->inverse ? linop_ifft_create(a->N, a->dims, a->flags) : linop_fft_create(a->N, a->dims, a->flags);

	a->result = wrap_linop(op);
	return 0;
}

bartorch_linop* bartorch_linop_fft(int N, const long* dims, unsigned long flags, int inverse, int centered)
{
	struct linop_fft_args a = { N, dims, flags, inverse, centered, NULL };
	return (0 == guarded(linop_fft_worker, &a)) ? a.result : NULL;
}

struct linop_cdiag_args { int N; const long* dims; unsigned long flags; const void* diag; bartorch_linop* result; };

static int linop_cdiag_worker(void* p)
{
	struct linop_cdiag_args* a = p;
	a->result = wrap_linop(linop_cdiag_create(a->N, a->dims, a->flags, a->diag));
	return 0;
}

bartorch_linop* bartorch_linop_cdiag(int N, const long* dims, unsigned long flags, const void* diag)
{
	struct linop_cdiag_args a = { N, dims, flags, diag, NULL };
	return (0 == guarded(linop_cdiag_worker, &a)) ? a.result : NULL;
}

struct linop_fmac_args { int N; const long* odims; const long* idims; const long* tdims; const void* tensor; bartorch_linop* result; };

static int linop_fmac_worker(void* p)
{
	struct linop_fmac_args* a = p;
	a->result = wrap_linop(linop_fmac_dims_create(a->N, a->odims, a->idims, a->tdims, a->tensor));
	return 0;
}

bartorch_linop* bartorch_linop_fmac(int N, const long* odims, const long* idims, const long* tdims, const void* tensor)
{
	struct linop_fmac_args a = { N, odims, idims, tdims, tensor, NULL };
	return (0 == guarded(linop_fmac_worker, &a)) ? a.result : NULL;
}

struct linop_sampling_args { const long* dims; const long* pat_dims; const void* pattern; bartorch_linop* result; };

static int linop_sampling_worker(void* p)
{
	struct linop_sampling_args* a = p;
	a->result = wrap_linop(linop_sampling_create(a->dims, a->pat_dims, a->pattern));
	return 0;
}

bartorch_linop* bartorch_linop_sampling(const long* dims, const long* pat_dims, const void* pattern)
{
	struct linop_sampling_args a = { dims, pat_dims, pattern, NULL };
	return (0 == guarded(linop_sampling_worker, &a)) ? a.result : NULL;
}

struct linop_nufft_args {

	int N; const long* ksp_dims; const long* cim_dims; const long* traj_dims; const void* traj;
	const long* wgh_dims; const void* weights;
	const long* bas_dims; const void* basis;
	int toeplitz; float os; float width;
	bartorch_linop* result;
};

static int linop_nufft_worker(void* p)
{
	struct linop_nufft_args* a = p;

	struct nufft_conf_s conf = nufft_conf_defaults;
	conf.toeplitz = (0 != a->toeplitz);

	/* Zero is what says nobody asked, so that a caller who asks for BART's
	 * own factor of two is told apart from one who asked for nothing. */
	conf.os = (a->os > 0.f) ? a->os : 0.;

	conf.width = (a->width > 0.f) ? a->width : 0.;

	/* Weights and a subspace basis belong to the operator rather than to
	 * something chained onto it: the normal is a point spread function over
	 * both, which a chain could not be. */
	long wgh_dims[a->N];

	if (NULL == a->weights)
		md_singleton_dims(a->N, wgh_dims);
	else
		md_copy_dims(a->N, wgh_dims, a->wgh_dims);

	a->result = wrap_linop(nufft_create2(a->N, a->ksp_dims, a->cim_dims, a->traj_dims, a->traj,
			wgh_dims, a->weights, a->bas_dims, a->basis, conf));
	return 0;
}

bartorch_linop* bartorch_linop_nufft(int N, const long* ksp_dims, const long* cim_dims, const long* traj_dims,
		const void* traj, const long* wgh_dims, const void* weights,
		const long* bas_dims, const void* basis, int toeplitz, float os, float width)
{
	struct linop_nufft_args a = { N, ksp_dims, cim_dims, traj_dims, traj,
		wgh_dims, weights, bas_dims, basis, toeplitz, os, width, NULL };
	return (0 == guarded(linop_nufft_worker, &a)) ? a.result : NULL;
}

struct linop_pair_args { const bartorch_linop* a; const bartorch_linop* b; bartorch_linop* result; };

static int linop_chain_worker(void* p)
{
	struct linop_pair_args* a = p;
	a->result = wrap_linop(linop_chain(a->a->op, a->b->op));
	return 0;
}

/* b applied after a. */
bartorch_linop* bartorch_linop_chain(const bartorch_linop* a, const bartorch_linop* b)
{
	struct linop_pair_args args = { a, b, NULL };
	return (0 == guarded(linop_chain_worker, &args)) ? args.result : NULL;
}

static int linop_plus_worker(void* p)
{
	struct linop_pair_args* a = p;
	a->result = wrap_linop(linop_plus(a->a->op, a->b->op));
	return 0;
}

bartorch_linop* bartorch_linop_plus(const bartorch_linop* a, const bartorch_linop* b)
{
	struct linop_pair_args args = { a, b, NULL };
	return (0 == guarded(linop_plus_worker, &args)) ? args.result : NULL;
}

int bartorch_linop_domain(const bartorch_linop* h, int N, long* dims)
{
	const struct iovec_s* iov = linop_domain(h->op);
	copy_dims(iov, N, dims);
	return iov->N;
}

int bartorch_linop_codomain(const bartorch_linop* h, int N, long* dims)
{
	const struct iovec_s* iov = linop_codomain(h->op);
	copy_dims(iov, N, dims);
	return iov->N;
}

struct linop_apply_args { const bartorch_linop* h; void* dst; const void* src; int mode; };

static int linop_apply_worker(void* p)
{
	struct linop_apply_args* a = p;

	switch (a->mode) {
	case 0: linop_forward_unchecked(a->h->op, a->dst, a->src); break;
	case 1: linop_adjoint_unchecked(a->h->op, a->dst, a->src); break;
	default: linop_normal_unchecked(a->h->op, a->dst, a->src); break;
	}

	return 0;
}

int bartorch_linop_forward(const bartorch_linop* h, void* dst, const void* src)
{
	struct linop_apply_args a = { h, dst, src, 0 };
	return guarded(linop_apply_worker, &a);
}

int bartorch_linop_adjoint(const bartorch_linop* h, void* dst, const void* src)
{
	struct linop_apply_args a = { h, dst, src, 1 };
	return guarded(linop_apply_worker, &a);
}

int bartorch_linop_normal(const bartorch_linop* h, void* dst, const void* src)
{
	struct linop_apply_args a = { h, dst, src, 2 };
	return guarded(linop_apply_worker, &a);
}

void bartorch_linop_free(bartorch_linop* h)
{
	if (NULL == h)
		return;

	linop_free(h->op);
	xfree(h);
}


/* --- least squares ---------------------------------------------------------- */

struct lsqr_args {

	const bartorch_linop* A; int maxiter; float lambda; float tol; int warmstart;
	void* x; const void* y;
};

static int lsqr_worker(void* p)
{
	struct lsqr_args* a = p;

	struct iter_conjgrad_conf cg = iter_conjgrad_defaults;
	cg.maxiter = a->maxiter;
	cg.l2lambda = a->lambda;
	cg.tol = a->tol;

	struct lsqr_conf conf = lsqr_defaults;
	conf.warmstart = (0 != a->warmstart);

	const struct iovec_s* dom = linop_domain(a->A->op);
	const struct iovec_s* cod = linop_codomain(a->A->op);

	lsqr(dom->N, &conf, iter_conjgrad, CAST_UP(&cg), a->A->op, NULL, dom->dims, a->x, cod->dims, a->y, NULL);
	return 0;
}

int bartorch_lsqr(const bartorch_linop* A, int maxiter, float lambda, float tol, int warmstart, void* x, const void* y)
{
	struct lsqr_args a = { A, maxiter, lambda, tol, warmstart, x, y };
	return guarded(lsqr_worker, &a);
}


/* --- host-defined nonlinear operator ---------------------------------------- */

struct cb_nlop_data {

	nlop_data_t super;
	bartorch_apply_fn forward;
	bartorch_apply_fn derivative;
	bartorch_apply_fn adjoint;
	void* ctx;
	bartorch_release_fn release;
};

static DEF_TYPEID(cb_nlop_data);

static void cb_nl_forward(const nlop_data_t* _d, complex float* dst, const complex float* src)
{
	const struct cb_nlop_data* d = CAST_DOWN(cb_nlop_data, _d);

	if (0 != d->forward(d->ctx, dst, src))
		error("bartorch: forward callback failed\n");
}

static void cb_nl_derivative(const nlop_data_t* _d, int o, int i, complex float* dst, const complex float* src)
{
	(void)o; (void)i;
	const struct cb_nlop_data* d = CAST_DOWN(cb_nlop_data, _d);

	if (0 != d->derivative(d->ctx, dst, src))
		error("bartorch: derivative callback failed\n");
}

static void cb_nl_adjoint(const nlop_data_t* _d, int o, int i, complex float* dst, const complex float* src)
{
	(void)o; (void)i;
	const struct cb_nlop_data* d = CAST_DOWN(cb_nlop_data, _d);

	if (0 != d->adjoint(d->ctx, dst, src))
		error("bartorch: adjoint callback failed\n");
}

static void cb_nl_del(const nlop_data_t* _d)
{
	const struct cb_nlop_data* d = CAST_DOWN(cb_nlop_data, _d);

	if (NULL != d->release)
		d->release(d->ctx);

	xfree(d);
}

struct nlop_callback_args {

	int ON; const long* odims; int IN; const long* idims;
	bartorch_apply_fn forward; bartorch_apply_fn derivative; bartorch_apply_fn adjoint;
	void* ctx; bartorch_release_fn release;
	bartorch_nlop* result;
};

static int nlop_callback_worker(void* p)
{
	struct nlop_callback_args* a = p;

	PTR_ALLOC(struct cb_nlop_data, d);
	SET_TYPEID(cb_nlop_data, d);
	d->super.clear_der = NULL;
	d->super.data_der = NULL;
	d->forward = a->forward;
	d->derivative = a->derivative;
	d->adjoint = a->adjoint;
	d->ctx = a->ctx;
	d->release = a->release;

	a->result = wrap_nlop(nlop_create(a->ON, a->odims, a->IN, a->idims, CAST_UP(PTR_PASS(d)),
			cb_nl_forward, cb_nl_derivative, cb_nl_adjoint, NULL, NULL, cb_nl_del));
	return 0;
}

bartorch_nlop* bartorch_nlop_callback(int ON, const long* odims, int IN, const long* idims,
		bartorch_apply_fn forward, bartorch_apply_fn derivative, bartorch_apply_fn adjoint,
		void* ctx, bartorch_release_fn release)
{
	struct nlop_callback_args a = { ON, odims, IN, idims, forward, derivative, adjoint, ctx, release, NULL };
	return (0 == guarded(nlop_callback_worker, &a)) ? a.result : NULL;
}

struct nlop_from_linop_args { const bartorch_linop* lin; bartorch_nlop* result; };

static int nlop_from_linop_worker(void* p)
{
	struct nlop_from_linop_args* a = p;
	a->result = wrap_nlop(nlop_from_linop(a->lin->op));
	return 0;
}

bartorch_nlop* bartorch_nlop_from_linop(const bartorch_linop* lin)
{
	struct nlop_from_linop_args a = { lin, NULL };
	return (0 == guarded(nlop_from_linop_worker, &a)) ? a.result : NULL;
}

struct nlop_pair_args { const bartorch_nlop* a; const bartorch_nlop* b; bartorch_nlop* result; };

static int nlop_chain_worker(void* p)
{
	struct nlop_pair_args* a = p;
	a->result = wrap_nlop(nlop_chain(a->a->op, a->b->op));
	return 0;
}

/* b applied after a. */
bartorch_nlop* bartorch_nlop_chain(const bartorch_nlop* a, const bartorch_nlop* b)
{
	struct nlop_pair_args args = { a, b, NULL };
	return (0 == guarded(nlop_chain_worker, &args)) ? args.result : NULL;
}

int bartorch_nlop_domain(const bartorch_nlop* h, int N, long* dims)
{
	const struct iovec_s* iov = nlop_domain(h->op);
	copy_dims(iov, N, dims);
	return iov->N;
}

int bartorch_nlop_codomain(const bartorch_nlop* h, int N, long* dims)
{
	const struct iovec_s* iov = nlop_codomain(h->op);
	copy_dims(iov, N, dims);
	return iov->N;
}

struct nlop_apply_args { const bartorch_nlop* h; void* dst; const void* src; int mode; };

static int nlop_apply_worker(void* p)
{
	struct nlop_apply_args* a = p;
	const struct iovec_s* dom = nlop_domain(a->h->op);
	const struct iovec_s* cod = nlop_codomain(a->h->op);

	switch (a->mode) {
	case 0: nlop_apply(a->h->op, cod->N, cod->dims, a->dst, dom->N, dom->dims, a->src); break;
	case 1: nlop_derivative(a->h->op, cod->N, cod->dims, a->dst, dom->N, dom->dims, a->src); break;
	default: nlop_adjoint(a->h->op, dom->N, dom->dims, a->dst, cod->N, cod->dims, a->src); break;
	}

	return 0;
}

int bartorch_nlop_apply(const bartorch_nlop* h, void* dst, const void* src)
{
	struct nlop_apply_args a = { h, dst, src, 0 };
	return guarded(nlop_apply_worker, &a);
}

int bartorch_nlop_derivative(const bartorch_nlop* h, void* dst, const void* src)
{
	struct nlop_apply_args a = { h, dst, src, 1 };
	return guarded(nlop_apply_worker, &a);
}

int bartorch_nlop_adjoint(const bartorch_nlop* h, void* dst, const void* src)
{
	struct nlop_apply_args a = { h, dst, src, 2 };
	return guarded(nlop_apply_worker, &a);
}

void bartorch_nlop_free(bartorch_nlop* h)
{
	if (NULL == h)
		return;

	nlop_free(h->op);
	xfree(h);
}


/* --- Gauss-Newton ------------------------------------------------------------ */

struct irgnm_args {

	const bartorch_nlop* F; int iter; float alpha; float alpha_min; float redu; int cgiter; float cgtol;
	void* x; const void* y; const void* xref;
};

static int irgnm_worker(void* p)
{
	struct irgnm_args* a = p;

	struct iter3_irgnm_conf conf = iter3_irgnm_defaults;
	conf.iter = a->iter;
	conf.alpha = a->alpha;
	conf.alpha_min = a->alpha_min;
	conf.redu = a->redu;
	conf.cgiter = a->cgiter;
	conf.cgtol = a->cgtol;

	const struct iovec_s* dom = nlop_domain(a->F->op);
	const struct iovec_s* cod = nlop_codomain(a->F->op);

	long N = 2 * md_calc_size(dom->N, dom->dims);
	long M = 2 * md_calc_size(cod->N, cod->dims);

	iter4_irgnm(CAST_UP(&conf), a->F->op, N, a->x, a->xref, M, a->y, NULL, (struct iter_op_s){ NULL, NULL });
	return 0;
}

int bartorch_irgnm(const bartorch_nlop* F, int iter, float alpha, float alpha_min, float redu, int cgiter, float cgtol,
		void* x, const void* y, const void* xref)
{
	struct irgnm_args a = { F, iter, alpha, alpha_min, redu, cgiter, cgtol, x, y, xref };
	return guarded(irgnm_worker, &a);
}
