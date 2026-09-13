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

#include "num/flpmath.h"
#include "num/iovec.h"
#include "num/multind.h"

#include "linops/fmac.h"
#include "linops/linop.h"
#include "linops/someops.h"
#include "linops/sum.h"
#include "linops/grad.h"

#include "sense/model.h"
#include "noncart/nufft.h"

#include "noir/model2.h"
#include "noir/model_net.h"

#include "nlops/cast.h"
#include "nlops/chain.h"
#include "nlops/const.h"
#include "nlops/nlop.h"
#include "nlops/someops.h"
#include "nlops/stack.h"
#include "nlops/tenmul.h"
#include "nlops/zexp.h"

#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/iter3.h"
#include "iter/iter4.h"
#include "iter/italgos.h"
#include "iter/lsqr.h"
#include "iter/misc.h"

#include "include/bartorch.h"
#include "substitute/backend.h"

struct bartorch_linop_s { const struct linop_s* op; };

/* What is behind a handle, for the translation units that drive BART's own
 * solvers with it.  The struct stays private to this file. */
const struct linop_s* bartorch_linop_unwrap(const struct bartorch_linop_s* h)
{
	return (NULL == h) ? NULL : h->op;
}
struct bartorch_nlop_s { const struct nlop_s* op; };

struct bartorch_noir_s { struct noir2_s model; };

struct bartorch_noir_net_s {

	struct noir2_net_config_s* config;
	struct noir2_net_s* model;
	int noncart;
	/* The config keeps pointers to these three rather than copies,
	 * so they have to outlive it. */
	struct nufft_conf_s nufft_conf;
	complex float* basis;
	complex float* mask;
};


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

/* An operator as a handle the ABI can hand out.  Not static: `iter.c` wraps
 * the transform a regularization term carries. */
bartorch_linop* bartorch_linop_wrap(const struct linop_s* op)
{
	if (NULL == op)
		return NULL;

	bartorch_linop* h = xmalloc(sizeof(*h));
	h->op = op;
	return h;
}

static bartorch_linop* wrap_linop(const struct linop_s* op)
{
	return bartorch_linop_wrap(op);
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

/* Sensitivities, either as maps or as the kernels they band-limit to. */
extern const struct linop_s* bartorch_sense_operator(const long max_dims[DIMS], const long sens_dims[DIMS],
		const _Complex float* sens, int kernels, const long ksp_dims[DIMS],
		const long traj_dims[DIMS], const _Complex float* traj,
		const long wgh_dims[DIMS], const _Complex float* weights,
		const long bas_dims[DIMS], const _Complex float* basis,
		const struct nufft_conf_s* conf, int modulated);

struct linop_sense_args {

	const long* max_dims; const long* ksp_dims;
	const long* sens_dims; const void* sens; int kernels;
	const long* traj_dims; const void* traj;
	const long* wgh_dims; const void* weights;
	const long* bas_dims; const void* basis;
	int toeplitz; int modulated;
	bartorch_linop* result;
};

static int linop_sense_worker(void* p)
{
	struct linop_sense_args* a = p;

	struct nufft_conf_s conf = nufft_conf_defaults;
	conf.toeplitz = (0 != a->toeplitz);
	conf.os = 0.;
	conf.width = 0.;

	a->result = wrap_linop(bartorch_sense_operator(a->max_dims, a->sens_dims, a->sens, a->kernels,
				a->ksp_dims, a->traj_dims, a->traj,
				a->wgh_dims, a->weights, a->bas_dims, a->basis, &conf, a->modulated));
	return 0;
}

bartorch_linop* bartorch_linop_sense(const long* max_dims, const long* ksp_dims,
		const long* sens_dims, const void* sens, int kernels,
		const long* traj_dims, const void* traj,
		const long* wgh_dims, const void* weights,
		const long* bas_dims, const void* basis, int toeplitz, int modulated)
{
	struct linop_sense_args a = { max_dims, ksp_dims, sens_dims, sens, kernels,
		traj_dims, traj, wgh_dims, weights, bas_dims, basis, toeplitz, modulated, NULL };
	return (0 == guarded(linop_sense_worker, &a)) ? a.result : NULL;
}

/* The coil multiply alone, over sensitivities held as maps or as kernels. */
extern const struct linop_s* bartorch_coils_operator(const long max_dims[DIMS], const long sens_dims[DIMS],
		const _Complex float* sens, int kernels);

struct linop_coils_args {

	const long* max_dims; const long* sens_dims; const void* sens; int kernels;
	bartorch_linop* result;
};

static int linop_coils_worker(void* p)
{
	struct linop_coils_args* a = p;

	a->result = wrap_linop(bartorch_coils_operator(a->max_dims, a->sens_dims, a->sens, a->kernels));
	return 0;
}

bartorch_linop* bartorch_linop_coils(const long* max_dims, const long* sens_dims, const void* sens, int kernels)
{
	struct linop_coils_args a = { max_dims, sens_dims, sens, kernels, NULL };
	return (0 == guarded(linop_coils_worker, &a)) ? a.result : NULL;
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

static int linop_with_normal_worker(void* p)
{
	struct linop_pair_args* a = p;

	auto dom = linop_domain(a->a->op);
	auto nrm_dom = linop_domain(a->b->op);
	auto nrm_cod = linop_codomain(a->b->op);

	/* `linop_from_ops` asserts this, and an assertion here would take the
	 * process rather than the call. */
	if (!md_check_equal_dims(dom->N, dom->dims, nrm_dom->dims, ~0UL)
	 || !md_check_equal_dims(dom->N, dom->dims, nrm_cod->dims, ~0UL))
		error("bartorch: a normal operator maps the domain to itself\n");

	/* `linop_from_ops` takes its own reference to each of these. */
	a->result = wrap_linop(linop_from_ops(a->a->op->forward, a->a->op->adjoint,
				a->b->op->forward, NULL));
	return 0;
}

/* The operator `a`, answering `normal` when it is asked for A^H A.
 *
 * BART derives a normal by chaining the adjoint onto the forward, which is
 * the two applications.  Where the product has a closed form -- a sampling
 * pattern and a subspace basis collapse into one kernel applied between the
 * transforms, and the frames never have to be made -- this is how that form
 * is attached.  Both sides stay BART operators, so what a solver drives is
 * still one operator in BART's own loop.
 */
bartorch_linop* bartorch_linop_with_normal(const bartorch_linop* a, const bartorch_linop* normal)
{
	struct linop_pair_args args = { a, normal, NULL };
	return (0 == guarded(linop_with_normal_worker, &args)) ? args.result : NULL;
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

struct linop_unary_args { const bartorch_linop* a; bartorch_linop* result; };

struct linop_stack_cod_args { int n; const bartorch_linop** ops; int stack_dim; bartorch_linop* result; };

static int linop_stack_cod_worker(void* p)
{
	struct linop_stack_cod_args* a = p;

	const struct linop_s* ops[a->n];

	for (int i = 0; i < a->n; i++)
		ops[i] = a->ops[i]->op;

	a->result = wrap_linop(linop_stack_cod(a->n, ops, a->stack_dim));
	return 0;
}

/* One domain, codomains laid end to end along stack_dim: what stacking
 * operators means, and what a solver then drives as a single operator. */
bartorch_linop* bartorch_linop_stack_cod(int n, const bartorch_linop** ops, int stack_dim)
{
	if (1 > n)
		error("stacking needs at least one operator\n");

	struct linop_stack_cod_args a = { n, ops, stack_dim, NULL };
	return (0 == guarded(linop_stack_cod_worker, &a)) ? a.result : NULL;
}

struct linop_stack_args { int cod_dim; int dom_dim; const bartorch_linop* a; const bartorch_linop* b; bartorch_linop* result; };

static int linop_stack_worker(void* p)
{
	struct linop_stack_args* s = p;
	s->result = wrap_linop(linop_stack(s->cod_dim, s->dom_dim, s->a->op, s->b->op));
	return 0;
}

/* Both sides stacked: separate inputs to separate outputs, which is a block
 * diagonal. */
bartorch_linop* bartorch_linop_stack(int cod_dim, int dom_dim, const bartorch_linop* a, const bartorch_linop* b)
{
	struct linop_stack_args s = { cod_dim, dom_dim, a, b, NULL };
	return (0 == guarded(linop_stack_worker, &s)) ? s.result : NULL;
}

static int linop_adjoint_op_worker(void* p)
{
	struct linop_unary_args* a = p;
	a->result = wrap_linop(linop_get_adjoint(a->a->op));
	return 0;
}

/* A^H as an operator, sharing A's operators rather than wrapping it. */
bartorch_linop* bartorch_linop_adjoint_op(const bartorch_linop* a)
{
	struct linop_unary_args args = { a, NULL };
	return (0 == guarded(linop_adjoint_op_worker, &args)) ? args.result : NULL;
}

static int linop_normal_op_worker(void* p)
{
	struct linop_unary_args* a = p;

	/* linop_create composes forward and adjoint when a constructor gives no
	 * normal of its own, so this holds for everything reachable from here;
	 * say so rather than dereference a null if some constructor does not. */
	if (NULL == a->a->op->normal)
		error("this operator carries no normal operator\n");

	a->result = wrap_linop(linop_get_normal(a->a->op));
	return 0;
}

/* A^H A as an operator, which for an encoding built with toeplitz=true is the
 * point-spread convolution rather than the two applications. */
bartorch_linop* bartorch_linop_normal_op(const bartorch_linop* a)
{
	struct linop_unary_args args = { a, NULL };
	return (0 == guarded(linop_normal_op_worker, &args)) ? args.result : NULL;
}

struct linop_scale_args { int N; const long* dims; float re; float im; bartorch_linop* result; };

static int linop_scale_worker(void* p)
{
	struct linop_scale_args* a = p;
	a->result = wrap_linop(linop_scale_create(a->N, a->dims, a->re + a->im * I));
	return 0;
}

/* The scale is passed as two floats: a complex float is not in the ABI. */
bartorch_linop* bartorch_linop_scale(int N, const long* dims, float re, float im)
{
	struct linop_scale_args args = { N, dims, re, im, NULL };
	return (0 == guarded(linop_scale_worker, &args)) ? args.result : NULL;
}

struct linop_dims_args { int N; const long* dims; bartorch_linop* result; };

static int linop_zconj_worker(void* p)
{
	struct linop_dims_args* a = p;
	a->result = wrap_linop(linop_zconj_create(a->N, a->dims));
	return 0;
}

bartorch_linop* bartorch_linop_zconj(int N, const long* dims)
{
	struct linop_dims_args args = { N, dims, NULL };
	return (0 == guarded(linop_zconj_worker, &args)) ? args.result : NULL;
}

static int linop_identity_worker(void* p)
{
	struct linop_dims_args* a = p;
	a->result = wrap_linop(linop_identity_create(a->N, a->dims));
	return 0;
}

bartorch_linop* bartorch_linop_identity(int N, const long* dims)
{
	struct linop_dims_args args = { N, dims, NULL };
	return (0 == guarded(linop_identity_worker, &args)) ? args.result : NULL;
}

struct linop_null_args { int NO; const long* odims; int NI; const long* idims; bartorch_linop* result; };

static int linop_null_worker(void* p)
{
	struct linop_null_args* a = p;
	a->result = wrap_linop(linop_null_create(a->NO, a->odims, a->NI, a->idims));
	return 0;
}

bartorch_linop* bartorch_linop_null(int NO, const long* odims, int NI, const long* idims)
{
	struct linop_null_args args = { NO, odims, NI, idims, NULL };
	return (0 == guarded(linop_null_worker, &args)) ? args.result : NULL;
}

struct linop_norm_args { const bartorch_linop* a; double result; };

static int linop_norm_worker(void* p)
{
	struct linop_norm_args* a = p;

	/* The power iteration runs on A^H A.  An operator that does not carry
	 * its own normal gets one composed here, the way linop_stack does. */
	const struct operator_s* normal = a->a->op->normal;
	bool composed = (NULL == normal);

	if (composed)
		normal = operator_chain(a->a->op->forward, a->a->op->adjoint);

	a->result = estimate_maxeigenval(normal);

	if (composed)
		operator_free(normal);

	return 0;
}

/* The largest eigenvalue of A^H A, so that the spectral norm of A is its
 * square root.  BART estimates it by a power iteration from a random start,
 * drawn from its process-global generator: the answer moves slightly from one
 * call to the next.  A negative return means the estimate failed. */
double bartorch_linop_maxeigen(const bartorch_linop* a)
{
	struct linop_norm_args args = { a, -1. };
	return (0 == guarded(linop_norm_worker, &args)) ? args.result : -1.;
}

/* Whether BART can solve (A^H A + lambda I) x = b in closed form for this
 * operator.  A constructor says so by giving linop_create a norm_inv, and in
 * BART as it stands only the sum and average operators do -- everything else
 * leaves it null, and linop_pseudo_inv would assert on it.  So this is asked
 * before the call rather than discovered by aborting inside it. */
/* --- operators that rearrange, reduce or restrict a shape ------------------
 *
 * Each is one BART constructor.  Shapes and per-axis vectors arrive already
 * reversed and padded to DIMS by the host, so nothing here reorders anything.
 */

struct linop_flagged_args { int N; const long* dims; unsigned long flags; const void* data; bartorch_linop* result; };

static int linop_rdiag_worker(void* p)
{
	struct linop_flagged_args* a = p;
	a->result = wrap_linop(linop_rdiag_create(a->N, a->dims, a->flags, a->data));
	return 0;
}

/* md_zrmul: real parts by real parts and imaginary by imaginary, which is a
 * diagonal on each of the two components and not a real-valued diagonal. */
bartorch_linop* bartorch_linop_rdiag(int N, const long* dims, unsigned long flags, const void* diag)
{
	struct linop_flagged_args a = { N, dims, flags, diag, NULL };
	return (0 == guarded(linop_rdiag_worker, &a)) ? a.result : NULL;
}

struct linop_matrix_args { int N; const long* odims; const long* idims; const long* mdims; const void* matrix; bartorch_linop* result; };

static int linop_matrix_worker(void* p)
{
	struct linop_matrix_args* a = p;
	a->result = wrap_linop(linop_matrix_create(a->N, a->odims, a->idims, a->mdims, a->matrix));
	return 0;
}

bartorch_linop* bartorch_linop_matrix(int N, const long* odims, const long* idims, const long* mdims, const void* matrix)
{
	struct linop_matrix_args a = { N, odims, idims, mdims, matrix, NULL };
	return (0 == guarded(linop_matrix_worker, &a)) ? a.result : NULL;
}

struct linop_conv_args { int N; unsigned long flags; int ctype; int cmode; const long* odims; const long* idims; const long* kdims; const void* kernel; bartorch_linop* result; };

static int linop_conv_worker(void* p)
{
	struct linop_conv_args* a = p;
	a->result = wrap_linop(linop_conv_create(a->N, a->flags, (enum conv_type)a->ctype,
				(enum conv_mode)a->cmode, a->odims, a->idims, a->kdims, a->kernel));
	return 0;
}

bartorch_linop* bartorch_linop_conv(int N, unsigned long flags, int ctype, int cmode,
		const long* odims, const long* idims, const long* kdims, const void* kernel)
{
	struct linop_conv_args a = { N, flags, ctype, cmode, odims, idims, kdims, kernel, NULL };
	return (0 == guarded(linop_conv_worker, &a)) ? a.result : NULL;
}

struct linop_grad_args { int N; const long* dims; int d; unsigned long flags; bartorch_linop* result; };

static int linop_grad_worker(void* p)
{
	struct linop_grad_args* a = p;
	a->result = wrap_linop(linop_grad_create(a->N, a->dims, a->d, a->flags));
	return 0;
}

/* The finite differences along the flagged axes, stacked along d, which must
 * be a dimension the input has only one of. */
bartorch_linop* bartorch_linop_grad(int N, const long* dims, int d, unsigned long flags)
{
	struct linop_grad_args a = { N, dims, d, flags, NULL };
	return (0 == guarded(linop_grad_worker, &a)) ? a.result : NULL;
}

static int linop_zreal_worker(void* p)
{
	struct linop_flagged_args* a = p;
	a->result = wrap_linop(linop_zreal_create(a->N, a->dims));
	return 0;
}

bartorch_linop* bartorch_linop_zreal(int N, const long* dims)
{
	struct linop_flagged_args a = { N, dims, 0UL, NULL, NULL };
	return (0 == guarded(linop_zreal_worker, &a)) ? a.result : NULL;
}

static int linop_sum_worker(void* p)
{
	struct linop_flagged_args* a = p;
	a->result = wrap_linop(linop_sum_create(a->N, a->dims, a->flags));
	return 0;
}

/* The one BART operator with a closed-form pseudo-inverse. */
bartorch_linop* bartorch_linop_sum(int N, const long* dims, unsigned long flags)
{
	struct linop_flagged_args a = { N, dims, flags, NULL, NULL };
	return (0 == guarded(linop_sum_worker, &a)) ? a.result : NULL;
}

static int linop_scaled_sum_worker(void* p)
{
	struct linop_flagged_args* a = p;
	a->result = wrap_linop(linop_scaled_sum_create(a->N, a->dims, a->flags));
	return 0;
}

/* The sum divided by the square root of how many were summed, so that its
 * normal is an orthogonal projection.  That is the operator BART's closed-form
 * pseudo-inverse is written for. */
bartorch_linop* bartorch_linop_scaled_sum(int N, const long* dims, unsigned long flags)
{
	struct linop_flagged_args a = { N, dims, flags, NULL, NULL };
	return (0 == guarded(linop_scaled_sum_worker, &a)) ? a.result : NULL;
}

static int linop_avg_worker(void* p)
{
	struct linop_flagged_args* a = p;
	a->result = wrap_linop(linop_avg_create(a->N, a->dims, a->flags));
	return 0;
}

bartorch_linop* bartorch_linop_avg(int N, const long* dims, unsigned long flags)
{
	struct linop_flagged_args a = { N, dims, flags, NULL, NULL };
	return (0 == guarded(linop_avg_worker, &a)) ? a.result : NULL;
}

static int linop_repmat_worker(void* p)
{
	struct linop_flagged_args* a = p;
	a->result = wrap_linop(linop_repmat_create(a->N, a->dims, a->flags));
	return 0;
}

/* dims is the codomain: the domain is it with the flagged axes set to one. */
bartorch_linop* bartorch_linop_repmat(int N, const long* odims, unsigned long flags)
{
	struct linop_flagged_args a = { N, odims, flags, NULL, NULL };
	return (0 == guarded(linop_repmat_worker, &a)) ? a.result : NULL;
}

struct linop_hankel_args { int N; const long* dims; int dim; int window_dim; int window; bartorch_linop* result; };

static int linop_hankel_worker(void* p)
{
	struct linop_hankel_args* a = p;
	a->result = wrap_linop(linop_hankelization_create(a->N, a->dims, a->dim, a->window_dim, a->window));
	return 0;
}

/* A sliding window along dim, laid out along window_dim -- which the input
 * must have only one of.  BART builds it as a strided view, so nothing is
 * copied to make the windows overlap. */
bartorch_linop* bartorch_linop_hankel(int N, const long* dims, int dim, int window_dim, int window)
{
	struct linop_hankel_args a = { N, dims, dim, window_dim, window, NULL };
	return (0 == guarded(linop_hankel_worker, &a)) ? a.result : NULL;
}

static int linop_flip_worker(void* p)
{
	struct linop_flagged_args* a = p;
	a->result = wrap_linop(linop_flip_create(a->N, a->dims, a->flags));
	return 0;
}

bartorch_linop* bartorch_linop_flip(int N, const long* dims, unsigned long flags)
{
	struct linop_flagged_args a = { N, dims, flags, NULL, NULL };
	return (0 == guarded(linop_flip_worker, &a)) ? a.result : NULL;
}

struct linop_two_shapes_args { int NO; const long* odims; int NI; const long* idims; const long* pos; bartorch_linop* result; };

static int linop_reshape_worker(void* p)
{
	struct linop_two_shapes_args* a = p;
	a->result = wrap_linop(linop_reshape_create(a->NO, a->odims, a->NI, a->idims));
	return 0;
}

bartorch_linop* bartorch_linop_reshape(int NO, const long* odims, int NI, const long* idims)
{
	struct linop_two_shapes_args a = { NO, odims, NI, idims, NULL, NULL };
	return (0 == guarded(linop_reshape_worker, &a)) ? a.result : NULL;
}

static int linop_resize_worker(void* p)
{
	struct linop_two_shapes_args* a = p;
	a->result = wrap_linop(linop_resize_center_create(a->NO, a->odims, a->idims));
	return 0;
}

/* Centred: what BART's `resize -c` does, cropping or zero-filling about the
 * middle of each axis rather than the corner. */
bartorch_linop* bartorch_linop_resize(int N, const long* odims, const long* idims)
{
	struct linop_two_shapes_args a = { N, odims, N, idims, NULL, NULL };
	return (0 == guarded(linop_resize_worker, &a)) ? a.result : NULL;
}

static int linop_extract_worker(void* p)
{
	struct linop_two_shapes_args* a = p;
	a->result = wrap_linop(linop_extract_create(a->NI, a->pos, a->odims, a->idims));
	return 0;
}

bartorch_linop* bartorch_linop_extract(int N, const long* pos, const long* odims, const long* idims)
{
	struct linop_two_shapes_args a = { N, odims, N, idims, pos, NULL };
	return (0 == guarded(linop_extract_worker, &a)) ? a.result : NULL;
}

struct linop_transpose_args { int N; int a; int b; const long* dims; bartorch_linop* result; };

static int linop_transpose_worker(void* p)
{
	struct linop_transpose_args* t = p;
	t->result = wrap_linop(linop_transpose_create(t->N, t->a, t->b, t->dims));
	return 0;
}

bartorch_linop* bartorch_linop_transpose(int N, int a, int b, const long* dims)
{
	struct linop_transpose_args t = { N, a, b, dims, NULL };
	return (0 == guarded(linop_transpose_worker, &t)) ? t.result : NULL;
}

struct linop_permute_args { int N; const int* order; const long* idims; bartorch_linop* result; };

static int linop_permute_worker(void* p)
{
	struct linop_permute_args* a = p;
	a->result = wrap_linop(linop_permute_create(a->N, a->order, a->idims));
	return 0;
}

/* order is BART's: the output's dimension i is the input's order[i]. */
bartorch_linop* bartorch_linop_permute(int N, const int* order, const long* idims)
{
	struct linop_permute_args a = { N, order, idims, NULL };
	return (0 == guarded(linop_permute_worker, &a)) ? a.result : NULL;
}

struct linop_shift_args { int N; const long* dims; int dim; long shift; int pad; bartorch_linop* result; };

static int linop_shift_worker(void* p)
{
	struct linop_shift_args* a = p;
	a->result = wrap_linop(linop_shift_create(a->N, a->dims, a->dim, a->shift, (enum PADDING)a->pad));
	return 0;
}

bartorch_linop* bartorch_linop_shift(int N, const long* dims, int dim, long shift, int pad)
{
	struct linop_shift_args a = { N, dims, dim, shift, pad, NULL };
	return (0 == guarded(linop_shift_worker, &a)) ? a.result : NULL;
}

struct linop_padding_args { int N; const long* dims; int pad; const long* before; const long* after; bartorch_linop* result; };

static int linop_padding_worker(void* p)
{
	struct linop_padding_args* a = p;

	/* BART takes these non-const and does not write through them. */
	long before[DIMS];
	long after[DIMS];
	md_copy_dims(a->N, before, a->before);
	md_copy_dims(a->N, after, a->after);

	a->result = wrap_linop(linop_padding_create(a->N, a->dims, (enum PADDING)a->pad, before, after));
	return 0;
}

bartorch_linop* bartorch_linop_padding(int N, const long* dims, int pad, const long* before, const long* after)
{
	struct linop_padding_args a = { N, dims, pad, before, after, NULL };
	return (0 == guarded(linop_padding_worker, &a)) ? a.result : NULL;
}

int bartorch_linop_has_pseudo_inv(const bartorch_linop* h)
{
	return ((NULL != h) && (NULL != h->op->norm_inv)) ? 1 : 0;
}

struct linop_pinv_args { const bartorch_linop* h; float lambda; void* dst; const void* src; };

static int linop_pinv_worker(void* p)
{
	struct linop_pinv_args* a = p;
	const struct linop_s* op = a->h->op;

	if (NULL == op->norm_inv)
		error("this operator has no closed-form pseudo-inverse\n");

	const struct iovec_s* dom = linop_domain(op);
	complex float* adj = md_alloc_sameplace(dom->N, dom->dims, CFL_SIZE, a->dst);

	linop_adjoint_unchecked(op, adj, a->src);
	linop_norm_inv_unchecked(op, a->lambda, a->dst, adj);

	md_free(adj);
	return 0;
}

/* (A^H A + lambda I)^-1 A^H y, which is what linop_pseudo_inv does, without
 * its assert: the caller has asked bartorch_linop_has_pseudo_inv first. */
int bartorch_linop_pseudo_inv(const bartorch_linop* h, float lambda, void* dst, const void* src)
{
	struct linop_pinv_args args = { h, lambda, dst, src };
	return guarded(linop_pinv_worker, &args);
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

/* --- host-defined nonlinear operator of many arguments ---------------------- */

struct cb_generic_data {

	nlop_data_t super;
	bartorch_generic_apply_fn forward;
	bartorch_pair_apply_fn derivative;
	bartorch_pair_apply_fn adjoint;
	void* ctx;
	bartorch_release_fn release;
};

static DEF_TYPEID(cb_generic_data);

static void cbg_forward(const nlop_data_t* _d, int N, complex float* args[N])
{
	const struct cb_generic_data* d = CAST_DOWN(cb_generic_data, _d);

	if (0 != d->forward(d->ctx, N, (void**)args))
		error("bartorch: forward callback failed\n");
}

static void cbg_derivative(const nlop_data_t* _d, int o, int i, complex float* dst, const complex float* src)
{
	const struct cb_generic_data* d = CAST_DOWN(cb_generic_data, _d);

	if (0 != d->derivative(d->ctx, o, i, dst, src))
		error("bartorch: derivative callback failed\n");
}

static void cbg_adjoint(const nlop_data_t* _d, int o, int i, complex float* dst, const complex float* src)
{
	const struct cb_generic_data* d = CAST_DOWN(cb_generic_data, _d);

	if (0 != d->adjoint(d->ctx, o, i, dst, src))
		error("bartorch: adjoint callback failed\n");
}

static void cbg_del(const nlop_data_t* _d)
{
	const struct cb_generic_data* d = CAST_DOWN(cb_generic_data, _d);

	if (NULL != d->release)
		d->release(d->ctx);

	xfree(d);
}

struct nlop_callback_generic_args {

	int OO; int ON; const long* odims;
	int II; int IN; const long* idims;
	bartorch_generic_apply_fn forward;
	bartorch_pair_apply_fn derivative;
	bartorch_pair_apply_fn adjoint;
	void* ctx; bartorch_release_fn release;
	bartorch_nlop* result;
};

static int nlop_callback_generic_worker(void* p)
{
	struct nlop_callback_generic_args* a = p;

	PTR_ALLOC(struct cb_generic_data, d);
	SET_TYPEID(cb_generic_data, d);
	d->super.clear_der = NULL;
	d->super.data_der = NULL;
	d->forward = a->forward;
	d->derivative = a->derivative;
	d->adjoint = a->adjoint;
	d->ctx = a->ctx;
	d->release = a->release;

	/* The shapes arrive flat, one argument after another; BART wants them
	 * as arrays of arrays, which is the same memory read differently. */
	const long (*od)[a->ON] = (const long (*)[a->ON])a->odims;
	const long (*id)[a->IN] = (const long (*)[a->IN])a->idims;

	/* One derivative and one adjoint for every (input, output) pair.  The
	 * two shims below dispatch on the pair themselves, so every entry is
	 * the same function; what BART reads from the table is *which* pairs
	 * exist at all. */
	nlop_der_fun_t der[a->II][a->OO];
	nlop_der_fun_t adj[a->II][a->OO];

	for (int i = 0; i < a->II; i++) {

		for (int o = 0; o < a->OO; o++) {

			der[i][o] = cbg_derivative;
			adj[i][o] = cbg_adjoint;
		}
	}

	a->result = wrap_nlop(nlop_generic_create(a->OO, a->ON, od, a->II, a->IN, id,
			CAST_UP(PTR_PASS(d)), cbg_forward, der, adj, NULL, NULL, cbg_del));
	return 0;
}

bartorch_nlop* bartorch_nlop_callback_generic(int OO, int ON, const long* odims,
		int II, int IN, const long* idims,
		bartorch_generic_apply_fn forward,
		bartorch_pair_apply_fn derivative,
		bartorch_pair_apply_fn adjoint,
		void* ctx, bartorch_release_fn release)
{
	if ((NULL == odims) || (NULL == idims) || (0 >= OO) || (0 >= II))
		return NULL;

	struct nlop_callback_generic_args args = {

		OO, ON, odims, II, IN, idims,
		forward, derivative, adjoint, ctx, release, NULL,
	};

	return (0 == guarded(nlop_callback_generic_worker, &args)) ? args.result : NULL;
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

/* --- how many arguments, and what shape each one is ------------------------
 *
 * BART's `nlop_s` is many inputs to many outputs; the wrapper started with
 * the one-in one-out case because that is what `bartorch_irgnm` needed.  The
 * model `nlinv` inverts is two inputs -- the image and the coil profiles --
 * so the arity has to be reachable before any of it can be expressed here.
 *
 * Arguments are counted BART's way throughout: outputs first, then inputs,
 * which is the order `nlop_generic_apply_unchecked` reads them in.
 */
int bartorch_nlop_inputs(const bartorch_nlop* h)
{
	return (NULL == h) ? -1 : nlop_get_nr_in_args(h->op);
}

int bartorch_nlop_outputs(const bartorch_nlop* h)
{
	return (NULL == h) ? -1 : nlop_get_nr_out_args(h->op);
}

int bartorch_nlop_input_domain(const bartorch_nlop* h, int i, int N, long* dims)
{
	if ((NULL == h) || (NULL == dims) || (0 > i) || (i >= nlop_get_nr_in_args(h->op)))
		return -1;

	const struct iovec_s* iov = nlop_generic_domain(h->op, i);

	if (iov->N > N)
		return -8;

	copy_dims(iov, N, dims);

	return iov->N;
}

int bartorch_nlop_output_codomain(const bartorch_nlop* h, int o, int N, long* dims)
{
	if ((NULL == h) || (NULL == dims) || (0 > o) || (o >= nlop_get_nr_out_args(h->op)))
		return -1;

	const struct iovec_s* iov = nlop_generic_codomain(h->op, o);

	if (iov->N > N)
		return -8;

	copy_dims(iov, N, dims);

	return iov->N;
}

/* Apply an operator of any arity.
 *
 * `args` is outputs then inputs, each a buffer the caller owns, which is what
 * `nlop_generic_apply_unchecked` takes.  It also fixes the point every
 * derivative is taken at, exactly as the one-argument `bartorch_nlop_apply`
 * does.
 */
struct nlop_generic_args { const bartorch_nlop* h; int nargs; void** args; };

static int nlop_generic_worker(void* p)
{
	struct nlop_generic_args* a = p;
	nlop_generic_apply_unchecked(a->h->op, a->nargs, a->args);
	return 0;
}

int bartorch_nlop_apply_generic(const bartorch_nlop* h, int nargs, void** args)
{
	if ((NULL == h) || (NULL == args))
		return -1;

	if (nargs != nlop_get_nr_in_args(h->op) + nlop_get_nr_out_args(h->op))
		return -1;

	struct nlop_generic_args a = { h, nargs, args };

	return guarded(nlop_generic_worker, &a);
}

/* The derivative of one output by one input, as a linear operator.
 *
 * `nlop_get_derivative` hands back a `linop_s`, so the whole linear surface --
 * its adjoint, its normal, a solve over it -- applies to it unchanged.  That
 * is how `noir/recon2.c` builds the inner problem of its Gauss-Newton steps,
 * and it is how one gets built here.
 *
 * The point is wherever the last application left it.
 */
struct nlop_derivative_args { const bartorch_nlop* h; int o; int i; bartorch_linop* result; };

static int nlop_derivative_linop_worker(void* p)
{
	struct nlop_derivative_args* a = p;
	a->result = bartorch_linop_wrap(linop_clone(nlop_get_derivative(a->h->op, a->o, a->i)));
	return 0;
}

bartorch_linop* bartorch_nlop_derivative_linop(const bartorch_nlop* h, int o, int i)
{
	if ((NULL == h)
	    || (0 > o) || (o >= nlop_get_nr_out_args(h->op))
	    || (0 > i) || (i >= nlop_get_nr_in_args(h->op)))
		return NULL;

	struct nlop_derivative_args a = { h, o, i, NULL };

	return (0 == guarded(nlop_derivative_linop_worker, &a)) ? a.result : NULL;
}

/* --- the algebra ----------------------------------------------------------
 *
 * `chain` was the whole of it, and it only says "all of a into all of b".
 * These are the rest of `nlops/chain.h`: one output into one input, two
 * operators side by side, an output tied back to an input, two inputs made
 * one, and the reorderings that make those usable.
 */
struct nlop_reshape_args { const bartorch_nlop* a; int at; int N; const long* dims; int output; bartorch_nlop* result; };

static int nlop_reshape_worker(void* p)
{
	struct nlop_reshape_args* a = p;
	const struct nlop_s* made = nlop_clone(a->a->op);

	made = a->output
		? nlop_reshape_out_F(made, a->at, a->N, a->dims)
		: nlop_reshape_in_F(made, a->at, a->N, a->dims);

	a->result = wrap_nlop(made);
	return 0;
}

static bartorch_nlop* nlop_reshape(const bartorch_nlop* a, int at, int N, const long* dims, int output)
{
	if ((NULL == a) || (NULL == dims) || (0 >= N))
		return NULL;

	struct nlop_reshape_args args = { a, at, N, dims, output, NULL };

	return (0 == guarded(nlop_reshape_worker, &args)) ? args.result : NULL;
}

bartorch_nlop* bartorch_nlop_reshape_in(const bartorch_nlop* a, int i, int N, const long* dims)
{
	return nlop_reshape(a, i, N, dims, 0);
}

bartorch_nlop* bartorch_nlop_reshape_out(const bartorch_nlop* a, int o, int N, const long* dims)
{
	return nlop_reshape(a, o, N, dims, 1);
}

struct nlop_chain2_args { const bartorch_nlop* a; int o; const bartorch_nlop* b; int i; bartorch_nlop* result; };

static int nlop_chain2_worker(void* p)
{
	struct nlop_chain2_args* a = p;
	a->result = wrap_nlop(nlop_chain2(a->a->op, a->o, a->b->op, a->i));
	return 0;
}

bartorch_nlop* bartorch_nlop_chain2(const bartorch_nlop* a, int o, const bartorch_nlop* b, int i)
{
	if ((NULL == a) || (NULL == b))
		return NULL;

	struct nlop_chain2_args args = { a, o, b, i, NULL };

	return (0 == guarded(nlop_chain2_worker, &args)) ? args.result : NULL;
}

static int nlop_combine_worker(void* p)
{
	struct nlop_pair_args* a = p;
	a->result = wrap_nlop(nlop_combine(a->a->op, a->b->op));
	return 0;
}

bartorch_nlop* bartorch_nlop_combine(const bartorch_nlop* a, const bartorch_nlop* b)
{
	if ((NULL == a) || (NULL == b))
		return NULL;

	struct nlop_pair_args args = { a, b, NULL };

	return (0 == guarded(nlop_combine_worker, &args)) ? args.result : NULL;
}

struct nlop_index2_args { const bartorch_nlop* x; int a; int b; int c; bartorch_nlop* result; };

static int nlop_link_worker(void* p)
{
	struct nlop_index2_args* v = p;
	v->result = wrap_nlop(nlop_link(v->x->op, v->a, v->b));
	return 0;
}

bartorch_nlop* bartorch_nlop_link(const bartorch_nlop* x, int oo, int ii)
{
	if (NULL == x)
		return NULL;

	struct nlop_index2_args v = { x, oo, ii, 0, NULL };

	return (0 == guarded(nlop_link_worker, &v)) ? v.result : NULL;
}

static int nlop_dup_worker(void* p)
{
	struct nlop_index2_args* v = p;
	v->result = wrap_nlop(nlop_dup(v->x->op, v->a, v->b));
	return 0;
}

bartorch_nlop* bartorch_nlop_dup(const bartorch_nlop* x, int a, int b)
{
	if (NULL == x)
		return NULL;

	struct nlop_index2_args v = { x, a, b, 0, NULL };

	return (0 == guarded(nlop_dup_worker, &v)) ? v.result : NULL;
}

static int nlop_stack_inputs_worker(void* p)
{
	struct nlop_index2_args* v = p;
	v->result = wrap_nlop(nlop_stack_inputs(v->x->op, v->a, v->b, v->c));
	return 0;
}

bartorch_nlop* bartorch_nlop_stack_inputs(const bartorch_nlop* x, int a, int b, int dim)
{
	if (NULL == x)
		return NULL;

	struct nlop_index2_args v = { x, a, b, dim, NULL };

	return (0 == guarded(nlop_stack_inputs_worker, &v)) ? v.result : NULL;
}

static int nlop_stack_outputs_worker(void* p)
{
	struct nlop_index2_args* v = p;
	v->result = wrap_nlop(nlop_stack_outputs(v->x->op, v->a, v->b, v->c));
	return 0;
}

bartorch_nlop* bartorch_nlop_stack_outputs(const bartorch_nlop* x, int a, int b, int dim)
{
	if (NULL == x)
		return NULL;

	struct nlop_index2_args v = { x, a, b, dim, NULL };

	return (0 == guarded(nlop_stack_outputs_worker, &v)) ? v.result : NULL;
}

struct nlop_permute_args { const bartorch_nlop* x; int n; const int* perm; int outputs; bartorch_nlop* result; };

static int nlop_permute_worker(void* p)
{
	struct nlop_permute_args* v = p;
	v->result = wrap_nlop(v->outputs
			? nlop_permute_outputs(v->x->op, v->n, v->perm)
			: nlop_permute_inputs(v->x->op, v->n, v->perm));
	return 0;
}

bartorch_nlop* bartorch_nlop_permute(const bartorch_nlop* x, int outputs, int n, const int* perm)
{
	if ((NULL == x) || (NULL == perm))
		return NULL;

	struct nlop_permute_args v = { x, n, perm, outputs, NULL };

	return (0 == guarded(nlop_permute_worker, &v)) ? v.result : NULL;
}

struct nlop_index1_args { const bartorch_nlop* x; int i; bartorch_nlop* result; };

/* Many arguments made one.
 *
 * `nlop_flatten` reshapes every input into one flat vector and every output
 * into another, which is how `noir/recon2.c` hands a two-unknown model to a
 * solver that knows only one vector: the image and the coil coefficients are
 * laid out one after the other, in argument order.
 */
static int nlop_flatten_worker(void* p)
{
	struct nlop_index1_args* v = p;

	const struct nlop_s* op = v->i ? (const struct nlop_s*)nlop_flatten_inputs_F(nlop_clone(v->x->op))
				       : (const struct nlop_s*)nlop_flatten(v->x->op);

	/* BART's flatten declares its vector at rank one.  Everything else in
	 * this wrapper is declared over all sixteen axes, and `lsqr2_create`
	 * checks the rank it was given against the operator's own -- so a
	 * flattened model handed to a solver asserts unless the vector is
	 * restated the way the rest of the library states it. */
	long dims[DIMS];
	md_singleton_dims(DIMS, dims);

	dims[0] = md_calc_size(nlop_generic_domain(op, 0)->N, nlop_generic_domain(op, 0)->dims);
	op = nlop_reshape_in_F(op, 0, DIMS, dims);

	if (!v->i) {

		md_singleton_dims(DIMS, dims);
		dims[0] = md_calc_size(nlop_generic_codomain(op, 0)->N, nlop_generic_codomain(op, 0)->dims);
		op = nlop_reshape_out_F(op, 0, DIMS, dims);
	}

	v->result = wrap_nlop(op);

	return 0;
}

bartorch_nlop* bartorch_nlop_flatten(const bartorch_nlop* x, int inputs_only)
{
	if (NULL == x)
		return NULL;

	struct nlop_index1_args v = { x, inputs_only, NULL };

	return (0 == guarded(nlop_flatten_worker, &v)) ? v.result : NULL;
}


static int nlop_del_out_worker(void* p)
{
	struct nlop_index1_args* v = p;
	v->result = wrap_nlop(nlop_del_out(v->x->op, v->i));
	return 0;
}

bartorch_nlop* bartorch_nlop_del_out(const bartorch_nlop* x, int o)
{
	if (NULL == x)
		return NULL;

	struct nlop_index1_args v = { x, o, NULL };

	return (0 == guarded(nlop_del_out_worker, &v)) ? v.result : NULL;
}

/* --- the basic nonlinear operators ----------------------------------------
 *
 * The elementwise maps and the tensor product, which are what the algebra
 * above is for: `nlinv`'s model is a product of an image with coil profiles,
 * and every model BART fits to a relaxation curve is an exponential of
 * something.  A `long` dimension vector and BART's own flags throughout, as
 * everywhere else in this wrapper.
 */
struct nlop_dims_args {

	int N;
	const long* dims;
	const long* dims2;
	const long* dims3;
	float a;
	float b;
	unsigned long flags;
	bartorch_nlop* result;
};

static int nlop_tenmul_worker(void* p)
{
	struct nlop_dims_args* v = p;
	v->result = wrap_nlop(nlop_tenmul_create(v->N, v->dims, v->dims2, v->dims3));
	return 0;
}

bartorch_nlop* bartorch_nlop_tenmul(int N, const long* odims, const long* idims1, const long* idims2)
{
	if ((NULL == odims) || (NULL == idims1) || (NULL == idims2))
		return NULL;

	struct nlop_dims_args v = { N, odims, idims1, idims2, 0., 0., 0ul, NULL };

	return (0 == guarded(nlop_tenmul_worker, &v)) ? v.result : NULL;
}

static int nlop_zdiv_worker(void* p)
{
	struct nlop_dims_args* v = p;
	v->result = wrap_nlop((0. < v->a)
			? nlop_zdiv_reg_create(v->N, v->dims, v->a)
			: nlop_zdiv_create(v->N, v->dims));
	return 0;
}

bartorch_nlop* bartorch_nlop_zdiv(int N, const long* dims, float eps)
{
	if (NULL == dims)
		return NULL;

	struct nlop_dims_args v = { N, dims, NULL, NULL, eps, 0., 0ul, NULL };

	return (0 == guarded(nlop_zdiv_worker, &v)) ? v.result : NULL;
}

static int nlop_zaxpbz_worker(void* p)
{
	struct nlop_dims_args* v = p;
	v->result = wrap_nlop(nlop_zaxpbz2_create(v->N, v->dims, v->flags, v->a, v->flags, v->b));
	return 0;
}

bartorch_nlop* bartorch_nlop_zaxpbz(int N, const long* dims, float a, float b)
{
	if (NULL == dims)
		return NULL;

	struct nlop_dims_args v = { N, dims, NULL, NULL, a, b, ~0ul, NULL };

	return (0 == guarded(nlop_zaxpbz_worker, &v)) ? v.result : NULL;
}

static int nlop_zexp_worker(void* p)
{
	struct nlop_dims_args* v = p;
	v->result = wrap_nlop(nlop_zexp_create(v->N, v->dims));
	return 0;
}

bartorch_nlop* bartorch_nlop_zexp(int N, const long* dims)
{
	if (NULL == dims)
		return NULL;

	struct nlop_dims_args v = { N, dims, NULL, NULL, 0., 0., 0ul, NULL };

	return (0 == guarded(nlop_zexp_worker, &v)) ? v.result : NULL;
}

static int nlop_zlog_worker(void* p)
{
	struct nlop_dims_args* v = p;
	v->result = wrap_nlop(nlop_zlog_create(v->N, v->dims));
	return 0;
}

bartorch_nlop* bartorch_nlop_zlog(int N, const long* dims)
{
	if (NULL == dims)
		return NULL;

	struct nlop_dims_args v = { N, dims, NULL, NULL, 0., 0., 0ul, NULL };

	return (0 == guarded(nlop_zlog_worker, &v)) ? v.result : NULL;
}

static int nlop_zinv_worker(void* p)
{
	struct nlop_dims_args* v = p;
	v->result = wrap_nlop((0. < v->a)
			? nlop_zinv_reg_create(v->N, v->dims, v->a)
			: nlop_zinv_create(v->N, v->dims));
	return 0;
}

bartorch_nlop* bartorch_nlop_zinv(int N, const long* dims, float eps)
{
	if (NULL == dims)
		return NULL;

	struct nlop_dims_args v = { N, dims, NULL, NULL, eps, 0., 0ul, NULL };

	return (0 == guarded(nlop_zinv_worker, &v)) ? v.result : NULL;
}

static int nlop_zsqrt_worker(void* p)
{
	struct nlop_dims_args* v = p;
	v->result = wrap_nlop(nlop_zsqrt_create(v->N, v->dims));
	return 0;
}

bartorch_nlop* bartorch_nlop_zsqrt(int N, const long* dims)
{
	if (NULL == dims)
		return NULL;

	struct nlop_dims_args v = { N, dims, NULL, NULL, 0., 0., 0ul, NULL };

	return (0 == guarded(nlop_zsqrt_worker, &v)) ? v.result : NULL;
}

static int nlop_zspow_worker(void* p)
{
	struct nlop_dims_args* v = p;
	v->result = wrap_nlop(nlop_zspow_create(v->N, v->dims, v->a + v->b * 1.i));
	return 0;
}

bartorch_nlop* bartorch_nlop_zspow(int N, const long* dims, float re, float im)
{
	if (NULL == dims)
		return NULL;

	struct nlop_dims_args v = { N, dims, NULL, NULL, re, im, 0ul, NULL };

	return (0 == guarded(nlop_zspow_worker, &v)) ? v.result : NULL;
}

static int nlop_zsadd_worker(void* p)
{
	struct nlop_dims_args* v = p;
	v->result = wrap_nlop(nlop_zsadd_create(v->N, v->dims, v->a + v->b * 1.i));
	return 0;
}

bartorch_nlop* bartorch_nlop_zsadd(int N, const long* dims, float re, float im)
{
	if (NULL == dims)
		return NULL;

	struct nlop_dims_args v = { N, dims, NULL, NULL, re, im, 0ul, NULL };

	return (0 == guarded(nlop_zsadd_worker, &v)) ? v.result : NULL;
}

static int nlop_zabs_worker(void* p)
{
	struct nlop_dims_args* v = p;
	v->result = wrap_nlop(nlop_zabs_create(v->N, v->dims));
	return 0;
}

bartorch_nlop* bartorch_nlop_zabs(int N, const long* dims)
{
	if (NULL == dims)
		return NULL;

	struct nlop_dims_args v = { N, dims, NULL, NULL, 0., 0., 0ul, NULL };

	return (0 == guarded(nlop_zabs_worker, &v)) ? v.result : NULL;
}

static int nlop_smo_abs_worker(void* p)
{
	struct nlop_dims_args* v = p;
	v->result = wrap_nlop(nlop_smo_abs_create(v->N, v->dims, v->a));
	return 0;
}

bartorch_nlop* bartorch_nlop_smo_abs(int N, const long* dims, float eps)
{
	if (NULL == dims)
		return NULL;

	struct nlop_dims_args v = { N, dims, NULL, NULL, eps, 0., 0ul, NULL };

	return (0 == guarded(nlop_smo_abs_worker, &v)) ? v.result : NULL;
}

static int nlop_zrss_worker(void* p)
{
	struct nlop_dims_args* v = p;
	v->result = wrap_nlop((0. < v->a)
			? nlop_zrss_reg_create(v->N, v->dims, v->flags, v->a)
			: nlop_zrss_create(v->N, v->dims, v->flags));
	return 0;
}

bartorch_nlop* bartorch_nlop_zrss(int N, const long* dims, unsigned long flags, float eps)
{
	if (NULL == dims)
		return NULL;

	struct nlop_dims_args v = { N, dims, NULL, NULL, eps, 0., flags, NULL };

	return (0 == guarded(nlop_zrss_worker, &v)) ? v.result : NULL;
}

static int nlop_zss_worker(void* p)
{
	struct nlop_dims_args* v = p;
	v->result = wrap_nlop(nlop_zss_create(v->N, v->dims, v->flags));
	return 0;
}

bartorch_nlop* bartorch_nlop_zss(int N, const long* dims, unsigned long flags)
{
	if (NULL == dims)
		return NULL;

	struct nlop_dims_args v = { N, dims, NULL, NULL, 0., 0., flags, NULL };

	return (0 == guarded(nlop_zss_worker, &v)) ? v.result : NULL;
}

/* An operator of no inputs at all: it returns the tensor it was built with.
 *
 * `nlop_const_create` copies, so the caller's buffer is free after the call.
 * Combined and linked, this is how an input is pinned to a value.
 */
struct nlop_const_args { int N; const long* dims; const void* val; bartorch_nlop* result; };

static int nlop_const_worker(void* p)
{
	struct nlop_const_args* v = p;
	v->result = wrap_nlop(nlop_const_create(v->N, v->dims, true, v->val));
	return 0;
}

bartorch_nlop* bartorch_nlop_const(int N, const long* dims, const void* val)
{
	if ((NULL == dims) || (NULL == val))
		return NULL;

	struct nlop_const_args v = { N, dims, val, NULL };

	return (0 == guarded(nlop_const_worker, &v)) ? v.result : NULL;
}

struct nlop_set_const_args { const bartorch_nlop* a; int i; int N; const long* dims; const void* val; bartorch_nlop* result; };

static int nlop_set_input_const_worker(void* p)
{
	struct nlop_set_const_args* v = p;
	v->result = wrap_nlop(nlop_set_input_const(v->a->op, v->i, v->N, v->dims, true, v->val));
	return 0;
}

bartorch_nlop* bartorch_nlop_set_input_const(const bartorch_nlop* a, int i, int N, const long* dims, const void* val)
{
	if ((NULL == a) || (NULL == dims) || (NULL == val)
	    || (0 > i) || (i >= nlop_get_nr_in_args(a->op)))
		return NULL;

	struct nlop_set_const_args v = { a, i, N, dims, val, NULL };

	return (0 == guarded(nlop_set_input_const_worker, &v)) ? v.result : NULL;
}


void bartorch_nlop_free(bartorch_nlop* h)
{
	if (NULL == h)
		return;

	nlop_free(h->op);
	xfree(h);
}


/* --- the nonlinear SENSE model ---------------------------------------------
 *
 * `nlinv` does not fit an image against known sensitivities: it fits both at
 * once, and the map it inverts is
 *
 *	kspace = A[ (mask * image) * ifftuc(weights * ksens) ]
 *
 * which is exactly the encoding the linear operators here already build, with
 * the coils turned from a fixed tensor into a second unknown.  BART builds it
 * in `noir/model2.c` out of a `tenmul` and three linear operators, and this
 * hands that same model over rather than rebuilding it: `nlinv` and a fit
 * driven from here run the same arithmetic.
 *
 * The coils are unknown as k-space coefficients, not as maps.  `lop_coil`
 * carries the Sobolev weighting `(1 + a |k|^2)^(-b/2)` that keeps them smooth
 * and the transform to image space; what comes out of a fit is coefficients,
 * and `lop_coil` is what turns them into sensitivities.
 *
 * Off the grid the model is asymmetric: it returns gridded coil images rather
 * than k-space samples, and the data has to be gridded to match with the
 * adjoint of `lop_asym`.  On the grid `lop_asym` is the identity and the
 * model returns k-space.  `bartorch_noir_data` hands that operator out so a
 * caller can prepare the data BART's way in either case.
 */
struct noir_create_args {

	int N;
	const long* ksp_dims;
	const long* cim_dims;
	const long* img_dims;
	const long* kco_dims;
	const long* col_dims;
	const long* pat_dims;
	const void* pattern;
	const long* trj_dims;
	const void* traj;
	const long* wgh_dims;
	const void* weights;
	const long* bas_dims;
	const void* basis;
	const long* msk_dims;
	const void* mask;
	int noncart;
	int optimized;
	int toeplitz;
	struct noir2_model_conf_s conf;
	bartorch_noir* result;
};

static int noir_create_worker(void* p)
{
	struct noir_create_args* a = p;

	struct nufft_conf_s nufft_conf = nufft_conf_defaults;
	nufft_conf.toeplitz = (0 != a->toeplitz);

	a->conf.nufft_conf = &nufft_conf;
	a->conf.noncart = (0 != a->noncart);

	bartorch_noir* h = xmalloc(sizeof(*h));

	if (a->noncart) {

		h->model = (a->optimized ? noir2_noncart_optimized_create : noir2_noncart_create)(
				a->N, a->trj_dims, a->traj, a->wgh_dims, a->weights,
				a->bas_dims, a->basis, a->msk_dims, a->mask,
				a->ksp_dims, a->cim_dims, a->img_dims, a->kco_dims, a->col_dims,
				&a->conf);

	} else {

		h->model = noir2_cart_create(a->N, a->pat_dims, a->pattern,
				a->bas_dims, a->basis, a->msk_dims, a->mask,
				a->ksp_dims, a->cim_dims, a->img_dims, a->kco_dims, a->col_dims,
				&a->conf);
	}

	/* The conf is copied into the model, and the nufft conf it points at is
	 * on this stack; nufft_create2 has already read it. */
	h->model.model_conf.nufft_conf = NULL;

	a->result = h;

	return 0;
}

bartorch_noir* bartorch_noir_create(int N,
		const long* ksp_dims, const long* cim_dims, const long* img_dims,
		const long* kco_dims, const long* col_dims,
		const long* pat_dims, const void* pattern,
		const long* trj_dims, const void* traj,
		const long* wgh_dims, const void* weights,
		const long* bas_dims, const void* basis,
		const long* msk_dims, const void* mask,
		int noncart, int optimized, int toeplitz,
		unsigned long fft_flags, unsigned long wght_flags,
		int rvc, int sos, float a, float b, float c,
		float oversampling_coils, int ret_os_coils)
{
	if ((NULL == ksp_dims) || (NULL == cim_dims) || (NULL == img_dims)
	    || (NULL == kco_dims) || (NULL == col_dims))
		return NULL;

	if (noncart ? (NULL == trj_dims) : (NULL == pat_dims))
		return NULL;

	struct noir_create_args args = {

		.N = N,
		.ksp_dims = ksp_dims, .cim_dims = cim_dims, .img_dims = img_dims,
		.kco_dims = kco_dims, .col_dims = col_dims,
		.pat_dims = pat_dims, .pattern = pattern,
		.trj_dims = trj_dims, .traj = traj,
		.wgh_dims = wgh_dims, .weights = weights,
		.bas_dims = bas_dims, .basis = basis,
		.msk_dims = msk_dims, .mask = mask,
		.noncart = noncart, .optimized = optimized, .toeplitz = toeplitz,
		.conf = noir2_model_conf_defaults,
		.result = NULL,
	};

	args.conf.fft_flags = fft_flags;
	args.conf.wght_flags = wght_flags;
	args.conf.rvc = (0 != rvc);
	args.conf.sos = (0 != sos);
	args.conf.a = a;
	args.conf.b = b;
	args.conf.c = c;
	args.conf.oversampling_coils = oversampling_coils;
	args.conf.ret_os_coils = (0 != ret_os_coils);

	return (0 == guarded(noir_create_worker, &args)) ? args.result : NULL;
}

struct noir_part_args { const bartorch_noir* h; int which; bartorch_nlop* nlop; bartorch_linop* linop; };

static int noir_model_worker(void* p)
{
	struct noir_part_args* v = p;
	v->nlop = wrap_nlop(nlop_clone(v->h->model.model));
	return 0;
}

bartorch_nlop* bartorch_noir_model(const bartorch_noir* h)
{
	if (NULL == h)
		return NULL;

	struct noir_part_args v = { h, 0, NULL, NULL };

	return (0 == guarded(noir_model_worker, &v)) ? v.nlop : NULL;
}

static int noir_linop_worker(void* p)
{
	struct noir_part_args* v = p;

	const struct linop_s* op = NULL;

	switch (v->which) {
	case 0: op = v->h->model.lop_coil; break;
	case 1: op = v->h->model.lop_im; break;
	case 2: op = v->h->model.lop_asym; break;
	default: op = v->h->model.lop_fft; break;
	}

	v->linop = wrap_linop(linop_clone(op));

	return 0;
}

static bartorch_linop* noir_linop(const bartorch_noir* h, int which)
{
	if (NULL == h)
		return NULL;

	struct noir_part_args v = { h, which, NULL, NULL };

	return (0 == guarded(noir_linop_worker, &v)) ? v.linop : NULL;
}

bartorch_linop* bartorch_noir_coils(const bartorch_noir* h)
{
	return noir_linop(h, 0);
}

bartorch_linop* bartorch_noir_image(const bartorch_noir* h)
{
	return noir_linop(h, 1);
}

bartorch_linop* bartorch_noir_data(const bartorch_noir* h)
{
	return noir_linop(h, 2);
}

bartorch_linop* bartorch_noir_transform(const bartorch_noir* h)
{
	return noir_linop(h, 3);
}

int bartorch_noir_dims(const bartorch_noir* h, int which, int N, long* dims)
{
	if ((NULL == h) || (NULL == dims) || (N < h->model.N))
		return -1;

	const long* src = NULL;

	switch (which) {
	case 0: src = h->model.ksp_dims; break;
	case 1: src = h->model.cim_dims; break;
	case 2: src = h->model.img_dims; break;
	case 3: src = h->model.col_dims; break;
	case 4: src = h->model.col_ten_dims; break;
	case 5: src = h->model.pat_dims; break;
	case 6: src = h->model.trj_dims; break;
	default: return -1;
	}

	/* The shape of the coil coefficients the model takes is not kept on the
	 * struct; it is the model's second input, which the arity queries
	 * report. */

	if (NULL == src)
		return -1;

	for (int i = 0; i < N; i++)
		dims[i] = (i < h->model.N) ? src[i] : 1;

	return h->model.N;
}

void bartorch_noir_free(bartorch_noir* h)
{
	if (NULL == h)
		return;

	noir2_free(&h->model);
	xfree(h);
}



/* --- the model as a network ------------------------------------------------ */

struct noir_net_create_args {

	int N;
	const long* ksp_dims;
	const long* cim_dims;
	const long* img_dims;
	const long* col_dims;
	const long* trj_dims;
	const long* wgh_dims;
	const long* bas_dims;
	const void* basis;
	const long* msk_dims;
	const void* mask;
	unsigned long batch_flag;
	int batch;
	struct noir2_model_conf_s conf;
	int toeplitz;
	bartorch_noir_net* result;
};

/* `noir2_net_config_create` keeps the pointer it is given rather than the
 * array, so anything the caller passed has to be taken a copy of here. */
static complex float* noir_net_keep(int N, const long* dims, const void* src)
{
	if ((NULL == dims) || (NULL == src))
		return NULL;

	long size = md_calc_size(N, dims);
	complex float* out = xmalloc((size_t)size * CFL_SIZE);
	md_copy(1, MD_DIMS(size), out, src, CFL_SIZE);
	return out;
}

static int noir_net_create_worker(void* p)
{
	struct noir_net_create_args* a = p;

	bartorch_noir_net* h = xmalloc(sizeof(*h));

	h->nufft_conf = nufft_conf_defaults;
	h->nufft_conf.toeplitz = (0 != a->toeplitz);
	h->basis = noir_net_keep(a->N, a->bas_dims, a->basis);
	h->mask = noir_net_keep(a->N, a->msk_dims, a->mask);

	a->conf.nufft_conf = &h->nufft_conf;
	a->conf.noncart = (NULL != a->trj_dims) && (1 != md_calc_size(a->N, a->trj_dims));
	h->noncart = a->conf.noncart;

	h->config = noir2_net_config_create(a->N,
			a->trj_dims, a->wgh_dims,
			a->bas_dims, h->basis,
			a->msk_dims, h->mask,
			a->ksp_dims, a->cim_dims, a->img_dims, a->col_dims,
			a->batch_flag, &a->conf);

	h->model = noir2_net_create(h->config, a->batch);

	a->result = h;
	return 0;
}

bartorch_noir_net* bartorch_noir_net_create(int N,
		const long* ksp_dims, const long* cim_dims,
		const long* img_dims, const long* col_dims,
		const long* trj_dims, const long* wgh_dims,
		const long* bas_dims, const void* basis,
		const long* msk_dims, const void* mask,
		unsigned long batch_flag, int batch,
		unsigned long fft_flags, unsigned long wght_flags,
		int rvc, int sos, float a, float b, float c, int toeplitz)
{
	if ((NULL == ksp_dims) || (NULL == cim_dims) || (NULL == img_dims) || (NULL == col_dims))
		return NULL;

	if (1 > batch)
		return NULL;

	struct noir_net_create_args args = {

		.N = N,
		.ksp_dims = ksp_dims, .cim_dims = cim_dims,
		.img_dims = img_dims, .col_dims = col_dims,
		.trj_dims = trj_dims, .wgh_dims = wgh_dims,
		.bas_dims = bas_dims, .basis = basis,
		.msk_dims = msk_dims, .mask = mask,
		.batch_flag = batch_flag, .batch = batch,
		.conf = noir2_model_conf_defaults,
		.toeplitz = toeplitz,
		.result = NULL,
	};

	args.conf.fft_flags = fft_flags;
	args.conf.wght_flags = wght_flags;
	args.conf.rvc = (0 != rvc);
	args.conf.sos = (0 != sos);
	args.conf.a = a;
	args.conf.b = b;
	args.conf.c = c;

	return (0 == guarded(noir_net_create_worker, &args)) ? args.result : NULL;
}

struct noir_net_part_args {

	const bartorch_noir_net* h;
	int which;
	int cgiter;
	float cgtol;
	float l2lambda;
	int iterations;
	float redu;
	float alpha_min;
	bartorch_nlop* nlop;
};

static int noir_net_part_worker(void* p)
{
	struct noir_net_part_args* v = p;
	struct noir2_net_s* model = v->h->model;

	struct iter_conjgrad_conf cgconf = iter_conjgrad_defaults;
	cgconf.maxiter = v->cgiter;
	cgconf.tol = v->cgtol;
	cgconf.l2lambda = v->l2lambda;

	const struct nlop_s* made = NULL;

	switch (v->which) {

	case 0: made = noir_gauss_newton_step_create(model, &cgconf); break;
	case 1: made = noir_gauss_newton_iter_create_create(model, &cgconf, v->iterations, v->redu, v->alpha_min); break;
	case 2: made = v->h->noncart ? noir_adjoint_nufft_create(model) : noir_adjoint_fft_create(model); break;
	case 3: made = noir_decomp_create(model); break;
	case 4: made = noir_split_create(model); break;
	case 5: made = noir_join_create(model); break;
	default: return -1;
	}

	v->nlop = wrap_nlop(made);
	return 0;
}

static bartorch_nlop* noir_net_part(const bartorch_noir_net* h, int which,
		int cgiter, float cgtol, float l2lambda, int iterations, float redu, float alpha_min)
{
	if (NULL == h)
		return NULL;

	struct noir_net_part_args v = {

		.h = h, .which = which,
		.cgiter = cgiter, .cgtol = cgtol, .l2lambda = l2lambda,
		.iterations = iterations, .redu = redu, .alpha_min = alpha_min,
		.nlop = NULL,
	};

	return (0 == guarded(noir_net_part_worker, &v)) ? v.nlop : NULL;
}

bartorch_nlop* bartorch_noir_net_step(const bartorch_noir_net* h, int cgiter, float cgtol, float l2lambda)
{
	return noir_net_part(h, 0, cgiter, cgtol, l2lambda, 0, 0., 0.);
}

bartorch_nlop* bartorch_noir_net_iterations(const bartorch_noir_net* h,
		int cgiter, float cgtol, float l2lambda, int iterations, float redu, float alpha_min)
{
	return noir_net_part(h, 1, cgiter, cgtol, l2lambda, iterations, redu, alpha_min);
}

bartorch_nlop* bartorch_noir_net_adjoint(const bartorch_noir_net* h)
{
	return noir_net_part(h, 2, 0, 0., 0., 0, 0., 0.);
}

bartorch_nlop* bartorch_noir_net_decompose(const bartorch_noir_net* h)
{
	return noir_net_part(h, 3, 0, 0., 0., 0, 0., 0.);
}

bartorch_nlop* bartorch_noir_net_split(const bartorch_noir_net* h)
{
	return noir_net_part(h, 4, 0, 0., 0., 0, 0., 0.);
}

bartorch_nlop* bartorch_noir_net_join(const bartorch_noir_net* h)
{
	return noir_net_part(h, 5, 0, 0., 0., 0, 0., 0.);
}

void bartorch_noir_net_free(bartorch_noir_net* h)
{
	if (NULL == h)
		return;

	noir2_net_free(h->model);
	noir2_net_config_free(h->config);

	if (NULL != h->basis)
		xfree(h->basis);
	if (NULL != h->mask)
		xfree(h->mask);

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

/* The second form, the one that takes a generic regularized least-squares
 * solver for its inner problem.
 *
 * `irgnm2` costs an extra application of the derivative and buys the ability
 * to solve the linearized problem with anything at all: `noir/recon2.c` and
 * `moba/iter_l1.c` hand it FISTA, ADMM or Chambolle-Pock built over
 * `nlop_get_derivative`, which is how a regularized `nlinv` or `moba` works.
 * A NULL solver is BART's own conjugate gradients, and that is what this
 * exposes: the outer loop written out in Python is held against it.
 */
struct irgnm2_args {

	const bartorch_nlop* F; int iter; float alpha; float alpha_min; float alpha_min0; float redu;
	int cgiter; float cgtol;
	void* x; const void* y; const void* xref;
};

static int irgnm2_worker(void* p)
{
	struct irgnm2_args* a = p;

	struct iter3_irgnm_conf conf = iter3_irgnm_defaults;
	conf.iter = a->iter;
	conf.alpha = a->alpha;
	conf.alpha_min = a->alpha_min;
	conf.alpha_min0 = a->alpha_min0;
	conf.redu = a->redu;
	conf.cgiter = a->cgiter;
	conf.cgtol = a->cgtol;

	const struct iovec_s* dom = nlop_domain(a->F->op);
	const struct iovec_s* cod = nlop_codomain(a->F->op);

	long N = 2 * md_calc_size(dom->N, dom->dims);
	long M = 2 * md_calc_size(cod->N, cod->dims);

	iter4_irgnm2(CAST_UP(&conf), a->F->op, N, a->x, a->xref, M, a->y, NULL,
			(struct iter_op_s){ NULL, NULL });
	return 0;
}

int bartorch_irgnm2(const bartorch_nlop* F, int iter, float alpha, float alpha_min, float alpha_min0,
		float redu, int cgiter, float cgtol, void* x, const void* y, const void* xref)
{
	struct irgnm2_args a = { F, iter, alpha, alpha_min, alpha_min0, redu, cgiter, cgtol, x, y, xref };
	return guarded(irgnm2_worker, &a);
}
