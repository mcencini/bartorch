/*
 * The solve BART's own tools run, assembled from here.
 *
 * `pics` turns its arguments into three things and hands them to `lsqr2`: the
 * proximal operators its `-R` strings name, the algorithm its solver flag
 * chooses, and the encoding.  This does the same, in the same order, with
 * BART's own functions -- `opt_reg_configure`, `italgo_config`, `lsqr2` --
 * so that an encoding built in Python and solved through here is the same
 * computation as the tool, and not a second implementation of it.
 *
 * Nothing here is an algorithm.  Every iteration is BART's.
 */

#include <assert.h>
#include <complex.h>
#include <stdbool.h>
#include <string.h>

#include "misc/misc.h"
#include "misc/mri.h"
#include "misc/debug.h"

#include "num/multind.h"
#include "num/flpmath.h"
#include "num/ops.h"
#include "num/ops_p.h"
#include "num/iovec.h"

#include "linops/linop.h"
#include "linops/someops.h"

#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/lsqr.h"
#include "iter/misc.h"
#include "iter/monitor.h"
#include "iter/prox.h"
#include "iter/thresh.h"

#include "wavelet/wavthresh.h"

#include "sense/optcom.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"

#include "include/bartorch.h"

/* The operator handles the ABI hands out; the host sees only the pointer. */
struct bartorch_linop_s;
extern const struct linop_s* bartorch_linop_unwrap(const struct bartorch_linop_s* h);
extern bartorch_linop* bartorch_linop_wrap(const struct linop_s* op);

/* BART's own letters for its regularization terms, from the run of
 * comparisons in `grecon/optreg.c`.  The host names a term rather than
 * spelling one, and this is the only place the two vocabularies meet. */
static int xform_by_name(const char* name, int* xform)
{
	struct { const char* name; int xform; } table[] = {
		{ "W",  L1WAV },   { "H",  NIHTWAV }, { "N",  NIHTIM },
		{ "L",  LLR },     { "T",  TV },      { "G",  TGV },
		{ "C",  ICTV },    { "V",  ICTGV },   { "P",  LAPLACE },
		{ "R1", IMAGL1 },  { "R2", IMAGL2 },  { "I",  L1IMG },
		{ "Q",  L2IMG },   { "S",  POS },     { "F",  FTL1 },
	};

	for (unsigned int i = 0; i < ARRAY_SIZE(table); i++) {

		if (0 == strcmp(name, table[i].name)) {

			*xform = table[i].xform;
			return 0;
		}
	}

	return -1;
}


static enum algo_t algo_by_name(const char* name)
{
	if (NULL == name)
		return ALGO_DEFAULT;
	if (0 == strcmp(name, "cg"))		return ALGO_CG;
	if (0 == strcmp(name, "ist"))		return ALGO_IST;
	if (0 == strcmp(name, "fista"))		return ALGO_FISTA;
	if (0 == strcmp(name, "admm"))		return ALGO_ADMM;
	if (0 == strcmp(name, "pridu"))		return ALGO_PRIDU;
	if (0 == strcmp(name, "niht"))		return ALGO_NIHT;
	if (0 == strcmp(name, "eulermaruyama"))	return ALGO_EULERMARUYAMA;
	return (enum algo_t)-1;
}

/* One regularization term, built once and kept.  What BART makes of a term is
 * a proximal operator and, for most of them, a transform to apply it through;
 * the two belong together and are handed to the solver as a pair. */
struct bartorch_prox_s {

	const struct operator_p_s* op;
	const struct linop_s* trafo;
	int xform;
};

int bartorch_prox_create(const char* kind, long xflags, long jflags, float lambda, int k,
		int llr_blk, const char* wavelet, int shift_mode,
		const long* img_dims, bartorch_prox** out)
{
	int xform;

	if (0 != xform_by_name(kind, &xform))
		return -4;

	struct opt_reg_s ropts;
	(void)opt_reg_init(&ropts);

	ropts.regs[0].xform = xform;
	ropts.regs[0].xflags = (unsigned long)xflags;
	ropts.regs[0].jflags = (unsigned long)jflags;
	ropts.regs[0].lambda = lambda;
	ropts.regs[0].k = k;
	ropts.regs[0].graph_file = NULL;
	ropts.regs[0].asl = false;
	ropts.r = 1;

	const struct operator_p_s* prox_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };
	const long (*sdims[NUM_REGS])[DIMS + 1] = { NULL };

	long dims[DIMS];
	md_copy_dims(DIMS, dims, img_dims);

	opt_reg_configure(DIMS, dims, &ropts, prox_ops, trafos, sdims,
			llr_blk, shift_mode, (NULL != wavelet) ? wavelet : "dau2", false, ITER_DIM);

	/* A term that extends the optimisation variable cannot be built on its
	 * own: what it adds is counted across the whole set, and the solve
	 * asserts that the total is what was reserved. */
	if (0 < ropts.svars) {

		opt_reg_free(&ropts, prox_ops, trafos);
		return -6;
	}

	PTR_ALLOC(struct bartorch_prox_s, p);
	p->op = prox_ops[0];
	p->trafo = trafos[0];
	p->xform = xform;
	*out = PTR_PASS(p);

	return 0;
}

/* The shape a term's proximal operator works on.
 *
 * Usually the image's.  A term with a transform in front of it -- total
 * variation's gradient, a wavelet, the Fourier transform an `F` term takes --
 * has its proximal operator on the far side of that transform, and then the
 * shape is the transform's codomain rather than the image.
 *
 * Returns the rank, or a negative code.  `dims` holds `N` entries and is
 * filled with ones past the rank.
 */
int bartorch_prox_domain(const bartorch_prox* h, int N, long* dims)
{
	if ((NULL == h) || (NULL == h->op) || (NULL == dims) || (1 > N))
		return -1;

	auto dom = operator_p_domain(h->op);

	if (dom->N > N)
		return -8;

	md_singleton_dims(N, dims);
	md_copy_dims(dom->N, dims, dom->dims);

	return dom->N;
}

/* prox_{gamma f}(src) into dst, over that shape.
 *
 * What the solvers call between their gradient steps, reached on its own so
 * that an iteration written outside the library can call the same operator
 * the library would have.  The shapes are the caller's to check, against
 * `bartorch_prox_domain`.
 */
int bartorch_prox_apply(const bartorch_prox* h, float gamma, void* dst, const void* src)
{
	if ((NULL == h) || (NULL == h->op) || (NULL == dst) || (NULL == src))
		return -1;

	operator_p_apply_unchecked(h->op, gamma, dst, src);

	return 0;
}

/* The transform a term carries, applied without making an operator of it.
 *
 * A gradient puts its components on an axis of their own past BART's sixteen,
 * so total variation's transform cannot be handed over as a `bartorch_linop`
 * at all -- and it is exactly the term an alternating-direction solver is
 * for.  This applies it in place instead, over the shapes
 * `bartorch_prox_domain` reports, which a caller can hold as a tensor of
 * whatever rank it likes.
 *
 * `mode` is 0 for the forward, 1 for the adjoint, 2 for the normal.
 */
int bartorch_prox_transform_apply(const bartorch_prox* h, int mode, void* dst, const void* src)
{
	if ((NULL == h) || (NULL == h->trafo) || (NULL == dst) || (NULL == src))
		return -1;

	switch (mode) {

	case 0: linop_forward_unchecked(h->trafo, dst, src); break;
	case 1: linop_adjoint_unchecked(h->trafo, dst, src); break;
	case 2: linop_normal_unchecked(h->trafo, dst, src); break;
	default: return -1;
	}

	return 0;
}

/* Put a term's own random generator back where a fresh term would have it.
 *
 * A wavelet threshold spins its transform by a random shift drawn from a
 * generator of its own, seeded at one when BART makes the operator.  The tool
 * builds a fresh operator per run; a term here is kept across solves, so
 * every solve rewinds it and a reused term answers as the tool does.
 * `bartorch_solve` does this itself; a loop written outside the library has
 * to ask.
 *
 * A term with no such generator is left alone, which is most of them.
 *
 * Returns 0, or a negative code.
 */
int bartorch_prox_rewind(const bartorch_prox* h)
{
	if ((NULL == h) || (NULL == h->op))
		return -1;

	if (L1WAV == h->xform)
		wavthresh_rand_state_set(h->op, 1);

	return 0;
}

/* Whether that transform is the identity.
 *
 * `iter2_chambolle_pock` asks `linop_is_identity` of the first term and, when
 * the answer is yes, makes it the primal proximal step instead of a dual --
 * which changes the iteration, not just its bookkeeping.  A loop written
 * outside the library has to split the terms the same way, and this is the
 * same question asked of the same operator.
 *
 * Returns 1, 0, or a negative code.
 */
int bartorch_prox_transform_is_identity(const bartorch_prox* h)
{
	if (NULL == h)
		return -1;

	if (NULL == h->trafo)
		return 1;

	return linop_is_identity(h->trafo) ? 1 : 0;
}

/* The transform a term applies before its proximal operator.
 *
 * `opt_reg_configure` gives every term one, and for most of them it is the
 * identity: a wavelet term carries its transform inside its proximal operator
 * rather than in front of it, which is why that one works on the image's own
 * shape.  Total variation is the other arrangement -- the gradient in front,
 * the threshold on its components -- and the Laplace term is a third, a real
 * convolution in front of a proximal operator that happens to be shaped like
 * the image.  So the shapes do not say which arrangement a term is; this does.
 *
 * The handle is the caller's to free.
 */
bartorch_linop* bartorch_prox_transform(const bartorch_prox* h)
{
	if ((NULL == h) || (NULL == h->trafo))
		return NULL;

	return bartorch_linop_wrap(linop_clone(h->trafo));
}

void bartorch_prox_free(bartorch_prox* h)
{
	if (NULL == h)
		return;

	if (NULL != h->op)
		operator_p_free(h->op);

	if (NULL != h->trafo)
		linop_free(h->trafo);

	xfree(h);
}

/* A monitor that does nothing but count.
 *
 * Every one of BART's iterations calls `iter_monitor` once at the top of each
 * step, so counting the calls counts the steps taken -- which for conjugate
 * gradients is what an alternating-direction solver budgets by: `admm` breaks
 * on `nr_invokes > maxiter`, and `nr_invokes` is the conjugate-gradient
 * iterations run across the whole solve.  There is no other way to see that
 * number from outside the library.
 */
struct counting_monitor {

	struct iter_monitor_s super;
	long count;
};

static void counting_monitor_fun(struct iter_monitor_s* monitor, const struct vec_iter_s* /*ops*/, const float* /*x*/)
{
	((struct counting_monitor*)monitor)->count++;
}

/* `lsqr`'s normal operator, rebuilt so its largest eigenvalue can be asked
 * for on its own.
 *
 * `normaleq_l2_apply` is `A^H A x + lambda x`, and this is that arithmetic and
 * not an equivalent: the estimate divides the step every iteration is taken
 * with, so a difference in its last bits is a difference in the answer.
 */
struct maxeigen_data {

	operator_data_t super;

	float lambda;
	long size;

	const struct linop_s* model_op;
};

static DEF_TYPEID(maxeigen_data);

static void maxeigen_apply(const operator_data_t* _data, int N, void* args[static N])
{
	const auto data = CAST_DOWN(maxeigen_data, _data);

	assert(2 == N);
	assert(args[0] != args[1]);

	linop_normal_unchecked(data->model_op, args[0], args[1]);
	md_axpy(1, MD_DIMS(data->size), args[0], data->lambda, args[1]);
}

static void maxeigen_del(const operator_data_t* _data)
{
	const auto data = CAST_DOWN(maxeigen_data, _data);

	linop_free(data->model_op);
	xfree(data);
}

/* The largest eigenvalue of the operator an iteration divides its step by.
 *
 * `pics -e` asks for it, and every iteration that takes a step -- `ist`,
 * `fista`, `eulermaruyama`, `chambolle_pock` -- divides by what comes back.
 * It is a power iteration from a random start, so it draws on BART's own
 * generator: a loop written outside the library has to ask for it here, at
 * the point in the sequence the library would have asked, or the draws that
 * follow it are different ones.
 *
 * `A` and `cclambda` are the encoding and the quadratic weight, together the
 * operator `lsqr` builds.  `proxes`, when given, are terms whose transforms
 * are added to it -- which is what `iter2_chambolle_pock` does, and only it;
 * the proximal iterations take the encoding alone.
 *
 * Returns 0 and writes `out`, or a negative code.
 */
int bartorch_maxeigen(const bartorch_linop* handle, float cclambda,
		int nprox, const bartorch_prox* const* proxes,
		int iterations, double* out)
{
	if ((NULL == handle) || (NULL == out) || (1 > iterations))
		return -1;

	if ((0 < nprox) && (NULL == proxes))
		return -1;

	const struct linop_s* model_op = bartorch_linop_unwrap(handle);

	if (NULL == model_op)
		return -1;

	auto iov = linop_domain(model_op);

	PTR_ALLOC(struct maxeigen_data, data);
	SET_TYPEID(maxeigen_data, data);

	data->lambda = cclambda;
	data->size = 2 * md_calc_size(iov->N, iov->dims);	// FIXME: assume complex
	data->model_op = linop_clone(model_op);

	const struct operator_s* normal = operator_create(iov->N, iov->dims, iov->N, iov->dims,
			CAST_UP(PTR_PASS(data)), maxeigen_apply, maxeigen_del);

	for (int i = 0; i < nprox; i++) {

		if ((NULL == proxes[i]) || (NULL == proxes[i]->trafo)) {

			operator_free(normal);
			return -1;
		}

		auto tmp = normal;
		normal = operator_plus_create(normal, proxes[i]->trafo->normal);
		operator_free(tmp);
	}

	*out = estimate_maxeigenval_sameplace(normal, iterations, NULL);

	operator_free(normal);

	return 0;
}

int bartorch_solve(const bartorch_linop* handle,
		const char* algorithm,
		const char* const* reg_kinds, const long* reg_xflags, const long* reg_jflags,
		const float* reg_lambda, const int* reg_k,
		const bartorch_prox* const* reg_ops, int n_reg,
		float cclambda, int maxiter, float step, int eigen, int hogwild,
		float admm_rho, int admm_maxitercg, float cg_tol,
		int admm_dynamic_rho, int admm_dynamic_tau, int admm_relative_norm, int admm_fast,
		float fista_p, float fista_q, float fista_r,
		float sigma_tau_ratio, int adaptive_step,
		int warmstart,
		const bartorch_linop* precond,
		const bartorch_linop* em_precond, float em_precond_diag, float em_precond_tol,
		int em_precond_maxiter,
		void* x, const void* y, long* iterations)
{
	const struct linop_s* model_op = bartorch_linop_unwrap(handle);

	if (NULL == model_op)
		return -1;

	/* The one preconditioner BART's least-squares solvers take.
	 *
	 * `lsqr2_create` chains it onto the normal operator and onto the
	 * adjoint, so what the iteration sees is `M (A^H A + lambda) x = M A^H y`
	 * -- left preconditioning by composition.  It is plumbed all the way
	 * through `sense_recon_create` and `pics.c` passes NULL, so nothing on
	 * the command line has ever used it.
	 *
	 * `conjgrad` itself has no preconditioner argument at all; this is the
	 * only place one enters.  M has to map the image to itself, and for
	 * conjugate gradients to mean anything it has to be positive definite. */
	const struct operator_s* precond_op = NULL;

	if (NULL != precond) {

		const struct linop_s* m = bartorch_linop_unwrap(precond);

		if (NULL == m)
			return -1;

		precond_op = m->forward;
	}

	enum algo_t algo = algo_by_name(algorithm);

	if ((enum algo_t)-1 == algo)
		return -2;

	/* The `-R` strings, read by BART's own parser, so that a specification
	 * means here exactly what it means on the command line. */
	struct opt_reg_s ropts;

	/* `opt_reg_init` returns a default for an unrelated flag, not success;
	 * `pics` ignores it.  Its `lambda` is left where it puts it: that field
	 * only ever reaches `opt_reg_configure`, which turns a bare `pics -r`
	 * into an implicit `-R Q` term, and here the terms are the caller's
	 * objects and nothing is configured. */
	(void)opt_reg_init(&ropts);

	/* Each term filled straight into the table `opt_reg` would have parsed a
	 * string into, so that what `opt_reg_configure` builds from here is what
	 * it builds for the tool. */
	if (n_reg > NUM_REGS)
		return -5;

	for (int i = 0; i < n_reg; i++) {

		int xform;

		if (0 != xform_by_name(reg_kinds[i], &xform))
			return -4;

		ropts.regs[i].xform = xform;
		ropts.regs[i].xflags = (unsigned long)reg_xflags[i];
		ropts.regs[i].jflags = (unsigned long)reg_jflags[i];
		ropts.regs[i].lambda = reg_lambda[i];
		ropts.regs[i].k = reg_k[i];
		ropts.regs[i].graph_file = NULL;
		ropts.regs[i].asl = false;
	}

	ropts.r = n_reg;

	const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };

	long img_dims[DIMS];
	md_copy_dims(DIMS, img_dims, linop_domain(model_op)->dims);

	long ksp_dims[DIMS];
	md_copy_dims(DIMS, ksp_dims, linop_codomain(model_op)->dims);

	/* The terms were built when the caller made them, so nothing is
	 * configured here: what the solver is given is the operators the
	 * objects have been holding. */
	for (int i = 0; i < n_reg; i++) {

		if (NULL == reg_ops[i])
			return -7;

		thresh_ops[i] = reg_ops[i]->op;
		trafos[i] = reg_ops[i]->trafo;

		/* A wavelet threshold spins its transform by a random shift drawn
		 * from a generator of its own, seeded at one when BART makes the
		 * operator.  The tool builds a fresh one per run; a term here is
		 * kept, so the generator is put back where a fresh one would have
		 * it and a reused term answers as the tool does. */
		if (L1WAV == reg_ops[i]->xform)
			wavthresh_rand_state_set(reg_ops[i]->op, 1);
	}

	int nr_penalties = ropts.r;

	if (ALGO_DEFAULT == algo)
		algo = italgo_choose(nr_penalties, ropts.regs);

	/* The step the tool settles on when none was asked for.  `italgo_config`
	 * takes whatever it is given, so a proximal-gradient iteration reached
	 * from here and one reached through `pics` would otherwise step
	 * differently -- which is a different answer, not a different default. */
	if (   ((ALGO_IST == algo) || (ALGO_FISTA == algo) || (ALGO_PRIDU == algo))
	    && (-1. == step))
		step = 0.95;

	struct admm_conf admm = {
		(bool)admm_dynamic_rho, (bool)admm_dynamic_tau, (bool)admm_relative_norm,
		(0. < admm_rho) ? admm_rho : iter_admm_defaults.rho,
		(0 < admm_maxitercg) ? admm_maxitercg : iter_admm_defaults.maxitercg,
		(bool)admm_fast };
	struct fista_conf fista = { { fista_p, fista_q, fista_r }, false };
	struct pridu_conf pridu = { (0. < sigma_tau_ratio) ? sigma_tau_ratio : 1., (bool)adaptive_step };

	struct iter it = italgo_config(algo, nr_penalties, ropts.regs, maxiter,
			step, eigen ? 30 : 0, hogwild, admm, fista, pridu, (bool)warmstart);

	/* `italgo_config` takes no tolerance and leaves conjugate gradients at
	 * BART's default of zero. */
	if (ALGO_CG == algo) {

		CAST_DOWN(iter_conjgrad_conf, CAST_DOWN(iter_call_s, it.iconf)->_conf)->tol = cg_tol;
		nr_penalties = 0;
	}

	/* The sampler's own preconditioner, which is a different thing from
	 * `lsqr`'s and the only genuinely preconditioned conjugate gradients in
	 * BART: `eulermaruyama_precond` solves `(M^H M + diag) o = x` with
	 * `conjgrad` at every step.  `italgo_config` cannot be told about it and
	 * `pics` has no flag for it, so this is the only way to reach it. */
	if ((ALGO_EULERMARUYAMA == algo) && (0. < em_precond_diag)) {

		struct iter_eulermaruyama_conf* em =
			CAST_DOWN(iter_eulermaruyama_conf, CAST_DOWN(iter_call_s, it.iconf)->_conf);

		em->precond_diag = em_precond_diag;
		em->precond_tol = em_precond_tol;
		em->precond_max_iter = em_precond_maxiter;
		em->precond_linop = (NULL == em_precond) ? NULL : bartorch_linop_unwrap(em_precond);
	}

	/* Only three of the iterations take the regularizers' transforms; the
	 * rest assert that they were not given any.  `pics` decides the same
	 * way, and getting it wrong is an assertion rather than a wrong answer. */
	bool trafos_cond = (   (ALGO_PRIDU == algo)
			    || (ALGO_ADMM == algo)
			    || (   (ALGO_NIHT == algo)
				&& (NIHTWAV == ropts.regs[0].xform)));

	struct lsqr_conf conf = lsqr_defaults;
	/* `pics` fills this from its own `-q`, not from `-r`: the first is the
	 * weight in the normal equations, the second the regularizers'. */
	conf.lambda = cclambda;
	conf.it_gpu = false;
	conf.warmstart = (bool)warmstart;

	/* `lsqr2_create` takes a reference and the composite drops it when it is
	 * freed, so the caller's own reference has to be one this owns: the
	 * handle's belongs to the host, and letting the driver's bookkeeping
	 * reach it corrupts the heap the moment the host frees the operator. */
	const struct linop_s* owned = linop_clone(model_op);

	struct counting_monitor counter = { { NULL, counting_monitor_fun, NULL, 0., 0. }, 0 };

	lsqr2(DIMS, &conf, it.italgo, it.iconf, owned,
			nr_penalties, thresh_ops, trafos_cond ? trafos : NULL,
			img_dims, (complex float*)x, ksp_dims, (const complex float*)y,
			precond_op, (NULL != iterations) ? &counter.super : NULL);

	if (NULL != iterations)
		*iterations = counter.count;

	linop_free(owned);
	italgo_config_free(it);

	return 0;
}

float bartorch_scaling_norm(long size, const void* image, float rescale, int compat, float p)
{
	/* On the host, and on a copy: the estimate is read off order statistics
	 * and BART sorts the array it is handed to get them. */
	complex float* tmp = md_alloc(1, MD_DIMS(size), sizeof(complex float));
	md_copy(1, MD_DIMS(size), tmp, image, sizeof(complex float));

	float scale = estimate_scaling_norm(rescale, (int)size, tmp, (bool)compat, p);

	md_free(tmp);

	return scale;
}

const char* bartorch_solve_error(int code)
{
	switch (code) {

	case  0: return "";
	case -1: return "the operator is not one this can solve against";
	case -2: return "no such algorithm";
	case -4: return "no such regularization term";
	case -5: return "more regularization terms than BART holds";
	case -6: return "this term extends the optimisation variable, and BART configures those with the whole set";
	case -7: return "a regularization term was not built";
	default: return "unknown";
	}
}
