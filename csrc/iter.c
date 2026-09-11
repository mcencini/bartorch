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
#include "num/ops.h"
#include "num/ops_p.h"
#include "num/iovec.h"

#include "linops/linop.h"

#include "iter/iter.h"
#include "iter/iter2.h"
#include "iter/lsqr.h"
#include "iter/prox.h"
#include "iter/thresh.h"

#include "grecon/optreg.h"
#include "grecon/italgo.h"

#include "include/bartorch.h"

/* The operator handles the ABI hands out; the host sees only the pointer. */
struct bartorch_linop_s;
extern const struct linop_s* bartorch_linop_unwrap(const struct bartorch_linop_s* h);

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

int bartorch_solve(const bartorch_linop* handle,
		const char* algorithm,
		const char* const* regularizers, int n_reg,
		float lambda, float cclambda, int maxiter, float step, int eigen, int hogwild,
		float admm_rho, int admm_maxitercg,
		float fista_p, float fista_q, float fista_r,
		int llr_blk, int shift_mode, const char* wavelet,
		int warmstart,
		void* x, const void* y)
{
	const struct linop_s* model_op = bartorch_linop_unwrap(handle);

	if (NULL == model_op)
		return -1;

	enum algo_t algo = algo_by_name(algorithm);

	if ((enum algo_t)-1 == algo)
		return -2;

	/* The `-R` strings, read by BART's own parser, so that a specification
	 * means here exactly what it means on the command line. */
	struct opt_reg_s ropts;

	/* `opt_reg_init` returns a default for an unrelated flag, not success;
	 * `pics` ignores it.  It leaves `lambda` at -1, which is what the
	 * regularizers read as "not given", so only overwrite it when it was. */
	(void)opt_reg_init(&ropts);

	if (0. <= lambda)
		ropts.lambda = lambda;

	/* BART's convention for a conversion function is that true is failure;
	 * `opts.c` prints the usage and errors on one. */
	for (int i = 0; i < n_reg; i++)
		if (opt_reg((void*)&ropts, 'R', regularizers[i]))
			return -4;

	const struct operator_p_s* thresh_ops[NUM_REGS] = { NULL };
	const struct linop_s* trafos[NUM_REGS] = { NULL };
	const long (*sdims[NUM_REGS])[DIMS + 1] = { NULL };

	long img_dims[DIMS];
	md_copy_dims(DIMS, img_dims, linop_domain(model_op)->dims);

	long ksp_dims[DIMS];
	md_copy_dims(DIMS, ksp_dims, linop_codomain(model_op)->dims);

	opt_reg_configure(DIMS, img_dims, &ropts, thresh_ops, trafos, sdims,
			llr_blk, shift_mode, (NULL != wavelet) ? wavelet : "dau2",
			false, ITER_DIM);

	int nr_penalties = ropts.r + ropts.sr;

	if (ALGO_DEFAULT == algo)
		algo = italgo_choose(nr_penalties, ropts.regs);

	struct admm_conf admm = { false, false, false,
		(0. < admm_rho) ? admm_rho : iter_admm_defaults.rho,
		(0 < admm_maxitercg) ? admm_maxitercg : iter_admm_defaults.maxitercg,
		false };
	struct fista_conf fista = { { fista_p, fista_q, fista_r }, false };
	struct pridu_conf pridu = { 1., false };

	struct iter it = italgo_config(algo, nr_penalties, ropts.regs, maxiter,
			step, eigen ? 30 : 0, hogwild, admm, fista, pridu, (bool)warmstart);

	if (ALGO_CG == algo)
		nr_penalties = 0;

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

	lsqr2(DIMS, &conf, it.italgo, it.iconf, owned,
			nr_penalties, thresh_ops, trafos_cond ? trafos : NULL,
			img_dims, (complex float*)x, ksp_dims, (const complex float*)y,
			NULL, NULL);

	linop_free(owned);
	italgo_config_free(it);
	opt_reg_free(&ropts, thresh_ops, trafos);

	return 0;
}

const char* bartorch_solve_error(int code)
{
	switch (code) {

	case  0: return "";
	case -1: return "the operator is not one this can solve against";
	case -2: return "no such algorithm";
	case -4: return "BART would not parse one of the regularizers";
	default: return "unknown";
	}
}
