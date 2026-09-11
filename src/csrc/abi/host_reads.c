/*
 * The entry points BART reads on the host.
 *
 * A tool works on the memory it is handed, and most of what it does goes
 * through `md_` operations that dispatch on where a pointer is.  These few
 * walk an array element by element instead -- sizing an image from a
 * trajectory, taking a median of k-space -- so a pointer on a card would be
 * dereferenced on the host.  BART guards them against its own virtual
 * pointers the same way and no differently, by working on a copy; these
 * answer for a device pointer, and the translation units they come from are
 * compiled with the originals under another name.
 */
#include <complex.h>
#include <stdbool.h>

#include "misc/mri.h"
#include "misc/mri2.h"

#include "num/flpmath.h"
#include "num/multind.h"

#include "sense/optcom.h"

#include "include/bartorch.h"

extern void bart_estimate_im_dims(int N, unsigned long flags, long dims[N], const long tdims[N], const complex float* traj);
extern void bart_estimate_fast_sq_im_dims(int N, long dims[3], const long tdims[N], const complex float* traj);
extern float bart_estimate_scaling_norm(float rescale, int imsize, complex float* tmpnorm, bool compat, float p);

/* A copy of the trajectory on the host, or NULL when it is already there. */
static complex float* host_copy(int N, const long tdims[N], const complex float* traj)
{
	if (!bartorch_on_device(traj))
		return NULL;

	complex float* host = md_alloc(N, tdims, CFL_SIZE);

	md_copy(N, tdims, host, traj, CFL_SIZE);

	return host;
}

void estimate_im_dims(int N, unsigned long flags, long dims[N], const long tdims[N], const complex float* traj)
{
	complex float* host = host_copy(N, tdims, traj);

	bart_estimate_im_dims(N, flags, dims, tdims, (NULL != host) ? host : traj);

	md_free(host);
}

void estimate_fast_sq_im_dims(int N, long dims[3], const long tdims[N], const complex float* traj)
{
	complex float* host = host_copy(N, tdims, traj);

	bart_estimate_fast_sq_im_dims(N, dims, tdims, (NULL != host) ? host : traj);

	md_free(host);
}

/* The scale a solve is normalised by, which BART finds by sorting k-space and
 * reading off a median.  It sorts what it is given, and BART's own guard sorts
 * a copy and throws it away, so this does no less. */
float estimate_scaling_norm(float rescale, int imsize, complex float* tmpnorm, bool compat, float p)
{
	if (!bartorch_on_device(tmpnorm))
		return bart_estimate_scaling_norm(rescale, imsize, tmpnorm, compat, p);

	long dims[1] = { imsize };
	complex float* host = md_alloc(1, dims, CFL_SIZE);

	md_copy(1, dims, host, tmpnorm, CFL_SIZE);

	float ret = bart_estimate_scaling_norm(rescale, imsize, host, compat, p);

	md_free(host);

	return ret;
}
