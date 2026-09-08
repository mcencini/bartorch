/*
 * Command execution, logging and threading behind the C ABI.
 */
#include <errno.h>
#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "misc/debug.h"
#include "misc/misc.h"
#include "misc/version.h"
#include "num/init.h"

#include "noncart/nufft.h"

#include "include/bartorch.h"
#include "backend.h"

extern int bart_command(int len, char* buf, int argc, char* argv[]);

/* A failed assertion reaches the caller instead of the process.
 *
 * BART checks its arguments with assert() and routes it through error() --
 * which the error catcher turns into a return code -- only under USE_DWARF,
 * which also wants libdw and libunwind for backtraces.  Without that define
 * assert() is glibc's, and glibc's calls abort(), so a shape a caller got
 * wrong would take the interpreter down with it.  Answering __assert_fail
 * here puts those assertions back on BART's own error path and needs neither
 * the define nor the libraries; the symbol is hidden, so it binds inside this
 * library and nothing outside it changes.
 */
void __assert_fail(const char* assertion, const char* file, unsigned int line, const char* function);

void __assert_fail(const char* assertion, const char* file, unsigned int line, const char* function)
{
	error("Assertion '%s' failed in %s:%u (%s)\n", assertion, file, line, function);
}

#ifndef BARTORCH_BUILD_INFO
#define BARTORCH_BUILD_INFO "unknown"
#endif

static pthread_mutex_t cmd_lock = PTHREAD_MUTEX_INITIALIZER;

static bartorch_log_fn g_log = NULL;
static void* g_log_ctx = NULL;

static char g_err[4096];

const char* bartorch_bart_version(void)
{
	return bart_version;
}

const char* bartorch_build_info(void)
{
	return BARTORCH_BUILD_INFO;
}

void bartorch_set_log_handler(bartorch_log_fn fn, void* ctx)
{
	g_log = fn;
	g_log_ctx = ctx;
}

static int effective_debug_level(void)
{
	if (-1 != debug_level)
		return debug_level;

	int level = DP_INFO;
	const char* str = getenv("BART_DEBUG_LEVEL");

	if (NULL != str) {

		errno = 0;
		long r = strtol(str, NULL, 10);

		if ((0 == errno) && (0 <= r) && (r < 10))
			level = (int)r;
	}

	return level;
}

void bartorch_set_debug_level(int level)
{
	debug_level = level;
}

int bartorch_get_debug_level(void)
{
	return effective_debug_level();
}

void bartorch_set_num_threads(int n)
{
	if (n < 1)
		n = 1;

	num_set_num_threads(n);
	bartorch_fft_set_num_threads(n);
}

/* What BART last said when it failed.
 *
 * A tool's message comes back with its return code; an operator has no return
 * code to carry one, so this is where the reason it refused is read from. */
const char* bartorch_last_error(void)
{
	return g_err;
}

void bartorch_clear_error(void)
{
	g_err[0] = '\0';
}

void bartorch_record_error(const char* msg)
{
	size_t used = strlen(g_err);

	if (used > 0 && used + 3 < sizeof(g_err)) {

		strcat(g_err, "; ");
		used += 2;
	}

	strncat(g_err, msg, sizeof(g_err) - used - 1);
}

static const char* level_name(int level)
{
	switch (level) {
	case DP_ERROR: return "ERROR";
	case DP_WARN: return "WARN";
	case DP_INFO: return "INFO";
	case DP_DEBUG1: return "DEBUG1";
	case DP_DEBUG2: return "DEBUG2";
	case DP_DEBUG3: return "DEBUG3";
	case DP_DEBUG4: return "DEBUG4";
	default: return "TRACE";
	}
}

void vendor_log(int level, const char* func_name, const char* file, unsigned int line, const char* message)
{
	if (level <= DP_ERROR)
		bartorch_record_error(message);

	if (level > effective_debug_level())
		return;

	if (NULL != g_log) {

		g_log(g_log_ctx, level, func_name, file, (int)line, message);
		return;
	}

	fprintf(stderr, "%s: %s\n", level_name(level), message);
	fflush(stderr);
}

int bartorch_command(int argc, const char* const* argv, char* out, size_t outlen, char* err, size_t errlen)
{
	pthread_mutex_lock(&cmd_lock);

	g_err[0] = '\0';

	char** av = calloc((size_t)argc + 2, sizeof(char*));
	av[0] = strdup("bart");

	for (int i = 0; i < argc; i++)
		av[i + 1] = strdup(argv[i]);

	/* The NUFFT options are a global the command line writes into, and this
	 * process runs more than one command, so an `-o` in one would otherwise
	 * still be in force for the next.  Zero is neither a grid nor a width
	 * anybody can ask for, which is what makes it mean "nobody asked":
	 * FINUFFT's own defaults then stand, and BART gets its two and its six
	 * back before it ever sees the conf. */
	nufft_conf_options = nufft_conf_defaults;
	nufft_conf_options.os = 0.;
	nufft_conf_options.width = 0.;

	bool have_out = (NULL != out) && (outlen > 0);

	if (have_out)
		out[0] = '\0';

	int ret = bart_command(have_out ? (int)outlen : 0, have_out ? out : NULL, argc + 1, av);

	for (int i = 0; i < argc + 1; i++)
		free(av[i]);

	free(av);

	if ((NULL != err) && (errlen > 0)) {

		strncpy(err, g_err, errlen - 1);
		err[errlen - 1] = '\0';
	}

	pthread_mutex_unlock(&cmd_lock);
	return ret;
}
