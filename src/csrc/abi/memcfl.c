/*
 * The in-memory array registry, replacing BART's misc/memcfl.c.
 *
 * BART resolves any file name ending in ".mem" here.  Arrays the host
 * registers stay the host's; arrays BART creates are allocated through
 * the host's allocator callback so they are host arrays from the start,
 * on whatever device the host chooses, and go back through the host's
 * free callback when unlinked.
 */
#include <complex.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "misc/debug.h"
#include "misc/memcfl.h"
#include "misc/misc.h"

#include "include/bartorch.h"
#include "substitute/backend.h"

enum owner { OWNER_HOST, OWNER_HOST_ALLOC, OWNER_XMALLOC };

struct entry {

	char* name;
	int D;
	long* dims;
	complex float* data;
	int refcount;
	enum owner owner;
	struct entry* next;
};

static struct entry* list = NULL;
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;

static bartorch_alloc_fn g_alloc = NULL;
static bartorch_free_fn g_free = NULL;
static void* g_ctx = NULL;

void bartorch_set_allocator(bartorch_alloc_fn alloc, bartorch_free_fn free_, void* ctx)
{
	pthread_mutex_lock(&lock);
	g_alloc = alloc;
	g_free = free_;
	g_ctx = ctx;
	pthread_mutex_unlock(&lock);
}

static struct entry* find_locked(const char* name)
{
	for (struct entry* e = list; NULL != e; e = e->next)
		if (0 == strcmp(e->name, name))
			return e;

	return NULL;
}

/* An array fits a request for D dimensions when every dimension beyond D is one. */
static bool fits_locked(const struct entry* e, int D)
{
	for (int i = D; i < e->D; i++)
		if (1 != e->dims[i])
			return false;

	return true;
}

static struct entry* find_data_locked(const complex float* data)
{
	for (struct entry* e = list; NULL != e; e = e->next)
		if (e->data == data)
			return e;

	return NULL;
}

static void insert_locked(const char* name, int D, const long* dims, complex float* data, enum owner owner)
{
	struct entry* e = malloc(sizeof(*e));
	e->name = strdup(name);
	e->D = D;
	e->dims = malloc(sizeof(long) * (size_t)D);
	memcpy(e->dims, dims, sizeof(long) * (size_t)D);
	e->data = data;
	e->refcount = 1;
	e->owner = owner;
	e->next = list;
	list = e;
}

static void release(struct entry* e)
{
	switch (e->owner) {

	case OWNER_HOST:
		break;

	case OWNER_HOST_ALLOC:
		if (NULL != g_free)
			g_free(g_ctx, e->data);
		break;

	case OWNER_XMALLOC:
		xfree(e->data);
		break;
	}

	free(e->name);
	free(e->dims);
	free(e);
}

static void unlink_locked(struct entry** ep)
{
	struct entry* e = *ep;
	*ep = e->next;
	release(e);
}

/* BART-facing API (misc/memcfl.h). */

void memcfl_register(const char* name, int D, const long dims[D], complex float* data, bool managed)
{
	pthread_mutex_lock(&lock);
	insert_locked(name, D, dims, data, managed ? OWNER_XMALLOC : OWNER_HOST);
	pthread_mutex_unlock(&lock);
}

complex float* memcfl_create(const char* name, int D, const long dims[D])
{
	pthread_mutex_lock(&lock);

	struct entry** ep = &list;

	while (NULL != *ep) {

		if (0 == strcmp((*ep)->name, name))
			unlink_locked(ep);
		else
			ep = &(*ep)->next;
	}

	bartorch_alloc_fn alloc = g_alloc;
	void* ctx = g_ctx;
	pthread_mutex_unlock(&lock);

	complex float* data = NULL;
	enum owner owner = OWNER_XMALLOC;

	if (NULL != alloc) {

		data = alloc(ctx, D, dims);
		owner = OWNER_HOST_ALLOC;

	} else {

		data = xmalloc((size_t)io_calc_size(D, dims, sizeof(complex float)));
	}

	if (NULL == data)
		error("bartorch: host allocation failed for %s\n", name);

	pthread_mutex_lock(&lock);
	insert_locked(name, D, dims, data, owner);
	pthread_mutex_unlock(&lock);

	return data;
}

bool memcfl_exists(const char* name)
{
	pthread_mutex_lock(&lock);
	bool found = (NULL != find_locked(name));
	pthread_mutex_unlock(&lock);
	return found;
}

const char** memcfl_list_all(void)
{
	pthread_mutex_lock(&lock);

	int count = 0;

	for (struct entry* e = list; NULL != e; e = e->next)
		count++;

	const char** out = xmalloc(sizeof(const char*) * (size_t)(count + 1));
	out[0] = (const char*)(uintptr_t)count;

	int i = 1;

	for (struct entry* e = list; NULL != e; e = e->next)
		out[i++] = e->name;

	pthread_mutex_unlock(&lock);
	return out;
}

complex float* memcfl_load(const char* name, int D, long dims[D])
{
	pthread_mutex_lock(&lock);

	struct entry* e = find_locked(name);

	if (NULL == e) {

		pthread_mutex_unlock(&lock);
		error("bartorch: no array registered as %s\n", name);
	}

	if (!fits_locked(e, D)) {

		pthread_mutex_unlock(&lock);
		error("bartorch: %s has %d dimensions, more than the %d requested\n", name, e->D, D);
	}

	for (int i = 0; i < D; i++)
		dims[i] = (i < e->D) ? e->dims[i] : 1;

	e->refcount++;
	complex float* data = e->data;

	pthread_mutex_unlock(&lock);
	return data;
}

bool memcfl_unmap(const complex float* p)
{
	pthread_mutex_lock(&lock);

	struct entry* e = find_data_locked(p);

	if (NULL == e) {

		pthread_mutex_unlock(&lock);
		return false;
	}

	if (e->refcount > 0)
		e->refcount--;

	pthread_mutex_unlock(&lock);
	return true;
}

void memcfl_unlink(const char* name)
{
	pthread_mutex_lock(&lock);

	for (struct entry** ep = &list; NULL != *ep; ep = &(*ep)->next) {

		if (0 == strcmp((*ep)->name, name)) {

			unlink_locked(ep);
			break;
		}
	}

	pthread_mutex_unlock(&lock);
}

/* Host-facing API (include/bartorch.h). */

int bartorch_register(const char* name, int D, const long* dims, void* data)
{
	if ((D < 1) || (D > BARTORCH_DIMS) || (NULL == data))
		return -1;

	pthread_mutex_lock(&lock);

	for (struct entry** ep = &list; NULL != *ep; ep = &(*ep)->next) {

		if (0 == strcmp((*ep)->name, name)) {

			unlink_locked(ep);
			break;
		}
	}

	insert_locked(name, D, dims, data, OWNER_HOST);
	pthread_mutex_unlock(&lock);
	return 0;
}

int bartorch_exists(const char* name)
{
	return memcfl_exists(name) ? 1 : 0;
}

int bartorch_lookup(const char* name, int D, long* dims, void** data)
{
	pthread_mutex_lock(&lock);

	struct entry* e = find_locked(name);

	if ((NULL == e) || !fits_locked(e, D)) {

		pthread_mutex_unlock(&lock);
		return -1;
	}

	for (int i = 0; i < D; i++)
		dims[i] = (i < e->D) ? e->dims[i] : 1;

	*data = e->data;

	pthread_mutex_unlock(&lock);
	return 0;
}

int bartorch_unlink(const char* name)
{
	pthread_mutex_lock(&lock);

	int found = -1;

	for (struct entry** ep = &list; NULL != *ep; ep = &(*ep)->next) {

		if (0 == strcmp((*ep)->name, name)) {

			unlink_locked(ep);
			found = 0;
			break;
		}
	}

	pthread_mutex_unlock(&lock);
	return found;
}

int bartorch_unlink_all(void)
{
	pthread_mutex_lock(&lock);

	int n = 0;

	while (NULL != list) {

		unlink_locked(&list);
		n++;
	}

	pthread_mutex_unlock(&lock);
	return n;
}
