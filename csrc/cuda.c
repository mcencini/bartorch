/*
 * The CUDA side of the C ABI.
 *
 * Every entry point exists in both builds; without CUDA they report that the
 * library has none.  BART finds device memory through
 * cudaPointerGetAttributes (CUDA_GET_CUDA_DEVICE_NUM), so a tensor the host
 * allocated on a device is recognised without being registered anywhere, and
 * an array BART creates comes from the host's allocator on whichever device
 * the host selected.
 *
 * BART keeps its own streams.  Ordering them against the caller's stream is
 * an event in each direction: BART's streams wait on what the caller has
 * already queued, and the caller waits on what BART leaves behind.
 */
#include <stdbool.h>
#include <stdlib.h>

#include "include/bartorch.h"

#ifdef USE_CUDA

#include <cuda_runtime_api.h>

#include "misc/misc.h"

#include "num/gpuops.h"
#include "num/init.h"

static int current_device = -1;

int bartorch_cuda_built(void)
{
	return 1;
}

int bartorch_cuda_device_count(void)
{
	int count = 0;

	if (cudaSuccess != cudaGetDeviceCount(&count)) {

		cudaGetLastError();
		return 0;
	}

	return count;
}

int bartorch_cuda_enable(int device)
{
	if (device < 0) {

		bart_use_gpu = false;
		current_device = -1;
		return 0;
	}

	if (device >= bartorch_cuda_device_count())
		return -1;

	if (cudaSuccess != cudaSetDevice(device))
		return -1;

	bart_use_gpu = true;
	num_init_gpu_support();
	current_device = device;
	return 0;
}

int bartorch_cuda_device(void)
{
	return current_device;
}

/* Two streams rather than BART's one.
 *
 * An operator that has to bring a slab of sensitivities across fetches the
 * next while the card works on this one, and that needs a stream to fetch on;
 * with one stream the loop waits for each crossing.  Nothing else in BART
 * changes: a stream count above one only lets its own parallel loops fan out,
 * which is what the count is for. */
__attribute__((constructor))
static void bartorch_streams_default(void)
{
	cuda_num_streams = 2;
}

int bartorch_cuda_set_streams(int n)
{
	if ((n < 1) || (n > CUDA_MAX_STREAMS))
		return -1;

	cuda_num_streams = n;
	return 0;
}

int bartorch_cuda_get_streams(void)
{
	return cuda_num_streams;
}

int bartorch_cuda_use_memcache(int enable)
{
	if (!enable)
		cuda_memcache_off();
	else
		cuda_memcache_clear();

	return 0;
}

/* Hold BART's streams until the work already queued on `stream` has run. */
int bartorch_cuda_wait_for_stream(void* stream)
{
	if (-1 == current_device)
		return -1;

	cudaEvent_t event;

	if (cudaSuccess != cudaEventCreateWithFlags(&event, cudaEventDisableTiming))
		return -1;

	int ret = 0;

	if (cudaSuccess != cudaEventRecord(event, (cudaStream_t)stream))
		ret = -1;

	for (int i = 0; (0 == ret) && (i < cuda_num_streams); i++)
		if (cudaSuccess != cudaStreamWaitEvent(cuda_get_stream_by_id(i), event, 0))
			ret = -1;

	cudaEventDestroy(event);
	return ret;
}

/* Hold `stream` until the work BART queued on its own streams has run. */
int bartorch_cuda_signal_stream(void* stream)
{
	if (-1 == current_device)
		return -1;

	int ret = 0;

	for (int i = 0; (0 == ret) && (i < cuda_num_streams); i++) {

		cudaEvent_t event;

		if (cudaSuccess != cudaEventCreateWithFlags(&event, cudaEventDisableTiming))
			return -1;

		if (   (cudaSuccess != cudaEventRecord(event, cuda_get_stream_by_id(i)))
		    || (cudaSuccess != cudaStreamWaitEvent((cudaStream_t)stream, event, 0)))
			ret = -1;

		cudaEventDestroy(event);
	}

	return ret;
}

/* Host memory a copy can leave the card's own engine to fetch.
 *
 * An asynchronous copy out of pageable memory is not one: the driver stages it
 * through a buffer of its own and the call blocks while it does, so nothing
 * overlaps.  Page-locked memory is what the engine can read directly, and it
 * is what the function a normal is streamed from is kept in.  Where there is
 * no card, or where the card will not lock that much, it is ordinary
 * memory. */
void* bartorch_host_alloc(long size, int pinned)
{
	if (0 >= size)
		return NULL;

	if (pinned && (-1 != current_device)) {

		void* ptr = NULL;

		if (cudaSuccess == cudaMallocHost(&ptr, (size_t)size))
			return ptr;

		cudaGetLastError();
	}

	return xmalloc((size_t)size);
}

void bartorch_host_free(void* ptr)
{
	if (NULL == ptr)
		return;

	struct cudaPointerAttributes attr;

	if ((cudaSuccess == cudaPointerGetAttributes(&attr, ptr))
			&& (cudaMemoryTypeHost == attr.type)) {

		cudaFreeHost(ptr);
		return;
	}

	cudaGetLastError();
	xfree(ptr);
}

/* A stream of its own for bringing the function over.
 *
 * BART takes its stream from the OpenMP thread it is on, so a copy issued
 * from the thread that is about to convolve lands on the stream that will
 * convolve and cannot run beside it.  This is a stream nothing else uses,
 * ordered against BART's by events alone -- which is what lets the set that
 * will be wanted next cross while the card works on the one it has, without
 * the second host thread that would make BART's own threading nested.
 *
 * Two slots, so `filled` says a slot has arrived and `freed` says the card has
 * finished reading it; a copy waits for the second before overwriting.  One of
 * these belongs to each operator that streams, because the slots it names are
 * that operator's.
 */
struct bartorch_stage {

	cudaStream_t stream;
	cudaEvent_t filled[2];
	cudaEvent_t freed[2];
};

int bartorch_cuda_stage_open(void** stage)
{
	if ((NULL == stage) || (-1 == current_device))
		return -1;

	struct bartorch_stage* s = xmalloc(sizeof *s);

	if (cudaSuccess != cudaStreamCreateWithFlags(&s->stream, cudaStreamNonBlocking)) {

		xfree(s);
		return -1;
	}

	for (int i = 0; i < 2; i++)
		if ((cudaSuccess != cudaEventCreateWithFlags(&s->filled[i], cudaEventDisableTiming))
				|| (cudaSuccess != cudaEventCreateWithFlags(&s->freed[i], cudaEventDisableTiming))) {

			cudaStreamDestroy(s->stream);
			xfree(s);
			return -1;
		}

	*stage = s;
	return 0;
}

void bartorch_cuda_stage_close(void* stage)
{
	if (NULL == stage)
		return;

	struct bartorch_stage* s = stage;

	cudaStreamSynchronize(s->stream);

	for (int i = 0; i < 2; i++) {

		cudaEventDestroy(s->filled[i]);
		cudaEventDestroy(s->freed[i]);
	}

	cudaStreamDestroy(s->stream);
	xfree(s);
}

/* Start a slot's crossing, once the card has finished reading what is in it. */
int bartorch_cuda_stage_copy(void* stage, int slot, void* dst, const void* src, long size)
{
	if ((NULL == stage) || (0 > slot) || (1 < slot))
		return -1;

	struct bartorch_stage* s = stage;

	if (   (cudaSuccess != cudaStreamWaitEvent(s->stream, s->freed[slot], 0))
	    || (cudaSuccess != cudaMemcpyAsync(dst, src, (size_t)size, cudaMemcpyHostToDevice, s->stream))
	    || (cudaSuccess != cudaEventRecord(s->filled[slot], s->stream)))
		return -1;

	return 0;
}

/* Hold BART's stream until a slot has arrived. */
int bartorch_cuda_stage_wait(void* stage, int slot)
{
	if ((NULL == stage) || (0 > slot) || (1 < slot))
		return -1;

	struct bartorch_stage* s = stage;

	return (cudaSuccess == cudaStreamWaitEvent(cuda_get_stream(), s->filled[slot], 0)) ? 0 : -1;
}

/* Say that everything BART has queued so far has finished with a slot. */
int bartorch_cuda_stage_release(void* stage, int slot)
{
	if ((NULL == stage) || (0 > slot) || (1 < slot))
		return -1;

	struct bartorch_stage* s = stage;

	return (cudaSuccess == cudaEventRecord(s->freed[slot], cuda_get_stream())) ? 0 : -1;
}

int bartorch_on_device(const void* ptr)
{
	return cuda_ondevice(ptr) ? 1 : 0;
}

long bartorch_cuda_free_memory(void)
{
	size_t free_bytes = 0;
	size_t total_bytes = 0;

	if (cudaSuccess != cudaMemGetInfo(&free_bytes, &total_bytes)) {

		cudaGetLastError();
		return -1;
	}

	return (long)free_bytes;
}

#else /* !USE_CUDA */

int bartorch_on_device(const void* ptr) { (void)ptr; return 0; }
int bartorch_cuda_built(void) { return 0; }
int bartorch_cuda_device_count(void) { return 0; }
int bartorch_cuda_enable(int device) { return (device < 0) ? 0 : -1; }
int bartorch_cuda_device(void) { return -1; }
/* Two streams rather than BART's one.
 *
 * An operator that has to bring a slab of sensitivities across fetches the
 * next while the card works on this one, and that needs a stream to fetch on;
 * with one stream the loop waits for each crossing.  Nothing else in BART
 * changes: a stream count above one only lets its own parallel loops fan out,
 * which is what the count is for. */
__attribute__((constructor))
static void bartorch_streams_default(void)
{
	cuda_num_streams = 2;
}

int bartorch_cuda_set_streams(int n) { (void)n; return -1; }
int bartorch_cuda_get_streams(void) { return 0; }
int bartorch_cuda_use_memcache(int enable) { (void)enable; return -1; }
int bartorch_cuda_wait_for_stream(void* stream) { (void)stream; return -1; }
int bartorch_cuda_signal_stream(void* stream) { (void)stream; return -1; }
void* bartorch_host_alloc(long size, int pinned) { (void)pinned; return (0 < size) ? xmalloc((size_t)size) : NULL; }
void bartorch_host_free(void* ptr) { if (NULL != ptr) xfree(ptr); }
int bartorch_cuda_stage_open(void** stage) { (void)stage; return -1; }
void bartorch_cuda_stage_close(void* stage) { (void)stage; }
int bartorch_cuda_stage_copy(void* stage, int slot, void* dst, const void* src, long size)
{ (void)stage; (void)slot; (void)dst; (void)src; (void)size; return -1; }
int bartorch_cuda_stage_wait(void* stage, int slot) { (void)stage; (void)slot; return -1; }
int bartorch_cuda_stage_release(void* stage, int slot) { (void)stage; (void)slot; return -1; }
long bartorch_cuda_free_memory(void) { return -1; }

#endif
