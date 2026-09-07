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

int bartorch_cuda_built(void) { return 0; }
int bartorch_cuda_device_count(void) { return 0; }
int bartorch_cuda_enable(int device) { return (device < 0) ? 0 : -1; }
int bartorch_cuda_device(void) { return -1; }
int bartorch_cuda_set_streams(int n) { (void)n; return -1; }
int bartorch_cuda_get_streams(void) { return 0; }
int bartorch_cuda_use_memcache(int enable) { (void)enable; return -1; }
int bartorch_cuda_wait_for_stream(void* stream) { (void)stream; return -1; }
int bartorch_cuda_signal_stream(void* stream) { (void)stream; return -1; }
long bartorch_cuda_free_memory(void) { return -1; }

#endif
