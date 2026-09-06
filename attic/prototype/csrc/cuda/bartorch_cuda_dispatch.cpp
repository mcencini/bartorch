/*
 * cuda/bartorch_cuda_dispatch.cpp — GPU dispatch mode for bartorch run().
 *
 * When all inputs to run() are CUDA tensors, bartorch enters "GPU dispatch
 * mode" before calling bart_command():
 *
 *   1. bartorch_enable_gpu_dispatch(device_id) is called.
 *   2. Input tensors are registered with their *device* pointers (zero-copy).
 *   3. __wrap_memcfl_create intercepts BART's output-CFL allocation and
 *      redirects it to cudaMalloc — so the output buffer also lives on the
 *      same GPU.  BART's fftc/linop code sees both input and output as GPU
 *      pointers → cuFFT / CUDA kernels are used throughout.
 *   4. After bart_command(), bartorch_disable_gpu_dispatch() resets the flag.
 *   5. The caller retrieves the output GPU pointer, copies it into a new
 *      torch::Tensor on the same device, and calls cudaFree() on the
 *      BART-allocated buffer.
 *
 * --wrap,memcfl_create must be added to the linker command so that
 * __wrap_memcfl_create overrides the symbol in bart_static.
 *
 * Thread safety
 * -------------
 * BART itself is not thread-safe (it uses global state), so concurrent
 * calls to run() are not supported.  The global dispatch flag is therefore
 * not guarded by a mutex.
 */

#ifdef USE_CUDA

#include <cuda_runtime.h>
#include <cstddef>

// BART forward declarations needed for io_calc_size and memcfl_register.
extern "C" {
    long   io_calc_size(int D, const long dims[], size_t element_size);
    void   memcfl_register(const char* name, int D, const long dims[],
                           void* data, int managed);
    // The "real" (original, pre-wrap) memcfl_create that allocates CPU memory.
    void*  __real_memcfl_create(const char* name, int D, const long dims[]);
}

// ---------------------------------------------------------------------------
// Global GPU dispatch state
// ---------------------------------------------------------------------------

static bool s_gpu_dispatch_active = false;
static int  s_gpu_dispatch_device = 0;

void bartorch_enable_gpu_dispatch(int device_id)
{
    s_gpu_dispatch_active = true;
    s_gpu_dispatch_device = device_id;
}

void bartorch_disable_gpu_dispatch()
{
    s_gpu_dispatch_active = false;
    s_gpu_dispatch_device = 0;
}

bool bartorch_is_gpu_dispatch_active()
{
    return s_gpu_dispatch_active;
}

int bartorch_gpu_dispatch_device()
{
    return s_gpu_dispatch_device;
}

// ---------------------------------------------------------------------------
// __wrap_memcfl_create — redirect CFL output allocation to GPU
// ---------------------------------------------------------------------------
//
// When GPU dispatch is active, allocate the output CFL buffer on the GPU
// via cudaMalloc and register it as a *non-managed* memcfl entry.
// "Non-managed" means memcfl_unlink() will NOT call xfree() on it.
// The caller (bartorch_ext.cpp::run()) is responsible for cudaFree().
//
// When GPU dispatch is inactive (CPU path), fall through to the real
// memcfl_create which allocates with xmalloc.
//
extern "C" void* __wrap_memcfl_create(const char* name, int D, const long dims[])
{
    if (!s_gpu_dispatch_active)
        return __real_memcfl_create(name, D, dims);

    // Compute byte size of the complex float buffer.
    long sz = io_calc_size(D, dims, sizeof(float) * 2);
    if (sz <= 0)
        return __real_memcfl_create(name, D, dims);  // fallback on bad dims

    // Set the correct CUDA device before allocating.
    cudaSetDevice(s_gpu_dispatch_device);

    void* ptr = nullptr;
    cudaError_t err = cudaMalloc(&ptr, (size_t)sz);
    if (err != cudaSuccess || ptr == nullptr) {
        // Allocation failed — fall back to CPU so BART still runs (on CPU).
        return __real_memcfl_create(name, D, dims);
    }

    // Zero-initialise so BART accumulate-adds don't see garbage.
    cudaMemset(ptr, 0, (size_t)sz);

    // Register as non-managed: memcfl_unlink won't call free() on ptr.
    memcfl_register(name, D, dims, ptr, /*managed=*/0);

    return ptr;
}

#endif  // USE_CUDA
