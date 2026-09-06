/*
 * cuda/cufinufft_grid.cu — cuFINUFFT-backed GPU gridder shim.
 *
 * Linked via:
 *   -Wl,--wrap,cuda_grid
 *   -Wl,--wrap,cuda_gridH
 *   -Wl,--wrap,cuda_apply_rolloff_correction2
 *
 * When BART's noncart/nufft.c detects GPU trajectory (cuda_ondevice(traj))
 * it calls cuda_grid / cuda_gridH (BART's Kaiser-Bessel GPU gridder in
 * noncart/gpu_grid.cu).  This shim intercepts those calls and replaces
 * them with cuFINUFFT type-1 / type-2 spread-only operations, exactly
 * mirroring what finufft_grid.cpp does on the CPU side.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Semantics
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  cuda_grid  (adjoint: nonuniform k-space → oversampled Cartesian grid):
 *      cuFINUFFT type-1, gpu_spreadinterponly=1
 *
 *  cuda_gridH (forward: oversampled Cartesian grid → nonuniform k-space):
 *      cuFINUFFT type-2, gpu_spreadinterponly=1
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Coordinate mapping (identical to finufft_grid.cpp CPU path)
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  x_cufinufft_d = conf->os * Re(traj[3j+d]) * 2π / grid_dims[d]
 *
 *  This maps BART's non-oversampled grid units to cuFINUFFT's [-π, π) range.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Rolloff correction
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  __wrap_cuda_apply_rolloff_correction2 applies the ES-kernel deconvolution
 *  weight on GPU.  The weight per spatial index p in dimension d is:
 *
 *    w(p, d) = 1 / hat_phi_ES( (p - n_d/2) / (n_d * os) )
 *
 *  Kernel parameters (tol=1e-6, upsampfac=2.0, from FINUFFT setup_spreader):
 *    nspread = 7,  ES_beta = 16.1,  ES_hw = 3.5
 *
 *  The rolloff weight array is pre-computed once per call using a 256-point
 *  midpoint quadrature and then applied in a simple element-wise CUDA kernel.
 */

#ifdef BARTORCH_USE_CUFINUFFT

#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cufinufft.h>

#include <cstdint>
#include <cmath>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

extern "C" {
#ifndef _Bool
typedef bool _Bool;
#endif
#include "noncart/grid.h"
#include "noncart/gpu_grid.h"
}

// ---------------------------------------------------------------------------
// Forward declarations for the "real" (pre-wrap) BART GPU gridder symbols.
// The linker renames cuda_grid → __real_cuda_grid etc. when --wrap is active.
// ---------------------------------------------------------------------------
extern "C" {
void __real_cuda_grid(const struct grid_conf_s* conf,
                      const long ksp_dims[4], const long trj_strs[4],
                      const _Complex float* traj,
                      const long grid_dims[4], const long grid_strs[4],
                      _Complex float* grid,
                      const long ksp_strs[4], const _Complex float* src);

void __real_cuda_gridH(const struct grid_conf_s* conf,
                       const long ksp_dims[4], const long trj_strs[4],
                       const _Complex float* traj,
                       const long ksp_strs[4], _Complex float* dst,
                       const long grid_dims[4], const long grid_strs[4],
                       const _Complex float* grid);

void __real_cuda_apply_rolloff_correction2(float os, float width, float beta,
                                           int N, const long dims[4],
                                           const long ostrs[4],
                                           _Complex float* dst,
                                           const long istrs[4],
                                           const _Complex float* src);
}

// ---------------------------------------------------------------------------
// Compile-time NUFFT accuracy for cuFINUFFT
// ---------------------------------------------------------------------------
static constexpr float CUFINUFFT_GRID_TOL = 1e-6f;

// ES-kernel parameters for nspread=7, upsampfac=2.0 (same as finufft_grid.cpp)
static constexpr float ES_BETA = 16.1f;
static constexpr float ES_HW   = 3.5f;   // nspread / 2

// =============================================================================
// cuFINUFFT plan cache
// =============================================================================
//
// Mirrors the CPU plan cache in finufft_grid.cpp.  Plans are keyed by the
// trajectory device pointer + grid geometry so that within a CG iterate (many
// forward/adjoint calls with identical trajectory) `makeplan`+`setpts` is only
// executed once.  The trajectory pointer is stable for the lifetime of a
// BartLinopHandle (encoding_op) and throughout a single bart_command() run
// (CLI, e.g. pics).
//
// Thread safety: protected by a std::mutex.
// Invalidation:
//   - BartLinopHandle::~BartLinopHandle() calls bartorch_cufinufft_invalidate_traj()
//     before freeing the trajectory tensor.
//   - After each bart_command() call, bartorch_cufinufft_flush_all() is called
//     to purge all entries (prevents ABA reuse of CLI-allocated CFL pointers).
// =============================================================================

struct CuFinufftPlanKey {
    const void* traj_ptr; ///< device pointer of the trajectory CFL
    long        Nx, Ny, Nz;
    long        n_transf;
    int         type;     ///< 1 = adjoint (cuda_grid), 2 = forward (cuda_gridH)
    int         ndim;

    bool operator==(const CuFinufftPlanKey& o) const noexcept
    {
        return traj_ptr == o.traj_ptr
            && Nx       == o.Nx   && Ny == o.Ny && Nz == o.Nz
            && n_transf == o.n_transf
            && type     == o.type && ndim == o.ndim;
    }
};

struct CuFinufftPlanKeyHash {
    // Boost-style hash_combine using the 64-bit golden ratio constant
    // (0x9e3779b97f4a7c15 = 2^64 / φ) for better avalanche behaviour than
    // a plain XOR — same technique used by Boost and the CPU-side cache.
    static std::size_t combine(std::size_t seed, std::size_t val) noexcept
    {
        return seed ^ (val + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
    }

    std::size_t operator()(const CuFinufftPlanKey& k) const noexcept
    {
        std::size_t h = std::hash<const void*>{}(k.traj_ptr);
        h = combine(h, std::hash<long>{}(k.Nx));
        h = combine(h, std::hash<long>{}(k.Ny));
        h = combine(h, std::hash<long>{}(k.Nz));
        h = combine(h, std::hash<long>{}(k.n_transf));
        h = combine(h, std::hash<int> {}(k.type));
        h = combine(h, std::hash<int> {}(k.ndim));
        return h;
    }
};

static std::mutex s_cu_plan_cache_mutex;
static std::unordered_map<CuFinufftPlanKey, cufinufftf_plan, CuFinufftPlanKeyHash>
    s_cu_plan_cache;
/// Secondary index: traj_ptr → list of keys, for O(1) invalidation.
static std::unordered_map<const void*, std::vector<CuFinufftPlanKey>>
    s_cu_traj_index;

/// Remove all cache entries whose traj_ptr matches @p ptr.
/// Called from BartLinopHandle's destructor before the trajectory tensor is
/// freed on the GPU side, preventing ABA pointer reuse.
extern "C" void bartorch_cufinufft_invalidate_traj(const void* ptr)
{
    std::unique_lock<std::mutex> lk(s_cu_plan_cache_mutex);
    auto idx_it = s_cu_traj_index.find(ptr);
    if (idx_it == s_cu_traj_index.end()) return;

    for (const auto& key : idx_it->second) {
        auto it = s_cu_plan_cache.find(key);
        if (it != s_cu_plan_cache.end()) {
            cufinufftf_destroy(it->second);
            s_cu_plan_cache.erase(it);
        }
    }
    s_cu_traj_index.erase(idx_it);
}

/// Flush the entire cuFINUFFT plan cache.
/// Called at the end of each bart_command() run to prevent ABA reuse of
/// CLI-allocated CFL trajectory pointers across successive command calls.
extern "C" void bartorch_cufinufft_flush_all()
{
    std::unique_lock<std::mutex> lk(s_cu_plan_cache_mutex);
    for (auto& kv : s_cu_plan_cache)
        cufinufftf_destroy(kv.second);
    s_cu_plan_cache.clear();
    s_cu_traj_index.clear();
}

// ---------------------------------------------------------------------------
// CUDA kernel: extract and scale trajectory coordinates from GPU memory
// ---------------------------------------------------------------------------
//
// BART traj layout: complex float, re(traj[3*j + d]) is the coordinate in
// non-oversampled grid units [-N_d/2, N_d/2].
// cuFINUFFT expects device float arrays in [-π, π).
//
// Scaling: x = conf->os * re(traj[3*j+d]) * 2π / grid_dims[d]
//
static __global__ void k_extract_coords(
    long          M,
    const float2* traj,           // device ptr, complex float → float2
    float         sx, float sy, float sz,
    int           ndim,
    float*        xj,
    float*        yj,
    float*        zj)
{
    long j = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= M) return;

    xj[j] = traj[3*j + 0].x * sx;
    if (ndim >= 2) yj[j] = traj[3*j + 1].x * sy;
    if (ndim >= 3) zj[j] = traj[3*j + 2].x * sz;
}

// ---------------------------------------------------------------------------
// CUDA kernel: apply ES rolloff weight  dst[i] = src[i] * w(i)
// ---------------------------------------------------------------------------
//
// The weight is 1 / hat_phi_ES(xi) where xi = (p - n/2) / (n * os).
// hat_phi_ES(xi) = 2 * ∫_0^{hw} exp(beta * sqrt(1 - (x/hw)^2)) cos(2π xi x) dx
// evaluated via 256-point midpoint quadrature.
//
static __device__ float es_kernel_ft(float xi_d, float beta, float hw)
{
    // 256-point midpoint quadrature on [0, hw]
    const int N_QUAD = 256;
    const float dx = hw / (float)N_QUAD;
    float sum = 0.0f;
    for (int k = 0; k < N_QUAD; ++k) {
        float x   = (k + 0.5f) * dx;
        float arg = 1.0f - (x / hw) * (x / hw);
        float ker = (arg > 0.0f) ? expf(beta * sqrtf(arg)) : 0.0f;
        sum += ker * cosf(2.0f * (float)M_PI * xi_d * x);
    }
    return 2.0f * sum * dx;
}

// 3-D rolloff kernel — N = total spatial voxels = Nx*Ny*Nz
static __global__ void k_apply_rolloff(
    long           N_spatial,
    long           Nx, long Ny, long Nz,
    float          os,
    float          beta, float hw,
    float2*        dst,            // cuFloatComplex → float2
    const float2*  src)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_spatial) return;

    // Recover (px, py, pz) from flat index
    long px = idx % Nx;
    long py = (idx / Nx) % Ny;
    long pz = (idx / Nx / Ny);

    float wx = 1.0f, wy = 1.0f, wz = 1.0f;

    {
        float xi = ((float)px - (float)Nx / 2.0f) / ((float)Nx * os);
        float h  = es_kernel_ft(xi, beta, hw);
        wx = (fabsf(h) > 1e-12f) ? 1.0f / h : 1.0f;
    }
    if (Ny > 1) {
        float xi = ((float)py - (float)Ny / 2.0f) / ((float)Ny * os);
        float h  = es_kernel_ft(xi, beta, hw);
        wy = (fabsf(h) > 1e-12f) ? 1.0f / h : 1.0f;
    }
    if (Nz > 1) {
        float xi = ((float)pz - (float)Nz / 2.0f) / ((float)Nz * os);
        float h  = es_kernel_ft(xi, beta, hw);
        wz = (fabsf(h) > 1e-12f) ? 1.0f / h : 1.0f;
    }

    float w = wx * wy * wz;
    dst[idx].x = src[idx].x * w;
    dst[idx].y = src[idx].y * w;
}

// ---------------------------------------------------------------------------
// Helper: spatial dimensionality from grid_dims[0..2]
// ---------------------------------------------------------------------------
static int spatial_ndim_gpu(const long grid_dims[4])
{
    if (grid_dims[2] > 1) return 3;
    if (grid_dims[1] > 1) return 2;
    return 1;
}

// ---------------------------------------------------------------------------
// RAII wrapper for CUDA device memory allocations used inside the plan helper
// ---------------------------------------------------------------------------
struct CuDevicePtr {
    float* ptr = nullptr;
    explicit CuDevicePtr(size_t n_floats)
    {
        if (n_floats > 0)
            cudaMalloc(&ptr, n_floats * sizeof(float));
    }
    ~CuDevicePtr() { if (ptr) cudaFree(ptr); }
    // Non-copyable
    CuDevicePtr(const CuDevicePtr&)            = delete;
    CuDevicePtr& operator=(const CuDevicePtr&) = delete;
    // Implicit conversion so it can be passed directly to CUDA APIs
    operator float*() const { return ptr; }
    bool ok() const { return ptr != nullptr; }
};

// ---------------------------------------------------------------------------
// get_or_make_cufinufft_plan — cache-aware plan helper
// ---------------------------------------------------------------------------
//
// On a cache hit (same traj_ptr + grid geometry): returns the cached plan.
// On a cache miss: calls makeplan + setpts, inserts into cache, then frees
//   the temporary device coordinate arrays (cuFINUFFT copies them into the
//   plan's internal storage during setpts).
//
// Returns nullptr if makeplan or setpts fails (caller should fall back to
// BART's native GPU gridder).
//
static cufinufftf_plan get_or_make_cufinufft_plan(
    const _Complex float* traj_c,   ///< device trajectory pointer (cache key)
    int                   type,     ///< 1 = adjoint, 2 = forward
    int                   ndim,
    long Nx, long Ny, long Nz,
    long M_per,
    int  n_transf,
    float sx, float sy, float sz)
{
    CuFinufftPlanKey key{ (const void*)traj_c, Nx, Ny, Nz,
                          (long)n_transf, type, ndim };

    {
        std::unique_lock<std::mutex> lk(s_cu_plan_cache_mutex);
        auto it = s_cu_plan_cache.find(key);
        if (it != s_cu_plan_cache.end())
            return it->second;   // cache hit: skip makeplan+setpts
    }

    // Cache miss — build plan.
    // Allocate temporary device coordinate arrays (freed automatically on scope exit).
    CuDevicePtr d_xj(M_per);
    CuDevicePtr d_yj(ndim >= 2 ? M_per : 0);
    CuDevicePtr d_zj(ndim >= 3 ? M_per : 0);

    if (!d_xj.ok() || (ndim >= 2 && !d_yj.ok()) || (ndim >= 3 && !d_zj.ok()))
        return nullptr;   // cudaMalloc failed

    // Extract trajectory coordinates on GPU.
    int threads = 256;
    int blocks  = (int)((M_per + threads - 1) / threads);
    k_extract_coords<<<blocks, threads>>>(
        M_per, (const float2*)traj_c,
        sx, sy, sz, ndim,
        d_xj, d_yj, d_zj);

    // Make cuFINUFFT plan.
    cufinufft_opts opts;
    cufinufft_default_opts(&opts);
    opts.gpu_spreadinterponly = 1;
    opts.upsampfac             = 2.0;
    opts.debug                 = 0;

    cufinufftf_plan plan = nullptr;
    int64_t n_modes[3] = { Nx, Ny, Nz };
    int ier = cufinufftf_makeplan(
        type, ndim, n_modes,
        +1, n_transf,
        CUFINUFFT_GRID_TOL, &plan, &opts);
    if (ier != 0 || plan == nullptr)
        return nullptr;   // d_xj/d_yj/d_zj freed by CuDevicePtr destructors

    // Set nonuniform points (cuFINUFFT copies coords into its internal buffers).
    ier = cufinufftf_setpts(plan, M_per, d_xj,
                            (ndim >= 2) ? (float*)d_yj : nullptr,
                            (ndim >= 3) ? (float*)d_zj : nullptr,
                            0, nullptr, nullptr, nullptr);
    // d_xj/d_yj/d_zj freed automatically at end of scope (CuDevicePtr destructor).

    if (ier != 0) {
        cufinufftf_destroy(plan);
        return nullptr;
    }

    // Insert into cache.
    {
        std::unique_lock<std::mutex> lk(s_cu_plan_cache_mutex);
        s_cu_plan_cache.emplace(key, plan);
        s_cu_traj_index[key.traj_ptr].push_back(key);
    }
    return plan;
}

// ---------------------------------------------------------------------------
// __wrap_cuda_grid — adjoint gridding: nonuniform k-space → Cartesian grid
// ---------------------------------------------------------------------------
//
// BART signature:
//   cuda_grid(conf, ksp_dims[4], trj_strs[4], traj, grid_dims[4], grid_strs[4],
//             grid, ksp_strs[4], src)
//
// Falls back to BART's KB GPU gridder when conf->os != 2.0 (decomp mode).
//
extern "C" void __wrap_cuda_grid(
    const struct grid_conf_s* conf,
    const long                ksp_dims[4],
    const long                trj_strs[4],
    const _Complex float*     traj_c,
    const long                grid_dims[4],
    const long                grid_strs[4],
    _Complex float*           grid_c,
    const long                ksp_strs[4],
    const _Complex float*     src_c)
{
    if (conf->os != 2.0f) {
        __real_cuda_grid(conf, ksp_dims, trj_strs, traj_c,
                         grid_dims, grid_strs, grid_c, ksp_strs, src_c);
        return;
    }

    const int  ndim  = spatial_ndim_gpu(grid_dims);
    const long Nx    = grid_dims[0];
    const long Ny    = (ndim >= 2) ? grid_dims[1] : 1L;
    const long Nz    = (ndim >= 3) ? grid_dims[2] : 1L;

    int  n_transf = (int)ksp_dims[3];
    long M_per    = ksp_dims[1] * ksp_dims[2];

    if (M_per == 0 || n_transf == 0) return;

    const float TWO_PI = 2.0f * (float)M_PI;
    float sx = conf->os * TWO_PI / (float)Nx;
    float sy = (ndim >= 2) ? conf->os * TWO_PI / (float)Ny : 0.f;
    float sz = (ndim >= 3) ? conf->os * TWO_PI / (float)Nz : 0.f;

    cufinufftf_plan plan = get_or_make_cufinufft_plan(
        traj_c, 1, ndim, Nx, Ny, Nz, M_per, n_transf, sx, sy, sz);
    if (plan == nullptr) {
        __real_cuda_grid(conf, ksp_dims, trj_strs, traj_c,
                         grid_dims, grid_strs, grid_c, ksp_strs, src_c);
        return;
    }

    // Zero the output grid (BART may not clear it before calling cuda_grid).
    cudaMemset(grid_c, 0,
               (size_t)(n_transf * Nx * Ny * Nz) * sizeof(cuFloatComplex));

    // type-1 execute: c = src (NU k-space), fk = grid (Cartesian)
    int ier = cufinufftf_execute(plan,
        (cuFloatComplex*)const_cast<_Complex float*>(src_c),
        (cuFloatComplex*)grid_c);

    if (ier != 0) {
        cudaMemset(grid_c, 0,
                   (size_t)(n_transf * Nx * Ny * Nz) * sizeof(cuFloatComplex));
        __real_cuda_grid(conf, ksp_dims, trj_strs, traj_c,
                         grid_dims, grid_strs, grid_c, ksp_strs, src_c);
    }
}

// ---------------------------------------------------------------------------
// __wrap_cuda_gridH — forward gridding: Cartesian grid → nonuniform k-space
// ---------------------------------------------------------------------------
extern "C" void __wrap_cuda_gridH(
    const struct grid_conf_s* conf,
    const long                ksp_dims[4],
    const long                trj_strs[4],
    const _Complex float*     traj_c,
    const long                ksp_strs[4],
    _Complex float*           dst_c,
    const long                grid_dims[4],
    const long                grid_strs[4],
    const _Complex float*     grid_c)
{
    if (conf->os != 2.0f) {
        __real_cuda_gridH(conf, ksp_dims, trj_strs, traj_c,
                          ksp_strs, dst_c, grid_dims, grid_strs, grid_c);
        return;
    }

    const int  ndim  = spatial_ndim_gpu(grid_dims);
    const long Nx    = grid_dims[0];
    const long Ny    = (ndim >= 2) ? grid_dims[1] : 1L;
    const long Nz    = (ndim >= 3) ? grid_dims[2] : 1L;

    int  n_transf = (int)ksp_dims[3];
    long M_per    = ksp_dims[1] * ksp_dims[2];

    if (M_per == 0 || n_transf == 0) return;

    const float TWO_PI = 2.0f * (float)M_PI;
    float sx = conf->os * TWO_PI / (float)Nx;
    float sy = (ndim >= 2) ? conf->os * TWO_PI / (float)Ny : 0.f;
    float sz = (ndim >= 3) ? conf->os * TWO_PI / (float)Nz : 0.f;

    cufinufftf_plan plan = get_or_make_cufinufft_plan(
        traj_c, 2, ndim, Nx, Ny, Nz, M_per, n_transf, sx, sy, sz);
    if (plan == nullptr) {
        __real_cuda_gridH(conf, ksp_dims, trj_strs, traj_c,
                          ksp_strs, dst_c, grid_dims, grid_strs, grid_c);
        return;
    }

    // Zero k-space output.
    cudaMemset(dst_c, 0,
               (size_t)(n_transf * M_per) * sizeof(cuFloatComplex));

    // type-2 execute: fk = grid (Cartesian, read-only), c = dst (NU output)
    int ier = cufinufftf_execute(plan,
        (cuFloatComplex*)dst_c,
        (cuFloatComplex*)const_cast<_Complex float*>(grid_c));

    if (ier != 0) {
        cudaMemset(dst_c, 0,
                   (size_t)(n_transf * M_per) * sizeof(cuFloatComplex));
        __real_cuda_gridH(conf, ksp_dims, trj_strs, traj_c,
                          ksp_strs, dst_c, grid_dims, grid_strs, grid_c);
    }
}

// ---------------------------------------------------------------------------
// __wrap_cuda_apply_rolloff_correction2 — ES-kernel rolloff on GPU
// ---------------------------------------------------------------------------
//
// Applied after spreading (cuda_grid) and FFT to deconvolve the ES kernel.
// The weight is computed per spatial voxel using the same 256-point quadrature
// as finufft_grid.cpp.
//
// When os != 2.0, delegate to BART's original KB GPU rolloff.
//
extern "C" void __wrap_cuda_apply_rolloff_correction2(
    float           os,
    float           width,
    float           beta_kb,
    int             N,
    const long      dims[4],
    const long      ostrs[4],
    _Complex float* dst_c,
    const long      istrs[4],
    const _Complex float* src_c)
{
    if (os != 2.0f) {
        __real_cuda_apply_rolloff_correction2(os, width, beta_kb, N,
                                              dims, ostrs, dst_c, istrs, src_c);
        return;
    }

    // Assume standard contiguous layout for dimensions [Nx, Ny, Nz, batch...].
    // batch = ∏ dims[3..N-1]
    long Nx = dims[0];
    long Ny = (N >= 2) ? dims[1] : 1L;
    long Nz = (N >= 3) ? dims[2] : 1L;
    long N_batch = 1;
    for (int d = 3; d < N; ++d) N_batch *= dims[d];
    long N_spatial = Nx * Ny * Nz;

    // ES kernel parameters (same as finufft_grid.cpp, nspread=7, upsampfac=2.0)
    float es_beta = ES_BETA;
    float es_hw   = ES_HW;

    int threads = 256;
    long total   = N_batch * N_spatial;
    int  blocks  = (int)((total + threads - 1) / threads);

    // Apply rolloff to each (batch, spatial) element independently.
    // For simplicity, treat all batches together (rolloff is the same per
    // spatial position regardless of coil/batch index).
    for (long b = 0; b < N_batch; ++b) {
        _Complex float* dst_b = dst_c + b * N_spatial;
        const _Complex float* src_b = src_c + b * N_spatial;

        k_apply_rolloff<<<blocks, threads>>>(
            N_spatial, Nx, Ny, Nz, os, es_beta, es_hw,
            (float2*)dst_b, (const float2*)src_b);
    }
}

#endif  // BARTORCH_USE_CUFINUFFT
