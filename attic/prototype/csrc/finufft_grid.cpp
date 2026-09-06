/*
 * finufft_grid.cpp — FINUFFT-backed spread/interpolate replacements for
 *                    BART's grid2() and grid2H().
 *
 * Linked via  -Wl,--wrap,grid2  -Wl,--wrap,grid2H
 *             -Wl,--wrap,rolloff_correction
 *             -Wl,--wrap,apply_rolloff_correction
 *             -Wl,--wrap,apply_rolloff_correction2
 * so that ALL BART CLI commands that reach the NUFFT codepath (nufft, pics,
 * moba, nlinv, …) automatically use FINUFFT's optimised multi-threaded
 * ES-kernel spreader in place of BART's single-threaded Kaiser-Bessel gridder,
 * AND use a matching ES-kernel rolloff correction instead of BART's KB one.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Semantics
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  grid2  (adjoint, nonuniform k-space → oversampled Cartesian grid):
 *      FINUFFT type-1, spreadinterponly=1
 *      f_k  = Σ_j  c_j · φ( (k/N) − x_j/(2π) )
 *      where φ is FINUFFT's piecewise-polynomial ES spreading kernel.
 *
 *  grid2H (forward, oversampled Cartesian grid → nonuniform k-space):
 *      FINUFFT type-2, spreadinterponly=1
 *      c_j  = Σ_k  f_k · φ( (k/N) − x_j/(2π) )
 *
 * Both operations are purely the spread/interpolation step.  The FFT step
 * remains in BART's nufft.c unchanged.  The rolloff (deconvolution) step is
 * also replaced — see __wrap_rolloff_correction and friends below.
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Coordinate mapping
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  BART trajectory element:
 *    real( traj[3*j + d] )  ∈ [-N_d/2, N_d/2]  (non-oversampled grid units)
 *
 *  BART's gridH() applies:  pos_d = conf->os × Re(traj[d]) + grid_dims[d]/2
 *    to get a 0-indexed position on the OVERSAMPLED grid of size grid_dims[d].
 *
 *  FINUFFT spread-only expects x_j ∈ [-π, π) where the positive π edge
 *  corresponds to grid position N_fine (wraps to 0).  Mapping:
 *
 *    x_finufft_d = conf->os × Re(traj[3j+d]) × 2π / grid_dims[d]
 *
 *  Derivation: pos_d / grid_dims[d] × 2π − π
 *     = (conf->os × Re(traj) + grid_dims[d]/2) / grid_dims[d] × 2π − π
 *     = conf->os × Re(traj) × 2π / grid_dims[d] + π − π
 *     = conf->os × Re(traj) × 2π / grid_dims[d]                     ✓
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Rolloff (deconvolution) correction — ES-kernel FT
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  After spreading (grid2) and FFT, BART calls apply_rolloff_correction to
 *  deconvolve the spreading kernel from image space.  The correct deconvolution
 *  factor for FINUFFT's ES ("exponential of semicircle") kernel is:
 *
 *    w(p, d) = 1 / hat_phi_ES( (p - n_d/2) / (n_d * os) )
 *
 *  where hat_phi_ES(xi) = 2 * ∫₀^{J/2} exp(β·√(1−(2x/J)²)) · cos(2π·xi·x) dx
 *  is the continuous FT of the ES kernel at cycles-per-sample frequency xi.
 *
 *  This replaces BART's KB correction (1 / ftkb(beta, xi*width) / width).
 *  Both corrections are derived from the respective kernel FTs; using matching
 *  spreading kernel and rolloff correction gives exact deconvolution.
 *
 *  Kernel parameters for FINUFFT_GRID_TOL=1e-6f and upsampfac=2.0
 *  (from FINUFFT's setup_spreader, src/spreadinterp.cpp):
 *    nspread = ceil(-log10(tol/10)) = 7
 *    betaovns = 2.30  (default for nspread > 4, upsampfac=2.0)
 *    ES_beta  = betaovns * nspread = 16.1
 *    ES_hw    = nspread / 2       = 3.5
 *
 *  The kernel FT integral is evaluated with a 256-point midpoint rule on
 *  [0, ES_hw], which gives machine-precision accuracy (the integrand is
 *  smooth and decays to exp(-ES_beta) ≈ 1e-7 at the integration boundary).
 *
 *  The __wrap symbols intercept:
 *    rolloff_correction         — fills a 3-D correction weight array
 *    apply_rolloff_correction2  — applies correction with strides + batch
 *    apply_rolloff_correction   — contiguous wrapper → apply_rolloff_correction2
 *
 * ─────────────────────────────────────────────────────────────────────────────
 * Batch / coil handling
 * ─────────────────────────────────────────────────────────────────────────────
 *
 *  FINUFFT's n_transf parameter is set to  ∏ ksp_dims[3..D-1]  so that coils
 *  (and any higher batch dims) are processed in a single finufftf_execute()
 *  call, exploiting FINUFFT's OpenMP-parallel batch loop.
 *
 *  Memory layout compatibility:
 *    BART ksp_dims = [1, M1, M2, Ncoils, ...]  (dim-0 fastest)
 *    → src[j + M*coil + M*Ncoils*frame + ...]  (M = M1*M2)
 *    FINUFFT batch: c[i*M + j] for transform i and sample j
 *    These are the same layout                                        ✓
 *
 *    BART grid_dims = [Nx, Ny, Nz, Ncoils, ...]
 *    → dst[n + N_spatial*coil + ...]  (N_spatial = Nx*Ny*Nz)
 *    FINUFFT batch: fk[i*N_spatial + n]
 *    These are the same layout                                        ✓
 */

#include <complex>
#include <cstring>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <finufft.h>   // single + double precision public API

// BART's grid_conf_s definition — must be included with C linkage.
// grid.h uses _Bool (C99 built-in) which is not a keyword in strict C++ mode
// (it would be __Bool or just bool).  We provide a C++ typedef alias before
// including the header.  GCC/Clang accept this as a harmless no-op on recent
// toolchains where _Bool is already defined as a built-in.
extern "C" {
#ifndef _Bool
typedef bool _Bool;  // C99 ↔ C++ compatibility alias for BART's grid.h
#endif
#include "noncart/grid.h"

// Forward declarations for BART's original (pre-wrap) gridding functions.
// In decomp mode (conf->os != 2.0) BART calls grid2/grid2H with os=1 and
// a non-zero conf->shift to implement its sub-grid decomposition.  FINUFFT
// spread-only is designed for the os=2 full-grid path; in decomp mode we fall
// back to the original KB gridder so that conf->shift and the halved width are
// honoured correctly.
void __real_grid2 (const struct grid_conf_s*, int, const long*, const _Complex float*,
                   const long*, _Complex float*, const long*, const _Complex float*);
void __real_grid2H(const struct grid_conf_s*, int, const long*, const _Complex float*,
                   const long*, _Complex float*, const long*, const _Complex float*);

// Forward declarations for BART's original (pre-wrap) rolloff functions.
// BART's decomp path (default, conf->os=1 for grid2 sub-calls) uses the KB
// gridder (via __real_grid2/__real_grid2H above).  The rolloff is precomputed
// and baked into data->linphase, so it must also use the KB kernel to match.
// These __real_* redirects ensure KB rolloff is always used, keeping the
// gridder + rolloff pair consistent.
void __real_rolloff_correction      (float, float, float, const long[3], _Complex float*);
void __real_apply_rolloff_correction(float, float, float, int, const long*, _Complex float*, const _Complex float*);
void __real_apply_rolloff_correction2(float, float, float, int, const long*, const long*, _Complex float*, const long*, const _Complex float*);
}

// ─────────────────────────────────────────────────────────────────────────────
// FINUFFT plan cache
//
// Plans are keyed by (traj_ptr, type, ndim, Nx, Ny, Nz, n_transf).
// Using the raw trajectory pointer as part of the key is safe because:
//  • bartorch keeps the trajectory tensor alive in BartLinopHandle::kept_ for
//    the lifetime of the linop, so the pointer is stable.
//  • BartLinopHandle's destructor calls bartorch_finufft_invalidate_traj()
//    (declared in lib_ops.cpp / lib_shim.c) before freeing the tensor, so stale
//    entries are removed before the pointer can be reused by a new allocation.
//
// Cache hit path: skip makeplan + setpts + destroy — only execute.
// Cache miss path: makeplan + setpts + execute, then cache the plan.
// ─────────────────────────────────────────────────────────────────────────────
struct FinufftPlanKey {
    const void* traj_ptr;   // raw data pointer of the BART trajectory buffer
    long        Nx, Ny, Nz; // uniform grid dims (oversampled)
    long        n_transf;   // batch size (coils × …)
    int         type;       // 1 (grid2/adjoint) or 2 (grid2H/forward)
    int         ndim;       // spatial dimensionality (1, 2, or 3)

    bool operator==(const FinufftPlanKey& o) const noexcept
    {
        return traj_ptr == o.traj_ptr
            && Nx == o.Nx && Ny == o.Ny && Nz == o.Nz
            && n_transf == o.n_transf && type == o.type && ndim == o.ndim;
    }
};

struct FinufftPlanKeyHash {
    // Boost-style hash_combine: avoids XOR-only poor avalanche behaviour.
    static std::size_t combine(std::size_t seed, std::size_t val) noexcept
    {
        return seed ^ (val + 0x9e3779b97f4a7c15ULL + (seed << 6) + (seed >> 2));
    }

    std::size_t operator()(const FinufftPlanKey& k) const noexcept
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

static std::mutex s_plan_cache_mutex;
static std::unordered_map<FinufftPlanKey, finufftf_plan, FinufftPlanKeyHash>
    s_plan_cache;
/// Secondary index: traj_ptr → list of keys, for O(1) invalidation.
static std::unordered_map<const void*, std::vector<FinufftPlanKey>>
    s_traj_index;

/// Remove all cache entries whose traj_ptr matches @p ptr.
/// Called from BartLinopHandle's destructor before the trajectory tensor is
/// freed, so the address cannot be reused by a new allocation while stale
/// entries remain in the cache.
extern "C" void bartorch_finufft_invalidate_traj(const void* ptr)
{
    std::unique_lock<std::mutex> lk(s_plan_cache_mutex);
    auto idx_it = s_traj_index.find(ptr);
    if (idx_it == s_traj_index.end()) return;

    for (const auto& key : idx_it->second) {
        auto it = s_plan_cache.find(key);
        if (it != s_plan_cache.end()) {
            finufftf_destroy(it->second);
            s_plan_cache.erase(it);
        }
    }
    s_traj_index.erase(idx_it);
}

/// Flush the entire CPU FINUFFT plan cache.
/// Called at the end of each bart_command() run to prevent ABA reuse of
/// CLI-allocated CFL trajectory pointers across successive command calls.
extern "C" void bartorch_finufft_flush_all()
{
    std::unique_lock<std::mutex> lk(s_plan_cache_mutex);
    for (auto& kv : s_plan_cache)
        finufftf_destroy(kv.second);
    s_plan_cache.clear();
    s_traj_index.clear();
}

// ─────────────────────────────────────────────────────────────────────────────
// Internal helpers
// ─────────────────────────────────────────────────────────────────────────────

static constexpr float FINUFFT_GRID_TOL = 1e-6f;  // float32 NUFFT accuracy

using FC = std::complex<float>;  // alias for readability

/// Deduce spatial dimensionality from grid_dims[0..2].
static int spatial_ndim(const long grid_dims[])
{
    if (grid_dims[2] > 1) return 3;
    if (grid_dims[1] > 1) return 2;
    return 1;
}

/// Extract BART trajectory into FINUFFT coordinate arrays (in [-π, π)).
///
/// BART layout: traj[3*j + d] is complex float; real part holds the
/// coordinate in non-oversampled grid units [-N_d/2, N_d/2].
/// conf->os scales to the oversampled grid; grid_dims[d] is the oversampled
/// size.  Mapping:  x_finufft = conf->os * Re(traj[d]) * 2π / grid_dims[d]
static void extract_coords(
    long                         M,
    const long                   trj_dims[],
    const FC*                    traj,
    const long                   grid_dims[],
    float                        os,
    int                          ndim,
    std::vector<float>&          xj,
    std::vector<float>&          yj,
    std::vector<float>&          zj)
{
    static constexpr float TWO_PI = 2.0f * static_cast<float>(M_PI);

    xj.resize(M);
    if (ndim >= 2) yj.resize(M);
    if (ndim >= 3) zj.resize(M);

    const float sx = os * TWO_PI / static_cast<float>(grid_dims[0]);
    const float sy = (ndim >= 2)
                   ? os * TWO_PI / static_cast<float>(grid_dims[1]) : 0.f;
    const float sz = (ndim >= 3)
                   ? os * TWO_PI / static_cast<float>(grid_dims[2]) : 0.f;

    for (long j = 0; j < M; ++j) {
        xj[j] = traj[3*j + 0].real() * sx;
        if (ndim >= 2) yj[j] = traj[3*j + 1].real() * sy;
        if (ndim >= 3) zj[j] = traj[3*j + 2].real() * sz;
    }
}

// ─────────────────────────────────────────────────────────────────────────────
// __wrap_grid2:  adjoint gridding — nonuniform k-space → Cartesian grid
//
// BART signature (noncart/grid.h):
//   void grid2(const struct grid_conf_s* conf,
//              int D, const long trj_dims[D], const complex float* traj,
//              const long grid_dims[D], complex float* dst,
//              const long ksp_dims[D],  const complex float* src)
//
// FINUFFT call: type-1, spreadinterponly=1
//   execute(plan, c=src (NU input), fk=dst (uniform output))
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void __wrap_grid2(
    const struct grid_conf_s* conf,
    int                       D,
    const long                trj_dims[],
    const _Complex float*     traj_c,
    const long                grid_dims[],
    _Complex float*           dst_c,
    const long                ksp_dims[],
    const _Complex float*     src_c)
{
    // In BART's default decomp mode grid2 is called with conf->os=1 (not 2) and
    // a non-zero conf->shift for each sub-grid.  FINUFFT spread-only is designed
    // for the os=2 full-grid path; fall back to BART's original KB gridder for
    // any call where os != 2.0 so that the decomp shifts are honoured correctly.
    if (conf->os != 2.0f) {
        __real_grid2(conf, D, trj_dims, traj_c, grid_dims, dst_c, ksp_dims, src_c);
        return;
    }

    // Reinterpret C complex ↔ C++ complex (same layout, guaranteed by standard)
    const FC* traj = reinterpret_cast<const FC*>(traj_c);
    const FC* src  = reinterpret_cast<const FC*>(src_c);
    FC*       dst  = reinterpret_cast<FC*>(dst_c);

    const int  ndim     = spatial_ndim(grid_dims);
    const long Nx       = grid_dims[0];
    const long Ny       = (ndim >= 2) ? grid_dims[1] : 1L;
    const long Nz       = (ndim >= 3) ? grid_dims[2] : 1L;
    const long N_spatial = Nx * Ny * Nz;

    // Total k-space samples M = ∏ trj_dims[1..D-1]  (trj_dims[0] = 3, trj_dims[3] = 1)
    long M = 1;
    for (int d = 1; d < D; ++d) M *= trj_dims[d];

    // Batch count = ∏ ksp_dims[3..D-1]  (coils + higher dims)
    long n_transf = 1;
    for (int d = 3; d < D; ++d) n_transf *= ksp_dims[d];

    if (M == 0 || n_transf == 0) return;

    // ── Plan cache lookup ────────────────────────────────────────────────────
    // Key on the raw trajectory pointer: stable because BartLinopHandle keeps
    // the tensor alive and invalidates the cache entry on destruction.
    const FinufftPlanKey key{ traj_c, Nx, Ny, Nz, n_transf, 1, ndim };

    finufft_opts opts;
    finufftf_default_opts(&opts);
    opts.spreadinterponly = 1;
    opts.upsampfac        = 2.0;
    opts.debug            = 0;
    opts.showwarn         = 0;

    finufftf_plan plan = nullptr;
    bool cache_hit     = false;
    {
        std::unique_lock<std::mutex> lk(s_plan_cache_mutex);
        auto it = s_plan_cache.find(key);
        if (it != s_plan_cache.end()) {
            plan      = it->second;
            cache_hit = true;
        }
    }

    if (!cache_hit) {
        // Cache miss: extract coords and make the plan.
        std::vector<float> xj, yj, zj;
        extract_coords(M, trj_dims, traj, grid_dims, conf->os, ndim, xj, yj, zj);

        int64_t n_modes[3] = { Nx, Ny, Nz };
        int ier = finufftf_makeplan(
            1, ndim, n_modes, +1, static_cast<int>(n_transf),
            FINUFFT_GRID_TOL, &plan, &opts);
        if (ier != 0 || plan == nullptr) return;

        ier = finufftf_setpts(plan, M, xj.data(),
            (ndim >= 2) ? yj.data() : nullptr,
            (ndim >= 3) ? zj.data() : nullptr,
            0, nullptr, nullptr, nullptr);
        if (ier != 0) { finufftf_destroy(plan); return; }

        // Insert into cache and secondary index.
        {
            std::unique_lock<std::mutex> lk(s_plan_cache_mutex);
            s_plan_cache.emplace(key, plan);
            s_traj_index[traj_c].push_back(key);
        }
    }

    // Zero the output: BART's caller may not clear dst before calling grid2
    std::memset(dst, 0,
        static_cast<std::size_t>(n_transf * N_spatial) * sizeof(FC));

    // type-1 execute: weights = src (NU k-space input), result = dst (grid output)
    // FINUFFT's C API takes non-const pointers even for read-only operands;
    // the const_cast is safe because for type-1 the weights array is only read.
    finufftf_execute(plan,
        const_cast<FC*>(src),
        dst);
}

// ─────────────────────────────────────────────────────────────────────────────
// __wrap_grid2H:  forward gridding — Cartesian grid → nonuniform k-space
//
// BART signature (noncart/grid.h):
//   void grid2H(const struct grid_conf_s* conf,
//               int D, const long trj_dims[D], const complex float* traj,
//               const long ksp_dims[D],  complex float* dst,
//               const long grid_dims[D], const complex float* src)
//
// FINUFFT call: type-2, spreadinterponly=1
//   execute(plan, c=dst (NU output), fk=src (uniform input))
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void __wrap_grid2H(
    const struct grid_conf_s* conf,
    int                       D,
    const long                trj_dims[],
    const _Complex float*     traj_c,
    const long                ksp_dims[],
    _Complex float*           dst_c,
    const long                grid_dims[],
    const _Complex float*     src_c)
{
    // Fall back to BART's original KB gridder in decomp mode (conf->os != 2.0).
    // See __wrap_grid2 for the detailed rationale.
    if (conf->os != 2.0f) {
        __real_grid2H(conf, D, trj_dims, traj_c, ksp_dims, dst_c, grid_dims, src_c);
        return;
    }

    const FC* traj = reinterpret_cast<const FC*>(traj_c);
    FC*       dst  = reinterpret_cast<FC*>(dst_c);
    const FC* src  = reinterpret_cast<const FC*>(src_c);

    const int  ndim = spatial_ndim(grid_dims);
    const long Nx   = grid_dims[0];
    const long Ny   = (ndim >= 2) ? grid_dims[1] : 1L;
    const long Nz   = (ndim >= 3) ? grid_dims[2] : 1L;

    long M = 1;
    for (int d = 1; d < D; ++d) M *= trj_dims[d];

    long n_transf = 1;
    for (int d = 3; d < D; ++d) n_transf *= ksp_dims[d];

    if (M == 0 || n_transf == 0) return;

    // ── Plan cache lookup ────────────────────────────────────────────────────
    const FinufftPlanKey key{ traj_c, Nx, Ny, Nz, n_transf, 2, ndim };

    finufft_opts opts;
    finufftf_default_opts(&opts);
    opts.spreadinterponly = 1;    // interpolation step only — no FFT, no deconvolution
    opts.upsampfac        = 2.0;  // same kernel design as grid2 (see comment there)
    opts.debug            = 0;
    opts.showwarn         = 0;

    finufftf_plan plan = nullptr;
    bool cache_hit     = false;
    {
        std::unique_lock<std::mutex> lk(s_plan_cache_mutex);
        auto it = s_plan_cache.find(key);
        if (it != s_plan_cache.end()) {
            plan      = it->second;
            cache_hit = true;
        }
    }

    if (!cache_hit) {
        std::vector<float> xj, yj, zj;
        extract_coords(M, trj_dims, traj, grid_dims, conf->os, ndim, xj, yj, zj);

        int64_t n_modes[3] = { Nx, Ny, Nz };
        int ier = finufftf_makeplan(
            2, ndim, n_modes, +1, static_cast<int>(n_transf),
            FINUFFT_GRID_TOL, &plan, &opts);
        if (ier != 0 || plan == nullptr) return;

        ier = finufftf_setpts(plan, M, xj.data(),
            (ndim >= 2) ? yj.data() : nullptr,
            (ndim >= 3) ? zj.data() : nullptr,
            0, nullptr, nullptr, nullptr);
        if (ier != 0) { finufftf_destroy(plan); return; }

        {
            std::unique_lock<std::mutex> lk(s_plan_cache_mutex);
            s_plan_cache.emplace(key, plan);
            s_traj_index[traj_c].push_back(key);
        }
    }

    // Zero k-space output
    std::memset(dst, 0, static_cast<std::size_t>(n_transf * M) * sizeof(FC));

    // type-2 execute: weights = dst (NU output, will be filled),
    //                 result  = src (uniform grid input, only read)
    // const_cast on src: FINUFFT's C API takes non-const even for read-only operands.
    finufftf_execute(plan,
        dst,
        const_cast<FC*>(src));
}

// =============================================================================
// Rolloff correction — delegates to BART's original KB rolloff
// =============================================================================
//
// BART's default decomp mode uses the KB gridder (via __real_grid2/__real_grid2H
// above).  The rolloff correction must match the gridder kernel.  We delegate
// all three rolloff wrappers to the original __real_* KB implementations so
// that gridder and rolloff are always consistent.
//
// ─────────────────────────────────────────────────────────────────────────────
// __wrap_rolloff_correction
//
// Delegates unconditionally to BART's original KB rolloff so that the
// gridder (KB in decomp mode, FINUFFT in non-decomp mode) and the rolloff
// are always consistent.  BART's decomp path is the default and uses the
// original __real_grid2/__real_grid2H (KB kernel); the KB rolloff from
// __real_rolloff_correction matches that kernel exactly.
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void __wrap_rolloff_correction(
    float           os,
    float           width,
    float           beta,
    const long      dim[3],
    _Complex float* dst_c)
{
    __real_rolloff_correction(os, width, beta, dim, dst_c);
}

// ─────────────────────────────────────────────────────────────────────────────
// __wrap_apply_rolloff_correction2  (delegates to KB original)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void __wrap_apply_rolloff_correction2(
    float                os,
    float                width,
    float                beta,
    int                  N,
    const long           dims[],
    const long           ostrs[],
    _Complex float*      dst_c,
    const long           istrs[],
    const _Complex float* src_c)
{
    __real_apply_rolloff_correction2(os, width, beta, N, dims, ostrs, dst_c, istrs, src_c);
}

// ─────────────────────────────────────────────────────────────────────────────
// __wrap_apply_rolloff_correction  (delegates to KB original)
// ─────────────────────────────────────────────────────────────────────────────
extern "C" void __wrap_apply_rolloff_correction(
    float                os,
    float                width,
    float                beta,
    int                  N,
    const long           dims[],
    _Complex float*      dst_c,
    const _Complex float* src_c)
{
    __real_apply_rolloff_correction(os, width, beta, N, dims, dst_c, src_c);
}
