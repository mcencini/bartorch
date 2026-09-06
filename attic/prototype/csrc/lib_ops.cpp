/*
 * lib_ops.cpp — pybind11 bindings for persistent BART library operators.
 *
 * Exposes to Python:
 *
 *   BartLinopHandle         opaque owner of a BART linop_s* (auto-freed on GC)
 *   create_encoding_op()    wraps bartorch_create_encoding_op() from lib_shim.c
 *   init_lib_ops()          registers everything into the _bartorch_ext module
 *
 * C/C++ boundary design
 * ---------------------
 * BART headers use C99 extensions (_Complex float, VLAs, GNU __auto_type) that
 * are not portable C++.  lib_shim.c (compiled as C99) handles all interaction
 * with BART headers and exposes only void* data pointers and plain C structs
 * to this file.  Declarations here use void* for complex data pointers; on all
 * supported platforms void* and _Complex float* share the same size and calling
 * convention (ABI-compatible).
 *
 * Axis convention
 * ---------------
 * All tensor shapes crossing the Python boundary use bartorch C-order convention
 * (last index varies fastest, e.g. (nc, nz, ny, nx)).  The helpers
 * c_to_bart_dims() and bart_to_c_shape() perform the reversal to/from BART's
 * 16-element Fortran-order dim arrays.
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <memory>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

static constexpr int BART_DIMS = 16;
static constexpr int COIL_DIM  = 3;  /* matches BART's COIL_DIM from misc/mri.h */
static constexpr int MAPS_DIM  = 4;  /* matches BART's MAPS_DIM */
static constexpr int COEFF_DIM = 6;  /* matches BART's COEFF_DIM (subspace) */

// ---------------------------------------------------------------------------
// BART shim forward declarations
// ---------------------------------------------------------------------------
// All complex data pointers are declared void* here (ABI-compatible with
// _Complex float* on all platforms we support).
// ---------------------------------------------------------------------------
extern "C" {
    struct linop_s;  // opaque

    void bartorch_linop_get_domain_dims   (const struct linop_s *op, long dims[16]);
    void bartorch_linop_get_codomain_dims (const struct linop_s *op, long dims[16]);

    const struct linop_s *bartorch_create_encoding_op(
        const long img_dims[16],
        const long ksp_dims[16],
        const long map_dims[16], const void *maps,
        const long pat_dims[16], const void *pattern,
        const long traj_dims[16], const void *traj,
        const long bas_dims[16],  const void *basis,
        int use_cuda);

    void bartorch_linop_forward (const struct linop_s *op, void *dst, const void *src);
    void bartorch_linop_adjoint (const struct linop_s *op, void *dst, const void *src);
    void bartorch_linop_normal  (const struct linop_s *op, void *dst, const void *src);

    void bartorch_conjgrad_solve(
        const struct linop_s *op,
        int maxiter, float lambda, float tol,
        const long x_dims[16], void *x,
        const long y_dims[16], const void *y);

    void bartorch_linop_free(const struct linop_s *op);

    /// Invalidate CPU FINUFFT plan-cache entries whose traj_ptr matches ptr.
    /// Defined in finufft_grid.cpp; declared here so BartLinopHandle can call
    /// it at destruction time to prevent use-after-free of trajectory pointers.
    void bartorch_finufft_invalidate_traj(const void *ptr);

#ifdef BARTORCH_USE_CUFINUFFT
    /// Invalidate GPU cuFINUFFT plan-cache entries whose traj_ptr matches ptr.
    /// Defined in cuda/cufinufft_grid.cu.
    void bartorch_cufinufft_invalidate_traj(const void *ptr);
#endif
}
// misc/types.h (pulled in transitively by BART headers included by lib_shim.c)
// defines `#define auto __auto_type`.  Since this TU includes no BART headers
// we don't need the undef here, but add it defensively in case some torch
// header chain triggers it.
#undef auto

// ---------------------------------------------------------------------------
// Shape conversion helpers
// ---------------------------------------------------------------------------

// Convert a C-order Python shape to a BART Fortran-order dim array of length
// BART_DIMS, padding with 1s.
static void c_to_bart_dims(const std::vector<int64_t> &cshape,
                            long bart_dims[BART_DIMS])
{
    for (int i = 0; i < BART_DIMS; ++i) bart_dims[i] = 1;
    int n = (int)cshape.size();
    for (int j = 0; j < n; ++j)
        bart_dims[j] = (long)cshape[n - 1 - j];  // reverse: C-order → Fortran
}

// Convert a BART Fortran-order dim array to a trimmed C-order shape.
// Trailing size-1 dimensions are removed (but at least min_ndim dims kept).
static std::vector<int64_t> bart_to_c_shape(const long bart_dims[BART_DIMS],
                                             int min_ndim = 1)
{
    int ndim_ = BART_DIMS;
    while (ndim_ > min_ndim && bart_dims[ndim_ - 1] == 1) --ndim_;
    std::vector<int64_t> shape(ndim_);
    for (int j = 0; j < ndim_; ++j)
        shape[j] = (int64_t)bart_dims[ndim_ - 1 - j];
    return shape;
}

// Ensure a tensor is complex64, C-contiguous, and cloned (so BART cannot
// corrupt the original Python tensor via its internal pointer arithmetic).
static torch::Tensor to_bart_tensor(torch::Tensor t)
{
    return t.to(torch::kComplexFloat).contiguous().clone();
}

// ---------------------------------------------------------------------------
// BartLinopHandle
// ---------------------------------------------------------------------------

/**
 * BartLinopHandle — Python-visible owner of a BART linop_s*.
 *
 * Constructed by :func:`create_encoding_op`.  Holds the BART operator pointer
 * and keeps the input data tensors (maps, traj, basis, pattern) alive so that
 * BART's internal pointers into them remain valid.  The operator is freed
 * automatically when the Python object is garbage-collected.
 *
 * Thread safety: not thread-safe; use one handle per thread.
 */
class BartLinopHandle {
public:
    BartLinopHandle(const struct linop_s *op,
                    std::vector<torch::Tensor> kept_alive,
                    const void *traj_data_ptr = nullptr)
        : op_(op), kept_(std::move(kept_alive)), traj_data_ptr_(traj_data_ptr)
    {
        // Extract domain / codomain Fortran dims from BART and convert to
        // C-order shapes for Python consumption.
        long idims[BART_DIMS], odims[BART_DIMS];
        bartorch_linop_get_domain_dims  (op_, idims);
        bartorch_linop_get_codomain_dims(op_, odims);
        ishape_ = bart_to_c_shape(idims);
        oshape_ = bart_to_c_shape(odims);
    }

    ~BartLinopHandle()
    {
        // Invalidate FINUFFT plan-cache entries for this operator's trajectory
        // BEFORE freeing the linop and BEFORE the kept_ tensors are destroyed.
        // This prevents the ABA problem where a new allocation reuses the same
        // pointer address and incorrectly gets a cache hit with stale setpts.
        if (traj_data_ptr_) {
            bartorch_finufft_invalidate_traj(traj_data_ptr_);
#ifdef BARTORCH_USE_CUFINUFFT
            bartorch_cufinufft_invalidate_traj(traj_data_ptr_);
#endif
        }

        if (op_) {
            bartorch_linop_free(op_);
            op_ = nullptr;
        }
    }

    // Non-copyable, moveable
    BartLinopHandle(const BartLinopHandle &)            = delete;
    BartLinopHandle &operator=(const BartLinopHandle &) = delete;

    const std::vector<int64_t> &ishape() const { return ishape_; }
    const std::vector<int64_t> &oshape() const { return oshape_; }

    /**
     * Apply the operator to a tensor.
     *
     * mode  0 = forward  (domain → codomain,  A  x)
     *       1 = adjoint  (codomain → domain,  A^H y)
     *       2 = normal   (domain → domain,    A^H A x)
     */
    torch::Tensor apply(torch::Tensor x, int mode) const
    {
        if (!op_) throw std::runtime_error("BartLinopHandle: operator has been freed");

        x = x.to(torch::kComplexFloat).contiguous().clone();

        // Select output shape based on mode.
        const std::vector<int64_t> &out_shape = (mode == 0) ? oshape_ : ishape_;

        auto result = torch::zeros(
            torch::IntArrayRef(out_shape.data(), out_shape.size()),
            torch::TensorOptions().dtype(torch::kComplexFloat));

        switch (mode) {
        case 0:
            bartorch_linop_forward(op_, result.data_ptr(), x.data_ptr());
            break;
        case 1:
            bartorch_linop_adjoint(op_, result.data_ptr(), x.data_ptr());
            break;
        case 2:
            bartorch_linop_normal (op_, result.data_ptr(), x.data_ptr());
            break;
        default:
            throw std::invalid_argument("mode must be 0 (forward), 1 (adjoint), or 2 (normal)");
        }
        return result;
    }

    /**
     * CG solve: (A^H A + lambda*I) x = A^H y.
     *
     * The entire iteration runs in C via BART's lsqr + iter_conjgrad with no
     * Python callbacks in the inner loop.
     */
    torch::Tensor solve(torch::Tensor y, int maxiter, float lambda, float tol) const
    {
        if (!op_) throw std::runtime_error("BartLinopHandle: operator has been freed");

        y = y.to(torch::kComplexFloat).contiguous().clone();

        // Solution lives in the domain.
        auto x = torch::zeros(
            torch::IntArrayRef(ishape_.data(), ishape_.size()),
            torch::TensorOptions().dtype(torch::kComplexFloat));

        // Obtain the full 16-element BART Fortran dim arrays for the lsqr call.
        long x_dims[BART_DIMS], y_dims[BART_DIMS];
        bartorch_linop_get_domain_dims  (op_, x_dims);
        bartorch_linop_get_codomain_dims(op_, y_dims);

        bartorch_conjgrad_solve(
            op_, maxiter, lambda, tol,
            x_dims, x.data_ptr(),
            y_dims, y.data_ptr());

        return x;
    }

private:
    const struct linop_s       *op_;
    std::vector<torch::Tensor>  kept_;         // keep inputs alive for op_'s lifetime
    const void                 *traj_data_ptr_; // raw trajectory ptr for cache invalidation
    std::vector<int64_t>        ishape_; // domain shape in C-order
    std::vector<int64_t>        oshape_; // codomain shape in C-order
};

// ---------------------------------------------------------------------------
// Factory: create_encoding_op
// ---------------------------------------------------------------------------

/**
 * Create a persistent BART encoding operator for SENSE MRI reconstruction.
 *
 * Wraps BART's ``pics_model()`` which builds a Cartesian or non-Cartesian
 * SENSE forward model:
 *
 *   Cartesian (traj=None):      E = P ∘ FFT ∘ coil-expansion
 *   Non-Cartesian (traj≠None):  E = NUFFT ∘ coil-expansion
 *   With basis (basis≠None):    prepends subspace projection to the above
 *
 * All tensor shapes use bartorch **C-order** convention.
 *
 * Parameters
 * ----------
 * maps_t     Sensitivity maps ``(nmaps, nc, [nz,] ny, nx)``, complex64.
 * ksp_shape  K-space output shape in C-order.
 *            - Cartesian: ``(nc, [nz,] ny, nx)`` — matches maps but coil kept,
 *              maps dim set to 1.  Inferred from maps if empty.
 *            - Non-Cartesian: ``(nc, nspokes, nsamples, 1)`` or similar.
 * pattern_t  Undersampling mask in C-order, optional (None → full k-space).
 * traj_t     Non-Cartesian trajectory in C-order ``(..., 3)``, optional.
 * basis_t    Subspace basis in C-order ``(ncoeff, nt)``, optional.
 * use_cuda   1 to allocate on GPU (requires CUDA build), 0 for CPU.
 *            When 1 and traj is provided, also enables GPU gridding
 *            (conf.gpu_gridding=true) so the NUFFT runs on the GPU.
 */
static py::object create_encoding_op(
    torch::Tensor        maps_t,
    std::vector<int64_t> ksp_shape,
    py::object           pattern_py,
    py::object           traj_py,
    py::object           basis_py,
    int                  use_cuda)
{
    // ── Prepare maps ────────────────────────────────────────────────────────
    maps_t = to_bart_tensor(maps_t);

    long map_dims[BART_DIMS];
    c_to_bart_dims(maps_t.sizes().vec(), map_dims);

    // ── Compute img_dims from maps ──────────────────────────────────────────
    //
    // BART maps Fortran layout: [nx, ny, nz, nc, nmaps, 1, …]
    //   map_dims[COIL_DIM=3] = nc
    //   map_dims[MAPS_DIM=4] = nmaps
    //
    // img_dims = map_dims with COIL_DIM zeroed (image has no coil dimension).
    long img_dims[BART_DIMS];
    for (int i = 0; i < BART_DIMS; ++i) img_dims[i] = map_dims[i];
    img_dims[COIL_DIM] = 1;

    // ── Compute ksp_dims ───────────────────────────────────────────────────
    long ksp_dims[BART_DIMS];
    if (!ksp_shape.empty()) {
        c_to_bart_dims(ksp_shape, ksp_dims);
    } else {
        // Cartesian: infer from maps — same spatial/coil layout, MAPS_DIM→1.
        for (int i = 0; i < BART_DIMS; ++i) ksp_dims[i] = map_dims[i];
        ksp_dims[MAPS_DIM] = 1;
    }

    // ── Optional inputs ─────────────────────────────────────────────────────
    torch::Tensor pattern_t, traj_t, basis_t;
    long pat_dims [BART_DIMS] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    long traj_dims[BART_DIMS] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};
    long bas_dims [BART_DIMS] = {1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1};

    bool has_traj  = !traj_py.is_none();
    bool has_basis = !basis_py.is_none();

    if (!pattern_py.is_none()) {
        pattern_t = to_bart_tensor(pattern_py.cast<torch::Tensor>());
        c_to_bart_dims(pattern_t.sizes().vec(), pat_dims);
    } else {
        // BART's linop_sampling_create always dereferences pat_dims, so we
        // must pass a valid dim array and a non-NULL all-ones pattern even for
        // full k-space (no undersampling).  Build pat_dims = ksp_dims with the
        // coil dimension (index COIL_DIM=3) collapsed to 1, then create a
        // matching all-ones tensor that we keep alive in the handle.
        long pd[BART_DIMS];
        for (int i = 0; i < BART_DIMS; ++i) pd[i] = ksp_dims[i];
        pd[COIL_DIM] = 1;
        auto pat_cshape = bart_to_c_shape(pd);
        pattern_t = torch::ones(
            torch::IntArrayRef(pat_cshape.data(), pat_cshape.size()),
            torch::TensorOptions().dtype(torch::kComplexFloat)).clone();
        c_to_bart_dims(pattern_t.sizes().vec(), pat_dims);
    }
    if (has_traj) {
        traj_t = to_bart_tensor(traj_py.cast<torch::Tensor>());
        c_to_bart_dims(traj_t.sizes().vec(), traj_dims);
    }
    if (has_basis) {
        basis_t = to_bart_tensor(basis_py.cast<torch::Tensor>());
        c_to_bart_dims(basis_t.sizes().vec(), bas_dims);
        // BART expects the basis in Fortran layout (1,…,1,nt,ncoeff,1,…) with
        // nt at index 5 (TE_DIM) and ncoeff at index 6 (COEFF_DIM).
        // C-order input (ncoeff, nt) maps directly: bshape[0]=ncoeff, bshape[1]=nt.
        auto bshape = basis_t.sizes().vec();
        if (bshape.size() >= 2) {
            for (int i = 0; i < BART_DIMS; ++i) bas_dims[i] = 1;
            bas_dims[5] = (long)bshape[1];  // TE_DIM=5:    nt
            bas_dims[6] = (long)bshape[0];  // COEFF_DIM=6: ncoeff
            img_dims[6] = (long)bshape[0];  // image gains COEFF_DIM
        }
    }

    // ── Call the C shim ─────────────────────────────────────────────────────
    const struct linop_s *op = bartorch_create_encoding_op(
        img_dims, ksp_dims,
        map_dims, maps_t.data_ptr(),
        pat_dims, pattern_t.data_ptr(),   // always non-NULL (all-ones or user mask)
        traj_dims, has_traj  ? traj_t.data_ptr()  : nullptr,
        bas_dims,  has_basis ? basis_t.data_ptr() : nullptr,
        use_cuda);

    if (!op)
        throw std::runtime_error(
            "bartorch: pics_model returned NULL — check dimension compatibility");

    // Keep tensors alive for the handle's lifetime (BART may keep raw pointers
    // into them for internal computations between apply calls).
    std::vector<torch::Tensor> keep;
    keep.push_back(maps_t);
    keep.push_back(pattern_t);   // always kept (all-ones or user-supplied)
    if (has_traj)  keep.push_back(traj_t);
    if (has_basis) keep.push_back(basis_t);

    // Record the raw trajectory pointer for FINUFFT plan-cache invalidation.
    const void *traj_ptr = has_traj ? traj_t.data_ptr() : nullptr;

    return py::cast(std::make_shared<BartLinopHandle>(op, std::move(keep), traj_ptr));
}

// ---------------------------------------------------------------------------
// Module registration
// ---------------------------------------------------------------------------

void init_lib_ops(py::module_ &m)
{
    py::class_<BartLinopHandle, std::shared_ptr<BartLinopHandle>>(
        m, "BartLinopHandle",
        "Opaque owner of a BART linop_s* linear operator.\n\n"
        "Created by :func:`create_encoding_op`.  Auto-freed when the Python "
        "object is garbage-collected.  Apply via :meth:`apply`; solve via "
        ":meth:`solve`.")
        .def_property_readonly("ishape", &BartLinopHandle::ishape,
            "Domain (input) shape in C-order.")
        .def_property_readonly("oshape", &BartLinopHandle::oshape,
            "Codomain (output) shape in C-order.")
        .def("apply", &BartLinopHandle::apply,
            py::arg("x"), py::arg("mode"),
            "Apply operator: mode=0 forward (A x), 1 adjoint (A^H y), 2 normal (A^H A x).")
        .def("solve", &BartLinopHandle::solve,
            py::arg("y"),
            py::arg("maxiter") = 30,
            py::arg("lambda_")  = 0.0f,
            py::arg("tol")     = 1e-6f,
            "CG solve (A^H A + lambda*I) x = A^H y entirely in C.");

    m.def("create_encoding_op",
        &create_encoding_op,
        py::arg("maps"),
        py::arg("ksp_shape")  = std::vector<int64_t>{},
        py::arg("pattern")    = py::none(),
        py::arg("traj")       = py::none(),
        py::arg("basis")      = py::none(),
        py::arg("use_cuda")   = 0,
        "Create a persistent BART SENSE encoding operator (Cartesian or non-Cartesian).\n\n"
        "Parameters\n----------\n"
        "maps : torch.Tensor\n"
        "    Sensitivity maps, C-order ``(nmaps, nc, [nz,] ny, nx)``, complex64.\n"
        "ksp_shape : list[int], optional\n"
        "    K-space shape in C-order.  Inferred from maps for Cartesian (default).\n"
        "pattern : torch.Tensor, optional\n"
        "    Undersampling mask (Cartesian).\n"
        "traj : torch.Tensor, optional\n"
        "    Non-Cartesian trajectory, C-order ``(..., 3)``.\n"
        "basis : torch.Tensor, optional\n"
        "    Subspace basis, C-order ``(ncoeff, nt)``.\n"
        "use_cuda : int\n"
        "    1 to use GPU (requires CUDA build), 0 for CPU (default).\n"
        "    When 1 and traj is provided, also enables GPU gridding.\n");
}
