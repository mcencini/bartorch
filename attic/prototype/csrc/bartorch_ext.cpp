/*
 * bartorch_ext.cpp — PyTorch C++ extension entry point.
 *
 * Implements the generic ``run()`` dispatcher which:
 *
 *  1. Registers input tensor data_ptr()s in BART's in-memory CFL registry
 *     via ``memcfl_register(…, managed=false)`` under ``_bt_inN.mem`` names.
 *     Input tensors are C-order; their shape is reversed to BART Fortran order
 *     at registration time (no data copy needed — identical byte layout).
 *
 *  2. Calls ``bart_command()`` with a fully-formed argv.
 *
 *  3. Retrieves the output from the in-memory CFL registry
 *     (``_bt_out0.mem``), copies it to a new contiguous C-order
 *     ``complex64`` tensor, reverses the BART Fortran dims to bartorch
 *     C-order, and returns it.
 *
 *  4. Cleans up all ``*.mem`` registrations.
 *
 * The ``*.mem`` name suffix causes BART's ``io.c`` to route all
 * ``create_cfl`` / ``load_cfl`` calls through the in-memory CFL registry
 * instead of touching disk.
 *
 * No ``bart_embed_api.h`` — we call BART's own internal functions directly
 * (``misc/memcfl.h``, ``misc/mmio.h``, ``misc/debug.h``).
 */

#include <torch/extension.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

// Python.h needed for PyErr_Occurred/Fetch/Clear used in run() to drain
// any exception BART's BART_WITH_PYTHON error handler may have set.
#include <Python.h>

#include <algorithm>
#include <atomic>
#include <cstring>
#include <regex>
#include <sstream>
#include <string>
#include <vector>

// Forward-declare only the BART functions we need, using C++-safe signatures.
// BART's own headers (misc/memcfl.h, misc/debug.h) use C99 VLAs and
// 'complex float*' types which are not valid C++.
// On all supported platforms void* and _Complex float* share the same size
// and calling convention, so these redeclarations are ABI-compatible.
extern "C" {
    void  memcfl_register(const char* name, int D, const long* dims,
                          void* data, int managed);
    bool  memcfl_exists(const char* name);
    void* memcfl_load(const char* name, int D, long* dims);
    bool  memcfl_unmap(const void* p);
    void  memcfl_unlink(const char* name);

    extern int debug_level;
    // Minimal debug-level constants matching BART's enum debug_levels.
    enum { BARTORCH_DP_ERROR = 0, BARTORCH_DP_WARN = 1, BARTORCH_DP_INFO = 2 };

    // bart_command is implemented in bart.c and linked via bart_static.
    int bart_command(int len, char* out, int argc, char* argv[]);
}
// misc/types.h (pulled in transitively) defines `#define auto __auto_type`
// which breaks C++.  Undo it immediately.
#undef auto

// Torch-prior registry functions — implemented in torch_prior.cpp.
void bartorch_register_torch_prior(const std::string& name,
                                   pybind11::object   fn,
                                   std::vector<long>  dims);
void bartorch_unregister_torch_prior(const std::string& name);

// lib_ops.cpp registration function.
void init_lib_ops(py::module_& m);

// FINUFFT plan-cache flush — implemented in finufft_grid.cpp and
// cuda/cufinufft_grid.cu.  Called after each bart_command() run to prevent
// ABA reuse of CLI-allocated CFL trajectory pointers across commands.
#ifdef BARTORCH_USE_FINUFFT
extern "C" void bartorch_finufft_flush_all();
#endif
#ifdef BARTORCH_USE_CUFINUFFT
extern "C" void bartorch_cufinufft_flush_all();
#endif

#ifdef USE_CUDA
#include <cuda_runtime.h>
#include "cuda/cuda_bridge.hpp"

// GPU dispatch mode functions — implemented in cuda/bartorch_cuda_dispatch.cpp.
// When active, __wrap_memcfl_create allocates BART's output CFL buffer on the
// GPU via cudaMalloc instead of xmalloc, enabling full GPU zero-copy.
void bartorch_enable_gpu_dispatch(int device_id);
void bartorch_disable_gpu_dispatch();
bool bartorch_is_gpu_dispatch_active();
int  bartorch_gpu_dispatch_device();

// BART GPU initialisation — from num/init.h / num/gpuops.h.
extern "C" {
    extern bool bart_use_gpu;
    void num_init_gpu_support(void);
}
#endif

namespace py = pybind11;

static constexpr int BART_DIMS = 16;

// ---------------------------------------------------------------------------
// Argv construction helpers
// ---------------------------------------------------------------------------

/**
 * Stringify a Python scalar / tuple value for a BART argv entry.
 *
 * Tuples are formatted as ``a:b:c`` (used by VEC2/VEC3 options).
 */
static std::string to_bart_str(py::object val)
{
    std::ostringstream oss;
    if (py::isinstance<py::bool_>(val)) {
        // Bare bool – caller should have skipped True-only flags already.
        oss << (val.cast<bool>() ? "1" : "0");
    } else if (py::isinstance<py::int_>(val)) {
        oss << val.cast<long long>();
    } else if (py::isinstance<py::float_>(val)) {
        oss << val.cast<double>();
    } else if (py::isinstance<py::str>(val)) {
        oss << val.cast<std::string>();
    } else if (py::isinstance<py::tuple>(val)) {
        py::tuple t = val.cast<py::tuple>();
        for (size_t i = 0; i < t.size(); ++i) {
            if (i > 0) oss << ":";
            py::object elem = py::reinterpret_borrow<py::object>(t[i]);
            if (py::isinstance<py::int_>(elem))      oss << elem.cast<long long>();
            else if (py::isinstance<py::float_>(elem)) oss << elem.cast<double>();
            else oss << py::str(elem).cast<std::string>();
        }
    } else {
        oss << py::str(val).cast<std::string>();
    }
    return oss.str();
}

/**
 * Build the BART argv vector from all dispatch arguments.
 *
 * Layout: [op_name] [flags…] [positional_scalars…] [input_names…] [output_name]
 *
 * kwargs protocol
 * ---------------
 * - None / False  → skipped
 * - Single-char key ``x``:
 *     - True     → ``-x``  (bare flag)
 *     - value    → ``-x value``
 * - Key ``flag_N`` (synthetic numeric flag, e.g. ``flag_3`` → ``-3``):
 *     - True     → ``-3``
 *     - value    → ``-3 value``  (rare)
 * - Key ending with ``_<digits>`` (list-expanded flag, e.g. ``R_0``):
 *     strip suffix → apply single/long-flag rule on base key
 * - Multi-char key ``long_name``:
 *     replace ``_`` with ``-`` → ``--long-name``
 *     - True     → ``--long-name``
 *     - value    → ``--long-name value``
 *
 * positional_args
 * ---------------
 * Plain scalar values (e.g. bitmask for fft) inserted between flags and
 * input file names.
 */
static std::vector<std::string> build_argv(
    const std::string&              op_name,
    const std::vector<std::string>& input_names,
    const std::string&              output_name,
    const py::list&                 positional_args,
    const py::dict&                 kwargs)
{
    std::vector<std::string> parts;
    // bart_command() passes argv to main_bart(), which calls basename(argv[0])
    // and expects either "bart" (then reads the tool from argv[1]) or the tool
    // name directly.  When argv[0] == "bart" the standard dispatch path is
    // triggered, which calls parse_bart_opts() and initialises the order[]
    // array before handing off to the tool.  Without this the array is
    // uninitialised and loop_step() may segfault.
    parts.push_back("bart");
    parts.push_back(op_name);

    // 1. Flags from kwargs -----------------------------------------------
    for (auto item : kwargs) {
        std::string key = item.first.cast<std::string>();
        py::object  val = py::reinterpret_borrow<py::object>(item.second);

        // Skip None and False
        if (val.is_none()) continue;
        if (py::isinstance<py::bool_>(val) && !val.cast<bool>()) continue;

        // Detect list-expanded suffix: R_0, R_1, …  →  base = "R"
        std::string base = key;
        {
            auto us = key.rfind('_');
            if (us != std::string::npos && us + 1 < key.size()) {
                std::string suffix = key.substr(us + 1);
                bool all_digits = !suffix.empty();
                for (char c : suffix) if (!isdigit((unsigned char)c)) { all_digits = false; break; }
                if (all_digits) base = key.substr(0, us);
            }
        }

        // Determine the flag string
        std::string flag;
        if (base.size() == 1) {
            flag = "-" + base;
        } else if (base.size() > 5 && base.substr(0, 5) == "flag_") {
            // flag_3 → -3  (numeric flag characters)
            flag = "-" + base.substr(5);
        } else {
            // Long flag: underscores → hyphens
            std::string ln = base;
            std::replace(ln.begin(), ln.end(), '_', '-');
            flag = "--" + ln;
        }
        parts.push_back(flag);

        // Add value if not a bare True flag
        if (!(py::isinstance<py::bool_>(val) && val.cast<bool>())) {
            parts.push_back(to_bart_str(val));
        }
    }

    // 2. Positional scalar args ------------------------------------------
    for (auto item : positional_args) {
        py::object val = py::reinterpret_borrow<py::object>(item);
        if (val.is_none()) continue;
        parts.push_back(to_bart_str(val));
    }

    // 3. Input CFL names -------------------------------------------------
    for (const auto& n : input_names) parts.push_back(n);

    // 4. Output CFL name (omitted for scalar-output tools) ---------------
    if (!output_name.empty())
        parts.push_back(output_name);

    return parts;
}

// ---------------------------------------------------------------------------
// Generic hot-path dispatcher
// ---------------------------------------------------------------------------

/**
 * run() — in-process BART command execution with zero-copy input registration.
 *
 * Parameters
 * ----------
 * op_name         BART tool name (e.g. "fft", "phantom")
 * py_inputs       list of complex64 torch.Tensor (C-order, one per input CFL)
 * output_dims_py  expected output shape hint (currently unused; BART infers it)
 * positional_args plain scalar positional argv entries (e.g. bitmask for fft)
 * kwargs          flag options: single-char → -f, multi-char → --long-flag
 *
 * Returns
 * -------
 * torch.Tensor  C-order complex64 result tensor
 *
 * Raises
 * ------
 * RuntimeError  if bart_command() returns non-zero
 */
static py::object run(
    const std::string& op_name,
    py::list           py_inputs,
    py::object         output_dims_py,
    py::list           positional_args,
    py::dict           kwargs)
{
    // Suppress BART's informational output during normal calls.
    // BARTORCH_DP_WARN (1) lets only errors and warnings through.
    int saved_debug = debug_level;
    if (debug_level < 0 || debug_level > BARTORCH_DP_WARN)
        debug_level = BARTORCH_DP_WARN;

    // ── 1. Register input tensors ──────────────────────────────────────────
    // Use a per-call counter in names so stale entries from a previous error
    // do not collide with the current call.
    static std::atomic<uint64_t> s_call_id{0};
    uint64_t call_id = s_call_id.fetch_add(1, std::memory_order_relaxed);

    // ── Detect whether all inputs are CUDA tensors (GPU zero-copy path) ────
    // GPU zero-copy: when every input lives on the same CUDA device we
    //   (a) register device pointers directly (no host-side clone),
    //   (b) enable GPU dispatch so __wrap_memcfl_create allocates the output
    //       CFL buffer via cudaMalloc instead of xmalloc, and
    //   (c) initialise BART's GPU subsystem so it routes ops to CUDA kernels.
    // If ANY input is CPU (or inputs are mixed-device), fall through to the
    // existing CPU path.
    bool   all_cuda    = false;
    int    cuda_device = -1;
#ifdef USE_CUDA
    if (!py_inputs.empty()) {
        all_cuda = true;
        for (size_t i = 0; i < (size_t)py_inputs.size(); ++i) {
            torch::Tensor ti = py_inputs[i].cast<torch::Tensor>();
            if (!ti.is_cuda()) { all_cuda = false; break; }
            int dev = ti.device().index();
            if (cuda_device < 0) {
                cuda_device = dev;
            } else if (cuda_device != dev) {
                all_cuda = false;  // mixed CUDA devices — use CPU path
                break;
            }
        }
    }
    if (all_cuda) {
        bartorch_enable_gpu_dispatch(cuda_device);
        // Initialise BART's CUDA subsystem for this device (idempotent after
        // first call per device).  num_init_gpu_support() sets bart_use_gpu=true
        // and calls cuda_init(), which is required for BART's cuFFT/cuBLAS code
        // paths to activate when it sees GPU pointers.
        bart_use_gpu = true;
        num_init_gpu_support();
    }
#endif

    std::vector<std::string>   input_names;
    std::vector<torch::Tensor> inputs;   // keep alive until after bart_command

    for (size_t i = 0; i < (size_t)py_inputs.size(); ++i) {
        torch::Tensor t = py_inputs[i].cast<torch::Tensor>();
        // GPU path: keep the tensor on its device (zero-copy input).
        // CPU path: ensure complex64 and C-contiguous, then always clone.
        // Cloning is required for CPU because some BART commands (e.g. pocsense)
        // modify their input buffers in-place.  For GPU tensors the same issue
        // applies, but BART's GPU kernels typically work out-of-place, and any
        // in-place modification would only affect our device-side clone anyway.
        if (all_cuda) {
            t = t.to(torch::kComplexFloat).contiguous();  // stays on GPU, no clone
        } else {
            t = t.to(torch::kComplexFloat).contiguous().clone();
        }

        inputs.push_back(t);

        std::string name = "_bt_" + std::to_string(call_id) +
                           "_in" + std::to_string(i) + ".mem";
        input_names.push_back(name);

        // A C-order contiguous tensor with Python shape [d_{N-1},…,d_0] has
        // IDENTICAL byte layout to a Fortran-order array with BART dims
        // [d_0,…,d_{N-1}].  So we reverse the shape for registration.
        auto shape = t.sizes().vec();
        int  ndim  = (int)shape.size();

        long bart_dims[BART_DIMS];
        for (int j = 0; j < BART_DIMS; ++j) bart_dims[j] = 1;
        for (int j = 0; j < ndim; ++j)
            bart_dims[j] = shape[ndim - 1 - j];  // reverse

        // managed=false → BART never frees our tensor's data pointer.
        memcfl_register(name.c_str(), BART_DIMS, bart_dims,
                        t.data_ptr(), /*managed=*/0);
    }

    // When output_dims_py is exactly Python False the tool produces a scalar
    // (or no) output and does NOT expect a CFL output file argument.
    // For all other values (None, a list, an int) the tool writes a CFL.
    bool want_cfl_output = !(
        py::isinstance<py::bool_>(output_dims_py) &&
        !output_dims_py.cast<bool>()
    );

    const std::string output_name = want_cfl_output
        ? "_bt_" + std::to_string(call_id) + "_out0.mem"
        : "";

    // ── 2. Build argv and call BART ────────────────────────────────────────
    auto parts = build_argv(op_name, input_names, output_name,
                            positional_args, kwargs);

    std::vector<char*> argv;
    argv.reserve(parts.size() + 1);
    for (auto& s : parts) argv.push_back(const_cast<char*>(s.c_str()));
    argv.push_back(nullptr);
    int argc = (int)parts.size();

    char err_buf[512] = {'\0'};
    // Clear any pending Python exception before calling bart_command.
    // With BART_WITH_PYTHON, BART's error() calls PyErr_SetString, which sets
    // a Python exception indicator.  pybind11 checks PyErr_Occurred() on
    // return from every bound function; a stale pending exception causes it to
    // re-raise instead of returning normally.  Clearing here ensures a clean
    // state before the BART call.
    if (PyErr_Occurred()) PyErr_Clear();
    int  ret          = bart_command(sizeof(err_buf), err_buf, argc, argv.data());
    // Capture the BART-set Python exception (if any) and clear it before
    // pybind11 sees it.  We propagate BART errors via ret != 0 / RuntimeError.
    std::string bart_py_err;
    if (PyErr_Occurred()) {
        PyObject *ptype = nullptr, *pvalue = nullptr, *ptb = nullptr;
        PyErr_Fetch(&ptype, &pvalue, &ptb);
        PyErr_NormalizeException(&ptype, &pvalue, &ptb);
        if (pvalue) {
            PyObject* str = PyObject_Str(pvalue);
            if (str) { bart_py_err = PyUnicode_AsUTF8(str); Py_DECREF(str); }
        }
        Py_XDECREF(ptype); Py_XDECREF(pvalue); Py_XDECREF(ptb);
    }

    debug_level = saved_debug;

    // ── 3. Disable GPU dispatch mode (reset before any early return) ───────
#ifdef USE_CUDA
    if (all_cuda)
        bartorch_disable_gpu_dispatch();
#endif

    // ── 3b. Flush NUFFT plan caches for CLI commands ───────────────────────
    // Plans are keyed by trajectory device pointer.  For CLI commands, the
    // trajectory lives in a temporary CFL buffer that will be freed in step 4.
    // Flushing now prevents ABA pointer reuse on the next bart_command() call.
    // (For encoding_op / BartLinopHandle, the trajectory lives longer and is
    // invalidated per-traj in BartLinopHandle::~BartLinopHandle() instead.)
#ifdef BARTORCH_USE_FINUFFT
    bartorch_finufft_flush_all();
#endif
#ifdef BARTORCH_USE_CUFINUFFT
    bartorch_cufinufft_flush_all();
#endif

    // ── 4. Clean up input registrations ───────────────────────────────────
    // After a successful bart_command() call:
    //   - Each input was loaded by the tool (refcount: 1 → 2) then unmapped
    //     (refcount: 2 → 1) by the BART tool code.
    //   - We call memcfl_unmap once (refcount: 1 → 0) before memcfl_unlink.
    //
    // After a failed bart_command() call:
    //   - The refcount may be 1 (tool never loaded the CFL) or 2 (loaded but
    //     longjmp was called before unmap_cfl).
    //   - We skip cleanup on error to avoid a cascade error from memcfl_unlink.
    //     The unique call_id in names ensures stale entries never collide.
    if (ret == 0) {
        for (size_t i = 0; i < inputs.size(); ++i) {
            memcfl_unmap(inputs[i].data_ptr());
            memcfl_unlink(input_names[i].c_str());
        }
    }
    // Note: on ret != 0, input entries are intentionally left in the registry
    // with their call-id-scoped names and will not interfere with future calls.

    // ── 5. Handle BART error ───────────────────────────────────────────────
    if (ret != 0) {
        if (memcfl_exists(output_name.c_str()))
            memcfl_unlink(output_name.c_str());
        std::string msg = "BART '" + op_name + "' failed (code " +
                          std::to_string(ret) + ")";
        if (err_buf[0]) msg += ": " + std::string(err_buf);
        else if (!bart_py_err.empty()) msg += ": " + bart_py_err;
        throw std::runtime_error(msg);
    }

    // ── 6. Retrieve scalar text output (nrmse, bitmask, …) ────────────────
    // For scalar-output tools (!want_cfl_output) we never registered an output
    // CFL, so skip the memcfl_exists check entirely.
    if (!want_cfl_output || !memcfl_exists(output_name.c_str())) {
        if (err_buf[0]) return py::str(std::string(err_buf));
        return py::none();
    }

    // ── 7. Retrieve CFL output ────────────────────────────────────────────
    long bart_dims_out[BART_DIMS] = {0};
    // memcfl_load increments refcount (0 → 1 after tool's unmap_cfl already
    // brought it to 0).
    void* ptr = memcfl_load(output_name.c_str(), BART_DIMS, bart_dims_out);
    TORCH_CHECK(ptr != nullptr,
                "bartorch: memcfl_load returned NULL for output '", output_name, "'");

    // Find effective ndim: trim trailing size-1 dims.
    int min_ndim_out = 1;
    if (!py::isinstance<py::bool_>(output_dims_py) && !output_dims_py.is_none()) {
        try {
            int hint = (int)py::len(output_dims_py);
            if (hint > min_ndim_out) min_ndim_out = hint;
        } catch (...) {}
    }

    int ndim_out = BART_DIMS;
    while (ndim_out > min_ndim_out && bart_dims_out[ndim_out - 1] == 1) --ndim_out;

    // Reverse BART Fortran dims → bartorch C-order shape.
    std::vector<int64_t> shape_out(ndim_out);
    for (int j = 0; j < ndim_out; ++j)
        shape_out[j] = bart_dims_out[ndim_out - 1 - j];

    // Total output byte count (all BART_DIMS, including padding 1s).
    size_t byte_count = 1;
    for (int j = 0; j < BART_DIMS; ++j) byte_count *= (size_t)bart_dims_out[j];
    byte_count *= sizeof(c10::complex<float>);

#ifdef USE_CUDA
    // ── 7a. GPU output path ──────────────────────────────────────────────
    // ptr is a cudaMalloc'd device buffer (from __wrap_memcfl_create).
    // Create the result tensor on the same CUDA device and do a D2D copy.
    // The D2D copy is much cheaper than a D2H+H2D round-trip and avoids the
    // shared-malloc aliasing bug that affects the CPU path (different allocators).
    if (all_cuda) {
        auto result = torch::empty(
            shape_out,
            torch::TensorOptions()
                .dtype(torch::kComplexFloat)
                .device(torch::kCUDA, cuda_device));

        cudaMemcpy(result.data_ptr(), ptr,
                   (size_t)result.numel() * sizeof(c10::complex<float>),
                   cudaMemcpyDeviceToDevice);

        memcfl_unmap(ptr);               // refcount 1 → 0
        memcfl_unlink(output_name.c_str()); // removes entry; managed=false, no xfree
        cudaFree(ptr);                   // free the cudaMalloc'd buffer

        // Synchronise so the caller sees fully written GPU data.
        cudaDeviceSynchronize();
        return py::cast(result);
    }
#endif

    // ── 7b. CPU output path ───────────────────────────────────────────────
    // Copy the BART output into a temporary buffer BEFORE calling torch::empty.
    //
    // Root cause of the "all-zeros on 2nd call" bug:
    //   BART and PyTorch share the same malloc pool (libtorch_cpu.so exports
    //   malloc/free and our extension links against it).  After BART's
    //   memcfl_unlink frees the output buffer, torch::empty can receive that
    //   same address from the shared allocator.  The subsequent memcpy would
    //   be a no-op (src == dst), and then memcfl_unlink would free the block
    //   that the result tensor still points to — leaving the tensor with freed
    //   (potentially zeroed by glibc) memory.
    //
    //   Fix: stash the raw bytes in a std::vector before releasing BART's
    //   buffer.  torch::empty is called only after we hold the data safely in
    //   the vector.  The vector uses its own (stack/heap) storage that the
    //   BART malloc pool cannot alias.
    std::vector<char> tmp(byte_count);
    std::memcpy(tmp.data(), ptr, byte_count);

    // Release BART's ownership of the output CFL now that data is in tmp.
    memcfl_unmap(ptr);               // refcount 1 → 0
    memcfl_unlink(output_name.c_str()); // free (managed=true, malloc'd by BART)

    // Allocate output tensor and copy from temporary buffer.
    auto result = torch::empty(shape_out,
                               torch::TensorOptions().dtype(torch::kComplexFloat));
    std::memcpy(result.data_ptr(), tmp.data(),
                (size_t)result.numel() * sizeof(c10::complex<float>));

    return py::cast(result);
}

// ---------------------------------------------------------------------------
// Module definition
// ---------------------------------------------------------------------------

PYBIND11_MODULE(_bartorch_ext, m) {
    m.doc() = "bartorch C++ extension — zero-copy PyTorch↔BART bridge";

    m.def("run", &run,
          py::arg("op_name"),
          py::arg("inputs"),
          py::arg("output_dims"),
          py::arg("positional_args"),
          py::arg("kwargs"),
          "Generic in-process BART command dispatcher.\n\n"
          "Registers input tensors in BART's in-memory CFL registry, calls\n"
          "bart_command(), retrieves and returns the output tensor.");

    m.def("register_torch_prior",
          &bartorch_register_torch_prior,
          py::arg("name"),
          py::arg("fn"),
          py::arg("dims"),
          "Register a Python denoiser callable in the global torch-prior "
          "registry so that __wrap_nlop_tf_create can find it when BART "
          "processes -R TF:{bartorch://<name>}:<lambda>.");

    m.def("unregister_torch_prior",
          &bartorch_unregister_torch_prior,
          py::arg("name"),
          "Remove a previously registered torch prior from the registry.");

    // Library operator bindings (BartLinopHandle, create_encoding_op, …)
    init_lib_ops(m);
}
