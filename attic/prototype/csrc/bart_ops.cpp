/*
 * bart_ops.cpp — Direct bindings for the most-used BART library functions.
 *
 * Each function in this file:
 *   1. Converts Python/torch arguments into a BART argv[] array.
 *   2. Registers input BartTensors via register_mem_cfl_non_managed().
 *   3. Pre-allocates output BartTensors with Fortran strides.
 *   4. Calls bart_command() in-process.
 *   5. Unlinks all *.mem names and returns the output tensor(s).
 *
 * Functions here mirror the SWIG-based API from the old bartpy, covering:
 *   - FFT / IFFT (num/fft)
 *   - Phantom generation (simu/phantom)
 *   - PICS reconstruction (src/pics)
 *   - ESPIRiT calibration (src/ecalib)
 *   - Linear operators (linops/)
 *   - Iterative algorithms (iter/)
 *
 * The implementation is currently a stub; see agents.md for the roadmap.
 */

#include <torch/extension.h>
#include <string>
#include <vector>

extern "C" {
#include "bart_embed_api.h"
}

#include "tensor_bridge.hpp"

namespace bartorch {
namespace ops {

// ---------------------------------------------------------------------------
// Stub: will be filled in as part of the implementation phase.
// ---------------------------------------------------------------------------

// Example signature (not yet wired up):
//
// torch::Tensor fft(const torch::Tensor& input, long flags,
//                   bool unitary, bool inverse, bool centered);

} // namespace ops
} // namespace bartorch
