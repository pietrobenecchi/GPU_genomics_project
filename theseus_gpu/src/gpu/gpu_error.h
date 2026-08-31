#pragma once

/**
 * @file gpu_error.h
 * @brief The backend's single error slot, shared by every GPU translation unit.
 *
 * One buffer per process, written by whichever CUDA call failed last and read
 * back through align_gpu.h's last_error(). Internal to src/gpu: the boundary
 * the rest of the project sees stays POD-only in align_gpu.h.
 */

#include <cuda_runtime.h>

namespace theseus {
namespace gpu {

/** @brief Record @p err against @p context, overwriting the previous message. */
void set_error(const char *context, cudaError_t err);

/** @brief Empty the slot, so an entry point that succeeds reports no error. */
void clear_error();

}  // namespace gpu
}  // namespace theseus
