/**
 * @file gpu_error.cu
 * @brief Storage for the error slot declared in gpu_error.h.
 *
 * The buffer is file-local: set_error, clear_error and last_error are the only
 * three things that touch it, and they all live here.
 */

#include "gpu/gpu_error.h"
#include "gpu/align_gpu.h"

#include <cstdio>

namespace theseus {
namespace gpu {

namespace {
char g_last_error[256] = {0};
}  // namespace

void set_error(const char *context, cudaError_t err) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s: %s", context,
                  cudaGetErrorString(err));
}

void clear_error() { g_last_error[0] = '\0'; }

const char *last_error() { return g_last_error; }

}  // namespace gpu
}  // namespace theseus
