// Storage dello slot d'errore dichiarato in gpu_error.h.
// Il buffer e' file-local: lo toccano solo set_error, clear_error e
// last_error, che stanno tutte qui.

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
