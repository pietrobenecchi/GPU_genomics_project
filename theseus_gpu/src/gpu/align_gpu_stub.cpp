// Backend usato quando si compila senza CUDA, cosi' theseus_proxy linka e gira
// uguale anche su macchine senza nvcc (con fallback CPU).

#include "gpu/align_gpu.h"

#include <cstdlib>

namespace theseus {
namespace gpu {

const char *last_error() { return ""; }

const TimingReport &last_timing() {
    static TimingReport timing;
    return timing;
}

Status align_batch(const BatchView &, const DeviceGraph *, DeviceWorkspace *, const int32_t *,
                   const int32_t *, AlignScoring, AlignOptions, AlignResult *, void *, int32_t *) {
    return Status::NotCompiled;
}

DeviceWorkspace *create_workspace() { return nullptr; }
void free_workspace(DeviceWorkspace *) {}

// Niente device, niente da page-lockare: la memoria host normale ha lo stesso
// contratto e il chiamante non vede differenza.
void *alloc_host_pinned(size_t bytes) { return std::malloc(bytes); }
void free_host_pinned(void *buffer) { std::free(buffer); }

DeviceGraph *upload_graph(const GraphCsrView &) { return nullptr; }

void free_graph(DeviceGraph *) {}

Status readback_graph(const DeviceGraph *, char *, int32_t *, int32_t *, int32_t *,
                      int32_t *) {
    return Status::NotCompiled;
}

}  // namespace gpu
}  // namespace theseus
