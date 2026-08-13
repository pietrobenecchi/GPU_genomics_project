/**
 * @file align_gpu_stub.cpp
 * @brief Backend used when the project is built without CUDA, so that
 * theseus_proxy links and runs identically on machines without nvcc.
 */

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

// No device, so nothing to page-lock for: plain host memory has the same
// contract and the caller cannot tell the difference.
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
