/**
 * @file align_gpu_stub.cpp
 * @brief Backend used when the project is built without CUDA, so that
 * theseus_proxy links and runs identically on machines without nvcc.
 */

#include "gpu/align_gpu.h"

namespace theseus {
namespace gpu {

const char *last_error() { return ""; }

const TimingReport &last_timing() {
    static TimingReport timing;
    return timing;
}

Status align_batch(const BatchView &, const DeviceGraph *, const int32_t *,
                   const int32_t *, AlignScoring, AlignOptions, AlignResult *, void *, int32_t *) {
    return Status::NotCompiled;
}

DeviceGraph *upload_graph(const GraphCsrView &) { return nullptr; }

void free_graph(DeviceGraph *) {}

Status readback_graph(const DeviceGraph *, char *, int32_t *, int32_t *, int32_t *,
                      int32_t *) {
    return Status::NotCompiled;
}

}  // namespace gpu
}  // namespace theseus
