#pragma once

/**
 * @file device_memory.h
 * @brief The two long-lived device allocations, and the calls that manage them.
 *
 * Both are opaque in align_gpu.h -- the project outside src/gpu only ever holds
 * pointers to them -- and defined here so that the batch orchestration can see
 * their members without also seeing the kernels.
 *
 * The workspace is grow-only and lives for the process: every batch, and every
 * chunk of a batch, reuses the same buffers, so a steady workload allocates
 * once and then never calls cudaMalloc again.
 */

#include "gpu/align_gpu.h"
#include "gpu/kernel_launch.h"
#include "query_state.h"

#include <cstddef>

namespace theseus {
namespace gpu {

/**
 * @brief The graph in device memory, plus the sizes needed to read it back.
 */
struct DeviceGraph {
    GraphCsrView view;  // Pointers below, as seen by kernels
    char *vertex_chars = nullptr;
    int32_t *vertex_offsets = nullptr;
    int32_t *edge_targets = nullptr;
    int32_t *edge_overlaps = nullptr;
    int32_t *edge_offsets = nullptr;
    int32_t num_chars = 0;
    int32_t num_edges = 0;
};

struct DeviceWorkspace {
    char *chars = nullptr;
    int32_t *offsets = nullptr;
    int32_t *start_node_ids = nullptr;
    int32_t *start_offsets = nullptr;
    AlignResult *results = nullptr;
    QueryState *states = nullptr;
    int32_t *lengths = nullptr;
    TracebackMeta *traceback_meta = nullptr;
    size_t query_capacity = 0;
    size_t batch_capacity = 0;

    // Staging for the packed traceback: the cells the backtrace will actually
    // read, end to end, plus where each query's slice starts. Sized on the
    // cells a batch really used, so they grow with the workload and not with
    // kBeyondScopeCapacity. Kept on the workspace so that the page lock is paid
    // once per process, like the other host buffers.
    Cell *packed_device = nullptr;
    int32_t *pack_offsets = nullptr;      // device, one base per query
    size_t packed_device_capacity = 0;    // in cells
    size_t pack_offsets_capacity = 0;     // in queries
    Cell *packed_host = nullptr;
    size_t packed_host_capacity = 0;      // in cells
    bool packed_host_pinned = false;
};

/**
 * @brief Grow @p workspace so that a chunk of @p chunk_capacity queries whose
 * text is @p chars_bytes long fits, reusing whatever is already big enough.
 *
 * Two independent tests, because the two halves grow for different reasons: the
 * query text with the sequences a chunk happens to carry, everything else with
 * the chunk capacity.
 *
 * On failure returns Status::CudaError with the failing allocation named in the
 * error slot, and leaves the workspace holding exactly what it held before.
 */
Status ensure_workspace_capacity(DeviceWorkspace *workspace,
                                 int32_t chunk_capacity, size_t chars_bytes);

}  // namespace gpu
}  // namespace theseus
