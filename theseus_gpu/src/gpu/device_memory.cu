/**
 * @file device_memory.cu
 * @brief Device allocations that outlive a single batch: the graph and the
 * workspace.
 *
 * Everything here is CUDA runtime calls and bookkeeping -- no kernel is written
 * in this file, and the one it launches it reaches through kernel_launch.h.
 */

#include "gpu/device_memory.h"
#include "gpu/gpu_error.h"
#include "gpu/kernel_launch.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>

namespace theseus {
namespace gpu {

namespace {

/**
 * @brief cudaMalloc + H2D for one array. A zero-length array yields a null
 * device pointer, which the kernels never dereference because the matching
 * offset range is empty.
 */
template <typename T>
bool upload_array(T **device_ptr, const T *host_ptr, size_t count,
                  const char *what) {
    *device_ptr = nullptr;
    if (count == 0) {
        return true;
    }

    cudaError_t err = cudaMalloc(device_ptr, sizeof(T) * count);
    if (err != cudaSuccess) {
        set_error(what, err);
        return false;
    }

    err = cudaMemcpy(*device_ptr, host_ptr, sizeof(T) * count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_error(what, err);
        return false;
    }
    return true;
}

/**
 * @brief cudaMalloc for an output array, filled with a sentinel.
 *
 * The sentinel matters: an output buffer must never be seeded with the values
 * the caller expects, or a kernel that writes nothing at all would still
 * compare equal. Anything the kernel skips comes back as 0xFF bytes and fails
 * the comparison loudly.
 */
template <typename T>
bool alloc_output_array(T **device_ptr, size_t count, const char *what) {
    *device_ptr = nullptr;
    if (count == 0) {
        return true;
    }

    cudaError_t err = cudaMalloc(device_ptr, sizeof(T) * count);
    if (err != cudaSuccess) {
        set_error(what, err);
        return false;
    }

    err = cudaMemset(*device_ptr, 0xFF, sizeof(T) * count);
    if (err != cudaSuccess) {
        set_error(what, err);
        return false;
    }
    return true;
}

/**
 * @brief D2H for one array.
 */
template <typename T>
bool download_array(T *host_ptr, const T *device_ptr, size_t count,
                    const char *what) {
    if (count == 0) {
        return true;
    }

    cudaError_t err = cudaMemcpy(host_ptr, device_ptr, sizeof(T) * count,
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        set_error(what, err);
        return false;
    }
    return true;
}

/**
 * @brief The per-query arrays of a workspace, as one set.
 *
 * A grow allocates seven buffers and has to undo all of them if any one fails,
 * which is what the ALLOC_GROWN macro this replaces was for. Holding them in a
 * struct with a release() lets each failure site say what it means once.
 */
struct GrownArrays {
    int32_t *offsets = nullptr;
    int32_t *start_node_ids = nullptr;
    int32_t *start_offsets = nullptr;
    AlignResult *results = nullptr;
    QueryState *states = nullptr;
    int32_t *lengths = nullptr;
    TracebackMeta *traceback_meta = nullptr;

    /** @brief Give back whatever was allocated. cudaFree(nullptr) is a no-op. */
    void release() {
        cudaFree(offsets);
        cudaFree(start_node_ids);
        cudaFree(start_offsets);
        cudaFree(results);
        cudaFree(states);
        cudaFree(lengths);
        cudaFree(traceback_meta);
    }
};

/** @brief cudaMalloc @p bytes into @p ptr, recording @p label if it fails. */
template <typename T>
bool grow(T **ptr, size_t bytes, const char *label) {
    const cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess) {
        set_error(label, err);
        return false;
    }
    return true;
}

}  // namespace

void *alloc_host_pinned(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    void *buffer = nullptr;
    const cudaError_t err = cudaHostAlloc(&buffer, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        set_error("cudaHostAlloc(host batch buffers)", err);
        return nullptr;
    }
    return buffer;
}

void free_host_pinned(void *buffer) {
    if (buffer != nullptr) {
        cudaFreeHost(buffer);
    }
}

DeviceWorkspace *create_workspace() { return new DeviceWorkspace(); }

void free_workspace(DeviceWorkspace *workspace) {
    if (workspace == nullptr) return;
    cudaFree(workspace->chars);
    cudaFree(workspace->offsets);
    cudaFree(workspace->start_node_ids);
    cudaFree(workspace->start_offsets);
    cudaFree(workspace->results);
    cudaFree(workspace->states);
    cudaFree(workspace->lengths);
    cudaFree(workspace->traceback_meta);
    cudaFree(workspace->packed_device);
    cudaFree(workspace->pack_offsets);
    if (workspace->packed_host != nullptr) {
        if (workspace->packed_host_pinned) {
            cudaFreeHost(workspace->packed_host);
        } else {
            std::free(workspace->packed_host);
        }
    }
    delete workspace;
}

Status ensure_workspace_capacity(DeviceWorkspace *workspace,
                                 int32_t chunk_capacity, size_t chars_bytes) {
    if (chars_bytes > workspace->query_capacity) {
        char *grown = nullptr;
        if (!grow(&grown, chars_bytes, "cudaMalloc(chars)")) {
            return Status::CudaError;
        }
        cudaFree(workspace->chars);
        workspace->chars = grown;
        workspace->query_capacity = chars_bytes;
    }

    if (static_cast<size_t>(chunk_capacity) <= workspace->batch_capacity) {
        return Status::Ok;
    }

    // Allocation sizes, as opposed to the copy sizes above. Every per-query
    // buffer is allocated for a full chunk even when this chunk is short, which
    // the last chunk of a batch always is: the capacity test below is what
    // decides whether to reallocate, so sizing the arrays on one short chunk
    // would leave them too small for the next full one while the test still
    // said the workspace was big enough.
    const size_t cap_q = static_cast<size_t>(chunk_capacity);
    // The one buffer sized on the chunk capacity rather than on this chunk's
    // query count: it is the largest thing in the workspace by three orders of
    // magnitude, and keeping it at the capacity is what lets every chunk reuse
    // the same allocation (and the same one-off zeroing) instead of
    // reallocating per chunk.
    const size_t states_bytes = sizeof(QueryState) * cap_q;

    GrownArrays grown;
    if (!grow(&grown.offsets, sizeof(int32_t) * (cap_q + 1),
              "cudaMalloc(offsets)") ||
        !grow(&grown.start_node_ids, sizeof(int32_t) * cap_q,
              "cudaMalloc(start_node_ids)") ||
        !grow(&grown.start_offsets, sizeof(int32_t) * cap_q,
              "cudaMalloc(start_offsets)") ||
        !grow(&grown.results, sizeof(AlignResult) * cap_q,
              "cudaMalloc(results)") ||
        !grow(&grown.states, states_bytes, "cudaMalloc(query_states)")) {
        grown.release();
        return Status::CudaError;
    }

        // cudaMalloc does not zero, and two things need it to be zero.
        //
        // sp_cleared has to read as 0 -- "no entry of sp_off has been cleared
        // yet" -- on a state's first use, or the first query would skip a clear
        // it still needs.
        //
        // And the argument that nothing else needs zeroing turned out to be
        // wrong. The commit that dropped the per-batch cudaMemset said it was
        // "an argument, not a proof, so it is checked rather than trusted";
        // checking it says no: on ebola_exact_smoke, 8 queries, initcheck
        // reports 73 758 reads of uninitialised device memory, all of them the
        // same site once deduplicated --
        //
        //     core_check_end            align_core.h:39
        //     core_extend_diagonal      align_core.h:281
        //     align_one                 align_gpu.cu:1149   the score-0 seed
        //
        // -- and every commit after it inherits them, while the commit before
        // it is clean. The reads have never changed a result: all ten datasets
        // match their golden byte for byte at 64, 128 and 256 threads. But a
        // value read out of memory nobody wrote is whatever the last tenant of
        // that DRAM left, so "it matched" is a property of one run, not of the
        // program.
        //
        // Zeroing here rather than per batch keeps what that commit was after.
        // The cost it removed was 4.4 MB per query on *every* batch -- 8.6 GB
        // for 2048 queries, 40.8 ms against a 4.7 ms kernel; this pays it once
        // per allocation, and the workspace is allocated once per process and
        // grown only when a batch needs more states than the last one. It also
        // subsumes the strided memset of sp_cleared that used to be here.
    const cudaError_t err = cudaMemset(grown.states, 0, states_bytes);
    if (err != cudaSuccess) {
        set_error("cudaMemset(query_states)", err);
        grown.release();
        return Status::CudaError;
    }

    if (!grow(&grown.lengths, sizeof(int32_t) * cap_q, "cudaMalloc(lengths)") ||
        !grow(&grown.traceback_meta, sizeof(TracebackMeta) * cap_q,
              "cudaMalloc(traceback_meta)")) {
        grown.release();
        return Status::CudaError;
    }

    cudaFree(workspace->offsets);
    cudaFree(workspace->start_node_ids);
    cudaFree(workspace->start_offsets);
    cudaFree(workspace->results);
    cudaFree(workspace->states);
    cudaFree(workspace->lengths);
    cudaFree(workspace->traceback_meta);
    workspace->offsets = grown.offsets;
    workspace->start_node_ids = grown.start_node_ids;
    workspace->start_offsets = grown.start_offsets;
    workspace->results = grown.results;
    workspace->states = grown.states;
    workspace->lengths = grown.lengths;
    workspace->traceback_meta = grown.traceback_meta;
    workspace->batch_capacity = cap_q;
    return Status::Ok;
}

DeviceGraph *upload_graph(const GraphCsrView &graph) {
    clear_error();

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        set_error("cudaGetDeviceCount",
                  err != cudaSuccess ? err : cudaErrorNoDevice);
        return nullptr;
    }

    const int32_t num_vertices = graph.num_vertices;
    const int32_t num_chars = graph.vertex_offsets[num_vertices];
    const int32_t num_edges = graph.edge_offsets[num_vertices];

    DeviceGraph *device_graph = new DeviceGraph();
    device_graph->num_chars = num_chars;
    device_graph->num_edges = num_edges;

    const bool uploaded =
        upload_array(&device_graph->vertex_chars, graph.vertex_chars,
                     static_cast<size_t>(num_chars), "cudaMalloc/Memcpy(vertex_chars)") &&
        upload_array(&device_graph->vertex_offsets, graph.vertex_offsets,
                     static_cast<size_t>(num_vertices) + 1, "cudaMalloc/Memcpy(vertex_offsets)") &&
        upload_array(&device_graph->edge_targets, graph.edge_targets,
                     static_cast<size_t>(num_edges), "cudaMalloc/Memcpy(edge_targets)") &&
        upload_array(&device_graph->edge_overlaps, graph.edge_overlaps,
                     static_cast<size_t>(num_edges), "cudaMalloc/Memcpy(edge_overlaps)") &&
        upload_array(&device_graph->edge_offsets, graph.edge_offsets,
                     static_cast<size_t>(num_vertices) + 1, "cudaMalloc/Memcpy(edge_offsets)");

    if (!uploaded) {
        free_graph(device_graph);
        return nullptr;
    }

    device_graph->view.vertex_chars = device_graph->vertex_chars;
    device_graph->view.vertex_offsets = device_graph->vertex_offsets;
    device_graph->view.edge_targets = device_graph->edge_targets;
    device_graph->view.edge_overlaps = device_graph->edge_overlaps;
    device_graph->view.edge_offsets = device_graph->edge_offsets;
    device_graph->view.num_vertices = num_vertices;

    return device_graph;
}

void free_graph(DeviceGraph *graph) {
    if (graph == nullptr) {
        return;
    }
    cudaFree(graph->vertex_chars);
    cudaFree(graph->vertex_offsets);
    cudaFree(graph->edge_targets);
    cudaFree(graph->edge_overlaps);
    cudaFree(graph->edge_offsets);
    delete graph;
}

Status readback_graph(const DeviceGraph *graph,
                      char *out_vertex_chars,
                      int32_t *out_vertex_offsets,
                      int32_t *out_edge_targets,
                      int32_t *out_edge_overlaps,
                      int32_t *out_edge_offsets) {
    clear_error();

    if (graph == nullptr) {
        return Status::NoDevice;
    }

    const int32_t num_vertices = graph->view.num_vertices;
    const int32_t num_chars = graph->num_chars;
    const int32_t num_edges = graph->num_edges;

    if (num_vertices <= 0) {
        // Nothing to read back, and a zero-block launch is not a legal config.
        return Status::Ok;
    }

    char *d_chars = nullptr;
    int32_t *d_vertex_offsets = nullptr;
    int32_t *d_edge_targets = nullptr;
    int32_t *d_edge_overlaps = nullptr;
    int32_t *d_edge_offsets = nullptr;
    Status status = Status::Ok;
    cudaError_t err = cudaSuccess;

    // Single exit path: every allocation below is released at `cleanup`.
    if (!alloc_output_array(&d_chars, static_cast<size_t>(num_chars),
                            "cudaMalloc(readback chars)") ||
        !alloc_output_array(&d_vertex_offsets, static_cast<size_t>(num_vertices) + 1,
                            "cudaMalloc(readback vertex_offsets)") ||
        !alloc_output_array(&d_edge_targets, static_cast<size_t>(num_edges),
                            "cudaMalloc(readback edge_targets)") ||
        !alloc_output_array(&d_edge_overlaps, static_cast<size_t>(num_edges),
                            "cudaMalloc(readback edge_overlaps)") ||
        !alloc_output_array(&d_edge_offsets, static_cast<size_t>(num_vertices) + 1,
                            "cudaMalloc(readback edge_offsets)")) {
        status = Status::CudaError;
        goto cleanup;
    }

    err = launch_graph_readback(graph->view, d_chars, d_vertex_offsets,
                                d_edge_targets, d_edge_overlaps, d_edge_offsets);
    if (err != cudaSuccess) {
        set_error("graph_readback_kernel launch", err);
        status = Status::CudaError;
        goto cleanup;
    }

    if (!download_array(out_vertex_chars, d_chars, static_cast<size_t>(num_chars),
                        "cudaMemcpy(readback chars D2H)") ||
        !download_array(out_vertex_offsets, d_vertex_offsets,
                        static_cast<size_t>(num_vertices) + 1, "cudaMemcpy(readback vertex_offsets D2H)") ||
        !download_array(out_edge_targets, d_edge_targets, static_cast<size_t>(num_edges),
                        "cudaMemcpy(readback edge_targets D2H)") ||
        !download_array(out_edge_overlaps, d_edge_overlaps, static_cast<size_t>(num_edges),
                        "cudaMemcpy(readback edge_overlaps D2H)") ||
        !download_array(out_edge_offsets, d_edge_offsets,
                        static_cast<size_t>(num_vertices) + 1, "cudaMemcpy(readback edge_offsets D2H)")) {
        status = Status::CudaError;
        goto cleanup;
    }

cleanup:
    cudaFree(d_chars);
    cudaFree(d_vertex_offsets);
    cudaFree(d_edge_targets);
    cudaFree(d_edge_overlaps);
    cudaFree(d_edge_offsets);
    return status;
}

}  // namespace gpu
}  // namespace theseus
