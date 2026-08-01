/**
 * @file align_gpu.cu
 * @brief CUDA backend for the Theseus proxy aligner.
 *
 * Version 0 maps one CUDA thread to one complete query alignment. Inside that
 * thread the Theseus wavefront algorithm remains serial and follows the CPU
 * control flow implemented in gpu/align_core.h.
 */

#include "gpu/align_gpu.h"
#include "query_state.h"
#include "gpu/align_core.h"

#include <cuda_runtime.h>

#include <cstdio>

namespace theseus {
namespace gpu {

namespace {

char g_last_error[256] = {0};

void set_error(const char *context, cudaError_t err) {
    std::snprintf(g_last_error, sizeof(g_last_error), "%s: %s", context,
                  cudaGetErrorString(err));
}

void clear_error() { g_last_error[0] = '\0'; }

/**
 * @brief One block per sequence, which is the geometry the alignment kernel
 * will use: a block owns a query and its wavefronts for the whole alignment.
 *
 * For now a block only reports the length of its own sequence, which is enough
 * to prove the offsets survived the upload intact.
 */
__global__ void seq_length_kernel(const int32_t *offsets, int32_t num_seqs,
                                  int32_t *out_seq_lengths) {
    const int32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= num_seqs) {
        return;
    }
    out_seq_lengths[i] = offsets[i + 1] - offsets[i];
}

__global__ void align_kernel(BatchView batch, GraphCsrView graph,
                             const int32_t *start_node_ids,
                             const int32_t *start_offsets,
                             AlignScoring scoring, QueryState *states,
                             AlignResult *results) {
    const int32_t q = blockIdx.x * blockDim.x + threadIdx.x;
    if (q >= batch.num_seqs) {
        return;
    }
    const int32_t begin = batch.offsets[q];
    const int32_t end = batch.offsets[q + 1];
    align_one(states[q], scoring, batch.chars + begin, end - begin, graph,
              start_node_ids[q], start_offsets[q], results[q]);
}

/**
 * @brief Copy the CSR back out, one block per vertex.
 *
 * Each block reaches its own text and out-edges through the offset arrays,
 * which is the traversal the alignment kernel will do. Reading the graph any
 * other way here would test the upload but not the traversal.
 */
__global__ void graph_readback_kernel(GraphCsrView graph,
                                      char *out_vertex_chars,
                                      int32_t *out_vertex_offsets,
                                      int32_t *out_edge_targets,
                                      int32_t *out_edge_overlaps,
                                      int32_t *out_edge_offsets) {
    const int32_t v = blockIdx.x;
    if (v >= graph.num_vertices) {
        return;
    }

    const int32_t text_begin = graph.vertex_offsets[v];
    const int32_t text_end = graph.vertex_offsets[v + 1];
    for (int32_t i = text_begin + threadIdx.x; i < text_end; i += blockDim.x) {
        out_vertex_chars[i] = graph.vertex_chars[i];
    }

    const int32_t edge_begin = graph.edge_offsets[v];
    const int32_t edge_end = graph.edge_offsets[v + 1];
    for (int32_t e = edge_begin + threadIdx.x; e < edge_end; e += blockDim.x) {
        out_edge_targets[e] = graph.edge_targets[e];
        out_edge_overlaps[e] = graph.edge_overlaps[e];
    }

    if (threadIdx.x == 0) {
        out_vertex_offsets[v] = text_begin;
        out_edge_offsets[v] = edge_begin;
        if (v == graph.num_vertices - 1) {
            out_vertex_offsets[v + 1] = text_end;
            out_edge_offsets[v + 1] = edge_end;
        }
    }
}

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

}  // namespace

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

const char *last_error() { return g_last_error; }

Status align_batch(const BatchView &batch,
                   const DeviceGraph *graph,
                   const int32_t *start_node_ids,
                   const int32_t *start_offsets,
                   AlignScoring scoring,
                   AlignResult *out_results,
                   void *out_query_states,
                   int32_t *out_seq_lengths) {
    clear_error();

    if (batch.num_seqs <= 0) {
        return Status::NotImplemented;
    }
    if (graph == nullptr) {
        return Status::NoDevice;
    }

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        if (err != cudaSuccess) {
            set_error("cudaGetDeviceCount", err);
        }
        return Status::NoDevice;
    }

    const size_t chars_bytes = static_cast<size_t>(batch.offsets[batch.num_seqs]);
    const size_t offsets_bytes = sizeof(int32_t) * (static_cast<size_t>(batch.num_seqs) + 1);
    const size_t per_query_i32_bytes = sizeof(int32_t) * static_cast<size_t>(batch.num_seqs);
    const size_t lengths_bytes = sizeof(int32_t) * static_cast<size_t>(batch.num_seqs);
    const size_t results_bytes = sizeof(AlignResult) * static_cast<size_t>(batch.num_seqs);
    const size_t states_bytes = sizeof(QueryState) * static_cast<size_t>(batch.num_seqs);

    char *d_chars = nullptr;
    int32_t *d_offsets = nullptr;
    int32_t *d_start_node_ids = nullptr;
    int32_t *d_start_offsets = nullptr;
    AlignResult *d_results = nullptr;
    QueryState *d_states = nullptr;
    int32_t *d_lengths = nullptr;
    BatchView device_batch{nullptr, nullptr, 0};
    Status status = Status::Ok;

    // Single exit path: every allocation below is released at `cleanup`.
    err = cudaMalloc(&d_chars, chars_bytes);
    if (err != cudaSuccess) {
        set_error("cudaMalloc(chars)", err);
        status = Status::CudaError;
        goto cleanup;
    }
    err = cudaMalloc(&d_offsets, offsets_bytes);
    if (err != cudaSuccess) {
        set_error("cudaMalloc(offsets)", err);
        status = Status::CudaError;
        goto cleanup;
    }
    err = cudaMalloc(&d_start_node_ids, per_query_i32_bytes);
    if (err != cudaSuccess) {
        set_error("cudaMalloc(start_node_ids)", err);
        status = Status::CudaError;
        goto cleanup;
    }
    err = cudaMalloc(&d_start_offsets, per_query_i32_bytes);
    if (err != cudaSuccess) {
        set_error("cudaMalloc(start_offsets)", err);
        status = Status::CudaError;
        goto cleanup;
    }
    err = cudaMalloc(&d_results, results_bytes);
    if (err != cudaSuccess) {
        set_error("cudaMalloc(results)", err);
        status = Status::CudaError;
        goto cleanup;
    }
    err = cudaMalloc(&d_states, states_bytes);
    if (err != cudaSuccess) {
        set_error("cudaMalloc(query_states)", err);
        status = Status::CudaError;
        goto cleanup;
    }
    err = cudaMalloc(&d_lengths, lengths_bytes);
    if (err != cudaSuccess) {
        set_error("cudaMalloc(lengths)", err);
        status = Status::CudaError;
        goto cleanup;
    }

    err = cudaMemcpy(d_chars, batch.chars, chars_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_error("cudaMemcpy(chars H2D)", err);
        status = Status::CudaError;
        goto cleanup;
    }
    err = cudaMemcpy(d_offsets, batch.offsets, offsets_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_error("cudaMemcpy(offsets H2D)", err);
        status = Status::CudaError;
        goto cleanup;
    }
    err = cudaMemcpy(d_start_node_ids, start_node_ids, per_query_i32_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_error("cudaMemcpy(start_node_ids H2D)", err);
        status = Status::CudaError;
        goto cleanup;
    }
    err = cudaMemcpy(d_start_offsets, start_offsets, per_query_i32_bytes, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_error("cudaMemcpy(start_offsets H2D)", err);
        status = Status::CudaError;
        goto cleanup;
    }
    err = cudaMemset(d_results, 0, results_bytes);
    if (err != cudaSuccess) {
        set_error("cudaMemset(results)", err);
        status = Status::CudaError;
        goto cleanup;
    }
    err = cudaMemset(d_states, 0, states_bytes);
    if (err != cudaSuccess) {
        set_error("cudaMemset(query_states)", err);
        status = Status::CudaError;
        goto cleanup;
    }

    constexpr int kThreadsPerBlock = 128;
    const int blocks = (batch.num_seqs + kThreadsPerBlock - 1) / kThreadsPerBlock;

    seq_length_kernel<<<blocks, kThreadsPerBlock>>>(d_offsets, batch.num_seqs, d_lengths);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_error("seq_length_kernel launch", err);
        status = Status::CudaError;
        goto cleanup;
    }

    device_batch = BatchView{d_chars, d_offsets, batch.num_seqs};
    align_kernel<<<blocks, kThreadsPerBlock>>>(device_batch, graph->view,
                                               d_start_node_ids, d_start_offsets,
                                               scoring, d_states, d_results);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_error("align_kernel launch", err);
        status = Status::CudaError;
        goto cleanup;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_error("align_kernel synchronize", err);
        status = Status::CudaError;
        goto cleanup;
    }

    err = cudaMemcpy(out_seq_lengths, d_lengths, lengths_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        set_error("cudaMemcpy(lengths D2H)", err);
        status = Status::CudaError;
        goto cleanup;
    }
    err = cudaMemcpy(out_results, d_results, results_bytes, cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        set_error("cudaMemcpy(results D2H)", err);
        status = Status::CudaError;
        goto cleanup;
    }
    if (out_query_states != nullptr) {
        err = cudaMemcpy(out_query_states, d_states, states_bytes, cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            set_error("cudaMemcpy(query_states D2H)", err);
            status = Status::CudaError;
            goto cleanup;
        }
    }

cleanup:
    cudaFree(d_chars);
    cudaFree(d_offsets);
    cudaFree(d_start_node_ids);
    cudaFree(d_start_offsets);
    cudaFree(d_results);
    cudaFree(d_states);
    cudaFree(d_lengths);
    return status;
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

    graph_readback_kernel<<<num_vertices, 32>>>(graph->view, d_chars, d_vertex_offsets,
                                                d_edge_targets, d_edge_overlaps,
                                                d_edge_offsets);
    err = cudaGetLastError();
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
