#pragma once

#include <cstdint>

/**
 * @file align_gpu.h
 * @brief Boundary between the C++20 aligner and the CUDA backend.
 *
 * This header is compiled by both the host compiler and nvcc, so it must stay
 * free of CUDA types and of anything the two toolchains could disagree on:
 * POD types and free functions only.
 */

namespace theseus {
namespace gpu {

/**
 * @brief Flat view over a batch of query sequences.
 *
 * The concatenated layout is what a kernel can consume directly: a single
 * upload for the whole batch instead of one transfer per query.
 */
struct BatchView {
    const char *chars;       // Concatenated sequences, no separators
    const int32_t *offsets;  // num_seqs + 1 entries; sequence i is [offsets[i], offsets[i + 1])
    int32_t num_seqs;
};

/**
 * @brief Flat CSR view of the reference graph.
 *
 * Only what the alignment kernel reads is here: vertex texts and outgoing
 * edges. Vertex names and in-edges never reach the device, they exist solely
 * for GAF and GFA output on the host.
 */
struct GraphCsrView {
    const char *vertex_chars;      // Concatenated vertex texts, no separators
    const int32_t *vertex_offsets; // num_vertices + 1; text of vertex v is [off[v], off[v + 1])
    const int32_t *edge_targets;   // Destination vertex of each out-edge
    const int32_t *edge_overlaps;  // Overlap length, parallel to edge_targets
    const int32_t *edge_offsets;   // num_vertices + 1; out-edges of v are [off[v], off[v + 1])
    int32_t num_vertices;
};

/**
 * @brief POD result slot written by one alignment thread.
 *
 * This is deliberately independent from Cell and Alignment: the CUDA boundary
 * only carries primitive fields, while the host reconstructs richer C++ objects
 * after copying results back. end_* mirrors the terminal Cell selected by the
 * wavefront search. from_matrix uses Cell::Matrix's numeric values without
 * including cell.h here, keeping this header free of host-only dependencies.
 */
struct AlignResult {
    int32_t score;
    int32_t end_vertex_id;
    int32_t end_offset;
    int32_t end_diag;
    int64_t end_prev_pos;
    int8_t end_from_matrix;
    int8_t reached_end;
    int8_t capacity_exceeded;
    int8_t reserved;
};

/**
 * @brief Internal alignment penalties in POD form for CUDA kernels.
 */
struct AlignScoring {
    int32_t mism;
    int32_t gapo;
    int32_t gape;
    int32_t nscores;
};

/**
 * @brief Opaque handle to the graph as it lives in device memory.
 *
 * Defined only inside the CUDA backend: the host side never looks inside it,
 * which is what keeps CUDA types out of code compiled by the host compiler.
 */
struct DeviceGraph;

enum class Status {
    Ok,              // Batch aligned on the device
    NotImplemented,  // Device reachable, but this build cannot run the requested GPU path
    NoDevice,        // Built with CUDA, but no usable device at runtime
    NotCompiled,     // Built without the CUDA backend
    CudaError,       // A CUDA call failed, see last_error()
};

inline const char *status_message(Status s) {
    switch (s) {
        case Status::Ok:
            return "aligned on device";
        case Status::NotImplemented:
            return "device reached, but requested GPU path is not implemented";
        case Status::NoDevice:
            return "no CUDA device available, fell back to CPU";
        case Status::NotCompiled:
            return "built without CUDA (-DTHESEUS_ENABLE_CUDA=ON to enable), fell back to CPU";
        case Status::CudaError:
            return "CUDA error, fell back to CPU";
    }
    return "unknown status";
}

/**
 * @brief Description of the last CUDA failure, or "" if none.
 */
const char *last_error();

/**
 * @brief Align a batch of queries on the device.
 *
 * Version 0 uploads @p batch, launches one CUDA thread per query, and
 * runs a complete serial Theseus alignment inside that thread. @p out_seq_lengths
 * receives each sequence length as computed on the device, which lets the caller
 * verify the upload against host-side data.
 *
 * @param batch             Sequences to align
 * @param graph             Graph already uploaded to device memory
 * @param start_node_ids    Host array of batch.num_seqs numeric start vertices
 * @param start_offsets     Host array of batch.num_seqs start offsets
 * @param scoring           Internal penalties and Scope ring size
 * @param out_results       Caller-owned buffer of batch.num_seqs result slots
 * @param out_query_states  Caller-owned QueryState buffer, opaque at this boundary
 * @param out_seq_lengths   Caller-owned buffer of batch.num_seqs entries
 * @return Status           Ok only when the alignment itself ran on the device
 */
Status align_batch(const BatchView &batch,
                   const DeviceGraph *graph,
                   const int32_t *start_node_ids,
                   const int32_t *start_offsets,
                   AlignScoring scoring,
                   AlignResult *out_results,
                   void *out_query_states,
                   int32_t *out_seq_lengths);

/**
 * @brief Copy the graph to device memory, once.
 *
 * The graph is read-only and shared by every query, so it is uploaded one time
 * and kept for the lifetime of the aligner rather than sent per batch.
 *
 * @param graph  CSR over host memory
 * @return       Handle to free with free_graph(), or nullptr on failure (see last_error())
 */
DeviceGraph *upload_graph(const GraphCsrView &graph);

/**
 * @brief Release a graph obtained from upload_graph(). Accepts nullptr.
 */
void free_graph(DeviceGraph *graph);

/**
 * @brief Read the uploaded graph back out of device memory.
 *
 * The copy is made by a kernel that walks the CSR the same way the alignment
 * kernel will, so a mismatch means device code cannot traverse the graph, not
 * merely that some bytes moved wrong. Every buffer must be sized as the CSR
 * that was uploaded.
 *
 * @param graph  Handle from upload_graph()
 * @return       Ok when the readback ran, whatever the contents turn out to be
 */
Status readback_graph(const DeviceGraph *graph,
                      char *out_vertex_chars,
                      int32_t *out_vertex_offsets,
                      int32_t *out_edge_targets,
                      int32_t *out_edge_overlaps,
                      int32_t *out_edge_offsets);

}  // namespace gpu
}  // namespace theseus
