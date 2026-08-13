#pragma once

#include <cstddef>
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

struct AlignOptions {
    int32_t threads_per_block = 128;
    int32_t collect_counters = 0;
};

struct TimingReport {
    float graph_ms = 0.0f;
    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    float end_to_end_ms = 0.0f;
};

/**
 * @brief Opaque handle to the graph as it lives in device memory.
 *
 * Defined only inside the CUDA backend: the host side never looks inside it,
 * which is what keeps CUDA types out of code compiled by the host compiler.
 */
struct DeviceGraph;
struct DeviceWorkspace;

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

const TimingReport &last_timing();

/**
 * @brief Align a batch of queries on the device.
 *
 * Uploads @p batch and launches one block per query, each block running a
 * complete Theseus alignment over its own QueryState. @p out_seq_lengths
 * receives each sequence length as computed on the device, which lets the caller
 * verify the upload against host-side data.
 *
 * @param batch             Sequences to align
 * @param graph             Graph already uploaded to device memory
 * @param start_node_ids    Host array of batch.num_seqs numeric start vertices
 * @param start_offsets     Host array of batch.num_seqs start offsets
 * @param scoring           Internal penalties and Scope ring size
 * @param out_results       Caller-owned buffer of batch.num_seqs result slots
 * @param out_query_states  Caller-owned CompactTracebackState buffer
 * @param out_seq_lengths   Caller-owned buffer of batch.num_seqs entries
 * @return Status           Ok only when the alignment itself ran on the device
 */
Status align_batch(const BatchView &batch,
                   const DeviceGraph *graph,
                   DeviceWorkspace *workspace,
                   const int32_t *start_node_ids,
                   const int32_t *start_offsets,
                   AlignScoring scoring,
                   AlignOptions options,
                   AlignResult *out_results,
                   void *out_query_states,
                   int32_t *out_seq_lengths);

DeviceWorkspace *create_workspace();
void free_workspace(DeviceWorkspace *workspace);

/**
 * @brief Page-locked host memory, for buffers the device copies into.
 *
 * The compact traceback state is the batch's whole D2H payload -- 288 KB per
 * query, 590 MB for 2048 -- and a freshly allocated pageable buffer pays a page
 * fault per page on the way in, on top of the staging copy the driver has to
 * make because it cannot DMA into pageable memory. Allocating it once, page
 * locked, and keeping it for the aligner's lifetime removes both.
 *
 * The size is in bytes and the contents are not initialised: align_batch
 * overwrites every buffer it is given. free_host_pinned accepts nullptr. On a
 * build without CUDA these fall back to malloc/free, so the caller needs no
 * second path.
 *
 * @return Page-locked buffer, or nullptr if the allocation failed
 */
void *alloc_host_pinned(size_t bytes);
void free_host_pinned(void *buffer);

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
