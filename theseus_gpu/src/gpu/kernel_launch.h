#pragma once

/**
 * @file kernel_launch.h
 * @brief Host-side entry to every kernel the backend owns.
 *
 * The `<<<>>>` syntax is nvcc's, not C++'s, so keeping it inside align_gpu.cu --
 * the one translation unit that holds device code -- is what lets the memory
 * management and the batch orchestration be ordinary C++ next to it. Each
 * wrapper launches its kernel and returns cudaGetLastError(); none of them
 * synchronises, because the caller decides where the batch waits.
 *
 * Every launch geometry is derived here rather than passed in: the block count
 * and the dynamic shared memory follow from the kernel, so the callers cannot
 * disagree with it about either.
 */

#include "gpu/align_gpu.h"
#include "query_state.h"

#include <cuda_runtime.h>

namespace theseus {
namespace gpu {

/**
 * @brief What the host reads back from a finished QueryState to lay out the
 * traceback: the three wavefront sizes, plus the diagnostics.
 *
 * Small and fixed, so it can be copied for the whole chunk in one go before
 * anything decides how many cells the packed copy will need.
 */
struct TracebackMeta {
    int32_t m_size;
    int32_t m_jumps_size;
    int32_t i_jumps_size;
    int32_t peak_wf;
    int32_t cap_required;
    int32_t cap_available;
    int8_t capacity_exceeded;
    int8_t cap_reason;
    int8_t reserved[2];
};

/** @brief Report each sequence's length, to prove the offsets survived the upload. */
cudaError_t launch_seq_length(int32_t threads_per_block,
                              const int32_t *offsets, int32_t num_seqs,
                              int32_t *out_seq_lengths);

/**
 * @brief The alignment itself: one block per query, over @p batch.num_seqs
 * blocks, with the shared memory the block size implies.
 */
cudaError_t launch_align_batch(int32_t threads_per_block,
                               const BatchView &batch,
                               const GraphCsrView &graph,
                               const int32_t *start_node_ids,
                               const int32_t *start_offsets,
                               AlignScoring scoring,
                               QueryState *states,
                               AlignResult *results);

/** @brief Gather the traceback sizes and diagnostics of @p count states. */
cudaError_t launch_traceback_meta(int32_t threads_per_block,
                                  const QueryState *states, int32_t count,
                                  TracebackMeta *metadata);

/**
 * @brief Gather the cells the host backtrace will read into one dense buffer,
 * at the offsets @p base, which the host derived from the metadata above.
 */
cudaError_t launch_pack_traceback(int32_t threads_per_block,
                                  const QueryState *states, int32_t count,
                                  const int32_t *base, Cell *packed);

/** @brief Copy the CSR back out of device memory, one block per vertex. */
cudaError_t launch_graph_readback(const GraphCsrView &graph,
                                  char *out_vertex_chars,
                                  int32_t *out_vertex_offsets,
                                  int32_t *out_edge_targets,
                                  int32_t *out_edge_overlaps,
                                  int32_t *out_edge_offsets);

}  // namespace gpu
}  // namespace theseus
