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
TimingReport g_last_timing;

/**
 * @brief Bytes of dynamic shared memory one config1 block needs.
 *
 * Both staging buffers hold exactly one tile, so the requirement scales with the
 * block size instead of with kScopeWavefrontCapacity. The layout here must match
 * the partitioning at the top of theseus_align_batch_config1_kernel.
 */
size_t config1_shared_bytes(int32_t threads_per_block) {
    return static_cast<size_t>(threads_per_block) *
           (2 * sizeof(Cell) + 2 * sizeof(int));
}

struct Config1ICandidateRanges {
    Range i_dense;
    int32_t i_jumps_size;
    Range m_dense;
    int32_t m_jumps_size;
    int64_t *i_jumps;
    int64_t *m_jumps;
    int32_t pos_prev_i;
    int32_t pos_prev_m;
    int32_t upper_bound;
    int32_t total;
};


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


__device__ Range config1_empty_range() {
    return Range{0, 0};
}

__device__ int32_t config1_range_len(Range range) {
    return static_cast<int32_t>(range.end - range.start);
}

__device__ Config1ICandidateRanges config1_prepare_i_candidate_ranges(
        QueryState &qs, const AlignScoring &scoring,
        const GraphCsrView &graph, int32_t score, int32_t v) {
    Config1ICandidateRanges ranges;
    ranges.i_dense = config1_empty_range();
    ranges.i_jumps_size = 0;
    ranges.m_dense = config1_empty_range();
    ranges.m_jumps_size = 0;
    ranges.i_jumps = nullptr;
    ranges.m_jumps = nullptr;
    ranges.pos_prev_i = score - scoring.gape;
    ranges.pos_prev_m = score - (scoring.gapo + scoring.gape);
    ranges.upper_bound = vertex_len(graph, v);
    ranges.total = 0;

    const int32_t v_id = vd_get_id(qs, v);
    if (v_id < 0) {
        return ranges;
    }

    if (ranges.pos_prev_i >= 0) {
        if (sc_i_pos_size(qs, ranges.pos_prev_i) > v_id) {
            ranges.i_dense = sc_i_pos(qs, ranges.pos_prev_i)[v_id];
        }
        const int32_t pos_prev_i_scope = vd_get_pos(qs, ranges.pos_prev_i);
        ranges.i_jumps = vd_i_jumps(qs, v, pos_prev_i_scope);
        ranges.i_jumps_size = vd_i_jumps_size(qs, v, pos_prev_i_scope);
    }
    if (ranges.pos_prev_m >= 0) {
        if (sc_m_pos_size(qs, ranges.pos_prev_m) > v_id) {
            ranges.m_dense = sc_m_pos(qs, ranges.pos_prev_m)[v_id];
        }
        const int32_t pos_prev_m_scope = vd_get_pos(qs, ranges.pos_prev_m);
        ranges.m_jumps = vd_m_jumps(qs, v, pos_prev_m_scope);
        ranges.m_jumps_size = vd_m_jumps_size(qs, v, pos_prev_m_scope);
    }

    ranges.total = config1_range_len(ranges.i_dense) + ranges.i_jumps_size +
                   config1_range_len(ranges.m_dense) + ranges.m_jumps_size;
    return ranges;
}

__device__ bool config1_make_i_candidate(QueryState &qs,
                                         const Config1ICandidateRanges &ranges,
                                         int32_t idx,
                                         int32_t query_len,
                                         Cell &candidate) {
    const int32_t i_dense_count = config1_range_len(ranges.i_dense);
    const int32_t i_jumps_begin = i_dense_count;
    const int32_t m_dense_begin = i_jumps_begin + ranges.i_jumps_size;
    const int32_t m_jumps_begin = m_dense_begin + config1_range_len(ranges.m_dense);

    if (idx < i_dense_count) {
        candidate = sc_i_wf(qs, ranges.pos_prev_i)[ranges.i_dense.start + idx];
        candidate.diag += 1;
    } else if (idx < m_dense_begin) {
        const int32_t local = idx - i_jumps_begin;
        const int64_t pos = ranges.i_jumps[local];
        candidate = qs.bs_i_jumps_wf[pos];
        candidate.prev_pos = pos;
        candidate.from_matrix = Cell::Matrix::IJumps;
        candidate.diag += 1;
    } else if (idx < m_jumps_begin) {
        const int32_t local = idx - m_dense_begin;
        const int64_t pos = ranges.m_dense.start + local;
        candidate = qs.bs_m_wf[pos];
        candidate.prev_pos = pos;
        candidate.from_matrix = Cell::Matrix::M;
        candidate.diag += 1;
    } else {
        const int32_t local = idx - m_jumps_begin;
        const int64_t pos = ranges.m_jumps[local];
        candidate = qs.bs_m_jumps_wf[pos];
        candidate.prev_pos = pos;
        candidate.from_matrix = Cell::Matrix::MJumps;
        candidate.diag += 1;
    }

    const int32_t new_col = candidate.offset + candidate.diag;
    return candidate.offset <= query_len && new_col <= ranges.upper_bound;
}

__device__ void config1_merge_i_candidate(QueryState &qs,
                                          const Cell &candidate) {
    Cell &cell = sp_access_alloc(qs, candidate.diag);
    if (cell.offset < candidate.offset) {
        cell = candidate;
    }
}

constexpr int32_t kConfig1MaxWarps = 8;  // 256 threads, the largest block allowed

/**
 * @brief Block-parallel replacement for the densify loops of I, D and M.
 *
 * The serial form walks the active diagonals, keeps those still valid for this
 * vertex and appends them in order. That is a filter followed by a stream
 * compaction, and the test (vd_valid_diagonal) is a read-only scan of the
 * vertex's invalid segments, so every diagonal can be tested independently.
 * Only the write position is shared, and an exclusive prefix sum reproduces the
 * serial positions exactly, which is what keeps the output byte-identical.
 *
 * The prefix sum is a warp ballot plus a scan over the (at most 8) warp totals,
 * so it needs no external scan primitive and 9 ints of shared memory.
 *
 * Overflow keeps the serial semantics: elements that do not fit are dropped and
 * cap_fail records the buffer with the size the first failing push would have
 * reported. @p track_peak mirrors sc_wf_push updating sc_peak_wf, which
 * bs_push_back does not do.
 */
__device__ Range config1_densify(QueryState &qs, Cell::Matrix matrix, int32_t v,
                                 Cell *wf, int32_t &wf_size, int32_t capacity,
                                 CapBuffer cap_buffer, bool track_peak,
                                 int32_t *shared_warp_base, int32_t &shared_accum) {
    const int32_t ndiags = qs.sp_ndiags;   // uniform: densify never touches it
    const int32_t range_start = wf_size;   // read before thread 0 updates it
    const int32_t lane = threadIdx.x & 31;
    const int32_t warp = threadIdx.x >> 5;
    const int32_t nwarps = (static_cast<int32_t>(blockDim.x) + 31) >> 5;

    if (threadIdx.x == 0) {
        shared_accum = 0;
    }
    __syncthreads();

    for (int32_t tile = 0; tile < ndiags; tile += blockDim.x) {
        const int32_t di = tile + static_cast<int32_t>(threadIdx.x);
        int32_t flag = 0;
        Cell value{-1, -1, -1, -1, Cell::Matrix::None};
        if (di < ndiags) {
            const int32_t diag = qs.sp_diags[di];
            if (vd_valid_diagonal(qs, matrix, v, diag)) {
                flag = 1;
                value = sp_at(qs, diag);
            }
        }

        // Every thread reaches the ballot, including the out-of-range ones with
        // flag 0, so the warp stays converged.
        const unsigned ballot = __ballot_sync(0xffffffffu, flag != 0);
        const int32_t warp_prefix =
            __popc(ballot & ((1u << lane) - 1u));
        if (lane == 0) {
            shared_warp_base[warp] = __popc(ballot);
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            int32_t acc = shared_accum;
            for (int32_t w = 0; w < nwarps; ++w) {
                const int32_t count = shared_warp_base[w];
                shared_warp_base[w] = acc;   // exclusive base for this warp
                acc += count;
            }
            shared_accum = acc;
        }
        __syncthreads();

        if (flag != 0) {
            const int32_t pos = range_start + shared_warp_base[warp] + warp_prefix;
            if (pos < capacity) {
                wf[pos] = value;
            }
        }
        __syncthreads();   // shared_warp_base is reused by the next tile
    }

    if (threadIdx.x == 0) {
        const int32_t total = shared_accum;
        const int32_t room = capacity - range_start;
        const int32_t fits = room > 0 ? room : 0;
        const int32_t written = total < fits ? total : fits;
        if (total > fits) {
            cap_fail(qs, cap_buffer, capacity + 1, capacity);
        }
        wf_size = range_start + written;
        if (track_peak && wf_size > qs.sc_peak_wf) {
            qs.sc_peak_wf = wf_size;
        }
    }
    __syncthreads();

    Range r;
    r.start = range_start;
    r.end = wf_size;
    return r;
}

__device__ Range config1_finish_i_wavefront(QueryState &qs, int32_t score,
                                            int32_t v,
                                            int32_t *shared_warp_base,
                                            int32_t &shared_accum) {
    const Range new_range =
        config1_densify(qs, Cell::Matrix::I, v, sc_i_wf(qs, score),
                        sc_i_wf_size(qs, score), kScopeWavefrontCapacity,
                        kCapScopeWavefront, true, shared_warp_base, shared_accum);
    if (threadIdx.x == 0) {
        sc_pos_push(qs, sc_i_pos(qs, score), sc_i_pos_size(qs, score), new_range);
    }
    __syncthreads();
    return new_range;
}

/**
 * @brief Build the I candidates for one vertex and merge them into the scratchpad.
 *
 * The candidate space is consumed one tile of blockDim.x elements at a time, the
 * same way config1_extend_and_consume_m_cells consumes the M range. Staging a
 * tile rather than the whole candidate space is what lets the shared buffers be
 * sized on the block instead of on kScopeWavefrontCapacity, and it removes the
 * capacity ceiling that used to abort the alignment when a vertex produced more
 * than kScopeWavefrontCapacity candidates.
 *
 * The merge stays on thread 0: candidates can collide on a diagonal and the
 * winner is the largest offset, so merging in candidate order is what keeps the
 * result identical to the CPU. Tiling preserves that order exactly.
 */
__device__ void config1_generate_and_merge_i_candidates(
        QueryState &qs, const AlignScoring &scoring, const char *query,
        int32_t query_len,
        const GraphCsrView &graph, int32_t score, int32_t v,
        Config1ICandidateRanges &shared_i_ranges,
        Cell *shared_i_candidates, int *shared_i_valid,
        int &shared_i_count, int32_t *shared_warp_base, int32_t &shared_accum,
        int &block_end, Cell &block_end_cell) {
    if (threadIdx.x == 0) {
        shared_i_ranges = config1_prepare_i_candidate_ranges(qs, scoring, graph,
                                                             score, v);
        shared_i_count = shared_i_ranges.total;
    }
    __syncthreads();

    // Uniform across the block: written by thread 0 before the barrier above, so
    // every thread runs the same number of tiles and reaches the same barriers.
    const int32_t count = shared_i_count;
    const int32_t tile = static_cast<int32_t>(blockDim.x);

    for (int32_t tile_start = 0; tile_start < count; tile_start += tile) {
        const int32_t idx = tile_start + static_cast<int32_t>(threadIdx.x);
        if (idx < count) {
            Cell candidate{-1, -1, -1, -1, Cell::Matrix::None};
            const bool valid = config1_make_i_candidate(qs, shared_i_ranges, idx,
                                                        query_len, candidate);
            shared_i_candidates[threadIdx.x] = candidate;
            shared_i_valid[threadIdx.x] = valid ? 1 : 0;
        } else {
            shared_i_valid[threadIdx.x] = 0;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            const int32_t remaining = count - tile_start;
            const int32_t lanes = remaining < tile ? remaining : tile;
            for (int32_t lane = 0; lane < lanes; ++lane) {
                if (shared_i_valid[lane] == 0) {
                    continue;
                }
                config1_merge_i_candidate(qs, shared_i_candidates[lane]);
            }
        }
        __syncthreads();
    }

    const Range new_range = config1_finish_i_wavefront(qs, score, v,
                                                       shared_warp_base,
                                                       shared_accum);
    if (threadIdx.x == 0) {
        // Same tail as the CPU's next_I and as core_next_i: the I cells that
        // just reached the last column of this vertex open jumps into its
        // neighbours. core_store_m_jump can hit the end condition, so the
        // block's end state has to travel through here too.
        if (edge_begin(graph, v) < edge_end(graph, v)) {
            bool end = block_end != 0;
            Cell end_cell = block_end_cell;
            core_check_and_store_jumps(qs, query, query_len, graph, score, v,
                                       sc_i_wf(qs, score), new_range, end,
                                       end_cell);
            block_end = end ? 1 : 0;
            block_end_cell = end_cell;
        }
    }
    __syncthreads();
}

__device__ void config1_extend_and_consume_m_cells(QueryState &qs,
                                                   const char *query,
                                                   int32_t query_len,
                                                   const GraphCsrView &graph,
                                                   int32_t score,
                                                   int64_t range_start,
                                                   int64_t range_end,
                                                   Cell *shared_m_cells,
                                                   int *shared_m_valid,
                                                   int &block_end,
                                                   Cell &block_end_cell) {
    for (int64_t chunk_start = range_start; chunk_start < range_end;
         chunk_start += blockDim.x) {
        const int64_t idx = chunk_start + threadIdx.x;
        if (idx < range_end) {
            Cell cell = qs.bs_m_wf[idx];
            int32_t j = cell.diag + cell.offset;
            core_lcp(query, query_len, graph, cell.vertex_id, cell.offset, j);
            shared_m_cells[threadIdx.x] = cell;
            shared_m_valid[threadIdx.x] = 1;
        } else {
            shared_m_valid[threadIdx.x] = 0;
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            bool end = block_end != 0;
            Cell end_cell = block_end_cell;
            for (int32_t lane = 0; lane < blockDim.x; ++lane) {
                if (shared_m_valid[lane] == 0) {
                    continue;
                }
                const int64_t cell_idx = chunk_start + lane;
                Cell &cell = qs.bs_m_wf[cell_idx];
                cell = shared_m_cells[lane];
                core_check_end(cell, query_len, end, end_cell);
                const int32_t j = cell.diag + cell.offset;
                if (j == vertex_len(graph, cell.vertex_id) && cell.offset <= query_len &&
                    edge_begin(graph, cell.vertex_id) < edge_end(graph, cell.vertex_id)) {
                    core_store_m_jump(qs, query, query_len, graph, score,
                                      cell.vertex_id, cell, cell_idx,
                                      Cell::Matrix::M, end, end_cell);
                }
            }
            block_end = end ? 1 : 0;
            block_end_cell = end_cell;
        }
        __syncthreads();
    }
}

__device__ void config1_process_vertex(QueryState &qs,
                                       const AlignScoring &scoring,
                                       const char *query,
                                       int32_t query_len,
                                       const GraphCsrView &graph,
                                       int32_t score,
                                       int32_t v,
                                       int64_t &shared_range_start,
                                       int64_t &shared_range_end,
                                       Cell *shared_m_cells,
                                       int *shared_m_valid,
                                       Config1ICandidateRanges &shared_i_ranges,
                                       Cell *shared_i_candidates,
                                       int *shared_i_valid,
                                       int &shared_i_count,
                                       int32_t *shared_warp_base,
                                       int32_t &shared_accum,
                                       int &block_end,
                                       Cell &block_end_cell) {
    if (threadIdx.x == 0) {
        shared_range_start = 0;
        shared_range_end = 0;

    }
    __syncthreads();

    config1_generate_and_merge_i_candidates(qs, scoring, query, query_len, graph,
                                            score, v, shared_i_ranges,
                                            shared_i_candidates, shared_i_valid,
                                            shared_i_count, shared_warp_base,
                                            shared_accum, block_end,
                                            block_end_cell);

    // Sparsify stays on thread 0 (it merges into the shared scratchpad, where
    // candidates collide by diagonal); only the densify that follows it is a
    // filter+compaction the whole block can run.
    if (threadIdx.x == 0) {
        sp_reset(qs);
        core_next_d_sparsify(qs, scoring, query_len, graph, score, v);
    }
    __syncthreads();
    {
        const Range d_range =
            config1_densify(qs, Cell::Matrix::D, v, sc_d_wf(qs, score),
                            sc_d_wf_size(qs, score), kScopeWavefrontCapacity,
                            kCapScopeWavefront, true, shared_warp_base,
                            shared_accum);
        if (threadIdx.x == 0) {
            sc_pos_push(qs, sc_d_pos(qs, score), sc_d_pos_size(qs, score), d_range);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        sp_reset(qs);
        core_next_m_sparsify(qs, scoring, query_len, graph, score, v);
    }
    __syncthreads();
    {
        const Range m_range =
            config1_densify(qs, Cell::Matrix::M, v, qs.bs_m_wf, qs.bs_m_wf_size,
                            kBeyondScopeCapacity, kCapBeyondScope, false,
                            shared_warp_base, shared_accum);
        if (threadIdx.x == 0) {
            sc_pos_push(qs, sc_m_pos(qs, score), sc_m_pos_size(qs, score), m_range);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        sp_reset(qs);

        const int32_t v_pos = vd_get_id(qs, v);
        if (!qs.capacity_exceeded && v_pos >= 0 &&
            sc_m_pos_size(qs, score) > v_pos) {
            const Range cells_range = sc_m_pos(qs, score)[v_pos];
            shared_range_start = cells_range.start;
            shared_range_end = cells_range.end;
        }
    }
    __syncthreads();

    config1_extend_and_consume_m_cells(qs, query, query_len, graph, score,
                                       shared_range_start, shared_range_end,
                                       shared_m_cells, shared_m_valid,
                                       block_end, block_end_cell);
    __syncthreads();
}

/**
 * @brief Block-parallel vd_expand + vd_compact.
 *
 * Both walk the active vertices and touch only vertex `a`'s own segment arrays
 * and sizes, so the vertices are independent and one thread can own each. The
 * two passes are fused per vertex for the same reason: nothing vertex `a` reads
 * is written by any other vertex. Neither pass changes vd_num_active, so the
 * loop bound is stable.
 */
__device__ void config1_expand_and_compact(QueryState &qs,
                                           int &shared_num_active) {
    const int num_active = qs.vd_num_active;
    const int g = qs.vd_gape;
    for (int a = threadIdx.x; a < num_active; a += blockDim.x) {
        vd_expand_vec(qs.vd_m_invalid[a], qs.vd_m_invalid_size[a], g, g);
        vd_expand_vec(qs.vd_i_invalid[a], qs.vd_i_invalid_size[a], g, g);
        vd_expand_vec(qs.vd_d_invalid[a], qs.vd_d_invalid_size[a], g, g);
        qs.vd_m_invalid_size[a] =
            vd_compact_vec(qs.vd_m_invalid[a], qs.vd_m_invalid_size[a], g, g);
        qs.vd_i_invalid_size[a] =
            vd_compact_vec(qs.vd_i_invalid[a], qs.vd_i_invalid_size[a], g, g);
        qs.vd_d_invalid_size[a] =
            vd_compact_vec(qs.vd_d_invalid[a], qs.vd_d_invalid_size[a], g, g);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        shared_num_active = vd_num_active_vertices(qs);
    }
    __syncthreads();
}

__device__ void config1_compute_new_wave(QueryState &qs,
                                         const AlignScoring &scoring,
                                         const char *query,
                                         int32_t query_len,
                                         const GraphCsrView &graph,
                                         int32_t score,
                                         int &shared_num_active,
                                         int &shared_vertex,
                                         int64_t &shared_range_start,
                                         int64_t &shared_range_end,
                                         Cell *shared_m_cells,
                                         int *shared_m_valid,
                                         Config1ICandidateRanges &shared_i_ranges,
                                         Cell *shared_i_candidates,
                                         int *shared_i_valid,
                                         int &shared_i_count,
                                         int32_t *shared_warp_base,
                                         int32_t &shared_accum,
                                         int &block_end,
                                         Cell &block_end_cell) {
    config1_expand_and_compact(qs, shared_num_active);

    for (int32_t l = 0; l < shared_num_active; ++l) {
        if (threadIdx.x == 0) {
            shared_vertex = vd_get_vertex_id(qs, l);
        }
        __syncthreads();
        config1_process_vertex(qs, scoring, query, query_len, graph, score,
                               shared_vertex, shared_range_start,
                               shared_range_end, shared_m_cells,
                               shared_m_valid, shared_i_ranges,
                               shared_i_candidates, shared_i_valid,
                               shared_i_count, shared_warp_base, shared_accum,
                               block_end, block_end_cell);
    }
}

__device__ void align_one_config1(QueryState &qs, const AlignScoring &scoring,
                                  const char *query, int32_t query_len,
                                  const GraphCsrView &graph,
                                  int32_t start_node,
                                  int32_t start_offset,
                                  AlignResult &result,
                                  int &block_continue,
                                  int &block_end,
                                  int &block_score,
                                  int &shared_num_active,
                                  int &shared_vertex,
                                  int64_t &shared_range_start,
                                  int64_t &shared_range_end,
                                  Cell *shared_m_cells,
                                  int *shared_m_valid,
                                  Config1ICandidateRanges &shared_i_ranges,
                                  Cell *shared_i_candidates,
                                  int *shared_i_valid,
                                  int &shared_i_count,
                                  int32_t *shared_warp_base,
                                  int32_t &shared_accum,
                                  Cell &block_end_cell) {
    if (threadIdx.x == 0) {
        qs.capacity_exceeded = false;
        int32_t max_diag = 0;
        for (int32_t v = 0; v < graph.num_vertices; ++v) {
            const int32_t n = vertex_len(graph, v);
            if (n > max_diag) {
                max_diag = n;
            }
        }
        sp_init(qs, -query_len, max_diag);
        sc_init(qs, scoring.nscores);
        vd_init(qs, scoring.gapo, scoring.gape, scoring.nscores,
                graph.num_vertices);
        sc_new_alignment(qs);
        bs_new_alignment(qs);
        vd_new_alignment(qs);

        block_score = 0;
        block_end = 0;
        block_end_cell = Cell{-1, -1, -1, -1, Cell::Matrix::None};

        sc_new_score(qs, block_score);
        Cell init_condition;
        init_condition.offset = 0;
        init_condition.vertex_id = start_node;
        init_condition.diag = start_offset;
        init_condition.prev_pos = -1;
        init_condition.from_matrix = Cell::Matrix::None;

        bs_push_back(qs, qs.bs_m_jumps_wf, qs.bs_m_jumps_wf_size,
                     init_condition);
        vd_activate_vertex(qs, start_node);
        vd_jumps_push(qs, vd_m_jumps(qs, start_node, 0),
                      vd_m_jumps_size(qs, start_node, 0), 0);
    }
    __syncthreads();

    while (true) {
        if (threadIdx.x == 0) {
            block_continue = (block_end == 0 && !qs.capacity_exceeded) ? 1 : 0;
        }
        __syncthreads();
        const int continue_now = block_continue;
        __syncthreads();
        if (continue_now == 0) {
            break;
        }

        if (threadIdx.x == 0 && block_score == 0) {
            bool end = block_end != 0;
            Cell end_cell = block_end_cell;
            core_extend_diagonal(qs, query, query_len, graph, block_score,
                                 qs.bs_m_jumps_wf[0], qs.bs_m_jumps_wf[0], 0,
                                 Cell::Matrix::MJumps, end, end_cell);
            block_end = end ? 1 : 0;
            block_end_cell = end_cell;
        }
        __syncthreads();

        config1_compute_new_wave(qs, scoring, query, query_len, graph,
                                 block_score, shared_num_active, shared_vertex,
                                 shared_range_start, shared_range_end,
                                 shared_m_cells, shared_m_valid,
                                 shared_i_ranges, shared_i_candidates,
                                 shared_i_valid, shared_i_count,
                                 shared_warp_base, shared_accum,
                                 block_end, block_end_cell);

        if (threadIdx.x == 0) {
            ++block_score;
            sc_new_score(qs, block_score);
            vd_new_score(qs, block_score);
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        --block_score;
        result.score = block_score;
        result.end_vertex_id = block_end_cell.vertex_id;
        result.end_offset = block_end_cell.offset;
        result.end_diag = block_end_cell.diag;
        result.end_prev_pos = block_end_cell.prev_pos;
        result.end_from_matrix = static_cast<int8_t>(block_end_cell.from_matrix);
        result.reached_end = block_end != 0 ? 1 : 0;
        result.capacity_exceeded = qs.capacity_exceeded ? 1 : 0;
        result.reserved = 0;
    }
    __syncthreads();
}

__global__ void theseus_align_batch_config1_kernel(BatchView batch, GraphCsrView graph,
                                                   const int32_t *start_node_ids,
                                                   const int32_t *start_offsets,
                                                   AlignScoring scoring,
                                                   QueryState *states,
                                                   AlignResult *results) {
    const int32_t query_id = blockIdx.x;
    const int32_t tid = threadIdx.x;
    if (query_id >= batch.num_seqs) {
        return;
    }

    QueryState *state = &states[query_id];

    __shared__ int block_continue;
    __shared__ int block_end;
    __shared__ int block_score;
    __shared__ int shared_num_active;
    __shared__ int shared_vertex;
    __shared__ int64_t shared_range_start;
    __shared__ int64_t shared_range_end;
    __shared__ Config1ICandidateRanges shared_i_ranges;
    __shared__ int shared_i_count;
    __shared__ int32_t shared_warp_base[kConfig1MaxWarps];
    __shared__ int32_t shared_accum;
    __shared__ Cell block_end_cell;

    // One tile per staging buffer, sized at launch on blockDim.x. Cell first
    // because it is the most strictly aligned member; the layout must match
    // config1_shared_bytes().
    extern __shared__ unsigned char config1_smem[];
    Cell *shared_i_candidates = reinterpret_cast<Cell *>(config1_smem);
    Cell *shared_m_cells = shared_i_candidates + blockDim.x;
    int *shared_i_valid = reinterpret_cast<int *>(shared_m_cells + blockDim.x);
    int *shared_m_valid = shared_i_valid + blockDim.x;

    if (tid == 0) {
        block_continue = 0;
        block_end = 0;
        block_score = 0;
        shared_num_active = 0;
        shared_vertex = 0;
        shared_range_start = 0;
        shared_range_end = 0;
        shared_i_count = 0;
        block_end_cell = Cell{-1, -1, -1, -1, Cell::Matrix::None};
    }

    __syncthreads();

    const int32_t begin = batch.offsets[query_id];
    const int32_t end = batch.offsets[query_id + 1];
    align_one_config1(*state, scoring, batch.chars + begin, end - begin, graph,
                      start_node_ids[query_id], start_offsets[query_id],
                      results[query_id], block_continue, block_end, block_score,
                      shared_num_active, shared_vertex, shared_range_start,
                      shared_range_end, shared_m_cells, shared_m_valid,
                      shared_i_ranges, shared_i_candidates, shared_i_valid,
                      shared_i_count, shared_warp_base, shared_accum,
                      block_end_cell);
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

const TimingReport &last_timing() { return g_last_timing; }

Status align_batch(const BatchView &batch,
                   const DeviceGraph *graph,
                   const int32_t *start_node_ids,
                   const int32_t *start_offsets,
                   AlignScoring scoring,
                   AlignOptions options,
                   AlignResult *out_results,
                   void *out_query_states,
                   int32_t *out_seq_lengths) {
    clear_error();
    g_last_timing = TimingReport{};

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
    cudaEvent_t ev_start = nullptr;
    cudaEvent_t ev_h2d = nullptr;
    cudaEvent_t ev_kernel = nullptr;
    cudaEvent_t ev_d2h = nullptr;
    int blocks = 0;
    const int32_t threads_per_block =
        (options.threads_per_block == 64 || options.threads_per_block == 128 ||
         options.threads_per_block == 256)
            ? options.threads_per_block
            : 128;
    // Declared with the rest before the first `goto cleanup`: jumping over an
    // initialisation is ill-formed.
    const size_t config1_smem_bytes = config1_shared_bytes(threads_per_block);

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

    cudaEventCreate(&ev_start);
    cudaEventCreate(&ev_h2d);
    cudaEventCreate(&ev_kernel);
    cudaEventCreate(&ev_d2h);
    if (ev_start != nullptr) {
        cudaEventRecord(ev_start);
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

    if (ev_h2d != nullptr) {
        cudaEventRecord(ev_h2d);
    }

    blocks = (batch.num_seqs + threads_per_block - 1) / threads_per_block;

    seq_length_kernel<<<blocks, threads_per_block>>>(d_offsets, batch.num_seqs, d_lengths);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_error("seq_length_kernel launch", err);
        status = Status::CudaError;
        goto cleanup;
    }

    device_batch = BatchView{d_chars, d_offsets, batch.num_seqs};
    if (options.config == GpuConfig::Config1) {
        theseus_align_batch_config1_kernel<<<batch.num_seqs, threads_per_block,
                                             config1_smem_bytes>>>(
            device_batch, graph->view, d_start_node_ids, d_start_offsets, scoring,
            d_states, d_results);
    } else {
        align_kernel<<<blocks, threads_per_block>>>(device_batch, graph->view,
                                                    d_start_node_ids, d_start_offsets,
                                                    scoring, d_states, d_results);
    }
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
    if (ev_kernel != nullptr) {
        cudaEventRecord(ev_kernel);
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
    if (ev_d2h != nullptr) {
        cudaEventRecord(ev_d2h);
        cudaEventSynchronize(ev_d2h);
    }
    if (ev_start != nullptr && ev_h2d != nullptr && ev_kernel != nullptr && ev_d2h != nullptr) {
        cudaEventElapsedTime(&g_last_timing.h2d_ms, ev_start, ev_h2d);
        cudaEventElapsedTime(&g_last_timing.kernel_ms, ev_h2d, ev_kernel);
        cudaEventElapsedTime(&g_last_timing.d2h_ms, ev_kernel, ev_d2h);
        cudaEventElapsedTime(&g_last_timing.end_to_end_ms, ev_start, ev_d2h);
    }

cleanup:
    cudaFree(d_chars);
    cudaFree(d_offsets);
    cudaFree(d_start_node_ids);
    cudaFree(d_start_offsets);
    cudaFree(d_results);
    cudaFree(d_states);
    cudaFree(d_lengths);
    if (ev_start != nullptr) cudaEventDestroy(ev_start);
    if (ev_h2d != nullptr) cudaEventDestroy(ev_h2d);
    if (ev_kernel != nullptr) cudaEventDestroy(ev_kernel);
    if (ev_d2h != nullptr) cudaEventDestroy(ev_d2h);
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
