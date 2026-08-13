/**
 * @file align_gpu.cu
 * @brief CUDA backend for the Theseus proxy aligner.
 *
 * One block owns one query for the whole alignment. Inside the block the phases
 * that are independent per element run across the threads (the invalid-segment
 * aging, the I candidates, the densify of I/D/M, the LCP of the M cells); the
 * phases whose append order defines prev_pos, which is what the goldens compare,
 * stay on thread 0. The per-vertex recurrences it calls live in
 * gpu/align_core.h, shared with the CPU flattening.
 */

#include "gpu/align_gpu.h"
#include "query_state.h"
#include "gpu/align_core.h"

#include <cuda_runtime.h>

#include <cstdio>
#include <vector>

namespace theseus {
namespace gpu {

namespace {

char g_last_error[256] = {0};
TimingReport g_last_timing;

/**
 * @brief Bytes of dynamic shared memory one block needs.
 *
 * The staging buffers hold exactly one tile, so the requirement scales with the
 * block size instead of with kScopeWavefrontCapacity. The layout here must match
 * the partitioning at the top of theseus_align_batch_kernel.
 *
 * One Cell tile, not two: the M cells are no longer staged in shared memory
 * because the thread that extends a cell writes it straight back to bs_m_wf.
 */
size_t kernel_shared_bytes(int32_t threads_per_block) {
    return static_cast<size_t>(threads_per_block) *
           (sizeof(Cell) + 2 * sizeof(int));
}

/**
 * @brief One contiguous run of candidates a sparsify half folds into the
 * ScratchPad, described so that any thread can rebuild candidate @p l alone.
 *
 * The three core_sparsify_* helpers differ only in where the source index comes
 * from and whether they overwrite prev_pos/from_matrix, so one description
 * covers all of them:
 *
 *   core_sparsify_indel  dense run, keeps the source cell's prev_pos/from_matrix
 *   core_sparsify_m      dense run, rewrites them to (absolute index, M)
 *   core_sparsify_jumps  indirect run through a jump-position list, rewrites them
 */
struct SparsifyPart {
    const Cell *wf;            // source wavefront
    const int64_t *positions;  // jump positions, null for a dense run
    int64_t start;             // first index of a dense run
    int32_t count;
    int32_t offset_increase;
    int32_t shift_factor;
    bool rewrite_prev;
    Cell::Matrix from_matrix;
};

/** @brief The runs one sparsify half visits, in the order it visits them. */
struct SparsifyPlan {
    SparsifyPart parts[4];
    int32_t nparts;
    int32_t total;
    int32_t query_len;
    int32_t upper_bound;
};

struct ICandidateRanges {
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
 * @brief One block per sequence, the same geometry the alignment kernel uses: a
 * block owns a query and its wavefronts for the whole alignment.
 *
 * For now a block only reports the length of its own sequence, which is enough
 * to prove the offsets survived the upload intact.
 */
__global__ void seq_length_kernel(const int32_t *offsets, int32_t num_seqs,
                                  int32_t *out_seq_lengths) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ntx = static_cast<int32_t>(blockDim.x);
    const int32_t bx = static_cast<int32_t>(blockIdx.x);
    const int32_t tid = bx * ntx + tx;
    if (tid >= num_seqs) {
        return;
    }
    out_seq_lengths[tid] = offsets[tid + 1] - offsets[tid];
}

__device__ Range empty_range() {
    return Range{0, 0};
}

__device__ int32_t range_len(Range range) {
    return static_cast<int32_t>(range.end - range.start);
}

__device__ ICandidateRanges prepare_i_candidate_ranges(
                                                       QueryState &qs, const AlignScoring &scoring,
                                                       const GraphCsrView &graph, int32_t score, int32_t v) {
    ICandidateRanges ranges;
    ranges.i_dense = empty_range();
    ranges.i_jumps_size = 0;
    ranges.m_dense = empty_range();
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

    ranges.total = range_len(ranges.i_dense) + ranges.i_jumps_size +
                   range_len(ranges.m_dense) + ranges.m_jumps_size;
    return ranges;
}

__device__ bool make_i_candidate(QueryState &qs,
                                 const ICandidateRanges &ranges,
                                 int32_t idx,
                                 int32_t query_len,
                                 Cell &candidate) {
    const int32_t i_dense_count = range_len(ranges.i_dense);
    const int32_t i_jumps_begin = i_dense_count;
    const int32_t m_dense_begin = i_jumps_begin + ranges.i_jumps_size;
    const int32_t m_jumps_begin = m_dense_begin + range_len(ranges.m_dense);

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

constexpr int32_t kMaxWarps = 8;  // 256 threads, the largest block allowed

/**
 * @brief Allocate one slot per thread whose @p flag is set, in thread order, and
 * return this thread's slot.
 *
 * The aggregation densify has always used, lifted out so the merge phases can
 * reuse it instead of introducing a second scheme: a warp ballot gives the
 * within-warp prefix, then thread 0 folds the (at most 8) warp counts into
 * exclusive bases. @p shared_running is read as the first free slot and left at
 * the first free slot after this call, so consecutive tiles keep numbering
 * where the previous one stopped. That is what makes the slots identical to the
 * positions a serial append in the same order would have produced.
 *
 * Every thread of the block must call it: it contains barriers.
 */
__device__ int32_t block_prefix_alloc(int flag, int32_t *shared_warp_base,
                                      int32_t &shared_running) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ntx = static_cast<int32_t>(blockDim.x);
    const int32_t lane = tx & 31;
    const int32_t warp = tx >> 5;
    const int32_t nwarps = (ntx + 31) >> 5;

    // Every thread reaches the ballot, including the ones with flag 0, so the
    // warp stays converged.
    const unsigned ballot = __ballot_sync(0xffffffffu, flag != 0);
    const int32_t warp_prefix = __popc(ballot & ((1u << lane) - 1u));
    if (lane == 0) {
        shared_warp_base[warp] = __popc(ballot);
    }
    __syncthreads();

    if (tx == 0) {
        int32_t acc = shared_running;
        for (int32_t w = 0; w < nwarps; ++w) {
            const int32_t count = shared_warp_base[w];
            shared_warp_base[w] = acc;   // exclusive base for this warp
            acc += count;
        }
        shared_running = acc;
    }
    __syncthreads();

    const int32_t slot = shared_warp_base[warp] + warp_prefix;
    __syncthreads();   // shared_warp_base is scratch again for the caller
    return slot;
}

/**
 * @brief Block-parallel replacement for the serial merge of one tile of staged
 * candidates into the ScratchPad.
 *
 * The serial loop is
 *
 *     Cell &cell = sp_access_alloc(qs, c.diag);
 *     if (cell.offset < c.offset) cell = c;
 *
 * over the tile in index order, and it has two observable effects, not one.
 *
 * **The winner.** The comparison is strict, so a later candidate with an equal
 * offset never replaces an earlier one: the surviving cell is the candidate with
 * the largest offset and, among equals, the smallest index. Folding the tile to
 * that argmax and then comparing it once against the cell already there gives
 * the same result, because everything already in the cell came from a smaller
 * index and so wins its own ties.
 *
 * **The order of sp_diags.** access_alloc appends the diagonal on first touch,
 * and densify walks sp_diags in that order, so the append order decides the
 * order of the dense wavefront, hence the Ranges, hence the prev_pos of later
 * waves. That is the part a plain "max over offset" would silently get wrong.
 * Here the first toucher is the smallest index among the tile's valid
 * candidates on that diagonal, and the appends are numbered with the same
 * prefix-sum allocator densify uses, so they land exactly where the serial
 * appends would have.
 *
 * The two reductions are done with an O(tile) scan per thread over the staged
 * tile rather than with atomics on an array indexed by diagonal: such an array
 * would need one entry per diagonal over kScratchpadSpan (52 224), which does
 * not fit in shared memory and would cost more to clear than the merge costs to
 * run.
 *
 * The one thing this cannot reproduce is a candidate whose offset is negative.
 * Serially such a candidate leaves the cell at -1, so the *next* candidate on
 * that diagonal appends it a second time; here it would be appended once. Cell
 * offsets are query positions and are never negative, so the case is
 * unreachable — and it is checked rather than assumed: cap_fail fires if it
 * ever happens, instead of the output quietly diverging.
 *
 * Every thread of the block must call it: it contains barriers.
 */
__device__ void merge_candidate_tile(QueryState &qs,
                                     const Cell *shared_cells,
                                     const int *shared_valid,
                                     int32_t tile_len,
                                     int32_t *shared_warp_base,
                                     int32_t &shared_accum) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const bool mine = tx < tile_len && shared_valid[tx] != 0;
    const int32_t my_diag = mine ? shared_cells[tx].diag : 0;
    const int32_t my_off = mine ? shared_cells[tx].offset : 0;

    // Same guard as sp_access_alloc: out of window is a capacity failure, and
    // the candidate neither appends nor writes.
    const int32_t idx = my_diag - qs.sp_min_diag;
    const bool in_range = mine && idx >= 0 && idx < kScratchpadSpan;
    if (mine && !in_range) {
        cap_fail(qs, kCapScratchpadSpan, idx < 0 ? -idx : idx + 1,
                 kScratchpadSpan);
    }
    if (mine && my_off < 0) {
        // Unreachable (see above); never silent if it ever becomes reachable.
        cap_fail(qs, kCapScratchpadSpan, -1, kScratchpadSpan);
    }

    bool is_winner = in_range;
    bool is_first = in_range;
    if (in_range) {
        for (int32_t j = 0; j < tile_len; ++j) {
            if (j == tx || shared_valid[j] == 0 ||
                shared_cells[j].diag != my_diag) {
                continue;
            }
            const int32_t oj = shared_cells[j].offset;
            if (oj > my_off || (oj == my_off && j < tx)) {
                is_winner = false;
            }
            if (j < tx) {
                is_first = false;
            }
        }
    }

    // Read the resting state before any of this tile's writes: a diagonal is
    // appended only if the cell was still inactive, exactly as access_alloc
    // decides it.
    const bool needs_append = is_first && qs.sp_wf[idx].offset == -1;
    __syncthreads();

    if (tx == 0) {
        shared_accum = qs.sp_ndiags;
    }
    __syncthreads();
    const int32_t slot =
        block_prefix_alloc(needs_append ? 1 : 0, shared_warp_base, shared_accum);
    if (needs_append) {
        if (slot < kScratchpadSpan) {
            qs.sp_diags[slot] = my_diag;
        } else {
            cap_fail(qs, kCapScratchpadDiags, slot + 1, kScratchpadSpan);
        }
    }
    if (tx == 0) {
        qs.sp_ndiags =
            shared_accum < kScratchpadSpan ? shared_accum : kScratchpadSpan;
    }
    __syncthreads();

    // One winner per diagonal, so no two threads write the same cell.
    if (is_winner && qs.sp_wf[idx].offset < my_off) {
        qs.sp_wf[idx] = shared_cells[tx];
    }
    __syncthreads();
}

__device__ void plan_add(SparsifyPlan &plan, const Cell *wf,
                         const int64_t *positions, int64_t start, int32_t count,
                         int32_t offset_increase, int32_t shift_factor,
                         bool rewrite_prev, Cell::Matrix from_matrix) {
    if (count <= 0) {
        return;   // an empty run contributes nothing and no index space
    }
    SparsifyPart &part = plan.parts[plan.nparts];
    part.wf = wf;
    part.positions = positions;
    part.start = start;
    part.count = count;
    part.offset_increase = offset_increase;
    part.shift_factor = shift_factor;
    part.rewrite_prev = rewrite_prev;
    part.from_matrix = from_matrix;
    ++plan.nparts;
    plan.total += count;
}

/**
 * @brief The runs core_next_d_sparsify walks, in its order.
 *
 * Mirrors that function exactly, including that pos_prev_m_scope is computed
 * before the sign test and only used inside it.
 */
__device__ SparsifyPlan prepare_d_sparsify_plan(QueryState &qs,
                                                const AlignScoring &scoring,
                                                int32_t query_len,
                                                const GraphCsrView &graph,
                                                int32_t score, int32_t v) {
    SparsifyPlan plan;
    plan.nparts = 0;
    plan.total = 0;
    plan.query_len = query_len;
    plan.upper_bound = vertex_len(graph, v);

    const int32_t pos_prev_m = score - (scoring.gapo + scoring.gape);
    const int32_t pos_prev_d = score - scoring.gape;
    const int32_t pos_prev_m_scope = vd_get_pos(qs, pos_prev_m);
    const int32_t v_id = vd_get_id(qs, v);

    if (pos_prev_d >= 0 && sc_d_pos_size(qs, pos_prev_d) > v_id) {
        const Range r = sc_d_pos(qs, pos_prev_d)[v_id];
        plan_add(plan, sc_d_wf(qs, pos_prev_d), nullptr, r.start,
                 static_cast<int32_t>(r.end - r.start), 1, -1, false,
                 Cell::Matrix::None);
    }
    if (pos_prev_m >= 0) {
        if (sc_m_pos_size(qs, pos_prev_m) > v_id) {
            const Range r = sc_m_pos(qs, pos_prev_m)[v_id];
            plan_add(plan, qs.bs_m_wf, nullptr, r.start,
                     static_cast<int32_t>(r.end - r.start), 1, -1, true,
                     Cell::Matrix::M);
        }
        plan_add(plan, qs.bs_m_jumps_wf, vd_m_jumps(qs, v, pos_prev_m_scope), 0,
                 vd_m_jumps_size(qs, v, pos_prev_m_scope), 1, -1, true,
                 Cell::Matrix::MJumps);
    }
    return plan;
}

/** @brief The runs core_next_m_sparsify walks, in its order. */
__device__ SparsifyPlan prepare_m_sparsify_plan(QueryState &qs,
                                                const AlignScoring &scoring,
                                                int32_t query_len,
                                                const GraphCsrView &graph,
                                                int32_t score, int32_t v) {
    SparsifyPlan plan;
    plan.nparts = 0;
    plan.total = 0;
    plan.query_len = query_len;
    plan.upper_bound = vertex_len(graph, v);

    const int32_t pos_prev_m = score - scoring.mism;
    const int32_t pos_prev_m_scope = vd_get_pos(qs, pos_prev_m);
    const int32_t v_id = vd_get_id(qs, v);

    if (sc_d_pos_size(qs, score) > v_id) {
        const Range r = sc_d_pos(qs, score)[v_id];
        plan_add(plan, sc_d_wf(qs, score), nullptr, r.start,
                 static_cast<int32_t>(r.end - r.start), 0, 0, false,
                 Cell::Matrix::None);
    }
    if (sc_i_pos_size(qs, score) > v_id) {
        const Range r = sc_i_pos(qs, score)[v_id];
        plan_add(plan, sc_i_wf(qs, score), nullptr, r.start,
                 static_cast<int32_t>(r.end - r.start), 0, 0, false,
                 Cell::Matrix::None);
    }
    if (pos_prev_m >= 0) {
        if (sc_m_pos_size(qs, pos_prev_m) > v_id) {
            const Range r = sc_m_pos(qs, pos_prev_m)[v_id];
            plan_add(plan, qs.bs_m_wf, nullptr, r.start,
                     static_cast<int32_t>(r.end - r.start), 1, 0, true,
                     Cell::Matrix::M);
        }
        plan_add(plan, qs.bs_m_jumps_wf, vd_m_jumps(qs, v, pos_prev_m_scope), 0,
                 vd_m_jumps_size(qs, v, pos_prev_m_scope), 1, 0, true,
                 Cell::Matrix::MJumps);
    }
    return plan;
}

/**
 * @brief Rebuild candidate @p idx of a plan, and say whether it passes the
 * filter every core_sparsify_* applies before touching the ScratchPad.
 */
__device__ bool make_sparsify_candidate(const SparsifyPlan &plan, int32_t idx,
                                        Cell &out) {
    int32_t local = idx;
    for (int32_t p = 0; p < plan.nparts; ++p) {
        const SparsifyPart &part = plan.parts[p];
        if (local < part.count) {
            const int64_t pos =
                part.positions != nullptr ? part.positions[local]
                                          : part.start + local;
            Cell cell = part.wf[pos];
            if (part.rewrite_prev) {
                cell.prev_pos = pos;
                cell.from_matrix = part.from_matrix;
            }
            cell.diag += part.shift_factor;
            cell.offset += part.offset_increase;
            out = cell;
            const int32_t new_col = cell.offset + cell.diag;
            return cell.offset <= plan.query_len && new_col <= plan.upper_bound;
        }
        local -= part.count;
    }
    out = Cell{-1, -1, -1, -1, Cell::Matrix::None};
    return false;
}

/**
 * @brief Block-wide max, returned to every thread.
 *
 * max is associative and commutative and the values are exact int32, so the
 * result does not depend on the reduction order: this is the same number the
 * serial scan produced. Shaped like the prefix sum in densify (warp reduction,
 * then a fold over the at most 8 warp results on thread 0) and it borrows the
 * same @p shared_warp_base scratch, which nothing else is using at the point in
 * align_one where this runs.
 *
 * Every thread of the block must call it: it contains barriers.
 */
__device__ int32_t block_reduce_max(int32_t value, int32_t *shared_warp_base) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ntx = static_cast<int32_t>(blockDim.x);
    const int32_t lane = tx & 31;
    const int32_t warp = tx >> 5;
    const int32_t nwarps = (ntx + 31) >> 5;

    // ntx is 64, 128 or 256, so every warp is full and the whole mask is
    // the right one.
    for (int32_t delta = 16; delta > 0; delta >>= 1) {
        const int32_t other = __shfl_down_sync(0xffffffffu, value, delta);
        if (other > value) {
            value = other;
        }
    }
    if (lane == 0) {
        shared_warp_base[warp] = value;
    }
    __syncthreads();

    if (tx == 0) {
        int32_t acc = shared_warp_base[0];
        for (int32_t w = 1; w < nwarps; ++w) {
            if (shared_warp_base[w] > acc) {
                acc = shared_warp_base[w];
            }
        }
        shared_warp_base[0] = acc;
    }
    __syncthreads();

    const int32_t result = shared_warp_base[0];
    __syncthreads();   // shared_warp_base is scratch again for the caller
    return result;
}

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
__device__ Range densify(QueryState &qs, Cell::Matrix matrix, int32_t v,
                         Cell *wf, int32_t &wf_size, int32_t capacity,
                         CapBuffer cap_buffer, bool track_peak,
                         int32_t *shared_warp_base, int32_t &shared_accum) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ntx = static_cast<int32_t>(blockDim.x);
    const int32_t ndiags = qs.sp_ndiags;   // uniform: densify never touches it
    const int32_t range_start = wf_size;   // read before thread 0 updates it

    if (tx == 0) {
        shared_accum = 0;
    }
    __syncthreads();

    for (int32_t tile = 0; tile < ndiags; tile += ntx) {
        const int32_t di = tile + tx;
        int32_t flag = 0;
        Cell value{-1, -1, -1, -1, Cell::Matrix::None};
        if (di < ndiags) {
            const int32_t diag = qs.sp_diags[di];
            if (vd_valid_diagonal(qs, matrix, v, diag)) {
                flag = 1;
                value = sp_at(qs, diag);
            }
        }

        const int32_t slot =
            block_prefix_alloc(flag, shared_warp_base, shared_accum);
        if (flag != 0) {
            const int32_t pos = range_start + slot;
            if (pos < capacity) {
                wf[pos] = value;
            }
        }
        __syncthreads();   // shared_warp_base is reused by the next tile
    }

    if (tx == 0) {
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

/**
 * @brief Block-parallel sp_reset.
 *
 * sp_diags holds each touched diagonal once (access_alloc appends only on the
 * first touch), so the entries address distinct cells: no two threads write the
 * same one, and they all write the same -1. The list is only emptied after the
 * barrier, so nobody clears a diagonal another thread has not read yet.
 *
 * Every thread of the block must call it: it contains barriers.
 */
__device__ void sp_reset_block(QueryState &qs) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ntx = static_cast<int32_t>(blockDim.x);
    const int32_t ndiags = qs.sp_ndiags;   // uniform: nothing writes it here
    for (int32_t i = tx; i < ndiags; i += ntx) {
        sp_reset_one(qs, i);
    }
    __syncthreads();
    if (tx == 0) {
        qs.sp_ndiags = 0;
    }
    __syncthreads();
}

/**
 * @brief Fold a whole sparsify plan into the ScratchPad, one tile at a time.
 *
 * Tiled for the same reason the I candidates are: the staging buffers are sized
 * on the block, not on the candidate space, and the ScratchPad state carries
 * over from one tile to the next so the index order across tiles is preserved.
 *
 * Every thread of the block must call it: it contains barriers.
 */
__device__ void run_sparsify_plan(QueryState &qs, const SparsifyPlan &plan,
                                  Cell *shared_cells, int *shared_valid,
                                  int32_t *shared_warp_base,
                                  int32_t &shared_accum) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ntx = static_cast<int32_t>(blockDim.x);
    // Uniform: the plan lives in shared memory, written by thread 0 before the
    // barrier that got us here.
    const int32_t count = plan.total;

    for (int32_t tile_start = 0; tile_start < count; tile_start += ntx) {
        const int32_t idx = tile_start + tx;
        if (idx < count) {
            Cell candidate{-1, -1, -1, -1, Cell::Matrix::None};
            const bool valid = make_sparsify_candidate(plan, idx, candidate);
            shared_cells[tx] = candidate;
            shared_valid[tx] = valid ? 1 : 0;
        } else {
            shared_valid[tx] = 0;
        }
        __syncthreads();

        const int32_t remaining = count - tile_start;
        const int32_t lanes = remaining < ntx ? remaining : ntx;
        merge_candidate_tile(qs, shared_cells, shared_valid, lanes,
                             shared_warp_base, shared_accum);
    }
}

__device__ Range finish_i_wavefront(QueryState &qs, int32_t score,
                                    int32_t v,
                                    int32_t *shared_warp_base,
                                    int32_t &shared_accum) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const Range new_range =
        densify(qs, Cell::Matrix::I, v, sc_i_wf(qs, score),
                sc_i_wf_size(qs, score), kScopeWavefrontCapacity,
                kCapScopeWavefront, true, shared_warp_base, shared_accum);
    if (tx == 0) {
        sc_pos_push(qs, sc_i_pos(qs, score), sc_i_pos_size(qs, score), new_range);
    }
    __syncthreads();
    return new_range;
}

/**
 * @brief Build the I candidates for one vertex and merge them into the scratchpad.
 *
 * The candidate space is consumed one tile of ntx elements at a time, the
 * same way extend_and_consume_m_cells consumes the M range. Staging a
 * tile rather than the whole candidate space is what lets the shared buffers be
 * sized on the block instead of on kScopeWavefrontCapacity, and it removes the
 * capacity ceiling that used to abort the alignment when a vertex produced more
 * than kScopeWavefrontCapacity candidates.
 *
 * Candidates can collide on a diagonal, so the merge is not a plain scatter:
 * merge_candidate_tile reproduces both effects the serial order had, the winner
 * and the append order of sp_diags. Tiling preserves the index order exactly,
 * and the merge is done per tile because the ScratchPad state carries over from
 * one tile to the next.
 */
__device__ void generate_and_merge_i_candidates(
                                                QueryState &qs, const AlignScoring &scoring, const char *query,
                                                int32_t query_len,
                                                const GraphCsrView &graph, int32_t score, int32_t v,
                                                ICandidateRanges &shared_i_ranges,
                                                Cell *shared_i_candidates, int *shared_i_valid,
                                                int &shared_i_count, int32_t *shared_warp_base, int32_t &shared_accum,
                                                int &block_end, Cell &block_end_cell) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ntx = static_cast<int32_t>(blockDim.x);

    if (tx == 0) {
        shared_i_ranges = prepare_i_candidate_ranges(qs, scoring, graph,
                                                     score, v);
        shared_i_count = shared_i_ranges.total;
    }
    __syncthreads();

    // Uniform across the block: written by thread 0 before the barrier above, so
    // every thread runs the same number of tiles and reaches the same barriers.
    const int32_t count = shared_i_count;

    for (int32_t tile_start = 0; tile_start < count; tile_start += ntx) {
        const int32_t idx = tile_start + tx;
        if (idx < count) {
            Cell candidate{-1, -1, -1, -1, Cell::Matrix::None};
            const bool valid = make_i_candidate(qs, shared_i_ranges, idx,
                                                query_len, candidate);
            shared_i_candidates[tx] = candidate;
            shared_i_valid[tx] = valid ? 1 : 0;
        } else {
            shared_i_valid[tx] = 0;
        }
        __syncthreads();

        const int32_t remaining = count - tile_start;
        const int32_t lanes = remaining < ntx ? remaining : ntx;
        merge_candidate_tile(qs, shared_i_candidates, shared_i_valid, lanes,
                             shared_warp_base, shared_accum);
    }

    const Range new_range = finish_i_wavefront(qs, score, v,
                                               shared_warp_base,
                                               shared_accum);
    if (tx == 0) {
        // Same tail as the CPU's next_I: the I cells that just reached the last
        // column of this vertex open jumps into its neighbours.
        // core_store_m_jump can hit the end condition, so the block's end state
        // has to travel through here too.
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

__device__ void extend_and_consume_m_cells(QueryState &qs,
                                           const char *query,
                                           int32_t query_len,
                                           const GraphCsrView &graph,
                                           int32_t score,
                                           int64_t range_start,
                                           int64_t range_end,
                                           int *shared_m_valid,
                                           int &block_end,
                                           Cell &block_end_cell) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ntx = static_cast<int32_t>(blockDim.x);

    for (int64_t chunk_start = range_start; chunk_start < range_end;
         chunk_start += ntx) {
        const int64_t idx = chunk_start + tx;
        if (idx < range_end) {
            Cell cell = qs.bs_m_wf[idx];
            int32_t j = cell.diag + cell.offset;
            core_lcp(query, query_len, graph, cell.vertex_id, cell.offset, j);
            // The extending thread writes its own cell back. The serial loop
            // below used to do it lane by lane, so a cell was visible in
            // bs_m_wf only once the loop reached it; writing the whole tile up
            // front is invisible because nothing that loop calls reads
            // bs_m_wf. core_store_m_jump and the core_extend_diagonal
            // recursion under it push into bs_m_jumps_wf and bs_i_jumps_wf
            // only, and each iteration reads just the cell it is on.
            qs.bs_m_wf[idx] = cell;
            shared_m_valid[tx] = 1;
        } else {
            shared_m_valid[tx] = 0;
        }
        __syncthreads();

        if (tx == 0) {
            bool end = block_end != 0;
            Cell end_cell = block_end_cell;
            for (int32_t lane = 0; lane < ntx; ++lane) {
                if (shared_m_valid[lane] == 0) {
                    continue;
                }
                const int64_t cell_idx = chunk_start + lane;
                Cell &cell = qs.bs_m_wf[cell_idx];
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

__device__ void process_vertex(QueryState &qs,
                               const AlignScoring &scoring,
                               const char *query,
                               int32_t query_len,
                               const GraphCsrView &graph,
                               int32_t score,
                               int32_t v,
                               int64_t &shared_range_start,
                               int64_t &shared_range_end,
                               int *shared_m_valid,
                               ICandidateRanges &shared_i_ranges,
                               SparsifyPlan &shared_sparsify_plan,
                               Cell *shared_i_candidates,
                               int *shared_i_valid,
                               int &shared_i_count,
                               int32_t *shared_warp_base,
                               int32_t &shared_accum,
                               int &block_end,
                               Cell &block_end_cell) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);

    if (tx == 0) {
        shared_range_start = 0;
        shared_range_end = 0;

    }
    __syncthreads();

    generate_and_merge_i_candidates(qs, scoring, query, query_len, graph,
                                    score, v, shared_i_ranges,
                                    shared_i_candidates, shared_i_valid,
                                    shared_i_count, shared_warp_base,
                                    shared_accum, block_end,
                                    block_end_cell);

    // Reset, sparsify and densify are all block-wide now. Only the plan itself
    // is built on thread 0, and it is a handful of scalar reads.
    sp_reset_block(qs);
    if (tx == 0) {
        shared_sparsify_plan =
            prepare_d_sparsify_plan(qs, scoring, query_len, graph, score, v);
    }
    __syncthreads();
    run_sparsify_plan(qs, shared_sparsify_plan, shared_i_candidates,
                      shared_i_valid, shared_warp_base, shared_accum);
    {
        const Range d_range =
            densify(qs, Cell::Matrix::D, v, sc_d_wf(qs, score),
                    sc_d_wf_size(qs, score), kScopeWavefrontCapacity,
                    kCapScopeWavefront, true, shared_warp_base,
                    shared_accum);
        if (tx == 0) {
            sc_pos_push(qs, sc_d_pos(qs, score), sc_d_pos_size(qs, score), d_range);
        }
    }
    __syncthreads();

    sp_reset_block(qs);
    if (tx == 0) {
        shared_sparsify_plan =
            prepare_m_sparsify_plan(qs, scoring, query_len, graph, score, v);
    }
    __syncthreads();
    run_sparsify_plan(qs, shared_sparsify_plan, shared_i_candidates,
                      shared_i_valid, shared_warp_base, shared_accum);
    {
        const Range m_range =
            densify(qs, Cell::Matrix::M, v, qs.bs_m_wf, qs.bs_m_wf_size,
                    kBeyondScopeCapacity, kCapBeyondScope, false,
                    shared_warp_base, shared_accum);
        if (tx == 0) {
            sc_pos_push(qs, sc_m_pos(qs, score), sc_m_pos_size(qs, score), m_range);
        }
    }
    __syncthreads();

    sp_reset_block(qs);
    if (tx == 0) {
        const int32_t v_pos = vd_get_id(qs, v);
        if (!qs.capacity_exceeded && v_pos >= 0 &&
            sc_m_pos_size(qs, score) > v_pos) {
            const Range cells_range = sc_m_pos(qs, score)[v_pos];
            shared_range_start = cells_range.start;
            shared_range_end = cells_range.end;
        }
    }
    __syncthreads();

    extend_and_consume_m_cells(qs, query, query_len, graph, score,
                               shared_range_start, shared_range_end,
                               shared_m_valid,
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
__device__ void expand_and_compact(QueryState &qs,
                                   int &shared_num_active) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ntx = static_cast<int32_t>(blockDim.x);
    const int num_active = qs.vd_num_active;
    const int g = qs.vd_gape;
    for (int a = tx; a < num_active; a += ntx) {
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
    if (tx == 0) {
        shared_num_active = vd_num_active_vertices(qs);
    }
    __syncthreads();
}

__device__ void compute_new_wave(QueryState &qs,
                                 const AlignScoring &scoring,
                                 const char *query,
                                 int32_t query_len,
                                 const GraphCsrView &graph,
                                 int32_t score,
                                 int &shared_num_active,
                                 int &shared_vertex,
                                 int64_t &shared_range_start,
                                 int64_t &shared_range_end,
                                 int *shared_m_valid,
                                 ICandidateRanges &shared_i_ranges,
                                 SparsifyPlan &shared_sparsify_plan,
                                 Cell *shared_i_candidates,
                                 int *shared_i_valid,
                                 int &shared_i_count,
                                 int32_t *shared_warp_base,
                                 int32_t &shared_accum,
                                 int &block_end,
                                 Cell &block_end_cell) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);

    expand_and_compact(qs, shared_num_active);

    for (int32_t l = 0; l < shared_num_active; ++l) {
        if (tx == 0) {
            shared_vertex = vd_get_vertex_id(qs, l);
        }
        __syncthreads();
        process_vertex(qs, scoring, query, query_len, graph, score,
                       shared_vertex, shared_range_start,
                       shared_range_end, shared_m_valid, shared_i_ranges,
                       shared_sparsify_plan,
                       shared_i_candidates, shared_i_valid,
                       shared_i_count, shared_warp_base, shared_accum,
                       block_end, block_end_cell);
    }
}

__device__ void align_one(QueryState &qs, const AlignScoring &scoring,
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
                          int *shared_m_valid,
                          ICandidateRanges &shared_i_ranges,
                          SparsifyPlan &shared_sparsify_plan,
                          Cell *shared_i_candidates,
                          int *shared_i_valid,
                          int &shared_i_count,
                          int32_t *shared_warp_base,
                          int32_t &shared_accum,
                          int32_t &shared_span,
                          Cell &block_end_cell) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ntx = static_cast<int32_t>(blockDim.x);

    // The longest vertex of the graph, which sizes the ScratchPad window: a max
    // over the whole graph, so the block reduces it instead of thread 0 scanning
    // num_vertices on its own.
    int32_t local_max_diag = 0;
    for (int32_t v = tx; v < graph.num_vertices; v += ntx) {
        const int32_t n = vertex_len(graph, v);
        if (n > local_max_diag) {
            local_max_diag = n;
        }
    }
    const int32_t max_diag = block_reduce_max(local_max_diag, shared_warp_base);

    if (tx == 0) {
        qs.capacity_exceeded = false;
        // Scalar halves only; the block runs both clearing loops below.
        shared_span = sp_init_window(qs, -query_len, max_diag);
        sc_init(qs, scoring.nscores);
        vd_init_scalar(qs, scoring.gapo, scoring.gape, scoring.nscores,
                       graph.num_vertices);
        sc_new_alignment(qs);
        bs_new_alignment(qs);
        vd_new_alignment_scalar(qs);
    }
    __syncthreads();

    // Both bounds are uniform (shared_span written before the barrier,
    // graph.num_vertices a kernel argument), so every thread runs the same
    // number of iterations. The stores are independent and all write the same
    // value, so the result does not depend on which thread clears which entry.
    //
    // vd_init and vd_new_alignment clear the same prefix of vd_vertex_to_idx to
    // the same -1, one right after the other, so the two loops collapse into
    // this one pass with no change to the state either produced.
    const int32_t vd_fill = vd_map_fill_count(graph.num_vertices);
    for (int32_t i = tx; i < shared_span; i += ntx) {
        qs.sp_wf[i] = Cell{-1, -1, -1, -1, Cell::Matrix::None};
    }
    for (int32_t i = tx; i < vd_fill; i += ntx) {
        qs.vd_vertex_to_idx[i] = -1;
    }
    __syncthreads();

    if (tx == 0) {
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
        if (tx == 0) {
            block_continue = (block_end == 0 && !qs.capacity_exceeded) ? 1 : 0;
        }
        __syncthreads();
        const int continue_now = block_continue;
        __syncthreads();
        if (continue_now == 0) {
            break;
        }

        if (tx == 0 && block_score == 0) {
            bool end = block_end != 0;
            Cell end_cell = block_end_cell;
            core_extend_diagonal(qs, query, query_len, graph, block_score,
                                 qs.bs_m_jumps_wf[0], qs.bs_m_jumps_wf[0], 0,
                                 Cell::Matrix::MJumps, end, end_cell);
            block_end = end ? 1 : 0;
            block_end_cell = end_cell;
        }
        __syncthreads();

        compute_new_wave(qs, scoring, query, query_len, graph,
                         block_score, shared_num_active, shared_vertex,
                         shared_range_start, shared_range_end,
                         shared_m_valid,
                         shared_i_ranges, shared_sparsify_plan,
                         shared_i_candidates,
                         shared_i_valid, shared_i_count,
                         shared_warp_base, shared_accum,
                         block_end, block_end_cell);

        if (tx == 0) {
            ++block_score;
            sc_new_score(qs, block_score);
        }
        __syncthreads();

        // vd_new_score: one active vertex per thread. Each entry belongs to a
        // single vertex and they all get the same 0, so the result does not
        // depend on which thread clears which. block_score is uniform (thread 0
        // bumped it before the barrier) and nothing writes vd_num_active here.
        {
            const int pos = vd_new_score_slot(qs, block_score);
            for (int a = tx; a < qs.vd_num_active; a += ntx) {
                vd_new_score_one(qs, a, pos);
            }
        }
        __syncthreads();
    }

    if (tx == 0) {
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

__global__ void theseus_align_batch_kernel(BatchView batch, GraphCsrView graph,
                                           const int32_t *start_node_ids,
                                           const int32_t *start_offsets,
                                           AlignScoring scoring,
                                           QueryState *states,
                                           AlignResult *results) {
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ntx = static_cast<int32_t>(blockDim.x);
    const int32_t bx = static_cast<int32_t>(blockIdx.x);

    // One block per query: the block index is the query index.
    const int32_t query_id = bx;
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
    __shared__ ICandidateRanges shared_i_ranges;
    __shared__ SparsifyPlan shared_sparsify_plan;
    __shared__ int shared_i_count;
    __shared__ int32_t shared_warp_base[kMaxWarps];
    __shared__ int32_t shared_accum;
    __shared__ int32_t shared_span;
    __shared__ Cell block_end_cell;

    // One tile per staging buffer, sized at launch on ntx. Cell first
    // because it is the most strictly aligned member; the layout must match
    // kernel_shared_bytes().
    extern __shared__ unsigned char smem[];
    Cell *shared_i_candidates = reinterpret_cast<Cell *>(smem);
    int *shared_i_valid = reinterpret_cast<int *>(shared_i_candidates + ntx);
    int *shared_m_valid = shared_i_valid + ntx;

    if (tx == 0) {
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
    align_one(*state, scoring, batch.chars + begin, end - begin, graph,
              start_node_ids[query_id], start_offsets[query_id],
              results[query_id], block_continue, block_end, block_score,
              shared_num_active, shared_vertex, shared_range_start,
              shared_range_end, shared_m_valid,
              shared_i_ranges, shared_sparsify_plan, shared_i_candidates,
              shared_i_valid,
              shared_i_count, shared_warp_base, shared_accum, shared_span,
              block_end_cell);
}

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

__global__ void traceback_meta_kernel(const QueryState *states, int32_t count,
                                      TracebackMeta *metadata) {
    const int32_t i = static_cast<int32_t>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= count) return;
    const QueryState &state = states[i];
    metadata[i] = TracebackMeta{state.bs_m_wf_size,
                                state.bs_m_jumps_wf_size,
                                state.bs_i_jumps_wf_size,
                                state.sc_peak_wf,
                                state.cap_required,
                                state.cap_available,
                                static_cast<int8_t>(state.capacity_exceeded),
                                state.cap_reason,
                                {0, 0}};
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
    const int32_t tx = static_cast<int32_t>(threadIdx.x);
    const int32_t ntx = static_cast<int32_t>(blockDim.x);
    const int32_t bx = static_cast<int32_t>(blockIdx.x);

    // One block per vertex: the block index is the vertex id.
    const int32_t v = bx;
    if (v >= graph.num_vertices) {
        return;
    }

    const int32_t text_begin = graph.vertex_offsets[v];
    const int32_t text_end = graph.vertex_offsets[v + 1];
    for (int32_t i = text_begin + tx; i < text_end; i += ntx) {
        out_vertex_chars[i] = graph.vertex_chars[i];
    }

    const int32_t edge_begin = graph.edge_offsets[v];
    const int32_t edge_end = graph.edge_offsets[v + 1];
    for (int32_t e = edge_begin + tx; e < edge_end; e += ntx) {
        out_edge_targets[e] = graph.edge_targets[e];
        out_edge_overlaps[e] = graph.edge_overlaps[e];
    }

    if (tx == 0) {
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
};

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
    delete workspace;
}

const char *last_error() { return g_last_error; }

const TimingReport &last_timing() { return g_last_timing; }

Status align_batch(const BatchView &batch,
                   const DeviceGraph *graph,
                   DeviceWorkspace *workspace,
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
    if (graph == nullptr || workspace == nullptr) {
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

    char *d_chars = workspace->chars;
    int32_t *d_offsets = workspace->offsets;
    int32_t *d_start_node_ids = workspace->start_node_ids;
    int32_t *d_start_offsets = workspace->start_offsets;
    AlignResult *d_results = workspace->results;
    QueryState *d_states = workspace->states;
    int32_t *d_lengths = workspace->lengths;
    TracebackMeta *d_traceback_meta = workspace->traceback_meta;
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
    const size_t smem_bytes = kernel_shared_bytes(threads_per_block);

    if (chars_bytes > workspace->query_capacity) {
        char *grown = nullptr;
        err = cudaMalloc(&grown, chars_bytes);
        if (err != cudaSuccess) {
            set_error("cudaMalloc(chars)", err);
            status = Status::CudaError;
            goto cleanup;
        }
        if (workspace->chars != nullptr) cudaFree(workspace->chars);
        workspace->chars = d_chars = grown;
        workspace->query_capacity = chars_bytes;
    }
    if (static_cast<size_t>(batch.num_seqs) > workspace->batch_capacity) {
        int32_t *grown_offsets = nullptr;
        int32_t *grown_start_node_ids = nullptr;
        int32_t *grown_start_offsets = nullptr;
        AlignResult *grown_results = nullptr;
        QueryState *grown_states = nullptr;
        int32_t *grown_lengths = nullptr;
        TracebackMeta *grown_traceback_meta = nullptr;
#define ALLOC_GROWN(ptr, bytes, label)                                      \
        do {                                                                \
            err = cudaMalloc(&(ptr), (bytes));                              \
            if (err != cudaSuccess) {                                       \
                set_error((label), err);                                    \
                cudaFree(grown_offsets); cudaFree(grown_start_node_ids);    \
                cudaFree(grown_start_offsets); cudaFree(grown_results);     \
                cudaFree(grown_states); cudaFree(grown_lengths);            \
                cudaFree(grown_traceback_meta);                             \
                status = Status::CudaError;                                 \
                goto cleanup;                                               \
            }                                                               \
        } while (false)
        ALLOC_GROWN(grown_offsets, offsets_bytes, "cudaMalloc(offsets)");
        ALLOC_GROWN(grown_start_node_ids, per_query_i32_bytes, "cudaMalloc(start_node_ids)");
        ALLOC_GROWN(grown_start_offsets, per_query_i32_bytes, "cudaMalloc(start_offsets)");
        ALLOC_GROWN(grown_results, results_bytes, "cudaMalloc(results)");
        ALLOC_GROWN(grown_states, states_bytes, "cudaMalloc(query_states)");
        ALLOC_GROWN(grown_lengths, lengths_bytes, "cudaMalloc(lengths)");
        ALLOC_GROWN(grown_traceback_meta,
                    sizeof(TracebackMeta) * static_cast<size_t>(batch.num_seqs),
                    "cudaMalloc(traceback_meta)");
#undef ALLOC_GROWN
        if (workspace->offsets != nullptr) cudaFree(workspace->offsets);
        if (workspace->start_node_ids != nullptr) cudaFree(workspace->start_node_ids);
        if (workspace->start_offsets != nullptr) cudaFree(workspace->start_offsets);
        if (workspace->results != nullptr) cudaFree(workspace->results);
        if (workspace->states != nullptr) cudaFree(workspace->states);
        if (workspace->lengths != nullptr) cudaFree(workspace->lengths);
        if (workspace->traceback_meta != nullptr) cudaFree(workspace->traceback_meta);
        workspace->offsets = d_offsets = grown_offsets;
        workspace->start_node_ids = d_start_node_ids = grown_start_node_ids;
        workspace->start_offsets = d_start_offsets = grown_start_offsets;
        workspace->results = d_results = grown_results;
        workspace->states = d_states = grown_states;
        workspace->lengths = d_lengths = grown_lengths;
        workspace->traceback_meta = d_traceback_meta = grown_traceback_meta;
        workspace->batch_capacity = static_cast<size_t>(batch.num_seqs);
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
    theseus_align_batch_kernel<<<batch.num_seqs, threads_per_block,
                                         smem_bytes>>>(
        device_batch, graph->view, d_start_node_ids, d_start_offsets, scoring,
        d_states, d_results);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_error("theseus_align_batch_kernel launch", err);
        status = Status::CudaError;
        goto cleanup;
    }
    err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        set_error("theseus_align_batch_kernel synchronize", err);
        status = Status::CudaError;
        goto cleanup;
    }
    if (ev_kernel != nullptr) {
        cudaEventRecord(ev_kernel);
    }

    traceback_meta_kernel<<<blocks, threads_per_block>>>(
        d_states, batch.num_seqs, d_traceback_meta);
    err = cudaGetLastError();
    if (err != cudaSuccess) {
        set_error("traceback_meta_kernel launch", err);
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
        std::vector<TracebackMeta> metadata(static_cast<size_t>(batch.num_seqs));
        err = cudaMemcpy(metadata.data(), d_traceback_meta,
                         sizeof(TracebackMeta) * metadata.size(),
                         cudaMemcpyDeviceToHost);
        if (err != cudaSuccess) {
            set_error("cudaMemcpy(traceback metadata D2H)", err);
            status = Status::CudaError;
            goto cleanup;
        }
        CompactTracebackState *host_states =
            static_cast<CompactTracebackState *>(out_query_states);
        for (int32_t i = 0; i < batch.num_seqs; ++i) {
            const TracebackMeta &meta = metadata[static_cast<size_t>(i)];
            CompactTracebackState &host = host_states[i];
            host.bs_m_wf_size = meta.m_size;
            host.bs_m_jumps_wf_size = meta.m_jumps_size;
            host.bs_i_jumps_wf_size = meta.i_jumps_size;
            host.sc_peak_wf = meta.peak_wf;
            host.capacity_exceeded = meta.capacity_exceeded != 0;
            host.cap_reason = meta.cap_reason;
            host.cap_required = meta.cap_required;
            host.cap_available = meta.cap_available;
        }
#define COPY_TRACEBACK_2D(field, label)                                      \
        do {                                                                 \
            err = cudaMemcpy2D(host_states[0].field,                         \
                               sizeof(CompactTracebackState),                \
                               d_states[0].field, sizeof(QueryState),         \
                               sizeof(host_states[0].field),                  \
                               static_cast<size_t>(batch.num_seqs),           \
                               cudaMemcpyDeviceToHost);                       \
            if (err != cudaSuccess) {                                        \
                set_error((label), err);                                     \
                status = Status::CudaError;                                  \
                goto cleanup;                                                \
            }                                                                \
        } while (false)
        COPY_TRACEBACK_2D(bs_m_wf, "cudaMemcpy2D(traceback M D2H)");
        COPY_TRACEBACK_2D(bs_m_jumps_wf, "cudaMemcpy2D(traceback M jumps D2H)");
        COPY_TRACEBACK_2D(bs_i_jumps_wf, "cudaMemcpy2D(traceback I jumps D2H)");
#undef COPY_TRACEBACK_2D
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
