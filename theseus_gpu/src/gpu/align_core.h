#pragma once

#include "gpu/align_gpu.h"
#include "query_state.h"

namespace theseus {
namespace gpu {

THESEUS_HD inline int32_t vertex_len(const GraphCsrView &graph, int32_t v) {
    return graph.vertex_offsets[v + 1] - graph.vertex_offsets[v];
}

THESEUS_HD inline char vertex_char(const GraphCsrView &graph, int32_t v, int32_t j) {
    return graph.vertex_chars[graph.vertex_offsets[v] + j];
}

THESEUS_HD inline int32_t edge_begin(const GraphCsrView &graph, int32_t v) {
    return graph.edge_offsets[v];
}

THESEUS_HD inline int32_t edge_end(const GraphCsrView &graph, int32_t v) {
    return graph.edge_offsets[v + 1];
}

THESEUS_HD inline void core_lcp(const char *query, int32_t query_len,
                                const GraphCsrView &graph, int32_t v,
                                int32_t &offset, int32_t &j) {
    const int32_t n = vertex_len(graph, v);
    while (offset < query_len && j < n && query[offset] == vertex_char(graph, v, j)) {
        ++offset;
        ++j;
    }
}

THESEUS_HD inline void core_check_end(const Cell &cell, int32_t query_len,
                                      bool &end, Cell &end_cell) {
    if (cell.offset == query_len) {
        end = true;
        end_cell = cell;
    }
}

THESEUS_HD inline void core_sparsify_m(QueryState &qs, Cell *dense_wf,
                                       int32_t offset_increase,
                                       int32_t shift_factor,
                                       Range cells_range,
                                       int32_t query_len,
                                       int32_t upper_bound) {
    const int64_t len = cells_range.end - cells_range.start;
    for (int64_t l = 0; l < len; ++l) {
        Cell new_cell = dense_wf[cells_range.start + l];
        new_cell.diag += shift_factor;
        new_cell.offset += offset_increase;
        new_cell.from_matrix = Cell::Matrix::M;
        new_cell.prev_pos = cells_range.start + l;
        const int32_t new_col = new_cell.offset + new_cell.diag;
        if (new_cell.offset <= query_len && new_col <= upper_bound) {
            Cell &cell = sp_access_alloc(qs, new_cell.diag);
            if (cell.offset < new_cell.offset) {
                cell = new_cell;
            }
        }
    }
}

THESEUS_HD inline void core_sparsify_jumps(QueryState &qs, Cell *dense_wf,
                                           int64_t *jumps_positions,
                                           int32_t jumps_size,
                                           int32_t offset_increase,
                                           int32_t shift_factor,
                                           int32_t query_len,
                                           int32_t upper_bound,
                                           Cell::Matrix from_matrix) {
    for (int32_t l = 0; l < jumps_size; ++l) {
        const int64_t pos = jumps_positions[l];
        Cell new_cell = dense_wf[pos];
        new_cell.prev_pos = pos;
        new_cell.from_matrix = from_matrix;
        new_cell.diag += shift_factor;
        new_cell.offset += offset_increase;
        const int32_t new_col = new_cell.offset + new_cell.diag;
        if (new_cell.offset <= query_len && new_col <= upper_bound) {
            Cell &cell = sp_access_alloc(qs, new_cell.diag);
            if (cell.offset < new_cell.offset) {
                cell = new_cell;
            }
        }
    }
}

THESEUS_HD inline void core_sparsify_indel(QueryState &qs, Cell *dense_wf,
                                           int32_t offset_increase,
                                           int32_t shift_factor,
                                           Range cells_range,
                                           int32_t query_len,
                                           int32_t upper_bound) {
    const int64_t len = cells_range.end - cells_range.start;
    for (int64_t l = 0; l < len; ++l) {
        Cell new_cell = dense_wf[cells_range.start + l];
        new_cell.diag += shift_factor;
        new_cell.offset += offset_increase;
        const int32_t new_col = new_cell.offset + new_cell.diag;
        if (new_cell.offset <= query_len && new_col <= upper_bound) {
            Cell &cell = sp_access_alloc(qs, new_cell.diag);
            if (cell.offset < new_cell.offset) {
                cell = new_cell;
            }
        }
    }
}

THESEUS_HD void core_extend_diagonal(QueryState &qs, const char *query,
                                     int32_t query_len, const GraphCsrView &graph,
                                     int32_t score, Cell &curr_cell,
                                     Cell &prev_cell, int64_t prev_pos,
                                     Cell::Matrix from_matrix,
                                     bool &end, Cell &end_cell);

THESEUS_HD inline void core_store_m_jump(QueryState &qs, const char *query,
                                         int32_t query_len, const GraphCsrView &graph,
                                         int32_t score, int32_t curr_v,
                                         Cell &prev_cell, int64_t prev_pos,
                                         Cell::Matrix from_matrix,
                                         bool &end, Cell &end_cell) {
    vd_invalidate_m_jump(qs, vd_get_id(qs, prev_cell.vertex_id), prev_cell.diag);
    const int32_t pos_score = vd_get_pos(qs, score);
    const int32_t new_diag = -prev_cell.offset;
    Cell new_cell = prev_cell;
    new_cell.from_matrix = from_matrix;
    new_cell.prev_pos = prev_pos;

    for (int32_t e = edge_begin(graph, curr_v); e < edge_end(graph, curr_v); ++e) {
        new_cell.vertex_id = graph.edge_targets[e];
        new_cell.diag = new_diag + graph.edge_overlaps[e];
        vd_activate_vertex(qs, new_cell.vertex_id);
        if (vd_valid_diagonal(qs, Cell::Matrix::M, new_cell.vertex_id, new_cell.diag)) {
            const int32_t pos_new_cell = bs_push_back(qs, qs.bs_m_jumps_wf, qs.bs_m_jumps_wf_size, new_cell);
            vd_jumps_push(qs, vd_m_jumps(qs, new_cell.vertex_id, pos_score),
                          vd_m_jumps_size(qs, new_cell.vertex_id, pos_score), pos_new_cell);
            core_extend_diagonal(qs, query, query_len, graph, score, qs.bs_m_jumps_wf[pos_new_cell],
                                 qs.bs_m_jumps_wf[pos_new_cell], pos_new_cell, Cell::Matrix::MJumps,
                                 end, end_cell);
        }
    }
}

THESEUS_HD inline void core_store_i_jump(QueryState &qs, const GraphCsrView &graph,
                                         int32_t score, Cell &prev_cell,
                                         int64_t prev_pos, Cell::Matrix from_matrix) {
    struct Frame {
        int32_t vertex_id;
        Cell prev_cell;
        int64_t prev_pos;
        Cell::Matrix from_matrix;
        int32_t next_edge;
        bool invalidated;
    };

    Frame stack[kMaxIJumpStack];
    int32_t stack_size = 1;
    stack[0] = Frame{prev_cell.vertex_id, prev_cell, prev_pos, from_matrix, edge_begin(graph, prev_cell.vertex_id), false};

    while (stack_size > 0) {
        Frame &frame = stack[stack_size - 1];
        if (!frame.invalidated) {
            vd_invalidate_i_jump(qs, vd_get_id(qs, frame.prev_cell.vertex_id), frame.prev_cell.diag);
            frame.invalidated = true;
        }
        if (frame.next_edge >= edge_end(graph, frame.vertex_id)) {
            --stack_size;
            continue;
        }
        const int32_t e = frame.next_edge++;
        const int32_t pos_score = vd_get_pos(qs, score);
        const int32_t new_diag = -frame.prev_cell.offset;
        Cell new_cell = frame.prev_cell;
        new_cell.from_matrix = frame.from_matrix;
        new_cell.prev_pos = frame.prev_pos;
        new_cell.vertex_id = graph.edge_targets[e];
        new_cell.diag = new_diag + graph.edge_overlaps[e];
        vd_activate_vertex(qs, new_cell.vertex_id);

        if (vd_valid_diagonal(qs, Cell::Matrix::I, new_cell.vertex_id, new_cell.diag)) {
            const int32_t pos_new_cell = bs_push_back(qs, qs.bs_i_jumps_wf, qs.bs_i_jumps_wf_size, new_cell);
            vd_jumps_push(qs, vd_i_jumps(qs, new_cell.vertex_id, pos_score),
                          vd_i_jumps_size(qs, new_cell.vertex_id, pos_score), pos_new_cell);
            if (vertex_len(graph, new_cell.vertex_id) == 0) {
                if (stack_size >= kMaxIJumpStack) {
                    cap_fail(qs, kCapIJumpStack, stack_size + 1, kMaxIJumpStack);
                    continue;
                }
                stack[stack_size] = Frame{new_cell.vertex_id, qs.bs_i_jumps_wf[pos_new_cell],
                                          frame.prev_pos, Cell::Matrix::IJumps,
                                          edge_begin(graph, new_cell.vertex_id), false};
                ++stack_size;
            }
        }
    }
}

THESEUS_HD inline void core_check_and_store_jumps(QueryState &qs, const char *query,
                                                  int32_t query_len, const GraphCsrView &graph,
                                                  int32_t score, int32_t curr_v,
                                                  Cell *curr_wavefront,
                                                  Range cell_range,
                                                  bool &end, Cell &end_cell) {
    const int64_t len = cell_range.end - cell_range.start;
    const int32_t n = vertex_len(graph, curr_v);
    for (int64_t l = 0; l < len; ++l) {
        Cell &cell = curr_wavefront[cell_range.start + l];
        const int32_t curr_j = cell.diag + cell.offset;
        if (curr_j == n && cell.offset <= query_len) {
            core_store_m_jump(qs, query, query_len, graph, score, curr_v, cell,
                              cell.prev_pos, cell.from_matrix, end, end_cell);
            core_store_i_jump(qs, graph, score, cell, cell.prev_pos, cell.from_matrix);
        }
    }
}

/**
 * @brief Sparsify half of the D recurrence: fold every contribution that can
 * reach this vertex's D wavefront into the ScratchPad, where candidates landing
 * on the same diagonal collide and the largest offset wins.
 *
 * The densify half that used to follow it is now the block-parallel
 * filter+compaction in align_gpu.cu.
 */
THESEUS_HD inline void core_next_d_sparsify(QueryState &qs, const AlignScoring &scoring,
                                            int32_t query_len, const GraphCsrView &graph,
                                            int32_t score, int32_t v) {
    const int32_t upper_bound = vertex_len(graph, v);
    const int32_t pos_prev_m = score - (scoring.gapo + scoring.gape);
    const int32_t pos_prev_d = score - scoring.gape;
    const int32_t pos_prev_m_scope = vd_get_pos(qs, pos_prev_m);
    const int32_t v_id = vd_get_id(qs, v);

    if (pos_prev_d >= 0 && sc_d_pos_size(qs, pos_prev_d) > v_id) {
        core_sparsify_indel(qs, sc_d_wf(qs, pos_prev_d), 1, -1,
                            sc_d_pos(qs, pos_prev_d)[v_id], query_len, upper_bound);
    }
    if (pos_prev_m >= 0) {
        if (sc_m_pos_size(qs, pos_prev_m) > v_id) {
            core_sparsify_m(qs, qs.bs_m_wf, 1, -1, sc_m_pos(qs, pos_prev_m)[v_id],
                            query_len, upper_bound);
        }
        core_sparsify_jumps(qs, qs.bs_m_jumps_wf, vd_m_jumps(qs, v, pos_prev_m_scope),
                            vd_m_jumps_size(qs, v, pos_prev_m_scope), 1, -1,
                            query_len, upper_bound, Cell::Matrix::MJumps);
    }
}

/** @brief Sparsify half of the M recurrence. See core_next_d_sparsify. */
THESEUS_HD inline void core_next_m_sparsify(QueryState &qs, const AlignScoring &scoring,
                                            int32_t query_len, const GraphCsrView &graph,
                                            int32_t score, int32_t v) {
    const int32_t upper_bound = vertex_len(graph, v);
    const int32_t pos_prev_m = score - scoring.mism;
    const int32_t pos_prev_d = score;
    const int32_t pos_prev_i = score;
    const int32_t pos_prev_m_scope = vd_get_pos(qs, pos_prev_m);
    const int32_t v_id = vd_get_id(qs, v);

    if (sc_d_pos_size(qs, pos_prev_d) > v_id) {
        core_sparsify_indel(qs, sc_d_wf(qs, pos_prev_d), 0, 0,
                            sc_d_pos(qs, pos_prev_d)[v_id], query_len, upper_bound);
    }
    if (sc_i_pos_size(qs, pos_prev_i) > v_id) {
        core_sparsify_indel(qs, sc_i_wf(qs, pos_prev_i), 0, 0,
                            sc_i_pos(qs, pos_prev_i)[v_id], query_len, upper_bound);
    }
    if (pos_prev_m >= 0) {
        if (sc_m_pos_size(qs, pos_prev_m) > v_id) {
            core_sparsify_m(qs, qs.bs_m_wf, 1, 0, sc_m_pos(qs, pos_prev_m)[v_id],
                            query_len, upper_bound);
        }
        core_sparsify_jumps(qs, qs.bs_m_jumps_wf, vd_m_jumps(qs, v, pos_prev_m_scope),
                            vd_m_jumps_size(qs, v, pos_prev_m_scope), 1, 0,
                            query_len, upper_bound, Cell::Matrix::MJumps);
    }
}

THESEUS_HD inline void core_extend_diagonal(QueryState &qs, const char *query,
                                            int32_t query_len, const GraphCsrView &graph,
                                            int32_t score, Cell &curr_cell,
                                            Cell &prev_cell, int64_t prev_pos,
                                            Cell::Matrix from_matrix,
                                            bool &end, Cell &end_cell) {
    int32_t j = curr_cell.diag + curr_cell.offset;
    core_lcp(query, query_len, graph, curr_cell.vertex_id, curr_cell.offset, j);
    core_check_end(curr_cell, query_len, end, end_cell);
    if (j == vertex_len(graph, curr_cell.vertex_id) && curr_cell.offset <= query_len &&
        edge_begin(graph, curr_cell.vertex_id) < edge_end(graph, curr_cell.vertex_id)) {
        core_store_m_jump(qs, query, query_len, graph, score, curr_cell.vertex_id,
                          prev_cell, prev_pos, from_matrix, end, end_cell);
    }
}

}  // namespace gpu
}  // namespace theseus
