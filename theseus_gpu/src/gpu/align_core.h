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

THESEUS_HD inline void core_next_i(QueryState &qs, const AlignScoring &scoring,
                                   const char *query, int32_t query_len,
                                   const GraphCsrView &graph, int32_t score,
                                   int32_t v, bool &end, Cell &end_cell) {
    const int32_t upper_bound = vertex_len(graph, v);
    const int32_t pos_prev_m = score - (scoring.gapo + scoring.gape);
    const int32_t pos_prev_i = score - scoring.gape;
    const int32_t pos_prev_m_scope = vd_get_pos(qs, pos_prev_m);
    const int32_t pos_prev_i_scope = vd_get_pos(qs, pos_prev_i);
    const int32_t v_id = vd_get_id(qs, v);

    if (pos_prev_i >= 0) {
        if (sc_i_pos_size(qs, pos_prev_i) > v_id) {
            core_sparsify_indel(qs, sc_i_wf(qs, pos_prev_i), 0, 1,
                                sc_i_pos(qs, pos_prev_i)[v_id], query_len, upper_bound);
        }
        core_sparsify_jumps(qs, qs.bs_i_jumps_wf, vd_i_jumps(qs, v, pos_prev_i_scope),
                            vd_i_jumps_size(qs, v, pos_prev_i_scope), 0, 1,
                            query_len, upper_bound, Cell::Matrix::IJumps);
    }
    if (pos_prev_m >= 0) {
        if (sc_m_pos_size(qs, pos_prev_m) > v_id) {
            core_sparsify_m(qs, qs.bs_m_wf, 0, 1, sc_m_pos(qs, pos_prev_m)[v_id],
                            query_len, upper_bound);
        }
        core_sparsify_jumps(qs, qs.bs_m_jumps_wf, vd_m_jumps(qs, v, pos_prev_m_scope),
                            vd_m_jumps_size(qs, v, pos_prev_m_scope), 0, 1,
                            query_len, upper_bound, Cell::Matrix::MJumps);
    }

    Range new_range;
    new_range.start = sc_i_wf_size(qs, score);
    for (int32_t di = 0; di < qs.sp_ndiags; ++di) {
        const int32_t diag = qs.sp_diags[di];
        if (vd_valid_diagonal(qs, Cell::Matrix::I, v, diag)) {
            sc_wf_push(qs, sc_i_wf(qs, score), sc_i_wf_size(qs, score), sp_at(qs, diag));
        }
    }
    new_range.end = sc_i_wf_size(qs, score);
    sc_pos_push(qs, sc_i_pos(qs, score), sc_i_pos_size(qs, score), new_range);

    // Mirrors the tail of the CPU's next_I: every I cell that has just reached
    // the last column of this vertex opens M and I jumps into the neighbours.
    // Without it those jump candidates never enter the later wavefronts.
    if (edge_begin(graph, v) < edge_end(graph, v)) {
        core_check_and_store_jumps(qs, query, query_len, graph, score, v,
                                   sc_i_wf(qs, score), new_range, end, end_cell);
    }
}

/**
 * @brief Densify half of core_next_d: scan the active diagonals, keep the ones
 * still valid for this vertex, append them to the score's D wavefront.
 *
 * Split out so config1 can replace it with a block-parallel filter+compaction
 * while config0 keeps calling the serial pair through core_next_d.
 */
THESEUS_HD inline void core_next_d_densify(QueryState &qs, int32_t score,
                                           int32_t v) {
    Range new_range;
    new_range.start = sc_d_wf_size(qs, score);
    for (int32_t di = 0; di < qs.sp_ndiags; ++di) {
        const int32_t diag = qs.sp_diags[di];
        if (vd_valid_diagonal(qs, Cell::Matrix::D, v, diag)) {
            sc_wf_push(qs, sc_d_wf(qs, score), sc_d_wf_size(qs, score), sp_at(qs, diag));
        }
    }
    new_range.end = sc_d_wf_size(qs, score);
    sc_pos_push(qs, sc_d_pos(qs, score), sc_d_pos_size(qs, score), new_range);
}

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

THESEUS_HD inline void core_next_d(QueryState &qs, const AlignScoring &scoring,
                                   int32_t query_len, const GraphCsrView &graph,
                                   int32_t score, int32_t v) {
    core_next_d_sparsify(qs, scoring, query_len, graph, score, v);
    core_next_d_densify(qs, score, v);
}

/** @brief Densify half of core_next_m. See core_next_d_densify. */
THESEUS_HD inline void core_next_m_densify(QueryState &qs, int32_t score,
                                           int32_t v) {
    Range new_range;
    new_range.start = qs.bs_m_wf_size;
    for (int32_t di = 0; di < qs.sp_ndiags; ++di) {
        const int32_t diag = qs.sp_diags[di];
        if (vd_valid_diagonal(qs, Cell::Matrix::M, v, diag)) {
            bs_push_back(qs, qs.bs_m_wf, qs.bs_m_wf_size, sp_at(qs, diag));
        }
    }
    new_range.end = qs.bs_m_wf_size;
    sc_pos_push(qs, sc_m_pos(qs, score), sc_m_pos_size(qs, score), new_range);
}

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

THESEUS_HD inline void core_next_m(QueryState &qs, const AlignScoring &scoring,
                                   int32_t query_len, const GraphCsrView &graph,
                                   int32_t score, int32_t v) {
    core_next_m_sparsify(qs, scoring, query_len, graph, score, v);
    core_next_m_densify(qs, score, v);
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

THESEUS_HD inline void core_process_vertex(QueryState &qs, const AlignScoring &scoring,
                                           const char *query, int32_t query_len,
                                           const GraphCsrView &graph, int32_t score,
                                           int32_t v, bool &end, Cell &end_cell) {
    core_next_i(qs, scoring, query, query_len, graph, score, v, end, end_cell);
    sp_reset(qs);
    core_next_d(qs, scoring, query_len, graph, score, v);
    sp_reset(qs);
    core_next_m(qs, scoring, query_len, graph, score, v);
    sp_reset(qs);

    const int32_t v_pos = vd_get_id(qs, v);
    const Range cells_range = sc_m_pos(qs, score)[v_pos];
    for (int64_t idx = cells_range.start; idx < cells_range.end; ++idx) {
        core_extend_diagonal(qs, query, query_len, graph, score, qs.bs_m_wf[idx],
                             qs.bs_m_wf[idx], idx, Cell::Matrix::M, end, end_cell);
    }
}

THESEUS_HD inline void core_compute_new_wave(QueryState &qs, const AlignScoring &scoring,
                                             const char *query, int32_t query_len,
                                             const GraphCsrView &graph, int32_t score,
                                             bool &end, Cell &end_cell) {
    vd_expand(qs);
    vd_compact(qs);
    const int32_t num_active = vd_num_active_vertices(qs);
    for (int32_t l = 0; l < num_active; ++l) {
        core_process_vertex(qs, scoring, query, query_len, graph, score,
                            vd_get_vertex_id(qs, l), end, end_cell);
    }
}

THESEUS_HD inline void align_one(QueryState &qs, const AlignScoring &scoring,
                                 const char *query, int32_t query_len,
                                 const GraphCsrView &graph, int32_t start_node,
                                 int32_t start_offset, AlignResult &result) {
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
    vd_init(qs, scoring.gapo, scoring.gape, scoring.nscores, graph.num_vertices);
    sc_new_alignment(qs);
    bs_new_alignment(qs);
    vd_new_alignment(qs);

    int32_t score = 0;
    bool end = false;
    Cell end_cell{-1, -1, -1, -1, Cell::Matrix::None};

    sc_new_score(qs, score);
    Cell init_condition;
    init_condition.offset = 0;
    init_condition.vertex_id = start_node;
    init_condition.diag = start_offset;
    init_condition.prev_pos = -1;
    init_condition.from_matrix = Cell::Matrix::None;

    bs_push_back(qs, qs.bs_m_jumps_wf, qs.bs_m_jumps_wf_size, init_condition);
    vd_activate_vertex(qs, start_node);
    vd_jumps_push(qs, vd_m_jumps(qs, start_node, 0), vd_m_jumps_size(qs, start_node, 0), 0);

    while (!end && !qs.capacity_exceeded) {
        if (score == 0) {
            core_extend_diagonal(qs, query, query_len, graph, score, qs.bs_m_jumps_wf[0],
                                 qs.bs_m_jumps_wf[0], 0, Cell::Matrix::MJumps,
                                 end, end_cell);
        }
        core_compute_new_wave(qs, scoring, query, query_len, graph, score, end, end_cell);
        ++score;
        sc_new_score(qs, score);
        vd_new_score(qs, score);
    }
    --score;

    result.score = score;
    result.end_vertex_id = end_cell.vertex_id;
    result.end_offset = end_cell.offset;
    result.end_diag = end_cell.diag;
    result.end_prev_pos = end_cell.prev_pos;
    result.end_from_matrix = static_cast<int8_t>(end_cell.from_matrix);
    result.reached_end = end ? 1 : 0;
    result.capacity_exceeded = qs.capacity_exceeded ? 1 : 0;
    result.reserved = 0;
}

}  // namespace gpu
}  // namespace theseus
