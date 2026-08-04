/*
 *                             The MIT License
 *
 * Copyright (c) 2024 by Albert Jimenez-Blanco
 *
 * This file is part of #################### Theseus Library ####################.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 *
 */


#include <string_view>
#include "theseus_aligner_impl.h"

namespace theseus {

TheseusAlignerImpl::TheseusAlignerImpl(const Penalties &penalties,
                                       Graph &&graph) : _penalties(penalties),
                                                        _graph(std::move(graph)),
                                                        _internal_penalties(penalties) {
    // TODO: Gap-linear and dual affine-gap.
    const auto n_scores = std::max({_internal_penalties.gapo() +_internal_penalties.gape(),
                                  _internal_penalties.gapo() +_internal_penalties.gape(),
                                  _internal_penalties.mism()}) + 1;

    _internal_penalties = InternalPenalties(penalties);
    constexpr int expected_nvertices = std::max(1024, 0); // TODO: Set the expected number of vertices
    _vertices_data = std::make_unique<VerticesData>(penalties, n_scores, expected_nvertices);
    _qs = std::make_unique<QueryState>();
    _qs->capacity_exceeded = false;
    _qs->cap_reason = kCapNone;
    _qs->cap_required = 0;
    _qs->cap_available = 0;
    sp_init(*_qs, -1024, 1024);
    sc_init(*_qs, n_scores);
    vd_init(*_qs, penalties.gapo(), penalties.gape(), n_scores, _graph._vertices.size());
    _graph_csr = std::make_unique<gpu::GraphCsr>(_graph);
}

TheseusAlignerImpl::~TheseusAlignerImpl() {
    gpu::free_graph(_device_graph);
}

gpu::DeviceGraph *TheseusAlignerImpl::device_graph() {
    if (!_device_graph_attempted) {
        _device_graph_attempted = true;
        _device_graph = gpu::upload_graph(_graph_csr->view());
    }
    return _device_graph;
}

int32_t TheseusAlignerImpl::graph_vertex_id(const std::string &name) {
    return static_cast<int32_t>(_graph.get_id(name));
}

int32_t TheseusAlignerImpl::max_vertex_length() const {
    int32_t longest = 0;
    for (size_t i = 0; i < _graph._vertices.size(); ++i) {
        longest = std::max(longest, static_cast<int32_t>(_graph._vertices[i].value.size()));
    }
    return longest;
}

gpu::AlignResult TheseusAlignerImpl::last_align_result() const {
    gpu::AlignResult result;
    result.score = _score;
    result.end_vertex_id = _start_pos.vertex_id;
    result.end_offset = _start_pos.offset;
    result.end_diag = _start_pos.diag;
    result.end_prev_pos = _start_pos.prev_pos;
    result.end_from_matrix = static_cast<int8_t>(_start_pos.from_matrix);
    result.reached_end = _end ? 1 : 0;
    result.capacity_exceeded = _qs->capacity_exceeded ? 1 : 0;
    result.reserved = 0;
    return result;
}

Alignment TheseusAlignerImpl::alignment_from_gpu_result(
    std::string_view seq,
    int start_offset,
    const QueryState &state,
    const gpu::AlignResult &result) {
    *_qs = state;
    _seq = seq;
    _start_offset = start_offset;
    _score = result.score;
    _end = result.reached_end != 0;
    _start_pos.prev_pos = result.end_prev_pos;
    _start_pos.vertex_id = result.end_vertex_id;
    _start_pos.offset = result.end_offset;
    _start_pos.diag = result.end_diag;
    _start_pos.from_matrix = static_cast<Cell::Matrix>(result.end_from_matrix);
    _alignment.path.clear();
    _alignment.edit_op.clear();
    if (!_end || result.capacity_exceeded != 0) {
        return _alignment;
    }
    backtrace(0);
    return _alignment;
}

void TheseusAlignerImpl::new_alignment() {
    int max_diag = 0, v_n;
    for (int l = 0; l < _graph._vertices.size(); ++l) {
      v_n = _graph._vertices[l].value.size();
      max_diag = std::max(max_diag, v_n);
    }
    const int min_diag = -_seq.size();

    if (_qs->sp_max_diag < max_diag ||
        _qs->sp_min_diag > min_diag) {
        // TODO: Compute the max and min with a factor.
        // Mirrors the CPU ScratchPad, which reallocated to exactly this window
        // whenever either bound had to grow.
        sp_init(*_qs, min_diag, max_diag);
    }

    // Set data for first score
    sc_new_score(*_qs, _score);

    // TODO: Allow for different initial conditions. Now only global alignment.
    Cell init_condition;
    init_condition.offset = 0;
    init_condition.vertex_id = _start_node;
    init_condition.diag = _start_offset;
    init_condition.prev_pos = -1;

    // Initial vertex data
    bs_push_back(*_qs, _qs->bs_m_jumps_wf, _qs->bs_m_jumps_wf_size, init_condition);
    vd_activate_vertex(*_qs, _start_node);
    vd_jumps_push(*_qs, vd_m_jumps(*_qs, _start_node, 0), vd_m_jumps_size(*_qs, _start_node, 0), 0);

    // Alignment data
    _alignment.path.clear();
    _alignment.edit_op.clear();
}


// Process a given vertex with a given _score
void TheseusAlignerImpl::process_vertex(Graph::vertex* curr_v,
                                        int v) {

  // Perform the next operation
  int upper_bound = curr_v->value.size();
  next_I(curr_v, upper_bound, v);
  sp_reset(*_qs);
  next_D(upper_bound, v);
  sp_reset(*_qs);
  next_M(upper_bound, v);
  sp_reset(*_qs);

  // Perform the extend operations
  int v_pos = vd_get_id(*_qs, v);
  Range cells_range = sc_m_pos(*_qs, _score)[v_pos];
  for (Cell::pos_t idx = cells_range.start; idx < cells_range.end; ++idx) {
    extend_diagonal(curr_v, _qs->bs_m_wf[idx], v, _qs->bs_m_wf[idx], idx, Cell::Matrix::M);
  }
}


void TheseusAlignerImpl::compute_new_wave() {

  // Update invalid segments
  vd_expand(*_qs);
  vd_compact(*_qs);

  // Process all active vertices
  int num_active_vertices = vd_num_active_vertices(*_qs), v;
  for (int l = 0; l < num_active_vertices; ++l) {
    v = vd_get_vertex_id(*_qs, l);
    Graph::vertex* curr_v = &_graph._vertices[v];
    process_vertex(curr_v, v);
  }
}


Alignment TheseusAlignerImpl::align(
    std::string_view seq,
    std::string &start_node,
    int start_offset)
{
  sc_new_alignment(*_qs);
  bs_new_alignment(*_qs);
  vd_new_alignment(*_qs);
  _seq = seq;

  _start_node = _graph.get_id(start_node);
  _start_offset = start_offset;

  // Initialize data for the new alignment
  new_alignment();

  // TODO: Set initial conditions
  _score = 0;
  _end = false;
  // _graph.print_code_graphviz();

  // Find the optimal _score and an optimal alignment
  while (!_end)
  {
    // Compute the values of the new wave
    // Initial extend
    if (_score == 0) {
      extend_diagonal(&_graph._vertices[_start_node], _qs->bs_m_jumps_wf[0], _start_node, _qs->bs_m_jumps_wf[0], 0, Cell::Matrix::MJumps);
    }
    compute_new_wave();

    // Update _score
    _score = _score + 1;

    // Clear the corresponding waves and metadata from the scope
    sc_new_score(*_qs, _score);
    vd_new_score(*_qs, _score);
  }
  _score -= 1;

  // Backtrace
  backtrace(0);

  return _alignment;
}

  // Sparsify M data
  void TheseusAlignerImpl::sparsify_M_data(Cell * dense_wf,
                                           int offset_increase,
                                           int shift_factor,
                                           Range cells_range,
                                           int m,
                                           int upper_bound)
  {

    Cell::pos_t len = cells_range.end - cells_range.start, new_col;
    Cell new_cell;

    // Sparsify the active diagonals
    for (int l = 0; l < len; ++l)
    {
      new_cell = dense_wf[cells_range.start + l];
      new_cell.diag += shift_factor;
      new_cell.offset += offset_increase;
      new_cell.from_matrix = Cell::Matrix::M;
      new_cell.prev_pos = cells_range.start + l;
      new_col = new_cell.offset + new_cell.diag; // d = j - i -> j = d + i

      // Check validity
      if (new_cell.offset <= m && new_col <= upper_bound)
      { // If in bounds
        // Branchless push_back
        auto &cell = sp_access_alloc(*_qs, new_cell.diag);

        // If better offset
        const bool cmp = cell.offset < new_cell.offset;
        cell = (cmp) ? new_cell : cell;
      }
    }
  }

  // Sparsify jumps data
  void TheseusAlignerImpl::sparsify_jumps_data(Cell * dense_wf,
                                               Cell::pos_t * jumps_positions,
                                               int jumps_size,
                                               int offset_increase,
                                               int shift_factor,
                                               int m,
                                               int upper_bound,
                                               Cell::Matrix from_matrix)
  {
    int len = jumps_size, new_col, pos;
    Cell new_cell;

    // Sparsify the active diagonals
    for (int l = 0; l < len; ++l)
    {
      pos = jumps_positions[l];
      new_cell = dense_wf[pos];
      new_cell.prev_pos = pos;
      new_cell.from_matrix = from_matrix;
      new_cell.diag += shift_factor;
      new_cell.offset += offset_increase;
      new_col = new_cell.offset + new_cell.diag; // d = j - i -> j = d + i

      // Check validity
      if (new_cell.offset <= m && new_col <= upper_bound)
      { // If in bounds
        // Branchless push_back
        auto &cell = sp_access_alloc(*_qs, new_cell.diag);

        // If better offset
        const bool cmp = cell.offset < new_cell.offset;
        cell = (cmp) ? new_cell : cell;
      }
    }
  }

  // Sparsify indel
  void TheseusAlignerImpl::sparsify_indel_data(Cell * dense_wf,
                                               int offset_increase,
                                               int shift_factor,
                                               Range cells_range,
                                               int m,
                                               int upper_bound)
  {

    Cell::pos_t len = cells_range.end - cells_range.start, new_col;
    Cell new_cell;

    // Sparsify the active diagonals
    for (int l = 0; l < len; ++l)
    {
      // Vertex_id and previous matrix are the same as before
      new_cell = dense_wf[cells_range.start + l];
      new_cell.diag += shift_factor;
      new_cell.offset += offset_increase;
      new_col = new_cell.offset + new_cell.diag; // d = j - i -> j = d + i

      // Check validity
      if (new_cell.offset <= m && new_col <= upper_bound)
      { // If in bounds
        // Branchless push_back
        auto &cell = sp_access_alloc(*_qs, new_cell.diag);

        // If better offset
        const bool cmp = cell.offset < new_cell.offset;
        cell = (cmp) ? new_cell : cell;
      }
    }
  }

  // Compute next I matrix.
  // HOTSPOT: insertion propagation path (candidate for GPU offload).
  void TheseusAlignerImpl::next_I(Graph::vertex * curr_v,
                                  int upper_bound,
                                  int v)
  {

    // Sparsify data (put it in the scratch pad)
    int pos_prev_M = _score - (_internal_penalties.gapo() + _internal_penalties.gape()), pos_prev_I = _score - _internal_penalties.gape(), pos_prev_M_scope = vd_get_pos(*_qs, pos_prev_M);
    int pos_prev_I_scope = vd_get_pos(*_qs, pos_prev_I);

    // Come from an Insertion
    if (pos_prev_I >= 0) {
      if (sc_i_pos_size(*_qs, pos_prev_I) > vd_get_id(*_qs, v))
      {
        Range cells_range = sc_i_pos(*_qs, pos_prev_I)[vd_get_id(*_qs, v)];
        sparsify_indel_data(sc_i_wf(*_qs, pos_prev_I), 0, 1, cells_range, _seq.size(), upper_bound); // Sparsify I data
      };
      sparsify_jumps_data(_qs->bs_i_jumps_wf, vd_i_jumps(*_qs, v, pos_prev_I_scope), vd_i_jumps_size(*_qs, v, pos_prev_I_scope), 0, 1, _seq.size(), upper_bound, Cell::Matrix::IJumps);
    }

    // Come from M
    if (pos_prev_M >= 0) {
      if (sc_m_pos_size(*_qs, pos_prev_M) > vd_get_id(*_qs, v)) {
        Range cells_range = sc_m_pos(*_qs, pos_prev_M)[vd_get_id(*_qs, v)];
        sparsify_M_data(_qs->bs_m_wf, 0, 1, cells_range, _seq.size(), upper_bound); // Sparsify M data
      }
      sparsify_jumps_data(_qs->bs_m_jumps_wf, vd_m_jumps(*_qs, v, pos_prev_M_scope), vd_m_jumps_size(*_qs, v, pos_prev_M_scope), 0, 1, _seq.size(), upper_bound, Cell::Matrix::MJumps);
    }

    // Densify data (store it in the big wavefront)
    Range new_range;
    new_range.start = sc_i_wf_size(*_qs, _score);
    for (int di = 0; di < _qs->sp_ndiags; ++di) {
      int diag = _qs->sp_diags[di];
      if (vd_valid_diagonal(*_qs, Cell::Matrix::I, v, diag)) {
        sc_wf_push(*_qs, sc_i_wf(*_qs, _score), sc_i_wf_size(*_qs, _score), sp_at(*_qs, diag));     // Store Cell
      }
    }
    new_range.end = sc_i_wf_size(*_qs, _score);
    sc_pos_push(*_qs, sc_i_pos(*_qs, _score), sc_i_pos_size(*_qs, _score), new_range);

    // Check, store and invalidate new I jumps
    if (curr_v->out_edges.size() > 0) {
      check_and_store_jumps(curr_v, sc_i_wf(*_qs, _score), new_range);
    }
}


// Compute next D matrix
void TheseusAlignerImpl::next_D(int upper_bound,
                                int v)
{

  // Sparsify data (put it in the scratch pad)
  int pos_prev_M = _score - (_internal_penalties.gapo() + _internal_penalties.gape()), pos_prev_D = _score - _internal_penalties.gape(), pos_prev_M_scope = vd_get_pos(*_qs, pos_prev_M);

  // Come from a Deletion
  if (pos_prev_D >= 0 && sc_d_pos_size(*_qs, pos_prev_D) > vd_get_id(*_qs, v))
  {
    Range cells_range = sc_d_pos(*_qs, pos_prev_D)[vd_get_id(*_qs, v)];
    sparsify_indel_data(sc_d_wf(*_qs, pos_prev_D), 1, -1, cells_range, _seq.size(), upper_bound); // Sparsify D data
  }

  // Come from M
  if (pos_prev_M >= 0) {
    if (sc_m_pos_size(*_qs, pos_prev_M) > vd_get_id(*_qs, v))
    {
      Range cells_range = sc_m_pos(*_qs, pos_prev_M)[vd_get_id(*_qs, v)];
      sparsify_M_data(_qs->bs_m_wf, 1, -1, cells_range, _seq.size(), upper_bound); // Sparsify M data
    }
    sparsify_jumps_data(_qs->bs_m_jumps_wf, vd_m_jumps(*_qs, v, pos_prev_M_scope), vd_m_jumps_size(*_qs, v, pos_prev_M_scope), 1, -1, _seq.size(), upper_bound, Cell::Matrix::MJumps);
  }

  // Densify data (store it in the big wavefront)
  Range new_range;
  new_range.start = sc_d_wf_size(*_qs, _score);
  for (int di = 0; di < _qs->sp_ndiags; ++di) {
    int diag = _qs->sp_diags[di];
    if (vd_valid_diagonal(*_qs, Cell::Matrix::D, v, diag)) {
      sc_wf_push(*_qs, sc_d_wf(*_qs, _score), sc_d_wf_size(*_qs, _score), sp_at(*_qs, diag)); // Store Cell
    }
  }
  new_range.end = sc_d_wf_size(*_qs, _score);
  sc_pos_push(*_qs, sc_d_pos(*_qs, _score), sc_d_pos_size(*_qs, _score), new_range);
}


// Compute next M matrix
void TheseusAlignerImpl::next_M(int upper_bound,
                                int v) {

  // Sparsify data (put it in the scratch pad)
  int pos_prev_M = _score - _internal_penalties.mism(), pos_prev_D = _score, pos_prev_I = _score, pos_prev_M_scope = vd_get_pos(*_qs, pos_prev_M);

  // Come from a Deletion
  if (sc_d_pos_size(*_qs, pos_prev_D) > vd_get_id(*_qs, v))  {
    Range cells_range = sc_d_pos(*_qs, pos_prev_D)[vd_get_id(*_qs, v)];
    sparsify_indel_data(sc_d_wf(*_qs, pos_prev_D), 0, 0, cells_range, _seq.size(), upper_bound);  // Sparsify D data
  }

  // Come from an Insertion
  if (sc_i_pos_size(*_qs, pos_prev_I) > vd_get_id(*_qs, v))  {
    Range cells_range = sc_i_pos(*_qs, pos_prev_I)[vd_get_id(*_qs, v)];
    sparsify_indel_data(sc_i_wf(*_qs, pos_prev_I), 0, 0, cells_range, _seq.size(), upper_bound);  // Sparsify I data
  }

  // Come from M
  if (pos_prev_M >= 0) {
    if (sc_m_pos_size(*_qs, pos_prev_M) > vd_get_id(*_qs, v))  {
      Range cells_range = sc_m_pos(*_qs, pos_prev_M)[vd_get_id(*_qs, v)];
      sparsify_M_data(_qs->bs_m_wf, 1, 0, cells_range,  _seq.size(), upper_bound);  // Sparsify M data
    }
    sparsify_jumps_data(_qs->bs_m_jumps_wf, vd_m_jumps(*_qs, v, pos_prev_M_scope), vd_m_jumps_size(*_qs, v, pos_prev_M_scope), 1, 0, _seq.size(), upper_bound, Cell::Matrix::MJumps);
  }

  // Densify data (store it in the big wavefront)
  Range new_range;
  new_range.start = _qs->bs_m_wf_size;
  for (int di = 0; di < _qs->sp_ndiags; ++di) {
    int diag = _qs->sp_diags[di];
    if (vd_valid_diagonal(*_qs, Cell::Matrix::M, v, diag)) {
      bs_push_back(*_qs, _qs->bs_m_wf, _qs->bs_m_wf_size, sp_at(*_qs, diag));  // Store Cell
    }
  }
  new_range.end = _qs->bs_m_wf_size;
  sc_pos_push(*_qs, sc_m_pos(*_qs, _score), sc_m_pos_size(*_qs, _score), new_range);
}



// Store the jump in neighbours
void TheseusAlignerImpl::store_M_jump(Graph::vertex* curr_v,
                                      Cell &prev_cell,
                                      Cell::pos_t prev_pos,
                                      Cell::Matrix from_matrix) {

  // Invalidate the jumping diagonal
  vd_invalidate_m_jump(*_qs, vd_get_id(*_qs, prev_cell.vertex_id), prev_cell.diag);
  int pos_score = vd_get_pos(*_qs, _score);
  int new_diag = -prev_cell.offset;
  int num_out_v = curr_v->out_edges.size();
  Cell new_cell = prev_cell;
  new_cell.from_matrix = from_matrix;
  new_cell.prev_pos = prev_pos;

  for (int l = 0; l < num_out_v; ++l) {
    new_cell.vertex_id = curr_v->out_edges[l].to_vertex;
    new_cell.diag = new_diag + curr_v->out_edges[l].overlap;
    Graph::vertex* new_v = &_graph._vertices[new_cell.vertex_id];
    vd_activate_vertex(*_qs, new_cell.vertex_id);

    // Store jump and metadata
    bool valid_diag = vd_valid_diagonal(*_qs, Cell::Matrix::M, new_cell.vertex_id, new_cell.diag); // Extend only if it has not yet been visited
    if (valid_diag) { // Extend only if it has not yet been visited
      int pos_new_cell = bs_push_back(*_qs, _qs->bs_m_jumps_wf, _qs->bs_m_jumps_wf_size, new_cell);
      vd_jumps_push(*_qs, vd_m_jumps(*_qs, new_cell.vertex_id, pos_score), vd_m_jumps_size(*_qs, new_cell.vertex_id, pos_score), pos_new_cell);
      extend_diagonal(new_v, _qs->bs_m_jumps_wf[pos_new_cell], new_cell.vertex_id, _qs->bs_m_jumps_wf[pos_new_cell], pos_new_cell, Cell::Matrix::MJumps);
    }
  }
}


// Store insertion-jump transitions in neighbouring vertices.
// HOTSPOT: this fan-out can dominate runtime on dense/branchy graphs.
void TheseusAlignerImpl::store_I_jump(Graph::vertex* curr_v,
                                      Cell& prev_cell,
                                      Cell::pos_t prev_pos,
                                      Cell::Matrix from_matrix) {

  struct IJumpFrame {
    int vertex_id;
    Cell prev_cell;
    Cell::pos_t prev_pos;
    Cell::Matrix from_matrix;
    int next_edge;
    bool invalidated;
  };

  IJumpFrame stack[kMaxIJumpStack];
  int stack_size = 1;
  stack[0] = IJumpFrame{prev_cell.vertex_id, prev_cell, prev_pos, from_matrix, 0, false};

  while (stack_size > 0) {
    IJumpFrame &frame = stack[stack_size - 1];
    Graph::vertex *frame_v = &_graph._vertices[frame.vertex_id];

    if (!frame.invalidated) {
      vd_invalidate_i_jump(*_qs, vd_get_id(*_qs, frame.prev_cell.vertex_id), frame.prev_cell.diag);
      frame.invalidated = true;
    }

    if (frame.next_edge >= frame_v->out_edges.size()) {
      --stack_size;
      continue;
    }

    const auto &edge = frame_v->out_edges[frame.next_edge];
    ++frame.next_edge;

    int pos_score = vd_get_pos(*_qs, _score);
    int new_diag = -frame.prev_cell.offset;
    Cell new_cell = frame.prev_cell;
    new_cell.from_matrix = frame.from_matrix;
    new_cell.prev_pos = frame.prev_pos;
    new_cell.vertex_id = edge.to_vertex;
    new_cell.diag = new_diag + edge.overlap;
    vd_activate_vertex(*_qs, new_cell.vertex_id);

    bool valid_diag = vd_valid_diagonal(*_qs, Cell::Matrix::I, new_cell.vertex_id, new_cell.diag);
    if (valid_diag) {
      int pos_new_cell = bs_push_back(*_qs, _qs->bs_i_jumps_wf, _qs->bs_i_jumps_wf_size, new_cell);
      vd_jumps_push(*_qs, vd_i_jumps(*_qs, new_cell.vertex_id, pos_score), vd_i_jumps_size(*_qs, new_cell.vertex_id, pos_score), pos_new_cell);

      Graph::vertex *new_v = &_graph._vertices[new_cell.vertex_id];
      if (new_v->value.size() == 0) {
        if (stack_size >= kMaxIJumpStack) {
          cap_fail(*_qs, kCapIJumpStack, stack_size + 1, kMaxIJumpStack);
          continue;
        }
        stack[stack_size] = IJumpFrame{new_cell.vertex_id, _qs->bs_i_jumps_wf[pos_new_cell], frame.prev_pos, Cell::Matrix::IJumps, 0, false};
        ++stack_size;
      }
    }
  }

  (void)curr_v;
}


// Check and store I jumps (that is, those diagonals that have reached the last column of a vertex)
void TheseusAlignerImpl::check_and_store_jumps(Graph::vertex *curr_v,
                                               Cell *curr_wavefront,
                                               Range cell_range)
{

  Cell::pos_t len = cell_range.end - cell_range.start, diag, offset, curr_j, n = curr_v->value.size(), prev_pos;
  Cell::Matrix from_matrix;

  for (int l = 0; l < len; ++l) {
    diag = curr_wavefront[cell_range.start + l].diag;
    offset = curr_wavefront[cell_range.start + l].offset;
    curr_j = diag + offset;
    if (curr_j == n && offset <= _seq.size()) {
      from_matrix = curr_wavefront[cell_range.start + l].from_matrix;
      prev_pos = curr_wavefront[cell_range.start + l].prev_pos;
      store_M_jump(curr_v, curr_wavefront[cell_range.start + l], prev_pos, from_matrix);
      store_I_jump(curr_v, curr_wavefront[cell_range.start + l], prev_pos, from_matrix);
    }
  }
}


// Compute the Longest Common Prefix between two given sequences
void TheseusAlignerImpl::LCP(std::string_view query,
                             std::string &vertex_text,
                             int &offset,
                             int &j) {

    // Find LCP
    int len_seq_1 = query.size();
    int len_seq_2 = vertex_text.size();
    while (offset < len_seq_1 && j < len_seq_2 && query[offset] == vertex_text[j]) {
        offset = offset + 1;   // Update the f.r. of this diagonal
        j = j + 1;
    }
}


// TODO: Implement different end conditions as Global, Semi-Global...
void TheseusAlignerImpl::check_end_condition(Cell curr_data, // Offset and prev_index
                        int j,
                        int v) {

  (void)j;
  (void)v;
  if (curr_data.offset == _seq.size()) {
    _end = true;
    _start_pos = curr_data;
  }
}


// Extend a particular diagonal
void TheseusAlignerImpl::extend_diagonal(
    Graph::vertex *curr_v,
    Cell &curr_cell,
    int v,
    Cell &prev_cell,
    Cell::pos_t prev_pos,
    Cell::Matrix from_matrix) {

  // Longest Common prefix
  int j = curr_cell.diag + curr_cell.offset;
  LCP(_seq, curr_v->value, curr_cell.offset, j); // Find Longest Common Prefix

  // End condition
  check_end_condition(curr_cell, j, v); // Check end condition

  // Check jump
  if (j == curr_v->value.size() && curr_cell.offset <= _seq.size() && curr_v->out_edges.size() > 0) {
    store_M_jump(curr_v, prev_cell, prev_pos, from_matrix); // Store the jump in neighbours
  }
}


// Add matches to our backtracking vector
void TheseusAlignerImpl::add_matches(
    int start_matches,
    int end_matches) {

  int size = end_matches - start_matches;
  for (int k = 0; k < size; ++k) {
    _alignment.edit_op.push_back('M');
  }
}


// Add a mismatch to our backtracking vector
void TheseusAlignerImpl::add_mismatch()
{
  _alignment.edit_op.push_back('X');
}


// Add an insertion to our backtracking vector
void TheseusAlignerImpl::add_insertion()
{
  _alignment.edit_op.push_back('I');
}


// Add a deletion to our backtracking vector
void TheseusAlignerImpl::add_deletion()
{
  _alignment.edit_op.push_back('D');
}


void TheseusAlignerImpl::one_backtrace_step(
    Cell &curr_cell) {

  // Get previous cell
  // _score -= curr_cell.score_diff;
  Cell prev_cell;
  if (curr_cell.from_matrix == Cell::Matrix::M) prev_cell = _qs->bs_m_wf[curr_cell.prev_pos];
  else if (curr_cell.from_matrix == Cell::Matrix::MJumps) prev_cell = _qs->bs_m_jumps_wf[curr_cell.prev_pos];
  else prev_cell = _qs->bs_i_jumps_wf[curr_cell.prev_pos];

  // Inside the same vertex or jump
  int num_indels;
  if (curr_cell.vertex_id == prev_cell.vertex_id) { // Still in the same vertex
    if (curr_cell.diag == prev_cell.diag) {                                               // Mismatch
      if (curr_cell.offset > prev_cell.offset) {    // Consider 0 length vertices
        add_matches(prev_cell.offset + 1, curr_cell.offset);
        add_mismatch();
      }
    }
    else { // Indel
      if (curr_cell.diag < prev_cell.diag) {                                              // Deletion
        num_indels = prev_cell.diag - curr_cell.diag;
        add_matches(prev_cell.offset + num_indels, curr_cell.offset);
        for (int l = 0; l < num_indels; ++l) add_deletion();
      }
      else {                                                                                      // Insertion
        num_indels = curr_cell.diag - prev_cell.diag;
        add_matches(prev_cell.offset, curr_cell.offset);
        for (int l = 0; l < num_indels; ++l) add_insertion();
      }
    }
  }
  else {                                            // Jump
    add_matches(prev_cell.offset, curr_cell.offset);                    // Add the necessary matches
    _alignment.path.push_back(prev_cell.vertex_id); // Add the new vertex to the path
    int col_in_prev_v = prev_cell.diag + prev_cell.offset;
    int num_insertions = _graph._vertices[prev_cell.vertex_id].value.size() - col_in_prev_v;
    for (int l = 0; l < num_insertions; ++l) add_insertion();                   // Add the necessary insertions
  }

  curr_cell = prev_cell;
}


// Main function of the backtracking process
void TheseusAlignerImpl::backtrace(int initial_vertex)
{

  Cell curr_pos = _start_pos;
  _alignment.start_offset = _start_offset;
  _alignment.query_length = static_cast<int>(_seq.size());
  _alignment.end_offset = curr_pos.diag + curr_pos.offset; // Vertex offset = j
  _alignment.path.push_back(curr_pos.vertex_id);
  while (curr_pos.prev_pos != -1)
  {
    one_backtrace_step(curr_pos);
  }

  add_matches(0, curr_pos.offset); // Add the matches until the beginning of the sequence

  std::reverse(_alignment.edit_op.begin(), _alignment.edit_op.end());
  std::reverse(_alignment.path.begin(), _alignment.path.end());
}


// Output functions
// Print as GFA
void TheseusAlignerImpl::print_as_gfa(std::ofstream &out_stream) {
  _graph.print_as_gfa(out_stream);
}


// Print as GAF
void TheseusAlignerImpl::print_as_gaf(
    theseus::Alignment &alignment,
    std::ostream &out_stream,
    std::string seq_name) {

  // Field 1: Query name
  out_stream << seq_name;

  // Field 2: Query length
  out_stream << "\t" << alignment.query_length;

  // Field 3: Query start
  out_stream << "\t" << 0;

  // Field 4: Query end
  out_stream << "\t" << alignment.query_length;

  // Field 5: Strand
  out_stream << "\t" << "+"; // TODO: Support reverse strand

  // Field 6: Alignment path
  out_stream << "\t";
  for (int l = 0; l < alignment.path.size(); ++l) {
    out_stream << ">" << _graph._vertices[alignment.path[l]].name; // TODO: Support orientation
  }

  // Field 7: Target length
  int target_length = 0;
  for (int l = 0; l < alignment.path.size(); ++l) {
    target_length += _graph._vertices[alignment.path[l]].value.size();
  }
  out_stream << "\t" << target_length;

  // Field 8: Target start
  out_stream << "\t" << alignment.start_offset;

  // Field 9: Target end
  out_stream << "\t" << alignment.end_offset;

  // Field 10: Number of matching bases
  int num_matches = 0;
  for (int l = 0; l < alignment.edit_op.size(); ++l) {
    if (alignment.edit_op[l] == 'M') {
      num_matches += 1;
    }
  }
  out_stream << "\t" << num_matches;

  // Field 11: Alignment block length
  out_stream << "\t" << alignment.edit_op.size();

  // Field 12: Mapping quality
  out_stream << "\t" << 255; // TODO: Compute mapping quality

  // Optional fields
  out_stream << "\t" << "cg:Z:"; // CIGAR string
  std::string cigar = "";
  int count = 1;
  for (int l = 1; l < alignment.edit_op.size(); ++l) {
    if (alignment.edit_op[l] == alignment.edit_op[l - 1]) {
      count += 1;
    }
    else {
      cigar += std::to_string(count) + alignment.edit_op[l - 1];
      count = 1;
    }
  }
  if (alignment.edit_op.size() > 0) {
    cigar += std::to_string(count) + alignment.edit_op.back();
  }
  out_stream << cigar << "\n";
}

} // namespace theseus
