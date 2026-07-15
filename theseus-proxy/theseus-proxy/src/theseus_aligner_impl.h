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


#pragma once

#include <memory>
#include <string>
#include <queue>
#include <set>
#include <algorithm>

#include "theseus/alignment.h"
#include "theseus/penalties.h"

#include "graph.h"
#include "beyond_scope.h"
#include "cell.h"
#include "scope.h"
#include "scratchpad.h"
#include "vertices_data.h"
#include "wavefront.h"
#include "internal_penalties.h"

namespace theseus {

class TheseusAlignerImpl {
public:
    TheseusAlignerImpl(const Penalties &penalties,
                       Graph &&graph);

    /**
     * @brief Main alignment function. Aligns the given sequence to the graph
     * starting at the specified node and offset.
     *
     * @param seq               Sequence to be aligned
     * @param start_node        Starting node in the graph
     * @param start_offset      Starting offset within the starting node
     * @return                  Alignment object
     */
    Alignment align(std::string_view seq,
                    std::string &start_node,
                    int start_offset = 0);

    /**
     * @brief Output the current graph in GFA format.
     *
     * @param out_stream  Output stream to write the graph in GFA format
     */
    void print_as_gfa(std::ofstream &out_stream);

    /**
     * @brief Print the resulting alignment in GAF format.
     *
     * @param alignment Alignment to be printed
     * @param out_stream Output stream where the alignment will be printed
     */
    void print_as_gaf(
            theseus::Alignment &alignment,
            std::ostream &out_stream,
            std::string seq_name);

private:
    /**
     * @brief Initialize the data for a new alignment.
     *
     */
    void new_alignment();

    /**
     * @brief Process a given vertex at a given _score. This means performing
     * the next and extend operations.
     *
     * @param curr_v
     * @param v
     */
    void process_vertex(Graph::vertex *curr_v, int v);

    /**
     * @brief Compute the wave for a given score for all active vertices.
     *
     */
    void compute_new_wave();

    /**
     * @brief Sparsify the M data. This means storing the data in the scratchpad
     * to be later processed.
     *
     * @param curr_v
     * @param dense_wf
     * @param offset_increase
     * @param shift_factor
     * @param start_idx
     * @param end_idx
     * @param m
     * @param upper_bound
     * @param vertex_id
     * @param new_score_diff
     * @param prev_matrix
     */
    void sparsify_M_data(Cell::CellVector &dense_wf,
                         int offset_increase,
                         int shift_factor,
                         Scope::range cells_range,
                         int m,
                         int upper_bound);

    /**
     * @brief Sparsify the jumps data. This means storing the data in the scratchpad
     * to be later processed.
     *
     * @param curr_v
     * @param dense_wf
     * @param offset_increase
     * @param shift_factor
     * @param start_idx
     * @param end_idx
     * @param m
     * @param upper_bound
     * @param vertex_id
     * @param new_score_diff
     * @param prev_matrix
     */
    void sparsify_jumps_data(Cell::CellVector &dense_wf,
                             std::vector<Cell::pos_t> &jumps_positions,
                             int offset_increase,
                             int shift_factor,
                             int m,
                             int upper_bound,
                             Cell::Matrix from_matrix);

    /**
     * @brief Sparsify the indel (coming from I or D) data. This means storing
     * the data in the scratchpad to be later processed.
     *
     * @param curr_v
     * @param dense_wf
     * @param offset_increase
     * @param shift_factor
     * @param start_idx
     * @param end_idx
     * @param m
     * @param upper_bound
     * @param vertex_id
     * @param new_score_diff
     * @param prev_matrix
     */
    void sparsify_indel_data(Cell::CellVector &dense_wf,
                             int offset_increase,
                             int shift_factor,
                             Scope::range cells_range,
                             int m,
                             int upper_bound);

    /**
     * @brief Compute the next I matrix for a vertex v. This implies both sparsifying
     * the data in the scratchpad and storing it back on the new wavefront, once the
     * corresponding maximums and checks have been done.
     *
     * @param curr_v
     * @param upper_bound // Maximum value of the diagonal
     * @param v
     */
    void next_I(Graph::vertex *curr_v, int upper_bound, int v);


    /**
     * @brief Compute the next D matrix for a vertex v. This implies both sparsifying
     * the data in the scratchpad and storing it back on the new wavefront, once the
     * corresponding maximums and checks have been done.
     *
     * @param curr_v
     * @param upper_bound // Maximum value of the diagonal
     * @param v
     */
    void next_D(int upper_bound, int v);


    /**
     * @brief Compute the next M matrix for a vertex v. This implies both sparsifying
     * the data in the scratchpad and storing it back on the new wavefront, once the
     * corresponding maximums and checks have been done.
     *
     * @param curr_v
     * @param upper_bound // Maximum value of the diagonal
     * @param v
     */
    void next_M(int upper_bound, int v);

    /**
     * @brief Invalidate the diagonal associated to a jump in M, activate the newly
     * discovered vertices and store the jump in the neighbours.
     *
     * @param curr_v
     * @param prev_cell
     * @param prev_pos
     * @param prev_matrix
     * @param _score_diff
     */
    void store_M_jump(Graph::vertex *curr_v,
                      Cell &prev_cell,
                      Cell::pos_t prev_pos,
                      Cell::Matrix from_matrix);

    /**
     * @brief Invalidate the diagonal associated to a jump in I, activate the newly
     * discovered vertices and store the jump in the neighbours.
     *
     * @param curr_v
     * @param prev_cell
     * @param prev_pos
     * @param prev_matrix
     */
    void store_I_jump(Graph::vertex *curr_v,
                      Cell &prev_cell,
                      Cell::pos_t prev_pos,
                      Cell::Matrix from_matrix);

    /**
     * @brief Check and store I jumps (that is, those diagonals that have reached
     * the last column of a vertex for matrix I).
     *
     * @param curr_v
     * @param curr_wavefront
     * @param start_idx
     * @param end_idx
     * @param v
     */
    void check_and_store_jumps(Graph::vertex *curr_v,
                               Cell::CellVector &curr_wavefront,
                               Scope::range cell_range);

    /**
     * @brief Longest Common Prefix of two sequences.
     *
     * @param seq_1
     * @param seq_2
     * @param offset
     * @param j
     */
    void LCP(std::string_view query,
            std::string &vertex_text,
            int &offset,
            int &j);

    /**
     * @brief Check the end condition for the alignment.
     *
     * @param curr_data
     * @param j
     * @param v
     */
    void check_end_condition(Cell curr_data, int j, int v);

    /**
     * @brief Exyend a given diagonal for a given vertex and perform the necessary
     * jumps.
     *
     * @param curr_v
     * @param curr_cell
     * @param v
     * @param prev_cell
     * @param prev_pos
     * @param prev_matrix
     */
    void extend_diagonal(Graph::vertex *curr_v,
                         Cell &curr_cell,
                         int v,
                         Cell &prev_cell,
                         Cell::pos_t prev_pos,
                         Cell::Matrix from_matrix);

    /**
     * @brief Add matches to our backtracking vector.
     *
     * @param start_matches
     * @param end_matches
     */
    void add_matches(int start_matches, int end_matches);

    /**
     * @brief Add a mismatch to our backtracking vector.
     *
     */
    void add_mismatch();

    /**
     * @brief Add an insertion to our backtracking vector.
     *
     */
    void add_insertion();

    /**
     * @brief Add a deletion to our backtracking vector.
     *
     */
    void add_deletion();

    /**
     * @brief Perform a single step of the backtrace process.
     *
     * @param curr_cell
     * @param curr_v
     */
    void one_backtrace_step(Cell &curr_cell);

    /**
     * @brief Backtrace the alignment from the end vertex to the start vertex.
     *
     * @param initial_vertex
     */
    void backtrace(
        int initial_vertex);

    int32_t _score = 0;

    Penalties _penalties;
    InternalPenalties _internal_penalties;

    Graph _graph;   // The graph to align to

    bool _end = false;
    int _start_node;
    int _start_offset;
    Cell _start_pos;

    std::unique_ptr<ScratchPad> _scratchpad;

    std::unique_ptr<Scope> _scope;
    std::unique_ptr<BeyondScope> _beyond_scope;

    std::unique_ptr<VerticesData> _vertices_data;

    std::string_view _seq;

    Alignment _alignment;
};

}   // namespace theseus
