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
#include "cell.h"
#include "scratchpad.h"
#include "query_state.h"
#include "vertices_data.h"
#include "wavefront.h"
#include "internal_penalties.h"
#include "gpu/graph_csr.h"

namespace theseus {

/**
 * @brief Cells each wavefront is preallocated to hold.
 *
 * PROVISIONAL. Device code cannot reallocate, so every wavefront the GPU port
 * touches needs an upper bound fixed before the kernel launches. This number is
 * not that bound: it is the value the CPU has always used, and nothing has ever
 * tested it. Measured on the sample data (5 queries of 11-13 bases over an
 * 8-vertex graph) the wavefronts peak at 4 cells, so the value below is 256x
 * larger than that data can distinguish. Sizing a real bound from this dataset
 * would be meaningless.
 *
 * A real bound has to be derived from query length, vertex length and vertex
 * count, and measured on a realistic graph with real reads. Until then
 * Scope::capacity_exceeded() reports when the algorithm outgrows this value, so
 * that the day it happens is not the day a kernel silently reads past the end of
 * a buffer.
 *
 * The same constraint applies to BeyondScope, VerticesData and ScratchPad, which
 * also grow on demand and are not covered yet.
 */
constexpr int kProvisionalWavefrontCapacity = 1024;

class TheseusAlignerImpl {
public:
    TheseusAlignerImpl(const Penalties &penalties,
                       Graph &&graph);

    /**
     * @brief Releases the graph held in device memory, if any.
     */
    ~TheseusAlignerImpl();

    /**
     * @brief The CSR mirror of the graph, built once at construction.
     */
    const gpu::GraphCsr &graph_csr() const { return *_graph_csr; }

    /**
     * @brief Whether the alignments run so far outgrew the fixed wavefront
     * capacity, see kProvisionalWavefrontCapacity.
     */
    bool wavefront_capacity_exceeded() const { return _qs->capacity_exceeded; }

    /**
     * @brief Whether any fixed-capacity buffer in the flattened QueryState (the
     * ScratchPad span, the BeyondScope wavefronts, the Scope ring, ...) was too
     * small for some alignment. Same meaning as wavefront_capacity_exceeded():
     * harmless on the CPU, fatal for a device buffer, so it must never be silent.
     */
    bool query_state_capacity_exceeded() const { return _qs->capacity_exceeded; }

    /**
     * @brief Largest per-score wavefront size reached so far, to size a real
     * bound from.
     */
    std::ptrdiff_t peak_wavefront_capacity() const { return _qs->sc_peak_wf; }

    /** @brief Which fixed buffer ran out first, and the entries it wanted vs had. */
    int8_t  capacity_reason() const { return _qs->cap_reason; }
    int32_t capacity_required() const { return _qs->cap_required; }
    int32_t capacity_available() const { return _qs->cap_available; }

    /**
     * @brief The graph in device memory, uploaded on first use.
     *
     * Uploading is deferred rather than done at construction so that CPU-only
     * runs never touch the device. The upload happens once per aligner, not
     * once per batch: the graph is read-only and shared by every query.
     *
     * @return Handle owned by this object, or nullptr if no device is usable
     */
    gpu::DeviceGraph *device_graph();
    gpu::DeviceWorkspace *device_workspace() const { return _device_workspace; }

    /**
     * @brief Host-side lookup used before launching a GPU batch. Kernels receive
     * numeric vertex ids; graph names stay on the host for parsing and output.
     */
    int32_t graph_vertex_id(const std::string &name);

    /** @brief Longest vertex sequence in the graph; half of the ScratchPad bound. */
    int32_t max_vertex_length() const;

    /**
     * @brief POD copy of internal penalties for the CUDA backend.
     */
    gpu::AlignScoring gpu_scoring() const {
        return gpu::AlignScoring{_internal_penalties.mism(), _internal_penalties.gapo(),
                                 _internal_penalties.gape(), _qs->sc_nscores};
    }

    /**
     * @brief Result signature of the most recent CPU alignment, shaped like the
     * CUDA kernel output so the host can compare both paths before trusting the
     * device result for backtrace/output.
     */
    gpu::AlignResult last_align_result() const;

    /**
     * @brief Reconstruct an Alignment by running the existing host backtrace on
     * the compact traceback state copied back from the CUDA kernel.
     */
    Alignment alignment_from_gpu_result(std::string_view seq,
                                         int start_offset,
                                         const CompactTracebackState &state,
                                         const gpu::AlignResult &result,
                                         double *traceback_ms = nullptr);

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
    void sparsify_M_data(Cell *dense_wf,
                         int offset_increase,
                         int shift_factor,
                         Range cells_range,
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
    void sparsify_jumps_data(Cell *dense_wf,
                             Cell::pos_t *jumps_positions,
                             int jumps_size,
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
    void sparsify_indel_data(Cell *dense_wf,
                             int offset_increase,
                             int shift_factor,
                             Range cells_range,
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
                               Cell *curr_wavefront,
                               Range cell_range);

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
    void one_backtrace_step(Cell &curr_cell,
                            const TracebackWavefronts &wavefronts);

    /**
     * @brief Backtrace the alignment from the end vertex to the start vertex.
     *
     * @param initial_vertex
     * @param wavefronts  The BeyondScope wavefronts to walk: this aligner's own
     *                    QueryState after a CPU alignment, or the compact state
     *                    read back from the kernel.
     */
    void backtrace(int initial_vertex,
                   const TracebackWavefronts &wavefronts);

    int32_t _score = 0;

    Penalties _penalties;
    InternalPenalties _internal_penalties;

    Graph _graph;   // The graph to align to

    bool _end = false;
    int _start_node;
    int _start_offset;
    Cell _start_pos;

    // Per-query working set, being migrated to flat POD arrays one structure at
    // a time (see query_state.h). Heap-held because it is large. Currently holds
    // the flattened ScratchPad.
    std::unique_ptr<QueryState> _qs;

    std::unique_ptr<VerticesData> _vertices_data;

    std::unique_ptr<gpu::GraphCsr> _graph_csr;
    gpu::DeviceGraph *_device_graph = nullptr;
    gpu::DeviceWorkspace *_device_workspace = nullptr;
    bool _device_graph_attempted = false;   // Guards against retrying a failed upload

    std::string_view _seq;

    Alignment _alignment;
};

}   // namespace theseus
