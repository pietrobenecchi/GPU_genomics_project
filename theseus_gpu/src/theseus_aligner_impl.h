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

// Celle preallocate per ogni wavefront. PROVVISORIO: e' il valore che la CPU ha
// sempre usato, mai messo alla prova. Sul dataset giocattolo i wavefront arrivano
// a 4 celle, quindi da li' un bound vero non si ricava.

// Va derivato da lunghezza della query, dei vertici e loro numero, e misurato su
// un grafo realistico. Intanto capacity_exceeded() segnala quando si sfora.
constexpr int kProvisionalWavefrontCapacity = 1024;

class TheseusAlignerImpl {
public:
    TheseusAlignerImpl(const Penalties &penalties,
                       Graph &&graph);

    /**
     * @brief Releases the graph held in device memory, if any.
     */
    ~TheseusAlignerImpl();

    // La copia CSR del grafo, costruita una volta alla creazione.
    const gpu::GraphCsr &graph_csr() const { return *_graph_csr; }

    // Se gli allineamenti fatti finora hanno sforato la capacita' fissa dei
    // wavefront, vedi kProvisionalWavefrontCapacity.
    bool wavefront_capacity_exceeded() const { return _qs->capacity_exceeded; }

    // Se un buffer a capacita' fissa della QueryState non e' bastato per qualche
    // allineamento. Come wavefront_capacity_exceeded(): innocuo sulla CPU, fatale
    // su un buffer device, quindi non deve mai passare in silenzio.
    bool query_state_capacity_exceeded() const { return _qs->capacity_exceeded; }

    // Il wavefront per score piu' grande raggiunto finora, per ricavarne un bound.
    std::ptrdiff_t peak_wavefront_capacity() const { return _qs->sc_peak_wf; }

    // Quale buffer fisso e' finito per primo, e le entry che voleva contro quelle
    // che aveva.
    int8_t  capacity_reason() const { return _qs->cap_reason; }
    int32_t capacity_required() const { return _qs->cap_required; }
    int32_t capacity_available() const { return _qs->cap_available; }

    // Il grafo in memoria device, caricato al primo uso: differito cosi' un run
    // solo CPU non tocca mai la scheda. Una volta per aligner, non per batch,
    // perche' il grafo e' in sola lettura e lo condividono tutte le query.
    gpu::DeviceGraph *device_graph();
    gpu::DeviceWorkspace *device_workspace() const { return _device_workspace; }

    // I tre buffer host in cui align_batch scrive, page-locked e tenuti fra un
    // batch e l'altro. Erano std::vector per batch: 288 KB per query, cioe' 590 MB
    // mai toccati su 2048 query, piu' un page fault e una copia di staging a pagina.

    // Escono non inizializzati, align_batch li riempie tutti e tre per intero.
    // Torna false se l'allocazione fallisce: i puntatori restano nulli.
    bool host_batch_buffers(size_t queries,
                            CompactTracebackState **out_states,
                            gpu::AlignResult **out_results,
                            int32_t **out_lengths);

    // Lookup lato host, prima di lanciare un batch: i kernel ricevono id numerici,
    // i nomi dei vertici restano qui per il parsing e l'output.
    int32_t graph_vertex_id(const std::string &name);

    // Il vertice piu' lungo del grafo, meta' del bound della ScratchPad.
    int32_t max_vertex_length() const;

    // Copia POD delle penalita' interne, per il backend CUDA.
    gpu::AlignScoring gpu_scoring() const {
        return gpu::AlignScoring{_internal_penalties.mism(), _internal_penalties.gapo(),
                                 _internal_penalties.gape(), _qs->sc_nscores};
    }

    // Firma del risultato dell'ultimo allineamento CPU, nella forma dell'uscita
    // del kernel: cosi' l'host confronta i due percorsi prima di fidarsi.
    gpu::AlignResult last_align_result() const;

    // Ricostruisce un Alignment facendo girare il backtrace host sullo stato
    // compatto tornato dal kernel.
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

    // Page-locked host batch buffers, see host_batch_buffers(). _host_capacity
    // is in queries, and is 0 while nothing is allocated.
    size_t _host_capacity = 0;
    CompactTracebackState *_host_states = nullptr;
    gpu::AlignResult *_host_results = nullptr;
    int32_t *_host_lengths = nullptr;

    std::string_view _seq;

    Alignment _alignment;
};

}   // namespace theseus
