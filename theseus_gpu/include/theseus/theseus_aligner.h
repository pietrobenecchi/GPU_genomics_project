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
#include <istream>
#include <string>
#include <vector>

#include "theseus/penalties.h"
#include "theseus/alignment.h"

/**
 * @file theseus_aligner.h
 * @brief Header file for the TheseusAligner class. This class provides an interface
 * for aligning sequences to a graph given a starting position using the Theseus
 * algorithm.
 *
 *
 */

namespace theseus
{

    class TheseusAlignerImpl; // Forward declaration of the implementation class.

    // Diagonali che tiene la ScratchPad di una query: un batch ne vuole
    // max_vertex_length + max_query_length + 1, oltre non gira sul device.
    // Esposta per dimensionare un batch e per costruire input che la sforano.
    int scratchpad_span();

    // Cosa ha fatto davvero il backend con un batch. Ricade sulla CPU in
    // silenzio quando non ce la fa, quindi serve distinguere un run vero.
    struct GpuBatchReport
    {
        bool device_used = false;        // A CUDA kernel ran on this batch
        bool aligned_on_device = false;  // The alignment itself ran on the device
        // Gli allineamenti tornati vengono dalla QueryState del device. False vuol
        // dire che il kernel e' girato ma il risultato era inservibile e questi
        // sono quelli della CPU: output giusto, ma niente prove sul kernel.
        bool result_from_device = false;
        // I wavefront hanno sforato la capacita' fissa: innocuo sulla CPU, che
        // rialloca, ma questi dati su un buffer device ancora non ci girano.
        bool wavefront_capacity_exceeded = false;
        // Un buffer a capacita' fissa della QueryState non e' bastato. Stesso
        // significato di wavefront_capacity_exceeded, per quei buffer.
        bool query_state_capacity_exceeded = false;
        // Diagonali che vuole il batch contro quelle che tiene una QueryState,
        // riempite prima del lancio quando non ci sta; 0 vuol dire che ci sta.
        // Derivate dal grafo e dalle query: dicono a quanto alzare kScratchpadSpan.
        int scratchpad_span_required = 0;
        int scratchpad_span_available = 0;
        int gpu_threads_per_block = 128;
        float graph_ms = 0.0f;
        float h2d_ms = 0.0f;
        float kernel_ms = 0.0f;
        float d2h_ms = 0.0f;
        float end_to_end_ms = 0.0f;
        double batch_prepare_ms = 0.0;
        double graph_prepare_ms = 0.0;
        double host_buffers_ms = 0.0;
        double cpu_verification_ms = 0.0;
        double host_traceback_ms = 0.0;
        double alignment_construction_ms = 0.0;
        double align_batch_host_ms = 0.0;
        std::string message;             // Human readable backend status
    };

    class TheseusAligner
    {
    public:
        /**
         * Constructor
         *
         * @param penalties User defined alignment penalties
         * @param gfa_stream Input stream containing the graph in GFA format
         */
        TheseusAligner(const Penalties &penalties, std::istream &gfa_stream);

        /**
         * Class destructor
         *
         */
        ~TheseusAligner();

        /**
         * @brief Print the resulting alignment in GAF format.
         *
         * @param alignment Alignment to be printed
         * @param out_stream Output stream where the alignment will be printed
         */
        void print_alignment_as_gaf(
                theseus::Alignment &alignment,
                std::ostream &out_stream,
                std::string seq_name);

        /**
         * Main alignment function. Aligns the given sequence to the graph starting
         * from the specified node and offset.
         *
         * @param seq Sequence to be aligned
         * @param start_node Starting node in the graph
         * @param start_offset Starting offset within the starting node
         * @return Alignment
         */
        Alignment align(std::string_view seq,
                std::string &start_node,
                int start_offset = 0);

        /**
         * GPU alignment entry point for a single sequence.
         *
         * A lone sequence does not fill a device, so this always runs on the CPU
         * and exists only for API compatibility. Use align_batch_gpu instead.
         *
         * @param seq Sequence to be aligned
         * @param start_node Starting node in the graph
         * @param start_offset Starting offset within the starting node
         * @return Alignment
         */
        Alignment align_gpu(std::string_view seq,
                std::string &start_node,
                int start_offset = 0);

        // Entry point GPU per un batch di sequenze: si carica tutto una volta e
        // si assegna un blocco CUDA per query, cosi' il transfer si ammortizza
        // sul batch invece di pagarlo a sequenza.

        // Il kernel fa un allineamento completo per blocco e l'host ricostruisce
        // il GAF dallo stato tornato. @p verify_with_cpu fa girare anche la CPU
        // come oracolo esplicito. Gli allineamenti tornano nell'ordine di @p seqs.
        std::vector<Alignment> align_batch_gpu(
                const std::vector<std::string> &seqs,
                std::vector<std::string> &start_nodes,
                std::vector<int> &start_offsets,
                GpuBatchReport *report = nullptr,
                int gpu_threads_per_block = 128,
                bool verify_with_cpu = false);

    private:
        std::unique_ptr<TheseusAlignerImpl> aligner_impl_;
    };

} // namespace theseus
