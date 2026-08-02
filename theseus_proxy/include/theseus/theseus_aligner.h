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

    /**
     * @brief What the GPU backend actually did with a batch.
     *
     * The backend silently falls back to the CPU whenever it cannot do the work,
     * so callers need a way to tell a real device run from a fallback.
     */
    struct GpuBatchReport
    {
        bool device_used = false;        // A CUDA kernel ran on this batch
        bool aligned_on_device = false;  // The alignment itself ran on the device
        // The wavefronts outgrew their fixed capacity. Harmless on the CPU, which
        // reallocates, but it means this data cannot run on a device buffer yet.
        bool wavefront_capacity_exceeded = false;
        // A fixed-capacity buffer in the flattened QueryState (ScratchPad span,
        // BeyondScope wavefronts, ...) was too small. Same meaning as
        // wavefront_capacity_exceeded for those device-shaped buffers.
        bool query_state_capacity_exceeded = false;
        int gpu_config = 0;
        int gpu_threads_per_block = 128;
        float graph_ms = 0.0f;
        float h2d_ms = 0.0f;
        float kernel_ms = 0.0f;
        float d2h_ms = 0.0f;
        float end_to_end_ms = 0.0f;
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

        /**
         * GPU alignment entry point for a batch of sequences.
         *
         * This is the shape the device path needs: the whole batch is uploaded
         * once and one CUDA thread is assigned per query, so the transfer is amortised
         * across the batch instead of paid per sequence.
         *
         * The Version 0 kernel computes one complete serial alignment per CUDA thread.
         * The host validates kernel results against the CPU and reconstructs the
         * final GAF-compatible alignment from copied-back device state when the
         * signatures match. Pass @p report to find out what the backend did.
         *
         * @param seqs Sequences to be aligned
         * @param start_nodes Starting node in the graph, one per sequence
         * @param start_offsets Starting offset within the starting node, one per sequence
         * @param report Optional backend status, see GpuBatchReport
         * @return Alignments, in the same order as @p seqs
         */
        std::vector<Alignment> align_batch_gpu(
                const std::vector<std::string> &seqs,
                std::vector<std::string> &start_nodes,
                std::vector<int> &start_offsets,
                GpuBatchReport *report = nullptr,
                int gpu_config = 0,
                int gpu_threads_per_block = 128);

    private:
        std::unique_ptr<TheseusAlignerImpl> aligner_impl_;
    };

} // namespace theseus
