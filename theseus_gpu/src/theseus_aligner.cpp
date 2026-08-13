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


#include "theseus/theseus_aligner.h"

#include <cstdint>
#include <chrono>

#include "gpu/align_gpu.h"
#include "theseus_aligner_impl.h"

namespace theseus {

int scratchpad_span() { return kScratchpadSpan; }

TheseusAligner::TheseusAligner(const Penalties &penalties,
                               std::istream &gfa_stream)
{
    Graph graph(gfa_stream);
    aligner_impl_ = std::make_unique<TheseusAlignerImpl>(penalties, std::move(graph));
}


TheseusAligner::~TheseusAligner() {}

void TheseusAligner::print_alignment_as_gaf(
                theseus::Alignment &alignment,
                std::ostream &out_stream,
                std::string seq_name) {

    aligner_impl_->print_as_gaf(alignment, out_stream, seq_name);
}

/**
 * @brief Main alignment function for the Theseus aligner.
 *
 * @param seq
 * @param start_node
 * @param start_offset
 * @return Alignment
 */
Alignment TheseusAligner::align(
    std::string_view seq,
    std::string &start_node,
    int start_offset) {

    return aligner_impl_->align(seq, start_node, start_offset);
}

Alignment TheseusAligner::align_gpu(
    std::string_view seq,
    std::string &start_node,
    int start_offset) {

    // A single sequence never justifies a host-to-device transfer, so there is
    // no device path here. align_batch_gpu is where the GPU work belongs.
    return aligner_impl_->align(seq, start_node, start_offset);
}

namespace {

/**
 * @brief Read the uploaded graph back and check it against the host CSR.
 *
 * @param device_graph  Graph in device memory
 * @param host_csr      What the graph should be
 * @return              Description of the outcome, for GpuBatchReport::message
 */
std::string verify_device_graph(gpu::DeviceGraph *device_graph,
                                const gpu::GraphCsr &host_csr) {

    std::vector<char> chars(host_csr.num_chars());
    std::vector<int32_t> vertex_offsets(host_csr.num_vertices() + 1);
    std::vector<int32_t> edge_targets(host_csr.num_edges());
    std::vector<int32_t> edge_overlaps(host_csr.num_edges());
    std::vector<int32_t> edge_offsets(host_csr.num_vertices() + 1);

    const gpu::Status status =
        gpu::readback_graph(device_graph, chars.data(), vertex_offsets.data(),
                            edge_targets.data(), edge_overlaps.data(),
                            edge_offsets.data());
    if (status != gpu::Status::Ok) {
        return std::string("; graph readback failed: ") + gpu::status_message(status);
    }

    const bool matches = chars == host_csr.vertex_chars() &&
                         vertex_offsets == host_csr.vertex_offsets() &&
                         edge_targets == host_csr.edge_targets() &&
                         edge_overlaps == host_csr.edge_overlaps() &&
                         edge_offsets == host_csr.edge_offsets();
    if (!matches) {
        return "; GRAPH CSR MISMATCH, device cannot read the graph";
    }

    return "; graph CSR verified on device (" +
           std::to_string(host_csr.num_vertices()) + " vertices, " +
           std::to_string(host_csr.num_chars()) + " bases, " +
           std::to_string(host_csr.num_edges()) + " edges)";
}

bool same_align_result(const gpu::AlignResult &a, const gpu::AlignResult &b) {
    return a.score == b.score &&
           a.end_vertex_id == b.end_vertex_id &&
           a.end_offset == b.end_offset &&
           a.end_diag == b.end_diag &&
           a.end_prev_pos == b.end_prev_pos &&
           a.end_from_matrix == b.end_from_matrix &&
           a.reached_end == b.reached_end &&
           a.capacity_exceeded == b.capacity_exceeded;
}

std::string describe_align_result_mismatch(size_t idx, const gpu::AlignResult &gpu,
                                           const gpu::AlignResult &cpu) {
    return "; GPU ALIGN RESULT MISMATCH at query " + std::to_string(idx) +
           " gpu(score=" + std::to_string(gpu.score) +
           ",v=" + std::to_string(gpu.end_vertex_id) +
           ",off=" + std::to_string(gpu.end_offset) +
           ",diag=" + std::to_string(gpu.end_diag) +
           ",prev=" + std::to_string(gpu.end_prev_pos) +
           ",mat=" + std::to_string(gpu.end_from_matrix) +
           ",end=" + std::to_string(gpu.reached_end) +
           ",cap=" + std::to_string(gpu.capacity_exceeded) +
           ") cpu(score=" + std::to_string(cpu.score) +
           ",v=" + std::to_string(cpu.end_vertex_id) +
           ",off=" + std::to_string(cpu.end_offset) +
           ",diag=" + std::to_string(cpu.end_diag) +
           ",prev=" + std::to_string(cpu.end_prev_pos) +
           ",mat=" + std::to_string(cpu.end_from_matrix) +
           ",end=" + std::to_string(cpu.reached_end) +
           ",cap=" + std::to_string(cpu.capacity_exceeded) + ")";
}

}  // namespace

/**
 * @brief Batched GPU alignment entry point.
 *
 * @param seqs
 * @param start_nodes
 * @param start_offsets
 * @param report
 * @return std::vector<Alignment>
 */
std::vector<Alignment> TheseusAligner::align_batch_gpu(
    const std::vector<std::string> &seqs,
    std::vector<std::string> &start_nodes,
    std::vector<int> &start_offsets,
    GpuBatchReport *report,
    int gpu_threads_per_block,
    bool verify_with_cpu) {

    const auto batch_start = std::chrono::steady_clock::now();
    const auto elapsed_ms = [](const auto &start) {
        return std::chrono::duration<double, std::milli>(
            std::chrono::steady_clock::now() - start).count();
    };

    // Pre-flight the one bound that is derivable rather than measured.
    // extend_diagonal computes j = diag + offset with j indexing a vertex and
    // offset indexing the query, so diag lives in [-max_query, max_vertex] and
    // the ScratchPad needs max_vertex + max_query + 1 diagonals. Checking it here
    // turns "the kernel came back empty" into a number the caller can act on.
    if (report != nullptr && !seqs.empty()) {
        size_t max_query = 0;
        for (const auto &seq : seqs) {
            max_query = std::max(max_query, seq.size());
        }
        const long required =
            static_cast<long>(aligner_impl_->max_vertex_length()) +
            static_cast<long>(max_query) + 1;
        if (required > kScratchpadSpan) {
            report->scratchpad_span_required = static_cast<int>(required);
            report->scratchpad_span_available = kScratchpadSpan;
        }
    }

    const auto prepare_start = std::chrono::steady_clock::now();
    // Flatten into the concatenated layout the device consumes.
    std::vector<char> chars;
    std::vector<int32_t> offsets;
    std::vector<int32_t> start_node_ids;
    std::vector<int32_t> start_offset_values;
    offsets.reserve(seqs.size() + 1);
    start_node_ids.reserve(start_nodes.size());
    start_offset_values.reserve(start_offsets.size());
    offsets.push_back(0);
    for (size_t i = 0; i < seqs.size(); ++i) {
        const auto &seq = seqs[i];
        chars.insert(chars.end(), seq.begin(), seq.end());
        offsets.push_back(static_cast<int32_t>(chars.size()));
        start_node_ids.push_back(aligner_impl_->graph_vertex_id(start_nodes[i]));
        start_offset_values.push_back(static_cast<int32_t>(start_offsets[i]));
    }
    if (report != nullptr) report->batch_prepare_ms = elapsed_ms(prepare_start);

    // Uploaded once per aligner, on the first GPU batch.
    const auto graph_start = std::chrono::steady_clock::now();
    gpu::DeviceGraph *device_graph = aligner_impl_->device_graph();
    if (report != nullptr) report->graph_prepare_ms = elapsed_ms(graph_start);

    gpu::BatchView view{chars.data(), offsets.data(),
                        static_cast<int32_t>(seqs.size())};
    // Page-locked and kept across batches. They are the batch's whole D2H
    // payload, and a fresh pageable allocation made the copy pay a fault per
    // page plus the driver's staging copy; see host_batch_buffers().
    const auto host_buffers_start = std::chrono::steady_clock::now();
    int32_t *device_lengths = nullptr;
    gpu::AlignResult *device_results = nullptr;
    CompactTracebackState *device_states = nullptr;
    std::vector<int32_t> fallback_lengths;
    std::vector<gpu::AlignResult> fallback_results;
    std::vector<CompactTracebackState> fallback_states;
    if (!aligner_impl_->host_batch_buffers(seqs.size(), &device_states,
                                           &device_results, &device_lengths)) {
        // Page locking failed. The batch still runs, just without the benefit.
        fallback_lengths.assign(seqs.size(), -1);
        fallback_results.resize(seqs.size());
        fallback_states.resize(seqs.size());
        device_lengths = fallback_lengths.data();
        device_results = fallback_results.data();
        device_states = fallback_states.data();
    }
    // The -1 is what the layout check below tests against, and the buffer is
    // reused, so a stale length from the previous batch would pass it.
    std::fill(device_lengths, device_lengths + seqs.size(), -1);
    if (report != nullptr) report->host_buffers_ms = elapsed_ms(host_buffers_start);
    gpu::AlignOptions options;
    options.threads_per_block = gpu_threads_per_block;

    const gpu::Status status = gpu::align_batch(
        view, device_graph, aligner_impl_->device_workspace(),
        start_node_ids.data(), start_offset_values.data(),
        aligner_impl_->gpu_scoring(), options, device_results,
        device_states, device_lengths);

    if (report != nullptr) {
        report->device_used = (status == gpu::Status::Ok ||
                               status == gpu::Status::NotImplemented);
        report->aligned_on_device = (status == gpu::Status::Ok);
        report->gpu_threads_per_block = gpu_threads_per_block;
        const gpu::TimingReport &timing = gpu::last_timing();
        report->graph_ms = timing.graph_ms;
        report->h2d_ms = timing.h2d_ms;
        report->kernel_ms = timing.kernel_ms;
        report->d2h_ms = timing.d2h_ms;
        report->end_to_end_ms = timing.end_to_end_ms;
        report->message = std::string("threads/block ") +
                          std::to_string(gpu_threads_per_block) + "; " +
                          gpu::status_message(status);

        const char *error = gpu::last_error();
        if (error != nullptr && error[0] != '\0') {
            report->message += std::string(" (") + error + ")";
        }

        if (report->device_used && verify_with_cpu) {
            bool layout_ok = true;
            for (size_t i = 0; i < seqs.size(); ++i) {
                if (device_lengths[i] != static_cast<int32_t>(seqs[i].size())) {
                    layout_ok = false;
                    break;
                }
            }
            report->message += layout_ok
                    ? "; batch layout verified on device"
                    : "; BATCH LAYOUT MISMATCH, upload is wrong";

            report->message +=
                verify_device_graph(device_graph, aligner_impl_->graph_csr());
        }
    }

    std::vector<Alignment> alignments;
    std::vector<gpu::AlignResult> cpu_results;
    if (status != gpu::Status::Ok || verify_with_cpu) {
        const auto cpu_start = std::chrono::steady_clock::now();
        alignments.reserve(seqs.size());
        cpu_results.reserve(seqs.size());
        for (size_t i = 0; i < seqs.size(); ++i) {
            alignments.push_back(
                aligner_impl_->align(seqs[i], start_nodes[i], start_offsets[i]));
            cpu_results.push_back(aligner_impl_->last_align_result());
        }
        if (report != nullptr) report->cpu_verification_ms = elapsed_ms(cpu_start);
    }

    bool use_gpu_backtrace = status == gpu::Status::Ok && !verify_with_cpu;
    if (status == gpu::Status::Ok && verify_with_cpu) {
        bool result_ok = true;
        size_t mismatch_idx = 0;
        for (size_t i = 0; i < seqs.size(); ++i) {
            if (!same_align_result(device_results[i], cpu_results[i])) {
                result_ok = false;
                mismatch_idx = i;
                break;
            }
        }
        use_gpu_backtrace = result_ok;
        if (report != nullptr) {
            report->message += result_ok
                ? "; align kernel result verified against CPU"
                : describe_align_result_mismatch(mismatch_idx, device_results[mismatch_idx],
                                                 cpu_results[mismatch_idx]);
        }
    }

    if (use_gpu_backtrace) {
        const auto construction_start = std::chrono::steady_clock::now();
        double traceback_ms = 0.0;
        alignments.clear();
        alignments.reserve(seqs.size());
        for (size_t i = 0; i < seqs.size(); ++i) {
            alignments.push_back(aligner_impl_->alignment_from_gpu_result(
                seqs[i], start_offsets[i], device_states[i], device_results[i],
                &traceback_ms));
        }
        if (report != nullptr) {
            const double reconstruction_ms = elapsed_ms(construction_start);
            report->host_traceback_ms = traceback_ms;
            report->alignment_construction_ms = reconstruction_ms - traceback_ms;
            report->result_from_device = true;
            report->message += "; GAF reconstructed from compact GPU state with host backtrace";
            report->message += "; timing_ms h2d=" + std::to_string(report->h2d_ms) +
                               " kernel=" + std::to_string(report->kernel_ms) +
                               " d2h=" + std::to_string(report->d2h_ms) +
                               " total=" + std::to_string(report->end_to_end_ms);
        }
    }

    if (report != nullptr && status == gpu::Status::Ok) {
        for (size_t i = 0; i < seqs.size(); ++i) {
            const CompactTracebackState &state = device_states[i];
            if (state.capacity_exceeded) {
                report->query_state_capacity_exceeded = true;
                report->wavefront_capacity_exceeded = true;
                report->message += "; DEVICE QUERYSTATE CAPACITY EXCEEDED";
                break;
            }
        }
    }

    // Checked after aligning, not before: the wavefronts only grow while the
    // algorithm runs. A fixed-size device buffer would have been overrun here.
    // One check for every fixed-capacity buffer in the flattened QueryState:
    // ScratchPad span, BeyondScope wavefronts and the Scope ring. Their sizes are
    // fixed, so an overflow would have meant a device buffer read or write past
    // its end. Checked after aligning, not before: the buffers only fill while
    // the algorithm runs.
    if (report != nullptr && report->scratchpad_span_required > 0) {
        report->message += "; SCRATCHPAD TOO SMALL FOR THIS BATCH: needs " +
                           std::to_string(report->scratchpad_span_required) +
                           " diagonals, kScratchpadSpan is " +
                           std::to_string(report->scratchpad_span_available);
    }

    if (report != nullptr && verify_with_cpu &&
        aligner_impl_->query_state_capacity_exceeded()) {
        report->query_state_capacity_exceeded = true;
        report->wavefront_capacity_exceeded = true;  // same underlying cause
        // Name the buffer that ran out first and what it wanted. Listing every
        // capacity told you nothing about which one was the problem.
        report->message +=
            std::string("; QUERYSTATE CAPACITY EXCEEDED: ") +
            cap_buffer_name(aligner_impl_->capacity_reason()) + " needed " +
            std::to_string(aligner_impl_->capacity_required()) + ", has " +
            std::to_string(aligner_impl_->capacity_available()) +
            "; peak per-score wavefront was " +
            std::to_string(aligner_impl_->peak_wavefront_capacity()) +
            " cells; raise the matching bound in query_state.h before this data "
            "can run on device";
    }

    if (report != nullptr) report->align_batch_host_ms = elapsed_ms(batch_start);
    return alignments;
}

} // namespace theseus
