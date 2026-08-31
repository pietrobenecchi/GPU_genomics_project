/**
 * @file align_batch.cu
 * @brief The batch entry point: what happens around a launch, not inside one.
 *
 * A batch is split into chunks that fit the device, and each chunk walks the
 * same five phases -- grow, upload, launch, read results, read traceback. Each
 * phase is a function that returns a Status, so a failure returns from where it
 * happened; the `goto cleanup` chain this replaces existed only because the
 * timing events had to be destroyed on every path, which ChunkEvents now does.
 *
 * No device code here: the kernels are reached through kernel_launch.h and the
 * allocations through device_memory.h.
 */

#include "gpu/align_gpu.h"
#include "gpu/device_memory.h"
#include "gpu/gpu_error.h"
#include "gpu/kernel_launch.h"
#include "query_state.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace theseus {
namespace gpu {

namespace {

TimingReport g_last_timing;

/**
 * @brief The four events a chunk is timed against, destroyed however it exits.
 *
 * A null event is one that could not be created. Timing is a diagnostic, so
 * that costs the chunk its report and nothing else -- which is why every use
 * below is guarded rather than asserted.
 */
struct ChunkEvents {
    cudaEvent_t start = nullptr;
    cudaEvent_t h2d = nullptr;
    cudaEvent_t kernel = nullptr;
    cudaEvent_t d2h = nullptr;

    ChunkEvents() {
        cudaEventCreate(&start);
        cudaEventCreate(&h2d);
        cudaEventCreate(&kernel);
        cudaEventCreate(&d2h);
    }

    ~ChunkEvents() {
        if (start != nullptr) cudaEventDestroy(start);
        if (h2d != nullptr) cudaEventDestroy(h2d);
        if (kernel != nullptr) cudaEventDestroy(kernel);
        if (d2h != nullptr) cudaEventDestroy(d2h);
    }

    ChunkEvents(const ChunkEvents &) = delete;
    ChunkEvents &operator=(const ChunkEvents &) = delete;

    static void record(cudaEvent_t ev) {
        if (ev != nullptr) cudaEventRecord(ev);
    }

    /** @brief All four exist, so the three intervals between them mean something. */
    bool complete() const {
        return start != nullptr && h2d != nullptr && kernel != nullptr &&
               d2h != nullptr;
    }
};

/** @brief One cudaMemcpy, with @p what recorded in the error slot if it fails. */
Status copy(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind,
            const char *what) {
    const cudaError_t err = cudaMemcpy(dst, src, bytes, kind);
    if (err != cudaSuccess) {
        set_error(what, err);
        return Status::CudaError;
    }
    return Status::Ok;
}

/** @brief One kernel launch's error, named for the error slot. */
Status launched(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        set_error(what, err);
        return Status::CudaError;
    }
    return Status::Ok;
}

/**
 * @brief The block size to launch with: what the options ask for when it is one
 * of the three the kernel is validated at, and 128 otherwise.
 */
int32_t resolve_threads_per_block(const AlignOptions &options) {
    return (options.threads_per_block == 64 || options.threads_per_block == 128 ||
            options.threads_per_block == 256)
               ? options.threads_per_block
               : 128;
}

/**
 * @brief The chunk's inputs, host to device.
 *
 * The QueryState array is deliberately *not* zeroed here. It used to be, at
 * 4.4 MB per query -- 8.8 GB of writes for a 2048-query batch, more than
 * everything else in the H2D window put together and, at 40.8 ms against a
 * 4.7 ms kernel, the largest single cost of a batch after the D2H.
 *
 * Nothing needs it. align_one establishes every scalar it reads
 * (sp_init_window, sc_init, vd_init_scalar, bs_new_alignment, the cap_*
 * diagnostics) and clears the two arrays that are indexed without a size --
 * sp_off over the window and vd_vertex_to_idx over the graph. Every other
 * array in the QueryState is append-only behind a counter that those calls
 * set to zero: a cell of bs_m_wf beyond bs_m_wf_size, an InvalidSeg beyond
 * vd_m_invalid_size[a], a Scope wavefront entry beyond sc_i_wf_size[s] is
 * never read, and vd_activate_vertex zeroes a vertex's own counters the
 * first time it becomes active.
 *
 * That is an argument, not a proof, so it is checked rather than trusted:
 * compute-sanitizer --tool initcheck reports a read of uninitialised device
 * memory, and the regression runs under it.
 */
Status upload_batch(const BatchView &batch, DeviceWorkspace *workspace,
                    const int32_t *start_node_ids,
                    const int32_t *start_offsets) {
    const size_t chars_bytes = static_cast<size_t>(batch.offsets[batch.num_seqs]);
    const size_t offsets_bytes =
        sizeof(int32_t) * (static_cast<size_t>(batch.num_seqs) + 1);
    const size_t per_query_bytes =
        sizeof(int32_t) * static_cast<size_t>(batch.num_seqs);

    Status status = copy(workspace->chars, batch.chars, chars_bytes,
                         cudaMemcpyHostToDevice, "cudaMemcpy(chars H2D)");
    if (status != Status::Ok) return status;
    status = copy(workspace->offsets, batch.offsets, offsets_bytes,
                  cudaMemcpyHostToDevice, "cudaMemcpy(offsets H2D)");
    if (status != Status::Ok) return status;
    status = copy(workspace->start_node_ids, start_node_ids, per_query_bytes,
                  cudaMemcpyHostToDevice, "cudaMemcpy(start_node_ids H2D)");
    if (status != Status::Ok) return status;
    return copy(workspace->start_offsets, start_offsets, per_query_bytes,
                cudaMemcpyHostToDevice, "cudaMemcpy(start_offsets H2D)");
}

/** @brief The lengths probe and the alignment itself, waited on. */
Status run_alignment(const BatchView &device_batch, const DeviceGraph *graph,
                     DeviceWorkspace *workspace, AlignScoring scoring,
                     int32_t threads_per_block) {
    Status status = launched(
        launch_seq_length(threads_per_block, workspace->offsets,
                          device_batch.num_seqs, workspace->lengths),
        "seq_length_kernel launch");
    if (status != Status::Ok) return status;

    status = launched(
        launch_align_batch(threads_per_block, device_batch, graph->view,
                           workspace->start_node_ids, workspace->start_offsets,
                           scoring, workspace->states, workspace->results),
        "theseus_align_batch_kernel launch");
    if (status != Status::Ok) return status;

    return launched(cudaDeviceSynchronize(),
                    "theseus_align_batch_kernel synchronize");
}

/** @brief The two fixed-size outputs: one length and one AlignResult per query. */
Status download_results(const BatchView &batch, DeviceWorkspace *workspace,
                        AlignResult *out_results, int32_t *out_seq_lengths) {
    const size_t n = static_cast<size_t>(batch.num_seqs);
    const Status status =
        copy(out_seq_lengths, workspace->lengths, sizeof(int32_t) * n,
             cudaMemcpyDeviceToHost, "cudaMemcpy(lengths D2H)");
    if (status != Status::Ok) return status;
    return copy(out_results, workspace->results, sizeof(AlignResult) * n,
                cudaMemcpyDeviceToHost, "cudaMemcpy(results D2H)");
}

/**
 * @brief Grow the three packing buffers to hold @p total_cells more cells.
 *
 * The device buffer and the offset array only ever have to fit the current
 * chunk, so they are replaced when too small. The host buffer accumulates every
 * chunk of the batch, so it grows geometrically and keeps what is already in it.
 */
Status ensure_packed_capacity(DeviceWorkspace *workspace, int32_t num_seqs,
                              size_t total_cells, size_t packed_host_used) {
    cudaError_t err = cudaSuccess;
    const size_t cells = total_cells > 0 ? total_cells : 1;
    if (workspace->packed_device_capacity < cells) {
        cudaFree(workspace->packed_device);
        workspace->packed_device = nullptr;
        workspace->packed_device_capacity = 0;
        err = cudaMalloc(&workspace->packed_device, sizeof(Cell) * cells);
        if (err != cudaSuccess) {
            set_error("cudaMalloc(packed traceback)", err);
            return Status::CudaError;
        }
        workspace->packed_device_capacity = cells;
    }
    if (workspace->pack_offsets_capacity < static_cast<size_t>(num_seqs)) {
        cudaFree(workspace->pack_offsets);
        workspace->pack_offsets = nullptr;
        workspace->pack_offsets_capacity = 0;
        err = cudaMalloc(&workspace->pack_offsets,
                         sizeof(int32_t) * static_cast<size_t>(num_seqs));
        if (err != cudaSuccess) {
            set_error("cudaMalloc(pack offsets)", err);
            return Status::CudaError;
        }
        workspace->pack_offsets_capacity = static_cast<size_t>(num_seqs);
    }
    // Room for what earlier chunks already appended plus this chunk.
    // The grow copies the existing prefix across rather than dropping
    // it: those cells are this batch's results, and the caller has not
    // seen them yet.
    if (workspace->packed_host_capacity < packed_host_used + total_cells) {
        size_t want = workspace->packed_host_capacity * 2;
        if (want < packed_host_used + total_cells) {
            want = packed_host_used + total_cells;
        }
        // Page-locked if it can be: this is the batch's whole traceback
        // payload now, and the copy is the reason the buffer exists.
        // Pageable is a correct fallback, only slower.
        void *host_buffer = nullptr;
        bool pinned = cudaHostAlloc(&host_buffer, sizeof(Cell) * want,
                                    cudaHostAllocDefault) == cudaSuccess;
        if (!pinned) {
            cudaGetLastError();   // the failed alloc is handled, not propagated
            host_buffer = std::malloc(sizeof(Cell) * want);
        }
        if (host_buffer == nullptr) {
            set_error("alloc(packed traceback host)", cudaErrorMemoryAllocation);
            return Status::CudaError;
        }
        if (workspace->packed_host != nullptr) {
            if (packed_host_used > 0) {
                std::memcpy(host_buffer, workspace->packed_host,
                            sizeof(Cell) * (packed_host_used));
            }
            if (workspace->packed_host_pinned) {
                cudaFreeHost(workspace->packed_host);
            } else {
                std::free(workspace->packed_host);
            }
        }
        workspace->packed_host = static_cast<Cell *>(host_buffer);
        workspace->packed_host_pinned = pinned;
        workspace->packed_host_capacity = want;
    }
    return Status::Ok;
}

/**
 * @brief The variable-size output: what each query's backtrace will read.
 *
 * Three passes over the chunk, in this order because each needs the one before
 * it: the metadata says how many cells every query produced, the prefix sum of
 * those sizes says where each query's slice goes, and only then can the device
 * pack into a buffer the right size and copy it back in one transfer.
 */
Status download_traceback(const BatchView &batch, DeviceWorkspace *workspace,
                          int32_t threads_per_block, void *out_query_states,
                          size_t *packed_host_used,
                          std::vector<size_t> *packed_cell_offsets) {
    std::vector<TracebackMeta> metadata(static_cast<size_t>(batch.num_seqs));
    Status status = copy(metadata.data(), workspace->traceback_meta,
                         sizeof(TracebackMeta) * metadata.size(),
                         cudaMemcpyDeviceToHost,
                         "cudaMemcpy(traceback metadata D2H)");
    if (status != Status::Ok) return status;

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

    // Pack on the device, copy once. The layout is the exclusive prefix
    // sum of the three sizes just read back, computed here and handed to
    // the kernel, so host and device agree without a second scan.
    size_t total_cells = 0;
    std::vector<int32_t> base(static_cast<size_t>(batch.num_seqs));
    for (int32_t i = 0; i < batch.num_seqs; ++i) {
        base[static_cast<size_t>(i)] = static_cast<int32_t>(total_cells);
        const TracebackMeta &meta = metadata[static_cast<size_t>(i)];
        total_cells += static_cast<size_t>(meta.m_size) +
                       static_cast<size_t>(meta.m_jumps_size) +
                       static_cast<size_t>(meta.i_jumps_size);
        if (total_cells > 0x7fffffffu) {
            set_error("packed traceback exceeds 2^31 cells", cudaSuccess);
            return Status::CudaError;
        }
    }

    status = ensure_packed_capacity(workspace, batch.num_seqs, total_cells,
                                    *packed_host_used);
    if (status != Status::Ok) return status;

    status = copy(workspace->pack_offsets, base.data(),
                  sizeof(int32_t) * base.size(), cudaMemcpyHostToDevice,
                  "cudaMemcpy(pack offsets H2D)");
    if (status != Status::Ok) return status;

    if (total_cells > 0) {
        status = launched(
            launch_pack_traceback(threads_per_block, workspace->states,
                                  batch.num_seqs, workspace->pack_offsets,
                                  workspace->packed_device),
            "pack_traceback_kernel launch");
        if (status != Status::Ok) return status;

        status = copy(workspace->packed_host + *packed_host_used,
                      workspace->packed_device, sizeof(Cell) * total_cells,
                      cudaMemcpyDeviceToHost,
                      "cudaMemcpy(packed traceback D2H)");
        if (status != Status::Ok) return status;
    }

    // Offsets, not pointers: a later chunk may move the buffer.
    // align_batch turns these into pointers once every chunk is in.
    for (int32_t i = 0; i < batch.num_seqs; ++i) {
        packed_cell_offsets->push_back(
            *packed_host_used + static_cast<size_t>(base[static_cast<size_t>(i)]));
    }
    *packed_host_used += total_cells;
    return Status::Ok;
}

}  // namespace

const TimingReport &last_timing() { return g_last_timing; }

/**
 * @brief One launch's worth of a batch.
 *
 * This is what align_batch used to be, with three differences that let the
 * entry point below call it more than once for the same batch:
 *
 * - @p chunk_capacity, not @p batch.num_seqs, sizes the QueryState array. The
 *   workspace stays grow-only and is reused by every chunk;
 * - the packed traceback cells are *appended* to the host staging buffer at
 *   @p packed_host_used rather than written from its start, and each query's
 *   slice is recorded in @p packed_cell_offsets as an offset rather than a
 *   pointer. Growing the buffer between chunks moves it, so the pointers can
 *   only be formed once every chunk has landed -- align_batch does that;
 * - timings accumulate into @p timing instead of replacing it.
 *
 * Every out_* pointer is already the chunk's slice.
 */
static Status align_chunk(const BatchView &batch,
                          const DeviceGraph *graph,
                          DeviceWorkspace *workspace,
                          const int32_t *start_node_ids,
                          const int32_t *start_offsets,
                          AlignScoring scoring,
                          AlignOptions options,
                          AlignResult *out_results,
                          void *out_query_states,
                          int32_t *out_seq_lengths,
                          int32_t chunk_capacity,
                          size_t *packed_host_used,
                          std::vector<size_t> *packed_cell_offsets,
                          TimingReport *timing) {
    if (batch.num_seqs <= 0) {
        return Status::NotImplemented;
    }
    if (graph == nullptr || workspace == nullptr) {
        return Status::NoDevice;
    }

    int device_count = 0;
    const cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        if (err != cudaSuccess) {
            set_error("cudaGetDeviceCount", err);
        }
        return Status::NoDevice;
    }

    const int32_t threads_per_block = resolve_threads_per_block(options);
    const size_t chars_bytes = static_cast<size_t>(batch.offsets[batch.num_seqs]);

    Status status =
        ensure_workspace_capacity(workspace, chunk_capacity, chars_bytes);
    if (status != Status::Ok) return status;

    ChunkEvents events;
    ChunkEvents::record(events.start);

    status = upload_batch(batch, workspace, start_node_ids, start_offsets);
    if (status != Status::Ok) return status;
    ChunkEvents::record(events.h2d);

    // The kernel reads the offsets from device memory, so the view it is handed
    // is the workspace's copy of the chunk, not the caller's.
    const BatchView device_batch{workspace->chars, workspace->offsets,
                                 batch.num_seqs};
    status = run_alignment(device_batch, graph, workspace, scoring,
                           threads_per_block);
    if (status != Status::Ok) return status;
    ChunkEvents::record(events.kernel);

    status = launched(
        launch_traceback_meta(threads_per_block, workspace->states,
                              batch.num_seqs, workspace->traceback_meta),
        "traceback_meta_kernel launch");
    if (status != Status::Ok) return status;

    status = download_results(batch, workspace, out_results, out_seq_lengths);
    if (status != Status::Ok) return status;

    if (out_query_states != nullptr) {
        status = download_traceback(batch, workspace, threads_per_block,
                                    out_query_states, packed_host_used,
                                    packed_cell_offsets);
        if (status != Status::Ok) return status;
    }

    if (events.d2h != nullptr) {
        cudaEventRecord(events.d2h);
        cudaEventSynchronize(events.d2h);
    }
    if (events.complete()) {
        // Accumulated, not assigned: a batch is one or more chunks and the
        // caller is told what the whole batch cost.
        float h2d = 0.0f, kernel = 0.0f, d2h = 0.0f, total = 0.0f;
        cudaEventElapsedTime(&h2d, events.start, events.h2d);
        cudaEventElapsedTime(&kernel, events.h2d, events.kernel);
        cudaEventElapsedTime(&d2h, events.kernel, events.d2h);
        cudaEventElapsedTime(&total, events.start, events.d2h);
        timing->h2d_ms += h2d;
        timing->kernel_ms += kernel;
        timing->d2h_ms += d2h;
        timing->end_to_end_ms += total;
    }
    return Status::Ok;
}

/**
 * @brief Queries one launch may hold, from what the device can actually spare.
 *
 * The QueryState array is the only part of the workspace that scales with the
 * batch -- 3.95 MB each against a few tens of bytes for everything else -- so
 * it alone decides how many queries fit, and no other term is worth modelling.
 *
 * The budget is the free memory plus whatever the state array already holds,
 * because that allocation is reusable rather than lost, minus a reserve for the
 * graph, the staging buffers, the context and fragmentation. Derived from
 * cudaMemGetInfo at call time, so it follows the device it is running on
 * instead of a constant tuned to one card.
 *
 * Never below what is already allocated: that memory is paid for. Never above
 * the batch: a batch that fits is still one launch, exactly as before.
 */
static int32_t chunk_capacity_for(const DeviceWorkspace *workspace,
                                  int32_t num_seqs) {
    // Test hook. Chunking is only reachable on its own terms by a batch large
    // enough to exhaust the device, which is an awkward thing to require of a
    // regression run; forcing the size lets the same batch be put through one
    // launch and through many and the two outputs compared directly. Ignored
    // unless set to a positive number, so it costs a getenv per batch and
    // changes nothing otherwise.
    if (const char *forced = std::getenv("THESEUS_GPU_CHUNK")) {
        const int value = std::atoi(forced);
        if (value > 0) {
            return value < num_seqs ? value : num_seqs;
        }
    }

    const size_t state_bytes = sizeof(QueryState);
    size_t free_bytes = 0, total_bytes = 0;
    size_t budget = state_bytes;   // one query rather than a guess, if the query fails
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        const size_t committed = workspace->batch_capacity * state_bytes;
        const size_t reserve = size_t{192} << 20;
        const size_t pool = free_bytes + committed;
        budget = pool > reserve ? pool - reserve : 0;
    } else {
        cudaGetLastError();   // handled, not propagated
    }
    size_t n = budget / state_bytes;
    if (n < 1) {
        n = 1;
    }
    if (n < workspace->batch_capacity) {
        n = workspace->batch_capacity;
    }
    if (n > static_cast<size_t>(num_seqs)) {
        n = static_cast<size_t>(num_seqs);
    }
    return static_cast<int32_t>(n);
}

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

    const int32_t chunk = chunk_capacity_for(workspace, batch.num_seqs);

    CompactTracebackState *host_states =
        static_cast<CompactTracebackState *>(out_query_states);
    size_t packed_host_used = 0;
    std::vector<size_t> packed_cell_offsets;
    if (host_states != nullptr) {
        packed_cell_offsets.reserve(static_cast<size_t>(batch.num_seqs));
    }

    // Chunks are consumed in order and each writes its own slice of the output
    // arrays, so the batch comes back in the caller's order whether it took one
    // launch or twenty.
    for (int32_t start = 0; start < batch.num_seqs; start += chunk) {
        const int32_t n = (batch.num_seqs - start) < chunk ? (batch.num_seqs - start)
                                                           : chunk;

        // The kernel reads batch.offsets[query_id] and indexes batch.chars from
        // it, so a chunk needs its offsets rebased to its own first character.
        std::vector<int32_t> chunk_offsets(static_cast<size_t>(n) + 1);
        const int32_t base_char = batch.offsets[start];
        for (int32_t i = 0; i <= n; ++i) {
            chunk_offsets[static_cast<size_t>(i)] = batch.offsets[start + i] - base_char;
        }
        const BatchView chunk_view{batch.chars + base_char, chunk_offsets.data(), n};

        const Status status = align_chunk(
            chunk_view, graph, workspace, start_node_ids + start,
            start_offsets + start, scoring, options, out_results + start,
            host_states != nullptr ? host_states + start : nullptr,
            out_seq_lengths + start, chunk, &packed_host_used,
            &packed_cell_offsets, &g_last_timing);
        if (status != Status::Ok) {
            return status;
        }
    }

    // Every chunk has landed, so the staging buffer will not move again and the
    // recorded offsets can become the pointers the caller reads.
    if (host_states != nullptr) {
        for (size_t i = 0; i < packed_cell_offsets.size(); ++i) {
            CompactTracebackState &host = host_states[i];
            const Cell *slice = workspace->packed_host + packed_cell_offsets[i];
            host.bs_m_wf = slice;
            host.bs_m_jumps_wf = slice + host.bs_m_wf_size;
            host.bs_i_jumps_wf = slice + host.bs_m_wf_size +
                                 host.bs_m_jumps_wf_size;
        }
    }
    return Status::Ok;
}

}  // namespace gpu
}  // namespace theseus
