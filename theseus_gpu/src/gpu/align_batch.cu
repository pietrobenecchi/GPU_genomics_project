// Entry point del batch: attorno al lancio, non dentro. Il batch e' spezzato in
// chunk che stanno nel device, ognuno con le stesse cinque fasi: cresci, carica,
// lancia, leggi i risultati, leggi il traceback. Qui non c'e' codice device.

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

// I quattro eventi con cui si cronometra un chunk, distrutti comunque esca.
// Un evento nullo e' uno che non si e' potuto creare: il timing e' solo una
// diagnostica, quindi ogni uso e' protetto da un controllo invece che asserito.
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

    // Ci sono tutti e quattro, quindi i tre intervalli hanno senso.
    bool complete() const {
        return start != nullptr && h2d != nullptr && kernel != nullptr &&
               d2h != nullptr;
    }
};

// Una cudaMemcpy, con what registrato nello slot d'errore se fallisce.
Status copy(void *dst, const void *src, size_t bytes, cudaMemcpyKind kind,
            const char *what) {
    const cudaError_t err = cudaMemcpy(dst, src, bytes, kind);
    if (err != cudaSuccess) {
        set_error(what, err);
        return Status::CudaError;
    }
    return Status::Ok;
}

// L'errore di un lancio di kernel, con il nome per lo slot d'errore.
Status launched(cudaError_t err, const char *what) {
    if (err != cudaSuccess) {
        set_error(what, err);
        return Status::CudaError;
    }
    return Status::Ok;
}

// Larghezza del blocco con cui lanciare: quella chiesta nelle opzioni se e' una
// delle tre su cui il kernel e' validato (64/128/256), altrimenti 128.
int32_t resolve_threads_per_block(const AlignOptions &options) {
    return (options.threads_per_block == 64 || options.threads_per_block == 128 ||
            options.threads_per_block == 256)
               ? options.threads_per_block
               : 128;
}

// Gli input del chunk, da host a device. Le QueryState non si azzerano qui: a
// ogni batch costava 4.4 MB per query (40.8 ms su 2048, kernel da 4.7 ms). Si
// azzera una volta per allocazione in ensure_workspace_capacity.
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

// La sonda sulle lunghezze e l'allineamento vero, con l'attesa del device.
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

// I due output a taglia fissa: una lunghezza e un AlignResult per query.
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

// Fa crescere i tre buffer di packing per altre total_cells celle. Device e
// offset tengono solo il chunk corrente, quindi si rimpiazzano; quello host
// accumula tutti i chunk, quindi cresce geometricamente e conserva.
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
    // Spazio per cio' che i chunk precedenti hanno gia' accodato piu' questo.
    // La crescita ricopia il prefisso invece di buttarlo: sono risultati di
    // questo batch che il chiamante non ha ancora visto.
    if (workspace->packed_host_capacity < packed_host_used + total_cells) {
        size_t want = workspace->packed_host_capacity * 2;
        if (want < packed_host_used + total_cells) {
            want = packed_host_used + total_cells;
        }
        // Page-locked se si puo': qui passa tutto il payload del traceback e la
        // copia e' il motivo per cui il buffer esiste. Pageable e' un fallback
        // corretto, solo piu' lento.
        void *host_buffer = nullptr;
        bool pinned = cudaHostAlloc(&host_buffer, sizeof(Cell) * want,
                                    cudaHostAllocDefault) == cudaSuccess;
        if (!pinned) {
            cudaGetLastError();   // l'alloc fallita e' gestita, non propagata
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

// L'output a taglia variabile: le celle che leggera' il backtrace. Tre passate
// sul chunk, ognuna serve alla successiva: i metadati danno le taglie, la prefix
// sum le posizioni, poi il device compatta e si copia in un transfer solo.
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

    // Si compatta sul device e si copia una volta sola. Il layout e' la prefix
    // sum esclusiva delle tre dimensioni appena rilette, calcolata qui e passata
    // al kernel: host e device sono d'accordo senza una seconda scansione.
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

    // Offset, non puntatori: un chunk successivo puo' spostare il buffer.
    // align_batch li trasforma in puntatori quando sono arrivati tutti.
    for (int32_t i = 0; i < batch.num_seqs; ++i) {
        packed_cell_offsets->push_back(
            *packed_host_used + static_cast<size_t>(base[static_cast<size_t>(i)]));
    }
    *packed_host_used += total_cells;
    return Status::Ok;
}

}  // namespace

const TimingReport &last_timing() { return g_last_timing; }

// Un chunk di batch, cioe' un solo lancio. E' richiamabile piu' volte sullo
// stesso batch: dimensiona su chunk_capacity, accoda il traceback al buffer host
// per offset (la crescita lo sposta) e somma i tempi. Gli out_* sono gia' fette.
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

    // Il kernel legge gli offset dalla memoria device, quindi la view che riceve
    // e' la copia del chunk nel workspace, non quella del chiamante.
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
        // Sommati, non assegnati: un batch e' uno o piu' chunk e al chiamante
        // interessa il costo dell'intero batch.
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

// Quante query stanno in un lancio, dalla memoria del device. Decide da sola
// l'array delle QueryState (3.95 MB l'una, il resto sono decine di byte). Budget
// = libera + quella gia' negli stati - riserva, da cudaMemGetInfo a ogni chiamata.
static int32_t chunk_capacity_for(const DeviceWorkspace *workspace,
                                  int32_t num_seqs) {
    // Hook di test: il chunking si attiverebbe solo con un batch che satura il
    // device, scomodo in regressione. Forzare la taglia fa passare lo stesso
    // batch da uno e da tanti lanci. Ignorato se non e' positivo.
    if (const char *forced = std::getenv("THESEUS_GPU_CHUNK")) {
        const int value = std::atoi(forced);
        if (value > 0) {
            return value < num_seqs ? value : num_seqs;
        }
    }

    const size_t state_bytes = sizeof(QueryState);
    size_t free_bytes = 0, total_bytes = 0;
    size_t budget = state_bytes;   // se la query fallisce, una query invece di un'ipotesi
    if (cudaMemGetInfo(&free_bytes, &total_bytes) == cudaSuccess) {
        const size_t committed = workspace->batch_capacity * state_bytes;
        const size_t reserve = size_t{192} << 20;
        const size_t pool = free_bytes + committed;
        budget = pool > reserve ? pool - reserve : 0;
    } else {
        cudaGetLastError();   // gestito, non propagato
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

    // I chunk si consumano in ordine e ognuno scrive la propria fetta degli
    // array di output, quindi il batch torna nell'ordine del chiamante sia che
    // sia servito un lancio sia che ne siano serviti venti.
    for (int32_t start = 0; start < batch.num_seqs; start += chunk) {
        const int32_t n = (batch.num_seqs - start) < chunk ? (batch.num_seqs - start)
                                                           : chunk;

        // Il kernel legge batch.offsets[query_id] e da li' indicizza batch.chars,
        // quindi gli offset del chunk vanno ribasati sul suo primo carattere.
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

    // Tutti i chunk sono arrivati: il buffer di staging non si muove piu' e gli
    // offset registrati possono diventare i puntatori che legge il chiamante.
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
