#pragma once

#include <cstddef>
#include <cstdint>

// Confine fra l'aligner C++ e il backend CUDA.
// Lo compilano sia il compilatore host sia nvcc, quindi qui stanno solo tipi
// POD e funzioni libere: nessun tipo CUDA.

namespace theseus {
namespace gpu {

// Vista piatta su un batch di query: sequenze concatenate piu' gli offset.
// Layout che il kernel legge direttamente e che fa salire tutto il batch con
// un solo transfer invece di uno per query.
struct BatchView {
    const char *chars;       // Sequenze concatenate, senza separatori
    const int32_t *offsets;  // num_seqs + 1 entry; la sequenza i e' [offsets[i], offsets[i + 1])
    int32_t num_seqs;
};

// Grafo in CSR, con solo cio' che il kernel legge: testo dei vertici e archi
// uscenti. Nomi e in-edges restano sull'host, servono per l'output GAF/GFA.
struct GraphCsrView {
    const char *vertex_chars;      // Testi dei vertici concatenati, senza separatori
    const int32_t *vertex_offsets; // num_vertices + 1; text of vertex v is [off[v], off[v + 1])
    const int32_t *edge_targets;   // Vertice di destinazione di ogni arco uscente
    const int32_t *edge_overlaps;  // Lunghezza dell'overlap, parallelo a edge_targets
    const int32_t *edge_offsets;   // num_vertices + 1; out-edges of v are [off[v], off[v + 1])
    int32_t num_vertices;
};

// Slot di risultato POD, uno per query. Volutamente separato da Cell e
// Alignment: sul confine CUDA passano solo campi primitivi, gli oggetti C++ li
// ricostruisce l'host. I campi end_* sono la Cell finale scelta dal wavefront.
struct AlignResult {
    int32_t score;
    int32_t end_vertex_id;
    int32_t end_offset;
    int32_t end_diag;
    int64_t end_prev_pos;
    int8_t end_from_matrix;
    int8_t reached_end;
    int8_t capacity_exceeded;
    int8_t reserved;
};

// Penalita' di allineamento in forma POD, come le vuole il kernel.
struct AlignScoring {
    int32_t mism;
    int32_t gapo;
    int32_t gape;
    int32_t nscores;
};

struct AlignOptions {
    int32_t threads_per_block = 128;
    int32_t collect_counters = 0;
};

struct TimingReport {
    float graph_ms = 0.0f;
    float h2d_ms = 0.0f;
    float kernel_ms = 0.0f;
    float d2h_ms = 0.0f;
    float end_to_end_ms = 0.0f;
};

// Handle opachi alle allocazioni device. Definiti solo dentro il backend, cosi'
// il codice compilato dall'host non vede mai un tipo CUDA.
struct DeviceGraph;
struct DeviceWorkspace;

enum class Status {
    Ok,              // Batch allineato sul device
    NotImplemented,  // Device reachable, but this build cannot run the requested GPU path
    NoDevice,        // Compilato con CUDA, ma a runtime nessun device usabile
    NotCompiled,     // Compilato senza il backend CUDA
    CudaError,       // Una chiamata CUDA e' fallita, vedi last_error()
};

inline const char *status_message(Status s) {
    switch (s) {
        case Status::Ok:
            return "aligned on device";
        case Status::NotImplemented:
            return "device reached, but requested GPU path is not implemented";
        case Status::NoDevice:
            return "no CUDA device available, fell back to CPU";
        case Status::NotCompiled:
            return "built without CUDA (-DTHESEUS_ENABLE_CUDA=ON to enable), fell back to CPU";
        case Status::CudaError:
            return "CUDA error, fell back to CPU";
    }
    return "unknown status";
}

// Descrizione dell'ultimo errore CUDA, "" se non ce n'e' stato nessuno.
const char *last_error();

const TimingReport &last_timing();

// Allinea un batch sul device: un blocco per query, ognuno con la sua
// QueryState. out_seq_lengths riporta le lunghezze calcolate sul device, cosi'
// il chiamante puo' verificare l'upload. Ok solo se ha girato davvero la GPU.
Status align_batch(const BatchView &batch,
                   const DeviceGraph *graph,
                   DeviceWorkspace *workspace,
                   const int32_t *start_node_ids,
                   const int32_t *start_offsets,
                   AlignScoring scoring,
                   AlignOptions options,
                   AlignResult *out_results,
                   void *out_query_states,
                   int32_t *out_seq_lengths);

DeviceWorkspace *create_workspace();
void free_workspace(DeviceWorkspace *workspace);

// Memoria host page-locked per i buffer che il device riempie: il traceback e'
// tutto il payload D2H e un buffer pageable pagherebbe page fault piu' staging.
// Allocata una volta per la vita dell'aligner. Senza CUDA ricade su malloc/free.
void *alloc_host_pinned(size_t bytes);
void free_host_pinned(void *buffer);

// Copia il grafo in memoria device una volta sola: e' read-only e condiviso da
// tutte le query, quindi non si rimanda a ogni batch.
// Ritorna un handle da liberare con free_graph(), nullptr se fallisce.
DeviceGraph *upload_graph(const GraphCsrView &graph);

// Libera un grafo ottenuto da upload_graph(). Accetta nullptr.
void free_graph(DeviceGraph *graph);

// Rilegge il grafo dalla memoria device. La copia la fa un kernel che percorre
// il CSR come fara' il kernel di allineamento: se non torna, il problema e' la
// traversata sul device, non solo dei byte copiati male.
Status readback_graph(const DeviceGraph *graph,
                      char *out_vertex_chars,
                      int32_t *out_vertex_offsets,
                      int32_t *out_edge_targets,
                      int32_t *out_edge_overlaps,
                      int32_t *out_edge_offsets);

}  // namespace gpu
}  // namespace theseus
