#pragma once

// Grafo e workspace, le due allocazioni che sopravvivono al singolo batch.
// Opache in align_gpu.h, qui visibili all'orchestrazione ma non ai kernel. Il
// workspace cresce e basta: a regime non si chiama piu' cudaMalloc.

#include "gpu/align_gpu.h"
#include "gpu/kernel_launch.h"
#include "query_state.h"

#include <cstddef>

namespace theseus {
namespace gpu {

// Il grafo in memoria device, piu' le dimensioni che servono per rileggerlo.
struct DeviceGraph {
    GraphCsrView view;  // I puntatori qui sotto, come li vedono i kernel
    char *vertex_chars = nullptr;
    int32_t *vertex_offsets = nullptr;
    int32_t *edge_targets = nullptr;
    int32_t *edge_overlaps = nullptr;
    int32_t *edge_offsets = nullptr;
    int32_t num_chars = 0;
    int32_t num_edges = 0;
};

struct DeviceWorkspace {
    char *chars = nullptr;
    int32_t *offsets = nullptr;
    int32_t *start_node_ids = nullptr;
    int32_t *start_offsets = nullptr;
    AlignResult *results = nullptr;
    QueryState *states = nullptr;
    int32_t *lengths = nullptr;
    TracebackMeta *traceback_meta = nullptr;
    size_t query_capacity = 0;
    size_t batch_capacity = 0;

    // Staging del traceback compattato: solo le celle che il backtrace legge,
    // piu' l'inizio della fetta di ogni query. Dimensionati sulle celle usate dal
    // batch, non su kBeyondScopeCapacity; qui il page lock si paga una volta.
    Cell *packed_device = nullptr;
    int32_t *pack_offsets = nullptr;      // device, one base per query
    size_t packed_device_capacity = 0;    // in celle
    size_t pack_offsets_capacity = 0;     // in query
    Cell *packed_host = nullptr;
    size_t packed_host_capacity = 0;      // in celle
    bool packed_host_pinned = false;
};

// Fa crescere il workspace a un chunk di chunk_capacity query con chars_bytes di
// testo, riusando cio' che basta gia'. Due test separati: il testo segue le
// sequenze, il resto la capacita'. Se fallisce lascia il workspace com'era.
Status ensure_workspace_capacity(DeviceWorkspace *workspace,
                                 int32_t chunk_capacity, size_t chars_bytes);

}  // namespace gpu
}  // namespace theseus
