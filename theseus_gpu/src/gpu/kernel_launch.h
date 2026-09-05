#pragma once

// Ingresso host a tutti i kernel: la sintassi `<<<>>>` e' di nvcc, quindi resta
// chiusa in align_gpu.cu e il resto puo' essere C++ normale. Ogni wrapper lancia
// e ritorna cudaGetLastError(), non sincronizza, e decide lui la geometria.

#include "gpu/align_gpu.h"
#include "query_state.h"

#include <cuda_runtime.h>

namespace theseus {
namespace gpu {

// Quello che l'host rilegge da una QueryState finita per impaginare il
// traceback: le tre dimensioni dei wavefront piu' le diagnostiche.
// Piccola e a taglia fissa, quindi si copia in un colpo per tutto il chunk.
struct TracebackMeta {
    int32_t m_size;
    int32_t m_jumps_size;
    int32_t i_jumps_size;
    int32_t peak_wf;
    int32_t cap_required;
    int32_t cap_available;
    int8_t capacity_exceeded;
    int8_t cap_reason;
    int8_t reserved[2];
};

// Riporta la lunghezza di ogni sequenza: prova che gli offset sono
// sopravvissuti all'upload.
cudaError_t launch_seq_length(int32_t threads_per_block,
                              const int32_t *offsets, int32_t num_seqs,
                              int32_t *out_seq_lengths);

// L'allineamento vero: un blocco per query, batch.num_seqs blocchi, con la
// shared memory che la larghezza del blocco richiede.
cudaError_t launch_align_batch(int32_t threads_per_block,
                               const BatchView &batch,
                               const GraphCsrView &graph,
                               const int32_t *start_node_ids,
                               const int32_t *start_offsets,
                               AlignScoring scoring,
                               QueryState *states,
                               AlignResult *results);

// Raccoglie dimensioni del traceback e diagnostiche di count stati.
cudaError_t launch_traceback_meta(int32_t threads_per_block,
                                  const QueryState *states, int32_t count,
                                  TracebackMeta *metadata);

// Compatta in un unico buffer denso le celle che leggera' il backtrace host,
// agli offset base che l'host ha ricavato dai metadati qui sopra.
cudaError_t launch_pack_traceback(int32_t threads_per_block,
                                  const QueryState *states, int32_t count,
                                  const int32_t *base, Cell *packed);

// Rilegge il CSR dalla memoria device, un blocco per vertice.
cudaError_t launch_graph_readback(const GraphCsrView &graph,
                                  char *out_vertex_chars,
                                  int32_t *out_vertex_offsets,
                                  int32_t *out_edge_targets,
                                  int32_t *out_edge_overlaps,
                                  int32_t *out_edge_offsets);

}  // namespace gpu
}  // namespace theseus
