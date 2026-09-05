// Allocazioni device che sopravvivono al singolo batch: grafo e workspace.
// Qui ci sono solo chiamate al runtime CUDA e contabilita': nessun kernel e'
// scritto in questo file, l'unico che lancia passa da kernel_launch.h.

#include "gpu/device_memory.h"
#include "gpu/gpu_error.h"
#include "gpu/kernel_launch.h"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdlib>

namespace theseus {
namespace gpu {

namespace {

// cudaMalloc + H2D di un array. Un array vuoto da' un puntatore nullo, che i
// kernel non dereferenziano mai perche' il range di offset corrispondente e' vuoto.
template <typename T>
bool upload_array(T **device_ptr, const T *host_ptr, size_t count,
                  const char *what) {
    *device_ptr = nullptr;
    if (count == 0) {
        return true;
    }

    cudaError_t err = cudaMalloc(device_ptr, sizeof(T) * count);
    if (err != cudaSuccess) {
        set_error(what, err);
        return false;
    }

    err = cudaMemcpy(*device_ptr, host_ptr, sizeof(T) * count, cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        set_error(what, err);
        return false;
    }
    return true;
}

// cudaMalloc di un array di output, riempito con un sentinella 0xFF.
// Il sentinella serve: se il buffer partisse gia' con i valori attesi, un
// kernel che non scrive niente risulterebbe comunque uguale al riferimento.
template <typename T>
bool alloc_output_array(T **device_ptr, size_t count, const char *what) {
    *device_ptr = nullptr;
    if (count == 0) {
        return true;
    }

    cudaError_t err = cudaMalloc(device_ptr, sizeof(T) * count);
    if (err != cudaSuccess) {
        set_error(what, err);
        return false;
    }

    err = cudaMemset(*device_ptr, 0xFF, sizeof(T) * count);
    if (err != cudaSuccess) {
        set_error(what, err);
        return false;
    }
    return true;
}

// D2H di un array.
template <typename T>
bool download_array(T *host_ptr, const T *device_ptr, size_t count,
                    const char *what) {
    if (count == 0) {
        return true;
    }

    cudaError_t err = cudaMemcpy(host_ptr, device_ptr, sizeof(T) * count,
                                 cudaMemcpyDeviceToHost);
    if (err != cudaSuccess) {
        set_error(what, err);
        return false;
    }
    return true;
}

// Gli array per-query del workspace, tenuti insieme.
// Una crescita alloca sette buffer e deve disfarli tutti se uno solo fallisce:
// raggrupparli con una release() evita di ripetere la pulizia a ogni errore.
struct GrownArrays {
    int32_t *offsets = nullptr;
    int32_t *start_node_ids = nullptr;
    int32_t *start_offsets = nullptr;
    AlignResult *results = nullptr;
    QueryState *states = nullptr;
    int32_t *lengths = nullptr;
    TracebackMeta *traceback_meta = nullptr;

    // Restituisce quello che era stato allocato. cudaFree(nullptr) non fa nulla.
    void release() {
        cudaFree(offsets);
        cudaFree(start_node_ids);
        cudaFree(start_offsets);
        cudaFree(results);
        cudaFree(states);
        cudaFree(lengths);
        cudaFree(traceback_meta);
    }
};

// cudaMalloc di bytes in ptr, registrando label nello slot d'errore se fallisce.
template <typename T>
bool grow(T **ptr, size_t bytes, const char *label) {
    const cudaError_t err = cudaMalloc(ptr, bytes);
    if (err != cudaSuccess) {
        set_error(label, err);
        return false;
    }
    return true;
}

}  // namespace

void *alloc_host_pinned(size_t bytes) {
    if (bytes == 0) {
        return nullptr;
    }
    void *buffer = nullptr;
    const cudaError_t err = cudaHostAlloc(&buffer, bytes, cudaHostAllocDefault);
    if (err != cudaSuccess) {
        set_error("cudaHostAlloc(host batch buffers)", err);
        return nullptr;
    }
    return buffer;
}

void free_host_pinned(void *buffer) {
    if (buffer != nullptr) {
        cudaFreeHost(buffer);
    }
}

DeviceWorkspace *create_workspace() { return new DeviceWorkspace(); }

void free_workspace(DeviceWorkspace *workspace) {
    if (workspace == nullptr) return;
    cudaFree(workspace->chars);
    cudaFree(workspace->offsets);
    cudaFree(workspace->start_node_ids);
    cudaFree(workspace->start_offsets);
    cudaFree(workspace->results);
    cudaFree(workspace->states);
    cudaFree(workspace->lengths);
    cudaFree(workspace->traceback_meta);
    cudaFree(workspace->packed_device);
    cudaFree(workspace->pack_offsets);
    if (workspace->packed_host != nullptr) {
        if (workspace->packed_host_pinned) {
            cudaFreeHost(workspace->packed_host);
        } else {
            std::free(workspace->packed_host);
        }
    }
    delete workspace;
}

Status ensure_workspace_capacity(DeviceWorkspace *workspace,
                                 int32_t chunk_capacity, size_t chars_bytes) {
    if (chars_bytes > workspace->query_capacity) {
        char *grown = nullptr;
        if (!grow(&grown, chars_bytes, "cudaMalloc(chars)")) {
            return Status::CudaError;
        }
        cudaFree(workspace->chars);
        workspace->chars = grown;
        workspace->query_capacity = chars_bytes;
    }

    if (static_cast<size_t>(chunk_capacity) <= workspace->batch_capacity) {
        return Status::Ok;
    }

    // Dimensioni di allocazione, non di copia: ogni buffer per-query si alloca
    // per un chunk pieno anche quando questo e' corto (l'ultimo lo e' sempre).
    // Altrimenti resta piccolo per il prossimo e il test di capacita' non se ne accorge.
    const size_t cap_q = static_cast<size_t>(chunk_capacity);
    // L'array delle QueryState e' tre ordini di grandezza sopra tutto il resto:
    // tenerlo alla capacita' del chunk fa riusare a ogni chunk la stessa
    // allocazione, e lo stesso azzeramento una tantum.
    const size_t states_bytes = sizeof(QueryState) * cap_q;

    GrownArrays grown;
    if (!grow(&grown.offsets, sizeof(int32_t) * (cap_q + 1),
              "cudaMalloc(offsets)") ||
        !grow(&grown.start_node_ids, sizeof(int32_t) * cap_q,
              "cudaMalloc(start_node_ids)") ||
        !grow(&grown.start_offsets, sizeof(int32_t) * cap_q,
              "cudaMalloc(start_offsets)") ||
        !grow(&grown.results, sizeof(AlignResult) * cap_q,
              "cudaMalloc(results)") ||
        !grow(&grown.states, states_bytes, "cudaMalloc(query_states)")) {
        grown.release();
        return Status::CudaError;
    }

    // cudaMalloc non azzera e sp_cleared deve leggersi 0 al primo uso: senza
    // memset initcheck segnala 73 758 letture non inizializzate su 8 query. La
    // regressione passa comunque, non toglierlo senza rilanciare initcheck.

    // Si azzera una volta per allocazione, non per batch: per batch costava
    // 4.4 MB per query, 8.6 GB e 40.8 ms su 2048, contro un kernel da 4.7 ms.
    const cudaError_t err = cudaMemset(grown.states, 0, states_bytes);
    if (err != cudaSuccess) {
        set_error("cudaMemset(query_states)", err);
        grown.release();
        return Status::CudaError;
    }

    if (!grow(&grown.lengths, sizeof(int32_t) * cap_q, "cudaMalloc(lengths)") ||
        !grow(&grown.traceback_meta, sizeof(TracebackMeta) * cap_q,
              "cudaMalloc(traceback_meta)")) {
        grown.release();
        return Status::CudaError;
    }

    cudaFree(workspace->offsets);
    cudaFree(workspace->start_node_ids);
    cudaFree(workspace->start_offsets);
    cudaFree(workspace->results);
    cudaFree(workspace->states);
    cudaFree(workspace->lengths);
    cudaFree(workspace->traceback_meta);
    workspace->offsets = grown.offsets;
    workspace->start_node_ids = grown.start_node_ids;
    workspace->start_offsets = grown.start_offsets;
    workspace->results = grown.results;
    workspace->states = grown.states;
    workspace->lengths = grown.lengths;
    workspace->traceback_meta = grown.traceback_meta;
    workspace->batch_capacity = cap_q;
    return Status::Ok;
}

DeviceGraph *upload_graph(const GraphCsrView &graph) {
    clear_error();

    int device_count = 0;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    if (err != cudaSuccess || device_count == 0) {
        set_error("cudaGetDeviceCount",
                  err != cudaSuccess ? err : cudaErrorNoDevice);
        return nullptr;
    }

    const int32_t num_vertices = graph.num_vertices;
    const int32_t num_chars = graph.vertex_offsets[num_vertices];
    const int32_t num_edges = graph.edge_offsets[num_vertices];

    DeviceGraph *device_graph = new DeviceGraph();
    device_graph->num_chars = num_chars;
    device_graph->num_edges = num_edges;

    const bool uploaded =
        upload_array(&device_graph->vertex_chars, graph.vertex_chars,
                     static_cast<size_t>(num_chars), "cudaMalloc/Memcpy(vertex_chars)") &&
        upload_array(&device_graph->vertex_offsets, graph.vertex_offsets,
                     static_cast<size_t>(num_vertices) + 1, "cudaMalloc/Memcpy(vertex_offsets)") &&
        upload_array(&device_graph->edge_targets, graph.edge_targets,
                     static_cast<size_t>(num_edges), "cudaMalloc/Memcpy(edge_targets)") &&
        upload_array(&device_graph->edge_overlaps, graph.edge_overlaps,
                     static_cast<size_t>(num_edges), "cudaMalloc/Memcpy(edge_overlaps)") &&
        upload_array(&device_graph->edge_offsets, graph.edge_offsets,
                     static_cast<size_t>(num_vertices) + 1, "cudaMalloc/Memcpy(edge_offsets)");

    if (!uploaded) {
        free_graph(device_graph);
        return nullptr;
    }

    device_graph->view.vertex_chars = device_graph->vertex_chars;
    device_graph->view.vertex_offsets = device_graph->vertex_offsets;
    device_graph->view.edge_targets = device_graph->edge_targets;
    device_graph->view.edge_overlaps = device_graph->edge_overlaps;
    device_graph->view.edge_offsets = device_graph->edge_offsets;
    device_graph->view.num_vertices = num_vertices;

    return device_graph;
}

void free_graph(DeviceGraph *graph) {
    if (graph == nullptr) {
        return;
    }
    cudaFree(graph->vertex_chars);
    cudaFree(graph->vertex_offsets);
    cudaFree(graph->edge_targets);
    cudaFree(graph->edge_overlaps);
    cudaFree(graph->edge_offsets);
    delete graph;
}

Status readback_graph(const DeviceGraph *graph,
                      char *out_vertex_chars,
                      int32_t *out_vertex_offsets,
                      int32_t *out_edge_targets,
                      int32_t *out_edge_overlaps,
                      int32_t *out_edge_offsets) {
    clear_error();

    if (graph == nullptr) {
        return Status::NoDevice;
    }

    const int32_t num_vertices = graph->view.num_vertices;
    const int32_t num_chars = graph->num_chars;
    const int32_t num_edges = graph->num_edges;

    if (num_vertices <= 0) {
        // Niente da rileggere, e un lancio con zero blocchi non e' valido.
        return Status::Ok;
    }

    char *d_chars = nullptr;
    int32_t *d_vertex_offsets = nullptr;
    int32_t *d_edge_targets = nullptr;
    int32_t *d_edge_overlaps = nullptr;
    int32_t *d_edge_offsets = nullptr;
    Status status = Status::Ok;
    cudaError_t err = cudaSuccess;

    // Unica via d'uscita: tutte le allocazioni qui sotto si liberano in `cleanup`.
    if (!alloc_output_array(&d_chars, static_cast<size_t>(num_chars),
                            "cudaMalloc(readback chars)") ||
        !alloc_output_array(&d_vertex_offsets, static_cast<size_t>(num_vertices) + 1,
                            "cudaMalloc(readback vertex_offsets)") ||
        !alloc_output_array(&d_edge_targets, static_cast<size_t>(num_edges),
                            "cudaMalloc(readback edge_targets)") ||
        !alloc_output_array(&d_edge_overlaps, static_cast<size_t>(num_edges),
                            "cudaMalloc(readback edge_overlaps)") ||
        !alloc_output_array(&d_edge_offsets, static_cast<size_t>(num_vertices) + 1,
                            "cudaMalloc(readback edge_offsets)")) {
        status = Status::CudaError;
        goto cleanup;
    }

    err = launch_graph_readback(graph->view, d_chars, d_vertex_offsets,
                                d_edge_targets, d_edge_overlaps, d_edge_offsets);
    if (err != cudaSuccess) {
        set_error("graph_readback_kernel launch", err);
        status = Status::CudaError;
        goto cleanup;
    }

    if (!download_array(out_vertex_chars, d_chars, static_cast<size_t>(num_chars),
                        "cudaMemcpy(readback chars D2H)") ||
        !download_array(out_vertex_offsets, d_vertex_offsets,
                        static_cast<size_t>(num_vertices) + 1, "cudaMemcpy(readback vertex_offsets D2H)") ||
        !download_array(out_edge_targets, d_edge_targets, static_cast<size_t>(num_edges),
                        "cudaMemcpy(readback edge_targets D2H)") ||
        !download_array(out_edge_overlaps, d_edge_overlaps, static_cast<size_t>(num_edges),
                        "cudaMemcpy(readback edge_overlaps D2H)") ||
        !download_array(out_edge_offsets, d_edge_offsets,
                        static_cast<size_t>(num_vertices) + 1, "cudaMemcpy(readback edge_offsets D2H)")) {
        status = Status::CudaError;
        goto cleanup;
    }

cleanup:
    cudaFree(d_chars);
    cudaFree(d_vertex_offsets);
    cudaFree(d_edge_targets);
    cudaFree(d_edge_overlaps);
    cudaFree(d_edge_offsets);
    return status;
}

}  // namespace gpu
}  // namespace theseus
