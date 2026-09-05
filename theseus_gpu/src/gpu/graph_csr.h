#pragma once

#include <cstdint>
#include <vector>

#include "graph.h"
#include "gpu/align_gpu.h"

// Copia CSR host-side di un Graph.
// Normale C++ host: possiede std::vector ed e' compilato dal compilatore host.
// Verso il backend CUDA passa solo la view() che espone.

namespace theseus {
namespace gpu {

class GraphCsr {
public:
    // Appiattisce graph in buffer CSR. Nomi dei vertici e in-edges vengono
    // scartati: l'allineamento non li legge, servono solo all'output host.
    explicit GraphCsr(const Graph &graph);

    // Vista sui buffer posseduti, pronta per upload_graph().
    // Valida finche' vive questo oggetto.
    GraphCsrView view() const;

    int32_t num_vertices() const { return num_vertices_; }
    int32_t num_chars() const { return static_cast<int32_t>(vertex_chars_.size()); }
    int32_t num_edges() const { return static_cast<int32_t>(edge_targets_.size()); }

    const std::vector<char> &vertex_chars() const { return vertex_chars_; }
    const std::vector<int32_t> &vertex_offsets() const { return vertex_offsets_; }
    const std::vector<int32_t> &edge_targets() const { return edge_targets_; }
    const std::vector<int32_t> &edge_overlaps() const { return edge_overlaps_; }
    const std::vector<int32_t> &edge_offsets() const { return edge_offsets_; }

private:
    std::vector<char> vertex_chars_;
    std::vector<int32_t> vertex_offsets_;
    std::vector<int32_t> edge_targets_;
    std::vector<int32_t> edge_overlaps_;
    std::vector<int32_t> edge_offsets_;
    int32_t num_vertices_ = 0;
};

}  // namespace gpu
}  // namespace theseus
