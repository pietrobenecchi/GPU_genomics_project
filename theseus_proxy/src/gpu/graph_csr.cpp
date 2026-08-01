#include "gpu/graph_csr.h"

namespace theseus {
namespace gpu {

GraphCsr::GraphCsr(const Graph &graph) {
    const std::vector<Graph::vertex> &vertices = graph._vertices;
    num_vertices_ = static_cast<int32_t>(vertices.size());

    vertex_offsets_.reserve(vertices.size() + 1);
    edge_offsets_.reserve(vertices.size() + 1);
    vertex_offsets_.push_back(0);
    edge_offsets_.push_back(0);

    for (const Graph::vertex &v : vertices) {
        vertex_chars_.insert(vertex_chars_.end(), v.value.begin(), v.value.end());
        vertex_offsets_.push_back(static_cast<int32_t>(vertex_chars_.size()));

        for (const Graph::edge &e : v.out_edges) {
            edge_targets_.push_back(static_cast<int32_t>(e.to_vertex));
            edge_overlaps_.push_back(static_cast<int32_t>(e.overlap));
        }
        edge_offsets_.push_back(static_cast<int32_t>(edge_targets_.size()));
    }
}

GraphCsrView GraphCsr::view() const {
    GraphCsrView view;
    view.vertex_chars = vertex_chars_.data();
    view.vertex_offsets = vertex_offsets_.data();
    view.edge_targets = edge_targets_.data();
    view.edge_overlaps = edge_overlaps_.data();
    view.edge_offsets = edge_offsets_.data();
    view.num_vertices = num_vertices_;
    return view;
}

}  // namespace gpu
}  // namespace theseus
