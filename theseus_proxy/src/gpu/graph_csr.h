#pragma once

#include <cstdint>
#include <vector>

#include "graph.h"
#include "gpu/align_gpu.h"

/**
 * @file graph_csr.h
 * @brief Host-side CSR mirror of a Graph.
 *
 * This is ordinary host C++: it owns std::vectors and is compiled by the host
 * compiler. Only the view() it hands out crosses into the CUDA backend.
 */

namespace theseus {
namespace gpu {

class GraphCsr {
public:
    /**
     * @brief Flatten @p graph into CSR buffers.
     *
     * Vertex names and in-edges are dropped: the alignment never reads them,
     * they only serve GAF and GFA output on the host.
     */
    explicit GraphCsr(const Graph &graph);

    /**
     * @brief View over the owned buffers, ready for upload_graph().
     *
     * Valid as long as this object lives.
     */
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
