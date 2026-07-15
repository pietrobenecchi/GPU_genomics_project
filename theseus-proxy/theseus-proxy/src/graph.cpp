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


#include <vector>
#include <string>
#include "graph.h"
#include "gfa_graph.h"

namespace theseus {
    Graph::Graph(std::istream &gfa_stream) {
        GfaGraph gfa_graph(gfa_stream);

        // Add nodes
        _vertices.reserve(gfa_graph.gfa_nodes.size());
        for (int i = 0; i < gfa_graph.gfa_nodes.size(); ++i) {
            vertex v;
            v.name = gfa_graph.gfa_nodes[i].name;
            v.value = gfa_graph.gfa_nodes[i].seq;
            _vertices.push_back(v);
        }

        // Add edges
        for (int i = 0; i < gfa_graph.gfa_edges.size(); ++i) {
            edge e;
            e.from_vertex = gfa_graph.gfa_edges[i].from_node;
            e.to_vertex = gfa_graph.gfa_edges[i].to_node;
            e.overlap = gfa_graph.gfa_edges[i].overlap;
            _vertices[e.from_vertex].out_edges.push_back(e);
            _vertices[e.to_vertex].in_edges.push_back(e);
        }

        // Create name to id mapping
        for (int i = 0; i < _vertices.size(); ++i) {
            name_to_id_[_vertices[i].name] = i;
        }
    }
}