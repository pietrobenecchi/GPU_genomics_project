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


#pragma once

#include <cstdint>
#include <type_traits>
#include <typeinfo>

#include "vector.h"

/**
 * Defines the Cell structure used in the Theseus alignment algorithm.
 *
 * Fields:
 * - prev_pos: Position of the cell where the optimal path to the current cell
 * comes from.
 * - vertex_id: ID of the vertex in the graph where this cell is located.
 * - offset: Offset in the query.
 * - diag: Diagonal in the dynamic programming matrix of the current vertex.
 * - from_matrix: Matrix from which the current cell was derived (M, I, D, MJumps,
 *  IJumps).
 *
 *
 */

namespace theseus {

constexpr ptrdiff_t realloc_wavefront_policy(std::ptrdiff_t capacity,
                                             std::ptrdiff_t required_size)
{
    return required_size * 1.5;
};

// WARNING: We want Cell to be a simple struct so it is standard layout and
// trivial. This way, resizes of Vector<Cell> are free.
struct Cell {
    using CellVector = Vector<Cell, true>;

    using vertex_t = int32_t; // TODO: This should be here?
    using idx2d_t = int32_t;
    using pos_t = int64_t;
    using score_t = int32_t;

    enum class edit_t : int8_t {
        None,
        M,
        Ins,
        Del};

    enum class Matrix : int8_t {
        None,
        M,
        MJumps,
        I,
        IJumps,
        D,
        I2,
        I2Jumps,
        D2
    };

    pos_t prev_pos;
    vertex_t vertex_id;
    idx2d_t offset;
    idx2d_t diag;
    Matrix from_matrix;
};

}   // namespace theseus
