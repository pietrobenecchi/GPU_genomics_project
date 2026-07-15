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

#include <span>

#include "wavefront.h"
#include "cell.h"
#include "vector.h"

/* The scratchpad is the main merging data structure of Theseus. It serves as a
 common coordinate system containing all the possible existing diagonals in any
 dp matrix computed. The scratchpad allows to combine (in a process that we call
 sparsify) the data from the depending previous wavefronts yielding a score "s"
 in a given vertex, into a single wavefront containing their maximum offsets.
*/

namespace theseus {

class ScratchPad {
public:
    using size_type = ptrdiff_t;
    using diag_type = Cell::idx2d_t;

    /**
     * @brief Construct a new Scratch Pad object
     *
     * @param min_diag
     * @param max_diag
     */
    ScratchPad(diag_type min_diag, diag_type max_diag) :
        _wf(min_diag, max_diag, Cell{-1, -1, -1, -1, Cell::Matrix::None}) {

        _diags.realloc(_wf.size());
    }

    // TODO:
    Cell& access_alloc(diag_type diag) {
        // If the diagonal was not yet in the wavefront (offset is -1), add the
        // diagonal to _diags.
        auto size = _diags.size();

        // We are writing out of boundaries but inside capacity.
        // This is okay with a theseus::vector of trivial types (this is the case).
        _diags[size] = diag;

        size += _wf[diag].offset == -1;

        _diags.resize_unsafe(size);

        return _wf[diag];
    }

    // TODO:
    Cell& operator[](diag_type diag) { return _wf[diag]; }
    const Cell& operator[](diag_type diag) const { return _wf[diag]; }

    /**
     * @brief Return the number of active diagonals in the wavefront. That is,
     * the number of diagonals that have been modified since the last reset.
     *
     * @return size_type
     */
    size_type nactive_diags() {
        return _diags.size();
    }

    /**
     * @brief Return the minimum diagonal of the wavefront.
     *
     * @return diag_type
     */
    diag_type min_diag() const {
        return _wf.min_diag();
    }

    /**
     * @brief Return the maximum diagonal of the wavefront.
     *
     * @return diag_type
     */
    diag_type max_diag() const {
        return _wf.max_diag();
    }

    /**
     * @brief Return the active diagonals in the wavefront.
     *
     * @return std::span<const diag_type>
     */
    std::span<const diag_type> active_diags() {
        return std::span<diag_type>(_diags.data(), _diags.data() + _diags.size());
    }

    /**
     * @brief Reset the scratchpad. This means setting the offsets to a default
     * value (-1) and resizing the vector of diagonals to 0 (none have been changed
     * yet).
     *
     */
    void reset() {
        for (const auto diag : _diags) {
            _wf[diag].offset = -1;
        }
        _diags.resize(0);
    }

private:
    Wavefront<Cell> _wf;
    Vector<diag_type, true> _diags;
};

} // namespace theseus