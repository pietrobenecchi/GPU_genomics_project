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


#include "theseus/penalties.h"

// User defined alignment penalties
namespace theseus {

// Linear gap penalties constructor
Penalties::Penalties(penalty_t match, penalty_t mismatch, penalty_t gape)
    : type_(Type::Linear),
      match_(match),
      mismatch_(mismatch),
      gapo_(0),
      gape_(gape),
      gapo2_(0),
      gape2_(0) {}

// Affine gap penalties constructor
Penalties::Penalties(penalty_t match, penalty_t mismatch, penalty_t gapo, penalty_t gape)
    : type_(Type::Affine),
      match_(match),
      mismatch_(mismatch),
      gapo_(gapo),
      gape_(gape),
      gapo2_(0),
      gape2_(0) {}

// Dual affine gap penalties constructor
Penalties::Penalties(penalty_t match,
                     penalty_t mismatch,
                     penalty_t gapo,
                     penalty_t gape,
                     penalty_t gapo2,
                     penalty_t gape2)
    : type_(Type::DualAffine),
      match_(match),
      mismatch_(mismatch),
      gapo_(gapo),
      gape_(gape),
      gapo2_(gapo2),
      gape2_(gape2) {}

}   // namespace theseus