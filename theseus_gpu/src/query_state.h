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

#include "cell.h"

#ifdef __CUDACC__
#define THESEUS_HD __host__ __device__
#else
#define THESEUS_HD
#endif

/*
 * QueryState is the whole per-query working set of the aligner, laid out as flat
 * POD arrays with fixed capacities so that device code (which cannot allocate or
 * grow a buffer once a kernel is running) can own one instance per query.
 *
 * It is being filled in one structure at a time, following the CPU-first method:
 * each dynamic structure of the aligner is replaced by a fixed-capacity flat
 * mirror here, the algorithm is repointed at it, and the output is checked
 * byte-for-byte against the baseline before moving on. The first structure moved
 * is the ScratchPad.
 *
 * Every capacity here is PROVISIONAL, tuned to the toy sample dataset, exactly
 * like kProvisionalWavefrontCapacity. When a real dataset arrives the peaks have
 * to be measured and the bounds re-derived. Overflow is never silent: it sets
 * QueryState::capacity_exceeded so the day a buffer is too small is not the day a
 * kernel reads past its end.
 */

namespace theseus {

/**
 * @brief Diagonals the flattened ScratchPad wavefront is built to hold.
 *
 * Derived, not guessed. `extend_diagonal` computes j = diag + offset, where j
 * indexes a vertex sequence and offset indexes the query, so
 *
 *     diag in [-max_query_len, max_vertex_len]
 *     span  = max_vertex_len + max_query_len + 1
 *
 * `new_alignment` already computes exactly that window and calls sp_init with
 * it; this constant is the compile-time ceiling that window may reach. It must
 * cover the largest graph in the validation set, which is what the numbers below
 * are measured from (100 bp GGBS reads):
 *
 *     graph   segments   longest vertex   span needed
 *     ebola          7            9 063         9 164
 *     c4            16           52 006        52 107
 *
 * Growing this grows QueryState by 28 bytes per diagonal (a Cell plus its entry
 * in the active-diagonal list), and one QueryState is allocated per query, so it
 * trades directly against batch size in device memory: at this value QueryState
 * is 4.2 MB and a 512-query batch needs 2.1 GB. A graph with longer vertices
 * (MHC, yeast) will need a larger value, or a sparse ScratchPad instead of this
 * dense one. Overflow is never silent: align_batch_gpu reports the required
 * span before launching, and sp_init records it through cap_fail.
 */
constexpr int kScratchpadSpan = 52224;  // covers c4's 52 006 bp vertex + 100 bp reads

/**
 * @brief Cells each accumulating backtrace wavefront (BeyondScope) is built to
 * hold.
 *
 * These wavefronts grow for the whole alignment and are re-read by the
 * backtrace, so this is the dominant per-query buffer. Matches the CPU
 * BeyondScope's initial 4096. PROVISIONAL, like every capacity here.
 */
constexpr int kBeyondScopeCapacity = 4096;

/**
 * @brief Ring length (number of scores) the flattened Scope is built to hold.
 *
 * The CPU Scope sizes its ring at runtime from the penalties
 * (max(gapo+gape, mism)+1, i.e. 5 with the default penalties). Device arrays
 * need a compile-time bound; this is it. PROVISIONAL: guarded, and overflow sets
 * capacity_exceeded rather than indexing out of range.
 */
constexpr int kMaxScores = 8;

/**
 * @brief Cells each per-score Scope wavefront (I and D) is built to hold, and
 * range entries each per-score position vector (M/I/D) is built to hold.
 *
 * The Scope wavefronts are the ones kProvisionalWavefrontCapacity was written
 * about; the position vectors hold one range per active vertex. Both PROVISIONAL.
 * Kept equal to kProvisionalWavefrontCapacity (defined in theseus_aligner_impl.h,
 * which cannot be included here without a cycle).
 */
constexpr int kScopeWavefrontCapacity = 1024;
constexpr int kScopePosCapacity = 1024;

/**
 * @brief A half-open range into a wavefront, one per active vertex. Mirrors the
 * POD Scope::range so the flattened Scope needs no dependency on the Scope class.
 */
struct Range {
    int64_t start;
    int64_t end;
};

/**
 * @brief Fixed bounds for the flattened VerticesData. All PROVISIONAL and
 * guarded: kMaxVertices bounds the graph size (the vertex->active-index map),
 * kMaxActiveVertices the vertices touched in one alignment, kMaxInvalidSegments
 * the invalid-diagonal segments held per matrix per vertex, kMaxJumpsPerScore
 * the jump positions recorded per score per vertex.
 */
constexpr int kMaxVertices = 2048;
constexpr int kMaxActiveVertices = 256;
constexpr int kMaxInvalidSegments = 64;
constexpr int kMaxJumpsPerScore = 32;
// Derived, not guessed. The stack only grows on a chain of zero-length
// vertices, and the measured peak across all four GGBS datasets plus the sample
// graph is 1 (store_I_jump is reached only by c4_err; none of the graphs holds a
// zero-length segment). 8 leaves room for a chain of seven. It is a device-side
// local array of 56-byte frames, so 256 cost 14 336 bytes of local memory per
// thread and dominated the kernel's register/spill budget. Overflow is not
// silent: cap_fail records kCapIJumpStack with the depth that was needed.
constexpr int kMaxIJumpStack = 8;

/**
 * @brief One run of invalid diagonals for a vertex/matrix, with the countdowns
 * that grow it. Mirrors VerticesData::InvalidData (Segment + rem_up/rem_down)
 * flattened into one POD.
 */
struct InvalidSeg {
    int32_t start_d;   // inclusive
    int32_t end_d;     // inclusive
    int32_t rem_up;    // scores left before the segment grows one diagonal up
    int32_t rem_down;  // scores left before it grows one diagonal down
};

struct QueryState {
    // ---- ScratchPad -------------------------------------------------------
    // A wavefront indexed by diagonal over [sp_min_diag, sp_max_diag], plus the
    // list of diagonals touched since the last reset (the sparse view the
    // densify step iterates). sp_wf[diag - sp_min_diag] is the cell of `diag`.
    // A cell is "inactive" when its offset is -1, which is both the resting
    // value and how access_alloc detects a first touch.
    //
    // The offset is split out of the cell into sp_off, its own dense array, and
    // sp_off is the *authority*: every activity test and every offset
    // comparison reads it, and only it is cleared when the window opens. That
    // makes the per-query clear one word per diagonal instead of the six a Cell
    // occupies, which matters because the clear runs over the whole window
    // (52 107 diagonals on c4) while the alignment touches a handful of them.
    //
    // sp_wf is the cold half: the payload of the cells that are actually
    // active, never cleared, read only through sp_diags. Its own `offset` field
    // is written along with the rest of the cell and read back as part of the
    // payload, so an active cell agrees with sp_off; an inactive one holds
    // whatever the last active cell on that diagonal left, exactly as it
    // already did between scores, since sp_reset has always cleared the offset
    // alone. See profiling/campagna_cuda/01_invariante_scratchpad.md.
    Cell    sp_wf[kScratchpadSpan];
    int32_t sp_off[kScratchpadSpan];
    Cell    sp_overflow_cell;
    int32_t sp_diags[kScratchpadSpan];
    // How much of sp_off has ever been set to -1, in entries from 0. The
    // ScratchPad returns to its resting state by itself -- every write is to a
    // diagonal that is in sp_diags, and process_vertex ends with an sp_reset
    // that walks sp_diags -- so a prefix cleared once stays clear for every
    // later query on this state, and only the part never yet cleared has to be.
    // 0 means "nothing cleared", which is what a fresh QueryState must read as.
    // See profiling/campagna_cuda/04_clear_pigro.md.
    int32_t sp_cleared;
    int32_t sp_min_diag;
    int32_t sp_max_diag;
    int32_t sp_ndiags;

    // ---- BeyondScope ------------------------------------------------------
    // The append-only wavefronts kept until backtrace: the M densify output and
    // the M/I jump cells. Each is a flat array plus a running size. Being fixed
    // arrays, references into them never move (the CPU CellVector could realloc
    // mid-extend), so this is byte-identical and strictly safer as long as the
    // capacity holds. The unused I2-jumps wavefront is omitted.
    Cell    bs_m_wf[kBeyondScopeCapacity];
    Cell    bs_m_jumps_wf[kBeyondScopeCapacity];
    Cell    bs_i_jumps_wf[kBeyondScopeCapacity];
    int32_t bs_m_wf_size;
    int32_t bs_m_jumps_wf_size;
    int32_t bs_i_jumps_wf_size;

    // ---- Scope (ring of nscores wavefronts) -------------------------------
    // Indexed by score % sc_nscores. Each slot holds the per-score I and D dense
    // wavefronts (the M dense wavefront lives in BeyondScope) and the M/I/D
    // position ranges, one range per active vertex. sc_peak_wf is the largest
    // per-score wavefront seen, kept to size a real bound from.
    Cell    sc_i_wf[kMaxScores][kScopeWavefrontCapacity];
    Cell    sc_d_wf[kMaxScores][kScopeWavefrontCapacity];
    int32_t sc_i_wf_size[kMaxScores];
    int32_t sc_d_wf_size[kMaxScores];
    Range   sc_m_pos[kMaxScores][kScopePosCapacity];
    Range   sc_i_pos[kMaxScores][kScopePosCapacity];
    Range   sc_d_pos[kMaxScores][kScopePosCapacity];
    int32_t sc_m_pos_size[kMaxScores];
    int32_t sc_i_pos_size[kMaxScores];
    int32_t sc_d_pos_size[kMaxScores];
    int32_t sc_nscores;
    int32_t sc_peak_wf;

    // ---- VerticesData -----------------------------------------------------
    // Active vertices are packed into [0, vd_num_active); vd_vertex_to_idx maps a
    // graph vertex id to its active index (-1 if not active). Per active vertex:
    // the invalid-diagonal segments of matrices M/I/D, and the jump positions
    // recorded per score for the M and I jump wavefronts. vd_gapo/vd_gape are the
    // public penalties VerticesData grows its segments with.
    int32_t vd_gapo;
    int32_t vd_gape;
    int32_t vd_nscores;
    int32_t vd_num_vertices;
    int32_t vd_num_active;
    int32_t vd_vertex_to_idx[kMaxVertices];
    int32_t vd_vertex_id[kMaxActiveVertices];

    InvalidSeg vd_m_invalid[kMaxActiveVertices][kMaxInvalidSegments];
    InvalidSeg vd_i_invalid[kMaxActiveVertices][kMaxInvalidSegments];
    InvalidSeg vd_d_invalid[kMaxActiveVertices][kMaxInvalidSegments];
    int32_t vd_m_invalid_size[kMaxActiveVertices];
    int32_t vd_i_invalid_size[kMaxActiveVertices];
    int32_t vd_d_invalid_size[kMaxActiveVertices];

    int64_t vd_m_jumps_pos[kMaxActiveVertices][kMaxScores][kMaxJumpsPerScore];
    int64_t vd_i_jumps_pos[kMaxActiveVertices][kMaxScores][kMaxJumpsPerScore];
    int32_t vd_m_jumps_pos_size[kMaxActiveVertices][kMaxScores];
    int32_t vd_i_jumps_pos_size[kMaxActiveVertices][kMaxScores];

    // Set true the first time any fixed capacity above is too small. Checked by
    // the host after an alignment; never silently ignored.
    bool capacity_exceeded;

    // Which buffer ran out first, and by how much. A bare boolean told you an
    // overflow happened somewhere among fourteen call sites, which is not enough
    // to size anything: these say which one and what it wanted. Only the FIRST
    // overflow is recorded, because later ones are usually consequences of it.
    int8_t  cap_reason;    // a CapBuffer value
    int32_t cap_required;  // entries the caller asked for
    int32_t cap_available; // entries the buffer has
};

/**
 * @brief Everything the host needs to reconstruct one device alignment, and
 * nothing else.
 *
 * QueryState is the kernel's working set, not the shape of a result. The
 * backtrace only ever reads the three BeyondScope wavefronts (through
 * Cell::from_matrix and Cell::prev_pos) plus the capacity diagnostics, so the
 * ScratchPad, the Scope ring and VerticesData never have to cross the bus or be
 * materialised host-side. That is the whole difference between a 4.2 MB
 * per-query result and a 288 KB one: a 512-query batch needs 147 MB here
 * instead of 2.1 GB.
 *
 * The default constructor is user-provided and empty on purpose. It makes the
 * type non-trivially-default-constructible so `std::vector<CompactTracebackState>(n)`
 * value-initialises by calling it — which does nothing — instead of zero-filling
 * a buffer that align_batch is about to overwrite in full.
 */
struct CompactTracebackState {
    CompactTracebackState() noexcept {}

    // Views into the packed buffer align_batch fills, not storage.
    //
    // They used to be three Cell[kBeyondScopeCapacity] arrays, 288 KB per query
    // whatever the query needed, copied by one strided cudaMemcpy2D per field.
    // Instrumenting the CPU aligner -- same QueryState, same wavefronts, output
    // byte-identical -- says what those 288 KB carry: on c4_err_2k the M
    // wavefront averages 24,3 cells of 4096 and peaks at 438, the M jumps
    // average 1,0, the I jumps 0,0. **0,21 % of the bytes moved are ever read**,
    // and on c4_exact_2k 0,01 %.
    //
    // So align_batch packs the used prefixes on the device and copies them once,
    // and these point into that buffer. Two consequences for the caller:
    //
    // - the cells live in the DeviceWorkspace's staging buffer, valid until the
    //   next align_batch on that workspace. The backtrace runs right after the
    //   call, which is the only use there has ever been, but it is a borrow now
    //   and not a copy;
    // - a size of 0 leaves its pointer pointing at the start of the query's
    //   slice, which is one past the previous query's cells. Nothing reads it,
    //   and the pointer is never null.
    const Cell *bs_m_wf;
    const Cell *bs_m_jumps_wf;
    const Cell *bs_i_jumps_wf;
    int32_t bs_m_wf_size;
    int32_t bs_m_jumps_wf_size;
    int32_t bs_i_jumps_wf_size;

    // Diagnostics, carried so a device-side overflow is still reported by the
    // host that never sees the buffer that overflowed.
    int32_t sc_peak_wf;
    bool    capacity_exceeded;
    int8_t  cap_reason;    // a CapBuffer value
    int32_t cap_required;
    int32_t cap_available;
};

/**
 * @brief Read-only view of the three wavefronts the backtrace walks.
 *
 * The backtrace is one algorithm over cells, indifferent to whether the cells
 * were produced by the CPU aligner into its QueryState or by the kernel into a
 * CompactTracebackState. Binding the two sources here keeps a single traceback
 * implementation instead of a per-step test of which one it is reading.
 */
struct TracebackWavefronts {
    const Cell *m;
    const Cell *m_jumps;
    const Cell *i_jumps;
};

THESEUS_HD inline TracebackWavefronts traceback_view(const QueryState &qs) {
    return TracebackWavefronts{qs.bs_m_wf, qs.bs_m_jumps_wf, qs.bs_i_jumps_wf};
}

THESEUS_HD inline TracebackWavefronts traceback_view(const CompactTracebackState &state) {
    return TracebackWavefronts{state.bs_m_wf, state.bs_m_jumps_wf, state.bs_i_jumps_wf};
}

/**
 * @brief Which fixed-capacity buffer overflowed. Values are stable: they are
 * copied back from the device and printed by the host.
 */
enum CapBuffer : int8_t {
    kCapNone = 0,
    kCapScratchpadSpan,   // ScratchPad diagonal window
    kCapScratchpadDiags,  // ScratchPad active-diagonal list
    kCapBeyondScope,      // accumulating backtrace wavefronts
    kCapScores,           // Scope ring length
    kCapScopeWavefront,   // per-score I/D wavefront cells
    kCapScopePos,         // per-score M/I/D position ranges
    kCapVertices,         // vertex -> active-index map domain
    kCapActiveVertices,   // vertices touched in one alignment
    kCapInvalidSegments,  // invalid-diagonal segments per matrix per vertex
    kCapJumpsPerScore,    // jump positions per vertex per score
    kCapIJumpStack,       // explicit stack depth in store_I_jump
};

/**
 * @brief Record an overflow: set the flag, and keep the first cause.
 */
THESEUS_HD inline void cap_fail(QueryState &qs, CapBuffer which, int32_t required,
                                int32_t available) {
    qs.capacity_exceeded = true;
    if (qs.cap_reason == kCapNone) {
        qs.cap_reason = which;
        qs.cap_required = required;
        qs.cap_available = available;
    }
}

/**
 * @brief Name of the buffer @p reason refers to, for host-side messages.
 */
inline const char *cap_buffer_name(int8_t reason) {
    switch (reason) {
        case kCapScratchpadSpan:  return "scratchpad diagonal span";
        case kCapScratchpadDiags: return "scratchpad active-diagonal list";
        case kCapBeyondScope:     return "beyond-scope wavefront";
        case kCapScores:          return "scope ring length";
        case kCapScopeWavefront:  return "scope wavefront cells";
        case kCapScopePos:        return "scope position ranges";
        case kCapVertices:        return "vertex index map";
        case kCapActiveVertices:  return "active vertices";
        case kCapInvalidSegments: return "invalid segments";
        case kCapJumpsPerScore:   return "jumps per score";
        case kCapIJumpStack:      return "I-jump stack";
        default:                  return "none";
    }
}

/**
 * @brief Put the ScratchPad window at [min_diag, max_diag] and clear it.
 *
 * Mirrors constructing a fresh CPU ScratchPad(min_diag, max_diag): every cell in
 * the window is set inactive (offset -1) and no diagonal is active. Called once
 * at construction and again only when the window has to grow, matching when the
 * CPU code reallocates.
 */
/**
 * @brief The scalar half of sp_init: set the window, empty the active list and
 * check the span. Returns the number of cells the caller still has to clear, or
 * 0 when the span does not fit (the cap failure is already recorded).
 *
 * Split out so the GPU can run the clearing loop across the block: the stores
 * are independent, and 52 224 of them on one thread was the largest serial
 * stretch in the kernel. sp_init below is this plus the serial loop, which is
 * what the CPU flattening still calls.
 */
THESEUS_HD inline int sp_init_window(QueryState &qs, int min_diag, int max_diag) {
    qs.sp_min_diag = min_diag;
    qs.sp_max_diag = max_diag;
    qs.sp_ndiags = 0;
    qs.sp_overflow_cell = Cell{-1, -1, -1, -1, Cell::Matrix::None};

    const int span = max_diag - min_diag + 1;
    if (span > kScratchpadSpan) {
        cap_fail(qs, kCapScratchpadSpan, span, kScratchpadSpan);
        return 0;
    }
    return span;
}

/**
 * @brief Entries of sp_off this query still has to clear, as [start, span).
 *
 * Only the offsets: sp_wf is payload and nothing reads it before a cell has
 * been made active, which writes it whole. And only the entries never cleared
 * before, because the ScratchPad puts itself back: every write goes to a
 * diagonal that sp_access appended to sp_diags, and process_vertex ends with an
 * sp_reset that walks sp_diags and sets each of them back to -1. So at the end
 * of a query every entry it touched is -1 again, and the prefix cleared once is
 * still clear. What is left is the entries a *longer* window reaches for the
 * first time.
 *
 * The exception is a capacity failure: past kScratchpadSpan diagonals the
 * append is dropped while the cell is still written, so a cell can stay dirty
 * with nothing recording it. align_one answers that by putting sp_cleared back
 * to 0, which makes the next query on this state clear everything again.
 */
THESEUS_HD inline int sp_clear_start(QueryState &qs, int span) {
    const int start = qs.sp_cleared < span ? qs.sp_cleared : span;
    if (span > qs.sp_cleared) {
        qs.sp_cleared = span;
    }
    return start;
}

THESEUS_HD inline void sp_init(QueryState &qs, int min_diag, int max_diag) {
    const int span = sp_init_window(qs, min_diag, max_diag);
    for (int i = sp_clear_start(qs, span); i < span; ++i) {
        qs.sp_off[i] = -1;
    }
}

/**
 * @brief The scratchpad cell of diagonal @p diag, without marking it active.
 */
THESEUS_HD inline Cell &sp_at(QueryState &qs, int diag) {
    const int idx = diag - qs.sp_min_diag;
    if (idx < 0 || idx >= kScratchpadSpan) {
        cap_fail(qs, kCapScratchpadSpan, idx < 0 ? -idx : idx + 1, kScratchpadSpan);
        qs.sp_overflow_cell = Cell{-1, -1, -1, -1, Cell::Matrix::None};
        return qs.sp_overflow_cell;
    }
    return qs.sp_wf[idx];
}

/**
 * @brief Merge one candidate into the scratchpad, keeping the larger offset.
 *
 * This is ScratchPad::access_alloc followed by the compare-and-store that every
 * caller performed on the reference it returned:
 *
 *     Cell &cell = sp_access_alloc(qs, c.diag);
 *     if (cell.offset < c.offset) cell = c;
 *
 * The two halves are one primitive now because the offset lives in sp_off and
 * the payload in sp_wf, so a caller holding a `Cell &` could no longer keep the
 * two in step on its own. The semantics are unchanged, field for field:
 *
 * - the diagonal is appended to sp_diags only when the cell was still inactive,
 *   which dedups by first touch and preserves first-touch order (densify walks
 *   sp_diags in that order, so the order decides the dense wavefront);
 * - the comparison stays strict, so an equal offset never displaces the cell
 *   already there;
 * - out of window is a capacity failure and the candidate neither appends nor
 *   writes, as before.
 */
THESEUS_HD inline void sp_merge_candidate(QueryState &qs, const Cell &c) {
    const int idx = c.diag - qs.sp_min_diag;
    if (idx < 0 || idx >= kScratchpadSpan) {
        cap_fail(qs, kCapScratchpadSpan, idx < 0 ? -idx : idx + 1, kScratchpadSpan);
        return;
    }
    if (qs.sp_off[idx] == -1) {
        if (qs.sp_ndiags >= kScratchpadSpan) {
            cap_fail(qs, kCapScratchpadDiags, qs.sp_ndiags + 1, kScratchpadSpan);
            return;
        }
        qs.sp_diags[qs.sp_ndiags] = c.diag;
        ++qs.sp_ndiags;
    }
    if (qs.sp_off[idx] < c.offset) {
        qs.sp_wf[idx] = c;
        qs.sp_off[idx] = c.offset;
    }
}

/**
 * @brief Reset the scratchpad: set every active diagonal back to inactive and
 * empty the active list. Only touched cells are cleared, matching the CPU code.
 */
THESEUS_HD inline void sp_reset_one(QueryState &qs, int i) {
    qs.sp_off[qs.sp_diags[i] - qs.sp_min_diag] = -1;
}

THESEUS_HD inline void sp_reset(QueryState &qs) {
    for (int i = 0; i < qs.sp_ndiags; ++i) {
        sp_reset_one(qs, i);
    }
    qs.sp_ndiags = 0;
}

/**
 * @brief Empty every BeyondScope wavefront for a new alignment.
 */
THESEUS_HD inline void bs_new_alignment(QueryState &qs) {
    qs.bs_m_wf_size = 0;
    qs.bs_m_jumps_wf_size = 0;
    qs.bs_i_jumps_wf_size = 0;
}

/**
 * @brief Append @p c to a BeyondScope wavefront and return the index it landed
 * at (mirrors capturing size() before a push_back).
 *
 * On overflow the cell is dropped and capacity_exceeded is set: the alignment is
 * already invalid, but nothing is written past the end of the array. The clamped
 * index keeps callers in bounds until the host sees the flag.
 */
THESEUS_HD inline int32_t bs_push_back(QueryState &qs, Cell *wf, int32_t &size,
                            const Cell &c) {
    if (size >= kBeyondScopeCapacity) {
        cap_fail(qs, kCapBeyondScope, size + 1, kBeyondScopeCapacity);
        return (size > 0) ? size - 1 : 0;
    }
    const int32_t pos = size;
    wf[pos] = c;
    size = pos + 1;
    return pos;
}

// ---- Scope ----------------------------------------------------------------

/**
 * @brief Set the Scope ring length and clear it. Mirrors constructing a Scope
 * with @p nscores slots. Over-long rings are clamped and flagged rather than
 * indexed out of range.
 */
THESEUS_HD inline void sc_init(QueryState &qs, int nscores) {
    if (nscores > kMaxScores) {
        cap_fail(qs, kCapScores, nscores, kMaxScores);
        nscores = kMaxScores;
    }
    qs.sc_nscores = nscores;
    qs.sc_peak_wf = 0;
    for (int s = 0; s < kMaxScores; ++s) {
        qs.sc_i_wf_size[s] = 0;
        qs.sc_d_wf_size[s] = 0;
        qs.sc_m_pos_size[s] = 0;
        qs.sc_i_pos_size[s] = 0;
        qs.sc_d_pos_size[s] = 0;
    }
}

/**
 * @brief Empty every ring slot for a new alignment.
 */
THESEUS_HD inline void sc_new_alignment(QueryState &qs) {
    for (int s = 0; s < qs.sc_nscores; ++s) {
        qs.sc_i_wf_size[s] = 0;
        qs.sc_d_wf_size[s] = 0;
        qs.sc_m_pos_size[s] = 0;
        qs.sc_i_pos_size[s] = 0;
        qs.sc_d_pos_size[s] = 0;
    }
}

/**
 * @brief Reset the ring slot of @p score, whose old contents are no longer in
 * scope. Mirrors Scope::new_score.
 */
THESEUS_HD inline void sc_new_score(QueryState &qs, int score) {
    const int s = score % qs.sc_nscores;
    qs.sc_i_wf_size[s] = 0;
    qs.sc_d_wf_size[s] = 0;
    qs.sc_m_pos_size[s] = 0;
    qs.sc_i_pos_size[s] = 0;
    qs.sc_d_pos_size[s] = 0;
}

// Ring-slot views, named like the Scope accessors so call sites read the same.
THESEUS_HD inline Cell *sc_i_wf(QueryState &qs, int score) { return qs.sc_i_wf[score % qs.sc_nscores]; }
THESEUS_HD inline Cell *sc_d_wf(QueryState &qs, int score) { return qs.sc_d_wf[score % qs.sc_nscores]; }
THESEUS_HD inline int32_t &sc_i_wf_size(QueryState &qs, int score) { return qs.sc_i_wf_size[score % qs.sc_nscores]; }
THESEUS_HD inline int32_t &sc_d_wf_size(QueryState &qs, int score) { return qs.sc_d_wf_size[score % qs.sc_nscores]; }
THESEUS_HD inline Range *sc_m_pos(QueryState &qs, int score) { return qs.sc_m_pos[score % qs.sc_nscores]; }
THESEUS_HD inline Range *sc_i_pos(QueryState &qs, int score) { return qs.sc_i_pos[score % qs.sc_nscores]; }
THESEUS_HD inline Range *sc_d_pos(QueryState &qs, int score) { return qs.sc_d_pos[score % qs.sc_nscores]; }
THESEUS_HD inline int32_t &sc_m_pos_size(QueryState &qs, int score) { return qs.sc_m_pos_size[score % qs.sc_nscores]; }
THESEUS_HD inline int32_t &sc_i_pos_size(QueryState &qs, int score) { return qs.sc_i_pos_size[score % qs.sc_nscores]; }
THESEUS_HD inline int32_t &sc_d_pos_size(QueryState &qs, int score) { return qs.sc_d_pos_size[score % qs.sc_nscores]; }

/**
 * @brief Append a cell to a Scope wavefront, tracking the peak size. On overflow
 * the cell is dropped and capacity_exceeded is set (nothing written past the
 * end); the alignment is already invalid by then.
 */
THESEUS_HD inline void sc_wf_push(QueryState &qs, Cell *wf, int32_t &size, const Cell &c) {
    if (size >= kScopeWavefrontCapacity) {
        cap_fail(qs, kCapScopeWavefront, size + 1, kScopeWavefrontCapacity);
        return;
    }
    wf[size] = c;
    ++size;
    if (size > qs.sc_peak_wf) {
        qs.sc_peak_wf = size;
    }
}

/**
 * @brief Append a range to a Scope position vector, guarding capacity.
 */
THESEUS_HD inline void sc_pos_push(QueryState &qs, Range *pos, int32_t &size, const Range &r) {
    if (size >= kScopePosCapacity) {
        cap_fail(qs, kCapScopePos, size + 1, kScopePosCapacity);
        return;
    }
    pos[size] = r;
    ++size;
}

// ---- VerticesData ---------------------------------------------------------

THESEUS_HD inline int32_t vd_min(int32_t a, int32_t b) { return a < b ? a : b; }
THESEUS_HD inline int32_t vd_max(int32_t a, int32_t b) { return a > b ? a : b; }

/**
 * @brief How many entries of vd_vertex_to_idx a graph of @p num_vertices owns.
 *
 * The one place the clamp to kMaxVertices lives, so the scalar halves below and
 * the block-parallel fill in the kernel cannot disagree on the bound.
 */
THESEUS_HD inline int vd_map_fill_count(int num_vertices) {
    return num_vertices > kMaxVertices ? kMaxVertices : num_vertices;
}

/**
 * @brief The scalar half of vd_init: penalties, ring length, graph size and the
 * capacity check. Returns how many entries of vd_vertex_to_idx the caller still
 * has to set to -1.
 *
 * Split out for the same reason as sp_init_window: the clearing loop is
 * independent per entry, so the GPU runs it across the block.
 */
THESEUS_HD inline int vd_init_scalar(QueryState &qs, int gapo, int gape, int nscores,
                                     int num_vertices) {
    qs.vd_gapo = gapo;
    qs.vd_gape = gape;
    qs.vd_nscores = nscores;
    qs.vd_num_vertices = num_vertices;
    qs.vd_num_active = 0;
    if (num_vertices > kMaxVertices) {
        cap_fail(qs, kCapVertices, num_vertices, kMaxVertices);
    }
    return vd_map_fill_count(num_vertices);
}

/**
 * @brief Set the penalties, ring length and graph size the vertices data works
 * with, and empty it. @p num_vertices is the domain of the vertex->index map.
 */
THESEUS_HD inline void vd_init(QueryState &qs, int gapo, int gape, int nscores,
                    int num_vertices) {
    const int n = vd_init_scalar(qs, gapo, gape, nscores, num_vertices);
    for (int i = 0; i < n; ++i) {
        qs.vd_vertex_to_idx[i] = -1;
    }
}

/** @brief The scalar half of vd_new_alignment. See vd_init_scalar. */
THESEUS_HD inline int vd_new_alignment_scalar(QueryState &qs) {
    qs.vd_num_active = 0;
    return vd_map_fill_count(qs.vd_num_vertices);
}

/**
 * @brief Drop all active vertices and mark every graph vertex inactive.
 */
THESEUS_HD inline void vd_new_alignment(QueryState &qs) {
    const int n = vd_new_alignment_scalar(qs);
    for (int i = 0; i < n; ++i) {
        qs.vd_vertex_to_idx[i] = -1;
    }
}

THESEUS_HD inline int vd_get_pos(const QueryState &qs, int score) { return score % qs.vd_nscores; }
THESEUS_HD inline int vd_get_id(const QueryState &qs, int vtx) { return qs.vd_vertex_to_idx[vtx]; }
THESEUS_HD inline int vd_get_vertex_id(const QueryState &qs, int idx) { return qs.vd_vertex_id[idx]; }
THESEUS_HD inline int vd_num_active_vertices(const QueryState &qs) { return qs.vd_num_active; }

/**
 * @brief Add @p vtx to the active set if it is not already there. A fresh active
 * vertex has empty invalid-segment and jump lists.
 */
THESEUS_HD inline void vd_activate_vertex(QueryState &qs, int vtx) {
    if (vtx >= kMaxVertices) {
        cap_fail(qs, kCapVertices, vtx + 1, kMaxVertices);
        return;
    }
    if (qs.vd_vertex_to_idx[vtx] != -1) {
        return;
    }
    const int idx = qs.vd_num_active;
    if (idx >= kMaxActiveVertices) {
        cap_fail(qs, kCapActiveVertices, idx + 1, kMaxActiveVertices);
        return;
    }
    qs.vd_vertex_id[idx] = vtx;
    qs.vd_m_invalid_size[idx] = 0;
    qs.vd_i_invalid_size[idx] = 0;
    qs.vd_d_invalid_size[idx] = 0;
    for (int s = 0; s < qs.vd_nscores; ++s) {
        qs.vd_m_jumps_pos_size[idx][s] = 0;
        qs.vd_i_jumps_pos_size[idx][s] = 0;
    }
    qs.vd_vertex_to_idx[vtx] = idx;
    qs.vd_num_active = idx + 1;
}

/**
 * @brief Clear the jump positions of every active vertex at @p score's ring slot.
 */
THESEUS_HD inline int vd_new_score_slot(const QueryState &qs, int score) {
    return score % qs.vd_nscores;
}

/** @brief Clear one active vertex's jump positions at ring slot @p pos. */
THESEUS_HD inline void vd_new_score_one(QueryState &qs, int a, int pos) {
    qs.vd_m_jumps_pos_size[a][pos] = 0;
    qs.vd_i_jumps_pos_size[a][pos] = 0;
}

THESEUS_HD inline void vd_new_score(QueryState &qs, int score) {
    const int pos = vd_new_score_slot(qs, score);
    for (int a = 0; a < qs.vd_num_active; ++a) {
        vd_new_score_one(qs, a, pos);
    }
}

/**
 * @brief Age every segment of one list by one score, growing it when a countdown
 * reaches zero. Mirrors VerticesData::expand_invalid_vector.
 */
THESEUS_HD inline void vd_expand_vec(InvalidSeg *v, int size, int def_up, int def_down) {
    for (int l = 0; l < size; ++l) {
        v[l].rem_down -= 1;
        v[l].rem_up -= 1;
        if (v[l].rem_up == 0) {
            v[l].rem_up = def_up;
            v[l].end_d += 1;
        }
        if (v[l].rem_down == 0) {
            v[l].rem_down = def_down;
            v[l].start_d -= 1;
        }
    }
}

THESEUS_HD inline void vd_expand(QueryState &qs) {
    const int g = qs.vd_gape;
    for (int a = 0; a < qs.vd_num_active; ++a) {
        vd_expand_vec(qs.vd_m_invalid[a], qs.vd_m_invalid_size[a], g, g);
        vd_expand_vec(qs.vd_i_invalid[a], qs.vd_i_invalid_size[a], g, g);
        vd_expand_vec(qs.vd_d_invalid[a], qs.vd_d_invalid_size[a], g, g);
    }
}

/**
 * @brief Sort one segment list by start diagonal and merge overlapping runs.
 * Mirrors VerticesData::compact_invalid_vector; the std::sort is replaced by an
 * insertion sort (segments per vertex are few) and the merge arithmetic, order
 * included, is kept verbatim. Returns the new size.
 */
THESEUS_HD inline int vd_compact_vec(InvalidSeg *v, int size, int def_up, int def_down) {
    if (size == 0) {
        return 0;
    }

    // Insertion sort by start_d (ascending), stable.
    for (int i = 1; i < size; ++i) {
        InvalidSeg key = v[i];
        int j = i - 1;
        while (j >= 0 && v[j].start_d > key.start_d) {
            v[j + 1] = v[j];
            --j;
        }
        v[j + 1] = key;
    }

    int k = 0;
    for (int l = 1; l < size; ++l) {
        if (v[k].end_d + 1 >= v[l].start_d) {
            v[k].end_d = vd_max(v[l].end_d, v[k].end_d);

            v[k].rem_down = vd_min(v[k].rem_down,
                                   v[l].rem_down +
                                       (v[l].start_d - v[k].start_d) * def_down);

            if (v[l].end_d > v[k].end_d) {
                v[k].rem_up = vd_min(v[l].rem_up,
                                     v[k].rem_up +
                                         (v[l].end_d - v[k].end_d) * def_up);
            } else {
                v[k].rem_up = vd_min(v[k].rem_up,
                                     v[l].rem_up +
                                         (v[k].end_d - v[l].end_d) * def_up);
            }
        } else {
            k += 1;
            v[k] = v[l];
        }
    }
    return k + 1;
}

THESEUS_HD inline void vd_compact(QueryState &qs) {
    const int g = qs.vd_gape;
    for (int a = 0; a < qs.vd_num_active; ++a) {
        qs.vd_m_invalid_size[a] = vd_compact_vec(qs.vd_m_invalid[a], qs.vd_m_invalid_size[a], g, g);
        qs.vd_i_invalid_size[a] = vd_compact_vec(qs.vd_i_invalid[a], qs.vd_i_invalid_size[a], g, g);
        qs.vd_d_invalid_size[a] = vd_compact_vec(qs.vd_d_invalid[a], qs.vd_d_invalid_size[a], g, g);
    }
}

THESEUS_HD inline void vd_invalid_push(QueryState &qs, InvalidSeg *v, int32_t &size,
                            const InvalidSeg &s) {
    if (size >= kMaxInvalidSegments) {
        cap_fail(qs, kCapInvalidSegments, size + 1, kMaxInvalidSegments);
        return;
    }
    v[size] = s;
    ++size;
}

/**
 * @brief Record the invalid runs created by an I-jump on active vertex @p idx.
 * Mirrors VerticesData::invalidate_i_jump.
 */
THESEUS_HD inline void vd_invalidate_i_jump(QueryState &qs, int idx, int diag) {
    const int gapo = qs.vd_gapo, gape = qs.vd_gape;
    InvalidSeg s;

    s.rem_down = gapo + gape;
    s.rem_up = gape;
    s.start_d = diag;
    s.end_d = diag;
    vd_invalid_push(qs, qs.vd_m_invalid[idx], qs.vd_m_invalid_size[idx], s);

    s.rem_down = 2 * gapo + 3 * gape;
    s.rem_up = gape;
    s.start_d = diag;
    s.end_d = diag;
    vd_invalid_push(qs, qs.vd_i_invalid[idx], qs.vd_i_invalid_size[idx], s);

    s.rem_down = gapo + gape;
    s.rem_up = gapo + 2 * gape;
    s.start_d = diag;
    s.end_d = diag - 1;
    vd_invalid_push(qs, qs.vd_d_invalid[idx], qs.vd_d_invalid_size[idx], s);
}

/**
 * @brief Record the invalid runs created by an M-jump on active vertex @p idx.
 * Mirrors VerticesData::invalidate_m_jump.
 */
THESEUS_HD inline void vd_invalidate_m_jump(QueryState &qs, int idx, int diag) {
    const int gapo = qs.vd_gapo, gape = qs.vd_gape;
    InvalidSeg s;

    s.rem_down = gapo + gape;
    s.rem_up = gapo + gape;
    s.start_d = diag;
    s.end_d = diag;
    vd_invalid_push(qs, qs.vd_m_invalid[idx], qs.vd_m_invalid_size[idx], s);

    s.rem_down = 2 * (gapo + gape);
    s.rem_up = gapo + gape;
    s.start_d = diag + 1;
    s.end_d = diag;
    vd_invalid_push(qs, qs.vd_i_invalid[idx], qs.vd_i_invalid_size[idx], s);

    s.rem_down = gapo + gape;
    s.rem_up = 2 * (gapo + gape);
    s.start_d = diag;
    s.end_d = diag - 1;
    vd_invalid_push(qs, qs.vd_d_invalid[idx], qs.vd_d_invalid_size[idx], s);
}

/**
 * @brief Whether @p diag is outside every invalid run of the given matrix for
 * vertex @p vtx. Mirrors VerticesData::valid_diagonal.
 */
THESEUS_HD inline bool vd_valid_diagonal(const QueryState &qs, Cell::Matrix matrix, int vtx,
                              int diag) {
    const int idx = qs.vd_vertex_to_idx[vtx];
    const InvalidSeg *v;
    int size;
    if (matrix == Cell::Matrix::M) {
        v = qs.vd_m_invalid[idx];
        size = qs.vd_m_invalid_size[idx];
    } else if (matrix == Cell::Matrix::I) {
        v = qs.vd_i_invalid[idx];
        size = qs.vd_i_invalid_size[idx];
    } else {
        v = qs.vd_d_invalid[idx];
        size = qs.vd_d_invalid_size[idx];
    }
    for (int l = 0; l < size; ++l) {
        if (v[l].start_d <= diag && diag <= v[l].end_d) {
            return false;
        }
    }
    return true;
}

// Jump-position lists of a vertex at a score's ring slot, and their sizes.
THESEUS_HD inline int64_t *vd_m_jumps(QueryState &qs, int vtx, int pos) { return qs.vd_m_jumps_pos[qs.vd_vertex_to_idx[vtx]][pos]; }
THESEUS_HD inline int64_t *vd_i_jumps(QueryState &qs, int vtx, int pos) { return qs.vd_i_jumps_pos[qs.vd_vertex_to_idx[vtx]][pos]; }
THESEUS_HD inline int32_t &vd_m_jumps_size(QueryState &qs, int vtx, int pos) { return qs.vd_m_jumps_pos_size[qs.vd_vertex_to_idx[vtx]][pos]; }
THESEUS_HD inline int32_t &vd_i_jumps_size(QueryState &qs, int vtx, int pos) { return qs.vd_i_jumps_pos_size[qs.vd_vertex_to_idx[vtx]][pos]; }

THESEUS_HD inline void vd_jumps_push(QueryState &qs, int64_t *arr, int32_t &size, int64_t val) {
    if (size >= kMaxJumpsPerScore) {
        cap_fail(qs, kCapJumpsPerScore, size + 1, kMaxJumpsPerScore);
        return;
    }
    arr[size] = val;
    ++size;
}

}  // namespace theseus
