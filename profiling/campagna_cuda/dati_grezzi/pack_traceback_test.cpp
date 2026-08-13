// Host check of the packed-traceback layout (align_gpu.cu).
//
// align_batch replaced three strided cudaMemcpy2D of kBeyondScopeCapacity cells
// per query with: an exclusive prefix sum of the three used sizes computed on
// the host, a device kernel that gathers each query's prefixes at that base,
// one D2H of the total, and three pointers per query back into the result.
//
// Host and device compute the layout from the same numbers, so the thing that
// can go wrong is the arithmetic, not the transfer. This simulates both sides
// and checks that every query reads back exactly the cells it wrote, in order,
// including the empty wavefronts that are the common case (46-47 % of densify
// calls end with nothing, and c4_exact_2k has an empty M wavefront on every
// query).
//
// Build: g++ -std=c++17 -O2 pack_traceback_test.cpp -o pack_traceback_test && ./pack_traceback_test

#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

constexpr int32_t kBeyondScopeCapacity = 4096;

struct Cell {
    int64_t prev_pos;
    int32_t vertex_id;
    int32_t offset;
    int32_t diag;
    int8_t from_matrix;
    bool operator!=(const Cell &o) const {
        return prev_pos != o.prev_pos || vertex_id != o.vertex_id ||
               offset != o.offset || diag != o.diag || from_matrix != o.from_matrix;
    }
};

// One query's device-side state, only the fields the packing reads.
struct State {
    std::vector<Cell> m, mj, ij;
};

Cell make_cell(int32_t q, int32_t which, int32_t k) {
    return Cell{static_cast<int64_t>(q) * 1000000 + which * 10000 + k, q, which, k,
                static_cast<int8_t>(which)};
}

bool run(uint32_t seed, int32_t count, bool force_empty) {
    std::mt19937 rng(seed);
    std::vector<State> states(static_cast<size_t>(count));
    for (int32_t q = 0; q < count; ++q) {
        auto pick = [&]() -> int32_t {
            if (force_empty) return 0;
            const int r = static_cast<int>(rng() % 100);
            if (r < 45) return 0;                                  // the common case
            if (r < 95) return static_cast<int32_t>(rng() % 64);
            return static_cast<int32_t>(rng() % kBeyondScopeCapacity);
        };
        State &s = states[static_cast<size_t>(q)];
        const int32_t nm = pick(), nmj = pick(), nij = pick();
        for (int32_t k = 0; k < nm; ++k) s.m.push_back(make_cell(q, 0, k));
        for (int32_t k = 0; k < nmj; ++k) s.mj.push_back(make_cell(q, 1, k));
        for (int32_t k = 0; k < nij; ++k) s.ij.push_back(make_cell(q, 2, k));
    }

    // --- host: the exclusive prefix sum align_batch computes from the metadata
    size_t total = 0;
    std::vector<int32_t> base(static_cast<size_t>(count));
    for (int32_t q = 0; q < count; ++q) {
        base[static_cast<size_t>(q)] = static_cast<int32_t>(total);
        const State &s = states[static_cast<size_t>(q)];
        total += s.m.size() + s.mj.size() + s.ij.size();
    }

    // --- device: one block per query, three prefixes at base[q]
    std::vector<Cell> packed(total > 0 ? total : 1, Cell{-9, -9, -9, -9, -9});
    for (int32_t q = 0; q < count; ++q) {
        const State &s = states[static_cast<size_t>(q)];
        Cell *out = packed.data() + base[static_cast<size_t>(q)];
        const int32_t m = static_cast<int32_t>(s.m.size());
        const int32_t mj = static_cast<int32_t>(s.mj.size());
        const int32_t ij = static_cast<int32_t>(s.ij.size());
        for (int32_t k = 0; k < m; ++k) out[k] = s.m[static_cast<size_t>(k)];
        for (int32_t k = 0; k < mj; ++k) out[m + k] = s.mj[static_cast<size_t>(k)];
        for (int32_t k = 0; k < ij; ++k) out[m + mj + k] = s.ij[static_cast<size_t>(k)];
    }

    // --- host: the three views align_batch hands to the backtrace
    for (int32_t q = 0; q < count; ++q) {
        const State &s = states[static_cast<size_t>(q)];
        const Cell *slice = packed.data() + base[static_cast<size_t>(q)];
        const int32_t m = static_cast<int32_t>(s.m.size());
        const int32_t mj = static_cast<int32_t>(s.mj.size());
        const Cell *view_m = slice;
        const Cell *view_mj = slice + m;
        const Cell *view_ij = slice + m + mj;
        for (size_t k = 0; k < s.m.size(); ++k)
            if (view_m[k] != s.m[k]) { std::printf("FAIL m q=%d k=%zu\n", q, k); return false; }
        for (size_t k = 0; k < s.mj.size(); ++k)
            if (view_mj[k] != s.mj[k]) { std::printf("FAIL mj q=%d k=%zu\n", q, k); return false; }
        for (size_t k = 0; k < s.ij.size(); ++k)
            if (view_ij[k] != s.ij[k]) { std::printf("FAIL ij q=%d k=%zu\n", q, k); return false; }
    }

    // Nothing left unwritten inside the packed range: the D2H copies exactly
    // `total` cells, so an unwritten one would reach the backtrace.
    for (size_t i = 0; i < total; ++i) {
        if (packed[i].vertex_id == -9) { std::printf("FAIL hole at %zu\n", i); return false; }
    }
    return true;
}

}  // namespace

int main() {
    int runs = 0;
    for (uint32_t seed = 1; seed <= 300; ++seed) {
        for (int32_t count : {1, 2, 7, 64, 512}) {
            if (!run(seed, count, false)) return 1;
            ++runs;
        }
    }
    if (!run(999, 2048, true)) return 1;   // every wavefront empty: total == 0
    ++runs;
    std::printf("OK: %d batches packed and read back, no holes\n", runs);
    return 0;
}
