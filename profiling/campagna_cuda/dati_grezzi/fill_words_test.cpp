// Exhaustive host check of the index arithmetic of fill_words (align_gpu.cu).
//
// The device version is thread coarsening on the clear: three loops, a scalar
// prologue up to the first 16-byte boundary, an int4 body, a scalar epilogue.
// What has to hold is that the three together cover [begin, end) exactly once
// and touch nothing else, for every alignment of the array, every window the
// query can ask for and every block size the kernel accepts.
//
// Build: g++ -std=c++17 -O2 fill_words_test.cpp -o fill_words_test && ./fill_words_test

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace {

constexpr int32_t kGuard = 8;     // words of padding checked for spill on both sides
constexpr int32_t kValue = -1;

// Same arithmetic as the device function, with the stores recorded instead.
void fill_words_sim(int32_t *base, int32_t begin, int32_t end, int32_t value,
                    int32_t ntx, std::vector<int> &writes) {
    if (begin >= end) {
        return;
    }
    const uintptr_t addr = reinterpret_cast<uintptr_t>(base + begin);
    const int32_t skip = static_cast<int32_t>((16u - (addr & 15u)) & 15u) >> 2;
    int32_t vec_begin = begin + skip;
    if (vec_begin > end) {
        vec_begin = end;
    }
    const int32_t nvec = (end - vec_begin) >> 2;
    const int32_t vec_end = vec_begin + (nvec << 2);

    for (int32_t tx = 0; tx < ntx; ++tx) {
        for (int32_t i = begin + tx; i < vec_begin; i += ntx) {
            base[i] = value;
            ++writes[static_cast<size_t>(i)];
        }
        for (int32_t i = tx; i < nvec; i += ntx) {
            for (int32_t w = 0; w < 4; ++w) {
                const int32_t idx = vec_begin + (i << 2) + w;
                base[idx] = value;
                ++writes[static_cast<size_t>(idx)];
            }
        }
        for (int32_t i = vec_end + tx; i < end; i += ntx) {
            base[i] = value;
            ++writes[static_cast<size_t>(i)];
        }
    }

    // The vector loop must be the one doing the work: at most three words at
    // each end may go through the scalar paths.
    if (skip > 3 || (end - vec_end) > 3) {
        std::printf("FAIL ragged ends too long: skip=%d tail=%d\n", skip, end - vec_end);
        std::exit(1);
    }
}

bool check(int32_t misalign_words, int32_t begin, int32_t end, int32_t ntx) {
    // A buffer whose base is deliberately off a 16-byte boundary by
    // misalign_words, the way an odd QueryState is (sizeof is 8 mod 16).
    std::vector<int32_t> storage(static_cast<size_t>(end + 3 * kGuard + 8), 0);
    // 64-bit aligned by construction, then pushed by whole words.
    int32_t *aligned = storage.data();
    while ((reinterpret_cast<uintptr_t>(aligned) & 15u) != 0) {
        ++aligned;
    }
    int32_t *base = aligned + misalign_words + kGuard;

    std::vector<int> writes(static_cast<size_t>(end + kGuard), 0);
    for (int32_t i = -kGuard; i < end + kGuard; ++i) {
        base[i] = 7;   // poison, including the guard on both sides
    }
    fill_words_sim(base, begin, end, kValue, ntx, writes);

    for (int32_t i = begin; i < end; ++i) {
        if (base[i] != kValue || writes[static_cast<size_t>(i)] != 1) {
            std::printf("FAIL covered: mis=%d begin=%d end=%d ntx=%d i=%d "
                        "value=%d writes=%d\n",
                        misalign_words, begin, end, ntx, i, base[i],
                        writes[static_cast<size_t>(i)]);
            return false;
        }
    }
    for (int32_t i = -kGuard; i < begin; ++i) {
        if (base[i] != 7) {
            std::printf("FAIL underrun: mis=%d begin=%d end=%d ntx=%d i=%d\n",
                        misalign_words, begin, end, ntx, i);
            return false;
        }
    }
    // The one that matters: a 128-bit store must never reach past `end`.
    for (int32_t i = end; i < end + kGuard; ++i) {
        if (base[i] != 7) {
            std::printf("FAIL overrun: mis=%d begin=%d end=%d ntx=%d i=%d\n",
                        misalign_words, begin, end, ntx, i);
            return false;
        }
    }
    return true;
}

}  // namespace

int main() {
    long cases = 0;
    for (int32_t mis = 0; mis < 4; ++mis) {
        for (int32_t ntx : {32, 64, 128, 256}) {
            for (int32_t begin = 0; begin <= 9; ++begin) {
                for (int32_t end = begin; end <= 600; ++end) {
                    if (!check(mis, begin, end, ntx)) {
                        return 1;
                    }
                    ++cases;
                }
            }
        }
    }
    // and the real shapes: the whole c4 window, and a window that starts inside
    for (int32_t mis = 0; mis < 4; ++mis) {
        for (int32_t begin : {0, 1, 2, 3, 4, 51999, 52223}) {
            if (!check(mis, begin, 52224, 128)) {
                return 1;
            }
            ++cases;
        }
    }
    std::printf("OK: %ld cases, coverage exact, no overrun\n", cases);
    return 0;
}
