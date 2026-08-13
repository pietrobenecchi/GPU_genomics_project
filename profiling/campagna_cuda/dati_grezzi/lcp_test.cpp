// Simulates warp_lcp's ballot arithmetic against the serial core_lcp on random
// data. Same control flow as the device version, with the 32 lanes evaluated in
// a loop instead of in parallel.
#include <cstdint>
#include <cstdio>
#include <random>
#include <string>

static void serial_lcp(const char *q, int32_t qlen, const char *v, int32_t n,
                       int32_t &offset, int32_t &j) {
    while (offset < qlen && j < n && q[offset] == v[j]) { ++offset; ++j; }
}

static void warp_lcp(const char *q, int32_t qlen, const char *v, int32_t n,
                     int32_t &offset, int32_t &j) {
    for (;;) {
        const int32_t room_q = qlen - offset;
        const int32_t room_v = n - j;
        const int32_t remaining = room_q < room_v ? room_q : room_v;
        if (remaining <= 0) return;
        const int32_t chunk = remaining < 32 ? remaining : 32;
        unsigned int matches = 0;
        for (int lane = 0; lane < 32; ++lane) {
            const bool match = lane < chunk && q[offset + lane] == v[j + lane];
            if (match) matches |= (1u << lane);
        }
        const unsigned int wanted = chunk == 32 ? 0xFFFFFFFFu : ((1u << chunk) - 1u);
        const unsigned int mismatches = (~matches) & wanted;
        const int32_t advance = mismatches != 0u ? (__builtin_ffs((int)mismatches) - 1) : chunk;
        offset += advance; j += advance;
        if (advance < chunk) return;
    }
}

int main() {
    std::mt19937 rng(12345);
    const char *alpha = "ACGT";
    long cases = 0, bad = 0;
    for (int trial = 0; trial < 400000; ++trial) {
        const int qlen = 1 + (int)(rng() % 140);
        const int n = 1 + (int)(rng() % 140);
        // Bias towards long common prefixes so the multi-chunk path is exercised.
        const int shared = (int)(rng() % (std::min(qlen, n) + 1));
        std::string q(qlen, 'A'), v(n, 'A');
        for (int i = 0; i < qlen; ++i) q[i] = alpha[rng() % 4];
        for (int i = 0; i < n; ++i) v[i] = alpha[rng() % 4];
        for (int i = 0; i < shared && i < qlen && i < n; ++i) v[i] = q[i];
        const int off0 = (int)(rng() % (qlen + 1));
        const int j0 = (int)(rng() % (n + 1));
        int32_t a_off = off0, a_j = j0, b_off = off0, b_j = j0;
        serial_lcp(q.c_str(), qlen, v.c_str(), n, a_off, a_j);
        warp_lcp(q.c_str(), qlen, v.c_str(), n, b_off, b_j);
        ++cases;
        if (a_off != b_off || a_j != b_j) {
            if (++bad <= 3)
                printf("MISMATCH qlen=%d n=%d off0=%d j0=%d serial=(%d,%d) warp=(%d,%d)\n",
                       qlen, n, off0, j0, a_off, a_j, b_off, b_j);
        }
    }
    printf("%ld casi, %ld divergenti\n", cases, bad);
    return bad != 0;
}
