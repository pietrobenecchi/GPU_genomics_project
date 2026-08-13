#!/usr/bin/env python3
"""Add clock64() phase counters to align_gpu.cu, in place.

Profiling-only: it is applied to a throwaway worktree, never committed, and the
anchors it keys on are lines that are identical in every commit of the campaign
so the same patcher instruments all of them the same way.

Phases: 0 sp_clear, 1 i, 2 d, 3 m, 4 extend, 5 loop total.
"""
import sys
from pathlib import Path

path = Path(sys.argv[1])
s = path.read_text()


def sub(old, new, count=1):
    global s
    n = s.count(old)
    if n != count:
        raise SystemExit(f"anchor found {n} times, expected {count}:\n{old[:200]}")
    s = s.replace(old, new)


PROLOGUE = """
// ---- phase profiling (build-time only, never committed enabled) ----------
__device__ unsigned long long d_phase[6];
#define PH_T0(v) long long v = clock64()
#define PH_ADD(i, v)                                                          \\
    do {                                                                      \\
        if (threadIdx.x == 0) {                                               \\
            atomicAdd(&d_phase[i],                                            \\
                      (unsigned long long)(clock64() - (v)));                 \\
        }                                                                     \\
    } while (0)
"""

# The counters go right after the first __global__ in the file, which puts them
# after every include and before every device function that uses them.
sub("__global__ void seq_length_kernel(", PROLOGUE + "\n__global__ void seq_length_kernel(")

# --- phase 0: the per-query clear ----------------------------------------
sub("""    const int32_t vd_fill = vd_map_fill_count(graph.num_vertices);""",
    """    PH_T0(ph_clear);
    const int32_t vd_fill = vd_map_fill_count(graph.num_vertices);""")
sub("""    for (int32_t i = tx; i < vd_fill; i += ntx) {
        qs.vd_vertex_to_idx[i] = -1;
    }
    __syncthreads();""",
    """    for (int32_t i = tx; i < vd_fill; i += ntx) {
        qs.vd_vertex_to_idx[i] = -1;
    }
    __syncthreads();
    PH_ADD(0, ph_clear);""")

# --- phase 1: I ------------------------------------------------------------
sub("""    generate_and_merge_i_candidates(qs, scoring, query, query_len, graph,
                                    score, v, shared_i_ranges,
                                    shared_i_candidates, shared_i_valid,
                                    shared_i_count, shared_warp_base,
                                    shared_accum, block_end,
                                    block_end_cell);""",
    """    PH_T0(ph_i);
    generate_and_merge_i_candidates(qs, scoring, query, query_len, graph,
                                    score, v, shared_i_ranges,
                                    shared_i_candidates, shared_i_valid,
                                    shared_i_count, shared_warp_base,
                                    shared_accum, block_end,
                                    block_end_cell);
    PH_ADD(1, ph_i);""")

# --- phase 2: D ------------------------------------------------------------
sub("""    sp_reset_block(qs);
    if (tx == 0) {
        shared_sparsify_plan =
            prepare_d_sparsify_plan(qs, scoring, query_len, graph, score, v);
    }""",
    """    PH_T0(ph_d);
    sp_reset_block(qs);
    if (tx == 0) {
        shared_sparsify_plan =
            prepare_d_sparsify_plan(qs, scoring, query_len, graph, score, v);
    }""")
sub("""        if (tx == 0) {
            sc_pos_push(qs, sc_d_pos(qs, score), sc_d_pos_size(qs, score), d_range);
        }
    }
    __syncthreads();""",
    """        if (tx == 0) {
            sc_pos_push(qs, sc_d_pos(qs, score), sc_d_pos_size(qs, score), d_range);
        }
    }
    __syncthreads();
    PH_ADD(2, ph_d);""")

# --- phase 3: M ------------------------------------------------------------
sub("""    sp_reset_block(qs);
    if (tx == 0) {
        shared_sparsify_plan =
            prepare_m_sparsify_plan(qs, scoring, query_len, graph, score, v);
    }""",
    """    PH_T0(ph_m);
    sp_reset_block(qs);
    if (tx == 0) {
        shared_sparsify_plan =
            prepare_m_sparsify_plan(qs, scoring, query_len, graph, score, v);
    }""")
sub("""        if (tx == 0) {
            sc_pos_push(qs, sc_m_pos(qs, score), sc_m_pos_size(qs, score), m_range);
        }
    }
    __syncthreads();""",
    """        if (tx == 0) {
            sc_pos_push(qs, sc_m_pos(qs, score), sc_m_pos_size(qs, score), m_range);
        }
    }
    __syncthreads();
    PH_ADD(3, ph_m);""")

# --- phase 4: extend (LCP + jumps) -----------------------------------------
sub("""    extend_and_consume_m_cells(qs, query, query_len, graph, score,
                               shared_range_start, shared_range_end,
                               shared_m_valid,
                               block_end, block_end_cell);
    __syncthreads();""",
    """    PH_T0(ph_ext);
    extend_and_consume_m_cells(qs, query, query_len, graph, score,
                               shared_range_start, shared_range_end,
                               shared_m_valid,
                               block_end, block_end_cell);
    __syncthreads();
    PH_ADD(4, ph_ext);""")

# --- phase 5: the whole score loop -----------------------------------------
sub("""    while (true) {
        if (tx == 0) {
            block_continue = (block_end == 0 && !qs.capacity_exceeded) ? 1 : 0;
        }""",
    """    PH_T0(ph_loop);
    while (true) {
        if (tx == 0) {
            block_continue = (block_end == 0 && !qs.capacity_exceeded) ? 1 : 0;
        }""")
sub("""    if (tx == 0) {
        --block_score;
        result.score = block_score;""",
    """    PH_ADD(5, ph_loop);
    if (tx == 0) {
        --block_score;
        result.score = block_score;""")

# --- host: zero before the launch, read back after -------------------------
sub("""    device_batch = BatchView{d_chars, d_offsets, batch.num_seqs};
    theseus_align_batch_kernel<<<batch.num_seqs, threads_per_block,""",
    """    {
        unsigned long long zero[6] = {0, 0, 0, 0, 0, 0};
        cudaMemcpyToSymbol(d_phase, zero, sizeof(zero));
    }
    device_batch = BatchView{d_chars, d_offsets, batch.num_seqs};
    theseus_align_batch_kernel<<<batch.num_seqs, threads_per_block,""")
sub("""    if (ev_kernel != nullptr) {
        cudaEventRecord(ev_kernel);
    }""",
    """    if (ev_kernel != nullptr) {
        cudaEventRecord(ev_kernel);
    }
    {
        unsigned long long ph[6] = {0, 0, 0, 0, 0, 0};
        cudaMemcpyFromSymbol(ph, d_phase, sizeof(ph));
        fprintf(stderr,
                "GPU phases: clear %llu i %llu d %llu m %llu extend %llu "
                "loop %llu\\n",
                ph[0], ph[1], ph[2], ph[3], ph[4], ph[5]);
    }""")

if "#include <cstdio>" not in s:
    s = s.replace("#include <cstdint>", "#include <cstdint>\n#include <cstdio>", 1)

path.write_text(s)
print(f"instrumented {path}")
