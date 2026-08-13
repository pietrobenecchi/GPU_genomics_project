#!/bin/bash
# Nsight Compute counters for each build x dataset, clocks already pinned by hand.
ROOT=/content/theseus
OUT=/content/ncu
mkdir -p $OUT/csv

M=""
add() { M="${M:+$M,}$1"; }
add l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
add l1tex__t_requests_pipe_lsu_mem_global_op_st.sum
add l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio
add l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
add l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio
add dram__bytes_read.sum
add dram__bytes_write.sum
add dram__throughput.avg.pct_of_peak_sustained_elapsed
add gpu__time_duration.sum
add launch__registers_per_thread
add launch__occupancy_limit_registers
add launch__occupancy_limit_shared_mem
add launch__occupancy_limit_blocks
add launch__occupancy_limit_warps
add launch__shared_mem_per_block_static
add sm__warps_active.avg.pct_of_peak_sustained_active
add smsp__average_warp_latency_per_inst_issued.ratio
add smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
add smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio
add smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
add smsp__average_warps_issue_stalled_short_scoreboard_per_issue_active.ratio
add smsp__inst_executed.sum
add sm__sass_inst_executed_op_global_st.sum
add sm__sass_inst_executed_op_global_ld.sum
add local__load_bytes.sum
add local__store_bytes.sum

run() { # build dataset graph queries threads
    local b=$1 ds=$2 g=$3 q=$4 t=$5
    ncu --clock-control none --kernel-name theseus_align_batch_kernel \
        --launch-count 1 --metrics "$M" --csv \
        /content/wt/$b/build-gpu/apps/seq2graph_proxy --backend gpu \
        --require-gpu-result --gpu-threads $t \
        -g $ROOT/$g -s $ROOT/$q -f /tmp/ncu.gaf \
        > $OUT/csv/${b}_${ds}_${t}.csv 2> $OUT/csv/${b}_${ds}_${t}.err
    echo "done $b $ds $t rc=$?"
}

G=theseus_gpu/data/validation/ggbs/graphs
Q=theseus_gpu/data/validation/ggbs/queries
for b in "$@"; do
    for t in 64 128 256; do
        run $b c4_err       $G/c4.gfa    $Q/c4_err.queries       $t
    done
    run $b c4_exact      $G/c4.gfa    $Q/c4_exact.queries      128
    run $b c4_err_2k     $G/c4.gfa    $Q/c4_err_2k.queries     128
    run $b c4_exact_2k   $G/c4.gfa    $Q/c4_exact_2k.queries   128
    run $b ebola_err_2k  $G/ebola.gfa $Q/ebola_err_2k.queries  128
    run $b ebola_exact_2k $G/ebola.gfa $Q/ebola_exact_2k.queries 128
done
echo DONE > /content/logs/ncu.done
