#!/bin/bash
# Second detached run, after stage 1 of campagna.sh closed the validation
# question (four commits, 30/30 each) and initcheck opened a new one.
#
#   B1  regfix (__noinline__ process_vertex): build + regression
#   B2  --repeat, per-process vs per-batch cost (handoff 3.4)
#   B3  timing matrix, now including regfix
#   B4  initcheck with -lineinfo on tiny inputs: *which* read is uninitialised,
#       and whether c1 -- the commit before the memset was dropped -- is clean
#   B5  reduced ncu counters
#
# B4 uses 8-record inputs on purpose: the full ebola_exact_smoke took 11 minutes
# under initcheck, and the question is which source line reads uninitialised
# memory, not how many times it does.

ROOT=/content/theseus
G=$ROOT/theseus_gpu/data/validation/ggbs/graphs
Q=$ROOT/theseus_gpu/data/validation/ggbs/queries
T=/content/tiny
P=/content/logs/progress_b.txt

mkdir -p $T /content/logs /content/ncu/csv
: > $P
note() { echo "[$(date +%H:%M:%S)] $*" >> $P; }

nvidia-smi -pm 1 > /dev/null 2>&1
nvidia-smi -lgc 1590,1590 > /dev/null 2>&1

# ------------------------------------------------------------------- B1 -----
git -C $ROOT fetch -q /content/theseus2.bundle 'refs/heads/*:refs/remotes/new/*' 2>>$P
name=regfix; wt=/content/wt/$name
rm -rf "$wt"; git -C $ROOT worktree prune
git -C $ROOT worktree add --detach "$wt" "$1" > /content/logs/wt_$name.log 2>&1 || note "WORKTREE FAIL"
cmake -S "$wt/theseus_gpu" -B "$wt/build-gpu" -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON -DTHESEUS_PROXY_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=75 > /content/logs/cmake_$name.log 2>&1 \
  && cmake --build "$wt/build-gpu" -j "$(nproc)" > /content/logs/build_$name.log 2>&1 \
  && note "built regfix ($1)" || note "BUILD FAIL regfix"
{
    ctest --test-dir "$wt/build-gpu" --output-on-failure 2>&1 | tail -8
    (cd "$wt" && python3 $ROOT/scripts/run_ggbs_gpu_regression.py --suite all \
        --build-dir "$wt/build-gpu" --output-dir /content/gpu_results/$name \
        --timeout 900) 2>&1
    echo "REGRESSION_EXIT $?"
} > /content/logs/regr_regfix.log 2>&1
note "regr regfix: $(grep -c PASS /content/logs/regr_regfix.log) pass; $(tail -1 /content/logs/regr_regfix.log)"
echo DONE > /content/logs/b1.done

# ------------------------------------------------------------------- B2 -----
rep() { # build dataset graph queries threads n
    /content/wt/$1/build-gpu/apps/seq2graph_proxy --backend gpu \
        --require-gpu-result --gpu-threads $5 --repeat $6 \
        -g $G/$3.gfa -s $Q/$4.queries -f /tmp/rep.gaf \
        > /content/logs/rep_${1}_${2}.out 2> /content/logs/rep_${1}_${2}.log
    note "repeat $1 $2 rc=$?"
}
for b in prelazy lazy tile regfix; do
    rep $b c4_err_2k      c4    c4_err_2k      128 5
    rep $b ebola_exact_2k ebola ebola_exact_2k 128 5
    rep $b c4_exact       c4    c4_exact       128 5
done
echo DONE > /content/logs/b2.done
note "B2 done"

# ------------------------------------------------------------------- B3 -----
python3 /content/timing.py c1 nomemset lazy warp tile regfix \
    > /content/logs/timing.log 2>&1
note "B3 timing rc=$?"
echo DONE > /content/logs/b3.done

# ------------------------------------------------------------------- B5 -----
# before B4: the counters answer whether 138 registers actually buy occupancy.
M=$(tr '\n' ',' <<'EOF' | sed 's/,$//'
gpu__time_duration.sum
launch__registers_per_thread
launch__occupancy_limit_registers
launch__occupancy_limit_shared_mem
launch__shared_mem_per_block_static
sm__warps_active.avg.pct_of_peak_sustained_active
smsp__average_warp_latency_per_inst_issued.ratio
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__inst_executed.sum
local__load_bytes.sum
local__store_bytes.sum
dram__bytes_read.sum
dram__bytes_write.sum
EOF
)
for b in c1 tile regfix; do
    for spec in "c4_err:c4:c4_err" "ebola_err_2k:ebola:ebola_err_2k"; do
        IFS=: read ds g q <<< "$spec"
        timeout 1200 ncu --clock-control none --kernel-name theseus_align_batch_kernel \
            --launch-count 1 --metrics "$M" --csv \
            /content/wt/$b/build-gpu/apps/seq2graph_proxy --backend gpu \
            --require-gpu-result --gpu-threads 128 \
            -g $G/$g.gfa -s $Q/$q.queries -f /tmp/ncu.gaf \
            > /content/ncu/csv/${b}_${ds}_128.csv 2> /content/ncu/csv/${b}_${ds}_128.err
        note "ncu $b $ds rc=$?"
    done
done
echo DONE > /content/logs/b5.done

# ------------------------------------------------------------------- B4 -----
head -16 $Q/ebola_exact_smoke.queries > $T/ebola_exact_8.queries
head -16 $Q/c4_err.queries            > $T/c4_err_8.queries
for spec in c1 nomemset regfix; do
    wt=/content/wt/$spec
    cmake -S "$wt/theseus_gpu" -B "$wt/build-li" -DCMAKE_BUILD_TYPE=Release \
          -DTHESEUS_PROXY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
          -DCMAKE_CUDA_FLAGS=-lineinfo > /content/logs/cmake_li_$spec.log 2>&1 \
      && cmake --build "$wt/build-li" -j "$(nproc)" > /content/logs/build_li_$spec.log 2>&1 \
      && note "built $spec -lineinfo" || { note "LINEINFO BUILD FAIL $spec"; continue; }
    for ds in ebola_exact_8 c4_err_8; do
        g=ebola; [ "$ds" = c4_err_8 ] && g=c4
        timeout 1800 compute-sanitizer --tool initcheck --print-limit 6 \
            "$wt/build-li/apps/seq2graph_proxy" --backend gpu --require-gpu-result \
            --gpu-threads 64 -g $G/$g.gfa -s $T/$ds.queries -f /tmp/san.gaf \
            > /content/logs/li_initcheck_${spec}_${ds}.log 2>&1
        note "initcheck $spec $ds: $(grep -m1 'ERROR SUMMARY' /content/logs/li_initcheck_${spec}_${ds}.log)"
    done
done
echo DONE > /content/logs/b4.done
note "ALL DONE"
