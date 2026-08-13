#!/bin/bash
# One detached run for the whole campaign, so the VM is up for the shortest
# time that still answers every open question in 06_handoff.md.
#
# Stages, in decreasing priority; each writes /content/logs/stage<N>.done so the
# session can be torn down as soon as the stage that matters has landed.
#
#   1  build + regression for the five commits never executed on a device
#   2  compute-sanitizer where the handoff says the argument needs a proof
#   3  --repeat, to split the per-process cost from the per-batch one (B and D)
#   4  timing matrix over the builds
#   5  reduced ncu counters (fewer points than ncu.sh: credits)
#
# Usage: setsid nohup /content/campagna.sh > /content/logs/campagna.log 2>&1 &

ROOT=/content/theseus
G=$ROOT/theseus_gpu/data/validation/ggbs/graphs
Q=$ROOT/theseus_gpu/data/validation/ggbs/queries
P=/content/logs/progress.txt

mkdir -p /content/wt /content/logs /content/gpu_results /content/ncu/csv
: > $P
note() { echo "[$(date +%H:%M:%S)] $*" >> $P; }

nvidia-smi -pm 1 > /dev/null 2>&1
nvidia-smi -lgc 1590,1590 >> /content/logs/clocks.log 2>&1
note "clocks: $(nvidia-smi --query-gpu=name,clocks.applications.gr --format=csv,noheader)"

# ---------------------------------------------------------------- stage 1 ---
# tile first: it is the tip and carries every other change, so if credits run
# out after one build the backend that would ship is the one that got tested.
build_one() { # name commit
    local name=$1 commit=$2 wt=/content/wt/$1
    rm -rf "$wt"; git -C $ROOT worktree prune
    git -C $ROOT worktree add --detach "$wt" "$commit" \
        > /content/logs/wt_$name.log 2>&1 || { note "WORKTREE FAIL $name"; return 1; }
    cmake -S "$wt/theseus_gpu" -B "$wt/build-gpu" \
          -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
          -DTHESEUS_PROXY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
          > /content/logs/cmake_$name.log 2>&1 || { note "CMAKE FAIL $name"; return 1; }
    cmake --build "$wt/build-gpu" -j "$(nproc)" \
          > /content/logs/build_$name.log 2>&1 || { note "BUILD FAIL $name"; return 1; }
    note "built $name ($commit)"
}

regr_one() { # name
    local name=$1 wt=/content/wt/$1
    {
        echo "===== $name : $(git -C "$wt" rev-parse --short HEAD) ====="
        ctest --test-dir "$wt/build-gpu" --output-on-failure 2>&1 | tail -8
        (cd "$wt" && python3 $ROOT/scripts/run_ggbs_gpu_regression.py \
            --suite all --build-dir "$wt/build-gpu" \
            --output-dir /content/gpu_results/$name --timeout 900) 2>&1
        echo "REGRESSION_EXIT $?"
    } > /content/logs/regr_$name.log 2>&1
    note "regr $name: $(grep -c 'PASS' /content/logs/regr_$name.log) pass / $(grep -c 'FAIL' /content/logs/regr_$name.log) fail; $(tail -1 /content/logs/regr_$name.log)"
}

for spec in tile:e71a0e1 nomemset:f1c93ea lazy:41bb8b1 warp:5dae2ce; do
    build_one "${spec%%:*}" "${spec##*:}" && regr_one "${spec%%:*}"
done
# measurement-only builds: prelazy is D's control, c1 the timing baseline (they
# were validated before, and mixing VMs is not allowed, so they get rebuilt here)
for spec in prelazy:d9f813f c1:d84fd82; do
    build_one "${spec%%:*}" "${spec##*:}"
done
echo DONE > /content/logs/stage1.done
note "stage 1 done"

# ---------------------------------------------------------------- stage 2 ---
# Small datasets only: the tools cost 10-50x and the questions are structural.
san() { # tool build dataset graph queries threads
    local tool=$1 b=$2 ds=$3 g=$4 q=$5 t=$6
    timeout 1800 compute-sanitizer --tool "$tool" \
        /content/wt/$b/build-gpu/apps/seq2graph_proxy --backend gpu \
        --require-gpu-result --gpu-threads $t \
        -g $G/$g.gfa -s $Q/$q.queries -f /tmp/san.gaf \
        > /content/logs/san_${tool}_${b}_${ds}_${t}.log 2>&1
    local rc=$?
    note "san $tool $b $ds $t rc=$rc errors=$(grep -c '= *ERROR\|error' /content/logs/san_${tool}_${b}_${ds}_${t}.log)"
}

# f1c93ea dropped the per-batch memset on an argument, not a measurement:
# initcheck is the proof. 41bb8b1 leaves sp_off dirty past the seen prefix.
for b in nomemset lazy; do
    san initcheck $b ebola_exact_smoke ebola ebola_exact_smoke 64
    san initcheck $b c4_err            c4    c4_err            256
    san memcheck  $b c4_err            c4    c4_err            256
done
# 5dae2ce moves the barriers, e71a0e1 puts the query in shared memory.
for b in warp tile; do
    san racecheck $b ebola_error_smoke ebola ebola_error_smoke 64
    san synccheck $b c4_err            c4    c4_err            256
    san memcheck  $b c4_err            c4    c4_err            256
done
echo DONE > /content/logs/stage2.done
note "stage 2 done"

# ---------------------------------------------------------------- stage 3 ---
rep() { # build dataset graph queries threads n
    local b=$1 ds=$2 g=$3 q=$4 t=$5 n=$6
    /content/wt/$b/build-gpu/apps/seq2graph_proxy --backend gpu \
        --require-gpu-result --gpu-threads $t --repeat $n \
        -g $G/$g.gfa -s $Q/$q.queries -f /tmp/rep.gaf \
        > /content/logs/rep_${b}_${ds}_${t}.out 2> /content/logs/rep_${b}_${ds}_${t}.log
    note "repeat $b $ds $t rc=$?"
}
for b in prelazy lazy tile; do
    rep $b c4_err_2k     c4    c4_err_2k     128 5
    rep $b ebola_exact_2k ebola ebola_exact_2k 128 5
    rep $b c4_exact      c4    c4_exact      128 5
done
echo DONE > /content/logs/stage3.done
note "stage 3 done"

# ---------------------------------------------------------------- stage 4 ---
python3 /content/timing.py c1 nomemset lazy warp tile \
    > /content/logs/timing.log 2>&1
note "stage 4 timing rc=$?"
echo DONE > /content/logs/stage4.done

# ---------------------------------------------------------------- stage 5 ---
# Reduced against ncu.sh: three builds x two datasets x one thread count.
M=$(tr '\n' ',' <<'EOF' | sed 's/,$//'
l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum
l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum
dram__bytes_read.sum
dram__bytes_write.sum
dram__throughput.avg.pct_of_peak_sustained_elapsed
gpu__time_duration.sum
launch__registers_per_thread
launch__occupancy_limit_registers
launch__shared_mem_per_block_static
sm__warps_active.avg.pct_of_peak_sustained_active
smsp__average_warp_latency_per_inst_issued.ratio
smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio
smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio
smsp__inst_executed.sum
local__load_bytes.sum
local__store_bytes.sum
EOF
)
for b in c1 warp tile; do
    for spec in "c4_err:c4:c4_err" "ebola_err_2k:ebola:ebola_err_2k"; do
        IFS=: read ds g q <<< "$spec"
        timeout 1800 ncu --clock-control none --kernel-name theseus_align_batch_kernel \
            --launch-count 1 --metrics "$M" --csv \
            /content/wt/$b/build-gpu/apps/seq2graph_proxy --backend gpu \
            --require-gpu-result --gpu-threads 128 \
            -g $G/$g.gfa -s $Q/$q.queries -f /tmp/ncu.gaf \
            > /content/ncu/csv/${b}_${ds}_128.csv 2> /content/ncu/csv/${b}_${ds}_128.err
        note "ncu $b $ds rc=$?"
    done
done
echo DONE > /content/logs/stage5.done
note "ALL DONE"
