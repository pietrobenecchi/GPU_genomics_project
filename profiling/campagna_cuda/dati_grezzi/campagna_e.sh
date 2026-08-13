#!/bin/bash
# Fifth run: the two commits of category 2 (thread coarsening), against their
# parent on the same VM.
#
#   base  7053193  azzeramento per allocazione   -- the comparison point
#   clear 02127db  coarsening del clear          -- 4 words per thread, int4
#   empty 23dc718  densify vuota senza barriere
#
# fill_words is new memory arithmetic on a 52 224-word window, so memcheck and
# initcheck matter here as much as the regression does.

ROOT=/content/theseus
G=$ROOT/theseus_gpu/data/validation/ggbs/graphs
Q=$ROOT/theseus_gpu/data/validation/ggbs/queries
T=/content/tiny
P=/content/logs/progress_e.txt
: > $P
note() { echo "[$(date +%H:%M:%S)] $*" >> $P; }

mkdir -p $T /content/logs /content/ncu/csv /content/wt
nvidia-smi -pm 1 > /dev/null 2>&1
nvidia-smi -lgc 1590,1590 > /dev/null 2>&1
git -C $ROOT fetch -q /content/theseus4.bundle 'refs/heads/*:refs/remotes/new4/*' 2>>$P

build() { # name commit
    local wt=/content/wt/$1
    rm -rf "$wt"; git -C $ROOT worktree prune
    git -C $ROOT worktree add --detach "$wt" "$2" > /content/logs/wt_$1.log 2>&1 || { note "WT FAIL $1"; return 1; }
    cmake -S "$wt/theseus_gpu" -B "$wt/build-gpu" -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_TESTING=ON -DTHESEUS_PROXY_ENABLE_CUDA=ON \
          -DCMAKE_CUDA_ARCHITECTURES=75 > /content/logs/cmake_$1.log 2>&1 \
      && cmake --build "$wt/build-gpu" -j "$(nproc)" > /content/logs/build_$1.log 2>&1 \
      && note "built $1 ($2)" || { note "BUILD FAIL $1"; return 1; }
}

regr() { # name
    local wt=/content/wt/$1
    {
        ctest --test-dir "$wt/build-gpu" --output-on-failure 2>&1 | tail -6
        (cd "$wt" && python3 $ROOT/scripts/run_ggbs_gpu_regression.py --suite all \
            --build-dir "$wt/build-gpu" --output-dir /content/gpu_results/$1 \
            --timeout 900) 2>&1
        echo "REGRESSION_EXIT $?"
    } > /content/logs/regr_$1.log 2>&1
    note "regr $1: $(grep -c PASS /content/logs/regr_$1.log) pass; $(tail -1 /content/logs/regr_$1.log)"
}

for spec in base:7053193 clear:02127db empty:23dc718; do
    build "${spec%%:*}" "${spec##*:}" && regr "${spec%%:*}"
done
echo DONE > /content/logs/e1.done

# -- sanitizers: fill_words is the reason -----------------------------------
head -16 $Q/ebola_exact_smoke.queries > $T/ebola_exact_8.queries
head -16 $Q/c4_err.queries            > $T/c4_err_8.queries
for b in clear empty; do
    for tool in memcheck initcheck racecheck; do
        for spec in "ebola_exact_8:ebola:64" "c4_err_8:c4:256"; do
            IFS=: read ds g t <<< "$spec"
            timeout 1800 compute-sanitizer --tool $tool --print-limit 20 \
                /content/wt/$b/build-gpu/apps/seq2graph_proxy --backend gpu \
                --require-gpu-result --gpu-threads $t \
                -g $G/$g.gfa -s $T/$ds.queries -f /tmp/san.gaf \
                > /content/logs/e_${tool}_${b}_${ds}_${t}.log 2>&1
            note "$tool $b $ds t$t: $(grep -m1 -E 'ERROR SUMMARY|RACECHECK SUMMARY' /content/logs/e_${tool}_${b}_${ds}_${t}.log)"
        done
    done
done
echo DONE > /content/logs/e2.done

# -- steady state, where the clear is supposed to disappear anyway ------------
for b in base clear empty; do
    for spec in "c4_err_2k:c4" "c4_exact:c4" "c4_exact_2k:c4" "ebola_exact_2k:ebola"; do
        IFS=: read ds g <<< "$spec"
        /content/wt/$b/build-gpu/apps/seq2graph_proxy --backend gpu \
            --require-gpu-result --gpu-threads 128 --repeat 5 \
            -g $G/$g.gfa -s $Q/$ds.queries -f /tmp/rep.gaf \
            > /content/logs/rep_${b}_${ds}.out 2> /content/logs/rep_${b}_${ds}.log
        note "repeat $b $ds rc=$?"
    done
done
echo DONE > /content/logs/e3.done

python3 /content/timing.py base clear empty > /content/logs/timing_e.log 2>&1
note "timing rc=$?"

M="gpu__time_duration.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,smsp__average_warp_latency_per_inst_issued.ratio,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio,l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum,l1tex__t_requests_pipe_lsu_mem_global_op_st.sum,dram__bytes_write.sum"
for b in base clear empty; do
    for spec in "c4_exact:c4" "c4_err:c4" "ebola_err_2k:ebola"; do
        IFS=: read ds g <<< "$spec"
        timeout 1200 ncu --clock-control none --kernel-name theseus_align_batch_kernel \
            --launch-count 1 --metrics "$M" --csv \
            /content/wt/$b/build-gpu/apps/seq2graph_proxy --backend gpu \
            --require-gpu-result --gpu-threads 128 \
            -g $G/$g.gfa -s $Q/$ds.queries -f /tmp/ncu.gaf \
            > /content/ncu/csv/e_${b}_${ds}_128.csv 2>/dev/null
        note "ncu $b $ds rc=$?"
    done
done

cd /content && tar czf /content/results_e.tgz logs/progress_e.txt logs/regr_base.log \
    logs/regr_clear.log logs/regr_empty.log logs/e_*.log logs/rep_*.log \
    logs/timing_e.log logs/timing.json ncu/csv/e_*.csv 2>/dev/null
echo DONE > /content/logs/e.done
note "ALL DONE"
