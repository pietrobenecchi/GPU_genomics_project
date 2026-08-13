#!/bin/bash
# Sixth run: the packed traceback D2H (e9435fe) against its parent (23dc718).
# The D2H was 95 % of the per-batch GPU time and 99,8 % of it was slack, so the
# numbers to watch are d2h_ms and host_buffers_ms, not the kernel.
ROOT=/content/theseus
G=$ROOT/theseus_gpu/data/validation/ggbs/graphs
Q=$ROOT/theseus_gpu/data/validation/ggbs/queries
T=/content/tiny
P=/content/logs/progress_f.txt
: > $P
note() { echo "[$(date +%H:%M:%S)] $*" >> $P; }
mkdir -p $T /content/logs /content/ncu/csv /content/wt
nvidia-smi -pm 1 >/dev/null 2>&1; nvidia-smi -lgc 1590,1590 >/dev/null 2>&1
git -C $ROOT fetch -q /content/theseus5.bundle 'refs/heads/*:refs/remotes/new5/*' 2>>$P

build() {
    local wt=/content/wt/$1
    rm -rf "$wt"; git -C $ROOT worktree prune
    git -C $ROOT worktree add --detach "$wt" "$2" > /content/logs/wt_$1.log 2>&1 || { note "WT FAIL $1"; return 1; }
    cmake -S "$wt/theseus_gpu" -B "$wt/build-gpu" -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_TESTING=ON -DTHESEUS_PROXY_ENABLE_CUDA=ON \
          -DCMAKE_CUDA_ARCHITECTURES=75 > /content/logs/cmake_$1.log 2>&1 \
      && cmake --build "$wt/build-gpu" -j "$(nproc)" > /content/logs/build_$1.log 2>&1 \
      && note "built $1 ($2)" || { note "BUILD FAIL $1"; return 1; }
}
regr() {
    local wt=/content/wt/$1
    {
        ctest --test-dir "$wt/build-gpu" --output-on-failure 2>&1 | tail -6
        (cd "$wt" && python3 $ROOT/scripts/run_ggbs_gpu_regression.py --suite all \
            --build-dir "$wt/build-gpu" --output-dir /content/gpu_results/$1 --timeout 900) 2>&1
        echo "REGRESSION_EXIT $?"
    } > /content/logs/regr_$1.log 2>&1
    note "regr $1: $(grep -c PASS /content/logs/regr_$1.log) pass; $(tail -1 /content/logs/regr_$1.log)"
}
for spec in prev:23dc718 packed:e9435fe; do
    build "${spec%%:*}" "${spec##*:}" && regr "${spec%%:*}"
done
echo DONE > /content/logs/f1.done

head -16 $Q/ebola_exact_smoke.queries > $T/ebola_exact_8.queries
head -16 $Q/c4_err.queries            > $T/c4_err_8.queries
for tool in memcheck initcheck racecheck; do
    for spec in "ebola_exact_8:ebola:64" "c4_err_8:c4:256"; do
        IFS=: read ds g t <<< "$spec"
        timeout 1800 compute-sanitizer --tool $tool --print-limit 20 \
            /content/wt/packed/build-gpu/apps/seq2graph_proxy --backend gpu \
            --require-gpu-result --gpu-threads $t -g $G/$g.gfa -s $T/$ds.queries \
            -f /tmp/san.gaf > /content/logs/f_${tool}_${ds}_${t}.log 2>&1
        note "$tool packed $ds t$t: $(grep -m1 -E 'ERROR SUMMARY|RACECHECK SUMMARY' /content/logs/f_${tool}_${ds}_${t}.log)"
    done
done
echo DONE > /content/logs/f2.done

for b in prev packed; do
    for spec in "c4_err_2k:c4" "c4_exact_2k:c4" "ebola_err_2k:ebola" "c4_err:c4"; do
        IFS=: read ds g <<< "$spec"
        /content/wt/$b/build-gpu/apps/seq2graph_proxy --backend gpu --require-gpu-result \
            --gpu-threads 128 --repeat 5 -g $G/$g.gfa -s $Q/$ds.queries -f /tmp/rep.gaf \
            > /content/logs/rep_${b}_${ds}.out 2> /content/logs/rep_${b}_${ds}.log
        note "repeat $b $ds rc=$?"
    done
done
python3 /content/timing.py prev packed > /content/logs/timing_f.log 2>&1
note "timing rc=$?"
cd /content && tar czf /content/results_f.tgz logs/progress_f.txt logs/regr_prev.log \
    logs/regr_packed.log logs/f_*.log logs/rep_prev_*.log logs/rep_packed_*.log \
    logs/timing_f.log 2>/dev/null
echo DONE > /content/logs/f.done
note "ALL DONE"
