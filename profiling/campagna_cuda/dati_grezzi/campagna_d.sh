#!/bin/bash
# Fourth run: validate the fix for what initcheck found (zero the QueryState
# array once per allocation instead of once per batch), and prove it closes the
# site rather than moving it.
#
#   D1  build + regression, all ten datasets x 64/128/256
#   D2  initcheck on the same 8-record inputs that produced the 73 758 reads
#   D3  the per-batch cost the fix must not bring back: --repeat, where zeroing
#       once per allocation and once per batch differ by construction
#   D4  timing + registers, so the fix is on record next to the others

ROOT=/content/theseus
G=$ROOT/theseus_gpu/data/validation/ggbs/graphs
Q=$ROOT/theseus_gpu/data/validation/ggbs/queries
T=/content/tiny
P=/content/logs/progress_d.txt
: > $P
note() { echo "[$(date +%H:%M:%S)] $*" >> $P; }

nvidia-smi -lgc 1590,1590 > /dev/null 2>&1
git -C $ROOT fetch -q /content/theseus3.bundle 'refs/heads/*:refs/remotes/new3/*' 2>>$P

name=zeroalloc; wt=/content/wt/$name
rm -rf "$wt"; git -C $ROOT worktree prune
git -C $ROOT worktree add --detach "$wt" "$1" > /content/logs/wt_$name.log 2>&1
for flags in "" "-lineinfo"; do
    dir=build-gpu; [ -n "$flags" ] && dir=build-li
    cmake -S "$wt/theseus_gpu" -B "$wt/$dir" -DCMAKE_BUILD_TYPE=Release \
          -DBUILD_TESTING=ON -DTHESEUS_PROXY_ENABLE_CUDA=ON \
          -DCMAKE_CUDA_ARCHITECTURES=75 ${flags:+-DCMAKE_CUDA_FLAGS=$flags} \
          > /content/logs/cmake_${name}_$dir.log 2>&1 \
      && cmake --build "$wt/$dir" -j "$(nproc)" > /content/logs/build_${name}_$dir.log 2>&1 \
      && note "built $name $dir" || note "BUILD FAIL $name $dir"
done

# -- D1 ----------------------------------------------------------------------
{
    ctest --test-dir "$wt/build-gpu" --output-on-failure 2>&1 | tail -8
    (cd "$wt" && python3 $ROOT/scripts/run_ggbs_gpu_regression.py --suite all \
        --build-dir "$wt/build-gpu" --output-dir /content/gpu_results/$name \
        --timeout 900) 2>&1
    echo "REGRESSION_EXIT $?"
} > /content/logs/regr_$name.log 2>&1
note "regr $name: $(grep -c PASS /content/logs/regr_$name.log) pass; $(tail -1 /content/logs/regr_$name.log)"

# -- D2 ----------------------------------------------------------------------
for spec in "ebola_exact_8:ebola:64" "c4_err_8:c4:256"; do
    IFS=: read ds g t <<< "$spec"
    timeout 1800 compute-sanitizer --tool initcheck --print-limit 20 \
        "$wt/build-li/apps/seq2graph_proxy" --backend gpu --require-gpu-result \
        --gpu-threads $t -g $G/$g.gfa -s $T/$ds.queries -f /tmp/san.gaf \
        > /content/logs/d_initcheck_${name}_${ds}.log 2>&1
    note "initcheck $name $ds: $(grep -m1 'ERROR SUMMARY' /content/logs/d_initcheck_${name}_${ds}.log)"
done
timeout 1800 compute-sanitizer --tool memcheck "$wt/build-gpu/apps/seq2graph_proxy" \
    --backend gpu --require-gpu-result --gpu-threads 256 \
    -g $G/c4.gfa -s $T/c4_err_8.queries -f /tmp/san.gaf \
    > /content/logs/d_memcheck_${name}.log 2>&1
note "memcheck $name: $(grep -m1 'ERROR SUMMARY' /content/logs/d_memcheck_${name}.log)"

# -- D3 ----------------------------------------------------------------------
for spec in "c4_err_2k:c4" "c4_exact:c4" "ebola_exact_2k:ebola"; do
    IFS=: read ds g <<< "$spec"
    "$wt/build-gpu/apps/seq2graph_proxy" --backend gpu --require-gpu-result \
        --gpu-threads 128 --repeat 5 -g $G/$g.gfa -s $Q/$ds.queries -f /tmp/rep.gaf \
        > /content/logs/rep_${name}_${ds}.out 2> /content/logs/rep_${name}_${ds}.log
    note "repeat $name $ds rc=$?"
done

# -- D4 ----------------------------------------------------------------------
python3 /content/timing.py zeroalloc > /content/logs/timing_zeroalloc.log 2>&1
note "timing rc=$?"
M="gpu__time_duration.sum,launch__registers_per_thread,sm__warps_active.avg.pct_of_peak_sustained_active,smsp__inst_executed.sum,smsp__average_warp_latency_per_inst_issued.ratio"
for spec in "c4_err:c4" "ebola_err_2k:ebola"; do
    IFS=: read ds g <<< "$spec"
    timeout 1200 ncu --clock-control none --kernel-name theseus_align_batch_kernel \
        --launch-count 1 --metrics "$M" --csv \
        "$wt/build-gpu/apps/seq2graph_proxy" --backend gpu --require-gpu-result \
        --gpu-threads 128 -g $G/$g.gfa -s $Q/$ds.queries -f /tmp/ncu.gaf \
        > /content/ncu/csv/${name}_${ds}_128.csv 2>/dev/null
    note "ncu $name $ds rc=$?"
done

cd /content && tar czf /content/results_d.tgz logs/progress_d.txt logs/regr_zeroalloc.log \
    logs/d_*.log logs/rep_zeroalloc_*.log logs/timing_zeroalloc.log logs/timing.json \
    ncu/csv/*.csv 2>/dev/null
echo DONE > /content/logs/d.done
note "ALL DONE"
