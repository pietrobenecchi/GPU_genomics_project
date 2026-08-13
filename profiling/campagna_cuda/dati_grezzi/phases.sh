#!/bin/bash
# Phase cycle counts from the instrumented builds, plus a regression pass on
# them so the numbers are known to describe correct behaviour.
ROOT=/content/theseus
G=$ROOT/theseus_gpu/data/validation/ggbs/graphs
Q=$ROOT/theseus_gpu/data/validation/ggbs/queries
OUT=/content/logs/phases.txt
: > $OUT

for b in base_ph c1_ph; do
    bin=/content/wt/$b/build-gpu/apps/seq2graph_proxy
    for spec in "c4_err:c4:c4_err" "c4_exact:c4:c4_exact" \
                "c4_err_2k:c4:c4_err_2k" "c4_exact_2k:c4:c4_exact_2k" \
                "ebola_err_2k:ebola:ebola_err_2k" "ebola_exact_2k:ebola:ebola_exact_2k" \
                "ebola_error_smoke:ebola:ebola_error_smoke"; do
        IFS=: read ds g q <<< "$spec"
        for t in 128; do
            # warm-up then one measured run
            $bin --backend gpu --require-gpu-result --gpu-threads $t \
                 -g $G/$g.gfa -s $Q/$q.queries -f /tmp/ph.gaf > /dev/null 2>/dev/null
            line=$($bin --backend gpu --require-gpu-result --gpu-threads $t \
                 -g $G/$g.gfa -s $Q/$q.queries -f /tmp/ph.gaf 2>&1 >/dev/null \
                 | grep "GPU phases:")
            echo "$b $ds $t $line" >> $OUT
        done
    done
done

echo "----- regressione sulle build strumentate -----" >> $OUT
for b in base_ph c1_ph; do
    (cd /content/wt/$b && python3 $ROOT/scripts/run_ggbs_gpu_regression.py \
        --suite all --build-dir /content/wt/$b/build-gpu \
        --output-dir /content/gpu_results/$b --timeout 900 2>&1 | tail -3) >> $OUT
done
echo DONE > /content/logs/phases.done
