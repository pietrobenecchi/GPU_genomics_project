#!/bin/bash
# Correctness matrix for each named worktree: ctest, then every dataset x 64/128/256.
cd /content/theseus || exit 1
for name in "$@"; do
    wt=/content/wt/$name
    log=/content/logs/regr_$name.log
    {
        echo "===== $name : $(git -C "$wt" rev-parse --short HEAD) ====="
        echo "----- ctest -----"
        ctest --test-dir "$wt/build-gpu" --output-on-failure 2>&1 | tail -12
        echo "----- regression, all datasets -----"
        # Run from the worktree so the relative dataset paths resolve there.
        (cd "$wt" && python3 /content/theseus/scripts/run_ggbs_gpu_regression.py \
            --suite all --build-dir "$wt/build-gpu" \
            --output-dir /content/gpu_results/$name --timeout 900) 2>&1
        echo "REGRESSION_EXIT $?"
    } > "$log" 2>&1
done
echo DONE > /content/logs/regr.done
