#!/bin/bash
# Build phase-instrumented worktrees. Usage: build_phase.sh <name>:<commit> ...
set -x
cd /content/theseus || exit 1
for spec in "$@"; do
    name="${spec%%:*}"; commit="${spec##*:}"
    wt=/content/wt/${name}_ph
    rm -rf "$wt"; git worktree prune
    git worktree add --detach "$wt" "$commit" || exit 1
    python3 /content/phase_patch.py "$wt/theseus_gpu/src/gpu/align_gpu.cu" || exit 1
    cmake -S "$wt/theseus_gpu" -B "$wt/build-gpu" \
          -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
          -DTHESEUS_PROXY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
          > "/content/logs/cmake_${name}_ph.log" 2>&1 || exit 1
    cmake --build "$wt/build-gpu" -j 2 > "/content/logs/build_${name}_ph.log" 2>&1 || exit 1
    echo "BUILT ${name}_ph $commit"
done
echo DONE > /content/logs/build_phase.done
