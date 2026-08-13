#!/bin/bash
# Build every commit named on the command line into /content/wt/<name>.
# Usage: build.sh <name>:<commit> [<name>:<commit> ...]
set -x
cd /content/theseus || exit 1
for spec in "$@"; do
    name="${spec%%:*}"
    commit="${spec##*:}"
    wt=/content/wt/$name
    rm -rf "$wt"
    git worktree prune
    git worktree add --detach "$wt" "$commit" || exit 1
    cmake -S "$wt/theseus_gpu" -B "$wt/build-gpu" \
          -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
          -DTHESEUS_PROXY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
          > "/content/logs/cmake_$name.log" 2>&1 || exit 1
    cmake --build "$wt/build-gpu" -j 2 > "/content/logs/build_$name.log" 2>&1 || exit 1
    echo "BUILT $name $commit"
done
echo DONE
