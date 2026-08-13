#!/bin/bash
# Third detached run: finish the sanitizer questions the handoff left open,
# now that -lineinfo and 8-record inputs make each one cost seconds.
#
#   C1  where the uninitialised reads are: full site distribution on nomemset,
#       and whether the tip (tile/regfix) inherits them
#   C2  racecheck + synccheck on warp (moved barriers) and tile (shared query)
#   C3  memcheck everywhere, cheap and worth having on record

ROOT=/content/theseus
G=$ROOT/theseus_gpu/data/validation/ggbs/graphs
T=/content/tiny
P=/content/logs/progress_c.txt
: > $P
note() { echo "[$(date +%H:%M:%S)] $*" >> $P; }

san() { # tool build dataset graph threads printlimit
    local tool=$1 b=$2 ds=$3 g=$4 t=$5 pl=${6:-500}
    local bin=/content/wt/$b/build-li/apps/seq2graph_proxy
    [ -x "$bin" ] || bin=/content/wt/$b/build-gpu/apps/seq2graph_proxy
    timeout 1800 compute-sanitizer --tool "$tool" --print-limit "$pl" \
        "$bin" --backend gpu --require-gpu-result --gpu-threads $t \
        -g $G/$g.gfa -s $T/$ds.queries -f /tmp/san.gaf \
        > /content/logs/c_${tool}_${b}_${ds}_${t}.log 2>&1
    note "$tool $b $ds t$t: $(grep -m1 'ERROR SUMMARY' /content/logs/c_${tool}_${b}_${ds}_${t}.log)"
}

# -- C1: sites, not counts ---------------------------------------------------
san initcheck nomemset ebola_exact_8 ebola 64 500
for b in tile regfix; do
    wt=/content/wt/$b
    if [ ! -x "$wt/build-li/apps/seq2graph_proxy" ]; then
        cmake -S "$wt/theseus_gpu" -B "$wt/build-li" -DCMAKE_BUILD_TYPE=Release \
              -DTHESEUS_PROXY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
              -DCMAKE_CUDA_FLAGS=-lineinfo > /content/logs/cmake_li_$b.log 2>&1 \
          && cmake --build "$wt/build-li" -j "$(nproc)" > /content/logs/build_li_$b.log 2>&1 \
          && note "built $b -lineinfo"
    fi
    san initcheck $b ebola_exact_8 ebola 64 200
done

# -- C2: the two commits that move barriers ----------------------------------
for b in warp tile regfix; do
    san racecheck $b ebola_exact_8 ebola 64 100
    san synccheck $b ebola_exact_8 ebola 64 100
    san racecheck $b c4_err_8      c4    256 100
done

# -- C3 ----------------------------------------------------------------------
for b in nomemset lazy warp tile regfix; do
    san memcheck $b c4_err_8 c4 256 50
done
echo DONE > /content/logs/c.done
note "ALL DONE"
