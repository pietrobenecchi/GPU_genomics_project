#!/bin/bash
# Fase B, seconda passata: stesse metriche ma con --clock-control none, cioe' ai clock
# bloccati da noi a 1590 MHz invece che al base clock (585 MHz) che ncu impone di default.
set -u
ROOT=/content/theseus
B=$ROOT/theseus_gpu
OUT=/content/out
mkdir -p $OUT/raw_boost $OUT/csv_boost $OUT/logs $OUT/roofline_boost
cd $ROOT

nvidia-smi -pm 1          >  $OUT/logs/clocks_boost.log 2>&1
nvidia-smi -lgc 1590,1590 >> $OUT/logs/clocks_boost.log 2>&1
nvidia-smi -q -d CLOCK    >> $OUT/logs/clocks_boost.log 2>&1

graph_of() { case $1 in ebola_*) echo ebola.gfa;; c4_*) echo c4.gfa;; esac; }
app() {
  echo "$B/build-gpu/apps/seq2graph_proxy --backend gpu --require-gpu-result --gpu-threads $2 \
-g $B/data/validation/ggbs/graphs/$(graph_of $1) \
-s $B/data/validation/ggbs/queries/$1.queries -f /tmp/pb2_${1}_${2}.gaf"
}

DATASETS="ebola_exact_smoke c4_exact ebola_error_smoke c4_err"
THREADS="64 128 256"
RUNS="1 2 3"

for ds in $DATASETS; do
  for t in $THREADS; do
    for r in $RUNS; do
      tag=${ds}_${t}_run${r}
      ncu --set full --clock-control none \
          --kernel-name theseus_align_batch_kernel --launch-count 1 \
          -f -o $OUT/raw_boost/$tag \
          $(app $ds $t) > $OUT/logs/boost_$tag.log 2>&1
      echo "NCU_RC_$tag=$?" >> $OUT/logs/boost_$tag.log
      ncu -i $OUT/raw_boost/$tag.ncu-rep --csv --page details \
          > $OUT/csv_boost/$tag.csv 2>> $OUT/logs/boost_$tag.log
      if [ "$r" = "1" ]; then
        ncu -i $OUT/raw_boost/$tag.ncu-rep --csv --page raw \
            > $OUT/roofline_boost/${ds}_${t}.raw.csv 2>/dev/null
      fi
    done
  done
done
echo "BOOST_PROFILING_DONE"
touch /content/phaseB2.done
