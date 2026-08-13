#!/bin/bash
# Fase B - baseline di profiling. Lanciato detached, segnala la fine con /content/phaseB.done
set -u
ROOT=/content/theseus
B=$ROOT/theseus_gpu
OUT=/content/out
mkdir -p $OUT/raw $OUT/csv $OUT/logs $OUT/roofline $OUT/srcattr
cd $ROOT

nvidia-smi -pm 1            >  $OUT/logs/clocks.log 2>&1
nvidia-smi -lgc 1590,1590   >> $OUT/logs/clocks.log 2>&1
nvidia-smi -q -d CLOCK      >> $OUT/logs/clocks.log 2>&1

graph_of() { case $1 in ebola_*) echo ebola.gfa;; c4_*) echo c4.gfa;; esac; }

app() {   # ds threads outgaf builddir
  echo "$B/$4/apps/seq2graph_proxy --backend gpu --require-gpu-result --gpu-threads $2 \
-g $B/data/validation/ggbs/graphs/$(graph_of $1) \
-s $B/data/validation/ggbs/queries/$1.queries -f $3"
}

DATASETS="ebola_exact_smoke c4_exact ebola_error_smoke c4_err"
THREADS="64 128 256"
RUNS="1 2 3"

# ---------- 1. Tempi end-to-end senza profiler ----------
: > $OUT/logs/timing_clean.log
for ds in $DATASETS; do
  for t in $THREADS; do
    for r in $RUNS; do
      echo "===== $ds $t run$r =====" >> $OUT/logs/timing_clean.log
      /usr/bin/time -f "WALL_SECONDS=%e MAXRSS_KB=%M" \
        $(app $ds $t /tmp/clean_${ds}_${t}.gaf build-gpu) \
        >> $OUT/logs/timing_clean.log 2>&1
    done
  done
done
echo "TIMING_CLEAN_DONE"

# ---------- 2. Profiling ncu --set full ----------
for ds in $DATASETS; do
  for t in $THREADS; do
    for r in $RUNS; do
      tag=${ds}_${t}_run${r}
      ncu --set full \
          --kernel-name theseus_align_batch_kernel --launch-count 1 \
          -f -o $OUT/raw/$tag \
          $(app $ds $t /tmp/prof_${ds}_${t}.gaf build-gpu) \
          > $OUT/logs/$tag.log 2>&1
      echo "NCU_RC_$tag=$?" >> $OUT/logs/$tag.log
      ncu -i $OUT/raw/$tag.ncu-rep --csv --page details \
          > $OUT/csv/$tag.csv 2>> $OUT/logs/$tag.log
      echo "EXPORT_RC_$tag=$?" >> $OUT/logs/$tag.log
      # sezione roofline isolata, una sola volta per combinazione
      if [ "$r" = "1" ]; then
        ncu -i $OUT/raw/$tag.ncu-rep --csv --page raw \
            > $OUT/roofline/${ds}_${t}.raw.csv 2>/dev/null
      fi
    done
  done
done
echo "PROFILING_DONE"

# ---------- 3. Build separata con -lineinfo per l'attribuzione per riga ----------
cmake -S $B -B $B/build-gpu-lineinfo -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
      -DTHESEUS_PROXY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_FLAGS=-lineinfo > $OUT/logs/build_lineinfo.log 2>&1
cmake --build $B/build-gpu-lineinfo -j 4 >> $OUT/logs/build_lineinfo.log 2>&1
echo "BUILD_LINEINFO_RC=$?" >> $OUT/logs/build_lineinfo.log

# la build con -lineinfo deve ancora passare la regressione, altrimenti
# l'attribuzione non descrive il binario validato
python3 scripts/run_ggbs_gpu_regression.py --suite all \
    --build-dir theseus_gpu/build-gpu-lineinfo \
    --output-dir theseus_gpu/data/validation/ggbs/gpu_results_lineinfo \
    > $OUT/logs/regressione_lineinfo.log 2>&1
echo "EXIT_REGR_LINEINFO=$?" >> $OUT/logs/regressione_lineinfo.log

for ds in $DATASETS; do
  tag=${ds}_128
  ncu --set full --import-source yes \
      --kernel-name theseus_align_batch_kernel --launch-count 1 \
      -f -o $OUT/srcattr/$tag \
      $(app $ds 128 /tmp/src_${ds}.gaf build-gpu-lineinfo) \
      > $OUT/logs/srcattr_$tag.log 2>&1
  echo "NCU_SRC_RC_$tag=$?" >> $OUT/logs/srcattr_$tag.log
  ncu -i $OUT/srcattr/$tag.ncu-rep --csv --page source --print-source no \
      > $OUT/srcattr/$tag.source.csv 2>> $OUT/logs/srcattr_$tag.log
  echo "EXPORT_SRC_RC_$tag=$?" >> $OUT/logs/srcattr_$tag.log
done
echo "SRCATTR_DONE"

touch /content/phaseB.done
