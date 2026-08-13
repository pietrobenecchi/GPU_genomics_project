#!/bin/bash
set -u
OUT=/content/out; mkdir -p $OUT/opmix $OUT/logs $OUT/env
cd /content
rm -rf /content/theseus && mkdir -p /content/theseus
tar xzf /content/theseus.tar.gz -C /content/theseus
cd /content/theseus
nvidia-smi > $OUT/env/nvidia-smi-sessione2.txt 2>&1
nvidia-smi -pm 1 >> $OUT/env/nvidia-smi-sessione2.txt 2>&1
nvidia-smi -lgc 1590,1590 >> $OUT/env/nvidia-smi-sessione2.txt 2>&1
nvidia-smi -q -d CLOCK >> $OUT/env/nvidia-smi-sessione2.txt 2>&1

cmake -S theseus_gpu -B theseus_gpu/build-gpu -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
      -DTHESEUS_PROXY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 > $OUT/logs/build2.log 2>&1
cmake --build theseus_gpu/build-gpu -j 4 >> $OUT/logs/build2.log 2>&1
echo "BUILD2_RC=$?" >> $OUT/logs/build2.log

B=/content/theseus/theseus_gpu
graph_of(){ case $1 in ebola_*) echo ebola.gfa;; c4_*) echo c4.gfa;; esac; }
app(){ echo "$B/build-gpu/apps/seq2graph_proxy --backend gpu --require-gpu-result --gpu-threads $2 \
-g $B/data/validation/ggbs/graphs/$(graph_of $1) -s $B/data/validation/ggbs/queries/$1.queries -f /tmp/g_${1}_${2}.gaf"; }

DATASETS="ebola_exact_smoke c4_exact ebola_error_smoke c4_err"

# --- 1. tempi end-to-end SENZA profiler, 5 ripetizioni, cronometrati in bash ---
: > $OUT/logs/timing_clean.log
for ds in $DATASETS; do for t in 64 128 256; do for r in 1 2 3 4 5; do
  s=$(date +%s.%N)
  $(app $ds $t) > /tmp/o.txt 2> /tmp/e.txt
  rc=$?
  e=$(date +%s.%N)
  echo "RUN ds=$ds threads=$t rep=$r rc=$rc wall_s=$(echo "$e - $s" | bc)" >> $OUT/logs/timing_clean.log
  grep -o 'Aligned [0-9]* sequences in [0-9]* microseconds' /tmp/o.txt >> $OUT/logs/timing_clean.log
  grep -o 'timing_ms .*' /tmp/e.txt >> $OUT/logs/timing_clean.log
done; done; done
echo TIMING_DONE

# --- 2. mix di istruzioni + settori per richiesta ---
M="sm__sass_thread_inst_executed_op_integer_pred_on.sum,sm__sass_thread_inst_executed_op_fp32_pred_on.sum,sm__sass_thread_inst_executed_op_fp64_pred_on.sum,sm__sass_thread_inst_executed_op_control_pred_on.sum,sm__sass_thread_inst_executed_op_memory_pred_on.sum,sm__sass_thread_inst_executed_op_conversion_pred_on.sum,sm__sass_thread_inst_executed_op_misc_pred_on.sum,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum"
for ds in $DATASETS; do for t in 64 128 256; do
  ncu --clock-control none --kernel-name theseus_align_batch_kernel --launch-count 1 \
      --metrics $M --csv $(app $ds $t) > $OUT/opmix/${ds}_${t}.csv 2> $OUT/opmix/${ds}_${t}.err
done; done
echo OPMIX_DONE
cd /content && tar czf /content/gap.tar.gz out/opmix out/logs/timing_clean.log out/logs/build2.log out/env/nvidia-smi-sessione2.txt
touch /content/gap.done
