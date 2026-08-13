#!/bin/bash
set -u
OUT=/content/out; mkdir -p $OUT/nsys $OUT/logs
rm -rf /content/theseus && mkdir -p /content/theseus
tar xzf /content/theseus.tar.gz -C /content/theseus
cd /content/theseus

nsys --version > $OUT/logs/nsys_version.txt 2>&1
nvidia-smi -pm 1 >/dev/null 2>&1
nvidia-smi -lgc 1590,1590 >> $OUT/logs/nsys_version.txt 2>&1

cmake -S theseus_gpu -B theseus_gpu/build-gpu -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
      -DTHESEUS_PROXY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 > $OUT/logs/build_nsys.log 2>&1
cmake --build theseus_gpu/build-gpu -j 4 >> $OUT/logs/build_nsys.log 2>&1
echo "BUILD_RC=$?" >> $OUT/logs/build_nsys.log

B=/content/theseus/theseus_gpu
graph_of(){ case $1 in ebola_*) echo ebola.gfa;; c4_*) echo c4.gfa;; esac; }

for ds in ebola_exact_smoke c4_exact ebola_error_smoke c4_err; do
  nsys profile \
     --trace=cuda,nvtx,osrt \
     --cuda-memory-usage=true \
     --force-overwrite=true \
     --output=$OUT/nsys/${ds}_128 \
     $B/build-gpu/apps/seq2graph_proxy --backend gpu --require-gpu-result --gpu-threads 128 \
     -g $B/data/validation/ggbs/graphs/$(graph_of $ds) \
     -s $B/data/validation/ggbs/queries/${ds}.queries -f /tmp/nsys_${ds}.gaf \
     > $OUT/logs/nsys_${ds}.log 2>&1
  echo "NSYS_RC_${ds}=$?" >> $OUT/logs/nsys_${ds}.log

  # riepiloghi testuali standard di nsys
  nsys stats --force-export=true \
     --report cuda_api_sum --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
     --report cuda_gpu_mem_size_sum --report cuda_gpu_sum \
     --format table $OUT/nsys/${ds}_128.nsys-rep \
     > $OUT/nsys/${ds}_128.stats.txt 2>&1
  nsys stats --force-export=true \
     --report cuda_api_sum --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
     --report cuda_gpu_mem_size_sum --report cuda_gpu_sum \
     --format csv --output $OUT/nsys/${ds}_128 \
     $OUT/nsys/${ds}_128.nsys-rep >> $OUT/logs/nsys_${ds}.log 2>&1
done
echo NSYS_DONE
cd /content && tar czf /content/nsys.tar.gz out/nsys out/logs
touch /content/nsys.done
