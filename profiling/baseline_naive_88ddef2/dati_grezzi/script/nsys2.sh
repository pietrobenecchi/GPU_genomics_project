#!/bin/bash
set -u
OUT=/content/out; mkdir -p $OUT/nsys $OUT/logs
U=https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/nsight-systems-2025.6.3_2025.6.3.541-1_amd64.deb
curl -sL -o /content/nsys.deb $U
dpkg -i /content/nsys.deb > $OUT/logs/install_nsys.log 2>&1 || apt-get -y -f install >> $OUT/logs/install_nsys.log 2>&1
export PATH=$PATH:$(dirname $(find /opt/nvidia/nsight-systems -name nsys -type f 2>/dev/null | head -1))
which nsys >> $OUT/logs/install_nsys.log 2>&1
nsys --version > $OUT/logs/nsys_version.txt 2>&1
echo "PATH_NSYS=$(which nsys)" >> $OUT/logs/nsys_version.txt

nvidia-smi -pm 1 >/dev/null 2>&1; nvidia-smi -lgc 1590,1590 >/dev/null 2>&1
B=/content/theseus/theseus_gpu
graph_of(){ case $1 in ebola_*) echo ebola.gfa;; c4_*) echo c4.gfa;; esac; }

for ds in ebola_exact_smoke c4_exact ebola_error_smoke c4_err; do
  nsys profile --trace=cuda,osrt --cuda-memory-usage=true --force-overwrite=true \
     --output=$OUT/nsys/${ds}_128 \
     $B/build-gpu/apps/seq2graph_proxy --backend gpu --require-gpu-result --gpu-threads 128 \
     -g $B/data/validation/ggbs/graphs/$(graph_of $ds) \
     -s $B/data/validation/ggbs/queries/${ds}.queries -f /tmp/nsys_${ds}.gaf \
     > $OUT/logs/nsys_${ds}.log 2>&1
  echo "NSYS_RC_${ds}=$?" >> $OUT/logs/nsys_${ds}.log

  nsys stats --force-export=true \
     --report cuda_api_sum --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
     --report cuda_gpu_mem_size_sum \
     --format table $OUT/nsys/${ds}_128.nsys-rep > $OUT/nsys/${ds}_128.stats.txt 2>&1
  nsys stats --force-export=true \
     --report cuda_api_sum --report cuda_gpu_kern_sum --report cuda_gpu_mem_time_sum \
     --report cuda_gpu_mem_size_sum \
     --format csv --output $OUT/nsys/${ds}_128 $OUT/nsys/${ds}_128.nsys-rep \
     >> $OUT/logs/nsys_${ds}.log 2>&1
done
echo NSYS2_DONE
cd /content && tar czf /content/nsys.tar.gz out/nsys out/logs
touch /content/nsys2.done
