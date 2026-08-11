# Come sono stati generati questi file

**Questo è l'unico file di questa cartella che non è stato prodotto da NVIDIA.**
Contiene solo i comandi eseguiti, nessun dato derivato. Se vuoi zero file non-NVIDIA,
cancellalo: il resto della cartella resta intatto e leggibile.

Tutto il resto è output diretto di `ncu` (Nsight Compute) e `nsys` (Nsight Systems),
salvato così com'è uscito, senza alcuna rielaborazione.

---

## Cosa è stato profilato

Commit `88ddef2225d5d72cdee194b9e4abcc24606f7ef6` del branch `main`, binario
`theseus_gpu/build-gpu/apps/seq2graph_proxy`, kernel `theseus_align_batch_kernel`.

Quattro dataset (`ebola_exact_smoke`, `ebola_error_smoke`, `c4_exact`, `c4_err`),
tre configurazioni di thread per blocco (64, 128, 256), tre ripetizioni.

Hardware: **Tesla T4** (CC 7.5), driver 580.82.07, CUDA 12.8.93,
Nsight Compute 2025.1.1.0, Nsight Systems 2025.6.3, su VM Google Colab.

Build profilata:

```bash
cmake -S theseus_gpu -B theseus_gpu/build-gpu -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON -DTHESEUS_PROXY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build theseus_gpu/build-gpu -j 4
```

Comando dell'applicazione, uguale in tutte le raccolte (`$THR` = 64, 128 o 256):

```bash
theseus_gpu/build-gpu/apps/seq2graph_proxy --backend gpu --require-gpu-result \
    --gpu-threads $THR -g <grafo>.gfa -s <query>.queries -f /tmp/out.gaf
```

---

## `nsight-compute/clock-1590MHz/`

Raccolta principale. Clock SM fissati a mano prima di profilare e **non** lasciati
gestire a `ncu`, da cui `--clock-control none`:

```bash
nvidia-smi -pm 1
nvidia-smi -lgc 1590,1590

ncu --set full --clock-control none \
    --kernel-name theseus_align_batch_kernel --launch-count 1 \
    -f -o <dataset>_<thr>_run<N> <comando applicazione>
```

Export CSV, dallo stesso report:

```bash
ncu -i <dataset>_<thr>_run<N>.ncu-rep --csv --page details > <dataset>_<thr>_run<N>.csv
ncu -i <dataset>_<thr>_run1.ncu-rep  --csv --page raw     > <dataset>_<thr>.raw.csv
```

- `*_run<N>.ncu-rep` — report binario, da aprire con `ncu-ui`
- `*_run<N>.csv` — pagina *details* (le sezioni con le metriche)
- `*.raw.csv` — pagina *raw* (tutte le metriche raccolte, ~1300 colonne), solo `run1`

## `nsight-compute/clock-base/`

Identica alla precedente **tranne l'assenza di `--clock-control none`**, cioè con il
comportamento di default di `ncu`, che fissa la GPU al clock base (585 MHz sulla T4).
Serve a un solo scopo: riprodurre le misure storiche del progetto, che erano state prese
con quel default.

```bash
ncu --set full --kernel-name theseus_align_batch_kernel --launch-count 1 \
    -f -o <dataset>_<thr>_run<N> <comando applicazione>

ncu -i <dataset>_<thr>_run<N>.ncu-rep --csv --page details > <dataset>_<thr>_run<N>.csv
ncu -i <dataset>_<thr>_run1.ncu-rep  --csv --page raw     > <dataset>_<thr>.raw.csv
```

**Di questa passata sono versionati i soli CSV, non i `.ncu-rep`.** I CSV contengono le
stesse metriche; i report binari pesavano 116 MB e servono solo se si vuole riaprire
questa passata secondaria nella GUI. Per rigenerarli basta rieseguire i comandi qui
sopra.

## `nsight-compute/lineinfo/`

Attribuzione del traffico alle righe di codice. Richiede una build separata compilata con
`-lineinfo` (`-DCMAKE_CUDA_FLAGS=-lineinfo`, per il resto identica), profilata a 128
thread:

```bash
ncu --set full --import-source yes \
    --kernel-name theseus_align_batch_kernel --launch-count 1 \
    -f -o <dataset>_128 <comando applicazione, build con -lineinfo>

ncu -i <dataset>_128.ncu-rep --page source --print-source "cuda,sass" --csv \
    > <dataset>_128.source.csv
```

`*.source.csv` ha una riga per riga di codice CUDA seguita dalle sue istruzioni SASS.

## `nsight-compute/metriche-mirate/`

Metriche non incluse in `--set full`, richieste per nome. Clock a 1590 MHz.

```bash
ncu --clock-control none --kernel-name theseus_align_batch_kernel --launch-count 1 \
    --metrics sm__sass_thread_inst_executed_op_integer_pred_on.sum,\
sm__sass_thread_inst_executed_op_fp32_pred_on.sum,\
sm__sass_thread_inst_executed_op_fp64_pred_on.sum,\
sm__sass_thread_inst_executed_op_control_pred_on.sum,\
sm__sass_thread_inst_executed_op_memory_pred_on.sum,\
sm__sass_thread_inst_executed_op_conversion_pred_on.sum,\
sm__sass_thread_inst_executed_op_misc_pred_on.sum,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,\
l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio,\
dram__bytes_read.sum,dram__bytes_write.sum,gpu__time_duration.sum \
    --csv <comando applicazione> > <dataset>_<thr>.csv
```

## `nsight-compute/testo/`

Gli stessi report di `clock-1590MHz/` (ripetizione `run1`) riversati in testo, per
leggerli senza aprire la GUI. Include le raccomandazioni automatiche di NVIDIA, le righe
`OPT`:

```bash
ncu --import clock-1590MHz/<dataset>_<thr>_run1.ncu-rep --page details > <dataset>_<thr>.txt
```

Nessun dato aggiunto: è lo stesso report, in un altro formato.

## `nsight-systems/`

Linea temporale dell'intera esecuzione, a 128 thread. Clock a 1590 MHz.

```bash
nsys profile --trace=cuda,osrt --cuda-memory-usage=true --force-overwrite=true \
     --output=<dataset>_128 <comando applicazione>

nsys stats --force-export=true \
     --report cuda_api_sum --report cuda_gpu_kern_sum \
     --report cuda_gpu_mem_time_sum --report cuda_gpu_mem_size_sum \
     --format table <dataset>_128.nsys-rep > <dataset>_128.stats.txt

nsys stats --force-export=true \
     --report cuda_api_sum --report cuda_gpu_kern_sum \
     --report cuda_gpu_mem_time_sum --report cuda_gpu_mem_size_sum \
     --format csv --output <dataset>_128 <dataset>_128.nsys-rep
```

- `*.nsys-rep` — da aprire con `nsys-ui`
- `*.stats.txt` — gli stessi riepiloghi in tabella di testo
- `*_cuda_*.csv` — gli stessi riepiloghi in CSV

I file `.sqlite` che `nsys stats` genera come cache non sono inclusi: si rigenerano da
soli alla prima invocazione di `nsys stats`.

---

## Aprire i report

Non serve una GPU per **leggere** un report già raccolto.

```bash
ncu-ui  nsight-compute/clock-1590MHz/c4_err_128_run1.ncu-rep
nsys-ui nsight-systems/c4_err_128.nsys-rep
```

Se gli strumenti non sono installati, i pacchetti sono autoconsistenti e non richiedono
root:

```bash
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-compute-2025.4.1_2025.4.1.2-1_amd64.deb
dpkg-deb -x nsight-compute-*.deb /tmp/nc
mv /tmp/nc/opt/nvidia/nsight-compute/2025.4.1 ~/.local/opt/nsight-compute
ln -s ~/.local/opt/nsight-compute/{ncu,ncu-ui} ~/.local/bin/

curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/nsight-systems-2025.6.3_2025.6.3.541-1_amd64.deb
dpkg-deb -x nsight-systems-*.deb /tmp/ns
mv /tmp/ns/opt/nvidia/nsight-systems/2025.6.3 ~/.local/opt/nsight-systems
ln -s ~/.local/opt/nsight-systems/bin/{nsys,nsys-ui} ~/.local/bin/
```

Un report scritto da `ncu` 2025.1.1 si apre con qualsiasi versione pari o successiva.

---

## Dimensioni

| | |
|---|---:|
| `nsight-compute/clock-1590MHz/` — 36 `.ncu-rep` | 118 MB |
| `nsight-compute/lineinfo/` — 4 `.ncu-rep` | 16 MB |
| tutti i CSV, i `.txt` e i `.nsys-rep` | 28 MB |
| **totale** | **161 MB** |

I 36 `.ncu-rep` della passata `clock-base` (116 MB) sono stati **esclusi
deliberatamente**: di quella passata restano i CSV con le stesse metriche.

Nessun singolo file supera i 5,7 MB, quindi non si incontra il limite di 100 MB per file
di GitHub né la soglia di avviso a 50 MB. Il peso è tutto nel numero di report binari.

I `.ncu-rep` sono l'unica forma in cui si possono aprire i grafici nativi di Nsight
(Roofline, diagramma della memoria, vista per riga di codice). I CSV accanto contengono
le **stesse metriche** in forma testuale, ma senza i grafici.

---

## Nota su una riga di avviso

I report di `clock-1590MHz/` mostrano, aprendoli:

> *Data collection happened without fixed GPU frequencies.*

È atteso e non indica un problema: significa che **`ncu` non ha gestito lui i clock**,
perché erano già stati fissati a 1590 MHz con `nvidia-smi -lgc`. I report di
`clock-base/` non hanno l'avviso perché lì è `ncu` a fissarli, al clock base.
