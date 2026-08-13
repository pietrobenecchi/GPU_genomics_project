# Ricreare la sessione di misura da zero

Tutto quello che serve per rieseguire Fase A e Fase B. Nessun passo implicito.

## 0. Hardware e software effettivamente usati

| | |
|---|---|
| GPU | **Tesla T4**, compute capability **7.5**, 15360 MiB, 40 SM |
| Driver | 580.82.07 (CUDA runtime del driver: 13.0) |
| CUDA toolkit | 12.8.93 (`/usr/local/cuda/bin/nvcc`, build `cuda_12.8.r12.8`) |
| Nsight Compute | 2025.1.1.0 (build 35528883), `/usr/local/cuda/bin/ncu` |
| Host compiler | gcc / g++ 11.4.0 (Ubuntu 11.4.0-1ubuntu1~22.04.3) |
| CMake | 3.31.10 |
| Python | 3.12.13 |
| SO | Ubuntu 22.04.5 LTS, kernel 6.6.122+ |
| Clock SM | **bloccato a 1590 MHz** (`nvidia-smi -lgc 1590,1590`, riuscito) |
| Clock memoria | 5001 MHz (massimo di targa, non regolabile su T4) |

`nvidia-smi` integrale in [`nvidia-smi.txt`](nvidia-smi.txt), versioni integrali in
[`versioni.txt`](versioni.txt), esito del blocco dei clock in
[`../logs/clocks.log`](../logs/clocks.log).

Tutto il software sopra è **preinstallato** sull'immagine GPU di Colab: non è stato
installato né aggiornato nulla sulla VM. Non serve un ambiente conda **sulla VM** —
l'ambiente conda `colab-cli` è solo sul lato locale, per il CLI che pilota la sessione.

## 1. Lato locale: il CLI

```bash
conda activate colab-cli     # env locale, contiene solo il pacchetto colab-cli
colab sessions               # verifica che l'autenticazione funzioni
```

Se `colab sessions` dà 401/403, rifare l'ADC con tutti e quattro gli scope:

```bash
gcloud auth application-default login \
  --scopes=openid,https://www.googleapis.com/auth/cloud-platform,\
https://www.googleapis.com/auth/userinfo.email,\
https://www.googleapis.com/auth/colaboratory
```

## 2. Allocare la VM

```bash
colab new -s theseus-gpu --gpu T4
colab status -s theseus-gpu     # deve dire "Hardware: T4"
```

**Verificare il modello prima di misurare.** Se Colab assegna L4/A100/H100 invece della
T4, i valori di picco (banda DRAM, SM, occupancy) cambiano e il confronto con lo storico
di `docs/optimization_log.md` non è più diretto: in quel caso `colab stop` e riprovare.

## 3. Portare il sorgente sulla VM

Dal checkout locale, al commit da misurare:

```bash
git archive --format=tar HEAD | gzip > /tmp/theseus.tar.gz
sha256sum /tmp/theseus.tar.gz     # 0a3c747e... per 88ddef2
colab upload -s theseus-gpu /tmp/theseus.tar.gz /content/theseus.tar.gz
```

Sulla VM il tarball va estratto in `/content/theseus`, che diventa la radice di lavoro:

```bash
mkdir -p /content/theseus && tar xzf /content/theseus.tar.gz -C /content/theseus
```

`git archive` porta solo i file tracciati: `theseus_gpu/docs/` è in `.gitignore` da
`88ddef2` e **non** finisce sulla VM. Non serve: la regressione e il profiling non lo
leggono.

## 4. Build

```bash
cd /content/theseus
cmake -S theseus_gpu -B theseus_gpu/build-gpu -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON -DTHESEUS_PROXY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build theseus_gpu/build-gpu -j 4
```

`-DCMAKE_CUDA_ARCHITECTURES=75` è obbligatorio: senza, `nvcc` genera per l'architettura
di default e il caricamento del modulo passa per il JIT.

Per l'attribuzione per riga sorgente serve una **seconda** build, separata, identica
tranne `-lineinfo`:

```bash
cmake -S theseus_gpu -B theseus_gpu/build-gpu-lineinfo -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON -DTHESEUS_PROXY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_FLAGS=-lineinfo
cmake --build theseus_gpu/build-gpu-lineinfo -j 4
```

Le misure di baseline vengono **tutte** da `build-gpu`, cioè dal binario che ha passato
la regressione. `build-gpu-lineinfo` serve solo alla mappa sorgente→traffico ed è a sua
volta ripassato in regressione (`logs/regressione_lineinfo.log`) per garantire che
descriva lo stesso comportamento.

## 5. Regressione

```bash
cd /content/theseus
python3 scripts/run_ggbs_gpu_regression.py --suite simple  --build-dir theseus_gpu/build-gpu
python3 scripts/run_ggbs_gpu_regression.py --suite complex --build-dir theseus_gpu/build-gpu
```

**Lanciarla detached.** `colab exec` va in timeout mentre il lavoro prosegue, quindi:

```bash
setsid nohup /content/regr.sh > /content/regr_outer.log 2>&1 < /dev/null &
# e polling su /content/regr.done
```

## 6. Igiene di misura

```bash
nvidia-smi -pm 1
nvidia-smi -lgc 1590,1590     # su questa VM è consentito e riesce
nvidia-smi -q -d CLOCK        # conferma: Graphics 1590 MHz, SM 1590 MHz
```

Con i clock bloccati la dispersione fra ripetizioni scende molto, ma **tre ripetizioni
per configurazione restano il minimo**: la VM è condivisa e la memoria della T4 non è
regolabile. Riportare sempre mediana e dispersione, mai un numero solo.

## 7. Profiling

Lo script integrale è [`../script/phaseB.sh`](../script/phaseB.sh), archiviato qui accanto. In sintesi,
per ogni dataset × {64,128,256} × 3 ripetizioni:

```bash
ncu --set full --kernel-name theseus_align_batch_kernel --launch-count 1 \
    -f -o raw/<dataset>_<threads>_run<N> \
    theseus_gpu/build-gpu/apps/seq2graph_proxy --backend gpu --require-gpu-result \
    --gpu-threads <threads> -g <grafo> -s <query> -f /tmp/out.gaf

ncu -i raw/<dataset>_<threads>_run<N>.ncu-rep --csv --page details \
    > csv/<dataset>_<threads>_run<N>.csv
```

`--launch-count 1` basta: `align_batch_gpu` lancia `theseus_align_batch_kernel`
**una sola volta** per invocazione (`align_gpu.cu:1537`), con `gridDim = num_seqs` e
`blockDim = threads_per_block`. `--set full` costa 31 passi di replay e circa 15 s sul
dataset più grande.

## 8. Chiudere

```bash
colab stop -s theseus-gpu
```

Obbligatorio: una VM ferma continua a consumare compute unit fino al tetto di 24 h.
