# Fase A — Ripristino della validità

**Esito: verde su tutte le combinazioni. Nessuna modifica al codice è stata necessaria.**

Commit validato: **`88ddef2225d5d72cdee194b9e4abcc24606f7ef6`** (`main`, "Untrack docs directory",
albero pulito, `git status` senza modifiche).

---

## 1. Cosa era da validare, e perché

`821eee3` era l'ultimo commit validato su GPU. Fra `821eee3` e `88ddef2` c'è **una sola
modifica al codice compilato**:

```
$ git diff --stat 821eee3 88ddef2
 .gitignore                                         |  10 +-
 theseus_gpu/README.md                              |   6 +-
 theseus_gpu/data/sample_output.gaf                 |   5 -
 theseus_gpu/docs/ggbs_validation_dataset_report.md | 410 -----
 theseus_gpu/docs/gpu_parallelism_analysis.md       | 383 -----
 theseus_gpu/docs/optimization_log.md               | 805 -----
 theseus_gpu/src/gpu/align_gpu.cu                   | 203 +++---
 7 files changed, 131 insertions(+), 1691 deletions(-)
```

Le 203 righe di `align_gpu.cu` sono il commit `7bb6479` ("Convenzione indici thread"),
mai passato per `nvcc`. Ispezionato prima di compilare: è un refactor puramente meccanico
che sostituisce ogni occorrenza di `threadIdx.x` / `blockDim.x` con due variabili locali
`tx` / `ntx` di tipo `int32_t` inizializzate una volta a inizio funzione. Nessun cambio
di semantica, nessun cambio di layout dati, nessun cambio di ordine di barriere.
`88ddef2` e `d25e13a` non toccano codice compilato (`.gitignore`, README, un `.gaf` di
esempio e la rimozione dal tracking di `docs/`).

## 2. Procedura seguita

Da `theseus_gpu/docs/handoff_parallelizzazione_kernel.md`, sezione 7 "Come si valida
(procedura, non teoria)", righe 286–305, **alla lettera**:

```bash
conda activate colab-cli
colab new -s theseus-gpu --gpu T4
# sulla VM:
cmake -S theseus_gpu -B theseus_gpu/build-gpu -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON -DTHESEUS_PROXY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build theseus_gpu/build-gpu -j 4

python3 scripts/run_ggbs_gpu_regression.py --suite simple  --build-dir theseus_gpu/build-gpu
python3 scripts/run_ggbs_gpu_regression.py --suite complex --build-dir theseus_gpu/build-gpu
```

Il sorgente è stato trasferito sulla VM come `git archive HEAD` (tarball di 1.7 MB,
sha256 `0a3c747eb390891644d6d4057344a4e5d84084d4f1b2e7a916d2ba15190bd29a`), quindi il
contenuto della VM è esattamente l'albero tracciato di `88ddef2`.

Come da "Cose imparate a caro prezzo in questa sessione" nello stesso handoff, la
regressione è stata lanciata **detached** (`setsid nohup ... &`) su file di log con
polling su un `.done`, per non farla morire nel timeout di `colab exec`.

## 3. Dataset trovati

Enumerati da `scripts/run_ggbs_gpu_regression.py` (dizionario `DATASETS`, righe 27–56)
e verificati come file presenti in `theseus_gpu/data/validation/ggbs/`:

| dataset | tier | grafo | query | golden |
|---|---|---|---|---|
| `ebola_exact_smoke` | simple | `graphs/ebola.gfa` | `queries/ebola_exact_smoke.queries` (256) | `golden/ebola_exact_smoke.cpu.gaf` |
| `c4_exact` | simple | `graphs/c4.gfa` | `queries/c4_exact.queries` (512) | `golden/c4_exact.cpu.gaf` |
| `ebola_error_smoke` | complex | `graphs/ebola.gfa` | `queries/ebola_error_smoke.queries` (256) | `golden/ebola_error_smoke.cpu.gaf` |
| `c4_err` | complex | `graphs/c4.gfa` | `queries/c4_err.queries` (512) | `golden/c4_err.cpu.gaf` |

Thread per blocco: `THREAD_COUNTS = [64, 128, 256]` (riga 18 dello stesso script).
Sono quattro dataset, non sei: `covid`, `yeast`, `MHC`, `ecoli` di GGBS **non hanno un
golden congelato in questo albero** e quindi non sono parte della regressione. Nessun
dataset è stato aggiunto o tolto rispetto alla procedura.

## 4. Compilazione

Log integrale: [`logs/build.log`](dati_grezzi/logs/build.log).

```
-- The CUDA compiler identification is NVIDIA 12.8.93 with host compiler GNU 11.4.0
-- theseus_proxy: CUDA backend ON, architectures: 75
-- Configuring done (3.6s)
...
[ 56%] Building CUDA object CMakeFiles/theseus_proxy.dir/src/gpu/align_gpu.cu.o
[100%] Built target seq2graph_gpu_validate
CONFIG_RC=0 BUILD_RC=0
```

**Zero errori e zero warning**, sia dalla configurazione sia dalla build: lo `stderr` di
entrambi i comandi è vuoto (visibile nel log come blocco `--- stderr ---` privo di
contenuto). In particolare `align_gpu.cu` compila pulito con `nvcc` 12.8 per `sm_75`.

## 5. Matrice dataset × thread per blocco

Log: [`logs/regressione_simple.log`](dati_grezzi/logs/regressione_simple.log),
[`logs/regressione_complex.log`](dati_grezzi/logs/regressione_complex.log).

| dataset | tier | 64 | 128 | 256 |
|---|---|---|---|---|
| `ebola_exact_smoke` | simple | PASS (256 query) | PASS (256) | PASS (256) |
| `c4_exact` | simple | PASS (512 query) | PASS (512) | PASS (512) |
| `ebola_error_smoke` | complex | PASS (256 query) | PASS (256) | PASS (256) |
| `c4_err` | complex | PASS (512 query) | PASS (512) | PASS (512) |

`EXIT_SIMPLE=0`, `EXIT_COMPLEX=0`. **12 combinazioni su 12.**

### Cosa verifica esattamente un PASS

Non è solo un confronto di file. Per ogni run lo script (`run_ggbs_gpu_regression.py`):

1. lancia il binario con `--require-gpu-result` (riga 203), che impedisce al backend di
   ripiegare silenziosamente sulla CPU e far passare la regressione con l'output del
   fallback;
2. richiede `--require-device` (default `True`, riga 383) ed esegue prima
   `seq2graph_gpu_validate --require-device`;
3. controlla lo `stderr` del run (`validate_gpu_stderr`, righe 210–223): devono comparire
   *"aligned on device"*, *"align kernel result verified against CPU"* e *"GAF
   reconstructed from GPU QueryState"*, e non devono comparire *"fallback"* o
   *"not implemented"*;
4. confronta **campo per campo** ogni riga GAF con il golden (`compare_gaf`, righe
   107–171): conteggio righe, righe duplicate, righe mancanti o in eccesso, e poi per
   ogni query tutti i 12 campi — `query_length`, `query_start`, `query_end`, `strand`,
   `path`, `target_length`, `terminal_start`, `terminal_offset`,
   `matches_or_score_proxy`, `block_length`, `mapq`, `cigar`.

### Verifica aggiuntiva: identità byte per byte

Il confronto campo per campo è per nome di query e non impone l'identità del file. Ho
aggiunto quindi un confronto SHA-256 fra ogni GAF prodotto e il rispettivo golden
([`logs/byte_diff.txt`](dati_grezzi/logs/byte_diff.txt)):

| dataset | sha256 (primi 16) | 64 | 128 | 256 |
|---|---|---|---|---|
| `ebola_exact_smoke` | `0301bb6308be749c` | IDENTICAL | IDENTICAL | IDENTICAL |
| `c4_exact` | `88166d3ed9242460` | IDENTICAL | IDENTICAL | IDENTICAL |
| `ebola_error_smoke` | `dd6af6a27147aad6` | IDENTICAL | IDENTICAL | IDENTICAL |
| `c4_err` | `7eb1cf240dcc6809` | IDENTICAL | IDENTICAL | IDENTICAL |

I dodici file di output hanno lo stesso digest del golden corrispondente: identità
byte per byte, non solo equivalenza campo per campo.

## 6. Modifiche al codice

**Nessuna.** L'albero di `88ddef2` compila e passa la regressione così com'è. Non è stata
toccata una riga, né di codice né di build.

## 7. Ambiente

Dettagli completi in [`env/nvidia-smi.txt`](dati_grezzi/env/nvidia-smi.txt),
[`env/versioni.txt`](dati_grezzi/env/versioni.txt) e [`env/setup_colab.md`](dati_grezzi/env/setup_colab.md).
In breve: **Tesla T4** (CC 7.5), driver 580.82.07, CUDA toolkit 12.8.93,
Nsight Compute 2025.1.1.0, Ubuntu 22.04, gcc 11.4.0, CMake 3.31.10.
L'architettura assegnata **è** una T4, quindi il confronto con le misure storiche resta
diretto.
