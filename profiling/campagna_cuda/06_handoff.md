# Handoff della campagna CUDA

Stato al 13 agosto 2026, branch `profili-nsight`, HEAD `715d419`.

Questo documento è scritto per chi riprende il lavoro **quando ci sarà di nuovo
una GPU**. Le risorse Colab sono esaurite: solo la T4 è abilitata sull'account e
il backend ha smesso di assegnarla (`Service Unavailable`). Tutto ciò che non
richiedeva un device è stato fatto; ciò che richiede di *eseguire* no.

---

## 1. Riassunto in una riga

Il kernel non è bandwidth-bound ma **issue-bound**, e con 226–239 registri ha il
25–28 % di occupancy. Una sola ottimizzazione ha spostato il tempo — lo split
hot/cold della ScratchPad, fino a **7,86×** sul kernel — ma il kernel è lo
**0,6 % del wall**, quindi sul tempo di processo non si vede.

---

## 2. I commit, e come stanno messi

`00ea1ef` è il punto di partenza (baseline di codice; `027c19b` e `2737a2d`
aggiungono solo dataset e note, il codice è identico).

| commit | cosa | compila `nvcc` | CPU golden | GPU 30/30 | misurato |
|---|---|:-:|:-:|:-:|:-:|
| `027c19b` | workload MEDIUM/LARGE + suite per taglia | — | 10/10 | ✅ | ✅ |
| `d84fd82` | **C1** — `sp_off` denso, split hot/cold | ✅ | 10/10 | ✅ | ✅ |
| `026b457` | **B** — buffer host pinned persistenti | ✅ | 7/7 | ✅ | parziale |
| `f1c93ea` | niente `cudaMemset` per batch | ✅ | 10/10 | ❌ | ❌ |
| `d9f813f` | flag `--repeat` (solo misura) | ✅ | ok | ❌ | ❌ |
| `41bb8b1` | **D** — clear pigro (`sp_cleared`) | ✅ | 10/10 | ❌ | ❌ |
| `5dae2ce` | **E4** — LCP warp-cooperativo | ✅ | 5/5 ctest | ❌ | ❌ |
| `e71a0e1` | **E5** — query in shared memory | ✅ | 5/5 ctest | ❌ | ❌ |

Gli altri (`3d09815`, `c6f2955`, `8aa0429`, `80accca`, `e15dff0`, `715d419`) sono
solo documentazione e dati grezzi.

⚠️ **`41bb8b1` è un `--amend` di quello che era `0a303ce`.** Conteneva un errore
di compilazione (`grown_chars` inesistente) propagato ai due commit successivi;
è stato corretto alla radice e la catena riscritta, così ogni commit compila e
resta bisectabile. Il ramo pre-riscrittura è conservato in `backup-precompile`.

---

## 3. Cosa fare appena c'è una GPU

In quest'ordine. Tutti gli script sono in `dati_grezzi/`, già scritti e usati.

### 3.1 Rimettere in piedi l'ambiente

```bash
conda activate colab-cli
colab new -s theseus-gpu --gpu T4
colab status -s theseus-gpu          # deve dire T4: non confrontare misure fra GPU diverse

git bundle create /tmp/theseus.bundle --all
colab upload -s theseus-gpu /tmp/theseus.bundle /content/theseus.bundle
# sulla VM: git clone /content/theseus.bundle /content/theseus
nvidia-smi -pm 1 && nvidia-smi -lgc 1590,1590     # su Colab riesce
```

Poi caricare `build.sh`, `regr.sh`, `timing.py`, `ncu.sh`, `phase_patch.py`,
`build_phase.sh`, `phases.sh` da `dati_grezzi/` in `/content/`.

**Lanciare sempre detached** (`setsid nohup ... &` + file di done): `colab exec`
va in timeout mentre il lavoro prosegue, e la sessione è caduta due volte durante
questa campagna. I risultati devono atterrare su file sulla VM, non sullo stdout.

### 3.2 Validare i cinque commit mai eseguiti — **priorità assoluta**

```bash
/content/build.sh nomemset:f1c93ea lazy:41bb8b1 warp:5dae2ce tile:e71a0e1
/content/regr.sh nomemset lazy warp tile
```

Ognuno deve dare **30/30** (10 dataset × 64/128/256), byte-identico ai golden,
con `--require-gpu-result`. Un fallback CPU che passa non è un pass.

Attenzione a **E4** (`5dae2ce`): è l'unico che cambia la struttura di
`extend_and_consume_m_cells` (un warp per cella invece di un thread per cella).
Se qualcosa diverge, è il primo indiziato. La logica del ballot è però già
verificata contro `core_lcp` su 400 000 casi casuali
(`dati_grezzi/lcp_test.cpp`, 0 divergenze), quindi un'eventuale divergenza sarà
nell'ordine delle celle o nella sincronizzazione, non nell'aritmetica.

### 3.3 `compute-sanitizer`, due volte e per due motivi diversi

```bash
compute-sanitizer --tool initcheck  <app> ...   # su f1c93ea e su 41bb8b1
compute-sanitizer --tool memcheck   <app> ...
compute-sanitizer --tool racecheck  <app> ...   # su 5dae2ce, che cambia i barrier
```

- su `f1c93ea` **initcheck è la prova che serve**: il memset per batch è stato
  tolto sulla base di un argomento (ogni campo letto è inizializzato o protetto
  da un contatore che lo è), non di una verifica;
- su `41bb8b1` serve perché il clear pigro lascia `sp_off` non azzerato oltre il
  prefisso già visto.

### 3.4 Chiudere le due domande aperte con `--repeat`

```bash
<app> --backend gpu --require-gpu-result --repeat 5 -g <grafo> -s <query> -f /tmp/o.gaf
```

Stampa una riga di timing per iterazione. Serve a separare il costo **per
processo** da quello **per batch**:

1. **B, costo del page lock.** `host_buffers_ms` alla prima iterazione contro le
   successive. Proxy locale già misurato: `mlock` di 576 MiB costa 83 ms
   (`dati_grezzi/pinbench.c`). Se `cudaHostAlloc` costa molto di più, spiega i
   +65/+87 ms di wall visti su `c4_exact_2k` e `ebola_exact_2k`.
2. **D, riuso degli slot.** `prelazy` (`d9f813f`) contro `lazy` (`41bb8b1`): dalla
   seconda iterazione il clear deve sparire, e sul tier simple vale il 57,7 % del
   kernel. È l'unico modo per far vedere che il clear pigro serve a qualcosa.

### 3.5 Misure di prestazione

```bash
python3 /content/timing.py c1 nomemset lazy warp tile     # matrice completa
/content/ncu.sh c1 warp tile                              # contatori
/content/build_phase.sh c1:d84fd82 tile:e71a0e1 && /content/phases.sh
```

`timing.py` fa warm-up + 3 run e riporta mediana, min e max. Il **wall è rumoroso
(±130 ms)**, il tempo di kernel no (1–3 %): non trarre conclusioni sul wall da
run singole.

### 3.6 La sonda `-maxrregcount`

I registri e gli spill sono **già misurati in locale** (§5). Manca solo l'effetto
sul tempo:

```bash
cmake -S theseus_gpu -B build-r168 ... -DCMAKE_CUDA_FLAGS="-maxrregcount=168"
```

168 è il candidato: +50 % di occupancy (8 → 12 warp) per 72 byte di spill store.
Va trattato come **sonda**, non come soluzione: la strada giusta resta ridurre il
live set.

---

## 4. Lavoro ancora da implementare

### 4.1 Riduzione naturale dei registri — **fattibile subito, senza GPU**

È la voce con il rapporto valore/costo migliore, ed è la strada che il brief
indica come preferibile a `maxrregcount`. Il ciclo modifica → ricompila → leggi i
registri dura pochi secondi in locale (§5). Sospetti, in ordine:

1. **inlining di una catena lunga**: `align_one` inlina `compute_new_wave` →
   `process_vertex` → sei fasi disgiunte nel tempo ma unite per il compilatore,
   quindi i temporanei di una restano vivi attraverso l'altra. Da provare:
   `__noinline__` su `process_vertex` o su `densify`;
2. **`Cell` costa 6 registri** (24 byte) e ce ne sono molte vive insieme;
3. **`Frame stack[kMaxIJumpStack]`** in `core_store_i_jump`, array locale di 448 B;
4. **live range lunghi** attraverso le barriere (`range_start/end`, i puntatori
   shared, `score`, `v`, `query_len`).

Soglie: ≤ 168 registri → 12 warp/SM, ≤ 128 → 16 warp/SM.

### 4.2 Categoria 2, thread coarsening — **non implementata**

L'unico punto del TASK E scoperto. Nota che la campagna ha prodotto la modifica
*opposta* e la misurerà (E4 è de-coarsening: da un thread per cella a un warp per
cella), quindi il segno del risultato di E4 dice già molto. Candidati: più
diagonali per thread nel clear, più celle per thread in `densify`. Da valutare
insieme ai registri, perché il coarsening li aumenta.

### 4.3 TASK F, parallelizzare i jump — **il pezzo grosso che resta**

Il profiling dopo C1 lo promuove a priorità. Sul tier simple il **71–74 % del
loop** è il «resto», dominato dall'estensione del seme a punteggio 0 che
`align_gpu.cu:1138` esegue **su thread 0 da solo**: per una read che allinea
esatta, quella è tutta l'alleanza.

Serializzato su thread 0 c'è anche: `store_M_jump`, `store_I_jump`, il fan-out
sugli archi, l'attivazione dei vertici, l'append dei metadati, e il ciclo seriale
in fondo a `extend_and_consume_m_cells`.

L'ostacolo strutturale: `core_extend_diagonal` è `THESEUS_HD`, **ricorsiva**
attraverso `core_store_m_jump` e condivisa con la CPU. Renderla cooperativa vuol
dire una variante device-only dell'intera catena. La strategia dovrà essere
deterministica — generazione parallela dei candidati, allocazione per prefix-sum,
commit in ordine — esattamente come già fa `merge_candidate_tile`, che è il
modello da copiare (vedi il suo commento a `align_gpu.cu:257`, che spiega perché
l'ordine di `sp_diags` è osservabile).

### 4.4 Cose emerse e non inseguite

- **Il tetto del batch.** `align_batch` fa **un solo launch senza chunking**:
  `sizeof(QueryState)` × query, quindi ~3 370 query su una T4. Le 10 000 read
  disponibili per dataset ne richiederebbero 42 GB. Il chunking sbloccherebbe i
  workload veri **e** renderebbe utili sia il clear pigro sia i buffer pinned,
  che oggi non si ammortizzano perché la CLI fa un batch e esce.
- **Il costo fisso per processo.** Su `c4_err_2k`: wall 735 ms, di cui kernel
  4,7, D2H 46,7, H2D 40,8 e **~640 ms di resto** — inizializzazione del contesto
  CUDA, upload del grafo, parsing, GAF. In un uso reale si ammortizza; nella CLI
  no.
- **Covid come controcampo.** 39 253 vertici, 95 440 archi, vertice più lungo 32:
  span 83 contro 52 107. Rovescia il profilo — clear trascurabile, dominante il
  clear di `vd_vertex_to_idx` e il fan-out. Richiede `kMaxVertices ≥ 40 960`,
  cioè una build separata (+156 KB per QueryState). Dettagli in `00_workload.md`
  §1 e §7.
- **`kMaxIJumpStack`** resta da riderivare da picchi reali (oggi 8, misurato 1).
- **`sp_at` guarda `idx < kScratchpadSpan`, non `idx < span`.** Fuori dalla
  finestra leggerebbe memoria non azzerata invece di `-1`. Non raggiungibile per
  costruzione (`diag ∈ [-query_len, max_diag]`), ma la guardia è più larga
  dell'invariante: valeva anche prima del clear pigro, che però la rende più
  visibile.

---

## 5. Ambiente: cosa si può fare **senza** GPU

Scoperta di questa sessione, e vale la pena non ridimenticarla: **compilare CUDA
non richiede una GPU né il driver**, solo il toolkit.

```bash
conda create -y -p ~/.local/opt/cudac -c nvidia cuda-nvcc=12.8 cuda-cudart-dev=12.8 cuda-cccl=12.8
export PATH=$HOME/.local/opt/cudac/bin:$PATH
cd theseus_gpu && nvcc -std=c++17 -arch=sm_75 -O3 -I src -I include -Xptxas -v \
    -c src/gpu/align_gpu.cu -o /dev/null
```

È la stessa versione della VM (12.8.93). Dà: che `align_gpu.cu` compili — cosa
che il build host **non** verifica, perché usa lo stub, ed è così che un errore è
sopravvissuto a tre commit — più registri, spill, stack frame, shared statica, e
quindi l'occupancy teorica (`warp/SM = 65536 / (registri × 32)` su T4).

Nsight Compute e Systems sono in `~/.local/opt/` e **leggono i report raccolti
altrove senza GPU**: `ncu --import x.ncu-rep --page details`, `nsys stats x.nsys-rep`.

Il path CPU condivide `query_state.h`, quindi valida C1 e D sul serio: con
`c4_err_2k` una sola `QueryState` serve 2048 query consecutive e solo la prima
azzera qualcosa. Tutti e dieci i golden passano.

```bash
for ds in ebola_exact_smoke ebola_error_smoke c4_exact c4_err c4_exact_1k \
          c4_err_1k c4_exact_2k c4_err_2k ebola_exact_2k ebola_err_2k; do
  ...  ./build-host/apps/seq2graph_proxy --backend cpu ... && cmp con il golden
done
```

---

## 6. I fatti da non riderivare

1. **`durata ≈ istruzioni × latenza per istruzione`.** Verificato su due
   workload: `c4_err_2k` 0,33 × 0,49 = 0,16 contro 0,18 misurato; `ebola_err_2k`
   0,71 × 1,13 = 0,80 contro 0,82. Il kernel **non è bandwidth-bound**: è per
   questo che il clear coalescente di `00ea1ef` non aveva cambiato nulla
   (traffico dimezzato, istruzioni invariate) e C1 sì (cinque store su sei tolti).
2. **L'invariante della ScratchPad.** L'attività dipende solo da
   `sp_off[idx] == -1`, e la prova è nel codice: `sp_reset_one` azzera **solo**
   l'offset da sempre, quindi il payload stantio delle celle inattive è già la
   norma nell'implementazione validata. In più la ScratchPad **si ripulisce da
   sola**: ogni scrittura va a una diagonale che sta in `sp_diags`, e
   `process_vertex` finisce con un reset che la percorre.
3. **Il limite di occupancy sono i registri, sempre.** 65 536 / (239 × 32) = 8
   warp su 32 = 25 %, identico a 64, 128 e 256 thread/blocco: cambia quanti
   blocchi ci stanno, non quanti warp.
4. **Il clear pigro è corretto ma inutile sull'uso attuale**, perché ogni
   `QueryState` serve una query per processo: «azzerare una volta per slot» e
   «una volta per query» sono la stessa frase. Serve il riuso degli slot.
5. **I dataset nuovi non sono sintetici**: stessi record GGBS, `--limit` più alto.
   Ogni set piccolo è un prefisso byte-esatto del suo omologo grande, e lo stesso
   vale per i golden — l'oracolo, rieseguito, ha riprodotto i golden congelati sul
   prefisso condiviso.
6. **`ncu` di default blocca i clock al base (585 MHz)** e sovrascrive
   `nvidia-smi -lgc`. Per decidere ottimizzazioni serve `--clock-control none`
   con i clock fissati a mano a 1590.

---

## 7. Rischi noti

- **Cinque commit non sono mai stati eseguiti.** Compilano, e il path CPU copre
  `query_state.h`, ma `merge_candidate_tile`, il clear e
  `extend_and_consume_m_cells` sono codice device che nessun test locale tocca.
- **E4 e E5 potrebbero peggiorare.** Sono attesi in perdita o neutri sui workload
  correnti (§4 e §5 di `05_categorie_cuda.md`), ed è un risultato valido. Se
  peggiorano vanno revertiti dal backend finale **conservando commit, profilo e
  motivazione**, come chiede il brief.
- **B ha una misura aperta** e due dataset su dieci con il wall peggiorato. Non
  va dichiarato chiuso prima di `host_buffers_ms`.
- **Le misure di prestazione vengono da una sola VM.** Non mescolarle con quelle
  di una VM nuova senza dirlo.
