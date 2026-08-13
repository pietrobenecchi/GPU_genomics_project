# Esecuzione su GPU: la sessione che chiude l'handoff

13 agosto 2026, seconda VM Colab con **Tesla T4** (CC 7.5, 15 360 MiB, driver
580.82.07, CUDA 12.8.93), clock SM fissati a 1590 MHz con `nvidia-smi -lgc` e
`ncu --clock-control none`. Dati grezzi in
[`dati_grezzi/esecuzione_gpu/`](dati_grezzi/esecuzione_gpu/), script in
`dati_grezzi/campagna{,_b,_c,_d}.sh`.

⚠️ **VM diversa** da quella di [`REPORT.md`](REPORT.md): i numeri qui dentro
stanno in piedi fra loro, non vanno mescolati con quelli della prima campagna.
Per questo `c1` (`d84fd82`) è stato ricompilato e rimisurato qui: è il termine di
paragone locale.

---

## 1. Che cosa è stato chiuso

| domanda lasciata aperta dall'handoff | esito |
|---|---|
| §3.2 i cinque commit mai eseguiti | **30/30 ciascuno**, byte-identici ai golden |
| §3.3 `compute-sanitizer` | memcheck, racecheck, synccheck puliti; **initcheck no** |
| §3.4 B, costo del page lock | 250 ms, **una volta per processo**, poi 0,0015 ms |
| §3.4 D, riuso degli slot | il clear pigro vale **5,1×** sul kernel del tier simple |
| §3.6 la sonda `-maxrregcount` | superata: 226 → **138 registri senza spill**, e misurata |
| §4.1 riduzione naturale dei registri | fatta: `__noinline__` su `process_vertex` |

E una cosa che l'handoff non prevedeva: initcheck ha trovato che l'argomento di
`f1c93ea` è sbagliato, e il commit `7053193` lo ripara conservandone il guadagno.

## 2. Validazione: i commit mai eseguiti passano tutti

`run_ggbs_gpu_regression.py --suite all` su 10 dataset × 64/128/256 thread, con
`--require-gpu-result` (un fallback CPU che passa non è un pass).

| commit | cosa | ctest | regressione |
|---|---|:-:|:-:|
| `f1c93ea` | niente `cudaMemset` per batch | ok | **30/30** |
| `41bb8b1` | clear pigro (`sp_cleared`) | ok | **30/30** |
| `5dae2ce` | E4, LCP warp-cooperativo | ok | **30/30** |
| `e71a0e1` | E5, query in shared memory | ok | **30/30** |
| `7d2ce8a` | registri con `__noinline__` (nuovo) | ok | **30/30** |
| `7053193` | azzeramento per allocazione (nuovo) | ok | **30/30** |

E4 era «il primo indiziato» perché cambia la struttura di
`extend_and_consume_m_cells`: non diverge, e racecheck non trova hazard.

## 3. initcheck: l'argomento di `f1c93ea` non regge

`f1c93ea` toglie il `cudaMemset` della QueryState per batch e lo motiva così, nel
suo stesso commento: *«That is an argument, not a proof, so it is checked rather
than trusted: compute-sanitizer --tool initcheck …»*. Quel controllo non era mai
stato eseguito. Eseguito, dice di no.

| build | initcheck, `ebola_exact_smoke` ridotto a 8 record |
|---|---:|
| `c1` (`d84fd82`), prima della rimozione | 8 |
| `f1c93ea` | **73 758** |
| `e71a0e1` (tile) | 73 758 |
| `7d2ce8a` (regfix) | 73 758 |
| `7053193` (fix) | **8** |

Gli 8 di `c1` sono di un'altra specie — `Uninitialized access … on access by
cudaMemcpy source`, cioè byte mai scritti dentro un buffer che la D2H ricopia, non
una lettura del kernel — e ci sono da prima della campagna: vedi §7.

I 73 758 invece sono letture del kernel, e **deduplicano su un sito solo**.
Compilando con `-lineinfo`:

```
Uninitialized __global__ memory read of size 4 bytes
  core_check_end        align_core.h:39     if (cell.offset == query_len)
  core_extend_diagonal  align_core.h:281
  align_one             align_gpu.cu:1149   l'estensione del seme a punteggio 0
  theseus_align_batch_kernel
```

cioè `qs.bs_m_jumps_wf[0]`, la cella che l'estensione del seme legge al punteggio
0. Nessuna di quelle letture ha mai cambiato un risultato — tutti e dieci i
dataset restano byte-identici ai golden a 64, 128 e 256 thread — ma un valore
letto da memoria che nessuno ha scritto è quello che ci ha lasciato l'inquilino
precedente di quella DRAM: «ha combaciato» è una proprietà di una esecuzione, non
del programma.

**Il fix (`7053193`) non rimette il memset per batch.** Azzera l'array delle
QueryState **una volta per allocazione**, nello stesso punto e con la stessa
forma del `cudaMemset2D` di `sp_cleared` che il clear pigro aveva già introdotto
lì (e che ora è compreso nel memset intero). Il costo che `f1c93ea` voleva
togliere era 4,4 MB per query **su ogni batch** — 8,6 GB per 2048 query, 40,8 ms
contro un kernel da 4,7 ms; così si paga una volta per processo, e il workspace
si rialloca solo quando un batch chiede più stati del precedente.

Misurato: initcheck torna a 8, memcheck 0, regressione 30/30, e il kernel non
cambia (2,33 ms contro 2,31 di `regfix` su `c4_err_2k` a regime). Sul wall della
CLI si vedono i ~45 ms del memset, perché la CLI fa un batch solo e esce.

### Gli altri strumenti

Su input da 8 record, `-lineinfo`, build `nomemset` / `lazy` / `warp` / `tile` /
`regfix` / `zeroalloc`:

| strumento | dove | esito |
|---|---|---|
| `memcheck` | tutte, `c4_err` 256 thread | **0 errori** |
| `racecheck` | warp, tile, regfix; ebola 64 e `c4_err` 256 | **0 hazard** |
| `synccheck` | warp, tile, regfix | **0 errori** |

## 4. La riduzione dei registri: 226 → 138, senza spill

L'handoff indicava la riduzione naturale del live set come la voce col miglior
rapporto valore/costo, e il primo sospetto era giusto: **l'inlining di una catena
lunga**. `align_one` inlina `compute_new_wave` → `process_vertex` → sei fasi che
sono disgiunte nel tempo (ognuna finisce su un `__syncthreads()`) ma che il
compilatore vede come un corpo unico, così i temporanei di una restano vivi
attraverso le altre.

Un `__noinline__` su `process_vertex` — un solo confine di frame, in cima al
lavoro per vertice — basta:

| variante | registri | stack | spill st/ld | warp/SM | occupancy teorica |
|---|---:|---:|---:|---:|---:|
| inline (`e71a0e1`) | 226 | 736 B | 0 / 0 | 8 | 25 % |
| `-maxrregcount=168` | 168 | 800 B | 72 / 168 B | 12 | 37,5 % |
| **`__noinline__ process_vertex`** | **138** | 768 B | **0 / 0** | 14 | 43,75 % |

Batte la sonda `-maxrregcount` su entrambi i fronti: più occupancy e nessuno
spill. Il taglio va fatto **esattamente lì**: marcare le fasi sotto di essa dà
168–176 registri, e marcarle *insieme* a `process_vertex` dà 168–169 — il
guadagno viene da un confine in cima, non da tanti confini.

Verificato a tre livelli: `ptxas -v` e `cuobjdump -res-usage` in locale (senza
GPU, §5 dell'handoff), e sul dispositivo con `launch__registers_per_thread`.

**Occupancy raggiunta**, Nsight a 128 thread/blocco:

| | `c1` | `tile` | `regfix` |
|---|---:|---:|---:|
| registri (ncu) | 239 | 226 | **138** |
| `sm__warps_active` `c4_err` | 23,93 % | 23,96 % | **35,09 %** |
| `sm__warps_active` `ebola_err_2k` | 24,66 % | 24,66 % | **36,45 %** |
| durata `c4_err` | 1451 µs | 1237 µs | 1355 µs |
| durata `ebola_err_2k` | 3464 µs | 2815 µs | **2380 µs** |

L'occupancy sale del 47 % e la durata segue **sul workload grande** (−15 % contro
`tile`), non su `c4_err`, dove peggiora del 9,5 %. Il perché è nei contatori:
su `c4_err` la latenza per istruzione emessa sale da 24,4 a 36,9 e gli stalli
`long_scoreboard` da 2,41 a 4,54, con 14 MB di letture DRAM in più — il prezzo
della chiamata ABI, che su un batch piccolo non viene ripagato dai warp in più.

## 5. Il regime che conta: con riuso degli slot

Le misure singole della CLI nascondono quasi tutto, perché un processo fa **un
batch** e paga 640 ms di costo fisso. Con `--repeat 5` gli stessi slot servono
cinque batch di fila, ed è il regime per cui il clear pigro e i buffer pinned
sono stati scritti. Kernel in ms, 128 thread, iterazione ≥ 1 (mediana su 4):

| build | `c4_err_2k` | contro `prelazy` | `c4_exact` | contro `prelazy` |
|---|---:|---:|---:|---:|
| `prelazy` (`d9f813f`) | 4,18 | — | 0,89 | — |
| `lazy` (`41bb8b1`) | 3,48 | 1,20× | 0,176 | **5,1×** |
| `tile` (`e71a0e1`) | 2,79 | 1,50× | 0,170 | 5,2× |
| `regfix` (`7d2ce8a`) | 2,31 | **1,81×** | 0,153 | **5,8×** |
| `zeroalloc` (`7053193`) | 2,33 | 1,80× | 0,151 | 5,9× |

Due cose che solo questo regime fa vedere:

1. **il clear pigro serve davvero**: 0,89 → 0,176 ms sul tier simple, cioè il
   57,7 % del kernel che il profiling gli attribuiva, tolto;
2. **`__noinline__` vale il 21 %** in più su `c4_err_2k` (2,79 → 2,31), molto più
   del +9,5 % che la misura a batch singolo suggeriva su `c4_err`.

Alla prima iterazione tutti pagano ~4,7 ms: è il clear completo, che per
definizione avviene una volta per slot.

## 6. Le due domande di §3.4, con i numeri

**B, costo del page lock.** `host_buffers_ms`, `c4_err_2k`:

| iterazione | 0 | 1 | 2 | 3 |
|---|---:|---:|---:|---:|
| `host_buffers` | **250,4 ms** | 0,0015 | 0,0015 | 0,0015 |
| `graph` (upload) | 159,8 ms | 0,0001 | 0,0001 | 0,0001 |

`cudaHostAlloc` costa **tre volte** il proxy `mlock` misurato in locale (83 ms per
576 MiB) ed è interamente **per processo**: spiega i +65/+87 ms di wall visti su
`c4_exact_2k` e `ebola_exact_2k`, che sono batch singoli. Su `c4_exact` sono
60,3 ms, proporzionali alla taglia del batch. Dalla seconda iterazione il costo
non esiste: **B è chiuso e va tenuto.**

**D, riuso degli slot.** È la tabella di §5: `prelazy` resta a 0,89 ms per ogni
batch, `lazy` scende a 0,176 dal secondo. Il clear pigro **è corretto e utile**;
resta vero che sull'uso attuale della CLI non si vede, perché ogni QueryState
serve una query per processo.

## 7. Quello che resta aperto

- **Gli 8 errori `cudaMemcpy source`**, presenti anche in `c1` e quindi
  precedenti alla campagna: byte mai scritti in un buffer device che la D2H
  ricopia (indiziati: il padding di `AlignResult` o `TracebackMeta`). Innocui
  finché nessuno legge quei byte a valle, ma sono l'unica voce di initcheck
  ancora accesa e non sono stati inseguiti.
- **Il chunking del batch** (§4.4 dell'handoff): resta il moltiplicatore vero,
  perché è ciò che rende il regime di §5 quello normale invece che una misura con
  `--repeat`. Oggi `align_batch` fa un launch solo, ~3 370 query su una T4.
- **TASK F, parallelizzare i jump**: non toccato. Dopo `regfix` la mappa non
  cambia — l'estensione del seme a punteggio 0 gira ancora su thread 0 da solo,
  e su `c4_err` gli stalli di barriera sono il 16,7 % delle emissioni.
- **Categoria 2, thread coarsening**: ancora non implementata. Nota che E4 (il
  de-coarsening) è risultato un **guadagno** a 256 thread, contro il suo commit
  padre `lazy`: `c4_err_2k` da 9,07 a 6,19 ms, `ebola_err_2k` da 8,42 a 5,73 —
  cioè il contrario di quanto §7
  dell'handoff si aspettava: era dato in perdita o neutro. Il segno dice che su
  questo kernel conviene *dividere* il lavoro fra più thread, non accorparlo.
- **`kMaxIJumpStack`** resta da riderivare da picchi reali.
