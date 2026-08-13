# Esecuzione su GPU: la sessione che chiude l'handoff

13 agosto 2026, seconda VM Colab con **Tesla T4** (CC 7.5, 15 360 MiB, driver
580.82.07, CUDA 12.8.93), clock SM fissati a 1590 MHz con `nvidia-smi -lgc` e
`ncu --clock-control none`. Dati grezzi in
[`dati_grezzi/esecuzione_gpu/`](dati_grezzi/esecuzione_gpu/), script in
`dati_grezzi/campagna{,_b,_c,_d,_e}.sh`. §8 viene da una terza sessione, il 13
agosto in serata, su una T4 di nuovo diversa: i confronti di §8 sono interni a
quella sessione e hanno il proprio termine di paragone ricompilato lì.

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

---

## 8. Categoria 2, thread coarsening: dove si applica e dove no

Era l'unico punto del TASK E senza un esperimento, e il motivo per cui era stato
rimandato — «239 registri sono già al limite, il coarsening li aumenta» — è
caduto con §4: a 138 registri ci sono 30 registri di margine prima di scendere
sotto i 12 warp/SM. Due bersagli, due esiti opposti.

### 8.1 Il clear: quattro parole per thread (`02127db`)

Dopo lo split hot/cold il clear resta la fase più grossa del tier simple — non il
7,7 % di `c4_err_2k`, ma il **57-69 % dei cicli** di una query esatta:

| dataset | clear | loop | clear / totale |
|---|---:|---:|---:|
| `c4_exact` | 62,7 M | 27,6 M | **69,4 %** |
| `c4_exact_2k` | 194,8 M | 142,9 M | **57,7 %** |
| `ebola_exact_2k` | 13,8 M | 64,1 M | 17,7 % |
| `c4_err_2k` | 35,8 M | 426,5 M | 7,7 % |

È una parola per ciascuna delle 52 224 diagonali della finestra, tutte contigue e
tutte con lo stesso valore: il caso da manuale per dare a ogni thread quattro
elementi e scriverli con uno store da 128 bit. `fill_words` fa questo, con un
prologo e un epilogo scalari perché l'allineamento **non si può assumere**:
`sp_off` sta su un confine di 16 byte dentro `QueryState`, ma
`sizeof(QueryState)` è 8 modulo 16, quindi uno stato su due parte sfasato e uno
store `int4` lì dentro romperebbe. Il conto degli indici è verificato in locale
su **95 468 combinazioni** di allineamento, finestra e thread per blocco —
copertura esatta, nessuno sconfinamento — in `dati_grezzi/fill_words_test.cpp`.

Nsight, 128 thread, contro il commit padre:

| | `c4_exact` | `c4_err` | `ebola_err_2k` |
|---|---:|---:|---:|
| durata | 1 028 → **813 µs** (1,27×) | 1 359 → 1 283 µs (1,06×) | 2 379 → 2 374 µs (1,00×) |
| istruzioni | 6,13 M → **3,80 M** (−38 %) | 20,9 M → 18,6 M (−11 %) | 67,4 M → 66,3 M (−2 %) |
| richieste di store | 960 k → **335 k** (−65 %) | 1 110 k → 485 k | 1 646 k → 1 205 k |
| byte scritti in DRAM | 140,8 → 135,2 MB | invariati | invariati |

Di nuovo la firma di C1: **i byte non cambiano, le istruzioni sì, e la durata
segue le istruzioni**. La quota di stalli da barriera *sale* (40,9 → 56,5 %) e non
è un peggioramento: è la fase senza barriere che si è accorciata.

**Il limite, detto chiaro.** Il clear coarsened vale sulla *prima* query di ogni
slot, che è l'unica in cui il clear gira. Con `--repeat` su `c4_exact_2k`:
iterazione 0 4,41 → **3,58 ms** (1,23×), iterazione ≥ 1 **0,391 → 0,392 ms**,
identiche — perché lì il clear pigro lo ha già tolto del tutto. Le due
ottimizzazioni non si sommano, si coprono: una rende la fase più veloce, l'altra
la fa sparire.

### 8.2 `densify`: il coarsening non si applica, e il dato lo dice

L'altro candidato erano le fasi `densify`, il **53 % del loop** sul tier complex.
Prima di scriverlo ho misurato quanto è grande il tile: `sp_ndiags` esce dal path
CPU, che usa la stessa `QueryState` e produce lo stesso output byte per byte, con
un istogramma sui quattro dataset.

| dataset | chiamate | media | max | con `ndiags == 0` |
|---|---:|---:|---:|---:|
| `c4_err` | 10 707 | 2,77 | 37 | 45,5 % |
| `c4_err_2k` | 41 400 | 2,57 | 41 | 46,9 % |
| `ebola_err_2k` | 41 568 | 2,49 | 41 | 47,4 % |
| `c4_exact` | 1 551 | 0,00 | 0 | 100 % |

**Il wavefront è 2,5 diagonali, con un massimo di 41.** Il loop di `densify` fa
quindi *un solo tile* anche a 64 thread per blocco: non c'è nessun round da
accorpare, e dare quattro diagonali a testa lascerebbe soltanto 11 thread attivi
su 41 invece di 41. Misurato prima di scriverlo, non dopo: il tentativo con le
celle in registri costa anche 48 registri (138 → 186 già a una diagonale per
thread), quindi avrebbe pagato occupancy per non guadagnare niente.

Il 53 % del loop non è dunque lavoro di densificazione: è il **costo fisso** di
densificare 2,5 celle — quattro `__syncthreads()` per tile, tre volte per vertice
(I, D, M). Questo è il vero bersaglio, e non è coarsening: è la direzione
opposta.

### 8.3 Il corollario gratis: `densify` vuota (`23dc718`)

L'istogramma dice anche che **il 46-47 % delle chiamate ha `ndiags == 0`**, e che
in quel caso la funzione esegue due barriere e una scrittura in shared per
produrre un Range identico a quello che aveva in mano entrando. Il ritorno
anticipato è la stessa cosa senza le barriere; l'unica scrittura saltata è
l'aggiornamento del picco, che non può muoversi quando nessuno spinge
(`range_start <= sc_peak_wf` per costruzione, e il codice seriale aggiorna il
picco solo quando spinge davvero).

Vale poco da solo — 1 % su `ebola_err_2k`, 0,2 % su `c4_exact` — ma è misurato e
non costa niente.

### 8.4 Esito

| | 30/30 | memcheck | initcheck | racecheck | tier simple | tier complex |
|---|:-:|:-:|:-:|:-:|---:|---:|
| `02127db` clear coarsened | ✅ | 0 | 8 (i preesistenti) | 0 hazard | **1,20-1,28×** | 1,00-1,08× |
| `23dc718` densify vuota | ✅ | 0 | 8 | 0 hazard | +0,2-4 % | +1 % |

Cumulativo sul kernel a batch singolo: `c4_exact_2k` 4,58 → **3,58 ms**,
`ebola_exact_2k` 1,55 → **1,29 ms**, `c4_exact_1k` 2,49 → **2,07 ms**; il tier
complex resta dov'era, come previsto, perché lì il clear è l'1,5-7,7 % dei cicli.

Una nota di onestà sulle misure: la matrice `timing.py` dava `ebola_err_2k` a 128
thread in peggioramento del 6 % con `23dc718`, ma Nsight sullo stesso punto dà
2 379 → 2 336 µs, cioè un miglioramento dell'1,8 %. Il wall e il kernel misurati
con `timing.py` su tre run hanno una dispersione che a questa scala li rende
inutilizzabili: **dove i due si contraddicono vale il contatore, non il
cronometro.**

**Con questo tutte e sei le categorie hanno un esperimento**, e la 2 ne ha due:
uno che si applica e uno che il dato ha respinto prima di scriverlo.

---

## 9. Fuori dalle sei categorie: la D2H del traceback (`e9435fe`)

Chiuse le sei categorie, il kernel su `c4_err_2k` a regime è 2,3 ms e la D2H
dello stesso batch è 46,8 ms: **il 95 % del tempo GPU per batch**. Prima di
toccarla ho misurato quanto di quel trasferimento serve, strumentando il path
CPU — stessa `QueryState`, stesse wavefront, output byte-identico:

| dataset | celle M (media / max) | M jumps | I jumps | usato dei 3 × 4096 copiati |
|---|---|---:|---:|---:|
| `c4_err_2k` | 24,3 / 438 | 1,0 | 0,0 | **0,21 %** |
| `ebola_err_2k` | 23,7 / 438 | 1,1 | 0,0 | **0,20 %** |
| `c4_exact_2k` | 0,0 / 0 | 1,0 | 0,0 | **0,01 %** |

La copia portava **288 KB per query** a taglia fissa — 576 MB per un batch da
2048 — per consegnare in media **607 byte**. Il 99,8 % era slack mai scritto.

**La modifica.** `CompactTracebackState` non contiene più i tre array da 4096
celle ma tre puntatori. `align_batch` calcola la somma prefissa esclusiva delle
tre dimensioni che ha appena letto dai metadati, `pack_traceback_kernel` (un
blocco per query) raduna i prefissi usati in un buffer denso a quella base, e una
sola D2H porta a casa esattamente quei byte; i puntatori indicano dentro il
buffer di staging, che vive sul workspace ed è page-locked una volta per
processo. Host e device derivano il layout dagli stessi numeri, quindi non serve
nessuna scansione sul device. L'aritmetica è verificata a parte su 1501 batch
simulati, comprese le wavefront vuote e il caso in cui l'intero batch è vuoto
(`dati_grezzi/pack_traceback_test.cpp`).

**Misurato**, 128 thread, `--repeat`, contro il commit padre:

| | `c4_err_2k` | `c4_exact_2k` | `ebola_err_2k` |
|---|---:|---:|---:|
| D2H | 46,81 → **0,30 ms** (154×) | 46,67 → **0,19 ms** (244×) | 46,68 → **0,29 ms** (161×) |
| tempo GPU per batch | 51,42 → **4,66 ms** (11,0×) | 50,61 → **3,86 ms** (13,1×) | 49,77 → **3,26 ms** (15,3×) |
| `host_buffers` (per processo) | 235 → **1,0 ms** | 332 → **0,94 ms** | 333 → **0,96 ms** |

Il secondo effetto non era previsto ma segue dallo stesso fatto: i 576 MiB
page-locked erano quasi tutti `_host_states`, quindi il `cudaHostAlloc` da 250 ms
di §6 si riduce a un millisecondo. Le due voci che §6 aveva isolato come «costo
per processo» e «costo per batch» **si sono ridotte insieme**, perché erano la
stessa struttura dati vista da due lati.

**È la prima modifica della campagna che muove il wall clock.** Fin qui il
ritornello era «il kernel è lo 0,6 % del wall»; qui il wall cade perché a cadere
sono le due voci che lo componevano davvero:

| dataset (128 thread) | wall prima | wall dopo | query/s |
|---|---:|---:|---|
| `c4_err_2k` | 905 ms | **389 ms** | 2263 → **5259** (2,32×) |
| `ebola_err_2k` | 904 ms | **390 ms** | 2266 → **5253** (2,32×) |
| `c4_exact_2k` | 734 ms | **389 ms** | 2789 → **5272** (1,89×) |
| `c4_err` | 437 ms | **283 ms** | 1172 → **1809** (1,54×) |

**Correttezza**: 30/30 su dieci dataset × 64/128/256, byte-identici ai golden;
memcheck 0, racecheck 0 hazard, initcheck ai soliti 8 preesistenti.

Due cose da sapere per chi tocca questo codice:

- le celle sono **prestate, non copiate**: vivono nel buffer del workspace fino
  alla `align_batch` successiva. Il backtrace gira subito dopo la chiamata, che è
  l'unico uso mai esistito, ma ora è un vincolo di lifetime e non un dettaglio;
- una wavefront di dimensione 0 lascia il suo puntatore all'inizio della fetta
  della query, cioè subito dopo le celle della query precedente. Nessuno lo
  legge, e non è mai nullo.
