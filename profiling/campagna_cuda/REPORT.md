# Campagna di ottimizzazione CUDA — report

Hardware: **Tesla T4** (CC 7.5, 15 360 MiB, 40 SM), driver 580.82.07, CUDA
12.8.93, su VM Colab. Clock SM bloccati a 1590 MHz con `nvidia-smi -lgc`,
`ncu --clock-control none`. **Tutte le misure di prestazione qui dentro vengono
da una sola VM**; dove non è così è scritto.

Documenti di dettaglio: [`00_workload.md`](00_workload.md),
[`01_invariante_scratchpad.md`](01_invariante_scratchpad.md),
[`02_scratchpad_hot_cold.md`](02_scratchpad_hot_cold.md),
[`03_pinned_d2h.md`](03_pinned_d2h.md), [`04_clear_pigro.md`](04_clear_pigro.md),
[`05_categorie_cuda.md`](05_categorie_cuda.md). Dati grezzi in `dati_grezzi/`.

---

## 1. Il risultato principale

Il kernel non è limitato dalla banda. È limitato dall'**emissione di istruzioni**,
e con il 25 % di occupancy non ha abbastanza warp per nascondere la latenza. Il
modello che regge su tutti i workload misurati è

```
durata  ≈  istruzioni emesse  ×  latenza per istruzione
```

| workload | istruzioni | latenza/istr | prodotto | durata misurata |
|---|---:|---:|---:|---:|
| `c4_err_2k` | 0,33× | 0,49× | 0,16× | **0,18×** |
| `ebola_err_2k` | 0,71× | 1,13× | 0,80× | **0,82×** |

È la chiave che spiega perché due ottimizzazioni della stessa categoria hanno
dato risultati opposti: `00ea1ef` aveva reso gli store **più economici** senza
toglierne nessuno (traffico dimezzato, durata invariata); `d84fd82` ne ha tolti
cinque su sei, e la durata è seguita.

## 2. Le sei categorie

| Optimization | Implementation | Correctness | Small | Medium | Large | Kept? | Reason |
|---|---|---|---|---|---|---|---|
| **1 Privatization** | nessuna modifica: già applicata | — | — | — | — | n/a | Un blocco = una query, niente è condiviso fra query; dentro il blocco gli slot si allocano con prefix-sum su shared, non con atomiche. La ScratchPad non è privatizzabile: 1,2 MB contro 64 KB di shared. |
| **2 Thread coarsening** | **non fatta** | — | — | — | — | — | Unico punto scoperto del TASK E. I candidati insistono su fasi che dopo C1 valgono il 7,7 %–20 % del loop, e con 239 registri già al limite il coarsening taglierebbe l'occupancy sotto il 25 %. |
| **3 Coalescing** (`00ea1ef` + `d84fd82`) | clear a parole; poi split hot/cold con `sp_off` denso | **30/30** GPU + 10/10 CPU | 1,14–5,90× | 4,09–7,08× | 1,22–**7,86×** | **sì** | Kernel fino a 7,86× più veloce, nessuna regressione, +4,7 % di memoria per query. |
| **4 Divergence** (`cd83c70`) | LCP warp-cooperativo con `__ballot_sync` + `__ffs` | logica: **0 divergenze su 400 000 casi**; GPU: **non validata** | — | — | — | da decidere | Non compilata: nessuna GPU disponibile. Atteso in perdita sul tier complex (LCP corti, molte celle: un warp per cella spreca 31 lane) e ininfluente sul simple, dove l'LCP che conta è quello seriale su thread 0. |
| **5 Tiling** (`00d7536`) | query in shared, 1 KB per blocco, fallback su global | GPU: **non validata** | — | — | — | da decidere | Non compilata. Atteso piccolo: la query sono 100 byte già in L1, con 1,23 settori per richiesta. |
| **6 Occupancy** | analisi completa, sonda `-maxrregcount` pronta | contatori raccolti | 25 % | 25 % | 25 % | — | Il limite sono **sempre** i registri: 65 536 / (239 × 32) = 8 warp su 32. Identica a 64/128/256 thread. Servono ≤ 128 registri per arrivare al 50 %. |

## 3. Le ottimizzazioni specifiche di Theseus

| Optimization | Implementation | Correctness | Risultato | Kept? |
|---|---|---|---|---|
| **Pinned D2H** (`026b457`) | i tre buffer host compatti allocati una volta con `cudaHostAlloc`, tenuti per la vita dell'aligner | **30/30** GPU | D2H **8,2–10,6×** (1,4 → 12,3 GiB/s, cioè banda di bus). Wall −10 % sui LARGE complex, ma **+8 % su due dataset**: sospetto il costo del page lock di 576 MiB, non ammortizzato perché la CLI fa un batch e esce. Da separare con `--repeat`. | **sì**, con la misura aperta |
| **Niente memset per batch** (`f1c93ea`) | tolto `cudaMemset(d_states, 0, ...)`: 4,4 MB × query, 8,8 GB a 2048 | CPU 10/10; GPU **non validata** | Il memset era dentro `h2d_ms`, 40,8 ms contro 4,7 ms di kernel. Da confermare con `initcheck`. | da decidere |
| **Clear pigro ScratchPad** (`0a303ce`) | `sp_cleared` traccia il prefisso già azzerato; l'epoch non serve | CPU **10/10**, incluso il riuso di un solo stato per 2048 query; GPU **non validata** | **Corretto ma inutile sull'uso attuale**: ogni `QueryState` serve una query per processo, quindi «azzerare una volta per slot» = «una volta per query». Paga solo con slot riusati. | sì (non nuoce) |
| **Parallelizzazione dei jump** (TASK F) | **non fatta** | — | Il profiling dopo C1 la promuove: sul tier simple il 71–74 % del loop è il «resto», dominato dall'estensione del seme a punteggio 0 che `align_gpu.cu:1138` fa **su thread 0 da solo**. | — |

## 4. Come cambia il comportamento al crescere del workload

Lo speedup di C1 cresce con il batch, perché con più blocchi residenti il clear
pesa di più:

| dataset | 512 q | 1024 q | 2048 q |
|---|---:|---:|---:|
| `c4_exact` | 5,90× | 7,08× | **7,86×** |
| `c4_err` | 3,67× | 4,09× | **5,06×** |

E cambia con la *forma* dell'input, non solo con la taglia. A parità di query e
di basi, `ebola_*_2k` ha uno span di ScratchPad 5,7× più piccolo di `c4_*_2k`:

| | clear prima | clear dopo | speedup kernel |
|---|---:|---:|---:|
| `c4_exact_2k` (span 52 107) | 66,0 % | 57,7 % | 7,86× |
| `ebola_err_2k` (span 9 164) | 8,0 % | 1,5 % | 1,22× |

Sul tier **complex** il clear è finito (7,7 % su c4) e la fase più grande è
diventata `extend`, con il 29–31 % del loop. Sul tier **simple** il clear resta
dominante (57,7 %) anche a una parola per diagonale, perché quelle read finiscono
a punteggio 0 e il loop non fa quasi nulla.

## 5. Dove va il tempo, davvero

Su `c4_err_2k`, 2048 query, dopo C1 e pinned:

| | ms | quota del wall |
|---|---:|---:|
| wall del processo | 735 | 100 % |
| — kernel | 4,7 | **0,6 %** |
| — D2H | 46,7 | 6,4 % |
| — H2D (di cui il memset era 40,8) | 40,8 | 5,5 % |
| — resto (contesto CUDA, upload grafo, parsing, GAF) | ~640 | 87 % |

Il kernel è lo **0,6 % del wall**. Le 5–8× guadagnate sul kernel valgono l'1 %
del tempo del processo. Il grosso è costo fisso per processo — l'inizializzazione
del contesto CUDA da sola sono centinaia di ms — che in un uso reale, con molti
batch per processo, si ammortizza; ed è esattamente l'uso in cui anche il clear
pigro e i buffer pinned iniziano a rendere. La CLI non è quel caso.

## 6. Che cosa manca, e perché

**Cinque commit non sono mai stati compilati sotto `nvcc`**: `f1c93ea`
(memset), `d9f813f` (flag `--repeat`), `0a303ce` (clear pigro), `cd83c70` (LCP
warp), `00d7536` (query in shared). Su questa macchina non c'è GPU né toolkit; la
sessione Colab è caduta due volte e poi il backend ha smesso di assegnare T4
(`Service Unavailable`), e l'account non ha diritto ad altri acceleratori.

Da fare, in ordine, tutto già scriptato in `dati_grezzi/`:

1. build dei cinque commit e **regressione 30/30** su ciascuno (`regr.sh`);
2. `compute-sanitizer --tool initcheck`, che qui serve due volte: per la
   rimozione del memset e per il clear pigro;
3. `timing.py` su tutta la matrice per i commit che sopravvivono;
4. `--repeat 5` su `prelazy` contro `lazy` e su `c1` contro `pin`: separa il
   costo per processo da quello per batch, e chiude le due domande aperte —
   quanto costa il page lock e quanto rende il riuso degli slot;
5. `ncu.sh` per i contatori, più la sonda `-maxrregcount=128` della categoria 6;
6. thread coarsening (categoria 2), l'unica non implementata.

Fino ad allora le righe «non validata» delle tabelle vanno lette come tali:
codice scritto e ragionato, non verificato.
