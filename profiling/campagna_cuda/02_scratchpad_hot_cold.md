# C1 — Split hot/cold della ScratchPad

Categoria **3 (coalescing / data layout)**, seconda applicazione dopo il clear
coalescente di `00ea1ef`.

- baseline: `027c19b` (codice identico a `00ea1ef`; i due commit intermedi
  aggiungono solo dataset e note)
- ottimizzazione: `d84fd82` "Offset denso della ScratchPad"
- hardware: **una sola VM Colab, Tesla T4** (15 360 MiB, CC 7.5, driver
  580.82.07, CUDA 12.8.93), clock SM bloccati a 1590 MHz con `nvidia-smi -lgc`,
  `ncu --clock-control none`

## 1. La modifica

`offset` esce dalla `Cell` della ScratchPad e diventa un array denso suo,
`int32_t sp_off[kScratchpadSpan]`, che è l'**autorità** sull'attività di una
cella e sul confronto di offset. `sp_wf` resta il payload freddo e **non viene
più azzerato**.

Il clear per query passa da `span × 6` parole a `span × 1`. Le ragioni per cui è
lecito sono in [`01_invariante_scratchpad.md`](01_invariante_scratchpad.md); la
più stringente è che `sp_reset_one` azzerava già il solo `offset` fra un
punteggio e l'altro, quindi il payload stantio delle celle inattive è già oggi
la norma nel codice validato.

Costo: la QueryState passa da 4 407 560 a 4 616 456 byte, **+208 896 B (+4,7 %)**.
Il tetto di batch su una T4 scende da ~3 530 a ~3 370 query.

## 2. Correttezza

| | |
|---|---|
| CTest | 5/5 su entrambi i commit |
| Regressione GPU | **30/30** (10 dataset × 64/128/256), byte-identica ai golden |
| Build strumentata per fasi | anch'essa 30/30 |

Sempre con `--require-gpu-result`, quindi nessun risultato viene dal fallback CPU.
Sul lato host, il path CPU (che condivide `query_state.h`) riproduce tutti e
dieci i golden byte per byte.

## 3. Tempo di kernel

Mediana di 3 run dopo un warm-up; la dispersione fra run è entro l'1–3 %.

| dataset | classe | span | base | C1 | speedup |
|---|---|---:|---:|---:|---:|
| `c4_exact_2k` | LARGE | 52 107 | 29,03 ms | 3,69 ms | **7,86×** |
| `c4_exact_1k` | MEDIUM | 52 107 | 14,27 ms | 2,02 ms | **7,08×** |
| `c4_exact` | SMALL | 52 107 | 6,96 ms | 1,18 ms | **5,90×** |
| `c4_err_2k` | LARGE | 52 107 | 23,62 ms | 4,67 ms | **5,06×** |
| `c4_err_1k` | MEDIUM | 52 107 | 12,25 ms | 2,99 ms | **4,09×** |
| `c4_err` | SMALL | 52 107 | 6,29 ms | 1,71 ms | **3,67×** |
| `ebola_exact_2k` | LARGE | 9 164 | 3,72 ms | 1,07 ms | **3,48×** |
| `ebola_exact_smoke` | SMALL | 9 164 | 0,88 ms | 0,43 ms | 2,03× |
| `ebola_err_2k` | LARGE | 9 164 | 4,73 ms | 3,89 ms | 1,22× |
| `ebola_error_smoke` | SMALL | 9 164 | 1,24 ms | 1,08 ms | 1,14× |

Il guadagno segue **esattamente** il peso che il clear aveva: massimo dove lo
span è grande e il loop fa poco (c4 + read esatte), minimo dove lo span è piccolo
e il loop fa tutto (ebola + read con errori). E **cresce con la taglia del batch**
— c4_exact 5,90× a 512 query, 7,08× a 1024, 7,86× a 2048 — perché con più blocchi
residenti il clear pesa di più: è la ragione per cui i workload LARGE servivano.

## 4. Perché ha funzionato (e perché il coalescing da solo non aveva funzionato)

Contatori Nsight, `c4_err_2k` a 128 thread:

| metrica | base | C1 | rapporto |
|---|---:|---:|---:|
| istruzioni eseguite | 236,8 M | 78,7 M | 0,33× |
| istruzioni di store globale | 20,92 M | 4,25 M | 0,20× |
| store sectors | 100,3 M | 18,1 M | 0,18× |
| DRAM write | 3,42 GB | 546 MB | 0,16× |
| DRAM read | 1,02 GB | 118 MB | 0,12× |
| DRAM % del picco | 59,5 % | 50,0 % | 0,84× |
| stall `lg_throttle` | 13,42 | 0,80 | 0,06× |
| stall `barrier` | 18,18 | 13,55 | 0,75× |
| latenza warp / istruzione | 49,53 | 24,41 | 0,49× |
| registri/thread | 239 | 239 | 1,00× |
| occupancy raggiunta | 24,9 % | 24,7 % | 0,99× |
| **durata** | | | **0,18×** |

Il modello che tiene insieme le due ottimizzazioni della categoria 3 è

```
durata  ≈  istruzioni emesse  ×  latenza per istruzione
```

- `c4_err_2k`: 0,33 × 0,49 = 0,16, misurato 0,18 ✓
- `ebola_err_2k`: 0,71 × 1,13 = 0,80, misurato 0,82 ✓

Da cui la spiegazione del risultato nullo precedente. Il clear coalescente di
`00ea1ef` aveva dimezzato il **traffico** lasciando invariato il **numero di
istruzioni**, e infatti non aveva cambiato la durata: il kernel non è
bandwidth-bound, è limitato dall'emissione, e con il 25 % di occupancy (239
registri) non ci sono abbastanza warp per nascondere la latenza. Questa modifica
toglie cinque store su sei invece di renderli più economici, e la durata segue.

Non è però un risultato sprecato: senza il layout a parole di `00ea1ef` i
`sp_off` non sarebbero contigui allo stesso modo, e i due interventi si leggono
come la stessa categoria applicata prima al *come* si scrive e poi al *quanto*.

## 5. Ripartizione per fase

`clock64()` per fase (build strumentata a parte, anch'essa passata in
regressione). Sono cicli di SM misurati mentre il blocco è residente, quindi
includono l'interferenza degli altri blocchi: vanno letti come **proporzioni**,
non come costi assoluti.

| build | dataset | clear | i | d | m | extend | resto |
|---|---|---:|---:|---:|---:|---:|---:|
| base | `c4_err_2k` | **42,4 %** | 19,5 % | 19,3 % | 23,4 % | 19,0 % | 18,8 % |
| C1 | `c4_err_2k` | **7,7 %** | 17,9 % | 15,9 % | 19,6 % | 29,2 % | 17,4 % |
| base | `c4_exact_2k` | **66,0 %** | 8,2 % | 8,7 % | 9,0 % | 0,2 % | 73,9 % |
| C1 | `c4_exact_2k` | **57,7 %** | 9,1 % | 10,5 % | 9,1 % | 0,4 % | 70,9 % |
| base | `ebola_err_2k` | **8,0 %** | 17,6 % | 16,3 % | 19,5 % | 28,5 % | 18,2 % |
| C1 | `ebola_err_2k` | **1,5 %** | 17,3 % | 15,6 % | 19,3 % | 30,8 % | 17,0 % |
| base | `ebola_exact_2k` | **64,7 %** | 11,6 % | 11,3 % | 8,8 % | 1,5 % | 66,7 % |
| C1 | `ebola_exact_2k` | **17,7 %** | 8,5 % | 11,9 % | 8,1 % | 0,6 % | 70,8 % |

Le percentuali di `i/d/m/extend/resto` sono quote **del loop**, quella del clear
è sul totale. I valori della baseline riproducono la misura storica su `c4_err`
(44,3 % contro 44 %) e `c4_exact` (74,5 % contro 71 %), il che dà fiducia nella
strumentazione.

Due cose cambiano il seguito della campagna:

1. **Sul tier complex il clear è finito** (7,7 % su c4, 1,5 % su ebola) e la fase
   più grande è diventata `extend`, cioè LCP + jump, con il 29–31 % del loop.
   Non è più vero che le tecniche che agiscono sul loop lavorano sulla metà più
   piccola.
2. **Sul tier simple il clear resta dominante** (57,7 % su `c4_exact_2k`) anche a
   una parola per diagonale, perché quelle read finiscono a punteggio 0 e il loop
   non fa quasi nulla. Solo un clear **O(1)** lo toglierebbe: è esattamente il
   TASK D.

C'è poi un terzo fatto, non richiesto ma evidente: nel tier simple il «resto» è
il 71–74 % del loop, e quasi tutto è l'estensione del seme a punteggio 0, che
`align_gpu.cu:1138` esegue **sul solo thread 0**. Per una read che allinea esatta
quella è tutta l'alleanza. È il candidato naturale dell'LCP warp-cooperativo
(TASK E4), che quindi non è un esperimento di cortesia.

## 6. Tempo di wall

Qui la modifica **non si vede**, ed è il dato più importante da riportare:

| dataset | wall base | wall C1 | D2H | kernel |
|---|---:|---:|---:|---:|
| `c4_err_2k` | 848,4 ms | 824,9 ms | 407 ms | 4,7 ms |
| `c4_exact_2k` | 837,9 ms | 829,3 ms | 410 ms | 3,7 ms |
| `c4_err` | 398,5 ms | 394,2 ms | 102 ms | 1,7 ms |

Su `c4_err_2k` il kernel è lo **0,57 % del wall**. Il resto è la D2H (49 %),
l'upload del grafo, la H2D e il lavoro host. Un kernel 5 volte più veloce sposta
il wall dell'1 %: la leva successiva è la D2H (TASK B), non il kernel.

Il wall è anche molto più rumoroso del kernel — la VM è condivisa, e due punti su
diciotto sono usciti di 130 ms rispetto agli altri due run dello stesso punto.
Il tempo di kernel è invece stabile all'1–3 %.

## 7. Verdetto

**Tenuta.** Speedup di kernel 1,14×–7,86× secondo il workload, nessuna
regressione da nessuna parte, correttezza byte-identica su 30 combinazioni, costo
+4,7 % di memoria per query.
