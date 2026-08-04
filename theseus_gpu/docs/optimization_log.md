# Registro ottimizzazioni GPU — config 1

Storico di ogni modifica alle prestazioni del backend CUDA, con i numeri
misurati che l'hanno motivata e quelli che l'hanno verificata. Una voce per
intervento, in ordine cronologico. Il vincolo che governa tutto resta invariato:
ogni voce deve restare **byte-identical** ai golden dell'oracle su tutti e
quattro i dataset.

> **Nota su config 0.** Dopo Opt #3 config 0 è stata rimossa da `main` e
> conservata sulla branch `legacy/config0`; il kernel rimasto ha perso il
> prefisso `config1_` da tutti i nomi. Le colonne e i confronti qui sotto
> continuano a nominarla perché è il **baseline contro cui ogni speedup è stato
> misurato**: riscriverli renderebbe i numeri illeggibili. Per riprodurre una
> qualunque di queste misure serve `git checkout legacy/config0`.

## Ambiente e metodo di misura

- **GPU**: Tesla T4 (sm_75, 15360 MiB) su Colab. 64 KB shared/SM, 65 536
  registri/SM, max 32 warp/SM.
- **Toolchain**: CUDA 12.8 (V12.8.93), CMake 3.31, `-DCMAKE_CUDA_ARCHITECTURES=75`.
- **Build**: `cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
  -DTHESEUS_PROXY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75`
  con `-DCMAKE_CUDA_FLAGS='--ptxas-options=-v'` per registri/shared/stack.
- **Tempo di kernel**: letto da `GPU timing:` su stderr di `seq2graph_proxy`,
  che viene da eventi CUDA attorno al solo lancio.

**Regola di misura, imparata sbagliando**: confrontare due build eseguendo prima
tutti i run "before" e poi tutti gli "after" produce numeri falsi. Il primo
benchmark dell'Opt #1 mostrava così una regressione del 30% su c4@256 che,
**alternando** before/after nello stesso istante termico, diventa 1.02×. Era
throttling della T4. Da allora ogni confronto è interleaved, mediana di 5–7 run.

## Stato del kernel nel tempo (ptxas, sm_75)

| stato | config1 reg | config1 smem | config1 stack | config0 reg | config0 stack |
|---|---|---|---|---|---|
| originale (pre-Opt #1) | 152 | 35 992 B statici | 136 B | 160 | 104 B |
| dopo Opt #1 | 152 | 160 B + 56×thr | 136 B | 160 | 104 B |
| dopo fix jump mancanti | 223 | 160 B + 56×thr | **14 480 B** | 255 | 14 448 B |
| dopo Opt #2 | 223 | 160 B + 56×thr | **592 B** | 255 | 560 B |
| dopo Opt #3 (stato attuale) | 234 | 192 B + 56×thr | 592 B | 255 | 560 B |

Con `sizeof(Cell) = 24`, la shared dinamica è `thread × 56` byte: 3 744 B a 64
thread, 7 328 a 128, 14 496 a 256.

---

## Opt #1 — Tiling del percorso I e shared memory dinamica

**Categoria**: tiling / riduzione del footprint on-chip → occupancy.

**Problema.** `config1_generate_and_merge_i_candidates` consumava tutto lo spazio
dei candidati in un colpo, quindi il buffer in shared era dimensionato sul caso
peggiore `kScopeWavefrontCapacity = 1024`. Con i buffer M da 256, il blocco
chiedeva 35 992 byte: su una T4 da 64 KB/SM significa **un solo blocco residente
per SM**, qualunque fosse il numero di thread.

**Intervento.**
1. Consumo dei candidati I a tile di `blockDim.x`, come già faceva
   `config1_extend_and_consume_m_cells` con il range M. L'ordine di merge è
   preservato tile per tile, lane per lane — è ciò che tiene il risultato
   identico alla CPU.
2. I quattro buffer di staging passano da `__shared__` statici a
   `extern __shared__` partizionato, dimensionato al lancio da
   `config1_shared_bytes(threads_per_block)`. Sparisce anche il `256` hardcoded
   accoppiato a mano al clamp di `threads_per_block` in un altro file.
3. Conseguenza di correttezza: il tetto `shared_i_count > kConfig1ICandidateCapacity`
   non esiste più (era un limite del buffer, non dell'algoritmo), e con esso la
   `cap_fail` mal etichettata che segnalava lo sfondamento come se fosse lo stack
   degli I-jump, con numeri inventati.

**Risultato**: shared per blocco **35 992 → 160 byte statici** più la dinamica.

**Misure** (interleaved, mediana di 7, dataset exact):

| dataset | thr | prima | dopo | speedup |
|---|---|---|---|---|
| ebola | 64 | 2.233 ms | 0.911 ms | **2.45×** |
| ebola | 128 | 1.811 ms | 1.211 ms | **1.50×** |
| ebola | 256 | 1.946 ms | 1.960 ms | 0.99× |
| c4 | 64 | 13.451 ms | 5.912 ms | **2.28×** |
| c4 | 128 | 16.204 ms | 6.185 ms | **2.62×** |
| c4 | 256 | 12.845 ms | 12.578 ms | 1.02× |

Il guadagno segue il modello di occupancy: a 64 e 128 thread la shared era il
vincolo che teneva 1 blocco per SM, rimuoverlo ne libera 3–6. A 256 thread sono
i **registri** a vincolare (256 × 152 = 38 912, più di metà dei 65 536 per SM),
quindi 1 blocco per SM prima e dopo — e infatti il tempo non cambia.

**Validazione**: simple tier 2/2 dataset, config 0 e 1 a 64/128/256, tutti
byte-identical. `ctest` 5/5. compute-sanitizer memcheck/initcheck/synccheck/
racecheck puliti a 64 e 256 thread.

---

## Fix — chiamata `check_and_store_jumps` persa nel porting

**Non è un'ottimizzazione**, ma cambia le prestazioni e va registrato perché
tutte le misure successive partono da qui.

**Bug.** La CPU chiude `next_I` con:

```cpp
if (curr_v->out_edges.size() > 0) {
  check_and_store_jumps(curr_v, sc_i_wf(*_qs, _score), new_range);
}
```

`core_next_i` in `align_core.h` non ce l'aveva. Le celle I che raggiungono
l'ultima colonna del loro vertice non aprivano mai i salti verso i vicini, quindi
le wavefront successive vedevano un insieme di candidati più piccolo.

Due indizi lo segnalavano da prima: `core_check_and_store_jumps` era definita e
mai chiamata, e `core_next_i` accettava un parametro `query` che non usava mai —
serviva esattamente a quella chiamata.

**Sintomo**: `c4_err` query 402 terminava sulla stessa cella con `prev_pos` 22
invece di 21. Differenza di traceback, non di punteggio, per questo restava
invisibile finché le read non richiedevano score > 0. Presente identico su
config 0, quindi non era codice config1.

**Fix**: chiamata ripristinata in `core_next_i` (copre config 0) e nel percorso I
separato di config 1, con `end`/`end_cell` propagati perché `core_store_m_jump`
può soddisfare la condizione di fine.

**Costo.** Rende `core_store_i_jump` raggiungibile per la prima volta, e il suo
`Frame stack[kMaxIJumpStack]` da 256 frame va in spill: **152 → 223 registri e
136 → 14 480 byte di stack**.

**Misure dopo il fix** (mediana di 5, contro la versione pre-Opt #1):

| dataset | thr | originale | dopo fix | |
|---|---|---|---|---|
| c4_err | 64 | 19.134 ms | 7.672 ms | 2.49× |
| c4_exact | 64 | 15.419 ms | 8.077 ms | 1.91× |
| c4_exact | 128 | 14.498 ms | 8.229 ms | 1.76× |
| c4_err | 128 | 15.526 ms | 11.312 ms | 1.37× |
| ebola_err | 64 | 3.127 ms | 3.322 ms | 0.94× |
| c4_exact | 256 | 14.139 ms | 14.904 ms | 0.95× |
| c4_err | 256 | 16.388 ms | 19.644 ms | 0.83× |
| ebola_exact | 64 | 1.669 ms | 2.241 ms | 0.74× |
| ebola_err | 128 | 2.560 ms | 3.459 ms | 0.74× |
| ebola_err | 256 | 3.384 ms | 5.136 ms | 0.66× |
| ebola_exact | 128 | 1.532 ms | 2.341 ms | 0.65× |
| ebola_exact | 256 | 1.680 ms | 2.997 ms | 0.56× |

Su ebola il lavoro aggiunto domina e il netto è una perdita: grafo piccolo con
vertici corti, quindi le celle I raggiungono di continuo il fine-vertice. Su c4
(32 vertici, 329 664 basi) ci arrivano di rado.

**Attenzione nel leggerle**: non è un confronto a parità di lavoro. La versione
originale era più veloce anche perché saltava calcoli obbligatori e produceva un
risultato sbagliato.

**Validazione**: con il fix, per la prima volta **4/4 dataset** su entrambi i
tier, tutte le config, byte-identical. Il complex tier risulta sbloccato: la
documentazione lo dava per non terminante, ma `c4_err` completa 512 query in
~1.9 s.

---

## Opt #2 — `kMaxIJumpStack` derivato dai dati e VerticesData parallela

**Categoria**: (a) riduzione footprint per thread — local memory / spill;
(b) riduzione della frazione seriale.

### (a) `kMaxIJumpStack` 256 → 8

**Problema.** Lo spill introdotto dal fix è quasi interamente quell'array:

```
Frame = vertex_id(4) + pad(4) + Cell(24) + prev_pos(8)
      + from_matrix(1) + pad(3) + next_edge(4) + invalidated(1) + pad(7) = 56 B
256 × 56 = 14 336 B          misurati 14 480 B  →  il 99% è questo
```

**Derivazione del bound.** Lo stack cresce solo su catene di vertici a lunghezza
zero. Due verifiche indipendenti:

1. Nessun grafo contiene segmenti a lunghezza zero:

   | grafo | segmenti S | a lunghezza zero | lunghezza minima |
   |---|---|---|---|
   | ebola.gfa | 7 | 0 | 19 |
   | c4.gfa | 16 | 0 | 1 |
   | sample_graph.gfa | 4 | 0 | 1 |

2. Picco reale misurato strumentando temporaneamente lo stack sulla CPU (che ha
   lo stesso `IJumpFrame`):

   | dataset | picco |
   |---|---|
   | ebola_exact_smoke | 0 |
   | ebola_error_smoke | 0 |
   | c4_exact | 0 |
   | **c4_err** | **1** |
   | sample_graph | 0 |

   `store_I_jump` è raggiunto **solo da c4_err**: quel percorso ha copertura di
   test minima, cosa da tenere presente.

Bound impostato a **8**, margine ×8 sul picco osservato, con `cap_fail` che resta
a segnalare l'overflow in modo non silenzioso. Strumentazione rimossa dopo la
misura.

**Risultato**: stack cumulativo **14 480 → 592 byte** (−96%). Registri invariati
a 223.

### (b) `vd_expand` / `vd_compact` parallele

`vd_expand` e `vd_compact` scorrono i vertici attivi e toccano solo gli array e
le dimensioni del vertice `a`: i vertici sono indipendenti, quindi un thread per
vertice. Le due passate sono fuse per lo stesso motivo — nulla che il vertice `a`
legge viene scritto da un altro. Nessuna delle due modifica `vd_num_active`,
quindi il bound del loop è stabile. Le versioni in `query_state.h` restano
intatte per CPU e config 0.

**Misure** (interleaved, mediana di 5, contro lo stato post-fix):

| dataset | @64 | @128 | @256 |
|---|---|---|---|
| ebola_exact | **2.41×** | 2.09× | 1.51× |
| c4_exact | 1.11× | 1.12× | 1.16× |
| ebola_err | 1.38× | 1.43× | 1.16× |
| c4_err | 1.25× | 1.15× | 1.20× |

Migliorano tutte e dodici le configurazioni. Il guadagno maggiore è su ebola,
esattamente il dataset regredito dal fix: lo spill era la causa e rimuoverlo la
annulla.

**Validazione**: 4/4 dataset, entrambi i tier, tutte le config, byte-identical.
`ctest` 5/5.

---

## Opt #3 — Densify parallelo (filtro + stream compaction) per I, D e M

**Categoria**: riduzione della frazione seriale.

**Problema.** Tre loop identici, uno per matrice, per vertice per score, tutti su
thread 0:

```cpp
for (int32_t di = 0; di < qs.sp_ndiags; ++di) {
    const int32_t diag = qs.sp_diags[di];
    if (vd_valid_diagonal(qs, matrix, v, diag)) push(...);
}
```

`vd_valid_diagonal` **non è O(1)**: scansiona linearmente i segmenti invalidi del
vertice (fino a `kMaxInvalidSegments = 64`). Ogni loop costa quindi
`Δ × segmenti` con `Δ = sp_ndiags`, che cresce con lo score. Le sparsify che li
precedono sono invece O(1) per candidato: la coda pesa più della testa.

**Intervento.**

1. `align_core.h`: `core_next_d` e `core_next_m` separate in
   `_sparsify` + `_densify`. `core_next_d`/`core_next_m` chiamano la coppia,
   quindi **config 0 resta identico** — è il baseline di misura e non va toccato.
2. `align_gpu.cu`: `config1_densify`, un filtro + stream compaction di blocco che
   sostituisce i tre loop in config 1. Il test è read-only, quindi ogni diagonale
   è indipendente; solo la posizione di scrittura è condivisa, e una **somma
   prefissa esclusiva riproduce esattamente le posizioni seriali** — è ciò che
   tiene il byte-identical.
3. La somma prefissa è un `__ballot_sync` + `__popc` per il prefisso dentro il
   warp, più uno scan sui totali dei (max 8) warp fatto da thread 0. Nessuna
   dipendenza da CUB, 9 interi di shared.
4. La sparsify resta su thread 0: fonde nello scratchpad condiviso, dove i
   candidati collidono per diagonale e l'ordine di merge è semantico.

L'overflow conserva la semantica seriale: gli elementi che non entrano vengono
scartati e `cap_fail` registra il buffer con la dimensione che avrebbe riportato
la prima push fallita. Il flag `track_peak` replica il fatto che `sc_wf_push`
aggiorna `sc_peak_wf` mentre `bs_push_back` no.

**Costo**: registri 223 → 234 (occupancy invariata: 4 blocchi/SM a 64 thread in
entrambi i casi), shared statica 160 → 192 B, stack invariato a 592 B.

**Misure** (interleaved, mediana di 5, contro il commit `daf657e`):

| dataset | @64 | @128 | @256 |
|---|---|---|---|
| ebola_exact | 2.44× | 2.06× | 1.63× |
| c4_exact | 1.23× | 1.07× | 1.19× |
| **ebola_err** | **1.94×** | 1.34× | **1.52×** |
| c4_err | 1.22× | 1.11× | 1.11× |

Questi numeri sono cumulativi (Opt #2 + Opt #3). Il contributo marginale del solo
densify, ottenuto come rapporto con le misure di Opt #2 contro lo stesso commit —
**stima approssimata, le due serie vengono da VM diverse**:

| dataset | @64 | @128 | @256 |
|---|---|---|---|
| ebola_exact | ~1.01× | ~0.99× | ~1.08× |
| c4_exact | ~1.11× | ~0.96× | ~1.03× |
| **ebola_err** | **~1.41×** | ~0.94× | **~1.31×** |
| c4_err | ~0.98× | ~0.97× | ~0.93× |

Il guadagno si concentra su `ebola_err`, e la spiegazione è coerente con la
diagnosi: il densify costa `Δ × segmenti`, e `Δ` cresce con lo score. Su ebola —
grafo piccolo, molti score, vertici corti — le diagonali attive pesano molto
rispetto al resto. Su c4 i vertici lunghissimi spostano il costo sull'LCP e il
densify conta poco, tanto che le barriere aggiunte lo rendono marginalmente
negativo (~0.93–0.98×).

**Lezione per i prossimi interventi**: il peso dei blocchi seriali dipende dalla
topologia del grafo. Ottimizzare per c4 e per ebola non è la stessa cosa, e una
media sui quattro dataset nasconde questa differenza.

**Validazione**: 4/4 dataset, entrambi i tier, config 0 e 1 a 64/128/256, tutti
byte-identical. `ctest` 5/5. compute-sanitizer memcheck/initcheck/synccheck/
racecheck puliti su `c4_err` a 64 e 256 thread.

---

## Profiling Nsight Compute

Prima run di profiling, fatta dopo Opt #2. Comando:

```bash
ncu --kernel-name regex:config1 --launch-count 1 --csv --metrics <lista> \
    seq2graph_proxy --backend gpu --gpu-config 1 --gpu-threads <T> ...
```

| run | thread attivi / warp-inst | occupancy | limite reg | limite smem | limite warp | stall barriera | stall memoria | sm throughput |
|---|---|---|---|---|---|---|---|---|
| c4_err @64 | **1.15** / 32 | 22.61% | 4 blocchi | 8 | 16 | 5.83 | 0.99 | 50.60% |
| c4_err @256 | **1.56** / 32 | 25.00% | 1 | 2 | 4 | **32.04** | 0.86 | — |
| c4_exact @64 | **1.03** / 32 | 25.00% | 4 | 8 | 16 | 5.98 | 0.84 | — |
| c4_exact @256 | **1.12** / 32 | 25.00% | 1 | 2 | 4 | **30.54** | 0.84 | — |

Metriche usate: `smsp__thread_inst_executed_per_inst_executed.ratio`,
`sm__warps_active.avg.pct_of_peak_sustained_active`,
`launch__occupancy_limit_{registers,shared_mem,warps}`,
`smsp__average_warps_issue_stalled_{barrier,long_scoreboard}_per_issue_active.ratio`,
`sm__throughput.avg.pct_of_peak_sustained_elapsed`.

**Tre conclusioni.**

1. **Gira di fatto un thread solo.** 1.03–1.56 thread attivi su 32 per istruzione
   eseguita: utilizzo del 3–5%.
2. **Non siamo memory-bound.** Stall di memoria 0.84–0.99 contro stall di
   barriera 5.8–32. I warp passano il tempo fermi al `__syncthreads()` ad
   aspettare thread 0.
3. **Lo stall di barriera esplode a 256 thread** (5.8 → 32): più warp aspettano
   lo stesso singolo thread. È la spiegazione quantitativa del perché 64 thread è
   sempre la configurazione migliore.

L'occupancy è limitata dai registri (223), non più dalla shared: 4 blocchi/SM a
64 thread contro gli 8 che la shared permetterebbe.

**Tentativo non riuscito**: profiling a livello di sorgente per attribuire il
costo ai singoli blocchi seriali. `ncu --page source --print-source sass
--import-source yes` con `-lineinfo` ha prodotto solo il SASS, senza mappatura
file:riga. Alternativa più affidabile da provare: strumentare il kernel con
`clock64()` per fase, accumulando in un buffer di debug.

---

## Mappa del parallelismo (stato attuale)

Per ogni score e vertice attivo. `T` = thread/blocco, `A` = vertici attivi,
`Δ` = `sp_ndiags`, `E` = grado uscente, `L` = lunghezza LCP.

| fase | trip count | stato |
|---|---|---|
| `vd_expand` + `vd_compact` | `A × 3 × segmenti` | **parallelo** (Opt #2) |
| costruzione candidati I | `nI / T` | **parallelo** (Opt #1) |
| merge candidati I | `nI` | seriale |
| densify I | `Δ × segmenti / T` | **parallelo** (Opt #3) |
| `core_check_and_store_jumps` | `iCells × E × L` | seriale |
| sparsify D | `nD` | seriale |
| densify D | `Δ × segmenti / T` | **parallelo** (Opt #3) |
| sparsify M | `nM` | seriale |
| densify M | `Δ × segmenti / T` | **parallelo** (Opt #3) |
| estensione LCP celle M | `mCells × L / T` | **parallelo** |
| write-back M + end check + jump | `mCells × E × L` | seriale |
| loop sui vertici attivi | `A` | seriale |

L'unica operazione con costo per elemento non costante è l'LCP: è
parallelizzata nell'estensione delle celle M, ma resta **seriale su thread 0** in
`core_check_and_store_jumps` e nel write-back M, dentro un loop sugli out-edge e
con `core_extend_diagonal` che può ricorrere.

## Cosa resta, in ordine

1. **Loop sugli out-edge in `core_store_m_jump`** — ora il candidato principale.
   Le push (`bs_push_back`, `vd_jumps_push`) sono append il cui ordine determina
   i `prev_pos`, cioè esattamente ciò su cui i golden sono confrontati:
   parallelizzarlo mantenendo il byte-identical richiede offset da prefix-sum più
   una worklist esplicita per la ricorsione. Non è un passo incrementale.
2. **Tiling delle sparsify di D e M** — resta l'unica parte non parallela di
   `core_next_d`/`core_next_m` dopo Opt #3. Le tre fasi sono disgiunte nel tempo,
   quindi possono riusare lo stesso buffer in shared: costo zero di occupancy.
   Priorità bassa: sono O(1) per candidato, ed è il motivo per cui Opt #3 ha
   preso di mira la coda invece della testa.
3. **Ridurre i registri** (223) — sono il vincolo di occupancy attuale.
   `__launch_bounds__` è un esperimento da una riga, ma rischia di reintrodurre
   spill.
4. **Loop sui vertici attivi** — il più grosso, ma `sc_wf_push` fa append
   nell'ordine dei vertici: servirebbero offset pre-calcolati per restare
   byte-identical. Ultimo.
