# Registro ottimizzazioni GPU — config 1

Storico di ogni modifica alle prestazioni del backend CUDA, con i numeri
misurati che l'hanno motivata e quelli che l'hanno verificata. Una voce per
intervento, in ordine cronologico. Il vincolo che governa tutto resta invariato:
ogni voce deve restare **byte-identical** ai golden dell'oracle su tutti e
quattro i dataset.

Questo file è **append-only**: le voci passate non si riscrivono mai, si
aggiungono soltanto. Ogni numero qui dentro è misurato contro uno stato preciso
del codice, quindi allineare una voce vecchia al codice di oggi la renderebbe
irriproducibile. Quando una modifica rende obsoleta una voce, si aggiunge una
voce nuova che spiega come rileggere le precedenti — è il caso di
*Struttura #1*, che va letta prima di tutto ciò che la precede.

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

## Struttura #1 — config 0 spostata su `legacy/config0`

**Non è un'ottimizzazione**: nessuna riga di logica del kernel è cambiata, non
c'è nessuna misura associata. È registrata qui perché **cambia come si leggono
tutte le voci sopra**.

**Cosa.** Config 0 (un thread CUDA per query intera) era il baseline di misura,
non un target di sviluppo. È stata rimossa da `main` e conservata sulla branch
`legacy/config0`, al commit `40c074d`, dove entrambe le configurazioni erano
presenti e validate insieme. Con una sola configurazione rimasta, la scelta fra
configurazioni non ha più significato: sono spariti anche `GpuConfig`,
`AlignOptions::config`, il parametro `gpu_config` di `align_batch_gpu`,
`GpuBatchReport::gpu_config` e il flag CLI `--gpu-config`.

Da `align_core.h` sono state rimosse le ricorrenze seriali che solo config 0
chiamava: `core_next_i`, `core_next_d`, `core_next_m`, `core_next_d_densify`,
`core_next_m_densify`, `core_process_vertex`, `core_compute_new_wave`,
`align_one`. Restano le metà `_sparsify`, l'LCP, la macchina dei jump e
`core_extend_diagonal`, che il kernel usa ancora.

**Come rileggere le voci precedenti.** Tutti gli speedup di Opt #1, Opt #2 e
Opt #3 sono misurati contro config 0 o contro uno stato in cui esisteva. Per
riprodurne uno qualunque: `git checkout legacy/config0`. Le colonne
`config0 reg` / `config0 stack` della tabella di stato restano valide per quella
branch.

**Rinomine.** Il kernel rimasto ha perso il prefisso `config1_` da ogni nome.
Le voci sopra usano i nomi vecchi; la corrispondenza è:

| prima | ora |
|---|---|
| `theseus_align_batch_config1_kernel` | `theseus_align_batch_kernel` |
| `align_one_config1` | `align_one` |
| `config1_densify` | `densify` |
| `config1_process_vertex` | `process_vertex` |
| `config1_compute_new_wave` | `compute_new_wave` |
| `config1_expand_and_compact` | `expand_and_compact` |
| `config1_generate_and_merge_i_candidates` | `generate_and_merge_i_candidates` |
| `config1_extend_and_consume_m_cells` | `extend_and_consume_m_cells` |
| `config1_prepare_i_candidate_ranges` | `prepare_i_candidate_ranges` |
| `config1_make_i_candidate` / `config1_merge_i_candidate` | `make_i_candidate` / `merge_i_candidate` |
| `config1_finish_i_wavefront` | `finish_i_wavefront` |
| `config1_shared_bytes` | `kernel_shared_bytes` |
| `Config1ICandidateRanges` | `ICandidateRanges` |
| `kConfig1MaxWarps` | `kMaxWarps` |

**Verifica.** Build non-CUDA e `ctest` 5/5 verdi, GAF byte-identical al
baseline. **Il `.cu` non è stato compilato**: la macchina su cui è stata fatta la
rimozione non ha nvcc, e il controllo è stato solo `g++ -fsyntax-only` contro
shim CUDA. La regressione GGBS su GPU va rieseguita prima di fidarsi di questo
stato.

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
2. **Tiling delle sparsify di D e M** — `core_next_d_sparsify` e
   `core_next_m_sparsify` restano l'unica parte non parallela del percorso D/M
   dopo Opt #3. Le tre fasi sono disgiunte nel tempo,
   quindi possono riusare lo stesso buffer in shared: costo zero di occupancy.
   Priorità bassa: sono O(1) per candidato, ed è il motivo per cui Opt #3 ha
   preso di mira la coda invece della testa.
3. **Ridurre i registri** (223) — sono il vincolo di occupancy attuale.
   `__launch_bounds__` è un esperimento da una riga, ma rischia di reintrodurre
   spill.
4. **Loop sui vertici attivi** — il più grosso, ma `sc_wf_push` fa append
   nell'ordine dei vertici: servirebbero offset pre-calcolati per restare
   byte-identical. Ultimo.

---

## Opt #4 — Distribuzione della classe A: i blocchi seriali senza vincoli d'ordine

**Categoria**: distribuzione sui thread del blocco di lavoro che stava su thread 0.

**Stato di partenza**: `835d0be`. **Commit**: `8ff0e4f`, `9b61b98`, `8e3be75`
(branch `parallelizza/classe-a`).

**Problema.** Un audit statico del kernel contava **22 blocchi
`if (threadIdx.x == 0)`**: fuori dalle quattro fasi già parallele, il kernel
girava su un thread solo mentre gli altri aspettavano alla barriera. È la stessa
cosa che il profiling Nsight sopra aveva già quantificato dall'altra parte —
1.03–1.56 thread attivi su 32, stall di barriera 30+ a 256 thread.

**Metodo.** Prima un inventario dei 22 blocchi, classificati per *cosa impedisce*
di distribuirli, non per quanto costano:

- **A** — distribuibile senza problemi d'ordine: il risultato non dipende da
  quale thread fa cosa.
- **B** — richiede atomici e un tie-break deterministico, perché contiene una
  risoluzione di collisioni.
- **C** — lavoro O(1) o una tantum, non vale la pena.
- **D** — ricorsione o dipendenza sequenziale, richiede ristrutturazione.

Questa voce copre **solo la classe A**, cioè i casi in cui la parallelizzazione
è per costruzione byte-identical: ogni thread scrive celle distinte, oppure
scrive lo stesso valore, oppure partecipa a una riduzione associativa. Nessuna
scelta fra parallelismo e identità bit-a-bit è stata necessaria qui — la classe
A è esattamente l'insieme in cui quella scelta non si pone.

**Cosa è stato distribuito.**

| blocco | prima | ora |
|---|---|---|
| `sp_init` | 52 224 store di `Cell` su thread 0, una volta per query | loop block-stride |
| `max_diag` | scansione di `graph.num_vertices` su thread 0 | `block_reduce_max` (riduzione warp + fold su ≤8 warp) |
| `vd_init` + `vd_new_alignment` | due scansioni di `num_vertices` su thread 0 | un solo loop block-stride |
| write-back celle M | thread 0 ricopiava l'intera tile da shared | il thread che fa l'LCP scrive direttamente in `bs_m_wf` |
| `sp_reset` ×3 per vertice per score | `sp_ndiags` store su thread 0 | loop block-stride |
| `vd_new_score` | `2 × vd_num_active` store su thread 0 | un vertice attivo per thread |

`sp_init` era di gran lunga il più pesante: per una read da 100 bp scriveva da
solo più byte di tutto il resto dell'allineamento.

**Perché resta byte-identical.** Tre argomenti distinti, uno per forma:

1. *Store dello stesso valore su celle distinte* (`sp_init`, `vd_init`,
   `sp_reset`, `vd_new_score`). Le voci di `sp_diags` sono diagonali distinte
   perché `access_alloc` appende solo al primo tocco; le altre sono indicizzate
   per costruzione. Chi scrive cosa è irrilevante.
2. *Riduzione associativa* (`max_diag`). `max` su interi esatti non dipende
   dall'ordine: è lo stesso numero della scansione seriale, non un'approssimazione.
3. *Anticipo di scritture che nessuno rilegge* (write-back M). Il loop seriale
   rendeva una cella visibile in `bs_m_wf` solo quando la raggiungeva. Scrivere
   tutta la tile in anticipo è invisibile perché niente in quel loop legge
   `bs_m_wf`: `core_store_m_jump` e la ricorsione di `core_extend_diagonal`
   sotto di lui spingono solo in `bs_m_jumps_wf` e `bs_i_jumps_wf`, e ogni
   iterazione legge solo la cella su cui si trova.

Per non far divergere la versione seriale della CPU da quella parallela del
kernel, il corpo di ogni loop vive in un posto solo: `sp_reset_one`,
`vd_new_score_one`, `vd_map_fill_count`, e le metà scalari `sp_init_window`,
`vd_init_scalar`, `vd_new_alignment_scalar`. `sp_init`, `vd_init`,
`vd_new_alignment`, `sp_reset` e `vd_new_score` restano la somma delle due metà
e sono ciò che la CPU continua a chiamare, invariate.

**Effetto collaterale sulla shared.** Con il write-back diretto,
`shared_m_cells` è morto: la shared dinamica per blocco passa da
`2 × sizeof(Cell) + 2 × sizeof(int)` a `sizeof(Cell) + 2 × sizeof(int)`, cioè da
**56 a 32 byte per thread** (a 256 thread: 14 336 → 8 192 B).

**ptxas (sm_75), before/after.**

| | registri | stack | spill | smem statica | smem dinamica |
|---|---|---|---|---|---|
| `835d0be` | 234 | 592 B | 0 | 192 B | 56 × thr |
| Opt #4 | 234 | 592 B | 0 | 192 B | **32 × thr** |

Registri, stack e spill **invariati**: la distribuzione non ha allargato il
footprint per thread, e i registri (234) restano il limite di occupancy.

**Tempo di kernel.** Mediana di 7 run interleaved before/after nello stesso
istante termico, come impone la regola di misura sopra.

| caso | before | after | speedup |
|---|---|---|---|
| c4_exact @64 | 6.685 ms | 6.703 ms | 0.997× |
| c4_exact @256 | 14.219 ms | **5.993 ms** | **2.37×** |
| c4_err @64 | 6.928 ms | 6.824 ms | 1.015× |
| c4_err @256 | 19.751 ms | **6.431 ms** | **3.07×** |

**Come leggere questi numeri — tre osservazioni.**

1. **A 64 thread non cambia niente** (0.997× e 1.015×). Il guadagno è tutto a
   256. Coerente con lo stall di barriera misurato dal profiling (5.8 a 64
   contro 30+ a 256): a 64 thread ci sono due warp che aspettano thread 0, a 256
   ce ne sono otto. Ma vuol dire anche che **a 64 thread il collo di bottiglia è
   altrove**, e non è ancora stato identificato: va profilato, non spiegato a
   tavolino.
2. **256 thread non è più la configurazione peggiore.** Prima era sistematicamente
   la più lenta (14.2 contro 6.7 su c4_exact); ora è la più veloce (5.99 contro
   6.70). È un'inversione qualitativa, non solo un miglioramento.
3. **La varianza è collassata.** I run "before" a 256 thread oscillavano fra
   12.6 e 29.3 ms; quelli "after" fra 5.95 e 6.14. La coda lunga era il tempo
   che gli 8 warp passavano fermi in barriera, e dipendeva da quanto lavoro
   seriale capitava a quella query.

**Cosa NON è stato toccato, e perché.**

- **Il fan-out sugli archi uscenti** in `core_check_and_store_jumps` /
  `core_store_m_jump` era stato ipotizzato di classe A. **Non lo è: è classe D.**
  Ogni iterazione del loop sugli archi tocca tre strutture il cui ordine di
  append è osservabile — `vd_activate_vertex` assegna l'indice attivo che diventa
  `v_id` e indicizza `sc_*_pos`; `bs_push_back` restituisce la posizione che
  diventa `prev_pos`, confrontato campo per campo dai golden; e
  `core_extend_diagonal` ricorre dentro `core_store_m_jump`, quindi la sequenza
  di push è un preorder DFS. Resta seriale.
- **I merge (classe B)**: candidati I, `core_next_d_sparsify`,
  `core_next_m_sparsify`. Restano seriali. Vedi la nota qui sotto: lo schema
  atomicMax da solo non basta.
- **La ricorsione mutua** `core_extend_diagonal` ↔ `core_store_m_jump`
  (`align_core.h:140` e `293`). Classe D, fuori portata per intervento.

**Nota per chi affronterà la classe B.** Lo schema ovvio — un `atomicMax` per
diagonale su `(offset << 32) | (~indice)`, così il massimo è sull'offset e a
parità vince l'indice più basso — riproduce il vincitore, ma **non basta**.
`sp_access_alloc` appende la diagonale a `sp_diags` **al primo tocco**, e
`densify` scorre `sp_diags` in quell'ordine: l'ordine di primo tocco determina
l'ordine delle celle nel wavefront denso, quindi i `Range`, quindi i `prev_pos`
delle onde successive. Serve una seconda riduzione (`atomicMin` sull'indice per
ricostruire il primo toccante) più una compaction in ordine di indice — che è la
stessa macchina ballot+prefisso di `densify`. Va anche risolto il
dimensionamento dell'array temporaneo: una entry per diagonale su
`kScratchpadSpan` sono 52 224 slot, troppi per la shared.

**Verifica.** Regressione CPU-GPU su T4 dopo *ognuno* dei tre commit, non solo
alla fine: `ebola_exact_smoke`, `c4_exact`, `ebola_error_smoke`, `c4_err` a
64/128/256 thread — 12 run per commit, tutte PASS, output GPU identico ai golden
dell'oracle campo per campo. Build non-CUDA e `ctest` 5/5 verdi a ogni passo.

**Come rileggere le voci precedenti.** La riga «dopo Opt #3 (stato attuale)»
nella tabella *Stato del kernel nel tempo* non è più lo stato attuale: la
tabella ptxas qui sopra lo è. Registri, stack e smem statica sono invariati
rispetto a Opt #3; è cambiata solo la smem dinamica (56 → 32 byte/thread). La
sezione *Mappa del parallelismo* e la lista *Cosa resta* più sopra sono
anteriori a questa voce; quelle aggiornate sono qui sotto.

### Mappa del parallelismo dopo Opt #4

Per ogni score e vertice attivo. `T` = thread/blocco, `A` = vertici attivi,
`Δ` = `sp_ndiags`, `E` = grado uscente, `L` = lunghezza LCP.

| fase | trip count | stato |
|---|---|---|
| init `sp_init` | `span / T` | **parallelo** (Opt #4) |
| init `max_diag` | `V / T` + log | **parallelo** (Opt #4) |
| init mappa vertice→indice | `V / T` | **parallelo** (Opt #4) |
| `vd_expand` + `vd_compact` | `A × 3 × segmenti` | **parallelo** (Opt #2) |
| costruzione candidati I | `nI / T` | **parallelo** (Opt #1) |
| merge candidati I | `nI` | seriale — classe B |
| densify I | `Δ × segmenti / T` | **parallelo** (Opt #3) |
| `core_check_and_store_jumps` | `iCells × E × L` | seriale — classe D |
| `sp_reset` (×3) | `Δ / T` | **parallelo** (Opt #4) |
| sparsify D | `nD` | seriale — classe B |
| densify D | `Δ × segmenti / T` | **parallelo** (Opt #3) |
| sparsify M | `nM` | seriale — classe B |
| densify M | `Δ × segmenti / T` | **parallelo** (Opt #3) |
| estensione LCP celle M | `mCells × L / T` | **parallelo** |
| write-back celle M | `mCells / T` | **parallelo** (Opt #4) |
| end check + jump celle M | `mCells × E × L` | seriale — classe D |
| `vd_new_score` | `A / T` | **parallelo** (Opt #4) |
| loop sui vertici attivi | `A` | seriale |

### Cosa resta dopo Opt #4, in ordine

1. **Profilare a 64 thread.** È l'unico numero che Opt #4 non ha mosso, e non
   sappiamo perché. Prima di scegliere il prossimo intervento serve sapere dove
   va il tempo a quel block size, altrimenti si ottimizza al buio.
2. **I merge di classe B** — candidati I, sparsify D e M. Fattibile ma non
   naive: serve la doppia riduzione descritta sopra più il dimensionamento
   dell'array temporaneo.
3. **Il fan-out sugli out-edge** (classe D) — resta il candidato più grosso del
   percorso per vertice, e resta quello che richiede offset da prefix-sum più una
   worklist esplicita per la ricorsione.
4. **Ridurre i registri** (234) — sono ancora il limite di occupancy, e ora che
   la smem dinamica è scesa a 32 byte/thread lo sono in modo ancora più netto.

---

## Opt #5 — Classe B: le tre fasi di merge

**Categoria**: distribuzione sui thread di fasi con risoluzione di collisioni.

**Stato di partenza**: `bf44fb6` (fine Opt #4). **Commit**: `cd1dd88`, `891f89e`,
`654cb26`.

**Avvertenza, scritta prima di misurare.** Il profiling della sezione precedente
diceva che questa voce non avrebbe spostato i tempi: le fasi di merge sono
O(candidati per vertice), decine di celle, contro 642 MB di memset che dominano
il traffico. È stata fatta lo stesso per chiudere l'inventario dell'audit e
perché è la parte che *sembrava* impossibile mantenere byte-identical. La
previsione era ~1.0× e i numeri in fondo la confermano. Non è un fallimento
dell'intervento: è il motivo per cui il prossimo va scelto dal profilo, non
dall'inventario.

**Il problema, che non è quello che sembra.** Le tre fasi — merge dei candidati
I, `core_next_d_sparsify`, `core_next_m_sparsify` — fanno tutte la stessa cosa:

```cpp
Cell &cell = sp_access_alloc(qs, c.diag);
if (cell.offset < c.offset) cell = c;
```

su una sequenza di candidati in ordine di indice. Sembra un massimo per
diagonale. **Non lo è: gli effetti osservabili sono due.**

1. **Il vincitore.** Il confronto è stretto, quindi un candidato successivo con
   offset uguale non sostituisce mai il precedente: sopravvive l'argmax su
   (offset, −indice).
2. **L'ordine di `sp_diags`.** `access_alloc` appende la diagonale al primo
   tocco, e `densify` scorre `sp_diags` in quell'ordine. L'ordine di append
   decide l'ordine del wavefront denso → i `Range` → i `prev_pos` delle onde
   successive, che è esattamente ciò su cui i golden sono confrontati campo per
   campo.

Il punto (2) è quello che uno schema "atomicMax sull'offset" sbaglierebbe **in
silenzio**: produrrebbe le celle giuste nell'ordine sbagliato, e il primo
sintomo sarebbe un `prev_pos` diverso molti score dopo.

**Come sono riprodotti.** `merge_candidate_tile` lavora su una tile di
candidati messa in shared, e per ogni thread calcola due predicati con una
scansione O(tile) sulla tile:

- *sono il vincitore* — nessun altro candidato valido sulla mia diagonale ha
  offset maggiore, o offset uguale e indice minore;
- *sono il primo toccante* — nessun candidato valido sulla mia diagonale ha
  indice minore.

Gli append dei primi toccanti sono numerati con `block_prefix_alloc`, cioè lo
stesso allocatore a prefix-sum che `densify` usava già: è stato estratto e ora
`densify` lo chiama, quindi nel kernel c'è **un solo schema di aggregazione, non
due**. Ridurre la tile all'argmax e confrontarlo una volta con la cella già
presente dà lo stesso risultato del confronto uno per uno, perché ciò che è già
nella cella viene da un indice più piccolo e vince le proprie parità.

**Sull'impacchettamento a 64 bit: i bit ci stanno, ma non serve.** La domanda
era se `(offset << 32) | (~indice)` perde bit. Non li perde:

- `offset` è una posizione nella query, non negativa, limitata dal filtro
  `offset <= query_len` (100 bp nei dataset GGBS, `int32_t` in generale);
- l'indice del candidato è limitato dallo spazio dei candidati di una fase,
  cioè `kScopeWavefrontCapacity + kBeyondScopeCapacity + 2 × kMaxJumpsPerScore`
  = **5 184**.

Entrambi stanno in 32 bit con margini enormi. **Lo schema non è stato usato lo
stesso**, per un motivo che non ha a che vedere con i bit: `atomicMax` vuole un
array indicizzato per diagonale, e le diagonali vivono su `kScratchpadSpan` =
52 224. Un array così non sta in shared, e in globale costerebbe più azzerarlo
che eseguire il merge — lo stesso errore che il profiling ha appena trovato in
`sp_init`. La scansione O(tile) per thread costa `blockDim.x` confronti in
shared e non alloca niente.

**Le fasi D e M hanno richiesto un'enumerazione unificata.**
`core_next_d_sparsify` visita tre run in sequenza (indel sul wavefront D,
`sparsify_m` su `bs_m_wf`, `sparsify_jumps` su `bs_m_jumps_wf`), M ne visita
quattro. Senza un indice unico su tutto lo spazio l'ordine *fra* una run e
l'altra andrebbe perso — ed è quell'ordine a decidere chi tocca per primo una
diagonale. `SparsifyPlan` descrive le run in ordine e `make_sparsify_candidate`
ricostruisce il candidato `idx` da solo; i tre `core_sparsify_*` differiscono
solo per due cose (da dove viene l'indice sorgente, e se riscrivono
`prev_pos`/`from_matrix`), quindi una sola descrizione li copre tutti.

Il piano vive in shared, scritto da thread 0: sono 4 run da ~40 byte, e tenerlo
nei registri di ogni thread avrebbe fatto spill proprio dove i registri sono il
limite di occupancy. Le tile riusano `shared_i_candidates` e `shared_i_valid`,
liberi a quel punto: **nessuna shared dinamica in più**.

**Un limite noto, verificato invece che assunto.** Un candidato con offset
negativo lascerebbe serialmente la cella a −1, facendola appendere una seconda
volta dal candidato successivo; la versione parallela la appenderebbe una volta
sola. Gli offset sono posizioni nella query e non sono mai negativi, quindi il
caso è irraggiungibile — e `cap_fail` scatta se mai diventasse raggiungibile,
invece di far divergere l'output in silenzio.

**ptxas (sm_75), before/after.**

| | registri | stack | spill | smem statica | smem dinamica |
|---|---|---|---|---|---|
| `bf44fb6` (Opt #4) | 234 | 592 B | 0 | 192 B | 32 × thr |
| Opt #5 | 239 | 736 B | 0 | **368 B** | 32 × thr |

I 176 byte di smem statica in più sono `SparsifyPlan`. I 5 registri in più non
cambiano il limite di occupancy: 65 536 / 239 / 64 dà ancora 4 blocchi per SM a
64 thread, come con 234.

**Tempo di kernel.** Mediana di 7 run interleaved, stesso metodo di Opt #4.

| caso | base (Opt #4) | classe B | speedup |
|---|---|---|---|
| c4_exact @64 | 6.767 ms | 6.673 ms | 1.014× |
| c4_exact @256 | 6.022 ms | 6.068 ms | 0.992× |
| c4_err @64 | 6.833 ms | 6.695 ms | 1.021× |
| c4_err @256 | 6.393 ms | 6.521 ms | 0.980× |

**Nessun cambiamento**, come previsto: tutto entro ±2%, cioè dentro il rumore
del metodo. Il kernel resta bandwidth-bound.

**Verifica.** Regressione CPU-GPU su T4 costruendo e validando **ogni commit
separatamente**, non solo lo stato finale: `base`, `i_merge`, `d_sparsify`,
`m_sparsify`, ognuno su `ebola_exact_smoke`, `c4_exact`, `ebola_error_smoke`,
`c4_err` a 64/128/256 thread. 12 run per commit, tutte PASS.

Il primo giro ha trovato un errore di compilazione in `i_merge`
(`block_prefix_alloc` usata prima della definizione): i tre commit sono stati
ricostruiti con la correzione al posto giusto, così ogni commit intermedio
compila davvero e la validazione per-commit vale qualcosa.

### Mappa del parallelismo dopo Opt #5

Sostituisce quella di Opt #4 per le sole righe cambiate.

| fase | trip count | stato |
|---|---|---|
| merge candidati I | `nI / T` | **parallelo** (Opt #5) |
| sparsify D | `nD / T` | **parallelo** (Opt #5) |
| sparsify M | `nM / T` | **parallelo** (Opt #5) |
| `core_check_and_store_jumps` | `iCells × E × L` | seriale — classe D |
| end check + jump celle M | `mCells × E × L` | seriale — classe D |
| loop sui vertici attivi | `A` | seriale |

Dentro `process_vertex` restano su thread 0 solo: la costruzione del piano
(poche letture scalari), i due `sc_pos_push`, la lettura del range delle celle M
e le due fasi di classe D. **L'inventario dell'audit è chiuso per le classi A e
B**; restano C (che non vale la pena) e D.

### Cosa resta dopo Opt #5, in ordine

1. **Lo ScratchPad.** `sp_init` azzera 52 224 celle per query ed è l'80% delle
   scritture DRAM. È l'unica cosa nel kernel che, secondo il profilo, possa
   ancora cambiare i tempi in modo sostanziale. Non è parallelizzazione: è
   restringere la finestra, o sostituire il sentinella con un contatore di
   epoca, o azzerare una volta per batch invece che per query. Tutte e tre
   toccano struttura dati o derivazione dei bound, quindi vanno progettate con
   l'argomento di byte-identicità davanti, non dopo.
2. **Il fan-out sugli out-edge** (classe D) — invariato rispetto a Opt #4.
3. **Ridurre i registri** (239) — restano il limite di occupancy.
