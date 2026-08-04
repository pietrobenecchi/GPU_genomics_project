# Analisi del parallelismo, della completezza algoritmica e delle ottimizzazioni

Analisi statica del backend CUDA in `theseus_gpu/`. Nessuna modifica al codice.
I riferimenti sono `file:riga` relativi a `theseus_gpu/`.

Il codice analizzato è lo stato di lavoro corrente (`src/gpu/align_gpu.cu`,
`src/gpu/align_core.h`, `src/query_state.h` risultano modificati rispetto a
`daf657e`).

---

## Sezione 1 — Mappatura del parallelismo

### 1.1 I quattro kernel

Il file `src/gpu/align_gpu.cu` definisce quattro `__global__`. Solo due fanno
allineamento; gli altri due sono verifica dell'upload.

**`seq_length_kernel`** — `src/gpu/align_gpu.cu:66`.
Firma `(const int32_t *offsets, int32_t num_seqs, int32_t *out_seq_lengths)`.
Lanciato da `align_batch` a `src/gpu/align_gpu.cu:1055` con
`<<<blocks, threads_per_block>>>`, dove `blocks = ceil(num_seqs /
threads_per_block)` (`:1053`). Una volta per chiamata di `align_batch`, sempre,
anche quando poi gira config 1. Un thread = una query, e il suo lavoro concreto
è una sottrazione: `offsets[i+1] - offsets[i]`. Nessuna sincronizzazione, nessun
atomico, nessuna shared. Serve solo a far dire all'host "il layout del batch è
sopravvissuto all'upload" (`src/theseus_aligner.cpp:254-262`).

**`align_kernel`** (config 0) — `src/gpu/align_gpu.cu:75`.
Lanciato a `:1070` con la stessa geometria `<<<blocks, threads_per_block>>>`,
una volta per batch, quando `options.config != Config1`.
**Un thread = un allineamento completo.** Il blocco non ha alcun significato
algoritmico: è solo un contenitore di 64/128/256 query indipendenti. Il corpo è
`align_one` (`src/gpu/align_core.h:421`), cioè il control flow CPU integrale —
loop sugli score, loop sui vertici attivi, sparsify/densify, extend, jump.
Sincronizzazioni: **nessuna**. Nessun `__syncthreads`, nessun atomico, nessuna
shared memory. L'unica barriera è quella implicita di fine kernel
(`cudaDeviceSynchronize` a `:1080`).

**`theseus_align_batch_config1_kernel`** (config 1) — `src/gpu/align_gpu.cu:704`.
Lanciato a `:1065` con `<<<batch.num_seqs, threads_per_block,
config1_smem_bytes>>>`: **un blocco per query**, `blockIdx.x` è l'indice di query
(`:710`), e il numero di blocchi è esattamente il numero di query, non un
`ceil`. Una volta per batch.
Cosa sia un thread dipende dalla fase — è la parte interessante ed è dettagliata
in 1.2. La barriera è `__syncthreads()`, usata in modo pervasivo (circa 25
occorrenze nel percorso config1). **Nessun atomico** in tutto il file; la
riduzione della frontiera usa `__ballot_sync`/`__popc` (`:238-242`).

**`graph_readback_kernel`** — `src/gpu/align_gpu.cu:773`.
Lanciato da `readback_graph` a `:1238` con `<<<num_vertices, 32>>>`, chiamato
dall'host in `verify_device_graph` (`src/theseus_aligner.cpp:103`), una volta per
batch GPU. Un blocco = un vertice; i 32 thread copiano in parallelo il testo del
vertice (`:786`) e i suoi archi uscenti (`:792`) con loop a passo `blockDim.x`.
È l'unico kernel del file con accessi globali perfettamente coalescenti, e non
partecipa all'allineamento.

### 1.2 Config 1: cosa è un thread, fase per fase

Il ciclo esterno è in `align_one_config1` (`src/gpu/align_gpu.cu:591`):
un `while (true)` (`:650`) che itera sugli score, e dentro, per ogni score, un
loop seriale sui vertici attivi in `config1_compute_new_wave` (`:576`). Dentro
un vertice, `config1_process_vertex` (`:434`) alterna cinque fasi. In tre di
esse il thread ha un'unità di lavoro concreta, nelle altre il blocco degenera a
un thread solo.

| fase | funzione | unità di lavoro di un thread |
|---|---|---|
| aging dei segmenti invalidi | `config1_expand_and_compact:532` | **un vertice attivo** (`for (a = threadIdx.x; a < num_active; a += blockDim.x)`, `:536`) — expand+compact fusi su M, I, D di quel vertice |
| generazione candidati I | `config1_generate_and_merge_i_candidates:316` | **un candidato I** dello spazio concatenato (I densi, I-jumps, M densi, M-jumps), `idx = tile_start + threadIdx.x` (`:337`); costruisce la Cell e valuta il predicato di validità |
| merge candidati I | `:349-358` | **thread 0 soltanto**: scorre le lane del tile e fonde nello scratchpad |
| densify I / D / M | `config1_densify:209` | **una diagonale attiva** `qs.sp_diags[tile + threadIdx.x]` (`:225`): valuta `vd_valid_diagonal` e partecipa alla compaction |
| sparsify D / M | `:470-473`, `:487-490` | **thread 0 soltanto** (`core_next_d_sparsify`, `core_next_m_sparsify`) |
| extend LCP delle celle M | `config1_extend_and_consume_m_cells:383` | **una cella M** `bs_m_wf[chunk_start + threadIdx.x]` (`:396`): esegue `core_lcp` su di essa |
| write-back M + end check + jump | `:408-429` | **thread 0 soltanto**: riscrive le celle, controlla la fine, apre gli M-jump |
| `core_check_and_store_jumps` sulla wavefront I | `:365-379` | **thread 0 soltanto** |

Tutta l'inizializzazione (`:613-647`), l'incremento dello score con
`sc_new_score`/`vd_new_score` (`:681-685`) e la scrittura del risultato
(`:689-700`) sono su thread 0.

La struttura ricorrente è: *fase parallela → `__syncthreads()` → fase seriale su
thread 0 → `__syncthreads()`*. Il motivo è dichiarato nei commenti ed è
algoritmico, non di comodo: le push (`bs_push_back`, `sc_wf_push`,
`vd_jumps_push`) sono append il cui **ordine determina i `prev_pos`**, cioè
esattamente il campo su cui i golden vengono confrontati byte per byte
(commenti a `:311-314`, `:466-469`).

Il profiling già registrato in `docs/optimization_log.md` misura la conseguenza:
1.03–1.56 thread attivi su 32 per istruzione eseguita, con stall di barriera
5.8 (a 64 thread) e 32.0 (a 256 thread) contro stall di memoria < 1.

### 1.3 Sincronizzazioni e loro granularità

- **`__syncthreads()`**: unica primitiva di sincronizzazione usata. Granularità
  blocco = query. Compare in tre ruoli distinti:
  1. pubblicazione di uno scalare scritto da thread 0 e letto da tutti — es.
     `shared_i_count` (`:328-333`), `shared_vertex` (`:577-580`),
     `shared_range_start/end` (`:503-514`), `block_continue` (`:651-657`);
  2. protezione del riuso dei buffer di staging tra un tile e il successivo —
     `:263` ("shared_warp_base is reused by the next tile"), `:347`, `:359`,
     `:406`, `:430`;
  3. barriera tra fasi e tra score consecutivi (`:686`).
- **`__ballot_sync(0xffffffffu, ...)`** (`:238`): sincronizzazione a livello di
  warp per la somma prefissa della compaction. Il commento a `:236-237` è
  esplicito sul fatto che tutti i thread, anche quelli fuori range con `flag=0`,
  raggiungono la ballot, così il warp resta convergente.
- **Atomici**: nessuno. `grep atomicAdd` sull'intero `src/` non dà risultati.
  I contatori condivisi (`shared_accum`, `wf_size`) sono aggiornati da thread 0
  dentro una regione protetta da barriere (`:246-254`, `:266-278`).
- **Barriera implicita tra lanci**: `seq_length_kernel` e il kernel di
  allineamento sono lanciati sullo stream di default senza sincronizzazione
  intermedia; l'unica `cudaDeviceSynchronize` è a `:1080`, dopo entrambi.
- **Barriera tra score**: esiste ma è **intra-blocco** (`:686`). Poiché un blocco
  = una query e le query sono indipendenti, non serve alcuna barriera globale
  di griglia: due query possono trovarsi a score diversi nello stesso istante.
  Non c'è né cooperative-groups né grid sync.

### 1.4 Cosa resta su CPU

- Parsing GFA e costruzione del grafo (`src/gfa_graph.cpp`, `src/graph.cpp`),
  flatten in CSR (`src/gpu/graph_csr.cpp`).
- Concatenazione del batch (chars + offsets + start node/offset),
  `src/theseus_aligner.cpp:198-212`.
- Upload del grafo, **una volta sola** per aligner, alla prima batch GPU:
  `src/theseus_aligner_impl.cpp:61-67`. Coerente con il caso d'uso "grafo fisso,
  molte query".
- Calcolo di `n_scores` e delle penalità interne
  (`src/theseus_aligner_impl.cpp:40-53`).
- **L'allineamento CPU completo di ogni singola query**, sempre, anche quando la
  GPU ha avuto successo: `src/theseus_aligner.cpp:273-277`. Serve al confronto
  di `src/theseus_aligner.cpp:283-295`, che è ciò che abilita
  `use_gpu_backtrace`. Conseguenza misurabile: il tempo *wall* di
  `align_batch_gpu` non può in nessun caso scendere sotto il tempo CPU; solo
  `kernel_ms` (eventi CUDA attorno al solo lancio, `:1114-1119`) descrive la GPU.
- **Il backtrace intero**, `src/theseus_aligner_impl.cpp:685-747`, riavvolto sul
  `QueryState` scaricato dal device (`alignment_from_gpu_result`, `:95-117`, che
  fa `*_qs = state` e poi chiama `backtrace(0)`).
- Emissione CIGAR/GAF (`src/theseus_aligner_impl.cpp:758+`).
- Verifica capacità e diagnostica (`src/theseus_aligner.cpp:321-342`).
- Readback e confronto del grafo (`src/theseus_aligner.cpp:93-103`).

Il trasferimento D2H include l'intero `QueryState` per query
(`src/gpu/align_gpu.cu:1102-1108`): con `kScratchpadSpan = 52224` sono ~4.2 MB
per query (`src/query_state.h:87` e commento associato).

---

## Sezione 2 — Completezza rispetto all'algoritmo

Nota di lettura preliminare: la convenzione degli assi qui è che `I` **consuma il
vertice** (`diag += 1`, `offset` invariato, quindi `j = diag + offset` cresce) e
`D` **consuma la query** (`diag -= 1`, `offset += 1`, quindi `j` resta fermo). Si
legge nei parametri `(offset_increase, shift_factor)` passati alle sparsify:
`(0, +1)` per I (`src/gpu/align_core.h:233`), `(1, -1)` per D (`:301`). Questo
spiega perché esistano wavefront di jump per M e per I ma non per D, e va tenuto
presente leggendo il resto della sezione.

**Loop esterno per score crescente con barriera — PRESENTE.**
Config 1: `while (true)` a `src/gpu/align_gpu.cu:650`, incremento a `:682`,
`__syncthreads()` di chiusura a `:686`. La condizione di continuazione è
calcolata da thread 0 in `block_continue` (`:652`) e riletta da tutti dopo la
barriera (`:655`), che è il modo corretto di rendere uniforme un branch di
blocco. Config 0: `while (!end && !qs.capacity_exceeded)` a
`src/gpu/align_core.h:456`, senza barriere perché è single-thread.
La barriera è di blocco, non di griglia — vedi 1.3.

**Fase EXTEND (LCP dentro il nodo) — PRESENTE.**
`core_lcp` a `src/gpu/align_core.h:25-33`. Avanza `offset` e `j` finché
`query[offset] == vertex_char(graph, v, j)`, con i due limiti `offset <
query_len` e `j < n`. Chiamata da `core_extend_diagonal`
(`src/gpu/align_core.h:380`) e, in config 1, direttamente dai thread su ogni
cella M (`src/gpu/align_gpu.cu:400`). **È l'unica fase con costo per elemento
non costante che sia effettivamente parallelizzata**, e solo nel ramo delle celle
M: dentro `core_check_and_store_jumps` e dentro il write-back resta su thread 0.

**Traboccamento a costo zero verso TUTTI i successori, con rinumerazione della
diagonale — PRESENTE.**
`core_store_m_jump` a `src/gpu/align_core.h:119-145`. Il loop
`for (e = edge_begin(graph, curr_v); e < edge_end(graph, curr_v); ++e)` (`:132`)
copre tutti gli archi uscenti. La rinumerazione è
`new_cell.diag = new_diag + graph.edge_overlaps[e]` con
`new_diag = -prev_cell.offset` (`:128`, `:134`): la nuova diagonale è ricalcolata
nel sistema di coordinate del vertice destinazione, con correzione per l'overlap
dell'arco. Nessun costo aggiunto — la cella entra nella stessa wavefront di score.
Il gemello per I è `core_store_i_jump` (`:147-199`), con la stessa aritmetica a
`:180`.
La condizione che innesca il traboccamento è `j == vertex_len(...)`, cioè bordo
del nodo raggiunto: `src/gpu/align_core.h:382` e `:212`,
`src/gpu/align_gpu.cu:420-421`.

**Natura iterativa dell'extend — PRESENTE, ma realizzata in due forme diverse e
nessuna delle due è un loop a frontiera.**
Per il ramo M è **ricorsione mutua**: `core_extend_diagonal` chiama
`core_store_m_jump` (`src/gpu/align_core.h:384`), che per ogni arco richiama
`core_extend_diagonal` sulla cella appena creata (`:140-142`). La dichiarazione
anticipata a `:112-117` esiste proprio per chiudere il ciclo. Non c'è bound
esplicito sulla profondità né `cap_fail` su questa ricorsione.
Per il ramo I è uno **stack esplicito**: `Frame stack[kMaxIJumpStack]`
(`src/gpu/align_core.h:159`), con push solo quando il vertice destinazione ha
lunghezza zero (`:187-196`) e `cap_fail(kCapIJumpStack, ...)` all'overflow
(`:189`). `kMaxIJumpStack` è 8 (`src/query_state.h:148`), derivato da un picco
misurato di 1 su c4_err.
In config 1 entrambe le forme girano **interamente su thread 0**
(`src/gpu/align_gpu.cu:408-429` e `:365-379`).

**Invalidazione delle diagonali, insiemi separati per M/I/D — PRESENTE, ed è più
ricca di una semplice marcatura.**
Gli insiemi sono tre array distinti per vertice attivo:
`vd_m_invalid`, `vd_i_invalid`, `vd_d_invalid` (`src/query_state.h:221-223`),
interrogati per matrice da `vd_valid_diagonal` (`:730-751`). La marcatura avviene
al momento del salto, sulla coppia (vertice sorgente, diagonale):
`vd_invalidate_m_jump` (`:703-724`) e `vd_invalidate_i_jump` (`:676-697`), chiamate
rispettivamente a `src/gpu/align_core.h:125` e `:166`.
La struttura non è un bit per diagonale ma un **elenco di segmenti che si
allargano nel tempo**: ogni `InvalidSeg` porta due countdown `rem_up`/`rem_down`
(`src/query_state.h:155-160`), decrementati a ogni score da `vd_expand_vec`
(`:581-594`), che estende il segmento di una diagonale quando un countdown arriva
a zero; `vd_compact_vec` (`:611-651`) poi ordina e fonde i segmenti sovrapposti.
Questo è il pezzo parallelizzato in config 1 da `config1_expand_and_compact`
(`src/gpu/align_gpu.cu:532-552`).
Il test è **lineare** nei segmenti (`:745-750`, fino a
`kMaxInvalidSegments = 64`), quindi `vd_valid_diagonal` non è O(1) — è la ragione
per cui il densify pesava.

**Fase NEXT, ricorrenza WFA affine — PRESENTE, con la simmetria completa.**
- Mismatch: `core_next_m_sparsify`, `pos_prev_m = score - scoring.mism`
  (`src/gpu/align_core.h:341`), contributo M→M con `(offset_increase=1,
  shift_factor=0)` (`:357`).
- Chiusura del gap: sempre in `core_next_m_sparsify`, i contributi da D e da I
  **allo stesso score** (`pos_prev_d = pos_prev_i = score`, `:342-343`) con
  `(0, 0)` (`:348`, `:352`).
- Apertura del gap: `pos_prev_m = score - (gapo + gape)` in
  `core_next_i_...`/`core_next_d_sparsify` (`:225`, `:295`).
- Estensione del gap: `pos_prev_i = score - gape` (`:226`),
  `pos_prev_d = score - gape` (`:296`).
Le tre matrici sono materializzate: I e D nella Scope ad anello
(`sc_i_wf`, `sc_d_wf`, `src/query_state.h:194-195`), M nella BeyondScope
`bs_m_wf` (`:182`) perché serve al backtrace.

**Contributi cross-nodo per I e D — PRESENTE per I, ASSENTE per D (e per
costruzione non serve), gather.**
I ha una wavefront di jump dedicata, `bs_i_jumps_wf` (`src/query_state.h:184`),
alimentata da `core_store_i_jump` e consumata da `core_sparsify_jumps` con
`Cell::Matrix::IJumps` (`src/gpu/align_core.h:236-238`). M ne ha una analoga,
`bs_m_jumps_wf`, consumata sia dal percorso I (`:245-247`) sia da quello D
(`:309-311`) sia da quello M (`:360-362`).
Per D **non esiste** né `bs_d_jumps_wf` né `vd_d_jumps`: D riceve contributo
cross-nodo solo indirettamente, via gli M-jump. È coerente con la convenzione
degli assi ricordata sopra (D non consuma il vertice, quindi un gap D non
attraversa un bordo di nodo), ma va registrato come asimmetria strutturale.
Il meccanismo è **scatter al momento della creazione, gather al momento del
consumo**: chi salta scrive la posizione nella lista del vertice *destinazione*
(`vd_jumps_push(qs, vd_m_jumps(qs, new_cell.vertex_id, pos_score), ...)`,
`src/gpu/align_core.h:138-139`); chi elabora il vertice `v` legge la propria
lista (`vd_i_jumps(qs, v, pos_prev_i_scope)`, `:236`). **Nessun atomico** in
nessuno dei due lati: entrambi girano su thread 0.

**Risoluzione delle collisioni sulla stessa diagonale — PRESENTE, vince l'offset
maggiore.**
Il confronto è letteralmente `if (cell.offset < new_cell.offset) cell = new_cell`
e compare in tre punti di `align_core.h` — `:58-61` (sparsify M), `:84-86`
(sparsify jumps), `:104-107` (sparsify indel) — e nel gemello config1
`config1_merge_i_candidate` (`src/gpu/align_gpu.cu:181-187`).
Il rendez-vous è lo ScratchPad denso indicizzato per diagonale,
`sp_access_alloc` (`src/query_state.h:341-358`), che aggiunge la diagonale alla
lista attiva solo al primo tocco. La lista `sp_diags` è quindi in **ordine di
primo tocco**, non ordinata per valore di diagonale.
Il merge è deliberatamente tenuto seriale in config 1: il commento a
`src/gpu/align_gpu.cu:313-314` dice che l'ordine dei candidati è semantico e che
il tiling lo preserva esattamente.

**Gestione dei cicli — PARZIALE / NON VERIFICABILE DAL CODICE.**
Non esiste né un visited set esplicito, né un limite di profondità, né un
controllo di aciclicità del grafo. Quello che di fatto limita la propagazione è
l'invalidazione: `vd_activate_vertex` (`src/query_state.h:541-564`) deduplica i
vertici, e prima di ogni push il codice verifica
`vd_valid_diagonal(..., new_cell.vertex_id, new_cell.diag)`
(`src/gpu/align_core.h:136` per M, `:183` per I). Tornare sulla stessa coppia
(vertice, diagonale) già invalidata è quindi bloccato. Un ciclo che rientri su
una diagonale **diversa** non è bloccato da questo meccanismo, e nel ramo M la
ricorsione che ne seguirebbe non ha alcun bound. L'unico bound esplicito è
`kMaxIJumpStack` (`src/query_state.h:148`), che copre solo le catene di vertici a
lunghezza zero.
Va detto che i grafi di validazione non esercitano il caso: ebola ha 7 nodi e 8
archi, e `docs/optimization_log.md` registra che nessuno dei tre grafi usati
contiene segmenti a lunghezza zero.

**Condizione di terminazione — PRESENTE, ma è una sola e nient'altro.**
`core_check_end` (`src/gpu/align_core.h:35-41`): `cell.offset == query_len`, cioè
query interamente consumata — allineamento **globale** sulla query, come dichiara
il TODO a `src/theseus_aligner_impl.cpp:138`. Il loop esce quando `end` è vero o
quando `capacity_exceeded` è vero (`src/gpu/align_gpu.cu:652`,
`src/gpu/align_core.h:456`). Non c'è un tetto sullo score, né una terminazione
per "nessuna cella attiva rimasta", né un limite di iterazioni: se `end` non
viene mai raggiunto e nessuna capacità trabocca, il loop non ha altra uscita.
Nota che `sc_new_score`/`vd_new_score` riciclano gli slot dell'anello e non
segnalano nulla, quindi lo score può crescere indefinitamente.

**Backtrace — ASSENTE dalla GPU, presente sull'host.**
Il kernel produce solo la cella terminale in `AlignResult`
(`src/gpu/align_gpu.cu:689-700`; struct a `src/gpu/align_gpu.h:54-64`). Il
percorso viene ricostruito interamente sull'host da
`TheseusAlignerImpl::backtrace` (`src/theseus_aligner_impl.cpp:730-747`) e
`one_backtrace_step` (`:685-726`), che risalgono i `prev_pos` dentro
`bs_m_wf` / `bs_m_jumps_wf` / `bs_i_jumps_wf` (`:691-693`) del `QueryState`
scaricato dal device. È il motivo per cui l'intero `QueryState` — non solo il
risultato — deve tornare in D2H (`src/gpu/align_gpu.cu:1102-1108`), ed è la voce
dominante del traffico device→host.

---

## Sezione 3 — Ottimizzazioni

| Ottimizzazione | Stato | Dove nel codice (file:riga) | Cosa è stato fatto concretamente | Evidenza nel codice | Cosa manca |
|---|---|---|---|---|---|
| **1. Maximizing occupancy** | parziale | `align_gpu.cu:33-36`, `:189`, `:951-955`, `:1065-1066`, `:734-738`; `query_state.h:148` | La shared di config 1 è dinamica e dimensionata sul blocco, non sul caso peggiore: `config1_shared_bytes(threads) = threads * (2*sizeof(Cell) + 2*sizeof(int))` = 56 B/thread, passata al lancio come terzo argomento di configurazione. Il blocco è ristretto a {64,128,256} con default 128. `kMaxIJumpStack` è stato portato a 8 con una derivazione scritta nel commento, riducendo la local memory per thread. | `config1_shared_bytes` (`:33`); `extern __shared__ unsigned char config1_smem[]` con partizionamento manuale (`:734-738`) e il commento "the layout here must match the partitioning"; `constexpr int32_t kConfig1MaxWarps = 8;  // 256 threads, the largest block allowed` (`:189`); il clamp esplicito `(options.threads_per_block == 64 \|\| == 128 \|\| == 256) ? ... : 128` (`:951-955`); il commento di derivazione di `kMaxIJumpStack` (`query_state.h:141-148`) che cita 56 B/frame e 14 336 B. Segni di tuning consapevole: costanti nominate e commentate, non numeri nudi. | Nessun `__launch_bounds__` in tutto `src/` (grep senza risultati), quindi nessun tetto dichiarato sui registri — che è il vincolo di occupancy corrente (234 reg secondo `docs/optimization_log.md`, contro gli 8 blocchi/SM che la shared permetterebbe). Nessun `cudaOccupancyMaxPotentialBlockSize`, nessuna scelta automatica del blocco: `threads_per_block` arriva dalla CLI (`apps/seq2graph_proxy.cpp:50`) con default 128, che non è il valore migliore secondo le misure registrate. Config 0 non usa shared e non è dimensionato affatto. Nessun `cudaFuncSetAttribute` per il carve-out L1/shared. |
| **2. Coalesced global memory accesses** | parziale | `align_gpu.cu:225`, `:396`, `:257-262`, `:786-795`; `query_state.h:169-244` | Nei tre loop paralleli l'indice più veloce è `threadIdx.x` sommato a una base di tile, quindi thread contigui leggono elementi contigui: diagonali attive (`sp_diags`), celle M (`bs_m_wf`), e la scrittura compattata del densify. Il kernel di readback del grafo usa lo stesso schema su `vertex_chars` e sugli archi. | `const int32_t di = tile + static_cast<int32_t>(threadIdx.x);` (`:225`) seguito da `qs.sp_diags[di]` (`:229`) — `int32_t`, accesso pienamente coalescente; `const int64_t idx = chunk_start + threadIdx.x;` (`:396`) seguito da `qs.bs_m_wf[idx]`; `const int32_t pos = range_start + shared_warp_base[warp] + warp_prefix;` (`:258`) — posizioni consecutive tra le lane attive; `graph_readback_kernel` con `for (i = text_begin + threadIdx.x; i < text_end; i += blockDim.x)` (`:786`). | `sizeof(Cell) = 24` (`cell.h:93-97`) non è potenza di due, quindi ogni accesso a un array di Cell attraversa i settori in modo disallineato; non c'è né SoA né padding a 32 B. Lo ScratchPad è indicizzato **per valore di diagonale** (`sp_at`/`sp_access_alloc`, `query_state.h:325-358`), quindi `sp_wf[diag - min_diag]` non ha alcuna relazione con `threadIdx.x` — ed è comunque letto solo da thread 0 nelle sparsify. `core_lcp` legge il testo del vertice un byte per volta (`align_core.h:29`) e in config 0 ogni thread è su una query e un vertice diversi, quindi gli accessi a `vertex_chars` di un warp sono completamente sparsi. Non esiste la copia coalescente global→shared seguita da accessi irregolari in shared: i due buffer di staging (`shared_i_candidates`, `shared_m_cells`) servono a passare dati a thread 0 per la fase seriale, non a riorganizzare un pattern di accesso. Query, testo dei vertici e `vertex_offsets` non vengono mai staged. `QueryState` è per query e occupa ~4.2 MB, quindi due query adiacenti sono lontanissime in memoria. |
| **3. Minimizing control divergence** | assente | `align_core.h:25-33`, `:745-750` (in `query_state.h`); `align_gpu.cu:236-238`, `:349-358`, `:408-429` | L'unico intervento in questa direzione è mantenere il warp convergente sulla ballot: anche i thread fuori range entrano nella `__ballot_sync` con `flag = 0`. | `core_lcp` è un `while` **byte per byte**: `while (offset < query_len && j < n && query[offset] == vertex_char(graph, v, j))` (`align_core.h:29`) — nessun caricamento a 4/8 byte, nessuno XOR, nessun `__clz`/`__ffs`/`__byte_perm`. La lista delle celle attive è `sp_diags`, riempita in ordine di primo tocco da `sp_access_alloc` (`query_state.h:348-356`) e mai riordinata; il raggruppamento per nodo esiste solo perché il loop esterno sui vertici è seriale (`align_gpu.cu:576`), non perché la lista sia ordinata. Nei loop caldi il branch dominante è `if (threadIdx.x == 0)` (`:349`, `:408`, `:470`, `:487`, `:503`): non è divergenza classica ma serializzazione totale del blocco. `vd_valid_diagonal` (`query_state.h:745-750`) è un loop con trip count dipendente dai dati del vertice, eseguito per thread nel densify. Il commento a `:236-237` documenta l'unica scelta consapevole sulla convergenza. | Confronto LCP vettorizzato a parole (assente). Ordinamento o raggruppamento della lista di celle attive (assente). Riduzione dei blocchi `threadIdx.x == 0` nei loop caldi (assente per merge candidati I, sparsify D/M, write-back M, `core_check_and_store_jumps`, loop sui vertici attivi, loop sugli out-edge). |
| **4. Tiling of reused data** | parziale | `align_gpu.cu:718-738`, `:336-346`, `:394-405` | In shared ci sono: i due buffer di staging da un tile ciascuno (`shared_i_candidates`, `shared_m_cells`, `Cell`) più i due array di flag (`shared_i_valid`, `shared_m_valid`, `int`), tutti dinamici e dimensionati su `blockDim.x`; e in shared statica gli scalari di blocco (`block_continue`, `block_end`, `block_score`, `shared_num_active`, `shared_vertex`, `shared_range_start/end`, `shared_i_count`, `shared_accum`, `block_end_cell`, `shared_i_ranges`) più `shared_warp_base[kConfig1MaxWarps]`. | Le dichiarazioni `__shared__` a `:718-729` e il partizionamento di `config1_smem` a `:734-738`; il riempimento del tile a `:342-343` e `:401-402`. `shared_warp_base` è l'unico dato in shared realmente *riusato* nel calcolo (i totali per warp della somma prefissa, `:241-258`). | La query **non** è in shared: `core_lcp` la legge da global a ogni confronto (`align_core.h:29`), e la stessa query viene riletta da ogni cella M di ogni vertice di ogni score. Il testo dei vertici non è in shared. L'array delle lunghezze dei nodi non è in shared né in registro: `vertex_len` (`align_core.h:9-11`) rifà due letture globali da `vertex_offsets` a ogni chiamata, e viene chiamato più volte per cella (`align_gpu.cu:420`, `align_core.h:208`, `:382`, `:187`, e in ogni `upper_bound`). Le wavefront (`sc_i_wf`, `sc_d_wf`, `bs_m_wf`) e lo ScratchPad restano interamente in global dentro `QueryState`. Le frontiere non sono in shared. I due buffer di staging non sono tiling di dati riusati: ogni elemento vi transita una volta sola. |
| **5. Privatization** | parziale | `align_gpu.cu:219-222`, `:238-242`, `:246-262`, `:266-278` | Il contatore della frontiera del densify è privatizzato in due livelli: prefisso **dentro il warp** via `__ballot_sync` + `__popc` senza toccare memoria, poi un solo intero per warp scritto in shared e uno scan seriale sugli (al più 8) totali di warp fatto da thread 0, che aggiorna l'accumulatore di blocco `shared_accum` in shared. Nessun atomico viene mai eseguito. | `const unsigned ballot = __ballot_sync(0xffffffffu, flag != 0);` e `__popc(ballot & ((1u << lane) - 1u))` (`:238-240`); `if (lane == 0) shared_warp_base[warp] = __popc(ballot);` (`:241-242`) — una scrittura in shared per warp, non per thread; lo scan esclusivo `for (w = 0; w < nwarps; ++w) { count = shared_warp_base[w]; shared_warp_base[w] = acc; acc += count; }` (`:247-253`); il commento a `:204-206` ("warp ballot plus a scan over the (at most 8) warp totals... 9 ints of shared memory"). Nessun `atomicAdd`/`atomicCAS` in tutto `src/`. | La privatizzazione esiste **solo** nel densify. Tutti gli altri contatori di frontiera sono aggiornati serialmente da thread 0 senza aggregazione perché non c'è nulla da aggregare: `bs_push_back` (`query_state.h:388-398`), `sc_wf_push` (`:466-476`), `sc_pos_push` (`:481-488`), `vd_jumps_push` (`:759-766`) girano tutti in regioni `threadIdx.x == 0`. Lo scan sui totali di warp è seriale su thread 0 e non usa `__shfl_sync`. Nessuna riduzione tra blocchi (non serve: i blocchi sono query indipendenti). |
| **6. Thread coarsening** | parziale | `align_gpu.cu:224`, `:336`, `:394`, `:536`; `:75-88`, `:710` | Tutti e quattro i loop paralleli sono a passo `blockDim.x`, quindi un thread gestisce più unità di lavoro quando le celle attive superano i thread del blocco: diagonali attive, candidati I, celle M, vertici attivi. Sono block-stride, non grid-stride, coerentemente col fatto che un blocco possiede una query. | `for (int32_t tile = 0; tile < ndiags; tile += blockDim.x)` (`:224`); `for (int32_t tile_start = 0; tile_start < count; tile_start += tile)` con `tile = blockDim.x` (`:334-336`); `for (int64_t chunk_start = range_start; chunk_start < range_end; chunk_start += blockDim.x)` (`:394-395`); `for (int a = threadIdx.x; a < num_active; a += blockDim.x)` (`:536`). Il commento a `:307-310` motiva il tiling dei candidati I ("Staging a tile rather than the whole candidate space..."). | Nessuna forma di coarsening sull'asse delle query: config 1 è **una query per blocco** e basta (`const int32_t query_id = blockIdx.x;`, `:710`, con `<<<batch.num_seqs, ...>>>` a `:1065`); non c'è un loop `for (q = blockIdx.x; q < num_seqs; q += gridDim.x)`. Config 0 è una query per thread con guardia `if (q >= batch.num_seqs) return;` (`:80-83`) e nessun grid-stride: se le query superano `blocks*threads` non vengono allineate — non succede perché `blocks` è calcolato su `num_seqs` (`:1053`), ma il loop non c'è. Nessun thread elabora più celle M o più candidati I *contemporaneamente* per ammortizzare le letture (il tiling è sequenziale, non srotolato). |

---

## Dubbi

Punti che non sono determinabili dal codice e che vanno chiariti a voce.

1. **Cicli nel grafo: sono ammessi nel dominio, o è garantito che il grafo sia
   un DAG?** Dalla lettura, l'unico argomento di terminazione della ricorsione
   `core_extend_diagonal` ↔ `core_store_m_jump` (`align_core.h:140`, `:384`) è
   l'invalidazione della coppia (vertice, diagonale). Se un ciclo può far
   rientrare la propagazione sullo stesso vertice con una diagonale diversa, non
   vedo cosa la fermi, e non c'è né bound di profondità né `cap_fail` su quel
   percorso (a differenza degli I-jump, che hanno `kMaxIJumpStack`). Se il grafo
   è per costruzione aciclico la domanda decade, ma va detto esplicitamente
   perché il codice non lo verifica e non lo documenta.

2. **La ricorsione device di `core_store_m_jump` ha un bound noto?** nvcc la
   compila con stack per thread. `docs/optimization_log.md` riporta 592 B di
   stack cumulativo dopo Opt #2, ma non so se quel numero includa la profondità
   massima della ricorsione o solo il frame singolo. Serve sapere se qualcuno ha
   misurato la profondità reale, come è stato fatto per `kMaxIJumpStack`.

3. **`kMaxIJumpStack = 8` è tarato su un picco misurato di 1, raggiunto solo da
   `c4_err`.** È un percorso con copertura di test bassissima. La domanda è se
   quel bound debba reggere anche grafi con segmenti a lunghezza zero (MHC,
   yeast, ecoli non sono nel set di validazione) o se sia accettato come
   provvisorio.

4. **L'allineamento CPU di ogni query in `align_batch_gpu`
   (`theseus_aligner.cpp:273-277`) è permanente o è un ponteggio di
   validazione?** Cambia completamente cosa significhi misurare le prestazioni:
   finché c'è, l'unico numero confrontabile è `kernel_ms`, mai il wall time.

5. **Il D2H dell'intero `QueryState` (~4.2 MB/query) è un requisito o una
   conseguenza del backtrace su host?** Se il backtrace legge solo
   `bs_m_wf`/`bs_m_jumps_wf`/`bs_i_jumps_wf` (`theseus_aligner_impl.cpp:691-693`),
   lo ScratchPad — che è la parte che domina i 4.2 MB — non sembra servire
   all'host. Voglio conferma che non ci sia un altro consumatore.

6. **Assenza di jump per la matrice D**: la mia lettura è che sia corretta per
   costruzione, perché con la convenzione degli assi di questo codice D non
   consuma il vertice e quindi un gap D non attraversa mai un bordo di nodo. Va
   confermato, perché è esattamente il punto in cui la traccia richiesta
   ("contributi cross-nodo per I **e D**") e il codice divergono.

7. **Il default `threads_per_block = 128`** (`align_gpu.cu:955`,
   `apps/seq2graph_proxy.cpp:50`) non coincide con la configurazione che le
   misure registrate danno per migliore (64). È una scelta deliberata o un
   residuo?

8. **Config 0 è ancora un target vivo o è solo il baseline di misura?**
   Il commento a `docs/optimization_log.md` (Opt #3, punto 1) dice che non va
   toccato perché è il baseline. Se è così, le sue proprietà (nessuna shared,
   nessuna barriera, un thread per query) non vanno valutate come ottimizzazioni
   mancanti, e l'ho assunto nella sezione 3 — ma l'ho assunto, non verificato.
