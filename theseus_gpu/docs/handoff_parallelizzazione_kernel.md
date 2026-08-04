# Handoff — distribuzione del kernel sui thread del blocco

Stato al commit `821eee3`, branch **`parallelizza/classe-a`** (partito da `main`
a `835d0be`, **mai fatto merge né push**). Leggi questo prima di toccare
`src/gpu/align_gpu.cu`.

Il registro con i numeri misurati è `docs/optimization_log.md`, voci **Opt #4**
(classe A) e **Opt #5** (classe B). Questo documento è il contesto: perché sono
state fatte quelle scelte, cosa blocca il resto, e cosa conviene fare dopo.

---

## 1. Il punto di partenza

Un audit statico aveva contato **22 blocchi `if (threadIdx.x == 0)`** nel kernel
di allineamento: fuori da quattro fasi già parallele, il kernel girava su un
thread solo mentre gli altri aspettavano alla barriera. Il profiling Nsight
precedente lo confermava dall'altra parte: **1.03–1.56 thread attivi su 32** per
istruzione eseguita, stall di barriera 30+ a 256 thread.

Il vincolo che governa tutto: l'output GPU deve restare **bit-identico**, non
equivalente per score. Il confronto è campo per campo (`same_align_result`,
`src/theseus_aligner.cpp:125`) contro i golden dell'oracle, e deve passare a ogni
singolo step.

Il punto delicato è la risoluzione delle collisioni sulle diagonali:
`if (cell.offset < new_cell.offset) cell = new_cell`, con confronto **stretto**.
A parità di offset vince il primo arrivato in ordine di indice.

---

## 2. Il metodo: l'inventario per classe

I 22 blocchi (più uno scritto `if (tid == 0)` che il grep non prende, e uno che
sta in un altro kernel) sono stati classificati per **cosa impedisce di
distribuirli**, non per quanto costano:

| classe | significato |
|---|---|
| **A** | distribuibile senza problemi d'ordine: il risultato non dipende da quale thread fa cosa |
| **B** | contiene una risoluzione di collisioni: serve riprodurre il tie-break |
| **C** | lavoro O(1) o una tantum, non vale la pena |
| **D** | ricorsione o dipendenza sequenziale: richiede ristrutturazione |

Questa classificazione è il lascito più utile della sessione, perché dice **dove
il byte-identical è gratis e dove va dimostrato**.

### Due correzioni all'ipotesi iniziale

- **Il fan-out sugli archi uscenti era stato ipotizzato classe A. È classe D.**
  Motivazione dettagliata in §5.
- **`sc_new_score` era stato ipotizzato classe A. È classe C**: sono 5 store
  scalari. È `vd_new_score` quello con un loop.

---

## 3. Classe A — fatta (Opt #4)

Commit `8ff0e4f`, `9b61b98`, `8e3be75`.

| blocco | prima | ora |
|---|---|---|
| `sp_init` | 52 224 store di `Cell` su thread 0, per query | loop block-stride |
| `max_diag` | scansione di `graph.num_vertices` | `block_reduce_max` |
| `vd_init` + `vd_new_alignment` | due scansioni di `num_vertices` | un solo loop |
| write-back celle M | thread 0 ricopiava la tile da shared | il thread che fa l'LCP scrive in `bs_m_wf` |
| `sp_reset` ×3 per vertice per score | `sp_ndiags` store | loop block-stride |
| `vd_new_score` | `2 × vd_num_active` store | un vertice attivo per thread |

**Perché resta byte-identical** — tre argomenti distinti, uno per forma:

1. *Store dello stesso valore su celle distinte*. Le voci di `sp_diags` sono
   diagonali distinte perché `access_alloc` appende solo al primo tocco. Chi
   scrive cosa è irrilevante.
2. *Riduzione associativa* (`max_diag`). `max` su interi esatti non dipende
   dall'ordine: è lo stesso numero, non un'approssimazione.
3. *Anticipo di scritture che nessuno rilegge* (write-back M). Verificato, non
   assunto: `core_store_m_jump` e la ricorsione sotto di lui spingono solo in
   `bs_m_jumps_wf` e `bs_i_jumps_wf`, e ogni iterazione legge solo la cella su
   cui si trova.

**Effetto collaterale**: `shared_m_cells` è morto, shared dinamica da 56 a 32
byte per thread.

**Risultato**: thread attivi da 1.03–1.56 a **17–27** su 32. Tempi: **2.37× e
3.07× a 256 thread**, invariato a 64. 256 thread da configurazione peggiore a
migliore, e la varianza è collassata (12.6–29.3 ms → 5.95–6.14 ms).

---

## 4. Classe B — fatta (Opt #5)

Commit `cd1dd88`, `891f89e`, `654cb26`.

### Il punto tecnico che conta

Le tre fasi di merge sembrano un massimo per diagonale. **Non lo sono: gli
effetti osservabili sono due.**

1. **Il vincitore.** Confronto stretto ⇒ sopravvive l'argmax su (offset, −indice).
2. **L'ordine di `sp_diags`.** `access_alloc` appende la diagonale al primo
   tocco, `densify` scorre `sp_diags` in quell'ordine ⇒ l'ordine di append decide
   l'ordine del wavefront denso → i `Range` → i **`prev_pos`** delle onde
   successive, che è ciò su cui i golden sono confrontati.

Il punto (2) è quello che uno schema "atomicMax sull'offset" sbaglierebbe **in
silenzio**: celle giuste, ordine sbagliato, e il primo sintomo sarebbe un
`prev_pos` diverso molti score dopo. È esattamente il tipo di bug che aveva già
bloccato il tier complex una volta.

### Come sono riprodotti

`merge_candidate_tile` lavora su una tile in shared e calcola per ogni thread due
predicati con una scansione O(tile): *sono il vincitore* e *sono il primo
toccante*. Gli append usano `block_prefix_alloc`, estratto da `densify`: **un
solo schema di aggregazione nel kernel, non due**.

### Sull'impacchettamento a 64 bit — domanda posta, risposta misurata

I bit ci stanno con margine: `offset` è una posizione nella query, non negativa,
limitata da `query_len`; l'indice del candidato è limitato da
`kScopeWavefrontCapacity + kBeyondScopeCapacity + 2 × kMaxJumpsPerScore` =
**5 184**. Entrambi in 32 bit.

**Lo schema non è stato usato lo stesso**, per un motivo che non ha a che vedere
con i bit: `atomicMax` vuole un array indicizzato per diagonale, e le diagonali
vivono su `kScratchpadSpan` = **52 224**. Non sta in shared, e in globale
costerebbe più azzerarlo che eseguire il merge — lo stesso errore che il
profiling ha poi trovato in `sp_init`.

### Enumerazione unificata per D e M

`core_next_d_sparsify` visita tre run in sequenza, M ne visita quattro. Senza un
indice unico su tutto lo spazio l'ordine *fra* le run andrebbe perso, ed è
quell'ordine a decidere chi tocca per primo una diagonale. `SparsifyPlan` +
`make_sparsify_candidate` risolvono questo; i tre `core_sparsify_*` differiscono
solo per due cose (da dove viene l'indice sorgente, se riscrivono
`prev_pos`/`from_matrix`).

### Limite noto, verificato invece che assunto

Un candidato con **offset negativo** lascerebbe serialmente la cella a −1,
facendola appendere una seconda volta dal candidato successivo; la versione
parallela la appenderebbe una volta sola. Gli offset sono posizioni nella query e
non sono mai negativi, quindi il caso è irraggiungibile — e `cap_fail` scatta se
mai diventasse raggiungibile, invece di far divergere l'output in silenzio.

**Risultato**: 1.014×, 0.992×, 1.021×, 0.980×. **Nessun cambiamento**, previsto
prima di misurare. Vedi §6.

---

## 5. Cosa ci blocca

### 5.1 La classe D — e perché non è aggirabile a poco prezzo

Restano seriali `core_check_and_store_jumps` e la coda end-check + jump del
percorso M. Il loop sugli archi uscenti vive dentro `core_store_m_jump`
(`align_core.h:132`) e ogni iterazione tocca **tre** strutture il cui ordine di
append è osservabile:

1. `vd_activate_vertex` assegna l'indice attivo del vertice, che è `v_id`, che
   indicizza `sc_*_pos`. Cambiare l'ordine di attivazione **rinumera tutto**.
2. `bs_push_back` restituisce la posizione che diventa `prev_pos`, confrontato
   campo per campo dai golden.
3. `core_extend_diagonal` ricorre dentro `core_store_m_jump`, quindi la sequenza
   di push è un **preorder DFS**: gli append della ricorsione dell'arco *e*
   precedono quelli dell'arco *e+1*.

Lo stesso vale per `core_store_i_jump`. Parallelizzarlo mantenendo il
byte-identical richiede offset da prefix-sum **più una worklist esplicita per la
ricorsione**. Non è un passo incrementale, ed è la ragione per cui il piano
originale diceva esplicitamente di non toccarlo.

La ricorsione mutua `core_extend_diagonal` ↔ `core_store_m_jump`
(`align_core.h:140` e `293`) è la stessa cosa vista da un'altra angolazione.

### 5.2 Il vero blocco non è il parallelismo: è la banda

Il profiling fatto dopo Opt #4 dice che **il kernel non è più limitato dalla
serializzazione**:

| | c4_exact@64 | c4_exact@256 | c4_err@64 | c4_err@256 |
|---|---|---|---|---|
| thread attivi / 32 | 27.01 | 27.56 | 17.09 | 18.97 |
| stall barriera | 23.9 | 9.5 | 12.6 | 26.0 |
| **stall `lg_throttle`** | **91.9** | **100.1** | **31.5** | 26.6 |
| DRAM % picco | 70.96 | 74.39 | 56.91 | 46.95 |
| L1TEX % picco | 76.69 | 91.19 | 57.95 | 57.71 |
| **SM (matematica) % picco** | **6.99** | **6.50** | **6.38** | **4.88** |

`lg_throttle` = la pipe load/store è satura. Le unità di calcolo sono ferme.
`c4_exact@256` muove **1,541 GB in ~6,0 ms = 257 GB/s** contro i ~320 di picco
della T4: **il tempo del kernel è ormai spiegato dal traffico DRAM.**

### 5.3 E l'80% di quel traffico è un memset di celle che nessuno legge

`sp_init` azzera `kScratchpadSpan` = 52 224 celle per query. Con
`sizeof(Cell) = 24`:

- **c4**: 512 × 52 224 × 24 B = **641,7 MB** su 797,9 misurati = **80,4%**
- **ebola**: 256 × 9 164 × 24 B = **56,3 MB** su 69,5 misurati = **81,0%**

Tre conferme indipendenti:

1. `c4_exact` e `c4_err` scrivono **quasi gli stessi byte** (797,9 vs 799,2 MB)
   nonostante uno sia a score 0 e l'altro faccia più wavefront: il traffico non
   dipende dal lavoro di allineamento.
2. Rapporto c4/ebola **predetto 11,4×, misurato 11,48×**.
3. La quota predetta è l'81% su entrambi i grafi, che hanno span diversi di 5,7×.

**Questo spiega perché 64 thread non si è mai mosso**: bandwidth-bound, e a 64
thread ci sono già 4 blocchi residenti per SM (limite registri) che fornivano
abbastanza parallelismo di memoria anche quando il fill era seriale. A 256 thread
c'è 1 blocco per SM, quindi lì la serializzazione **non** era coperta — ed è
esattamente lì che Opt #4 ha dato 2,4–3,1×.

**Nota sui 21–23 settori per richiesta di store**: non è scatter. 32 thread × 24 B
contigui = 768 B = 24 settori, quindi è il valore atteso per una scrittura
contigua di uno struct AoS da 24 byte. Non spreca banda, satura la pipe per
istruzione. **La leva è scrivere meno byte, non riordinare.**

### 5.4 Il vincolo della sessione

Il lavoro è stato fatto sotto la regola «non toccare l'algoritmo, la struttura
dati o le costanti». Le classi A e B stanno dentro quella regola. **La cosa che
sposterebbe i tempi sta fuori.** Questo non è un difetto del lavoro fatto: è il
risultato che il lavoro fatto ha reso visibile.

---

## 6. Cosa fare dopo, in ordine

### 6.1 Lo ScratchPad — l'unica leva che il profilo indica

Non è parallelizzazione. Tre strade, tutte da progettare **con l'argomento di
byte-identicità davanti, non dopo**:

1. **Restringere la finestra.** Lo span è dimensionato sul vertice più lungo del
   grafo (52 006 bp in c4), ma una read da 100 bp raggiunge una banda di
   diagonali molto più stretta. Attenzione: `sp_at` e `sp_access_alloc`
   indicizzano su `sp_min_diag`, quindi cambiare la finestra cambia gli indici e
   va verificato che nessuna diagonale raggiungibile cada fuori.
2. **Epoch counter** al posto del sentinella `offset == -1`: elimina il clear del
   tutto. Cambia `Cell`, quindi cambia il layout della struttura più usata del
   kernel — e `Cell` è anche ciò che la CPU confronta.
3. **Azzerare una volta per batch invece che per query.** `sp_reset` già
   ripristina ogni cella toccata, quindi *forse* il clear per query è ridondante
   dopo il primo. **Da verificare, non da assumere**: la memoria arriva da
   `cudaMalloc` non inizializzata, e va controllato se e come le `QueryState`
   vengono riusate fra batch.

La (3) è la meno invasiva e la prima da valutare.

### 6.2 Classe C — la valutazione, che è "quasi tutta no"

Restano ~15 blocchi di classe C: `sc_pos_push`, `sc_new_score`, la scrittura di
`AlignResult`, gli init di shared, `shared_vertex = vd_get_vertex_id(l)`, i
`shared_range_*`. Sono da 1 a 5 store scalari ciascuno.

**Non conviene distribuirli.** Ognuno costerebbe una o due `__syncthreads()` in
più per risparmiare pochi store, e le barriere sono già uno stall misurabile
(9.5–26 per issue attivo). L'unico con un loop vero è il fold su `nwarps` dentro
`block_prefix_alloc` e `block_reduce_max`, che è ≤ 8 iterazioni ed è già la
primitiva di aggregazione.

Se l'obiettivo è la completezza dell'esercizio, si può fare; se è il tempo, no.
Va detto esplicitamente nel log invece di lasciare l'inventario "aperto".

### 6.3 Classe D — solo dopo lo ScratchPad

Serve, in ordine: una worklist esplicita che sostituisca la ricorsione, poi
offset pre-calcolati da prefix-sum per `vd_activate_vertex` e `bs_push_back` che
riproducano la numerazione del preorder DFS. È il pezzo più grosso rimasto e va
affrontato quando il kernel non è più bandwidth-bound, altrimenti non si misura
nulla.

### 6.4 Registri

239, ancora il limite di occupancy (4 blocchi/SM a 64 thread). `__launch_bounds__`
è un esperimento da una riga ma rischia di reintrodurre spill — lo stack è già a
736 byte.

---

## 7. Come si valida (procedura, non teoria)

Non c'è GPU locale. La build locale **compila lo stub**, non il `.cu`: `ctest`
verde non dice niente sul kernel.

```bash
conda activate colab-cli
colab new -s theseus-gpu --gpu T4
```

Poi sulla VM: estrarre il progetto, `cmake -DTHESEUS_PROXY_ENABLE_CUDA=ON
-DCMAKE_CUDA_ARCHITECTURES=75`, e

```bash
python3 scripts/run_ggbs_gpu_regression.py --suite simple  --build-dir theseus_gpu/build-gpu
python3 scripts/run_ggbs_gpu_regression.py --suite complex --build-dir theseus_gpu/build-gpu
```

**`colab stop -s theseus-gpu` quando hai finito**: una VM ferma continua a
consumare compute unit.

### Cose imparate a caro prezzo in questa sessione

- **`colab exec` va in timeout** durante una regressione lunga anche se il lavoro
  prosegue. Lancia il comando detached (`setsid`) su un file di log e fai
  polling su un `.done`.
- **Valida commit per commit, non solo lo stato finale.** Il primo giro della
  classe B ha trovato un errore di compilazione nel primo dei tre commit
  (`block_prefix_alloc` usata prima della definizione): i tre commit sono stati
  ricostruiti con la correzione al posto giusto, altrimenti la validazione
  per-commit non valeva niente. Lo script che costruisce e valida ogni albero in
  sequenza è il pattern giusto: **una sola sessione GPU, attribuzione completa.**
- **Ogni confronto di tempi va interleaved** before/after nello stesso istante
  termico, mediana di 7. È già la regola in `optimization_log.md` e vale ancora:
  la T4 throttla.

---

## 8. I commit di questa sessione

```
821eee3  Registra Opt #5: classe B
654cb26  Classe B: sparsify M parallela
891f89e  Classe B: sparsify D parallela
cd1dd88  Classe B: merge parallelo dei candidati I
bf44fb6  Registra Opt #4: distribuzione della classe A
8e3be75  Chiudi la classe A: write-back M, i tre sp_reset, vd_new_score
9b61b98  Riduci sul blocco max_diag e l'azzeramento della mappa vertice->indice
8ff0e4f  Distribuisci sul blocco l'azzeramento dello ScratchPad
```

Tutti su **`parallelizza/classe-a`**, mai fatto merge su `main` né push. Ogni
commit di codice è stato validato su T4 contro i golden dell'oracle su tutti e
quattro i dataset a 64/128/256 thread.

`align_core.h` e `theseus_aligner_impl.cpp` sono **invariati**: le funzioni
seriali (`sp_init`, `sp_reset`, `vd_init`, `vd_new_alignment`, `vd_new_score`,
`core_next_d_sparsify`, `core_next_m_sparsify`) restano quelle che la CPU chiama,
e dove sono state divise in metà scalare + loop, la somma delle due metà è
esattamente la funzione di prima.
