# Fase D — Analisi statica dello ScratchPad

**Solo analisi.** Nessuna modifica al codice, nessuna proposta di patch. Questo file è un
inventario di fatti con riferimenti a file e riga, più una valutazione delle invarianti
che un futuro *clear* logico dovrebbe preservare.

Commit analizzato: `88ddef2225d5d72cdee194b9e4abcc24606f7ef6`.

---

## Domanda 1 — Le istanze di `QueryState` sono riusate fra batch?

**No. Sono allocate ex novo, azzerate e liberate a ogni invocazione.** Un contatore di
epoca non può sopravvivere fra invocazioni: ogni chiamata riparte da memoria azzerata.

### La catena di allocazione, per riga

| dove | riga | cosa succede |
|---|---|---|
| `src/gpu/align_gpu.cu` | 1408 | `states_bytes = sizeof(QueryState) * batch.num_seqs` |
| `src/gpu/align_gpu.cu` | 1415 | `QueryState *d_states = nullptr;` — variabile **locale** ad `align_batch` |
| `src/gpu/align_gpu.cu` | 1464 | `cudaMalloc(&d_states, states_bytes)` |
| `src/gpu/align_gpu.cu` | 1515 | `cudaMemset(d_states, 0, states_bytes)` |
| `src/gpu/align_gpu.cu` | 1537 | lancio del kernel, `d_states` come argomento |
| `src/gpu/align_gpu.cu` | 1570 | `cudaMemcpy(out_query_states, d_states, …, D2H)` |
| `src/gpu/align_gpu.cu` | 1594 | `cudaFree(d_states)` nel blocco `cleanup` |

`d_states` non è né statico né membro: nasce e muore dentro `align_batch`. Non esiste
nessuna cache di buffer, nessun pool, nessun puntatore persistente fra chiamate. Il
`cleanup` è raggiunto da tutti i percorsi di uscita (i `goto cleanup` sugli errori e la
caduta naturale in fondo), quindi la `cudaFree` avviene sempre.

Lato host, `src/theseus_aligner.cpp:218` costruisce
`std::vector<QueryState> device_states(seqs.size())` come variabile locale di
`align_batch_gpu`: anche la copia host è nuova a ogni chiamata e viene distrutta
all'uscita.

Per contrasto, il grafo **è** riusato: `src/theseus_aligner.cpp:215` prende
`aligner_impl_->device_graph()`, con il commento «Uploaded once per aligner, on the first
GPU batch». La differenza di ciclo di vita fra grafo e `QueryState` è deliberata.

### Quante volte viene invocato `align_batch_gpu` per processo

Una sola volta, in tutti i binari di questo albero:
`apps/seq2graph_proxy.cpp:207`, `apps/seq2graph_gpu_benchmark.cpp:53`,
`apps/seq2graph_gpu_validate.cpp:74,122,148`. Non c'è nessun ciclo di batching: la griglia
è `gridDim = batch.num_seqs` (`align_gpu.cu:1537`), cioè un blocco per query e tutte le
query in un lancio solo. In `seq2graph_gpu_validate` le tre chiamate sono tre casi di test
distinti, ciascuno con la propria allocazione.

### Conseguenza per un contatore di epoca

1. **Non deve sopravvivere fra invocazioni, e non può.** La memoria è nuova a ogni
   chiamata, quindi il problema «l'epoca residua della chiamata precedente» non esiste.
2. **Lo stato iniziale è zero, non indefinito**, per via della `cudaMemset` a riga 1515.
   Un campo epoca partirebbe quindi da 0 su tutti i `kScratchpadSpan` elementi. Questo è
   un vincolo, non una comodità: se anche l'epoca *corrente* partisse da 0, tutte le celle
   risulterebbero valide fin dall'inizio, cioè esattamente il contrario di quello che
   serve. La prima epoca valida deve essere diversa dal valore che la `cudaMemset` lascia.
3. **La `cudaMemset` di riga 1515 azzera anche `sp_wf`**, cioè scrive già oggi
   `sizeof(QueryState) × num_seqs` byte — 1,07 GB per un batch da 256 query, 2,15 GB per
   uno da 512. È traffico che precede il kernel e che il profiling del kernel **non**
   vede; va tenuto presente perché un clear logico dentro il kernel non lo elimina.
4. **Un'epoca a 32 bit non può traboccare entro una singola invocazione** con i dati
   attuali: l'epoca avanzerebbe una volta per `sp_reset_block`, cioè tre volte per vertice
   per score (`align_gpu.cu:881, 894, 913`), e i conteggi misurati sono di ordini di
   grandezza inferiori. Resta però una condizione da verificare, non da assumere: se
   l'epoca traboccasse, celle vecchie tornerebbero valide silenziosamente.

---

## Domanda 2 — Esistono letture di campi diversi da `offset` senza aver prima controllato il sentinella?

**Sì: esattamente una.** È `sp_at` chiamata da `densify` in `src/gpu/align_gpu.cu:593`.
Non è un bug: è protetta da un'invariante diversa — l'appartenenza alla lista dei
diagonali attivi — e non dal sentinella. Ma è l'unico sito che un clear logico deve
trattare con attenzione, perché è l'unico che si porta via `prev_pos`, `vertex_id`,
`diag` e `from_matrix` oltre a `offset`.

### Inventario completo dei siti che toccano `qs.sp_wf[...]`

Classificazione: **S** = legge solo `offset` come sentinella (`== -1`); **O** = legge solo
`offset` ma come *ordinamento* (confronto `<`, che dipende dal fatto che −1 sia minore di
ogni offset valido); **F** = legge o scrive la Cell intera; **W** = sola scrittura.

| # | file:riga | codice | classe |
|---|---|---|---|
| 1 | `src/query_state.h:333` | `qs.sp_wf[i] = Cell{-1,…}` — `sp_init`, ciclo seriale (percorso CPU) | W |
| 2 | `src/gpu/align_gpu.cu:1072` | `qs.sp_wf[i] = Cell{-1,…}` — il clear distribuito sul blocco | W |
| 3 | `src/query_state.h:347` | `return qs.sp_wf[idx];` — `sp_at`, **restituisce un riferimento** | (dipende dal chiamante) |
| 4 | `src/query_state.h:363` | `if (qs.sp_wf[idx].offset == -1)` — `sp_access_alloc`, decide l'append | **S** |
| 5 | `src/query_state.h:372` | `return qs.sp_wf[idx];` — `sp_access_alloc`, **riferimento** | (dipende dal chiamante) |
| 6 | `src/query_state.h:380` | `qs.sp_wf[…].offset = -1;` — `sp_reset_one` | W (solo `offset`) |
| 7 | `src/gpu/align_gpu.cu:342` | `qs.sp_wf[idx].offset == -1` — `needs_append` in `merge_candidate_tile` | **S** |
| 8 | `src/gpu/align_gpu.cu:365` | `if (is_winner && qs.sp_wf[idx].offset < my_off)` | **O** |
| 9 | `src/gpu/align_gpu.cu:366` | `qs.sp_wf[idx] = shared_cells[tx];` | W (Cell intera) |
| 10 | `src/gpu/align_gpu.cu:593` | `value = sp_at(qs, diag);` — **copia la Cell intera** | **F (lettura)** |

Chiamanti dei riferimenti restituiti da `sp_access_alloc` (riga 5), tutti in
`src/gpu/align_core.h`, tutti con la stessa forma:

| # | file:riga | codice | classe |
|---|---|---|---|
| 11 | `align_core.h:58–61` | `Cell &cell = sp_access_alloc(…); if (cell.offset < new_cell.offset) cell = new_cell;` — `core_sparsify_wavefront` | **O** in lettura, W in scrittura |
| 12 | `align_core.h:83–86` | idem — `core_sparsify_jumps` | **O** / W |
| 13 | `align_core.h:104–107` | idem — `core_sparsify_indel` | **O** / W |

`sp_at` (riga 3) ha un unico chiamante in tutto l'albero GPU: la riga 10. Non ci sono
altri usi.

### Il sito n. 10, in dettaglio

```cpp
// src/gpu/align_gpu.cu:585-594, dentro densify()
for (int32_t tile = 0; tile < ndiags; tile += ntx) {
    const int32_t di = tile + tx;
    int32_t flag = 0;
    Cell value{-1, -1, -1, -1, Cell::Matrix::None};
    if (di < ndiags) {
        const int32_t diag = qs.sp_diags[di];
        if (vd_valid_diagonal(qs, matrix, v, diag)) {
            flag = 1;
            value = sp_at(qs, diag);        // <-- copia tutti e cinque i campi
        }
    }
```

Non c'è nessun test su `value.offset == -1`. Quello che rende la lettura corretta è che
`diag` non è un diagonale arbitrario: viene da `qs.sp_diags[di]` con `di < qs.sp_ndiags`,
cioè dalla **lista dei diagonali attivi**. Una diagonale entra in quella lista solo in due
punti, ed entrambi la aggiungono esattamente quando la cella smette di essere vuota:

- `src/query_state.h:363–370` (`sp_access_alloc`): l'append avviene nel ramo
  `if (qs.sp_wf[idx].offset == -1)`, e il chiamante scrive subito dopo;
- `src/gpu/align_gpu.cu:342–353` (`merge_candidate_tile`): `needs_append` è calcolato con
  lo stesso test `offset == -1`, e la scrittura della cella avviene a riga 366.

Quindi vale l'invariante: *se una diagonale è in `sp_diags[0..sp_ndiags)`, la sua cella è
stata scritta per intero almeno una volta dall'ultimo reset.* La lettura completa a riga
593 è coperta da questa invariante, non dal sentinella.

### Il caso che l'invariante non copre, e che oggi non si verifica

`sp_reset_one` (riga 6) rimette `offset = -1` **lasciando stantii gli altri quattro
campi**, e `sp_reset_block` (`align_gpu.cu:639–651`) azzera poi `sp_ndiags`. Dopo un
reset la lista attiva è vuota, quindi la riga 593 non può raggiungere una cella con
`offset = -1` e campi stantii. La correttezza dipende dal fatto che i due azzeramenti
avvengano insieme: `sp_ndiags = 0` è dentro `sp_reset_block` (riga 648), dopo il
`__syncthreads()` che chiude il ciclo dei `sp_reset_one`.

### Siti che leggono i *metadati* dello ScratchPad, non le celle

Per completezza, non sono letture di celle e non entrano nella classificazione sopra:
`qs.sp_min_diag` a `query_state.h:341, 357, 380` e `align_gpu.cu:310`;
`qs.sp_ndiags` a `align_gpu.cu:346, 359, 577, 642, 648`;
`qs.sp_diags[...]` a `query_state.h:369, 380` e `align_gpu.cu:353, 590`.
`qs.sp_overflow_cell` (`query_state.h:320, 344, 360, 366`) è la cella-discarica restituita
in caso di sfondamento: viene riscritta prima di ogni ritorno, quindi non conserva stato.

---

## Quali invarianti dovrebbe preservare un futuro clear logico

Valutazione, non progetto. Un clear logico — qualunque forma prenda — deve continuare a
garantire queste cinque cose, perché sono ciò su cui il codice attuale si appoggia:

1. **`offset == -1` deve restare l'unico predicato di «cella vuota».** È testato in tre
   punti (siti 4, 7 e implicitamente 8) e ciascuno decide un *append* alla lista attiva.
   Se una cella logicamente vuota potesse presentarsi con un `offset` diverso da −1, la
   lista attiva cambierebbe contenuto.

2. **Deve restare vero che −1 è minore di ogni `offset` valido** (siti 8, 11, 12, 13). I
   confronti `cell.offset < new_cell.offset` non sono test di vuoto: sono la riduzione di
   massimo che sceglie il candidato vincente su una diagonale. Una cella stantia con un
   `offset` grande sopravvissuta a un clear logico **vincerebbe** contro il candidato
   nuovo, in silenzio e senza sfondare nessuna capacità. È il modo più probabile in cui un
   clear logico sbagliato romperebbe l'identità byte per byte.

3. **L'ordine di primo tocco di `sp_diags` deve restare identico.** Il commento a
   `align_gpu.cu:273–275` lo dice esplicitamente: «access_alloc appends the diagonal on
   first touch, and densify walks sp_diags in that order, so the append order decides the
   [order]». L'ordine di append è parte dell'output osservabile, perché decide l'ordine
   delle celle in `bs_m_wf`, e quindi i `prev_pos` che finiscono nel GAF. Se il clear
   logico cambiasse quali celle risultano «al primo tocco», cambierebbe il file prodotto.

4. **`sp_ndiags = 0` e l'invalidamento delle celle devono restare atomici rispetto al
   blocco.** Oggi lo sono per costruzione (`sp_reset_block`, `align_gpu.cu:639–651`, con
   un `__syncthreads()` prima e uno dopo). L'invariante che protegge il sito 10 è
   esattamente questa: nessun thread può leggere `sp_diags[di]` con `di < sp_ndiags`
   mentre un altro sta invalidando.

5. **Il valore che `cudaMemset` lascia (zero) non deve essere un'epoca valida.**
   `align_gpu.cu:1515` azzera tutta la `QueryState` prima del lancio; qualunque campo
   epoca partirebbe da 0 su tutte le celle. La prima epoca usata dal kernel deve quindi
   essere diversa da 0, altrimenti la memoria appena azzerata si presenta come piena di
   celle valide con `offset` 0 — che per il punto 2 è peggio del sentinella.

Un sesto punto, che non è un'invariante ma un limite di ciò che si può guadagnare: il
clear logico agisce dentro il kernel e quindi elimina il traffico del sito 2
(`align_gpu.cu:1072`), non quello della `cudaMemset` host-side di `align_gpu.cu:1515`.
Quest'ultima non compare in nessuna delle misure di Fase B, perché il profiling è
filtrato su `theseus_align_batch_kernel`.
