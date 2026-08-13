# D — Clear pigro della ScratchPad

Ottimizzazione specifica dell'architettura. Il TASK chiedeva di studiare uno
schema **generation/epoch** al posto del clear lineare. La risposta è che l'epoch
non serve, perché l'invariante vero del codice è più forte di quello che l'epoch
andrebbe a stabilire — ma che il guadagno, sull'uso attuale del programma, è
strutturalmente **zero**. Entrambe le cose vanno dette per intero.

- commit: `0a303ce` "Clear pigro della ScratchPad"
- baseline: `d9f813f`

## 1. Le verifiche che il TASK chiedeva

### La ScratchPad si ripulisce da sola

Ogni scrittura di una cella passa per uno di due punti, e in tutti e due la
diagonale finisce in `sp_diags`:

| punto | appende | scrive |
|---|---|---|
| `sp_merge_candidate` (CPU) | se `sp_off[idx] == -1` | se `sp_off[idx] < c.offset` |
| `merge_candidate_tile` (GPU, `align_gpu.cu:344/368`) | `is_first && sp_off[idx] == -1` | `is_winner && sp_off[idx] < my_off` |

Se la cella era inattiva (`-1`) e l'offset del candidato è ≥ 0 — e gli offset
sono posizioni nella query, mai negative, cosa già verificata da un `cap_fail`
apposta — allora **appendere e scrivere avvengono insieme**. Se era già attiva,
la diagonale è già nella lista.

E `sp_reset_block` percorre `sp_diags` rimettendo ogni cella a `-1`. In
`process_vertex` ce ne sono tre (`align_gpu.cu:885, 905, 924`), una dopo ciascuna
delle tre fasi che fondono candidati, e **l'ultima operazione che
`process_vertex` fa sulla ScratchPad è un reset**: dopo di essa resta solo
`extend_and_consume_m_cells`, che non la tocca. `compute_new_wave` non ha uscite
anticipate dal ciclo sui vertici, e `core_extend_diagonal` a punteggio 0 non
tocca la ScratchPad.

Quindi **alla fine di ogni query ogni entry toccata è di nuovo `-1` e
`sp_ndiags` è 0**. Un prefisso azzerato una volta resta azzerato.

### Riuso degli slot, lifetime fra score, wraparound, determinismo

- **reset intra-alignment**: già sparso, `O(ndiags)`, invariato.
- **lifetime fra score**: coperto da quanto sopra.
- **wraparound**: non c'è contatore, quindi non c'è. È il vantaggio principale
  rispetto all'epoch, che avrebbe avuto un contatore a 32 bit da sorvegliare.
- **accessi che assumono `offset == -1`**: sono i quattro di
  [`01_invariante_scratchpad.md`](01_invariante_scratchpad.md), tutti su `sp_off`.
- **determinismo**: nessun valore cambia, cambia solo quante volte si riscrive
  lo stesso `-1`.

### L'unico caso in cui l'invariante si rompe

Oltre `kScratchpadSpan` diagonali, `merge_candidate_tile` lascia cadere l'append
(`align_gpu.cu:354`, `cap_fail(kCapScratchpadDiags)`) ma la cella vincente viene
scritta lo stesso. Lì una cella resta attiva senza che nulla la resetti.

Il risultato di quella query è comunque scartato, ma lo **stato** no: la query
successiva sullo stesso slot erediterebbe la sporcizia. `align_one` chiude quindi
con `if (qs.capacity_exceeded) qs.sp_cleared = 0;`, che obbliga la prossima a
riazzerare tutto.

## 2. La modifica

Un campo, `int32_t sp_cleared`: quante entry di `sp_off` sono già state messe a
`-1`, contate da 0. `sp_init` azzera solo `[sp_cleared, span)` e alza
`sp_cleared`. Niente epoch, niente array in più, **nessun byte in più per query**
oltre ai 4 del contatore.

Serve un solo appiglio: `sp_cleared` deve leggere 0 al primo uso di uno stato, e
`cudaMalloc` non azzera. Da qui un `cudaMemset2D` di un `int` per stato — 8 KB
per 2048 — fatto **una volta per allocazione del workspace**, non per batch.

## 3. Perché su questo programma non serve a niente

`align_batch` lancia un blocco per query e dà alla query *i* lo stato *i*. La CLI
allinea **un batch e poi esce**. Quindi ogni `QueryState` serve esattamente una
query per processo, e «azzerare una volta per slot» e «azzerare una volta per
query» sono la stessa frase.

Il clear pigro paga solo se uno slot vede più di una query, cioè:

- più batch nello stesso processo (un server, o il chunking di un run grande —
  che oggi non c'è, vedi [`00_workload.md`](00_workload.md) §3);
- il path CPU, che riusa **una sola** `QueryState` per tutte le query in
  sequenza.

Il secondo caso non è ipotetico ed è la verifica più severa disponibile: sul path
CPU, con `c4_err_2k`, un unico stato serve 2048 query consecutive e **solo la
prima azzera qualcosa**. Tutti e dieci i golden passano byte per byte. Se
l'invariante di auto-pulizia fosse falso, quell'output divergerebbe.

Da qui la flag `--repeat` (`d9f813f`), che allinea lo stesso batch N volte con lo
stesso aligner: è ciò che rende misurabile sia questo sia l'ammortamento del page
lock del TASK B.

## 4. Stato delle misure

**Correttezza CPU: 10/10 golden byte-identici**, incluso il caso di riuso a 2048
query su un solo stato.

**Le misure GPU non sono state prese.** La sessione Colab è caduta due volte e
poi il servizio ha smesso di assegnare T4 (`Service Unavailable`). Restano da
fare, e sono già scriptate:

1. regressione 30/30 su `0a303ce`;
2. `--repeat 5` su `prelazy` contro `lazy`, per far vedere che dalla seconda
   iterazione il kernel del tier simple perde la quota di clear misurata in
   [`02_scratchpad_hot_cold.md`](02_scratchpad_hot_cold.md) §5 (57,7 % su
   `c4_exact_2k`);
3. `compute-sanitizer --tool initcheck`, che qui serve due volte: per il clear
   pigro e per la rimozione del memset per batch (`f1c93ea`).

## 5. Verdetto provvisorio

**Corretto, tenuto, ma senza effetto sull'uso attuale.** È il risultato più
interessante del TASK: il clear lineare non è eliminabile perché lo schema di
azzeramento è sbagliato, ma perché **ogni slot serve una query sola**. Le due vie
d'uscita vere sono riusare gli slot (chunking o processo persistente) o rendere
la ScratchPad sparsa, cioè togliere del tutto la dipendenza da `span` — la
«ScratchPad sparsa» che `CLAUDE.md` già prevede per i grafi con vertici lunghi.
