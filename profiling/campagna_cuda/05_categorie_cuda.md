# E — Le sei categorie CUDA

## 1. Privatization — *già sostanzialmente applicata*

Non è stata aggiunta nessuna modifica artificiale. Quello che c'è già:

| cosa | dove | livello |
|---|---|---|
| `QueryState` per query | `align_gpu.cu:1231`, `states[query_id]` | un blocco intero, in global |
| scalari di controllo del blocco | `block_score`, `block_end`, `block_end_cell`, `shared_num_active`, `shared_vertex`, `shared_range_*` | shared, 368 B statici |
| piano di sparsify | `shared_sparsify_plan`, `shared_i_ranges` | shared, costruito da thread 0 |
| staging dei candidati | `shared_i_candidates`, `shared_i_valid`, `shared_m_valid` | shared dinamica, `ntx` elementi |
| basi del prefix-sum | `shared_warp_base[kMaxWarps]` | shared, una per warp |
| candidati in costruzione | `Cell new_cell` dentro `make_i_candidate` / `make_sparsify_candidate` | registri, per thread |

Il modello «un blocco = una query» è di per sé la privatizzazione: nessuna
struttura è condivisa fra query, quindi non esiste contesa fra blocchi da
privatizzare. Dentro il blocco, i due punti che sarebbero atomiche in un design
naive — l'allocazione degli slot in `sp_diags` e in `bs_*_wf` — usano già
`block_prefix_alloc`, cioè un prefix-sum su `shared_warp_base` con
`__ballot_sync`/`__shfl`, non `atomicAdd` su global.

L'unico candidato residuo sarebbe privatizzare la ScratchPad in shared, ma non ci
sta: 52 107 diagonali × 24 B sono 1,2 MB contro i 64 KB di shared per SM. È
scritto anche nel codice, ad `align_gpu.cu:286`, come motivo per cui il merge fa
una scansione O(tile) invece di un array indicizzato per diagonale.

**Verdetto: `Privatization: already substantially implemented`**, ai punti sopra.

## 2. Thread coarsening — non implementata

È l'unica delle sei senza un esperimento. La ragione è che la campagna ha
prodotto la modifica *opposta* e l'ha misurata: l'LCP warp-cooperativo (§4) passa
da un thread per cella a un warp per cella, cioè **de-coarsening**. Il suo
risultato dice già che cosa farebbe il coarsening sui workload dove il
parallelismo per cella è abbondante, e con segno rovesciato.

I candidati restanti — più diagonali per thread nel clear, più celle per thread
in `densify` — insistono su fasi che dopo C1 valgono rispettivamente il 7,7 % e
una quota di `i/d/m` intorno al 16-20 % del loop, con 239 registri già al limite
(§6): il coarsening li aumenterebbe ancora, tagliando l'occupancy sotto il 25 %.

Resta da fare. È l'unico punto del TASK E non coperto, e non va contato come
misurato.

## 3. Coalesced accesses — due applicazioni, misurate

- `00ea1ef`, clear a parole invece che a `Cell`: traffico dimezzato, **durata
  invariata**.
- `d84fd82`, split hot/cold della ScratchPad: **kernel 1,14×–7,86×**.

Entrambe in [`02_scratchpad_hot_cold.md`](02_scratchpad_hot_cold.md), con il
modello `durata ≈ istruzioni × latenza/istruzione` che spiega perché la prima non
ha dato niente e la seconda sì.

## 4. Control divergence — LCP warp-cooperativo

Commit `cd83c70`. `extend_and_consume_m_cells` passa da un thread per cella a un
warp per cella: le 32 lane confrontano 32 caratteri, `__ballot_sync` li riduce a
una maschera e `__ffs` sul complemento dà la lunghezza della corsa iniziale di
match, cioè l'avanzamento.

Che restituisca **esattamente** ciò che restituisce `core_lcp` non è
un'affermazione: l'avanzamento è la lunghezza del prefisso di match, mai il
numero di lane che matchano, quindi un mismatch alla lane *k* ferma tutto a *k*
qualunque cosa dicano le lane successive. È stato verificato simulando le 32 lane
contro la versione seriale su **400 000 casi casuali**, con prefissi comuni
lunghi apposta per esercitare il caso multi-chunk: **0 divergenze**
(`dati_grezzi/lcp_test.cpp`). Endpoint, ordering, jump e tie-breaking sono decisi
dal chiamante a partire da `offset` e `j`, che sono identici, e il ciclo seriale
di thread 0 che segue vede le stesse celle nello stesso ordine di indice.

**Che cosa aspettarsi, dai dati che ho.** Il profiling per fase dice che `extend`
è il 29-31 % del loop sul tier complex e lo 0,2-0,6 % sul simple. Ma sul tier
complex le read hanno errori, quindi gli LCP sono corti e le celle molte: un warp
per cella spreca 31 lane e riduce il parallelismo per cella di 32×. È il caso in
cui **questa modifica dovrebbe peggiorare**.

Il tier simple è il contrario — LCP di 100 caratteri, pochissime celle — ma lì
l'LCP che conta non è questo. È quello a punteggio 0 di `align_gpu.cu:1138`, che
gira **su thread 0 da solo** e che il profiling attribuisce al «resto» del loop,
il 71-74 % sul tier simple. Quel punto sta dentro `core_extend_diagonal`, che è
`THESEUS_HD`, ricorsivo attraverso `core_store_m_jump` e condiviso con la CPU:
renderlo cooperativo vuol dire una variante device-only dell'intera catena, che è
materiale del TASK F (serializzazione su thread 0), non una micro-ottimizzazione.

**Non misurato su GPU.**

## 5. Tiling of reused data — query in shared memory

Commit `00d7536`. La query è l'unico input che ogni cella M del blocco rilegge:
`core_lcp` cammina su `query[offset..]` per ogni cella di ogni wavefront. Viene
copiata una volta per blocco in shared (`kQueryTileBytes` = 1024 B) e `align_one`
riceve quel puntatore; oltre quella soglia si continua a leggere da global, che è
il comportamento di prima, quindi il bound non deve essere generoso. Le 100 bp di
ogni dataset GGBS ci stanno con margine.

Costo: 1 KB di shared per blocco, che porta il totale da 4,5 KB a 5,5 KB a 128
thread. Non tocca l'occupancy, perché il limite sono i registri e non la shared
(§6): la shared consentirebbe 7 blocchi, i registri ne consentono 2.

**Che cosa aspettarsi.** Poco. La query sono 100 byte, letti da tutti i thread
del blocco quasi in contemporanea, quindi stanno già in L1 e i 3 933 442 load
sectors misurati su `c4_err_2k` sono già a 1,23 settori per richiesta, cioè quasi
perfettamente coalescenti. Va misurato lo stesso.

**Non misurato su GPU.**

## 6. Occupancy — l'analisi è netta, il limite sono i registri

Misure Nsight su `c4_err`, tre configurazioni, sia baseline sia C1:

| thread/blocco | registri/thread | blocchi permessi da: registri / shared / warp / hw | occupancy raggiunta |
|---:|---:|---|---:|
| 64 | 239 | **4** / 12 / 16 / 16 | 22,98 % |
| 128 | 239 | **2** / 7 / 8 / 16 | 24,93 % |
| 256 | 239 | **1** / 3 / 4 / 16 | 25,00 % |

Il limite è **sempre** il file dei registri, e il conto torna esattamente: la T4
ha 65 536 registri per SM, 239 × 32 = 7 648 per warp, 65 536 / 7 648 = 8,57 →
**8 warp per SM su 32 possibili = 25 %**. Ecco perché l'occupancy è identica a
64, 128 e 256 thread: cambia quanti blocchi ci stanno, non quanti warp.

Le soglie da superare per guadagnare qualcosa sono quindi precise:

| registri/thread | warp/SM | occupancy |
|---:|---:|---:|
| 239 (oggi) | 8 | 25 % |
| ≤ 170 | 12 | 37,5 % |
| ≤ 128 | 16 | 50 % |
| ≤ 96 | 21 | 65,6 % |

**Da dove vengono 239 registri.** Le cause candidate, in ordine di sospetto:

1. **Inlining aggressivo di una catena lunga.** `align_one` inlina
   `compute_new_wave` → `process_vertex` → `generate_and_merge_i_candidates`,
   `run_sparsify_plan`, `densify` ×2, `extend_and_consume_m_cells`. Sono fasi
   disgiunte nel tempo ma il compilatore le vede come un corpo unico, e i
   temporanei di una fase restano vivi attraverso l'altra.
2. **`Cell` costa 6 registri.** È 24 byte, e ce ne sono molte simultanee vive:
   `new_cell`, `curr_cell`, `prev_cell`, `end_cell`, `block_end_cell`, `value`
   dentro `densify`.
3. **`Frame stack[kMaxIJumpStack]`** in `core_store_i_jump`: array locale di 8
   frame da 56 B = 448 B. Già ridotto da 256 frame (14 KB, che spillavano), ma un
   array locale indicizzato dinamicamente vive in local memory e i suoi indirizzi
   occupano registri.
4. **Live range lunghi attraverso le barriere.** `range_start`/`range_end`,
   i puntatori shared, `score`, `v`, `query_len` attraversano l'intero
   `process_vertex`.

**Che cosa non è stato fatto.** Né la riduzione naturale del live set, né
l'esperimento `-maxrregcount`. Quest'ultimo è pronto come misura e costa una
build: `-maxrregcount=128` porta a 16 warp/SM e mostra il trade-off fra occupancy
guadagnata e spill; il TASK dice giustamente di usarlo solo come sonda e non come
soluzione, ed è così che va letto.

Va detto anche perché non è la prima leva. Il modello di §3 dice che la durata è
`istruzioni × latenza per istruzione`: raddoppiare i warp residenti aiuta il
secondo fattore, ma C1 ha già tagliato il primo di 3× e la latenza per istruzione
di 2×, e lo ha fatto senza toccare l'occupancy, rimasta a 24,7 %. L'occupancy è
il tetto residuo, non il collo di bottiglia che è stato appena rimosso.

## Stato delle misure

| categoria | implementata | correttezza CPU | correttezza GPU | misurata su GPU |
|---|---|---|---|---|
| 1 privatization | n/a (analisi) | — | — | — |
| 2 coarsening | **no** | — | — | — |
| 3 coalescing | sì ×2 | 10/10 | 30/30 | **sì** |
| 4 divergence | sì | 5/5 ctest | no | no |
| 5 tiling | sì | 5/5 ctest | no | no |
| 6 occupancy | analisi + sonda pronta | — | — | parziale (contatori raccolti) |

Le categorie 4 e 5 hanno commit isolati e compilano solo sotto `nvcc`, che su
questa macchina non c'è: la sessione Colab è caduta e il servizio ha poi smesso
di assegnare T4. **Non sono state compilate né validate su GPU**, e finché non lo
saranno vanno considerate scritte, non verificate.
