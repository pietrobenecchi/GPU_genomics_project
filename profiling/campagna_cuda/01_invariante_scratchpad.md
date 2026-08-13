# L'invariante della ScratchPad, riverificato sul codice

Prerequisito del TASK C: prima di toccare il layout bisogna riconfermare che
l'attività di una cella dipende **solo** da `offset == -1` e che gli altri campi
di una cella inattiva non vengono mai letti. Rifatto sul codice corrente
(commit `00ea1ef`), non sulla memoria della sessione precedente.

## 1. Chi legge una cella *potenzialmente inattiva*

Quattro punti in tutto, e leggono tutti e soli `.offset`:

| punto | espressione |
|---|---|
| `query_state.h:423` (`sp_access_alloc`) | `qs.sp_wf[idx].offset == -1` |
| `query_state.h:440` (`sp_reset_one`) | scrive `.offset = -1` |
| `align_gpu.cu:344` (`merge_candidate_tile`) | `qs.sp_wf[idx].offset == -1` |
| `align_gpu.cu:367` (`merge_candidate_tile`) | `qs.sp_wf[idx].offset < my_off` |

## 2. Chi legge una cella *intera*

Tre punti CPU (`theseus_aligner_impl.cpp:391, 435, 477`) e uno GPU
(`align_gpu.cu:595`), tutti nella stessa forma:

```cpp
for (di = 0; di < ndiags; ++di) {
    const int32_t diag = qs.sp_diags[di];
    ...
    value = sp_at(qs, diag);          // lettura dei 24 byte
}
```

Iterano su `sp_diags`, cioè sull'elenco delle diagonali che `sp_access_alloc` ha
*appeso al primo tocco*. Una diagonale entra in quell'elenco solo passando per la
riga 423, e chi la tocca poi ci scrive una `Cell` intera (`cell = new_cell`).
Quindi le letture da 24 byte cadono solo su celle scritte per intero.

## 3. La prova che chiude la questione

Non serve fidarsi dell'enumerazione: `sp_reset_one` è già oggi

```cpp
qs.sp_wf[qs.sp_diags[i] - qs.sp_min_diag].offset = -1;
```

Scrive **solo `.offset`** e lascia gli altri cinque campi come stanno. Il reset
gira fra un punteggio e l'altro (`align_gpu.cu:883, 903, 922`), quindi
**già nell'implementazione attuale — quella validata byte-identica ai golden —
ogni cella inattiva dopo il primo reset contiene payload stantio.** Se qualcuno
leggesse quei campi, l'output divergerebbe già adesso.

Ne segue che i cinque sesti di parole scritte dal clear iniziale sono spreco
dimostrato, non presunto.

## 4. Cosa cambia e cosa no

Il clear lineare a inizio query (`align_gpu.cu:1107`) scrive `span × 6` parole.
Serve che ne scriva `span × 1`, e la nota di sessione dice già perché non basta
scriverne una su sei in loco: a passo 24 byte lo store parziale costa il
read-modify-write dei settori. Da cui lo split del TASK C1: un array denso
`int32_t sp_off[kScratchpadSpan]` che diventa l'autorità sull'attività, con
`sp_wf` che resta il payload e **non viene più azzerato**.

Due punti da tenere d'occhio nell'implementazione:

- **tutti i mutatori passano per l'assegnazione di una `Cell` intera.** I sei
  siti (`align_core.h:58,83,104` e `theseus_aligner_impl.cpp:275,312,346`) hanno
  la stessa forma `cell = new_cell` sotto la guardia `cell.offset < new_cell.offset`,
  quindi si possono raccogliere in un'unica primitiva che aggiorna payload e
  `sp_off` insieme. Non esistono mutazioni parziali da inseguire.
- **`sp_wf` non inizializzato non è memoria letta non inizializzata.** Il
  workspace è persistente e `sp_wf` non verrà più azzerato, ma per la §2 nessuna
  lettura cade su una cella mai scritta. `compute-sanitizer --tool initcheck` è
  il controllo che lo verifica invece di assumerlo.
