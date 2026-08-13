# Workload della campagna

Censimento degli input già presenti nel repository e definizione della matrice
SMALL / MEDIUM / LARGE. Nessun contenuto è stato inventato o modificato: i
workload nuovi sono gli **stessi record GGBS** dei workload storici, convertiti
con un `--limit` più alto.

## 1. Grafi disponibili

Misurati direttamente dai GFA (`S` = segmenti, `L` = link):

| grafo | file | vertici | archi | bp totali | vertice più lungo |
|---|---|---:|---:|---:|---:|
| ebola | `data/validation/ggbs/graphs/ebola.gfa` | 7 | 8 | 18 925 | 9 063 |
| c4 | `data/validation/ggbs/graphs/c4.gfa` | 16 | 22 | 164 832 | 52 006 |
| covid | `external_datasets/.../b_cov.gfa` | 39 253 | 95 440 | 51 776 | 32 |
| yeast | `external_datasets/.../d_yeast.gfa` | 49 410 | 67 320 | 762 651 | 5 184 |
| MHC | `external_datasets/.../f_MHC-57.gfa` | 980 | 1 399 | 5 181 373 | 209 326 |

Solo ebola e c4 sono utilizzabili con la build corrente. Gli altri tre sfondano
un bound compilato in `query_state.h`, e vale la pena essere precisi su quale,
perché è l'informazione che dice quanto costerebbe aggiungerli:

- **covid** e **yeast** superano `kMaxVertices = 2048` (39 253 e 49 410
  vertici). Il costo è `vd_vertex_to_idx[kMaxVertices]`, un `int32` per vertice
  **per query**: portarlo a 40 960 aggiunge 156 KB alla QueryState, che passa da
  4,20 a 4,35 MB.
- **MHC** supera `kScratchpadSpan = 52 224`: gli servirebbe
  209 326 + 100 + 1 = 209 427 diagonali, cioè una ScratchPad 4 volte più grande
  (`sp_wf` da 1,20 a 4,79 MB, `sp_diags` da 0,20 a 0,80 MB) e una QueryState
  intorno ai 8,4 MB. È il caso che la nota in `CLAUDE.md` prevede quando dice
  che «un grafo con vertici più lunghi richiede o un valore più grande o una
  ScratchPad sparsa».

Nessuno dei tre è entrato nella matrice: sono costanti di compilazione, quindi
cambiarle sposta la baseline di *tutti* i workload e renderebbe incomparabili le
misure prima/dopo di ogni ottimizzazione. Covid resta però il candidato più
interessante come esperimento a sé (vedi §5).

## 2. Read disponibili

I JSON GGBS contengono **10 000 read ciascuno** (4 000 per covid), non 512:

| sorgente | read | bp | lunghezza |
|---|---:|---:|---:|
| `a_ebola_0M_reads[_err].json` | 10 000 | 1 000 000 | 100 |
| `e_C490_reads[_err].json` | 10 000 | 1 000 000 | 100 |
| `b_cov_reads[_err]4k_50.json` | 4 000 | 200 000 | 50 |
| `d_lievito`, `c_Ecoli`, `f_MHC-57` | 10 000 | 1 000 000 | 100 |

I workload storici ne usavano 256 (ebola) e 512 (c4). **Non serve costruire un
workload sintetico**: basta convertire più record dello stesso file.

## 3. Il tetto vero: la memoria del device

`align_batch` lancia **un blocco per query in un solo launch**, senza chunking, e
alloca `sizeof(QueryState) * num_seqs` sul device. La QueryState misurata è di
4 407 560 B = **4,203 MB**, così ripartiti:

| blocco | byte | quota |
|---|---:|---:|
| VerticesData | 1 863 720 | 42,3 % |
| ScratchPad (`sp_wf` + `sp_diags`) | 1 462 312 | 33,2 % |
| Scope | 786 600 | 17,8 % |
| BeyondScope | 294 928 | 6,7 % |

Su una T4 da 16 GB questo fissa il batch massimo praticabile intorno alle
**3 400 query**. Le 10 000 read disponibili richiederebbero 42 GB, quindi il
LARGE è stato fermato a **2048 query = 8,6 GB**, che lascia margine anche dopo
le ottimizzazioni che fanno crescere la QueryState (lo split della ScratchPad la
porta a ~4,4 MB, cioè 9,0 GB a 2048).

Il chunking del batch non è stato introdotto: cambierebbe la struttura di ciò
che si sta misurando (un launch diventa N launch) proprio mentre si confrontano
kernel. Il tetto è un risultato da riportare, non un ostacolo da aggirare.

## 4. La matrice

| classe | dataset | query | bp | grafo | span ScratchPad | memoria device |
|---|---|---:|---:|---|---:|---:|
| SMALL | `ebola_exact_smoke`, `ebola_error_smoke` | 256 | 25 600 | ebola | 9 164 | 1,08 GB |
| SMALL | `c4_exact`, `c4_err` | 512 | 51 200 | c4 | 52 107 | 2,15 GB |
| MEDIUM | `c4_exact_1k`, `c4_err_1k` | 1 024 | 102 400 | c4 | 52 107 | 4,30 GB |
| LARGE | `c4_exact_2k`, `c4_err_2k` | 2 048 | 204 800 | c4 | 52 107 | 8,61 GB |
| LARGE | `ebola_exact_2k`, `ebola_err_2k` | 2 048 | 204 800 | ebola | 9 164 | 8,61 GB |

Lo *span* è `vertice più lungo + lunghezza query + 1` ed è la dimensione del
clear della ScratchPad, cioè della fase che il profiling precedente dà al
44–71 % del kernel. Le due coppie LARGE hanno lo stesso numero di query e lo
stesso numero di basi ma **span diverso di 5,7×**: è il contrasto che serve per
capire se un'ottimizzazione agisce sul clear o sul loop.

Le due dimensioni restano indipendenti:

- `--suite simple` / `--suite complex` continuano a significare esattamente i
  quattro workload congelati (score 0 / score non nullo), come in `CLAUDE.md`;
- `--suite small|medium|large` seleziona per taglia.

## 5. Come sono stati costruiti (e perché non sono "performance-only")

```
python3 scripts/ggbs_json_to_theseus_queries.py \
    --json external_datasets/GGBS/input_data/JSON/json_files/e_C490_reads_err.json \
    --graph theseus_gpu/data/validation/ggbs/graphs/c4.gfa \
    --queries theseus_gpu/data/validation/ggbs/queries/c4_err_2k.queries \
    --metadata theseus_gpu/data/validation/ggbs/truth/c4_err_2k.metadata.json \
    --limit 2048 --overwrite
```

`--limit N` prende i **primi N record nell'ordine del file**, senza campionamento.
Da qui due proprietà verificate, non assunte:

1. ogni set piccolo è un **prefisso byte-esatto** del suo omologo grande
   (`head -1024 c4_err_2k.queries | cmp - c4_err.queries` passa per tutte e
   cinque le coppie);
2. lo stesso vale per i golden: i primi 512 risultati di `c4_err_2k.cpu.gaf`
   sono identici a `c4_err.cpu.gaf`.

La (2) è un controllo forte: l'oracolo in `cpu_oracle/`, rieseguito oggi, ha
riprodotto esattamente i golden congelati a suo tempo sul prefisso condiviso.

Ogni set nuovo ha quindi il **suo golden CPU congelato** dallo stesso oracolo
(`oracle 56663af`), con manifest e SHA-256. Non sono *performance-only workload*:
sono workload di correttezza a tutti gli effetti, e la matrice di correttezza può
girarci sopra. I quattro golden storici restano comunque l'oracolo primario.

L'oracolo CPU allinea 2048 read c4_err in 5,2 ms.

## 6. Metriche da raccogliere per ogni workload

queries, input size, graph size, wall, `align_batch`, kernel, H2D, D2H,
throughput query/s, e la ripartizione per fase (clear ScratchPad, I, D, M,
extend) con la strumentazione `clock64()` già usata per il commit `00ea1ef`.

Warm-up + 3 run stabili, clock bloccati a 1590 MHz, `ncu --clock-control none`.
Tutte le misure di questa campagna vengono da **una sola VM Colab con T4**:
confronti fra run di VM diverse sono segnalati esplicitamente dove capitano.

## 7. Covid come esperimento separato

Covid è l'unico input che rovescia il profilo: span 83 contro 52 107, quindi il
clear della ScratchPad diventa trascurabile, mentre i 39 253 vertici rendono
dominante il clear di `vd_vertex_to_idx` e i 95 440 archi stressano il fan-out e
la frontiera dei vertici attivi. È il controcampo naturale di questa campagna,
ma richiede `kMaxVertices ≥ 40 960`, cioè una build diversa. Va trattato come
esperimento a sé, con la sua baseline, non come una riga della matrice qui sopra.
