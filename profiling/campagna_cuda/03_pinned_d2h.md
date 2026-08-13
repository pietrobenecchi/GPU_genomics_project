# B — Buffer host pinned persistenti per la D2H

Ottimizzazione specifica dell'architettura, non una delle sei categorie.

- baseline: `d84fd82` (C1)
- ottimizzazione: `026b457` "Buffer host pinned persistenti"
- stessa VM Colab / T4 della misura C1, stessa sessione

## 1. La modifica

I tre buffer host in cui `align_batch` scrive (`CompactTracebackState[]`,
`AlignResult[]`, `int32_t[]`) erano `std::vector` costruiti a ogni batch. Ora
sono allocati **una volta con `cudaHostAlloc`** e tenuti per la vita
dell'aligner, cresciuti solo quando un batch ne chiede più dell'ultimo.

Il formato logico non cambia: sono gli stessi array, con gli stessi tipi, letti
dallo stesso `TracebackWavefronts`. Cambia solo dove stanno.

Due dettagli di correttezza:

- se `cudaHostAlloc` fallisce si ricade sui `std::vector` di prima, quindi il
  batch gira lo stesso;
- il buffer delle lunghezze viene riempito di `-1` prima di ogni batch. È il
  sentinella su cui si regge il controllo del layout dell'upload, e con un
  buffer riusato una lunghezza rimasta dal batch precedente lo avrebbe superato
  senza che il kernel avesse scritto niente.

Sul lato senza CUDA le due funzioni sono `malloc`/`free`, così il build con lo
stub non ha bisogno di un secondo percorso.

## 2. Correttezza

Regressione GPU **30/30** (10 dataset × 64/128/256), byte-identica ai golden,
sempre con `--require-gpu-result`. CTest 5/5. Path CPU: 7/7 golden.

## 3. Tempo di D2H

Payload trasferito: 288 KB per query (`sizeof(CompactTracebackState)` = 294 936 B).

| dataset | query | payload | D2H C1 | D2H pinned | fattore |
|---|---:|---:|---:|---:|---:|
| `c4_err_2k` | 2048 | 576 MiB | 407,1 ms | **46,7 ms** | 8,7× |
| `ebola_err_2k` | 2048 | 576 MiB | 402,7 ms | **46,7 ms** | 8,6× |
| `c4_exact_2k` | 2048 | 576 MiB | 409,6 ms | **46,7 ms** | 8,8× |
| `c4_err_1k` | 1024 | 288 MiB | 247,8 ms | **23,4 ms** | 10,6× |
| `c4_exact_1k` | 1024 | 288 MiB | 200,3 ms | **23,4 ms** | 8,6× |
| `c4_err` | 512 | 144 MiB | 102,5 ms | **11,8 ms** | 8,7× |
| `c4_exact` | 512 | 144 MiB | 100,8 ms | **11,8 ms** | 8,5× |
| `ebola_error_smoke` | 256 | 72 MiB | 51,5 ms | **6,0 ms** | 8,5× |
| `ebola_exact_smoke` | 256 | 72 MiB | 50,1 ms | **6,1 ms** | 8,2× |

In banda:

```
pageable   576 MiB / 407,1 ms  =  1,4 GiB/s
pinned     576 MiB /  46,7 ms  = 12,3 GiB/s
```

12,3 GiB/s è la banda che ci si aspetta da un PCIe 3.0 x16, cioè la copia ora va
a velocità di bus. 1,4 GiB/s era il costo di far passare ogni pagina per il
buffer di staging del driver, più il page fault di prima scrittura su
un'allocazione mai toccata. Il fattore è **costante su tutte le taglie**, come
deve essere se la causa è per-pagina.

Il tempo di kernel è invariato (entro il rumore), come atteso: la modifica non
tocca il device.

## 4. Il tempo di wall, e il rovescio della medaglia

| dataset | wall C1 | wall pinned | Δ |
|---|---:|---:|---:|
| `c4_err_2k` | 824,9 ms | 735,3 ms | **−89,6** |
| `ebola_err_2k` | 819,4 ms | 733,9 ms | **−85,5** |
| `c4_err_1k` | 630,1 ms | 517,9 ms | **−112,2** |
| `c4_exact_1k` | 557,3 ms | 515,9 ms | −41,4 |
| `c4_err` | 394,2 ms | 366,5 ms | −27,7 |
| `c4_exact` | 394,9 ms | 363,1 ms | −31,8 |
| `ebola_exact_smoke` | 315,5 ms | 296,3 ms | −19,2 |
| `c4_exact_2k` | 829,3 ms | **894,0 ms** | **+64,7** |
| `ebola_exact_2k` | 813,8 ms | **901,2 ms** | **+87,4** |
| `ebola_error_smoke` | 318,5 ms | 354,2 ms | +35,7 |

La D2H risparmia ~360 ms sui LARGE, ma il wall ne guadagna solo ~87, e su due
dataset su dieci **peggiora**. Il candidato è il costo di `cudaHostAlloc`: bloccare
in memoria 576 MiB non è gratis, e in questa applicazione si paga **una volta per
processo e non si ammortizza mai**, perché `seq2graph_proxy` allinea esattamente
un batch e poi esce. Il buffer persistente è progettato per un chiamante che fa
molti batch; la CLI non è quel chiamante.

**Questo punto non è ancora misurato.** La sessione Colab è caduta prima che
scaricassi il JSON con `host_buffers_ms`, che è cronometrato proprio attorno alla
chiamata che alloca. Finché non c'è quel numero, l'attribuzione qui sopra resta
un'ipotesi coerente con i dati e non un fatto: il wall su questa VM oscilla di
±130 ms fra run dello stesso punto, quindi i due segni positivi potrebbero anche
essere solo rumore.

Da misurare alla prossima sessione:

1. `host_buffers_ms` per taglia di batch, cioè il costo del page lock;
2. lo stesso confronto con due batch nello stesso processo, dove l'allocazione si
   ammortizza.

## 5. Verdetto provvisorio

**Tenuta.** La D2H va 8,2–10,6× più veloce e raggiunge la banda del bus, la
correttezza è intatta e il fallback copre il caso in cui il page lock non
riesca. Il beneficio sul wall della CLI è reale ma parziale (−10 % sui LARGE
complex) e va confermato separando il costo di allocazione, che è l'unica cosa
che questa modifica può aver peggiorato.
