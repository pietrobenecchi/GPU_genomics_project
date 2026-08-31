# covid e yeast: i dati sono qui, il codice per eseguirli no

Questi quattro dataset — `covid_exact`, `covid_err`, `yeast_exact`, `yeast_err`,
più i sottoinsiemi `_64`, `_7` e `_380` — sono convertiti da GGBS e pronti, con
i golden CPU congelati e i manifest che ne registrano la provenienza.

**Su questa branch non girano.** Il kernel muore con `illegal memory access` e
l'aligner CPU con un segfault. Non è un difetto dei dati: sono cinque bound del
codice tarati sui due grafi giocattolo, e i due nuovi grafi li superano tutti.

## Perché i grafi giocattolo non lo mostravano

| grafo | nodi | archi | nodi che ramificano | vertice più lungo |
|---|---:|---:|---:|---:|
| ebola | 7 | 8 | 2 | 9.068 |
| c4 | 16 | 22 | 4 | 52.006 |
| **covid** | **39.253** | **95.440** | **24.089** | 32 |
| **yeast** | **49.410** | **67.320** | **17.713** | 5.184 |

covid ramifica in 24.089 punti contro i 4 di c4. Tutto quello che si rompe si
rompe sulla ramificazione, non sulla dimensione: covid e yeast hanno vertici
*più corti*, quindi la ScratchPad — che è dimensionata sulla lunghezza del
vertice — costa meno, non di più.

## I cinque bound, e di quanto sono superati

| bound | valore qui | serve | note |
|---|---:|---:|---|
| stack del device per thread | 1.024 B | ~19.000 B | la ricorsione M-jump costa 264 B/livello e arriva a 72 su yeast |
| `kMaxVertices` | 2.048 | 98.820 | è lo spazio degli **id**, che è il doppio dei nodi (due orientazioni) |
| `kMaxJumpsPerScore` | 32 | 33 | covid, query 6 |
| `kMaxActiveVertices` | 256 | 257 | covid, query 15 |
| guardia su `vd_get_id` | assente | — | legge fuori array e restituisce un indice negativo |

Le correzioni stanno sulla branch **`covid-yeast`**, otto commit. Sono tenute
fuori da `main` perché alzare `kMaxActiveVertices` porta la `QueryState` da 3,95
a 8,30 MB **per ogni dataset**, anche ebola che di vertici attivi ne tocca
quattro. Misurando l'occupazione reale, due terzi di quella crescita sono
spreco: `vd_*_jumps_pos` occupa 4 MB per contenere un picco di 2 jump. La strada
giusta è rendere sparsa la VerticesData, non alzare i bound; finché non è fatto,
il costo non va su `main`.

## Come rieseguirli

```bash
git checkout covid-yeast
cd theseus_gpu && cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release \
    -DTHESEUS_PROXY_ENABLE_CUDA=ON && cmake --build build-gpu -j
python3 ../scripts/run_ggbs_gpu_regression.py --dataset yeast_exact \
    --build-dir theseus_gpu/build-gpu --require-device
```

Su T4, `yeast_exact` e `yeast_err` (2048 query ciascuno) passano sul device a
64, 128 e 256 thread con GAF byte-identico alla CPU. covid è validato a 64
query: il riferimento CPU non finisce 256 query in dieci minuti su quel grafo,
che è un costo dell'algoritmo e non un difetto.
