# Profili NVIDIA del porting GPU di Theseus

Otto campagne di profiling su **Tesla T4** (CC 7.5), una per implementazione
del kernel `theseus_align_batch_kernel`, dalla versione naive a quella
ottimizzata. Tutti i file di questa cartella sono **output diretto di `ncu`
(Nsight Compute) e `nsys` (Nsight Systems)**, salvati come sono usciti: nessuna
rielaborazione. L'unica eccezione è questo README.

Ambiente: driver 580.82.07, CUDA 12.8.93, Nsight Compute 2025.1.1.0, Nsight
Systems 2025.6.3, su VM Google Colab.

## Come è organizzata

Una cartella per implementazione, numerata in ordine cronologico e nominata con
l'ottimizzazione introdotta e il commit che la contiene:

```
00-riferimento-pre-ottimizzazione/          termine di paragone della campagna ScratchPad
01-baseline-naive-88ddef2/                  la prima versione GPU funzionante
02-scratchpad-offset-denso-d84fd82/         una parola per diagonale invece di sei
03-query-in-shared-e71a0e1/                 la query staged in shared memory
04-registri-noinline-7d2ce8a/               __noinline__ in cima alla catena
05-azzeramento-per-allocazione-7053193/     il memset una volta per allocazione
06-coarsening-del-clear-02127db/            quattro parole per thread, int4
07-densify-vuota-23dc718/                   densify senza barriere sul caso vuoto
```

Ogni commit è raggiungibile da `main`: `git checkout <sha>` ricostruisce la
versione profilata.

Dentro ogni cartella:

- `nsight-compute/` — report `.ncu-rep` completi, da aprire con `ncu-ui` o
  `ncu --import <file> --page details`
- `nsight-compute-lineinfo/` — gli stessi con `-lineinfo`, per attribuire i
  contatori alla riga di sorgente
- `nsight-systems/` — timeline `.nsys-rep`, da aprire con `nsys-ui` o
  `nsys stats <file>`
- `metriche/` — export CSV di `ncu` a metriche mirate
- `testo/` — il riepilogo testuale di `ncu`

Solo la campagna `01` ha i `.ncu-rep` completi (40 report: 4 dataset × 64/128/256
thread × 3 ripetizioni). Le altre hanno export CSV a metriche mirate, che è
quello che serviva per rispondere alle domande specifiche di ogni passo.

## Che cosa ha detto il profiling, e che cosa è cambiato

### 01 — La baseline naive (`88ddef2`)

La prima misura ha spostato l'attenzione dove non ce l'aveva nessuno. Il kernel
non era il problema:

| voce | costo | quota del tempo GPU |
|---|---|---|
| copia D2H dell'array `QueryState` | — | **59–77 %** |
| `cudaMalloc` della `QueryState` | 136–169 ms | 21–40 % |
| `cudaMemset` prima del kernel | 4,85–9,67 ms | più del kernel su 3 dataset su 4 |
| il kernel vero e proprio | — | il resto |

Dentro il kernel, due colli di bottiglia:

1. **Il clear della ScratchPad** vale il **44–71 % dei cicli** per blocco, e
   scrive sei parole per diagonale quando ne basta una.
2. **239 registri per thread** inchiodano l'occupancy al **25 %** — 8 warp per
   SM su 32 — in tutte e tre le configurazioni di thread. Il report chiudeva
   indicando `__launch_bounds__` come «l'esperimento da una riga» mai provato,
   col sospetto che reintroducesse spill.

### 02 — ScratchPad a offset denso (`d84fd82`)

Il clear scriveva `Cell` intere, sei parole, quattro delle quali il sentinella
`-1` di campi che nessuno legge finché la cella è inattiva. L'offset è stato
spostato in un array denso suo, ed è l'unica cosa che il clear deve stabilire.

Prima di arrivarci, un passo intermedio ha reso il clear coalescente scrivendo
parole invece di struct: i settori di store sono passati da **61,2 M a 25,1 M**
(da 22,3 a 4,8 per richiesta) e il traffico DRAM si è dimezzato — **senza che la
durata del kernel cambiasse**. È il risultato più istruttivo della campagna: il
kernel non è limitato dalla banda, quindi dimezzare il traffico non lo accelera.
Quello che paga è scrivere di meno, non scrivere meglio.

### 03 — Query in shared memory (`e71a0e1`)

La query è l'unico input che ogni cella M rilegge: `core_lcp` la percorre per
ogni cella di ogni wavefront. Viene copiata una volta per blocco in shared
memory (1 KB, con fallback su global per query più lunghe).

### 04 — Registri con `__noinline__` (`7d2ce8a`)

Un `__noinline__` su `process_vertex`, in cima alla catena di chiamate, porta i
registri da **239 a 138**: l'occupancy passa dal 25 % al 43 %. È il singolo
cambiamento più efficace di tutta la campagna, ed è una riga.

### 05–07 — Il clear pigro e i suoi affinamenti (`7053193`, `02127db`, `23dc718`)

`sp_cleared` traccia il prefisso già azzerato, così il clear non ripete lavoro
fra un punteggio e l'altro; il `cudaMemset` dell'array `QueryState` passa da una
volta per batch a una volta per *allocazione*, e sparisce dal costo per batch.

Il regime in cui questo si vede è `--repeat`, dove un processo serve più batch
di fila invece di uno solo e poi uscire. Kernel in ms, 128 thread, mediana:

| build | `c4_err_2k` | vs partenza | `c4_exact` | vs partenza |
|---|---:|---:|---:|---:|
| `d9f813f` (prima del clear pigro) | 4,18 | — | 0,89 | — |
| `41bb8b1` (clear pigro) | 3,48 | 1,20× | 0,176 | **5,1×** |
| `e71a0e1` (query in shared) | 2,79 | 1,50× | 0,170 | 5,2× |
| `7d2ce8a` (`__noinline__`) | 2,31 | **1,81×** | 0,153 | **5,8×** |
| `7053193` (azzeramento per allocazione) | 2,33 | 1,80× | 0,151 | 5,9× |

Sul tier `exact` il clear pigro toglie da solo il 57,7 % del kernel — esattamente
la quota che il profiling gli attribuiva.

## Le ottimizzazioni fuori dal kernel

Il profiling della baseline diceva che il grosso non era nel kernel, ed è lì che
sono arrivati i guadagni più grandi, misurati con `nsys` e con i contatori
dell'applicazione anziché con `ncu`:

- **Buffer host pinned** (`026b457`): i buffer di destinazione allocati una volta
  con `cudaHostAlloc` invece che a ogni batch. La D2H passa da 1,4 a **12,3
  GiB/s**, la banda del bus: **8,2–10,6×**.
- **Traceback compattato** (`e9435fe`): invece di copiare l'intera `QueryState`,
  il device impacchetta le sole celle che il backtrace leggerà davvero. La D2H
  per query passa da **288 KB a circa 600 byte**.
- **Workspace persistente**: la `QueryState` non viene più allocata e liberata a
  ogni chiamata, il che toglie i 136–169 ms di `cudaMalloc`.

## 08–10 — Lo stato attuale, e l'esperimento mai provato

La versione su `main` oggi è `b375a77` (cartella `09`). Raggruppa lo stato
condiviso del blocco in una struct `BlockShared` e aggiunge
`__launch_bounds__(256, 2)` al kernel — l'esperimento che il report della
baseline aveva indicato e mai eseguito. La cartella `08` è la versione
immediatamente precedente (`6e0b4f8`, stato sparso in diciotto `__shared__`), e
la `10` contiene due varianti di controllo costruite apposta per isolare i due
cambiamenti: lo stato sparso con `-maxrregcount=128`, e `BlockShared` senza
`launch_bounds`.

Il sospetto che `launch_bounds` reintroducesse spill era infondato: **128
registri, zero byte di spill**. E la catena registri → occupancy è confermata
esattamente, su tutti e quattro i dataset:

| variante | registri | occupancy raggiunta |
|---|---:|---:|
| `BlockShared` senza bound (`10`) | 175 | 22–25 % |
| stato sparso (`08`) | 138 | 30–37 % |
| stato sparso + bound (`10`) | 128 | 36–49 % |
| `BlockShared` + bound (`09`) | 128 | 35–49 % |

Le quattro varianti eseguono lo stesso lavoro: 17,9–18,6 M istruzioni, e settori
di load/store identici alla cifra. La differenza è solo quanti warp stanno
residenti.

**Ma l'occupancy più alta non produce, qui, un kernel più veloce.** Durata del
kernel a 128 thread, singolo lancio sotto `ncu`, in microsecondi:

| variante | reg | `ebola_exact` | `ebola_error` | `c4_exact` | `c4_err` |
|---|---:|---:|---:|---:|---:|
| stato sparso (`08`) | 138 | **118** | **643** | **798** | 1722 |
| `BlockShared` senza bound | 175 | 143 | 829 | 752 | **1268** |
| sparso + bound | 128 | 131 | 847 | 824 | 1345 |
| `BlockShared` + bound (`09`) | 128 | 159 | 817 | 815 | 1384 |

Su tre dataset su quattro la versione con l'occupancy peggiore è la più veloce.

Questo **contraddice** una misura precedente fatta con i contatori `cudaEvent`
dell'applicazione, non con `ncu`: su `c4_err_2k` (2048 query), a regime con
`--repeat 5`, `09` dava 2,14 ms contro i 4,17 di `08` — un 1,95× che qui non si
vede.

Le due misure differiscono in tre cose insieme: il dataset (512 query contro
2048, cioè 512 blocchi contro 2048 su 40 SM), il regime (singolo lancio a freddo
contro cinque batch di fila) e lo strumento (`ncu`, che serializza e strumenta,
contro `cudaEvent`). L'ipotesi più semplice è che il guadagno sia **dipendente
dalla scala** — con 51 blocchi per SM l'occupancy conta, con 13 no — ma è
un'ipotesi: `c4_err_2k` non è fra i dataset profilati qui, scelti per essere
confrontabili con la campagna `01`.

**Quindi: il meccanismo è dimostrato, il guadagno no.** Quello che serve per
chiudere è un A/B con `cudaEvent` delle quattro varianti su entrambi i dataset,
senza `ncu` di mezzo. Non è stato fatto.

## Avvertenze di lettura

**I tempi assoluti non sono confrontabili fra campagne diverse.** Sono stati
raccolti in sessioni diverse, su VM diverse, e la T4 di Colab varia i clock: la
campagna `01` ha i clock fissati a mano (`clock-1590MHz` e `clock-base` sono due
serie separate proprio per questo), le altre no. Ogni tabella qui sopra confronta
build misurate **nella stessa sessione**, ed è l'unico confronto che regge.

**L'occupancy è sempre stata limitata dai registri**, mai dalla shared memory né
dal numero di blocchi. Questo vale in tutte e otto le campagne, ed è il motivo
per cui `__launch_bounds__` è finito per essere la leva che contava.

**Il kernel è una frazione piccola del tempo di processo.** Su `c4_err_2k` vale
lo 0,6 % del wall: il resto è inizializzazione del contesto CUDA, upload del
grafo, parsing e scrittura del GAF. Le accelerazioni del kernel documentate qui
si vedono in un uso con molti batch per processo, non nella CLI che ne fa uno.
