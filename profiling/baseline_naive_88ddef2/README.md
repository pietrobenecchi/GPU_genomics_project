# Baseline di profiling — GPU naive, commit `88ddef2`

**Parti da qui.** Questa pagina dice cosa c'è e cosa aprire. Niente numeri: quelli
stanno nei file elencati sotto.

> **I dati grezzi non sono qui.** Stanno in **[`profile-cuda/`](../../profile-cuda)**,
> alla radice del repository: quella cartella contiene *solo* file prodotti da Nsight
> Compute e Nsight Systems, senza nessuna rielaborazione, ed è pensata per essere
> committata da sola. Questa cartella qui contiene l'**analisi**: grafici, tabelle e
> testo, tutti derivati da quei file.

---

## In 30 secondi

Tre cose, misurate su Tesla T4 con gli strumenti NVIDIA:

1. **Il kernel è limitato dalla banda di memoria, non dal calcolo.** Usa il 67–81 % della
   banda DRAM della T4 e il 3 % delle unità di calcolo.
2. **Circa metà di quel traffico è azzeramento di memoria che nessuno rilegge**
   (`sp_init`). Sulle sole scritture è l'80 %.
3. **Il kernel è però lo 0,1–0,9 % del tempo GPU.** Il resto è la copia device→host
   dell'array `QueryState` (2,26 GB) e la `cudaMalloc` che lo alloca.

Tutto è stato validato prima di misurare: il commit passa la regressione CPU-GPU su
4 dataset × 3 configurazioni di thread, con output **identico byte per byte**.

---

## Cosa aprire, in ordine

| # | File | Cosa risponde |
|---|---|---|
| 1 | [`risultati/grafici/`](risultati/grafici) | i quattro grafici di sintesi — **il modo più veloce di capire i risultati** |
| 2 | [`risultati/tabelle/`](risultati/tabelle) | gli stessi dati in CSV, apribili con Excel o LibreOffice |
| 3 | [`REPORT.md`](REPORT.md) | l'analisi completa, con il ragionamento e i limiti della misura |
| 4 | [`../../profile-cuda/nsight-compute/testo/`](../../profile-cuda/nsight-compute/testo) | i report Nsight Compute in testo, uno per configurazione |
| 5 | [`../../profile-cuda/nsight-systems/`](../../profile-cuda/nsight-systems) | i report Nsight Systems, la linea temporale |
| 6 | [`00_regressione.md`](00_regressione.md) | la prova che il commit misurato è corretto |
| 7 | [`01_analisi_scratchpad.md`](01_analisi_scratchpad.md) | analisi statica dello ScratchPad (nessuna modifica al codice) |

Chi ha fretta legge i grafici e la sezione «In una riga» in cima a `REPORT.md`.

---

## I grafici

| file | cosa mostra |
|---|---|
| [`1_banda_vs_picco.png`](risultati/grafici/1_banda_vs_picco.png) | quanta banda DRAM usa il kernel rispetto al massimo della T4 |
| [`2_traffico_dram.png`](risultati/grafici/2_traffico_dram.png) | quanta parte del traffico è lavoro utile e quanta è azzeramento |
| [`3_roofline_intero.png`](risultati/grafici/3_roofline_intero.png) | il roofline, in operazioni intere invece che in FLOP |
| [`4_tempo_gpu.png`](risultati/grafici/4_tempo_gpu.png) | di cosa è fatto il tempo sulla GPU — la fetta del kernel |

## Le tabelle

Tutte in CSV con intestazioni per esteso, una riga per configurazione.

| file | contenuto |
|---|---|
| `1_riepilogo.csv` | **la tabella principale**: durata, banda, occupancy, traffico, per dataset × thread |
| `2_stall_e_memoria.csv` | dove si ferma il kernel e quanto sono efficienti gli accessi |
| `3_tempi_end_to_end.csv` | tempo del processo intero, misurato con i CUDA event |
| `4_attribuzione_sorgente.csv` | quale riga di codice genera il traffico |
| `5_confronto_storico.csv` | i numeri nuovi accanto a quelli di `docs/handoff_parallelizzazione_kernel.md` |
| `6_dati_roofline.csv` | i punti del roofline e come sono calcolati |
| `7_tempo_gpu_nsys.csv` | la scomposizione del tempo GPU misurata da Nsight Systems |

---

## Aprire i report NVIDIA

Nsight Compute 2025.4.1 e Nsight Systems 2025.6.3 sono **già installati** su questa
macchina, in `~/.local/opt/`, senza root. I comandi `ncu`, `ncu-ui`, `nsys` e `nsys-ui`
sono in `~/.local/bin`, già nel `PATH`. **Per leggere un report non serve una GPU.**

```bash
ncu-ui  profile-cuda/nsight-compute/clock-1590MHz/c4_err_128_run1.ncu-rep
nsys-ui profile-cuda/nsight-systems/c4_err_128.nsys-rep
```

Nella finestra di Nsight Compute: **Details** ha tutte le sezioni con i grafici a colori,
**Source** mostra il costo riga per riga, e **GPU Speed Of Light Roofline Chart** è il
roofline nativo che dalla riga di comando non si può esportare come immagine.

Senza aprire nulla, gli stessi report in testo:

```bash
less profile-cuda/nsight-compute/testo/c4_err_128.txt
less profile-cuda/nsight-systems/c4_err_128.stats.txt
```

Comandi di raccolta e istruzioni di reinstallazione:
[`profile-cuda/COMANDI.md`](../../profile-cuda/COMANDI.md).

---

## Come sono nati i numeri

Le misure vengono **interamente da Nsight Compute e Nsight Systems**, eseguiti su una
Tesla T4 affittata su Google Colab. Nulla è stato misurato a mano.

Quello che sta in *questa* cartella è solo la presentazione: i quattro PNG e i CSV di
riepilogo, perché la riga di comando di `ncu` non disegna grafici e non sa affiancare 12
configurazioni in una tabella sola. Ogni valore risale a un file in
[`profile-cuda/`](../../profile-cuda).

---

## `dati_grezzi/` — cosa resta qui

Non serve aprirlo per capire i risultati. Serve per rifarli o per contestarli. Contiene
solo materiale **non** prodotto dal profiler: i file NVIDIA sono tutti in `profile-cuda/`.

| cartella | contenuto |
|---|---|
| `logs/` | build, regressioni, blocco dei clock, tempi end-to-end, log di ogni run |
| `env/` | `nvidia-smi`, versioni degli strumenti, `setup_colab.md` per rifare tutto |
| `script/` | tutto ciò che è stato eseguito, sulla VM e in locale, verbatim |
| `agg.json`, `timing.json`, `nsys_agg.json` | i dati aggregati da cui nascono grafici e tabelle |

---

## Dimensioni, se devi committare

| cartella | peso | |
|---|---:|---|
| `profile-cuda/` (radice del repo) | ~277 MB | i file NVIDIA |
| `profiling/baseline_naive_88ddef2/` | ~1 MB | questa analisi |

Questa cartella è leggera e si committa senza pensarci. Per `profile-cuda/` vedi la nota
sulle dimensioni in [`profile-cuda/COMANDI.md`](../../profile-cuda/COMANDI.md).
