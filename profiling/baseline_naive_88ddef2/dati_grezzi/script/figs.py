#!/usr/bin/env python3
"""Le tre figure del report. Palette validata (slot 1-4 della reference dataviz)."""
import json, csv, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

AGG, OPMIX, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
OUT.mkdir(parents=True, exist_ok=True)

DS = ['ebola_exact_smoke', 'ebola_error_smoke', 'c4_exact', 'c4_err']
SHORT = {'ebola_exact_smoke': 'ebola_exact', 'ebola_error_smoke': 'ebola_error',
         'c4_exact': 'c4_exact', 'c4_err': 'c4_err'}
TH = [64, 128, 256]
# palette categorica validata: blu, arancio, acqua, giallo
C = {'ebola_exact_smoke': '#2a78d6', 'ebola_error_smoke': '#1baf7a',
     'c4_exact': '#eb6834', 'c4_err': '#eda100'}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d8d7d2'

agg = json.loads(AGG.read_text())['boost']
PEAK = sum(e['dram_gbs'] / (e['dram_pct'] / 100) for e in agg.values()) / len(agg)

opmix = {}
for ds in DS:
    for th in TH:
        p = OPMIX / f'{ds}_{th}.csv'
        if not p.exists():
            continue
        v = {}
        for row in csv.DictReader(l for l in p.read_text().splitlines() if l.startswith('"')):
            try:
                v[row['Metric Name']] = float(row['Metric Value'].replace(',', ''))
            except (KeyError, ValueError, TypeError):
                pass
        opmix[(ds, th)] = v

combos = [(ds, th) for ds in DS for th in TH]
labels = [f'{SHORT[ds]}\n{th}' for ds, th in combos]


def style(ax):
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8, length=3, color=GRID)
    for lbl in ax.get_xticklabels() + ax.get_yticklabels():
        lbl.set_color(INK2)


# ---- Figura 1: banda raggiunta vs picco -----------------------------------
fig, ax = plt.subplots(figsize=(10.5, 4.8))
vals = [agg[f'{ds}|{th}']['dram_gbs'] for ds, th in combos]
pcts = [agg[f'{ds}|{th}']['dram_pct'] for ds, th in combos]
xs = [i + 0.16 * (i // 3) for i in range(len(combos))]   # spazio fra i gruppi
bars = ax.bar(xs, vals, width=0.82, color=[C[ds] for ds, _ in combos])
ax.axhline(PEAK, color=INK, ls='--', lw=1.4, zorder=3)
ax.text(xs[0] - 0.5, PEAK + 8,
        f'picco DRAM sostenuto misurato: {PEAK:.0f} GB/s  (targa T4: 320 GB/s)',
        fontsize=9, color=INK, va='bottom')
for x, b, p in zip(xs, bars, pcts):
    ax.text(x, b.get_height() + 5, f'{p:.0f}%', ha='center', fontsize=8.5, color=INK)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel('banda DRAM raggiunta  (GB/s)', color=INK2, fontsize=9)
ax.set_ylim(0, 352)
ax.set_yticks([0, 100, 200, 300])
ax.set_title('Banda DRAM raggiunta rispetto al picco\n'
             'T4, clock SM bloccato a 1590 MHz — mediana di 3 ripetizioni',
             fontsize=11, color=INK, loc='left')
ax.grid(axis='y', color=GRID, lw=0.7, alpha=0.6)
ax.set_axisbelow(True)
style(ax)
ax.legend(handles=[Patch(facecolor=C[d], label=SHORT[d]) for d in DS],
          frameon=False, fontsize=8, ncol=4, loc='lower left',
          bbox_to_anchor=(0, -0.30), labelcolor=INK2)
fig.tight_layout()
fig.savefig(OUT / 'banda_vs_picco.png', dpi=170, facecolor='white')
plt.close(fig)

# ---- Figura 2: da dove vengono i byte -------------------------------------
# Composizione => 100% impilato su scala lineare. I totali assoluti, che
# differiscono di un ordine di grandezza fra ebola e c4, sono etichette dirette:
# impilare valori assoluti su un asse logaritmico falserebbe le proporzioni.
fig, ax = plt.subplots(figsize=(10.5, 4.8))
tots = [agg[f'{ds}|{th}']['dram_rd_MB'] + agg[f'{ds}|{th}']['dram_wr_MB']
        for ds, th in combos]
sp = [100 * agg[f'{ds}|{th}']['sp_init_MB'] / t for (ds, th), t in zip(combos, tots)]
wr_rest = [100 * (agg[f'{ds}|{th}']['dram_wr_MB'] - agg[f'{ds}|{th}']['sp_init_MB']) / t
           for (ds, th), t in zip(combos, tots)]
rd = [100 * agg[f'{ds}|{th}']['dram_rd_MB'] / t for (ds, th), t in zip(combos, tots)]
G = 0.55                                     # distanziatore fra i segmenti
ax.bar(xs, sp, width=0.82, color='#eb6834',
       label='azzeramento di sp_init (payload analitico)')
ax.bar(xs, [v - G for v in wr_rest], width=0.82, bottom=[s + G for s in sp],
       color='#eda100', label='altre scritture DRAM')
ax.bar(xs, [v - G for v in rd], width=0.82,
       bottom=[a + b + G for a, b in zip(sp, wr_rest)],
       color='#2a78d6', label='letture DRAM')
for x, s, t in zip(xs, sp, tots):
    ax.text(x, s / 2, f'{s:.0f}%', ha='center', va='center', fontsize=9,
            color='white', fontweight='bold')
    ax.text(x, 102, f'{t:,.0f}', ha='center', fontsize=7.5, color=INK2)
ax.text(xs[0] - 0.62, 108, 'traffico DRAM totale del kernel (MB):',
        fontsize=8, color=INK2)
ax.set_xticks(xs)
ax.set_xticklabels(labels, fontsize=7.5)
ax.set_ylabel('quota del traffico DRAM del kernel  (%)', color=INK2, fontsize=9)
ax.set_ylim(0, 116)
ax.set_yticks([0, 25, 50, 75, 100])
ax.set_title('Dove finiscono i byte: quanto è lavoro utile e quanto è azzeramento\n'
             'di memoria mai riletta — in arancio la quota di sp_init',
             fontsize=11, color=INK, loc='left')
ax.grid(axis='y', color=GRID, lw=0.7, alpha=0.6)
ax.set_axisbelow(True)
style(ax)
ax.legend(frameon=False, fontsize=8, ncol=3, loc='lower left',
          bbox_to_anchor=(0, -0.30), labelcolor=INK2)
fig.tight_layout()
fig.savefig(OUT / 'traffico_dram.png', dpi=170, facecolor='white')
plt.close(fig)

# ---- Figura 3: roofline in operazioni intere ------------------------------
PEAK_INST = 40 * 64 * 1.59e9        # 40 SM x 64 core INT32 x 1.59 GHz
fig, ax = plt.subplots(figsize=(7.8, 5.2))
xr = [10 ** (i / 30) for i in range(-90, 61)]
ax.plot(xr, [min(PEAK * 1e9 * x, PEAK_INST) / 1e9 for x in xr], color=INK, lw=1.8)
ax.text(0.9, PEAK_INST / 1e9 * 1.18, f'tetto INT32  {PEAK_INST/1e12:.2f} Tinst/s',
        fontsize=8, color=INK2)
ax.text(0.0115, 8.0, f'tetto banda {PEAK:.0f} GB/s', fontsize=8, color=INK2, rotation=33)
rows = []
for ds, th in combos:
    om = opmix.get((ds, th), {})
    ints = om.get('sm__sass_thread_inst_executed_op_integer_pred_on.sum')
    e = agg[f'{ds}|{th}']
    if not ints:
        continue
    byts = (e['dram_rd_MB'] + e['dram_wr_MB']) * 1e6
    x = ints / byts
    y = ints / (e['dur_us'] * 1e-6) / 1e9
    ax.plot(x, y, 'o', color=C[ds], ms=8, mec='white', mew=1.2, zorder=4)
    # etichetta diretta solo sul punto a 128 thread: gli altri due sono
    # nella stessa nuvola e le etichette si sovrapporrebbero
    if th == 128:
        off = {'ebola_exact_smoke': (-16, 16), 'ebola_error_smoke': (10, 5),
               'c4_exact': (-58, -6), 'c4_err': (12, -12)}[ds]
        ax.annotate(SHORT[ds], (x, y), fontsize=7.5, color=INK2,
                    xytext=off, textcoords='offset points',
                    arrowprops=dict(arrowstyle='-', lw=0.6, color=GRID,
                                    shrinkA=0, shrinkB=6))
    rows.append((ds, th, x, y))
ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(0.008, 60)
ax.set_ylim(2, 6000)
ax.set_xlabel('intensità operazionale  [istruzioni intere per byte DRAM]', color=INK2, fontsize=9)
ax.set_ylabel('throughput  [Ginst intere / s]', color=INK2, fontsize=9)
ax.set_title('Roofline in istruzioni intere, non in FLOP\n'
             'il kernel non esegue FP64 e quasi nessun FP32: un roofline in FLOP/byte\n'
             'lo collocherebbe a zero',
             fontsize=10.5, color=INK, loc='left')
ax.grid(alpha=0.3, which='both', color=GRID, lw=0.6)
ax.set_axisbelow(True)
style(ax)
ax.legend(handles=[Patch(facecolor=C[d], label=SHORT[d]) for d in DS],
          frameon=False, fontsize=8, ncol=2, loc='lower right', labelcolor=INK2)
fig.tight_layout()
fig.savefig(OUT / 'roofline_intero.png', dpi=170, facecolor='white')
plt.close(fig)

# ---- CSV di supporto ------------------------------------------------------
with open(OUT / 'roofline_data.csv', 'w', newline='') as fh:
    w = csv.writer(fh)
    w.writerow(['dataset', 'threads', 'duration_us', 'duration_spread_pct',
                'dram_GBs', 'dram_pct_of_peak', 'peak_GBs_measured', 'sm_pct_of_peak',
                'dram_read_MB', 'dram_write_MB', 'dram_total_MB', 'sp_init_payload_MB',
                'int_thread_inst', 'fp32_thread_inst', 'fp64_thread_inst',
                'int_ops_per_DRAM_byte', 'Ginst_int_per_s'])
    for ds, th in combos:
        e = agg[f'{ds}|{th}']
        om = opmix.get((ds, th), {})
        ints = om.get('sm__sass_thread_inst_executed_op_integer_pred_on.sum', 0)
        byts = (e['dram_rd_MB'] + e['dram_wr_MB']) * 1e6
        w.writerow([ds, th, round(e['dur_us'], 2), round(e['dur_spread_pct'], 2),
                    round(e['dram_gbs'], 2), round(e['dram_pct'], 2), round(PEAK, 1),
                    round(e['sm_pct'], 2), round(e['dram_rd_MB'], 1),
                    round(e['dram_wr_MB'], 1), round(e['dram_rd_MB'] + e['dram_wr_MB'], 1),
                    round(e['sp_init_MB'], 1), int(ints),
                    int(om.get('sm__sass_thread_inst_executed_op_fp32_pred_on.sum', 0)),
                    int(om.get('sm__sass_thread_inst_executed_op_fp64_pred_on.sum', 0)),
                    round(ints / byts, 4) if ints else '',
                    round(ints / (e['dur_us'] * 1e-6) / 1e9, 2) if ints else ''])
print(f'PEAK={PEAK:.1f} GB/s; scritte 3 figure + roofline_data.csv in {OUT}')
