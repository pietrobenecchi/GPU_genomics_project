#!/usr/bin/env python3
"""Figura 4: da cosa è fatto il tempo GPU. Dati misurati da Nsight Systems."""
import json, sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

A = json.loads(Path(sys.argv[1]).read_text())
OUT = Path(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=True)

DS = ['ebola_exact_smoke', 'ebola_error_smoke', 'c4_exact', 'c4_err']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#d8d7d2'
# palette validata, slot 1-5
VOCI = [
    ('copia device→host', 'size_memcpy Device-to-Host', 'mem_memcpy Device-to-Host', '#2a78d6'),
    ('cudaMalloc',        None, 'api_cudaMalloc',                                    '#eb6834'),
    ('cudaMemset',        None, 'mem_memset',                                        '#eda100'),
    ('kernel di allineamento', None, 'kern_theseus_align_batch_kernel',              '#1baf7a'),
    ('altri kernel + H2D', None, None,                                               '#e87ba4'),
]

fig, ax = plt.subplots(figsize=(10.5, 4.4))
ys = list(range(len(DS)))[::-1]
G = 1.2                                    # distanziatore fra i segmenti, in ms
for yi, d in zip(ys, DS):
    e = A[d]
    altri = e.get('kern_graph_readback_kernel', 0) + e.get('kern_seq_length_kernel', 0) \
        + e.get('mem_memcpy Host-to-Device', 0)
    vals = [e.get('mem_memcpy Device-to-Host', 0), e.get('api_cudaMalloc', 0),
            e.get('mem_memset', 0), e.get('kern_theseus_align_batch_kernel', 0), altri]
    tot = sum(vals)
    left = 0
    for (lab, _, _, col), v in zip(VOCI, vals):
        ax.barh(yi, max(v - G, 0.05), left=left, height=0.62, color=col,
                label=lab if yi == ys[0] else None)
        left += v
    ax.text(tot + 14, yi, f'{tot:,.0f} ms', va='center', fontsize=9, color=INK)
    k = e.get('kern_theseus_align_batch_kernel', 0)
    ax.text(tot + 108, yi, f'kernel {100*k/tot:.1f} %', va='center', fontsize=8.5,
            color=INK2)

ax.set_yticks(ys)
ax.set_yticklabels([d.replace('_smoke', '') for d in DS], fontsize=9)
ax.set_xlabel('tempo GPU (ms)  —  128 thread per blocco', color=INK2, fontsize=9)
ax.set_xlim(0, 860)
ax.set_title("Di cosa è fatto il tempo sulla GPU\n"
             "il kernel di allineamento è la fetta verde: circa l'1 % del totale",
             fontsize=11.5, color=INK, loc='left')
ax.grid(axis='x', color=GRID, lw=0.7, alpha=0.6)
ax.set_axisbelow(True)
ax.spines[['top', 'right', 'left']].set_visible(False)
ax.spines['bottom'].set_color(GRID)
ax.tick_params(colors=INK2, labelsize=8.5, length=3, color=GRID)
for l in ax.get_xticklabels() + ax.get_yticklabels():
    l.set_color(INK2)
ax.legend(frameon=False, fontsize=8.5, ncol=5, loc='lower left',
          bbox_to_anchor=(0, -0.32), labelcolor=INK2)
fig.tight_layout()
fig.savefig(OUT / '4_tempo_gpu.png', dpi=170, facecolor='white')
print('scritta', OUT / '4_tempo_gpu.png')
