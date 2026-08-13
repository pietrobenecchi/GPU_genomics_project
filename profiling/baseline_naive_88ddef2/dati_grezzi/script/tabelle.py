#!/usr/bin/env python3
"""Le tabelle leggibili: intestazioni in italiano, una tabella per domanda."""
import json, csv, sys
from pathlib import Path

AGG = json.loads(Path(sys.argv[1]).read_text())
TIM = json.loads(Path(sys.argv[2]).read_text())
OUT = Path(sys.argv[3]); OUT.mkdir(parents=True, exist_ok=True)

DS = ['ebola_exact_smoke', 'ebola_error_smoke', 'c4_exact', 'c4_err']
TH = [64, 128, 256]
combos = [(d, t) for d in DS for t in TH]
B = AGG['boost']


def w(name, header, rows):
    with open(OUT / name, 'w', newline='') as fh:
        c = csv.writer(fh)
        c.writerow(header)
        c.writerows(rows)
    print(f'  {name:34s} {len(rows)} righe')


# 1 --- riepilogo: una riga per configurazione, le metriche che contano
rows = []
for d, t in combos:
    e = B[f'{d}|{t}']
    tot = e['dram_rd_MB'] + e['dram_wr_MB']
    rows.append([d, t,
                 round(e['dur_us'], 1), round(e['dur_spread_pct'], 1),
                 round(e['dram_pct'], 1), round(e['dram_gbs'], 1), round(e['sm_pct'], 2),
                 round(e['dram_rd_MB'], 1), round(e['dram_wr_MB'], 1), round(tot, 1),
                 round(e['sp_init_MB'], 1),
                 round(100 * e['sp_init_MB'] / tot, 1),
                 round(100 * e['sp_init_MB'] / e['dram_wr_MB'], 1),
                 round(e['occ_ach'], 1), int(e['regs']), int(e['blk_reg']),
                 round(e['thr_per_warp'], 1),
                 round(e['l1_hit'], 1), round(e['l2_hit'], 1)])
w('1_riepilogo.csv',
  ['dataset', 'thread_per_blocco', 'durata_kernel_us', 'dispersione_%',
   'DRAM_%_del_picco', 'DRAM_GB_al_s', 'SM_%_del_picco',
   'DRAM_lette_MB', 'DRAM_scritte_MB', 'DRAM_totale_MB',
   'sp_init_MB', 'sp_init_%_del_totale', 'sp_init_%_delle_scritture',
   'occupancy_raggiunta_%', 'registri_per_thread', 'blocchi_residenti_per_SM',
   'thread_attivi_per_warp_su_32', 'L1_hit_%', 'L2_hit_%'], rows)

# 2 --- stall
rows = [[d, t] + [round(B[f'{d}|{t}'][k], 2) for k in
                  ('st_lg', 'st_barrier', 'st_lsb', 'st_mio', 'st_wait', 'st_noinst')]
        + [round(B[f'{d}|{t}']['sec_req_st'], 2), round(B[f'{d}|{t}']['sec_req_ld'], 2)]
        for d, t in combos]
w('2_stall_e_memoria.csv',
  ['dataset', 'thread_per_blocco', 'stall_pipe_load_store', 'stall_barriera',
   'stall_attesa_memoria_lunga', 'stall_mio', 'stall_wait', 'stall_nessuna_istruzione',
   'settori_per_richiesta_store', 'settori_per_richiesta_load'], rows)

# 3 --- tempi end-to-end
rows = []
for d, t in combos:
    e = TIM.get(f'{d}|{t}')
    if not e:
        continue
    rows.append([d, t, round(e['wall_s'], 3), round(e['wall_spread'], 1),
                 round(e['h2d_ms'], 2), round(e['kernel_ms'], 2), round(e['d2h_ms'], 1),
                 round(e['gpu_total_ms'], 1),
                 round(100 * e['kernel_ms'] / e['gpu_total_ms'], 2),
                 round(100 * e['d2h_ms'] / e['gpu_total_ms'], 1), e['n']])
w('3_tempi_end_to_end.csv',
  ['dataset', 'thread_per_blocco', 'processo_intero_s', 'dispersione_%',
   'copia_host_to_device_ms', 'kernel_ms', 'copia_device_to_host_ms',
   'totale_GPU_ms', 'kernel_%_del_tempo_GPU', 'D2H_%_del_tempo_GPU', 'ripetizioni'], rows)

# 4 --- attribuzione per riga sorgente
src = [('ebola_exact_smoke', 178.3, 63.5, 171.1, 56.3, 95.98, 56.30),
       ('ebola_error_smoke', 199.9, 80.1, 171.1, 56.3, 85.58, 56.30),
       ('c4_exact', 1960.2, 654.8, 1945.9, 640.5, 99.27, 640.29),
       ('c4_err', 2003.3, 687.5, 1945.9, 640.5, 97.13, 640.29)]
w('4_attribuzione_sorgente.csv',
  ['dataset_a_128_thread', 'settori_L2_totali_MB', 'settori_L2_ideali_MB',
   'riga_1072_settori_MB', 'riga_1072_ideali_MB', 'riga_1072_%_del_totale',
   'previsione_analitica_MB'],
  [[d, a, b, c, dd, e, f] for d, a, b, c, dd, e, f in src])

# 5 --- confronto con lo storico
w('5_confronto_storico.csv',
  ['metrica', 'configurazione', 'storico_handoff', 'misurato_clock_base',
   'misurato_clock_1590MHz'],
  [['DRAM % del picco', 'c4_exact@64', 70.96, 70.6, 71.6],
   ['DRAM % del picco', 'c4_exact@256', 74.39, 73.8, 81.2],
   ['DRAM % del picco', 'c4_err@64', 56.91, 59.6, 67.5],
   ['DRAM % del picco', 'c4_err@256', 46.95, 48.0, 77.5],
   ['SM % del picco', 'c4_exact@64', 6.99, 6.98, 2.84],
   ['SM % del picco', 'c4_exact@256', 6.50, 6.51, 2.94],
   ['SM % del picco', 'c4_err@64', 6.38, 6.65, 2.97],
   ['SM % del picco', 'c4_err@256', 4.88, 5.54, 3.70],
   ['stall pipe load/store', 'c4_exact@64', 91.9, 90.4, 234.9],
   ['stall pipe load/store', 'c4_exact@256', 100.1, 98.1, 228.0],
   ['stall pipe load/store', 'c4_err@64', 31.5, 29.8, 85.0],
   ['stall pipe load/store', 'c4_err@256', 26.6, 23.4, 35.1],
   ['thread attivi per warp', 'c4_exact@64', 27.01, 26.86, 26.86],
   ['thread attivi per warp', 'c4_exact@256', 27.56, 27.44, 27.44]])
print('fatto')
