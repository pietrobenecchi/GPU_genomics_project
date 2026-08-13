#!/usr/bin/env python3
"""Tabelle finali: unita' normalizzate, mediana + dispersione su 3 run."""
import csv, statistics, json, sys
from pathlib import Path

SP = Path('/tmp/claude-1000/-home-pietrobenecchi-Documents-GPU-genomics-project/'
          '78438445-b5a6-4dc8-a267-0c733b85a811/scratchpad/out')
DS = ['ebola_exact_smoke', 'ebola_error_smoke', 'c4_exact', 'c4_err']
TH = [64, 128, 256]
NQ = {'ebola_exact_smoke': 256, 'ebola_error_smoke': 256, 'c4_exact': 512, 'c4_err': 512}
SPAN = {'ebola_exact_smoke': 9164, 'ebola_error_smoke': 9164,
        'c4_exact': 52107, 'c4_err': 52107}
CELL = 24

TO_US = {'usecond': 1.0, 'us': 1.0, 'msecond': 1e3, 'ms': 1e3, 'second': 1e6,
         'nsecond': 1e-3, 'ns': 1e-3}


def num(s):
    s = (s or '').replace(',', '').strip()
    if s in ('', '-', 'n/a'):
        return None
    try:
        return float(s)
    except ValueError:
        return None


def details(path):
    out = {}
    for row in csv.DictReader(open(path, newline='')):
        v = num(row.get('Metric Value'))
        if v is None:
            continue
        name = row['Metric Name'].strip()
        unit = (row.get('Metric Unit') or '').strip()
        if name == 'Duration':
            v = v * TO_US.get(unit, 1.0)
        out[name] = v
    return out


def raws(path):
    rows = list(csv.DictReader(open(path, newline='')))
    return {k: num(v) for k, v in rows[-1].items() if k}


def load(dirname):
    t = {}
    for ds in DS:
        for th in TH:
            runs = [details(SP / dirname / f'{ds}_{th}_run{r}.csv')
                    for r in (1, 2, 3)
                    if (SP / dirname / f'{ds}_{th}_run{r}.csv').exists()]
            if runs:
                t[(ds, th)] = runs
    return t


def stat(runs, key):
    vals = [r.get(key) for r in runs]
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None
    m = statistics.median(vals)
    sp = 100 * (max(vals) - min(vals)) / m if m else 0.0
    return m, sp


def build(dirname, rawdir):
    T = load(dirname)
    R = {}
    for ds in DS:
        for th in TH:
            p = SP / rawdir / f'{ds}_{th}.raw.csv'
            if p.exists():
                R[(ds, th)] = raws(p)
    out = {}
    for ds in DS:
        for th in TH:
            runs = T.get((ds, th))
            if not runs:
                continue
            raw = R.get((ds, th), {})
            e = {}
            for k, label in [('Duration', 'dur_us'), ('DRAM Throughput', 'dram_pct'),
                             ('Compute (SM) Throughput', 'sm_pct'),
                             ('Memory Throughput', 'dram_gbs'),
                             ('L1/TEX Hit Rate', 'l1_hit'), ('L2 Hit Rate', 'l2_hit'),
                             ('Achieved Occupancy', 'occ_ach'),
                             ('Theoretical Occupancy', 'occ_theo'),
                             ('Achieved Active Warps Per SM', 'warps_ach'),
                             ('Registers Per Thread', 'regs'),
                             ('Block Limit Registers', 'blk_reg'),
                             ('Block Limit Shared Mem', 'blk_smem'),
                             ('Block Limit Warps', 'blk_warp'),
                             ('Static Shared Memory Per Block', 'smem_static_B'),
                             ('Dynamic Shared Memory Per Block', 'smem_dyn_KB'),
                             ('Waves Per SM', 'waves'),
                             ('Avg. Active Threads Per Warp', 'thr_per_warp'),
                             ('Avg. Not Predicated Off Threads Per Warp', 'thr_per_warp_np'),
                             ('Executed Instructions', 'inst_exec'),
                             ('Branch Efficiency', 'branch_eff'),
                             ('SM Frequency', 'sm_ghz'),
                             ('Warp Cycles Per Issued Instruction', 'wcpi')]:
                m, s = stat(runs, k)
                e[label] = m
                if label == 'dur_us':
                    e['dur_spread_pct'] = s
            g = raw.get
            e['dram_rd_MB'] = g('dram__bytes_read.sum')
            e['dram_wr_MB'] = g('dram__bytes_write.sum')
            e['dram_rd_sec'] = g('dram__sectors_read.sum')
            e['dram_wr_sec'] = g('dram__sectors_write.sum')
            e['g_st_sec'] = g('l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum')
            e['g_ld_sec'] = g('l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum')
            e['g_st_req'] = g('l1tex__t_requests_pipe_lsu_mem_global_op_st.sum')
            e['g_ld_req'] = g('l1tex__t_requests_pipe_lsu_mem_global_op_ld.sum')
            e['l_st_sec'] = g('l1tex__t_sectors_pipe_lsu_mem_local_op_st.sum')
            e['l_ld_sec'] = g('l1tex__t_sectors_pipe_lsu_mem_local_op_ld.sum')
            e['sec_req_st'] = (e['g_st_sec'] / e['g_st_req']) if e['g_st_req'] else None
            e['sec_req_ld'] = (e['g_ld_sec'] / e['g_ld_req']) if e['g_ld_req'] else None
            for k2, lab in [('smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio', 'st_barrier'),
                            ('smsp__average_warps_issue_stalled_lg_throttle_per_issue_active.ratio', 'st_lg'),
                            ('smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio', 'st_lsb'),
                            ('smsp__average_warps_issue_stalled_mio_throttle_per_issue_active.ratio', 'st_mio'),
                            ('smsp__average_warps_issue_stalled_wait_per_issue_active.ratio', 'st_wait'),
                            ('smsp__average_warps_issue_stalled_no_instruction_per_issue_active.ratio', 'st_noinst'),
                            ('sm__sass_thread_inst_executed_op_integer_pred_on.sum', 'int_thr_inst'),
                            ('smsp__thread_inst_executed_per_inst_executed.ratio', 'thr_per_inst'),
                            ('smsp__inst_executed.sum', 'warp_inst')]:
                e[lab] = g(k2)
            # analitico sp_init
            e['sp_init_MB'] = NQ[ds] * SPAN[ds] * CELL / 1e6
            e['dram_tot_MB'] = ((e['dram_rd_MB'] or 0) + (e['dram_wr_MB'] or 0))
            out[f'{ds}|{th}'] = e
    return out


if __name__ == '__main__':
    res = {'boost': build('csv_boost', 'roofline_boost'),
           'base': build('csv', 'roofline')}
    Path(sys.argv[1]).write_text(json.dumps(res, indent=1))
    print('written', sys.argv[1])
    for reg in ('boost', 'base'):
        print(f'\n===== {reg} =====')
        print(f'{"combo":28s} {"dur_us":>9s} {"±%":>5s} {"DRAM%":>6s} {"GB/s":>7s} {"SM%":>5s} '
              f'{"rdMB":>7s} {"wrMB":>7s} {"totMB":>8s} {"spinitMB":>9s} {"sp/tot":>7s} {"sp/wr":>6s}')
        for k, e in res[reg].items():
            tot = e['dram_tot_MB']
            print(f'{k:28s} {e["dur_us"]:9.1f} {e["dur_spread_pct"]:5.1f} {e["dram_pct"]:6.1f} '
                  f'{e["dram_gbs"]:7.1f} {e["sm_pct"]:5.2f} {e["dram_rd_MB"]:7.1f} {e["dram_wr_MB"]:7.1f} '
                  f'{tot:8.1f} {e["sp_init_MB"]:9.1f} {100*e["sp_init_MB"]/tot:6.1f}% '
                  f'{100*e["sp_init_MB"]/e["dram_wr_MB"]:5.1f}%')
