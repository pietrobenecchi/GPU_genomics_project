#!/usr/bin/env python3
"""Attribuisce i settori L2 globali alle righe di sorgente CUDA.

Il CSV di `ncu --page source --print-source cuda,sass --csv` alterna:
  - una riga "File Path", <path>
  - una riga "Function Name", <kernel>
  - una riga di intestazione colonne
  - righe con "Line No" valorizzato = riga CUDA aggregata
  - righe con "Line No" vuoto = singole istruzioni SASS sotto quella riga
Prendiamo solo le righe CUDA aggregate.
"""
import csv, sys, io
from collections import defaultdict


def parse(path):
    per_line = defaultdict(lambda: dict(st=0.0, ld=0.0, l2=0.0, l2_ideal=0.0,
                                        l2_local=0.0, inst=0.0, src=''))
    cur_file = None
    header = None
    with open(path, newline='') as fh:
        rd = csv.reader(fh)
        for row in rd:
            if not row:
                continue
            if row[0] == 'File Path':
                cur_file = row[1] if len(row) > 1 else None
                header = None
                continue
            if row[0] == 'Function Name':
                continue
            if row[0] == 'Line No' and len(row) > 2 and row[2] == 'Address':
                header = {name: i for i, name in enumerate(row)}
                # 'Source' compare due volte (CUDA e SASS): la prima e' la CUDA
                header['SourceCuda'] = row.index('Source')
                continue
            if header is None:
                continue
            lineno = row[0].strip()
            if not lineno:
                continue          # riga SASS, gia' contata nell'aggregato CUDA
            try:
                ln = int(lineno)
            except ValueError:
                continue

            def g(col):
                i = header.get(col)
                if i is None or i >= len(row):
                    return 0.0
                v = row[i].replace(',', '').strip()
                if v in ('', '-'):
                    return 0.0
                try:
                    return float(v)
                except ValueError:
                    return 0.0

            key = (cur_file, ln)
            e = per_line[key]
            e['l2'] += g('L2 Theoretical Sectors Global')
            e['l2_ideal'] += g('L2 Theoretical Sectors Global Ideal')
            e['l2_local'] += g('L2 Theoretical Sectors Local')
            e['inst'] += g('Instructions Executed')
            if not e['src']:
                e['src'] = row[header['SourceCuda']].strip()
    return per_line


if __name__ == '__main__':
    for path in sys.argv[1:]:
        per_line = parse(path)
        tot = sum(e['l2'] for e in per_line.values())
        tot_loc = sum(e['l2_local'] for e in per_line.values())
        print(f'\n########## {path}')
        print(f'L2 theoretical sectors global TOT = {tot:,.0f}  '
              f'({tot*32/1e6:,.1f} MB)   local = {tot_loc:,.0f} ({tot_loc*32/1e6:,.1f} MB)')
        rows = sorted(per_line.items(), key=lambda kv: -kv[1]['l2'])[:14]
        print(f'{"file:line":52s} {"sectors":>14s} {"MB":>9s} {"%":>6s}  source')
        for (f, ln), e in rows:
            if e['l2'] == 0:
                continue
            short = (f.split("/")[-1] if f else "?") + f':{ln}'
            print(f'{short:52s} {e["l2"]:14,.0f} {e["l2"]*32/1e6:9,.1f} '
                  f'{100*e["l2"]/tot:6.2f}  {e["src"][:70]}')
