#!/usr/bin/env python3
"""Wall/kernel/H2D/D2H timing for each build x dataset x threads.

One warm-up run per point (discarded) then N measured runs; the median is what
gets reported, with min/max so the dispersion is visible.
"""
import json, re, statistics, subprocess, sys, time
from pathlib import Path

ROOT = Path("/content/theseus")
sys.path.insert(0, str(ROOT / "scripts"))
import run_ggbs_gpu_regression as R

RUNS = 3
TIMING = re.compile(r"GPU timing: h2d ([\d.e+-]+) ms; kernel ([\d.e+-]+) ms; "
                    r"d2h ([\d.e+-]+) ms; total ([\d.e+-]+) ms")
STAGES = re.compile(r"align_batch ([\d.e+-]+) ms")


def one_run(binary, graph, queries, threads, out):
    t0 = time.perf_counter()
    p = subprocess.run([str(binary), "--backend", "gpu", "--require-gpu-result",
                        "--gpu-threads", str(threads),
                        "-g", str(graph), "-s", str(queries), "-f", out],
                       capture_output=True, text=True)
    wall = (time.perf_counter() - t0) * 1000.0
    if p.returncode != 0:
        raise RuntimeError(f"run failed: {p.stderr[-2000:]}")
    m = TIMING.search(p.stderr)
    s = STAGES.search(p.stderr)
    if not m:
        raise RuntimeError(f"no timing line: {p.stderr[-2000:]}")
    return {"wall_ms": wall, "h2d_ms": float(m.group(1)),
            "kernel_ms": float(m.group(2)), "d2h_ms": float(m.group(3)),
            "gpu_total_ms": float(m.group(4)),
            "align_batch_ms": float(s.group(1)) if s else None}


def main():
    builds = {name: Path(f"/content/wt/{name}/build-gpu/apps/seq2graph_proxy")
              for name in sys.argv[1:]}
    datasets = list(R.DATASETS)
    thread_counts = [128]
    sweep = {"c4_err", "c4_exact", "c4_err_2k", "ebola_err_2k"}

    out = []
    for build, binary in builds.items():
        for ds in datasets:
            item = R.DATASETS[ds]
            graph = ROOT / item["graph"]
            queries = ROOT / item["queries"]
            nq = sum(1 for l in queries.read_text().splitlines() if l.startswith(">"))
            for threads in (thread_counts + [64, 256] if ds in sweep else thread_counts):
                one_run(binary, graph, queries, threads, "/tmp/warm.gaf")  # warm-up
                runs = [one_run(binary, graph, queries, threads, "/tmp/out.gaf")
                        for _ in range(RUNS)]
                rec = {"build": build, "dataset": ds, "tier": item["tier"],
                       "size": item["size"], "threads": threads, "queries": nq,
                       "runs": runs}
                for key in ("wall_ms", "kernel_ms", "h2d_ms", "d2h_ms",
                            "gpu_total_ms", "align_batch_ms"):
                    vals = [r[key] for r in runs if r[key] is not None]
                    if vals:
                        rec[key] = statistics.median(vals)
                        rec[key + "_min"] = min(vals)
                        rec[key + "_max"] = max(vals)
                rec["qps"] = nq / (rec["wall_ms"] / 1000.0)
                out.append(rec)
                print(f"{build:5s} {ds:18s} thr {threads:3d}  "
                      f"wall {rec['wall_ms']:8.1f}  kernel {rec['kernel_ms']:7.2f}  "
                      f"h2d {rec['h2d_ms']:6.2f}  d2h {rec['d2h_ms']:8.2f}  "
                      f"{rec['qps']:7.1f} q/s", flush=True)
    Path("/content/logs/timing.json").write_text(json.dumps(out, indent=1))
    print("WROTE /content/logs/timing.json")


main()
