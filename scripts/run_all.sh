#!/usr/bin/env bash
#
# run_all.sh - build, run, validate and profile the Theseus CUDA backend in one
#              go, on a machine with a local NVIDIA GPU.
#
# Phases:
#   1 env      record GPU, driver, nvcc, CPU
#   2 build    configure + compile with CUDA enabled (auto-detected arch)
#   3 verify   ctest, the sample-graph baseline diff, and the full GGBS
#              CPU-vs-GPU regression at 64/128/256 threads per block
#   4 perf     --repeat runs at 64/128/256 threads, plus the CPU baseline
#   5 profile  ptxas registers/spill, and Nsight Compute on the kernel if ncu
#              is installed
#   6 summary  one readable report from all of the above
#
# Everything lands in a results directory; nothing is written into the source
# tree except the build directory.
#
# Usage:  scripts/run_all.sh [options]
#         scripts/run_all.sh --quick          # smoke datasets, no ncu
#         scripts/run_all.sh --phases verify  # only what you need
#
set -euo pipefail

# ------------------------------------------------------------------ defaults
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PROJ="$ROOT/theseus_gpu"
BUILD_DIR="$PROJ/build-gpu"
OUT_DIR="$ROOT/profiling/run_all_$(date +%Y%m%d_%H%M%S)"
ARCH=""                       # empty = auto-detect from the device
THREAD_SWEEP="64 128 256"
REPEAT=5
JOBS="$(nproc 2>/dev/null || echo 4)"
PHASES="env,build,verify,perf,profile,summary"
DATASETS=""                   # empty = pick from --quick
QUICK=0
USE_NCU=1
LOCK_CLOCKS=1
REGRESSION_TIMEOUT=900

usage() {
    sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --build-dir <dir>     build directory (default: theseus_gpu/build-gpu)
  --out-dir <dir>       results directory (default: profiling/run_all_<stamp>)
  --arch <NN>           CUDA architecture, e.g. 75 for a T4 (default: detected)
  --threads "<list>"    threads per block to sweep (default: "64 128 256")
  --repeat <n>          iterations per timed run (default: 5; iteration 0 pays
                        the per-process costs and is dropped from the median)
  --datasets "<list>"   datasets to time (default: the four 2k sets, or the two
                        smoke sets with --quick)
  --jobs <n>            parallel compile jobs (default: nproc)
  --phases <list>       comma-separated subset of
                        env,build,verify,perf,profile,summary
  --quick               smoke datasets, no ncu, --repeat 2
  --no-ncu              skip Nsight Compute even if it is installed
  --no-clock-lock       do not try to pin the SM clocks (needs root otherwise)
  -h, --help            this text
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --build-dir)   BUILD_DIR="$2"; shift 2 ;;
        --out-dir)     OUT_DIR="$2"; shift 2 ;;
        --arch)        ARCH="$2"; shift 2 ;;
        --threads)     THREAD_SWEEP="$2"; shift 2 ;;
        --repeat)      REPEAT="$2"; shift 2 ;;
        --datasets)    DATASETS="$2"; shift 2 ;;
        --jobs)        JOBS="$2"; shift 2 ;;
        --phases)      PHASES="$2"; shift 2 ;;
        --quick)       QUICK=1; USE_NCU=0; REPEAT=2; shift ;;
        --no-ncu)      USE_NCU=0; shift ;;
        --no-clock-lock) LOCK_CLOCKS=0; shift ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 1 ;;
    esac
done

has_phase() { [[ ",$PHASES," == *",$1,"* ]]; }

G="$PROJ/data/validation/ggbs/graphs"
Q="$PROJ/data/validation/ggbs/queries"
GOLD="$PROJ/data/validation/ggbs/golden"
BIN="$BUILD_DIR/apps/seq2graph_proxy"

if [[ -z "$DATASETS" ]]; then
    if [[ $QUICK -eq 1 ]]; then
        DATASETS="ebola_exact_smoke ebola_error_smoke"
    else
        DATASETS="c4_exact_2k ebola_exact_2k c4_err_2k ebola_err_2k"
    fi
fi

# dataset -> graph. The names are GGBS labels for benchmark inputs; the "_err"
# sets are the ones whose reads carry errors, so they need a non-zero score.
graph_of() {
    case "$1" in
        ebola*) echo ebola ;;
        c4*)    echo c4 ;;
        *)      echo "no graph known for dataset $1" >&2; return 1 ;;
    esac
}

LOG="$OUT_DIR/logs"
mkdir -p "$LOG" "$OUT_DIR/ncu"
STATUS="$OUT_DIR/status.txt"
# Appended, not truncated: re-running a subset of the phases into an existing
# results directory must not erase what the other phases already recorded.
touch "$STATUS"

note() { printf '[%s] %s\n' "$(date +%H:%M:%S)" "$*" | tee -a "$STATUS"; }
fail() { note "FAIL: $*"; exit 1; }
section() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; note "--- $* ---"; }

# run <logbase> <cmd...>: capture stdout/stderr, append the wall clock
run() {
    local lg="$1"; shift
    local t0 t1 rc
    t0=$(date +%s%3N)
    set +e
    "$@" > "${lg}.out" 2> "${lg}.log"
    rc=$?
    set -e
    t1=$(date +%s%3N)
    echo "WALL_MS $((t1 - t0)) RC $rc" >> "${lg}.log"
    return $rc
}

note "run_all.sh starting"
note "repo      $ROOT"
note "commit    $(git -C "$ROOT" rev-parse --short HEAD 2>/dev/null || echo '?') on $(git -C "$ROOT" rev-parse --abbrev-ref HEAD 2>/dev/null || echo '?')"
note "results   $OUT_DIR"
note "phases    $PHASES"
note "datasets  $DATASETS"

# ------------------------------------------------------------------ 1. env
if has_phase env; then
    section "1. Environment"
    command -v nvidia-smi >/dev/null 2>&1 || fail "nvidia-smi not found: this script needs a local NVIDIA GPU"
    command -v nvcc       >/dev/null 2>&1 || fail "nvcc not found: install the CUDA toolkit or put it on PATH"
    command -v cmake      >/dev/null 2>&1 || fail "cmake not found"

    nvidia-smi --query-gpu=name,compute_cap,memory.total,driver_version,clocks.sm \
               --format=csv > "$LOG/env_gpu.csv" 2>&1 || true
    nvcc --version > "$LOG/env_nvcc.txt" 2>&1 || true
    { lscpu 2>/dev/null | grep -E 'Model name|^CPU\(s\)|MHz' || true
      echo "--- mem ---"; free -g 2>/dev/null | head -2 || true; } > "$LOG/env_cpu.txt"

    if [[ -z "$ARCH" ]]; then
        cc="$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d ' .')"
        [[ -n "$cc" ]] || fail "could not detect compute capability; pass --arch"
        ARCH="$cc"
    fi
    note "GPU       $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) (sm_$ARCH)"
    note "nvcc      $(nvcc --version | tail -1 | tr -s ' ')"

    # Pinning the clocks is what makes two timed runs comparable. It needs
    # permissions; if it fails, the runs still work but the numbers move with
    # temperature, so the summary says so.
    CLOCKS_PINNED=no
    if [[ $LOCK_CLOCKS -eq 1 ]]; then
        maxsm="$(nvidia-smi --query-gpu=clocks.max.sm --format=csv,noheader 2>/dev/null | tr -dc '0-9')"
        if [[ -n "$maxsm" ]] && nvidia-smi -pm 1 >/dev/null 2>&1 \
           && nvidia-smi -lgc "$maxsm,$maxsm" >/dev/null 2>&1; then
            CLOCKS_PINNED="yes ($maxsm MHz)"
        fi
    fi
    note "clocks    pinned: $CLOCKS_PINNED"
    echo "$ARCH" > "$OUT_DIR/arch.txt"
else
    [[ -n "$ARCH" ]] || ARCH=75
fi
[[ -f "$OUT_DIR/arch.txt" ]] || echo "$ARCH" > "$OUT_DIR/arch.txt"
ARCH="$(cat "$OUT_DIR/arch.txt")"

# ------------------------------------------------------------------ 2. build
if has_phase build; then
    section "2. Build (CUDA, sm_$ARCH)"
    run "$LOG/cmake_configure" \
        cmake -S "$PROJ" -B "$BUILD_DIR" \
              -DCMAKE_BUILD_TYPE=Release \
              -DBUILD_TESTING=ON \
              -DTHESEUS_PROXY_ENABLE_CUDA=ON \
              -DCMAKE_CUDA_ARCHITECTURES="$ARCH" \
              -DCMAKE_CUDA_FLAGS='--ptxas-options=-v' \
        || fail "cmake configure (see $LOG/cmake_configure.log)"
    run "$LOG/cmake_build" cmake --build "$BUILD_DIR" -j "$JOBS" \
        || fail "compile (see $LOG/cmake_build.log)"
    note "built     $(grep -c . "$LOG/cmake_build.log" 2>/dev/null || echo 0) lines of build log"

    # --ptxas-options=-v puts registers, spill and shared per kernel into the
    # build log. This is the cheapest occupancy diagnostic there is.
    grep -E "Compiling entry function|Function properties for|Used [0-9]+ registers|stack frame|spill" \
        "$LOG/cmake_build.log" > "$LOG/ptxas_all.txt" 2>/dev/null || true
    if [[ -s "$LOG/ptxas_all.txt" ]]; then
        note "ptxas     per-kernel registers and spill in $LOG/ptxas_all.txt"
    fi
fi
[[ -x "$BIN" ]] || fail "$BIN not found: run the build phase first"

# ---------------------------------------------------------------- 3. verify
if has_phase verify; then
    section "3. Correctness"

    # 3a. ctest: the sample graph, CPU and GPU, against baseline/sample_output.gaf
    if run "$LOG/ctest" ctest --test-dir "$BUILD_DIR" --output-on-failure; then
        note "ctest     PASS ($(grep -oE '[0-9]+% tests passed, [0-9]+ tests failed out of [0-9]+' "$LOG/ctest.out" | head -1))"
    else
        note "ctest     FAIL (see $LOG/ctest.out)"
    fi

    # 3b. the sample graph through the GPU explicitly, with --require-gpu-result.
    # Without that flag a kernel that produced nothing still writes the CPU
    # fallback's alignments, which compare equal to a CPU golden and read as a
    # pass. This step exists to make that impossible.
    run "$LOG/sample_gpu" "$BIN" \
        -g "$PROJ/data/sample_graph.gfa" \
        -s "$PROJ/data/sample_queries.fasta" \
        -f "$OUT_DIR/sample_gpu.gaf" \
        --backend gpu --require-gpu-result --gpu-threads 128 || true
    if cmp -s "$OUT_DIR/sample_gpu.gaf" "$PROJ/baseline/sample_output.gaf"; then
        note "sample    PASS byte-identical to baseline/sample_output.gaf"
    else
        note "sample    FAIL differs from baseline/sample_output.gaf"
    fi

    # 3c. the real thing: ten GGBS datasets x three block sizes against the
    # frozen CPU goldens produced by cpu_oracle/.
    if run "$LOG/regression" python3 "$ROOT/scripts/run_ggbs_gpu_regression.py" \
            --suite all --build-dir "$BUILD_DIR" \
            --output-dir "$OUT_DIR/gpu_results" \
            --timeout "$REGRESSION_TIMEOUT"; then
        note "regression PASS $(grep -E '^[0-9]+/[0-9]+ datasets' "$LOG/regression.out" | tail -1)"
    else
        note "regression FAIL (see $LOG/regression.out)"
    fi
fi

# ------------------------------------------------------------------ 4. perf
if has_phase perf; then
    section "4. Performance"
    for ds in $DATASETS; do
        g="$(graph_of "$ds")"
        [[ -f "$Q/$ds.queries" ]] || { note "skip      $ds (no queries file)"; continue; }

        for t in $THREAD_SWEEP; do
            run "$LOG/gpu_${ds}_t${t}" "$BIN" \
                -g "$G/$g.gfa" -s "$Q/$ds.queries" -f "$OUT_DIR/${ds}_t${t}.gaf" \
                --backend gpu --require-gpu-result --gpu-threads "$t" \
                --repeat "$REPEAT" || note "warn      gpu $ds @$t returned non-zero"
            k="$(grep -oE 'kernel [0-9.]+ ms' "$LOG/gpu_${ds}_t${t}.log" | tail -1 || true)"
            note "gpu       $ds @${t}thr  last-iteration $k"
        done

        # The CPU aligner on the same input, same binary, same penalties: the
        # only comparison term that is not from another machine.
        run "$LOG/cpu_${ds}" "$BIN" \
            -g "$G/$g.gfa" -s "$Q/$ds.queries" -f "$OUT_DIR/${ds}_cpu.gaf" \
            --backend cpu || note "warn      cpu $ds returned non-zero"

        # A timing is only worth reporting if the run did the whole job.
        if [[ -f "$GOLD/${ds}.cpu.gaf" ]]; then
            cmp -s "$OUT_DIR/${ds}_cpu.gaf" "$GOLD/${ds}.cpu.gaf" \
                && note "cpu       $ds output GOLDEN_OK" \
                || note "cpu       $ds output DIFFERS from golden"
        fi
    done
fi

# --------------------------------------------------------------- 5. profile
if has_phase profile; then
    section "5. Profiling"

    # ptxas again, standalone: registers, spill and stack for the kernel, with
    # no build-system noise around it.
    if run "$LOG/ptxas_probe" nvcc -std=c++17 -arch="sm_$ARCH" -O3 \
            -I "$PROJ/src" -I "$PROJ/include" -Xptxas -v \
            -c "$PROJ/src/gpu/align_gpu.cu" -o /dev/null; then
        # The file compiles five kernels. Attribute the numbers to
        # theseus_align_batch_kernel by name: taking the last "Used N registers"
        # in the log reports whichever kernel ptxas happened to emit last.
        python3 "$ROOT/scripts/parse_ptxas.py" "$LOG/ptxas_probe.log" \
            > "$LOG/ptxas_kernel.txt" 2>/dev/null || true
        if [[ -s "$LOG/ptxas_kernel.txt" ]]; then
            note "ptxas     theseus_align_batch_kernel: $(tr '\n' ' ' < "$LOG/ptxas_kernel.txt")"
        else
            note "ptxas     could not attribute the numbers (see $LOG/ptxas_probe.log)"
        fi
    else
        note "ptxas     probe failed (see $LOG/ptxas_probe.log)"
    fi

    if [[ $USE_NCU -eq 1 ]] && command -v ncu >/dev/null 2>&1; then
        # --clock-control none: ncu otherwise pins the GPU to its *base* clock
        # and overrides nvidia-smi -lgc, which changes achieved bandwidth by
        # nearly 3x and makes the numbers incomparable with the untimed runs.
        M="gpu__time_duration.sum"
        M="$M,launch__registers_per_thread"
        M="$M,launch__occupancy_limit_registers"
        M="$M,launch__occupancy_limit_shared_mem"
        M="$M,sm__warps_active.avg.pct_of_peak_sustained_active"
        M="$M,smsp__inst_executed.sum"
        M="$M,smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio"
        M="$M,smsp__average_warps_issue_stalled_long_scoreboard_per_issue_active.ratio"
        M="$M,dram__bytes_read.sum"
        M="$M,dram__bytes_write.sum"
        M="$M,dram__throughput.avg.pct_of_peak_sustained_elapsed"
        M="$M,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio"
        M="$M,local__store_bytes.sum"
        M="$M,smsp__thread_inst_executed_per_inst_executed.ratio"

        for ds in $DATASETS; do
            g="$(graph_of "$ds")"
            [[ -f "$Q/$ds.queries" ]] || continue
            set +e
            ncu --clock-control none \
                --kernel-name theseus_align_batch_kernel \
                --launch-count 1 --metrics "$M" --csv \
                "$BIN" --backend gpu --require-gpu-result --gpu-threads 128 \
                -g "$G/$g.gfa" -s "$Q/$ds.queries" -f "$OUT_DIR/ncu_$ds.gaf" \
                > "$OUT_DIR/ncu/${ds}.csv" 2> "$LOG/ncu_${ds}.log"
            rc=$?
            set -e
            if [[ $rc -eq 0 ]]; then
                note "ncu       $ds -> ncu/${ds}.csv ($(wc -l < "$OUT_DIR/ncu/${ds}.csv") lines)"
            else
                note "ncu       $ds failed rc=$rc (profiling often needs"
                note "          NVreg_RestrictProfilingToAdminUsers=0 or root; see $LOG/ncu_${ds}.log)"
            fi
        done
    elif [[ $USE_NCU -eq 1 ]]; then
        note "ncu       not installed, skipping Nsight Compute"
    fi
fi

# --------------------------------------------------------------- 6. summary
if has_phase summary; then
    section "6. Summary"
    REPORT="$OUT_DIR/REPORT.md"
    python3 - "$OUT_DIR" "$LOG" "$REPORT" "$DATASETS" "$THREAD_SWEEP" <<'PY'
import csv, os, re, sys

out_dir, log_dir, report, datasets, threads = sys.argv[1:6]
datasets = datasets.split()
threads = threads.split()

def read(p):
    try:
        with open(p, errors="replace") as f:
            return f.read()
    except OSError:
        return ""

def kernel_times(path):
    """Every --repeat iteration prints its own timing line. Iteration 0 pays the
    per-process costs (CUDA context, graph upload, page locking) and the full
    ScratchPad clear, so the steady state is the median of the rest."""
    vals = [float(m) for m in re.findall(r"kernel ([0-9.eE+-]+) ms", read(path))]
    return vals

def median(xs):
    if not xs:
        return None
    s = sorted(xs)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

def cpu_ms(path):
    m = re.search(r"Aligned \d+ sequences in (\d+) microseconds", read(path))
    return float(m.group(1)) / 1000.0 if m else None

lines = ["# run_all.sh report", ""]
status = read(os.path.join(out_dir, "status.txt"))

# --- environment
lines += ["## Environment", "", "```"]
lines += [read(os.path.join(log_dir, "env_gpu.csv")).strip()]
lines += [read(os.path.join(log_dir, "env_nvcc.txt")).strip().splitlines()[-1:][0]
          if read(os.path.join(log_dir, "env_nvcc.txt")).strip() else ""]
lines += [read(os.path.join(log_dir, "env_cpu.txt")).strip(), "```", ""]

# --- correctness
lines += ["## Correctness", ""]
for key, label in (("ctest", "ctest"), ("sample", "sample baseline"),
                   ("regression", "GGBS regression")):
    hit = [l for l in status.splitlines() if re.search(rf"\]\s+{key}\b", l)]
    msg = hit[-1].split("]", 1)[1].strip() if hit else "not run"
    lines += [f"- **{label}**: " + re.sub(rf"^{key}\s+", "", msg)]
lines += [""]

# --- performance
rows = []
for ds in datasets:
    cpu = cpu_ms(os.path.join(log_dir, f"cpu_{ds}.out"))
    row = {"dataset": ds, "cpu": cpu}
    for t in threads:
        vals = kernel_times(os.path.join(log_dir, f"gpu_{ds}_t{t}.log"))
        steady = median(vals[1:]) if len(vals) > 1 else (vals[0] if vals else None)
        row[t] = steady
    rows.append(row)

if any(any(r.get(t) for t in threads) for r in rows):
    lines += ["## Kernel time, steady state (ms)", "",
              "Median of iterations 1..n-1; iteration 0 pays the per-process "
              "costs and the full ScratchPad clear, so it is dropped by "
              "construction.", ""]
    head = "| dataset | " + " | ".join(f"{t} thr" for t in threads) + \
           " | CPU (1 core, whole align) | best ratio |"
    lines += [head, "|" + "---|" * (len(threads) + 3)]
    for r in rows:
        cells = []
        best = None
        for t in threads:
            v = r.get(t)
            cells.append(f"{v:.3f}" if v is not None else "-")
            if v is not None and (best is None or v < best):
                best = v
        cpu = r["cpu"]
        ratio = f"{cpu / best:.1f}x" if (cpu and best) else "-"
        lines += ["| `%s` | %s | %s | %s |" % (
            r["dataset"], " | ".join(cells),
            f"{cpu:.3f}" if cpu else "-", ratio)]
    lines += ["",
              "> The ratio compares the **kernel** with the CPU doing the whole "
              "alignment, so it favours the GPU. It is not an end-to-end "
              "speedup: the host still pays backtrace, alignment construction "
              "and GAF serialisation in both cases.", ""]

# --- occupancy / registers, for theseus_align_batch_kernel specifically
ptx = read(os.path.join(log_dir, "ptxas_kernel.txt"))
if ptx.strip():
    vals = dict(l.split("=", 1) for l in ptx.split() if "=" in l)
    lines += ["## Kernel resources (ptxas, `theseus_align_batch_kernel`)", "",
              f"- registers per thread: {vals.get('registers', '?')}",
              f"- stack frame: {vals.get('stack', '?')} B",
              f"- spill stores: {vals.get('spill', '?')} B",
              "",
              "Spill inside the kernel's own frame is what limits occupancy; the "
              "`__noinline__` device functions it calls report their own frames "
              "separately in `logs/ptxas_probe.log`.", ""]

# --- ncu
ncu_dir = os.path.join(out_dir, "ncu")
csvs = sorted(f for f in os.listdir(ncu_dir)) if os.path.isdir(ncu_dir) else []
wanted = {
    "sm__warps_active.avg.pct_of_peak_sustained_active": "achieved occupancy %",
    "dram__throughput.avg.pct_of_peak_sustained_elapsed": "DRAM % of peak",
    "smsp__average_warps_issue_stalled_barrier_per_issue_active.ratio": "barrier stall",
    "smsp__thread_inst_executed_per_inst_executed.ratio": "threads active / inst",
    "launch__registers_per_thread": "registers/thread",
}
ncu_rows = []
for name in csvs:
    path = os.path.join(ncu_dir, name)
    text = read(path)
    start = text.find('"ID"')
    if start < 0:
        continue
    try:
        rdr = list(csv.DictReader(text[start:].splitlines()))
    except Exception:
        continue
    vals = {}
    for r in rdr:
        m = (r.get("Metric Name") or "").strip()
        if m in wanted:
            vals[wanted[m]] = (r.get("Metric Value") or "").strip()
    if vals:
        ncu_rows.append((name.replace(".csv", ""), vals))
if ncu_rows:
    cols = list(wanted.values())
    lines += ["## Nsight Compute (128 threads, first launch)", "",
              "| dataset | " + " | ".join(cols) + " |",
              "|" + "---|" * (len(cols) + 1)]
    for ds, vals in ncu_rows:
        lines += ["| `%s` | %s |" % (ds, " | ".join(vals.get(c, "-") for c in cols))]
    lines += [""]

lines += ["## Raw data", "",
          "- `logs/` one `.log`/`.out` pair per run, each with its `WALL_MS`"]
if ncu_rows:
    lines += ["- `ncu/` Nsight Compute CSV per dataset"]
lines += ["- `status.txt` the phase-by-phase trace", ""]

with open(report, "w") as f:
    f.write("\n".join(l for l in lines if l is not None) + "\n")
print(open(report).read())
PY
    note "report    $REPORT"
fi

section "Done"
note "results in $OUT_DIR"
