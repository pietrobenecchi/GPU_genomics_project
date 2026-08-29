# GPU_genomics_project — Theseus on CUDA

A GPU port of **Theseus**, a sequence-to-graph aligner. Given a query sequence
and a variation graph in GFA format it finds the minimum-penalty alignment path
through the graph and writes it out as GAF. The goal of the project is to move
the aligner onto CUDA while keeping the GPU output **byte-identical** to the CPU
baseline, and to measure what the port actually buys.

---

## 1. How the repository is organised

```
theseus_gpu/         ← THE PROJECT. The aligner under development: host code + CUDA kernel
cpu_oracle/          ← the frozen reference CPU aligner, read-only, produces the goldens
papers/              ← the two papers the work is based on
scripts/             ← build/run/profile driver, dataset conversion, regression
profiling/           ← the profiling campaigns, with their raw data
tests/               ← unit tests for the conversion tools
external_datasets/   ← vendored GGBS benchmark + Zenodo archives (untracked, ~2.7 GB)
```

**`theseus_gpu/` is the real project** — everything under development lives here:

```
theseus_gpu/
  src/gpu/            the CUDA backend
    align_gpu.cu      the kernel and the host side of the batch launch
    align_gpu.h       the host<->device boundary: POD types only
    align_core.h      the device-callable algorithm (shared with the CPU verifier)
    align_gpu_stub.cpp  compiled instead of the .cu when CUDA is off
    graph_csr.{h,cpp} the graph in CSR form, uploaded once and kept resident
  src/                the flattened CPU implementation: fallback, and the
                      verifier the kernel is checked against
  include/theseus/    public headers
  apps/               seq2graph_proxy (the CLI), seq2graph_gpu_validate,
                      seq2graph_gpu_benchmark
  data/               tiny sample graph/queries, plus data/validation/ggbs
  baseline/           frozen CPU output: the regression oracle
  docs/               optimisation log, parallelism analysis, dataset report
  GPU_PORT_PROMPT.md  the porting plan and its constraints
```

**`cpu_oracle/`** is the pre-flattening CPU aligner, frozen at commit `56663af`.
It is not part of the port and is never modified: it exists to produce the golden
GAF files that every GPU run is compared against. Each golden's manifest records
the oracle binary's SHA-256 that produced it.

**`profiling/`** holds three campaigns, each with the raw logs and CSVs behind
its conclusions: `baseline_naive_88ddef2/` (the first naive GPU version),
`campagna_cuda/` (the per-category CUDA analysis) and `campagna_finale/` (the
closing report — `REPORT_FINALE.md` is the single document with all the numbers).

> `profiling/`, `profile-cuda/` and `theseus_gpu/docs/` are **working material
> kept out of the repository** (see `.gitignore`): notes, campaign reports and
> 161 MB of raw Nsight dumps. They live in the working tree, so the paths
> referenced throughout this README resolve on a development machine but are not
> part of a clone. Everything a clone needs to build, run and validate the
> aligner is tracked.

Both CMake projects still declare the project and library name `theseus_proxy`.

---

## 2. What the algorithm does

Theseus is **WFA** (wavefront alignment) with gap-affine penalties, generalised
from a linear reference to a graph.

WFA explores the alignment by increasing score instead of filling a
dynamic-programming matrix: for score `s` it stores, per diagonal, the furthest
cell reachable at that score, and extends it along matches. The matrix is never
materialised — only the wavefronts are. Three wavefronts are kept per score, one
per state of the gap-affine model: `M` (match/mismatch), `I` (insertion) and `D`
(deletion).

The graph changes what a diagonal is. A cell is not `(query position, reference
position)` but `(query position, vertex, offset inside that vertex)`. One step
of the algorithm is:

1. **`next_I` / `next_D` / `next_M`** — build the candidate cells for score `s`
   from the wavefronts at `s-x`, `s-o-e` and `s-e`: *sparsify* them into a
   per-diagonal scratchpad, then *densify* back into a compact wavefront.
2. **`extend_diagonal`** — extend each `M` cell along the longest common prefix
   between the query suffix and the vertex text.
3. **`check_and_store_jumps` / `store_M_jump` / `store_I_jump`** — a cell that
   reaches the last column of a vertex has to continue in *every* successor. It
   is pushed as a "jump" on each out-edge, the successor vertex is activated,
   and extension continues there. This is the part with no counterpart in linear
   WFA, and the part that is inherently ordered.
4. The score is incremented and the loop repeats until a cell reaches the end of
   the query; a **backtrace** then walks the stored jumps back to the seed and
   produces the path and the CIGAR.

Each query is seeded at a position given in the input (start vertex, offset,
orientation), so the aligner extends from a known anchor rather than searching
the whole graph. Penalties default to match 0, mismatch 2, gap-open 3,
gap-extend 1.

**On the GPU the mapping is one CUDA block per query**: `threadIdx.x` is
cooperation *inside* a single query (densify, the LCP, the scratchpad clear, the
seed extension), never a second query. Per-query state is one 3.95 MB
`QueryState` in global memory, entirely private to its block, so no two blocks
contend. The graph is uploaded once in CSR form and stays resident; batches are
chunked from `cudaMemGetInfo` at call time so a batch larger than device memory
no longer fails.

> **CIGAR convention.** The emitted CIGAR has `I` and `D` **inverted relative to
> SAM**: here `M`, `X` and `D` consume the query and `I` does not. Every golden
> satisfies `M + X + D == query length`.

---

## 3. What was implemented, by CUDA optimisation category

Every entry below was implemented and measured on a Tesla T4, and every one had
to keep the output byte-identical to the goldens. The score is **how far the
category was explored and exploited where it made sense**, not a percentage of
performance. Full detail, with the measurement that motivated and the one that
verified each change, is in `theseus_gpu/docs/optimization_log.md` (append-only)
and `profiling/campagna_finale/REPORT_FINALE.md` §13.

| # | Category | /100 | What was implemented | Measured result |
|---|---|---:|---|---|
| 1 | **Privatization** | 85 | One block per query, no state shared between queries; slot allocation by shared-memory prefix sum instead of atomics; control scalars, sparsify plan and candidate staging kept in shared | No global atomics on the hot path. The one remaining candidate — the ScratchPad in shared — is impossible: 1.2 MB against 64 KB per SM |
| 2 | **Thread coarsening** | 75 | Two opposite experiments: ScratchPad clear at 4 words/thread with `int4` stores (**kept**), and coarsening of `densify` (**rejected before writing it**, from the wavefront histogram) | Clear: instructions −38 %, 1.27× faster, indices verified over 95 468 combinations. Densify: mean wavefront is 2.5 diagonals, max 41 — a single tile even at 64 threads |
| 3 | **Coalescing** | 90 | Clear rewritten word-wise instead of `Cell`-wise; hot/cold split of the ScratchPad so the frequently written field is a dense separate array | First: traffic halved, **duration unchanged** — a negative result, kept, because it disproved the bandwidth-bound hypothesis. Second: kernel **1.14–7.86×**, the largest single speedup of the project |
| 4 | **Divergence / serialisation** | 70 | Warp-cooperative LCP (`__ballot_sync` + `__ffs`); score-0 seed extension measured on warp 0 instead of thread 0 | 0 divergences over 400 000 simulated cases; the seed change is worth **1.37–1.38×** on the simple tier |
| 5 | **Tiling of reused data** | 65 | Query text staged in shared memory, 1 KB per block, with a fallback to global above the size threshold | Registers 239 → 226. Measured *before* committing: global loads were already at 1.03–1.23 sectors per request, so the headroom was small and known to be small |
| 6 | **Occupancy** | 88 | Full analysis; an 8-point `-maxrregcount` probe (run locally, no GPU needed); then a natural live-set reduction with one `__noinline__` at the top of the call chain | **226 → 138 registers with 0 spill** — better than the probe on both fronts. Achieved occupancy 33–37 %, and the demonstration that more occupancy is *not* faster: 128 threads has the highest occupancy and is 1.31× slower than 64 |

Two further pieces of work do not belong to a single category:

* **Data transfer** — the traceback D2H was 95 % of GPU time because it copied
  three fixed 4096-cell arrays per query. Replacing it with a prefix-sum-packed
  dense buffer made the D2H **154–244×** faster (46.8 → 0.30 ms); 99.79 % of the
  transfer had been slack. Persistent page-locked host buffers took the D2H from
  1.4 to 12.3 GiB/s.
* **Correctness under sanitizers** — memcheck, racecheck and synccheck are
  clean; **initcheck** caught a real defect the regression could not see
  (dropping the per-batch `cudaMemset` left the score-0 seed extension reading
  uninitialised device memory), which is why the `QueryState` array is zeroed
  once per *allocation*.

**CUDA streams were never attempted** and are reported as such, not as rejected.

### Where it ends up

Tesla T4 (sm_75) against one core of an Intel Xeon @ 2.00 GHz, same machine,
same binary, 2048-query batches, kernel at steady state:

| dataset | CPU (whole alignment) | GPU kernel | ratio |
|---|---:|---:|---:|
| `c4_exact_2k` | 3.484 ms | 0.281 ms | 12.4× |
| `ebola_exact_2k` | 3.570 ms | 0.264 ms | 13.5× |
| `c4_err_2k` | 7.623 ms | 2.255 ms | 3.4× |
| `ebola_err_2k` | 7.554 ms | 2.191 ms | 3.4× |

**End to end the GPU pipeline does not beat the single-threaded CPU**, and no
end-to-end speedup should be claimed on these inputs: the reads are 100 bp, so
the work per query is a few microseconds even on the CPU, and it does not repay
the fixed per-process cost nor the host-side backtrace, alignment construction
and GAF serialisation that both paths pay. The residual bottleneck inside the
kernel is the ordered serial work on thread 0 — `check_and_store_jumps` and the
M-jump DFS, i.e. the out-edge fan-out.

---

## 4. The `papers/` folder

Two papers, one per side of the work:

* **`theseus_paper.pdf`** — *Theseus: Fast and Optimal Affine-Gap
  Sequence-to-Graph Alignment* (Jiménez-Blanco, López-Villellas, Moure, Moretó,
  Marco-Sola; bioRxiv 2026). The algorithm being ported. It is the source for
  the diagonal-transition formulation on a graph, the sparse-data strategy that
  makes it tractable, and the optimality guarantee the port has to preserve —
  which is exactly why the acceptance criterion here is byte-identity rather
  than "close enough".
* **`TSUNAMI_A_GPU_Implementation_of_the_WFA_Algorithm.pdf`** — *TSUNAMI: a GPU
  implementation of the WFA algorithm* (Gerometta, Zeni, Santambrogio, PACT
  2023). The reference point for putting WFA on a GPU: how the wavefronts are
  laid out, how work is mapped to blocks and threads, and what its reported
  speedups do and do not cover. It handles the **linear** case, so it gives the
  parallelisation patterns but not the graph traversal — the jumps across
  out-edges, which are the ordered part, have no equivalent there.

---

## 5. Build

Assumes a local NVIDIA GPU with the CUDA toolkit installed (`nvcc` and
`nvidia-smi` on `PATH`) and CMake ≥ 3.18.

```bash
cd theseus_gpu
cmake -S . -B build-gpu \
      -DCMAKE_BUILD_TYPE=Release \
      -DBUILD_TESTING=ON \
      -DTHESEUS_PROXY_ENABLE_CUDA=ON \
      -DCMAKE_CUDA_ARCHITECTURES=75 \
      -DCMAKE_CUDA_FLAGS='--ptxas-options=-v'
cmake --build build-gpu -j"$(nproc)"
ctest --test-dir build-gpu --output-on-failure
```

Set `-DCMAKE_CUDA_ARCHITECTURES` to your device (75 = T4; check with
`nvidia-smi --query-gpu=compute_cap --format=csv,noheader`).
`--ptxas-options=-v` makes the build log print registers, spill and shared
memory per kernel — the cheapest occupancy diagnostic available.

---

## 6. Run

```bash
cd theseus_gpu
./build-gpu/apps/seq2graph_proxy \
  -g data/sample_graph.gfa \
  -s data/sample_queries.fasta \
  -f out.gaf \
  --backend gpu --gpu-threads 64 --require-gpu-result
```

```
  -m, --match <int>            Match penalty (default 0)
  -x, --mismatch <int>         Mismatch penalty (default 2)
  -o, --gapo <int>             Gap-open penalty (default 3)
  -e, --gape <int>             Gap-extend penalty (default 1)
  -g, --graph_file <file>      Graph in GFA format (required)
  -s, --sequences_file <file>  FASTA with start metadata (required)
  -f, --output_file <file>     Output GAF path (required)
  -b, --backend <cpu|gpu>      Alignment backend (default cpu)
      --gpu-threads <64|128|256>  Threads per GPU block (default 128)
      --require-gpu-result     Fail unless the output came from the GPU kernel
      --verify-gpu-with-cpu    Also run the CPU aligner and compare endpoints
      --repeat <int>           Align the same batch N times through one aligner
                               (measurement only: separates per-batch cost from
                               per-process cost)
```

**Always pass `--require-gpu-result`.** Without it, a kernel that produced
nothing still writes the CPU fallback's alignments, which compare equal to a CPU
golden and read as a pass. `THESEUS_GPU_CHUNK=<n>` forces the batch chunk size;
it is the only way to reach the chunking path with a small batch.

Input format — one metadata line per record, sequence on the next; orientation
is `+` or `-` and defaults to `+`:

```text
> <start_vertex_name> <start_offset> <orientation>
<sequence>
```

---

## 7. Build, run, validate and profile in one command

`scripts/run_all.sh` does the whole loop on a machine with a local GPU: it
detects the architecture, builds with CUDA on, runs ctest and the full
CPU-vs-GPU regression, times the kernel across block sizes against the CPU on
the same machine, collects `ptxas` registers/spill and — if `ncu` is installed —
an Nsight Compute pass, then writes one `REPORT.md` out of all of it.

```bash
scripts/run_all.sh                 # everything: ~3 min on a T4
scripts/run_all.sh --quick         # smoke datasets for the timings, no ncu: ~2 min
scripts/run_all.sh --phases verify # just the correctness phase
```

`--quick` shortens only the *timing* datasets: the full ten-dataset regression
runs either way, because correctness is not the part worth cutting.

Results land in `profiling/run_all_<timestamp>/`:

```
REPORT.md      the summary: environment, correctness, kernel times, ptxas, ncu
status.txt     the phase-by-phase trace, printed live while it runs
logs/          one .log/.out pair per run, each ending in its own WALL_MS
ncu/           one Nsight Compute CSV per dataset
gpu_results/   the GAF the regression produced
```

`REPORT.md` looks like this (a real T4 run, abridged):

```markdown
## Correctness
- **ctest**: PASS (100% tests passed, 0 tests failed out of 5)
- **sample baseline**: PASS byte-identical to baseline/sample_output.gaf
- **GGBS regression**: PASS 10/10 datasets passed (...)

## Kernel time, steady state (ms)
| dataset          | 64 thr | 128 thr | 256 thr | CPU (1 core) | best ratio |
| `c4_exact_2k`    |  0.208 |   0.274 |   0.674 |        3.354 |      16.1x |
| `c4_err_2k`      |  1.673 |   2.251 |   5.050 |       12.543 |       7.5x |

## Kernel resources (ptxas, `theseus_align_batch_kernel`)
- registers per thread: 138 · stack frame: 768 B · spill stores: 0 B
```

The timings are the median of iterations 1..n-1: iteration 0 pays the CUDA
context, the graph upload, the page locking and the full ScratchPad clear, so it
is dropped by construction rather than averaged in.

Useful options: `--arch NN` (skip detection), `--threads "64 128"`,
`--datasets "c4_err_2k"`, `--repeat N`, `--no-ncu`, `--no-clock-lock`,
`--out-dir <dir>`, `--build-dir <dir>`. Run `scripts/run_all.sh --help` for all
of them. Re-running a subset of the phases into an existing `--out-dir` keeps
what the other phases already recorded, so `--phases profile,summary` refreshes
the profile without losing the correctness results above it.

Two things the script handles that are easy to get wrong by hand: it pins the SM
clocks with `nvidia-smi -lgc` before timing (and says so in the report if it
could not), and it passes `--clock-control none` to `ncu`, which otherwise pins
the GPU to its *base* clock and silently overrides that lock — on a T4 that is
585 MHz against 1590, which moves achieved-bandwidth numbers by nearly 3×.

If `ncu` reports `ERR_NVGPUCTRPERM`, profiling counters are restricted to root
on your system; run that phase with `sudo`, or set
`NVreg_RestrictProfilingToAdminUsers=0` for the `nvidia` module. The rest of the
script does not need elevated privileges.

---

## 8. Validation

The rule that governs every change: `--backend gpu` output must stay
**byte-identical** to the frozen CPU goldens. Regression authority is CPU-vs-GPU
exact equality, not the benchmark's approximate position truth.

* `ctest` covers the sample graph on both backends against
  `theseus_gpu/baseline/sample_output.gaf`.
* The dataset-level check runs the kernel at 64, 128 and 256 threads per block
  over ten GGBS datasets and diffs each against its golden:

```bash
python3 scripts/run_ggbs_gpu_regression.py --suite simple  --build-dir theseus_gpu/build-gpu
python3 scripts/run_ggbs_gpu_regression.py --suite complex --build-dir theseus_gpu/build-gpu
python3 scripts/run_ggbs_gpu_regression.py --suite all     --build-dir theseus_gpu/build-gpu
```

`simple` is the score-0 datasets; `complex` is the ones whose reads carry errors
and therefore need at least one non-zero-score wavefront. **30/30 byte-identical**
on the final commit.

Under `compute-sanitizer`, memcheck / racecheck / synccheck / initcheck are all
clean. Re-run `--tool initcheck` after touching the `QueryState` zeroing: the
regression passes either way and will not tell you.

Supporting tools: `scripts/ggbs_json_to_theseus_queries.py` converts GGBS JSON
into Theseus queries, `scripts/generate_ggbs_cpu_golden.py` freezes a CPU
reference plus its manifest, and `tests/test_ggbs_tools.py` unit-tests the
conversion.

### Datasets

Validation uses **GGBS** (Genome Graphs Benchmark Suite), a published benchmark
for sequence-to-graph aligners — [Zenodo record
12207360](https://zenodo.org/records/12207360),
[repository](https://github.com/Mirkocoggi/GGBS). Its directories are named after
the organism each reference genome comes from (`ebola`, `covid`, `yeast`, `C4`,
`MHC`, `ecoli`); those names are kept identical to upstream purely as labels for
benchmark inputs, so results stay comparable with the published benchmark. The
graphs were picked for their topology, not their biology: `ebola` is a
7-node/8-edge graph used as the fastest smoke test, `c4` is the larger one. Reads
are simulated by the benchmark, with a matched "error" variant that exercises the
mismatch and indel paths.

---

## 9. Known bounds

`kScratchpadSpan` is derived, not guessed: `max_vertex_length + max_query_length
+ 1` (9164 for ebola, 52107 for c4, hence the constant 52224). One `QueryState`
is 3.95 MB, so a graph with longer vertices needs either a bigger value or a
sparse ScratchPad; `align_batch_gpu` reports the required span before launching.

The `Scope`, `BeyondScope` and `VerticesData` bounds are still provisional, tuned
on a toy dataset where wavefronts reach 4 cells. Overflow is never silent:
`cap_fail` records which buffer ran out and by how much.

---

## 10. Further reading

* `theseus_gpu/GPU_PORT_PROMPT.md` — the porting plan and its constraints.
* `theseus_gpu/docs/optimization_log.md` — every performance change with the
  numbers that motivated and verified it (append-only).
* `theseus_gpu/docs/gpu_parallelism_analysis.md` — static analysis of the kernel.
* `theseus_gpu/docs/ggbs_validation_dataset_report.md` — the validation dataset.
* `profiling/campagna_finale/REPORT_FINALE.md` — the final performance report.
