# GPU_genomics_project — Theseus on CUDA

A GPU port of **Theseus**, a sequence-to-graph aligner. It takes a variation
graph in GFA format and a set of query sequences, each seeded at a known
position, and writes the minimum-penalty alignment path as GAF: coordinates,
score and CIGAR. This is read mapping, the same task as BWA or minimap2.

The algorithm is **WFA** (wavefront alignment) with gap-affine penalties,
generalised from a linear reference to a graph. WFA explores by increasing
score instead of filling a DP matrix: for each score it keeps only the furthest
cell per diagonal, in three wavefronts (`M`, `I`, `D`), and extends them along
matches. On a graph a cell is `(query position, vertex, offset in the vertex)`,
and a cell that reaches the last column of a vertex has to continue in *every*
successor — the "jumps", which are the inherently ordered part of the work.

---

## Build

CPU only — no CUDA toolkit needed, the GPU backend compiles to a stub that falls
back to the CPU path:

```bash
cd theseus_gpu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON
cmake --build build -j
```

With CUDA (needs `nvcc`; developed against a T4):

```bash
cd theseus_gpu
cmake -S . -B build-gpu -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=ON \
      -DTHESEUS_PROXY_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=75
cmake --build build-gpu -j
```

Set `-DCMAKE_CUDA_ARCHITECTURES` to your device (75 = T4). Add
`-DCMAKE_CUDA_FLAGS='--ptxas-options=-v'` to see registers and spill per kernel.

Run the aligner:

```bash
./build-gpu/apps/seq2graph_proxy \
  -g data/sample_graph.gfa -s data/sample_queries.fasta -f out.gaf \
  -b gpu --gpu-threads 64 --require-gpu-result
```

**Always pass `--require-gpu-result`.** Without it a kernel that produced nothing
still writes the CPU fallback's alignments, which compare equal to a CPU golden
and read as a pass.

---

## Tests

Sample graph, both backends, against `theseus_gpu/baseline/sample_output.gaf`:

```bash
ctest --test-dir build-gpu --output-on-failure
```

Full CPU-vs-GPU regression: ten GGBS datasets at 64, 128 and 256 threads per
block, each diffed against its frozen golden. `simple` is the score-0 datasets,
`complex` the ones whose reads carry errors.

```bash
python3 scripts/run_ggbs_gpu_regression.py --suite simple  --build-dir theseus_gpu/build-gpu
python3 scripts/run_ggbs_gpu_regression.py --suite complex --build-dir theseus_gpu/build-gpu
python3 scripts/run_ggbs_gpu_regression.py --suite all     --build-dir theseus_gpu/build-gpu
```

Everything in one command — `scripts/run_all.sh` detects the architecture,
builds with CUDA on, runs ctest and the full regression, times the kernel at
64/128/256 threads against the CPU on the same machine, collects ptxas registers
and an Nsight Compute pass if `ncu` is installed, and writes a single
`REPORT.md`:

```bash
scripts/run_all.sh                  # everything: ~3 min on a T4
scripts/run_all.sh --quick          # smoke datasets for the timings, no ncu
scripts/run_all.sh --phases verify  # just the correctness phase
```

`--quick` shortens only the timing datasets; the ten-dataset regression runs
either way. Results land in `profiling/run_all_<timestamp>/`: `REPORT.md`,
the live phase trace, one log per run, the Nsight CSVs and the GAF produced.
Other options: `--arch NN`, `--threads "64 128"`, `--datasets c4_err_2k`,
`--repeat N`, `--no-ncu`, `--out-dir <dir>`. `scripts/run_all.sh --help` for all.

Under `compute-sanitizer`, memcheck / racecheck / synccheck / initcheck are
clean. Re-run `--tool initcheck` after touching the `QueryState` zeroing: the
regression passes either way and will not tell you.

---

## Optimisations

The six CUDA optimisation categories. Everything below was implemented and
measured on a Tesla T4, and every change had to keep the output byte-identical
to the goldens.

| Category | Where it was applied | What was done |
|---|---|---|
| **1. Privatization** | **One block per query**<br>**Slot allocation without atomics** | A block owns one query and its whole 3.95 MB `QueryState`, so two blocks never contend and nothing is left to privatise between them. Inside the block, the two places a naive design would use `atomicAdd` — appending to `sp_diags` and to the wavefronts — go through a ballot-based prefix sum on shared memory instead (`block_prefix_alloc`). Putting the ScratchPad itself in shared is impossible: 1.2 MB against 64 KB per SM. |
| **2. Thread coarsening** | **Four diagonals per thread in the ScratchPad clear** | The clear (`fill_words`) writes one word per diagonal across the whole window, so each thread takes four of them and issues a single 128-bit `int4` store, with the ragged ends filled one word at a time because the alignment is derived at run time. **Instructions −38 %, 1.27×.** Coarsening `densify` was measured before writing it and dropped: the mean wavefront is 2.5 diagonals, one tile even at 64 threads. |
| **3. Coalescing** | **The ScratchPad written word-wise, not `Cell`-wise**<br>**Hot/cold split of the ScratchPad** | Storing whole `Cell`s meant a 24-byte stride, where every field store reached across the warp's entire span. Rewriting the clear word-wise halved DRAM traffic and left the duration **unchanged** — a negative result, kept, because it is what disproved the bandwidth-bound hypothesis. Moving the only field read while a cell is inactive into its own dense array (`sp_off`, leaving the payload in `sp_wf`) then cut the clear from six words per diagonal to one: **kernel 1.14–7.86×**, the largest single speedup of the project. |
| **4. Divergence / serialisation** | **A warp per cell in the LCP**<br>**The score-0 seed extension off thread 0** | The LCP advanced one character at a time on one thread; now 32 lanes compare 32 characters, `__ballot_sync` turns that into a mask and `__ffs` on its complement gives the advance (`warp_lcp`). It returns the same `offset` and `j` as the serial version — verified on 400 000 simulated cases, 0 divergences. Measuring the seed extension on warp 0 (`warp_seed_offset`), instead of leaving it alone on thread 0 while the block waits, is worth **1.37–1.38×** on the simple tier. |
| **5. Tiling of reused data** | **The query text staged in shared memory** | The query is the one input every `M` cell re-reads, so it is copied once per block into 1 KB of shared (`kQueryTileBytes`), with a fallback to global above that size so nothing depends on the bound being generous. **Registers 239 → 226.** Measured before committing, and the measurement said the headroom was small: global loads were already at 1.03–1.23 sectors per request. |
| **6. Occupancy** | **One frame boundary in the per-vertex call chain**<br>**A register cap on the kernel** | nvcc inlined the whole per-vertex chain into one body, so the kernel paid the union of every phase's live set at once. One `__noinline__` on `process_vertex`, at the top of that chain, gives the per-vertex work its own frame: **226 → 138 registers with 0 spill**, better on both fronts than an 8-point `-maxrregcount` probe. `__launch_bounds__` then holds ptxas at 128. It also produced the counter-example: more occupancy is not faster, 128 threads has the highest occupancy and is 1.31× slower than 64. |
| **Outside the categories** | **The traceback download**<br>**Correctness under the sanitizers** | The traceback D2H was 95 % of GPU time because it copied three fixed-size arrays per query; packing it on the device into a dense prefix-summed buffer (`pack_traceback_kernel`) made it **154–244× faster** (46.8 → 0.30 ms), and persistent pinned host buffers took it from 1.4 to 12.3 GiB/s. `compute-sanitizer --tool initcheck` caught a defect the regression could not see, which is why the `QueryState` array is zeroed once per allocation. |
