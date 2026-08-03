# Minimal repro: CPU aligner does not terminate on a non-zero score

Found 2026-08-03 while generating the GGBS CPU reference outputs.

## Symptom

On the GGBS `C4` graph the CPU backend returns instantly for a read that matches
the graph exactly, and does not terminate for the same read with a single
substitution. Insertion and deletion behave like the substitution. Resident
memory stays flat at ~2.3 MB throughout, because the runaway is in a recycled
ring buffer rather than in a growing allocation.

## Root cause

Confirmed under gdb, 8 s into the `c4_sub1` run:

```
_score                  = 158574429      # racing at ~25M/s
_end                    = false
_qs->capacity_exceeded  = true
_qs->sc_peak_wf         = 0              # no cell was ever stored in a Scope wavefront
```

The chain is:

1. Node `2` of C4 is 6388 bp, so `next_I`/`next_D`/`next_M` run with
   `upper_bound = 6388`, far past the fixed capacities in `query_state.h`
   (`kScopePosCapacity = kScopeWavefrontCapacity = 1024`, `kMaxScores = 8`).
2. `sc_pos_push` (`query_state.h:401`) reacts to overflow by setting
   `capacity_exceeded` and **returning without storing the range**. The write is
   dropped and the caller carries on as if it had succeeded.
3. With the ranges gone the wavefronts stay empty (`sc_peak_wf == 0`), so no M
   cell ever satisfies `curr_data.offset == _seq.size()` in
   `check_end_condition` (`theseus_aligner_impl.cpp:611`), the only place that
   sets `_end`.
4. `while (!_end)` in `align` (`theseus_aligner_impl.cpp:205`) therefore never
   exits, incrementing `_score` forever.
5. Nothing reads `capacity_exceeded` on this path until after `align()` returns.
   The check at `theseus_aligner_impl.cpp:101` belongs to
   `alignment_from_gpu_result`, and `theseus_aligner.cpp:299` runs after the
   call. Both are unreachable while the loop spins.

`_score` is `int32_t`, so at ~25M/s it reaches `INT_MAX` in roughly 85 s and the
increment becomes signed overflow — undefined behaviour, not just a hang.

The alignment mathematics is not implicated: the same code produces correct
mismatch and indel alignments on the toy graph. What is wrong is that a
fixed-capacity guard drops data silently and the termination condition cannot
survive that loss.

All four queries below start at the same seed, `>2 4096 +`, and are 100 bp taken
from node `2` of `theseus_proxy/data/validation/ggbs/graphs/c4.gfa`:

| File | Edit vs graph | Result |
|---|---|---|
| `c4_exact.queries` | none | completes in ~0.007 s |
| `c4_sub1.queries` | 1 substitution at offset 50 | no termination (>25 s, killed) |
| `c4_ins1.queries` | 1 insertion at offset 50 | no termination (>25 s, killed) |
| `c4_del1.queries` | 1 deletion at offset 50 | no termination (>25 s, killed) |

## Reproduce

```bash
cd theseus_proxy
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release && cmake --build build -j
time ./build/apps/seq2graph_proxy --backend cpu \
    -g data/validation/ggbs/graphs/c4.gfa \
    -s data/validation/repro/c4_exact.queries -f /tmp/exact.gaf   # fast
time ./build/apps/seq2graph_proxy --backend cpu \
    -g data/validation/ggbs/graphs/c4.gfa \
    -s data/validation/repro/c4_sub1.queries -f /tmp/sub1.gaf     # hangs
```

## Cross-check against the pre-flattening implementation

Commit `56663af` ("Add theseus proxy project") predates the GPU work and holds
the original aligner, built on the `std::vector`-backed `Scope`, `ScratchPad`,
`VerticesData` and `BeyondScope` classes with no fixed capacity — zero references
to `QueryState`. Building and running it settles the question:

```bash
git archive 56663af theseus-proxy/theseus-proxy | tar -x -C /tmp/fullcpu --strip-components=2
cmake -S /tmp/fullcpu -B /tmp/fullcpu/build -DCMAKE_BUILD_TYPE=Release
cmake --build /tmp/fullcpu/build -j
/tmp/fullcpu/build/apps/seq2graph_proxy \
    -g theseus_proxy/data/validation/ggbs/graphs/c4.gfa \
    -s theseus_proxy/data/validation/repro/c4_sub1.queries -f /tmp/sub1.gaf
```

| Case | Current (flattened) | Commit `56663af` |
|---|---|---|
| `c4_exact` | 0.008 s, `100M` | 0.008 s, `100M` |
| `c4_sub1` | no termination | 0.006 s, `50M1X49M` |
| `c4_ins1` | no termination | 0.007 s, `50M1D49M` |
| `c4_del1` | no termination | 0.006 s, `51M1I49M` |
| `ebola_error_smoke` (256 reads) | no termination | 0.005 s, 256 rows |
| `c4_err` (512 reads) | no termination | 0.012 s, 512 rows |

The old implementation is also byte-identical to the current one everywhere the
current one terminates: the committed `baseline/sample_output.gaf`, the 256-row
`ebola_exact_smoke` golden and the 512-row `c4_exact` golden all match exactly.
So the two agree on results and differ only in whether they finish, which places
the defect in the fixed-capacity flattening and not in the alignment logic.

Note the I/D letters follow the aligner's own convention and read inverted
relative to these filenames, which are named from the query's point of view.

## What this is not

- Not a conversion bug. The exact read matches its graph window at 0 mismatches
  in 100 bp, which confirms the `node_id`/`offset`/orientation mapping is right.
- Not "the mismatch path was never written". The toy dataset in
  `data/sample_queries.fasta` exercises mismatch, insertion and deletion, and
  `baseline/sample_output.gaf` records `7M1X3M`, `1M1D11M` and `6M1I5M` for it.
  That graph is 15 bp across 4 nodes and stays inside the capacities.
- Not a CUDA bug: the GPU backend is a stub in this build, so this is entirely
  the CPU path. It is however a direct consequence of the fixed-capacity
  `QueryState` flattening done for the port.

The difference between the working and failing cases is graph size, not which
edit operation is involved: score 0 terminates on both graphs because the seed
extension reaches the end of the query without needing the Scope at all.

`GPU_PORT_PROMPT.md` predicted this: *"il bound attuale (1024) è provvisorio,
tarato su un dataset giocattolo dove i wavefront arrivano a 4 celle. Va rifatto
con dati reali — non fidarti del valore."* This is that failure, on real data.

## Consequence

The `*_err` halves of the GGBS validation suite cannot have a CPU reference
frozen until this is resolved, because generating one requires the CPU aligner to
finish on reads that carry sequencing errors. The exact halves are unaffected and
their goldens are committed.
