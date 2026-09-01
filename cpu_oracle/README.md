# Theseus CPU oracle

A frozen copy of the CPU aligner as it stood before the GPU port, vendored
verbatim from commit `56663af`. It exists to produce the golden GAF files every
GPU run is compared against: the regression authority is CPU-vs-GPU byte
equality, so the goldens have to come from an implementation nobody edits.

Treat this tree as read-only. If something needs to change, the thing to change
is `theseus_gpu/`, not this snapshot.

## Build

```bash
./cpu_oracle/build.sh
```

That rebuilds `bin/seq2graph_proxy_oracle` and checks it still reproduces
`theseus_gpu/baseline/sample_output.gaf` byte for byte. A prebuilt binary is
checked in, but it is an x86-64 Linux build: on any other machine run the
script rather than trusting it.

## Use

The CLI predates the `--backend` flag, so it takes no backend argument:

```bash
./cpu_oracle/bin/seq2graph_proxy_oracle \
    -g theseus_gpu/data/validation/ggbs/graphs/c4.gfa \
    -s theseus_gpu/data/validation/ggbs/queries/c4_err.queries \
    -f /tmp/c4_err.gaf
```

Normally you do not run it by hand. `scripts/generate_ggbs_cpu_golden.py
--no-backend-flag` drives it and records the binary's SHA-256 in the manifest,
which is how the goldens under `theseus_gpu/data/validation/ggbs/golden/` were
produced — re-run that when a golden has to be regenerated.
