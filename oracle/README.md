# Theseus CPU oracle

A frozen copy of the aligner as it stood **before** the GPU port flattened its
state into the fixed-capacity `QueryState`. This is the reference implementation
that produces the golden GAF files the GPU work is checked against.

## Provenance

Vendored verbatim from commit `56663af` ("Add theseus proxy project"), which
lived under `theseus-proxy/theseus-proxy/` at the time:

```bash
git archive 56663af theseus-proxy/theseus-proxy | tar -x -C oracle --strip-components=2
```

Only the original `baseline/` and `data/` directories were dropped, because the
live copies under `theseus_proxy/` are the ones everything else refers to, and
the original `README.md` was replaced by this file. Nothing else was edited:
treat this tree as read-only. If something needs to change, the thing to change
is `theseus_proxy/`, not this snapshot.

## Why it exists

It uses the `std::vector`-backed `Scope`, `ScratchPad`, `VerticesData` and
`BeyondScope` classes, which grow on demand. The current implementation replaced
them with fixed-capacity arrays sized for a toy dataset, and on real GGBS graphs
any read needing a non-zero alignment score overflows them, loses the write and
never terminates. Full analysis and evidence in
`theseus_proxy/data/validation/repro/README.md`.

So this snapshot is the only thing in the repo that can currently align a read
carrying sequencing errors against a real graph.

It agrees with the current implementation everywhere the current implementation
terminates — the committed toy baseline and both `*_exact` goldens match byte for
byte — so the two differ only in whether they finish, not in what they compute.

## Build

```bash
./oracle/build.sh
```

That rebuilds `bin/seq2graph_proxy_oracle` and verifies it still reproduces
`theseus_proxy/baseline/sample_output.gaf` byte for byte. A prebuilt binary is
checked in for convenience, but it is an x86-64 Linux build: on Colab or any
other machine, run the script rather than trusting it.

## Use

The CLI predates the `--backend` flag, so it takes no backend argument:

```bash
./oracle/bin/seq2graph_proxy_oracle \
    -g theseus_proxy/data/validation/ggbs/graphs/c4.gfa \
    -s theseus_proxy/data/validation/ggbs/queries/c4_err.queries \
    -f /tmp/c4_err.gaf
```

`scripts/generate_ggbs_cpu_golden.py --no-backend-flag` drives it and records its
SHA-256 in the manifest, which is how the goldens under
`theseus_proxy/data/validation/ggbs/golden/` were produced.
