# CPU Baseline

This directory stores deterministic CPU baseline outputs for the sample data.

The `cpu_sample_baseline_run` CTest test runs `seq2graph_proxy` against
`data/sample_graph.gfa` and `data/sample_queries.fasta`. The
`cpu_sample_baseline_compare` test compares the generated GAF against
`baseline/sample_output.gaf`.

Run from this directory with:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```
