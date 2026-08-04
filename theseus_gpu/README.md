# theseus_gpu

The working aligner: sequence-to-graph alignment, host code and CUDA kernel.
The frozen pre-flattening CPU aligner that produces the goldens lives in
`cpu_oracle/` at the repository root.

This folder intentionally excludes:
- MSA / POA graph-construction features
- MSA tools/tests

## What is included
- Core seq-to-graph aligner library (CMake target name: `theseus_proxy`)
- GPU backend under `src/gpu/` (CUDA kernel plus its CPU stub)
- Small proxy CLI app: `seq2graph_proxy`
- Tiny sample dataset for quick runs

## Build
```bash
cd theseus_gpu
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

## Run
```bash
./build/apps/seq2graph_proxy \
  -g data/sample_graph.gfa \
  -s data/sample_queries.fasta \
  -f data/sample_output.gaf
```

Input FASTA metadata format per sequence:
```text
> <start_vertex_name> <start_offset> <orientation>
<sequence>
```
Orientation is `+` or `-` (defaults to `+` if omitted).

## Notes
- Graph loading from GFA is kept because alignment needs a reference graph.
- No graph construction/update logic from MSA remains in this proxy project.
