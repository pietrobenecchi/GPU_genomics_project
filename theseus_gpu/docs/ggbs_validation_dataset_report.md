# GGBS validation dataset analysis for Theseus GPU

Date: 2026-08-02

Scope of this document: choosing which benchmark inputs to use as the regression
suite for the Theseus GPU port. Theseus is a sequence-to-graph aligner, so every
dataset below is an *input* to be aligned and every "read" is an existing
sequence to be mapped onto a graph. GGBS names its dataset directories after the
organism each public reference genome comes from; those names are kept as-is for
comparability with the published benchmark, and the datasets are assessed here
on graph topology and cost, not on biology.

Sources:
- Zenodo record: https://zenodo.org/records/12207360, DOI `10.5281/zenodo.12207360`
- Associated repository: https://github.com/Mirkocoggi/GGBS

Local copies:
- GitHub checkout: `external_datasets/GGBS`
- Zenodo archives: `external_datasets/GGBS_zenodo`

The Zenodo record exposes five archives: `Graphs.zip`, `short_match.zip`, `short_err.zip`, `long_match.zip`, `long_err.zip`. All were downloaded and their MD5 checksums match the Zenodo manifest:

| File | Size in Zenodo | MD5 |
|---|---:|---|
| `Graphs.zip` | 5,020,166 | `1bec2d6bd0d5c0ee4c44ec01e0425e69` |
| `short_match.zip` | 9,737,582 | `d0f858e35b61f0fbbd8e6668083921bd` |
| `short_err.zip` | 9,428,066 | `2f14000d93d26fa9cc047995900c70c2` |
| `long_match.zip` | 78,617,214 | `610158dfd8f04971225ab5b79e951f40` |
| `long_err.zip` | 84,262,218 | `691a9bb923ec87f080d7d72fe1ca3943` |

## 1. Dataset structure

The GitHub repository organizes runnable benchmark inputs under:

```text
input_data/
  TEST/<dataset>/{GRAPH,READS,TXT}
  IGNORE/<dataset>/{GRAPH,READS,TXT}
  JSON/json_files/*.json
  JSON/position/*.csv
```

`TEST` contains `ebola`, `covid`, `yeast`, `C4`, `MHC`. `ecoli` is present under `IGNORE`, but it is still useful for validation analysis because it has real graph/read/truth files.

The repository also contains:

- `zip_reads_folder/dati_fa_ggbs`: position CSV files grouped by `short_match`, `short_err`, `long_match`, `long_err`.
- `results`: historical aligner outputs and parsed position CSVs.
- `utils`: benchmark parsers and comparison scripts.

## 2. Dataset inventory

Observed from the checked-out GFA/read files:

| Dataset | Organism | Scope | Graph file | GFA bytes | Nodes | Edges | Graph sequence bp | Branching out-nodes | Paths | Reads | Avg read length | Formats |
|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|
| `ebola` | Ebola virus | TEST | `a_ebola_0M.gfa` | 19,109 | 7 | 8 | 18,925 | 2 | 0 | 10,000 | 100 | GFA, FASTA, FASTQ, TXT, CSV, JSON |
| `covid` | SARS-CoV-2 | TEST | `b_cov.gfa` | 2,518,978 | 39,253 | 95,440 | 51,776 | 24,089 | 1 | 4,000 | 50 | GFA, FASTA, FASTQ, TXT, CSV, JSON |
| `yeast` | Saccharomyces cerevisiae | TEST | `d_yeast.gfa` | 2,579,216 | 49,410 | 67,320 | 762,651 | 17,713 | 0 | 10,000 | 100 | GFA, FASTA, FASTQ, TXT, CSV, JSON |
| `C4` | human C4 locus | TEST | `e_C490.gfa` | 165,370 | 16 | 22 | 164,832 | 4 | 0 | 10,000 | 100 | GFA, FASTA, FASTQ, TXT, CSV, JSON |
| `MHC` | human MHC-57 | TEST | `f_MHC-57.gfa` | 5,211,546 | 980 | 1,399 | 5,181,373 | 372 | 0 | 10,000 | 100 | GFA, FASTA, FASTQ, TXT, CSV, JSON |
| `ecoli` | E. coli K-12 | IGNORE | `c_Ecoli_K12_blunt.gfa` | 4,912,984 | 1,048 | 877 | 4,888,879 | 311 | 0 | 10,000 | 100 | GFA, FASTA, FASTQ, TXT, CSV, JSON |

Graph type:

- All graph inputs are GFA.
- The repository README states the graphs were built with VG Toolkit.
- `covid` is highly fragmented and path-rich at node level: many short segment mappings per read.
- `ebola` and `C4` are small variation graphs, suitable for frequent regression.
- `yeast` is large in node count but still only a few MB.
- `MHC` and `ecoli` are biologically larger loci/genomes and should not be daily default validation unless sampled.

## 3. Query organization and headers

Each dataset provides:

- FASTA reads: `READS/FASTA/*.fa`
- FASTQ reads: `READS/FASTQ/*.fq`
- plain TXT sequence lists: `READS/TXT/*.txt` or dataset-level `TXT/*.txt`
- JSON truth-like records in `input_data/JSON/json_files`
- CSV start positions in `input_data/JSON/position`

Most benchmark-generated FASTA/FASTQ headers are simple numeric identifiers and do not encode start point metadata:

```text
>1
@1
```

Real examples:

- `input_data/TEST/ebola/READS/FASTA/a_ebola_0M_reads.fa`: `>1`
- `input_data/TEST/ebola/READS/FASTQ/a_ebola_0M_reads.fq`: `@1`
- `input_data/TEST/C4/READS/FASTA/e_C490_reads.fa`: `>1`
- `input_data/TEST/MHC/READS/FASTQ/f_MHC-57_reads.fq`: `@1`
- `input_data/TEST/covid/READS/FASTA/b_cov_reads_4k_50.fa`: `>1`

There are also small sample FASTA files with richer external read headers, but these are not the generated 4k/10k validation read sets:

```text
>ERR4440390.1.1 1 length=102
>101_516_1013_2:0:0_3:0:0_0/1
```

No generated FASTA/FASTQ header directly contains all fields Theseus currently requires: `start_node start_offset orientation`.

## 4. Ground truth analysis

Ground truth exists, but it is split across CSV and JSON rather than embedded in read headers.

CSV example, `input_data/JSON/position/a_ebola_0M_reads.csv`:

```csv
node_id,offset
105,424
107,3334
101,175
```

JSON example, `input_data/JSON/json_files/a_ebola_0M_reads.json`:

```json
{
  "identity": 1.0,
  "path": {
    "mapping": [
      {
        "edit": [{"from_length": 100, "to_length": 100}],
        "position": {"node_id": "105", "offset": "424"},
        "rank": "1"
      }
    ]
  },
  "score": 110,
  "sequence": "..."
}
```

The JSON files are more useful than the CSVs because they contain:

- `sequence`
- `score`
- `identity`
- `path.mapping`
- per-mapping `node_id`
- optional `offset`
- optional `rank`
- optional `is_reverse` in some datasets
- `edit` operations that can encode match/mismatch/insert/delete-like events

Observed JSON characteristics:

| JSON file | Reads | Multi-mapping reads | Orientation fields |
|---|---:|---:|---:|
| `a_ebola_0M_reads.json` | 10,000 | 215 | 0 |
| `a_ebola_0M_reads_err.json` | 10,000 | 213 | 0 |
| `b_cov_reads_4k_50.json` | 4,000 | 4,000 | 0 |
| `b_cov_reads_err4k_50.json` | 4,000 | 4,000 | 0 |
| `d_lievito_reads.json` | 10,000 | 7,971 | 0 |
| `d_lievito_reads_err.json` | 10,000 | 8,012 | 0 |
| `e_C490_reads.json` | 10,000 | 65 | 23 |
| `e_C490_reads_err.json` | 10,000 | 56 | 13 |
| `f_MHC-57_reads.json` | 10,000 | 138 | 0 |
| `f_MHC-57_reads_err.json` | 10,000 | 165 | 0 |
| `c_Ecoli_K12_blunt_reads.json` | 10,000 | 49 | 40 |
| `c_Ecoli_K12_blunt_reads_err.json` | 10,000 | 60 | 34 |

Example path crossing nodes in C4:

```json
[
  {"position": {"node_id": "3", "offset": "19885"}, "edit": [{"from_length": 40, "to_length": 40}]},
  {"position": {"node_id": "4"}, "edit": [{"from_length": 7, "to_length": 7}]},
  {"position": {"node_id": "5"}, "edit": [{"from_length": 53, "to_length": 53}]}
]
```

Example reverse orientation in E. coli:

```json
[
  {"position": {"node_id": "1107", "offset": "83"}, "edit": [{"from_length": 44, "to_length": 44}]},
  {"position": {"is_reverse": true, "node_id": "810"}, "edit": [{"from_length": 56, "to_length": 56}]}
]
```

Benchmark correctness in GGBS is approximate and position-oriented, not full-alignment-equivalence oriented. `utils/compare_csv.py` compares parsed aligner output CSVs against expected CSVs by row, requiring identical `node_id` and `abs(offset difference) < 10`. It then reports the fraction of matched rows. This is useful for benchmark-level mapping accuracy, but it is weaker than what Theseus needs for GPU regression because it does not compare exact path, traceback, CIGAR, GAF, or tie-breaking.

## 5. Compatibility with Theseus

Theseus currently requires query records in this shape:

```text
start_node start_offset orientation
sequence
```

GGBS read files are not directly compatible with that format.

Compatibility classification:

- FASTA/FASTQ headers: not compatible; headers are usually only `>1` or `@1`.
- TXT sequence files: not compatible by themselves; they contain sequences only.
- CSV position files: partially compatible; contain `node_id,offset`, but no orientation and no sequence.
- JSON files: best source for conversion; contain sequence, path start position, and sometimes `is_reverse`.

Therefore the dataset is case B/C:

- B: a conversion script is required.
- C: start point must be reconstructed from JSON or from CSV plus sequence file.

Recommended conversion source:

1. Use JSON as primary input.
2. Extract `start_node` from `path.mapping[0].position.node_id`.
3. Extract `start_offset` from `path.mapping[0].position.offset`, defaulting to `0` when absent.
4. Extract `orientation` from `is_reverse`; default to forward when absent.
5. Use `sequence` from JSON, not from separate FASTA/TXT, to avoid ID synchronization errors.
6. Preserve full JSON path as expected truth for validation comparison.

## 6. Dataset evaluation

### Ebola

- Cost: very low graph cost; 10k reads of length 100.
- Complexity: 7 nodes, 8 edges, 2 branching out-nodes.
- Strengths: daily smoke testing, graph jump sanity, traceback over small variation graph.
- Weaknesses: too small to stress active vertices, tie-breaking, large frontier behavior.
- Useful for: M extension, mismatch in `*_err`, insertion/deletion from error JSON edits, basic graph jump, traceback, CIGAR/GAF stability.

### C4

- Cost: low; graph is 165 KB, 16 nodes, 22 edges.
- Complexity: compact but biologically meaningful branching.
- Strengths: best daily validation candidate: real locus, small, branching, some multi-node paths, some reverse-orientation truth fields.
- Weaknesses: not enough active vertices to stress GPU scheduling.
- Useful for: M/I/D, path crossing, graph jump, traceback, GAF, tie-breaking in small branched graph.

### Covid

- Cost: moderate; only 4k reads of length 50, but graph has 39k nodes and 95k edges.
- Complexity: extremely fragmented graph, every JSON read observed has multiple mappings.
- Strengths: strong branching/active-vertices test, many graph transitions.
- Weaknesses: historical GGBS position accuracy for GraphAligner/VG is low on covid, so it should not be the first correctness oracle for Theseus unless using CPU-vs-GPU exact equality rather than external truth.
- Useful for: active vertices, graph jump, path length, GPU frontier behavior.

### Yeast

- Cost: moderate/high for frequent dev; 49k nodes, 67k edges, 10k reads.
- Complexity: many multi-mapping reads and many branch out-nodes.
- Strengths: milestone validation for path-heavy behavior.
- Weaknesses: larger than needed for daily regression; not the first choice before Config 1 stabilizes.
- Useful for: active vertices, long multi-node traceback, GAF path stability.

### MHC

- Cost: moderate/high because graph sequence bp is 5.18 Mb, with 10k reads.
- Complexity: 980 nodes, 1,399 edges, 372 branching out-nodes.
- Strengths: biologically realistic human variation graph.
- Weaknesses: not daily; better as milestone validation or profiling-adjacent correctness.
- Useful for: real human branching, graph jumps, traceback under non-trivial locus graph.

### E. coli

- Cost: moderate/high; 4.89 Mb graph sequence bp, 10k reads.
- Complexity: fewer edges than nodes, but contains reverse orientation examples.
- Strengths: useful for orientation-specific validation.
- Weaknesses: placed in `IGNORE` by GGBS; avoid as default suite unless we explicitly opt into ignored datasets.
- Useful for: orientation handling and reverse path validation.

## 7. Recommended fixed validation suite

Do not use the largest dataset as the daily default.

Recommended suite:

| Role | Dataset | Reads | Reason |
|---|---|---:|---|
| Smoke dataset | `ebola` exact + err subset | 100 to 500 selected reads | Fastest real graph. Good for every build and quick CPU/GPU path equality. |
| Daily validation dataset | `C4` exact + err subset/full | 1,000 selected reads initially, full 10,000 when stable | Small real graph with branching and multi-node paths. Best balance for frequent Config 1 regression. |
| Branching dataset | `covid` exact + err subset | 500 to 1,000 selected reads | Highly fragmented graph; best active-vertices and graph-jump stress without using profiling-size reads. |
| Milestone dataset | `MHC` exact + err subset/full | 1,000 selected reads, full 10,000 at milestones | Real human variation locus; stronger biological representativeness. |
| Orientation add-on | `ecoli` selected reverse-orientation reads | small curated subset | Use only for orientation regression because GGBS placed it under `IGNORE`. |

Final daily recommendation: `C4` should be the main daily validation dataset, with `ebola` smoke before it.

## 8. Proposed validation pipeline

For each selected dataset and read subset:

1. Convert JSON truth to Theseus query format and expected-truth format.
2. Run CPU baseline.
3. Run GPU Config 0.
4. Run GPU Config 1 with 64 threads.
5. Run GPU Config 1 with 128 threads.
6. Run GPU Config 1 with 256 threads.
7. Compare all GPU outputs against CPU output, not just against GGBS approximate position truth.

Required checks:

- same number of results
- same score
- same path
- same traceback
- same CIGAR
- same GAF
- same overflow count/status
- no crash
- no CPU fallback
- deterministic tie-breaking where CPU has defined tie-breaking

Important policy:

- GGBS CSV truth should be used only to verify that the selected reads are biologically anchored at expected start positions.
- Regression authority should be CPU Theseus vs GPU Theseus exact output equality.
- Once a CPU output snapshot is accepted, freeze it as the project oracle for that validation suite.

## 9. Scripts needed later

Do not implement these yet. The required scripts are:

1. `ggbs_json_to_theseus_queries`
   - Input: GGBS JSON.
   - Output: Theseus query file with `start_node start_offset orientation` and sequence.
   - Also emit stable read IDs.

2. `ggbs_json_to_truth`
   - Input: GGBS JSON.
   - Output: expected truth with read ID, score, full path mapping, edit operations, orientation, start node, start offset.

3. `select_validation_reads`
   - Input: full JSON.
   - Output: fixed subset manifests.
   - Selection criteria: exact, err, multi-node paths, branch crossing, reverse orientation where available, insert/delete/mismatch edits.

4. `compare_theseus_outputs`
   - Input: CPU output and GPU output.
   - Compare score, path, traceback, CIGAR, GAF, overflow, fallback flags.

5. `run_validation_matrix`
   - Runs CPU, Config 0, Config 1 at 64/128/256 threads.
   - Produces one machine-readable report per dataset/config.

## 10. Hand-built unit-test read set

The benchmark read sets exercise the aligner as a whole but do not isolate
individual code paths: a failing read rarely tells you *which* branch of
`process_vertex` broke. A small set of hand-built query reads fixes that. It is
a unit-test fixture for the aligner, in the same sense as any other test input.

Recommended construction:

- Graph: reuse the GGBS GFA unchanged — `C4` for compact branching, `ebola` for
  the tiny smoke graph. Nodes, edges, node sequences and branch topology all
  come from the benchmark file; nothing about the graph is invented.
- Queries: pick a known path through that graph, take the sequence it spells,
  and apply one controlled edit per query so that exactly one alignment
  behaviour is under test. Each query is paired with its expected GAF row.

One query per aligner code path:

- exact match within one node
- single mismatch (M path)
- insertion relative to graph (I path)
- deletion relative to graph (D path)
- read crossing two nodes
- read crossing a bifurcation
- graph jump over a selected branch
- tie-breaking with two equal-score paths, if the graph topology permits, or on
  a minimal subgraph derived from it

These queries do not replace GGBS validation and carry no biological meaning —
they are the smallest inputs that make a specific aligner branch fire, so a
regression points at one function instead of at 10,000 reads.

## 11. Final recommendation

Use daily:

- `ebola` smoke subset.
- `C4` validation subset/full, exact and err.

Use at every milestone:

- Full `C4`.
- `covid` branching subset.
- `MHC` subset/full after Config 1 is stable.
- Optional `ecoli` reverse-orientation subset.

Use only for profiling:

- Full high-read or long-read Zenodo archives, especially full `long_match.zip` and `long_err.zip`.
- Full `yeast`, `MHC`, and `ecoli` runs when measuring throughput rather than correctness.

Files that must be converted:

- Primary: `input_data/JSON/json_files/*.json`.
- Optional cross-check: `input_data/JSON/position/*.csv`.
- FASTA/FASTQ/TXT are not sufficient alone for Theseus because they lack start point metadata.

Scripts to write later:

- JSON-to-Theseus query converter.
- JSON-to-truth/oracle extractor.
- deterministic subset selector.
- exact CPU-vs-GPU output comparator.
- validation matrix runner.

Current conclusion:

GGBS is suitable for Theseus validation, but not directly through its FASTA/FASTQ headers. The stable validation strategy should use GGBS JSON as the canonical source, convert to Theseus query format, freeze CPU outputs as exact project oracles, and reserve the largest/full archives for milestone or profiling work.
