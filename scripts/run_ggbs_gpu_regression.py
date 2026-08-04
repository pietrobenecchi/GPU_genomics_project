#!/usr/bin/env python3
"""Run the Theseus GPU kernel and compare its GAF output with a saved CPU golden."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


# Threads per block the kernel accepts. The kernel itself no longer has
# variants: the one-thread-per-query config 0 lives on the legacy/config0
# branch and is not built here.
THREAD_COUNTS = [64, 128, 256]

# Datasets split by what they demand of the aligner, not by size.
#
#   simple  - every read matches its graph window exactly, so the seed extension
#             reaches the end of the query at score 0 and the Scope is never used.
#   complex - reads carry sequencing errors, so the alignment needs at least one
#             wavefront at a non-zero score. This is the tier that exercises the
#             fixed capacities in query_state.h, and the tier the current CPU
#             path cannot finish (see data/validation/repro/README.md).
DATASETS = {
    "ebola_exact_smoke": {
        "tier": "simple",
        "graph": "theseus_gpu/data/validation/ggbs/graphs/ebola.gfa",
        "queries": "theseus_gpu/data/validation/ggbs/queries/ebola_exact_smoke.queries",
        "metadata": "theseus_gpu/data/validation/ggbs/truth/ebola_exact_smoke.metadata.json",
        "golden": "theseus_gpu/data/validation/ggbs/golden/ebola_exact_smoke.cpu.gaf",
    },
    "c4_exact": {
        "tier": "simple",
        "graph": "theseus_gpu/data/validation/ggbs/graphs/c4.gfa",
        "queries": "theseus_gpu/data/validation/ggbs/queries/c4_exact.queries",
        "metadata": "theseus_gpu/data/validation/ggbs/truth/c4_exact.metadata.json",
        "golden": "theseus_gpu/data/validation/ggbs/golden/c4_exact.cpu.gaf",
    },
    "ebola_error_smoke": {
        "tier": "complex",
        "graph": "theseus_gpu/data/validation/ggbs/graphs/ebola.gfa",
        "queries": "theseus_gpu/data/validation/ggbs/queries/ebola_error_smoke.queries",
        "metadata": "theseus_gpu/data/validation/ggbs/truth/ebola_error_smoke.metadata.json",
        "golden": "theseus_gpu/data/validation/ggbs/golden/ebola_error_smoke.cpu.gaf",
    },
    "c4_err": {
        "tier": "complex",
        "graph": "theseus_gpu/data/validation/ggbs/graphs/c4.gfa",
        "queries": "theseus_gpu/data/validation/ggbs/queries/c4_err.queries",
        "metadata": "theseus_gpu/data/validation/ggbs/truth/c4_err.metadata.json",
        "golden": "theseus_gpu/data/validation/ggbs/golden/c4_err.cpu.gaf",
    },
}

TIERS = ("simple", "complex")


def datasets_in(suite: str) -> list[str]:
    if suite == "all":
        return list(DATASETS)
    return [name for name, item in DATASETS.items() if item["tier"] == suite]


@dataclass(frozen=True)
class GafRow:
    query_name: str
    fields: list[str]

    @property
    def query_index(self) -> int | None:
        if not self.query_name.startswith("seq_"):
            return None
        try:
            return int(self.query_name[4:])
        except ValueError:
            return None


def read_gaf(path: Path) -> list[GafRow]:
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.rstrip("\n").split("\t")
        if len(fields) < 12:
            raise RuntimeError(f"{path}:{line_no}: malformed GAF row with {len(fields)} fields")
        rows.append(GafRow(fields[0], fields))
    return rows


def query_count(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(">"))


def read_ids(metadata: Path | None) -> dict[int, str]:
    if metadata is None or not metadata.exists():
        return {}
    data = json.loads(metadata.read_text(encoding="utf-8"))
    return {
        int(item["query_index"]): str(item["read_id"])
        for item in data.get("queries", [])
        if "query_index" in item and "read_id" in item
    }


def compare_gaf(cpu_path: Path, gpu_path: Path, metadata: Path | None = None) -> list[str]:
    cpu_rows = read_gaf(cpu_path)
    gpu_rows = read_gaf(gpu_path)
    ids = read_ids(metadata)
    errors: list[str] = []

    if len(cpu_rows) != len(gpu_rows):
        errors.append(f"result count differs: CPU {len(cpu_rows)} GPU {len(gpu_rows)}")

    seen: set[str] = set()
    duplicates: set[str] = set()
    for row in gpu_rows:
        if row.query_name in seen:
            duplicates.add(row.query_name)
        seen.add(row.query_name)
    if duplicates:
        errors.append(f"duplicate GPU query rows: {', '.join(sorted(duplicates)[:10])}")

    cpu_by_name = {row.query_name: row for row in cpu_rows}
    gpu_by_name = {row.query_name: row for row in gpu_rows}
    missing = sorted(set(cpu_by_name) - set(gpu_by_name))
    extra = sorted(set(gpu_by_name) - set(cpu_by_name))
    if missing:
        errors.append(f"missing GPU rows: {', '.join(missing[:10])}")
    if extra:
        errors.append(f"unexpected GPU rows: {', '.join(extra[:10])}")

    labels = {
        1: "query_length",
        2: "query_start",
        3: "query_end",
        4: "strand",
        5: "path",
        6: "target_length",
        7: "terminal_start",
        8: "terminal_offset",
        9: "matches_or_score_proxy",
        10: "block_length",
        11: "mapq",
        12: "cigar",
    }
    for name in sorted(set(cpu_by_name) & set(gpu_by_name), key=lambda value: int(value[4:])):
        cpu = cpu_by_name[name].fields
        gpu = gpu_by_name[name].fields
        max_len = max(len(cpu), len(gpu))
        for idx in range(max_len):
            cpu_value = cpu[idx] if idx < len(cpu) else "<missing>"
            gpu_value = gpu[idx] if idx < len(gpu) else "<missing>"
            if cpu_value != gpu_value:
                qidx = cpu_by_name[name].query_index
                read_id = ids.get(qidx, "<unknown>")
                label = labels.get(idx, f"field_{idx + 1}")
                errors.append(
                    "Mismatch:\n"
                    f"query_index: {qidx}\n"
                    f"read_id: {read_id}\n"
                    f"field: {label}\n"
                    f"cpu: {cpu_value}\n"
                    f"gpu: {gpu_value}"
                )
                return errors
    return errors


def resolve_dataset(args: argparse.Namespace) -> tuple[Path, Path, Path, Path | None]:
    if args.dataset:
        if args.dataset not in DATASETS:
            raise RuntimeError(f"unknown dataset {args.dataset!r}")
        item = DATASETS[args.dataset]
        graph = Path(item["graph"])
        queries = Path(item["queries"])
        golden = Path(item["golden"])
        metadata = Path(item["metadata"])
    else:
        graph = args.graph
        queries = args.queries
        golden = args.golden
        metadata = args.metadata
    if graph is None or queries is None or golden is None:
        raise RuntimeError("provide --dataset or explicit --graph --queries --golden")
    return graph, queries, golden, metadata


def run_kernel(
    *,
    binary: Path,
    graph: Path,
    queries: Path,
    output: Path,
    threads: int | None,
    timeout: float | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [str(binary), "--backend", "gpu"]
    # Without this the backend falls back to the CPU when the kernel result is
    # unusable and still writes correct alignments, so the GAF compares equal to
    # the golden and the run looks like a pass.
    command += ["--require-gpu-result"]
    if threads is not None:
        command += ["--gpu-threads", str(threads)]
    command += ["-g", str(graph), "-s", str(queries), "-f", str(output)]
    return subprocess.run(command, capture_output=True, text=True, timeout=timeout)


def validate_gpu_stderr(stderr: str) -> list[str]:
    errors = []
    lower = stderr.lower()
    if "gpu backend:" not in lower:
        errors.append("seq2graph_proxy did not report GPU backend status")
    if "fallback" in lower or "cpu fallback" in lower:
        errors.append("GPU run reported CPU fallback")
    if "notimplemented" in lower or "not implemented" in lower:
        errors.append("GPU run reported an unimplemented device path")
    if "align kernel result verified against cpu" not in lower:
        errors.append("GPU kernel verification message is missing")
    if "gaf reconstructed from gpu querystate" not in lower:
        errors.append("GAF was not reconstructed from GPU QueryState")
    return errors


def check_device(args: argparse.Namespace) -> None:
    validator = args.build_dir / "apps" / "seq2graph_gpu_validate"
    if not args.require_device:
        return
    # A missing validator used to skip this check silently, which let the
    # regression "pass" on a machine with no GPU at all.
    if not validator.exists():
        raise RuntimeError(
            f"--require-device is set but the validator binary is missing: {validator}\n"
            "Build with -DTHESEUS_PROXY_ENABLE_CUDA=ON, or pass --no-require-device "
            "to compare against the golden without asserting device execution."
        )
    device = subprocess.run([str(validator), "--require-device"], capture_output=True, text=True)
    if device.returncode != 0:
        raise RuntimeError(
            f"CUDA device validation failed\nSTDOUT:\n{device.stdout}\nSTDERR:\n{device.stderr}"
        )


def run_dataset(
    name: str,
    graph: Path,
    queries: Path,
    golden: Path,
    metadata: Path | None,
    *,
    binary: Path,
    output_dir: Path,
    timeout: float | None,
) -> list[str]:
    """Run every thread count for one dataset. Returns failure descriptions."""
    for required in (graph, queries, golden):
        if not required.exists():
            raise RuntimeError(
                f"missing input: {required}\n"
                "Convert the queries with ggbs_json_to_theseus_queries.py and freeze the "
                "CPU reference with generate_ggbs_cpu_golden.py before running the regression."
            )
    expected_queries = query_count(queries)
    if len(read_gaf(golden)) != expected_queries:
        raise RuntimeError(f"{name}: golden result count does not match query count")

    output_dir.mkdir(parents=True, exist_ok=True)
    failures: list[str] = []
    for threads in THREAD_COUNTS:
        output = output_dir / f"{name}.threads{threads}.gaf"
        label = f"{name} threads {threads}"
        try:
            completed = run_kernel(
                binary=binary,
                graph=graph,
                queries=queries,
                output=output,
                threads=threads,
                timeout=timeout,
            )
        except subprocess.TimeoutExpired:
            failures.append(
                f"{label}: TIMEOUT after {timeout:g}s.\n"
                "The GPU path also runs the CPU aligner to verify the kernel, and on reads "
                "needing a non-zero score that call does not return. See "
                "theseus_gpu/data/validation/repro/README.md."
            )
            continue
        if completed.returncode != 0:
            failures.append(
                f"{label}: command failed with exit code {completed.returncode}\n"
                f"STDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
            )
            continue
        run_errors = validate_gpu_stderr(completed.stderr)
        run_errors.extend(compare_gaf(golden, output, metadata))
        if run_errors:
            failures.append(f"{label}: " + "\n".join(run_errors))
        else:
            print(f"{label}: PASS ({expected_queries} queries)")
    return failures


def main(argv: list[str] | None = None) -> int:
    args = parse_args(list(sys.argv[1:] if argv is None else argv))
    try:
        binary = args.build_dir / "apps" / "seq2graph_proxy"
        check_device(args)

        if args.suite:
            names = datasets_in(args.suite)
            if not names:
                raise RuntimeError(f"no datasets in suite {args.suite!r}")
            targets = [
                (
                    name,
                    Path(DATASETS[name]["graph"]),
                    Path(DATASETS[name]["queries"]),
                    Path(DATASETS[name]["golden"]),
                    Path(DATASETS[name]["metadata"]),
                )
                for name in names
            ]
        else:
            graph, queries, golden, metadata = resolve_dataset(args)
            targets = [(args.dataset or "ggbs", graph, queries, golden, metadata)]

        failures: list[str] = []
        passed: list[str] = []
        for name, graph, queries, golden, metadata in targets:
            if args.suite:
                print(f"--- {name} ({DATASETS[name]['tier']}) ---")
            dataset_failures = run_dataset(
                name,
                graph,
                queries,
                golden,
                metadata,
                binary=binary,
                output_dir=args.output_dir,
                timeout=args.timeout,
            )
            if dataset_failures:
                failures.extend(dataset_failures)
            else:
                passed.append(name)

        if args.suite:
            print(f"\n{len(passed)}/{len(targets)} datasets passed", end="")
            print(f" ({', '.join(passed)})" if passed else "")
        if failures:
            print("\n\n".join(failures), file=sys.stderr)
            return 1
        return 0
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=sorted(DATASETS))
    parser.add_argument(
        "--suite",
        choices=(*TIERS, "all"),
        help="run every dataset in a tier: 'simple' (score 0), 'complex' (needs a non-zero score), or 'all'",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=300.0,
        help="per-run wall-clock limit in seconds; 0 disables it (default: 300)",
    )
    parser.add_argument("--graph", type=Path)
    parser.add_argument("--queries", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--golden", type=Path)
    parser.add_argument("--build-dir", type=Path, default=Path("build-gpu"))
    parser.add_argument("--output-dir", type=Path, default=Path("theseus_gpu/data/validation/ggbs/gpu_results"))
    parser.add_argument("--require-device", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args(argv)
    if args.suite and args.dataset:
        parser.error("--suite and --dataset are mutually exclusive")
    if args.timeout is not None and args.timeout <= 0:
        args.timeout = None
    return args


if __name__ == "__main__":
    raise SystemExit(main())
