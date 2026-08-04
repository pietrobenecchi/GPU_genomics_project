#!/usr/bin/env python3
"""Run Theseus CPU once and save a protected golden GAF plus manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def count_queries(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.startswith(">"))


def count_results(path: Path) -> int:
    return sum(1 for line in path.read_text(encoding="utf-8").splitlines() if line.strip())


def git_commit() -> str | None:
    try:
        return subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True, text=True
        ).stdout.strip()
    except Exception:
        return None


def build_command(args: argparse.Namespace) -> list[str]:
    """The oracle under cpu_oracle/ predates --backend and rejects unknown options,
    so driving it means leaving the flag off."""
    command = [str(args.binary)]
    if args.backend_flag:
        command += ["--backend", "cpu"]
    command += ["-g", str(args.graph), "-s", str(args.queries), "-f", str(args.output_gaf)]
    return command


def run(args: argparse.Namespace) -> dict[str, object]:
    for out in (args.output_gaf, args.manifest):
        if out.exists() and not args.confirm_regeneration:
            raise RuntimeError(f"refusing to overwrite existing file: {out}")

    command = build_command(args)
    args.output_gaf.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    completed = subprocess.run(command, capture_output=True, text=True)
    elapsed = time.perf_counter() - start
    if completed.returncode != 0:
        raise RuntimeError(
            "CPU golden command failed with exit code "
            f"{completed.returncode}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"
        )

    query_count = count_queries(args.queries)
    result_count = count_results(args.output_gaf)
    if query_count != result_count:
        raise RuntimeError(
            f"CPU result count mismatch: {result_count} GAF rows for {query_count} queries"
        )

    manifest = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "git_commit": git_commit(),
        "source_json": str(args.source_json) if args.source_json else None,
        "source_json_sha256": sha256_file(args.source_json) if args.source_json else None,
        "binary": str(args.binary),
        "binary_sha256": sha256_file(args.binary),
        "producer": args.producer,
        "graph": str(args.graph),
        "graph_sha256": sha256_file(args.graph),
        "queries": str(args.queries),
        "queries_sha256": sha256_file(args.queries),
        "metadata": str(args.metadata) if args.metadata else None,
        "metadata_sha256": sha256_file(args.metadata) if args.metadata else None,
        "golden_gaf": str(args.output_gaf),
        "golden_gaf_sha256": sha256_file(args.output_gaf),
        "command": command,
        "query_count": query_count,
        "result_count": result_count,
        "overflow_or_error_count": sum(
            1
            for line in args.output_gaf.read_text(encoding="utf-8").splitlines()
            if "capacity" in line.lower() or "error" in line.lower() or "overflow" in line.lower()
        ),
        "elapsed_seconds": elapsed,
        "stdout": completed.stdout,
        "stderr": completed.stderr,
    }
    args.manifest.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, default=Path("build/apps/seq2graph_proxy"))
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--source-json", type=Path)
    parser.add_argument("--metadata", type=Path)
    parser.add_argument("--output-gaf", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--confirm-regeneration", action="store_true")
    parser.add_argument(
        "--backend-flag",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="pass --backend cpu; turn off for the pre-flattening oracle, whose CLI has no such flag",
    )
    parser.add_argument(
        "--producer",
        default=None,
        help="free-text provenance recorded in the manifest, e.g. 'oracle 56663af'",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    try:
        manifest = run(parse_args(list(sys.argv[1:] if argv is None else argv)))
    except RuntimeError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    print(
        f"wrote {manifest['golden_gaf']} with {manifest['result_count']} results "
        f"in {manifest['elapsed_seconds']:.3f}s"
    )
    print(f"golden_gaf_sha256 {manifest['golden_gaf_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
