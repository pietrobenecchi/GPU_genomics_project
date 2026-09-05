#!/usr/bin/env python3
"""Convert GGBS/vg-style JSON alignments into Theseus seeded queries."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


CONVERTER_VERSION = "ggbs-json-to-theseus-queries-v1"


class ConversionError(Exception):
    pass


@dataclass(frozen=True)
class ConvertedRecord:
    query_index: int
    read_id: str
    sequence: str
    start_node: str
    start_offset: int
    orientation: str
    source_score: Any
    identity: Any
    source_mapping: Any
    source_record_sha256: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_json(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def load_json_records(path: Path) -> list[dict[str, Any]]:
    text = path.read_text(encoding="utf-8")
    try:
        loaded = json.loads(text)
    except json.JSONDecodeError:
        records = []
        for line_no, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ConversionError(f"malformed JSON at line {line_no}: {exc}") from exc
            if not isinstance(item, dict):
                raise ConversionError(f"JSON line {line_no} is not an object")
            records.append(item)
        return records

    if isinstance(loaded, list):
        if not all(isinstance(item, dict) for item in loaded):
            raise ConversionError("top-level JSON array contains non-object records")
        return loaded
    if isinstance(loaded, dict):
        for key in ("records", "alignments", "reads"):
            value = loaded.get(key)
            if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                return value
        raise ConversionError("top-level JSON object has no records/alignments/reads array")
    raise ConversionError("JSON must be an array, JSON Lines, or an object containing records")


def load_gfa_nodes(path: Path) -> set[str]:
    nodes: set[str] = set()
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.startswith("S\t"):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) >= 2:
                nodes.add(fields[1])
    if not nodes:
        raise ConversionError(f"no GFA segment records found in {path}")
    return nodes


def read_id_list(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    return [item for item in ids if item and not item.startswith("#")]


def first_mapping(record: dict[str, Any]) -> dict[str, Any]:
    path = record.get("path")
    if not isinstance(path, dict):
        raise ConversionError("missing path object")
    mapping = path.get("mapping")
    if not isinstance(mapping, list) or not mapping:
        raise ConversionError("missing or empty path.mapping")
    first = mapping[0]
    if not isinstance(first, dict):
        raise ConversionError("path.mapping[0] is not an object")
    return first


def resolve_read_id(record: dict[str, Any], query_index: int) -> str:
    """Stable ID for a record, used both for dedup and for the index mapping."""
    read_id = record.get("name")
    if read_id is None:
        return f"record_{query_index}"
    return str(read_id)


def convert_one(
    record: dict[str, Any],
    query_index: int,
    gfa_nodes: set[str],
) -> ConvertedRecord:
    read_id = resolve_read_id(record, query_index)

    sequence = record.get("sequence")
    if not isinstance(sequence, str):
        raise ConversionError("missing sequence string")
    if sequence == "":
        raise ConversionError("empty sequence")

    mapping0 = first_mapping(record)
    position = mapping0.get("position")
    if not isinstance(position, dict):
        raise ConversionError("missing path.mapping[0].position")

    if "node_id" not in position:
        raise ConversionError("missing path.mapping[0].position.node_id")
    start_node = str(position["node_id"])
    if start_node not in gfa_nodes:
        raise ConversionError(f"node_id {start_node!r} is not present in the GFA")

    # vg omette gli scalari protobuf al valore di default, quindi un offset
    # assente vale 0 e non e' un record rotto. Confermato sui CSV di posizione
    # GGBS. Stessa regola per is_reverse qui sotto.
    try:
        start_offset = int(position.get("offset", 0))
    except (TypeError, ValueError) as exc:
        raise ConversionError("offset is not an integer") from exc
    if start_offset < 0:
        raise ConversionError("offset is negative")

    is_reverse = position.get("is_reverse", False)
    if not isinstance(is_reverse, bool):
        raise ConversionError("is_reverse must be boolean when present")
    orientation = "-" if is_reverse else "+"

    return ConvertedRecord(
        query_index=query_index,
        read_id=read_id,
        sequence=sequence,
        start_node=start_node,
        start_offset=start_offset,
        orientation=orientation,
        source_score=record.get("score"),
        identity=record.get("identity"),
        source_mapping=record.get("path", {}).get("mapping"),
        source_record_sha256=sha256_json(record),
    )


def select_records(
    records: list[dict[str, Any]],
    *,
    limit: int | None,
    read_ids: list[str] | None,
    sample_seed: int | None,
) -> list[dict[str, Any]]:
    selected = records
    if read_ids is not None:
        wanted = set(read_ids)
        selected = [record for record in records if str(record.get("name")) in wanted]
        found = {str(record.get("name")) for record in selected}
        missing = [read_id for read_id in read_ids if read_id not in found]
        if missing:
            raise ConversionError(f"read IDs not found: {', '.join(missing[:10])}")
    if sample_seed is not None:
        selected = list(selected)
        random.Random(sample_seed).shuffle(selected)
    if limit is not None:
        selected = selected[:limit]
    return selected


def write_queries(path: Path, converted: Iterable[ConvertedRecord]) -> None:
    with path.open("w", encoding="utf-8") as f:
        for item in converted:
            f.write(f">{item.start_node} {item.start_offset} {item.orientation}\n")
            f.write(f"{item.sequence}\n")


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except Exception:
        return None
    return result.stdout.strip()


def build_metadata(
    *,
    json_path: Path,
    graph_path: Path,
    queries_path: Path,
    converted: list[ConvertedRecord],
    invalid: list[dict[str, Any]],
    argv: list[str],
) -> dict[str, Any]:
    lengths = [len(item.sequence) for item in converted]
    total_bases = sum(lengths)
    return {
        "schema_version": 1,
        "converter_version": CONVERTER_VERSION,
        "converter_git_commit": git_commit(),
        "converter_arguments": argv,
        "source_json": str(json_path),
        "source_graph": str(graph_path),
        "source_json_sha256": sha256_file(json_path),
        "source_graph_sha256": sha256_file(graph_path),
        "queries": [
            {
                "query_index": item.query_index,
                "read_id": item.read_id,
                "sequence_length": len(item.sequence),
                "start_node": item.start_node,
                "start_offset": item.start_offset,
                "orientation": item.orientation,
                "identity": item.identity,
                "source_score": item.source_score,
                "source_mapping": item.source_mapping,
                "source_record_sha256": item.source_record_sha256,
            }
            for item in converted
        ],
        "stats": {
            "query_count": len(converted),
            "total_bases": total_bases,
            "minimum_length": min(lengths) if lengths else 0,
            "maximum_length": max(lengths) if lengths else 0,
            "mean_length": (total_bases / len(lengths)) if lengths else 0.0,
            "forward_count": sum(1 for item in converted if item.orientation == "+"),
            "reverse_count": sum(1 for item in converted if item.orientation == "-"),
            "invalid_record_count": len(invalid),
        },
        "invalid_records": invalid,
        "queries_sha256": sha256_file(queries_path),
    }


def convert(args: argparse.Namespace, argv: list[str]) -> dict[str, Any]:
    for out in (args.queries, args.metadata):
        if out.exists() and not args.overwrite:
            raise ConversionError(f"refusing to overwrite existing file: {out}")

    records = load_json_records(args.json)
    gfa_nodes = load_gfa_nodes(args.graph)
    ids = read_id_list(args.read_id_list) if args.read_id_list else None
    selected = select_records(
        records, limit=args.limit, read_ids=ids, sample_seed=args.sample_seed
    )

    seen_ids: set[str] = set()
    converted: list[ConvertedRecord] = []
    invalid: list[dict[str, Any]] = []
    for record in selected:
        read_id = resolve_read_id(record, len(converted))
        try:
            if read_id in seen_ids:
                raise ConversionError(f"duplicate read ID {read_id!r}")
            item = convert_one(record, len(converted), gfa_nodes)
            seen_ids.add(item.read_id)
            converted.append(item)
        except ConversionError as exc:
            invalid.append({"read_id": read_id, "reason": str(exc)})
            if not args.allow_invalid_records:
                raise ConversionError(f"invalid record {read_id}: {exc}") from exc

    args.queries.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    write_queries(args.queries, converted)
    metadata = build_metadata(
        json_path=args.json,
        graph_path=args.graph,
        queries_path=args.queries,
        converted=converted,
        invalid=invalid,
        argv=argv,
    )
    args.metadata.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return metadata


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--graph", type=Path, required=True)
    parser.add_argument("--queries", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--read-id-list", type=Path)
    parser.add_argument("--sample-seed", type=int)
    parser.add_argument("--allow-invalid-records", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    if args.limit is not None and args.limit < 0:
        parser.error("--limit must be non-negative")
    return args


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    try:
        metadata = convert(parse_args(argv), argv)
    except ConversionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    stats = metadata["stats"]
    print(
        "converted {query_count} records; invalid {invalid_record_count}; "
        "bases {total_bases}; len min/mean/max {minimum_length}/{mean_length:.2f}/{maximum_length}; "
        "forward/reverse {forward_count}/{reverse_count}".format(**stats)
    )
    print(f"queries_sha256 {metadata['queries_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
