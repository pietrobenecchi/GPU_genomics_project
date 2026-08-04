#!/usr/bin/env bash
# Rebuild the oracle binary from the vendored source in this directory.
#
# The checked-in bin/seq2graph_proxy_oracle is an x86-64 Linux build and will not
# run everywhere (Colab, macOS, a different glibc). This script is the portable
# guarantee: the source next to it is the real artifact.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
build="${1:-$here/build}"

cmake -S "$here" -B "$build" -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" -j

mkdir -p "$here/bin"
cp "$build/apps/seq2graph_proxy" "$here/bin/seq2graph_proxy_oracle"

echo
echo "oracle binary: $here/bin/seq2graph_proxy_oracle"
echo "sha256:        $(sha256sum "$here/bin/seq2graph_proxy_oracle" | cut -d' ' -f1)"
echo
echo "Sanity check against the committed toy baseline:"
"$here/bin/seq2graph_proxy_oracle" \
    -g "$here/../theseus_gpu/data/sample_graph.gfa" \
    -s "$here/../theseus_gpu/data/sample_queries.fasta" \
    -f "$build/sample_output.gaf"
if diff -q "$here/../theseus_gpu/baseline/sample_output.gaf" "$build/sample_output.gaf"; then
    echo "OK: byte-identical to theseus_gpu/baseline/sample_output.gaf"
else
    echo "FAIL: oracle output diverged from the committed baseline" >&2
    exit 1
fi
