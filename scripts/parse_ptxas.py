#!/usr/bin/env python3
"""Pull the resource numbers of one kernel out of a `ptxas -v` log.

`align_gpu.cu` compiles five kernels, so a plain `grep 'Used N registers'`
answers about whichever one ptxas emitted last (seq_length_kernel, 10
registers) rather than the one that matters. This walks the log per entry
function and reports the requested kernel by name.

Usage: parse_ptxas.py <log> [kernel-name]
Prints: registers=<n> stack=<bytes> spill=<bytes>, one per line.
"""

import re
import sys


def main() -> int:
    path = sys.argv[1]
    wanted = sys.argv[2] if len(sys.argv) > 2 else "theseus_align_batch_kernel"

    with open(path, errors="replace") as handle:
        text = handle.read()

    # Un blocco per entry function: i numeri di un kernel sono quelli fra la sua
    # riga "Compiling entry function" e la successiva.
    chunks = re.split(r"ptxas info\s*:\s*Compiling entry function", text)
    target = next((c for c in chunks[1:] if wanted in c.split("\n", 1)[0]), None)
    if target is None:
        return 1

    # Ci si ferma alle properties della prima funzione *diversa*: il frame di un
    # kernel e' quello il cui "Function properties for" nomina il kernel stesso,
    # e i callee __noinline__ che seguono riportano i loro a parte.
    head = target
    props = re.split(r"ptxas info\s*:\s*Function properties for ", target)
    own = next((p for p in props[1:] if wanted in p.split("\n", 1)[0]), "")

    regs = re.search(r"Used (\d+) registers", head)
    frame = re.search(r"(\d+) bytes stack frame, (\d+) bytes spill stores", own)

    print("registers=%s" % (regs.group(1) if regs else "?"))
    print("stack=%s" % (frame.group(1) if frame else "?"))
    print("spill=%s" % (frame.group(2) if frame else "?"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
