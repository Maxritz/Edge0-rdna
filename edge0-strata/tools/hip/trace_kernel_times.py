#!/usr/bin/env python
"""trace_kernel_times.py - per-kernel time from the HIP runtime's PAL dispatch trace (AMD_LOG_LEVEL=4).

The runtime logs one line per dispatch, in submission order, with a microsecond timestamp:

    :3:palvirtual.cpp :2785: 14461396656 us: [pid:...] !  kernel : _ZN6strata7kernels...native_gu_kernelILi11...

On a busy single stream the gap to the NEXT dispatch is how long this kernel kept the GPU, so summing those gaps per
kernel name attributes the trace to kernels.  That is the measurement this prints.  It is an attribution, not a
per-kernel profiler: a gap also contains the host's work between the two dispatches, so kernels that are cheap and
issued by a slow host read high, and the total is the wall time of the region, not the sum of pure GPU time.

  usage: trace_kernel_times.py <log> [top] [block_gap_ms]

Prints, per kernel: dispatches, attributed ms, share of the attributed total, and mean us per dispatch.
"""
import re
import sys
from collections import defaultdict

DISPATCH = re.compile(r"palvirtual\.cpp\s*:\d+\s*:\s*(\d+) us:.*kernel : (\S+)")

# The request is the longest stretch of back-to-back dispatches: loading the pack is host work, so the gaps either
# side of it are far longer than anything inside a prefill.  50 ms cleanly separates the two on this engine.
BLOCK_GAP_MS = 50.0


def label(mangled):
    """A readable tail of the mangled symbol: the function name with its template arguments."""
    # _ZN6strata7kernels12_GLOBAL__N_116native_gu_kernelILi11EEEvPKy... -> native_gu_kernelILi11
    m = re.search(r"(\d+)([A-Za-z_][A-Za-z0-9_]*)(I[^E]*E)", mangled)
    if m:
        return m.group(2) + m.group(3)
    m = re.search(r"(\d+)([A-Za-z_][A-Za-z0-9_]*)", mangled)
    return m.group(2) if m else mangled


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    path = sys.argv[1]
    top = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    block_gap_ms = float(sys.argv[3]) if len(sys.argv) > 3 else BLOCK_GAP_MS
    events = []
    with open(path, "r", errors="replace") as f:
        for line in f:
            m = DISPATCH.search(line)
            if m:
                events.append((int(m.group(1)), m.group(2)))
    if len(events) < 3:
        print(f"{path}: {len(events)} dispatch(es) - not enough to attribute")
        return 1
    # the request window: the longest run of dispatches whose consecutive gaps are all under BLOCK_GAP_MS
    best = (0, 0)
    lo = 0
    for i in range(1, len(events) + 1):
        if i == len(events) or events[i][0] - events[i - 1][0] > block_gap_ms * 1000.0:
            if events[i - 1][0] - events[lo][0] > events[best[1]][0] - events[best[0]][0]:
                best = (lo, i)
            lo = i
    lo, hi = best
    skipped = lo
    tail = len(events) - hi
    events = events[lo:hi]
    total = defaultdict(float)
    count = defaultdict(int)
    for i in range(len(events) - 1):
        ts, name = events[i]
        gap = events[i + 1][0] - ts
        if 0 <= gap < 5_000_000:          # a gap over 5 s is the region's end or an idle boundary, not a kernel
            total[name] += gap / 1000.0   # ms
            count[name] += 1
    span = (events[-1][0] - events[0][0]) / 1000.0
    attributed = sum(total.values())
    print(f"{path}: the request window is {len(events)} dispatches ({skipped} before, {tail} after), "
          f"span {span:.0f} ms, attributed {attributed:.0f} ms")
    print(f"{'kernel':44s} {'n':>7s} {'ms':>9s} {'share':>7s} {'us/dispatch':>12s}")
    for name, ms in sorted(total.items(), key=lambda kv: -kv[1])[:top]:
        print(f"{label(name)[:44]:44s} {count[name]:7d} {ms:9.0f} {100.0 * ms / attributed:6.1f}% "
              f"{1000.0 * ms / max(1, count[name]):12.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
