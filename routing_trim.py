"""How many experts could a rank cut actually remove?  Reads a --dump-routing file.

Format (src/program/generate.cpp, drive_pool_multi_impl): one record per (layer, position):
    int32 layer, int32 k, k int32 ids (rank order), k float weights
so record size is 8 + 8k.  The engine's --kv-style rank prune keeps ranks [0, R) of each token's
top-k and zeroes the rest, so the question this answers is:

  for each R:  how many routed ENTRIES disappear, and — what decides the bytes on this engine —
               how many (layer, expert) pairs are never needed at all.

Bytes are what matter: a distinct expert costs one 1.16 MB blob per layer per chunk whether one token
routes to it or thirty.  A rank cut that removes 20% of the entries but 2% of the distinct experts saves
almost nothing.
"""
import struct
import sys
from collections import defaultdict

path = sys.argv[1] if len(sys.argv) > 1 else "routing.bin"
with open(path, "rb") as f:
    data = f.read()

# discover k from the first record
layer0, k0 = struct.unpack_from("<ii", data, 0)
rec = 8 + 8 * k0
n = len(data) // rec
print(f"{path}: {len(data)} bytes, k={k0}, {n} records (layer,position)")
assert n * rec == len(data), "trailing bytes: not a whole number of records"

# rank_pairs[layer][expert] = bitmask of the ranks at which that expert was ever routed in that layer
rank_pairs = defaultdict(lambda: defaultdict(int))
entries = 0
ids_all = []
for i in range(n):
    off = i * rec
    layer, k = struct.unpack_from("<ii", data, off)
    ids = struct.unpack_from(f"<{k}i", data, off + 8)
    toks = (layer, ids)
    ids_all.append(toks)
    entries += k
    for r, e in enumerate(ids):
        if 0 <= e < 4096:
            rank_pairs[layer][e] |= (1 << r)

pairs_all = sum(len(v) for v in rank_pairs.values())
print(f"layers={len(rank_pairs)}  routed entries={entries}  distinct (layer,expert) pairs={pairs_all}")
print()
print(f"{'R':>3} {'kept ranks':>11} {'entries dropped':>16} {'pairs dropped':>14} {'blob bytes saved':>17} {'pair saved':>11}")
BLOB = 1163 * 1024
for R in range(2, k0 + 1):
    drop_pairs = 0
    for layer, per_expert in rank_pairs.items():
        keep = (1 << R) - 1
        for e, mask in per_expert.items():
            if (mask & keep) == 0:      # never routed inside the kept ranks
                drop_pairs += 1
    drop_entries = entries - n * R
    print(f"{R:>3} {'0..'+str(R-1):>11} {drop_entries:>13} ({100.0*drop_entries/entries:4.1f}%) "
          f"{drop_pairs:>8} ({100.0*drop_pairs/pairs_all:4.1f}%) "
          f"{drop_pairs*BLOB/2**30:>12.0f} GiB {100.0*drop_pairs/pairs_all:>9.1f}%")
print()
print("'pairs' is the byte-relevant quantity: one blob per distinct (layer,expert) per chunk.")
