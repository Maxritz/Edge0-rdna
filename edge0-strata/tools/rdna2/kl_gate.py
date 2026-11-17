#!/usr/bin/env python3
"""KL / NLL quality gate for a latent-table change (Q4_0 for the D/U tables).

The repo has no quality gate for a table-format change, which is what has blocked Q4_0 for the latent D/U
tables.  This is that gate, and it is deliberately narrow: it compares the model's own next-token distribution
under two latent tables on the SAME tokens, so the only variable is the table.

How the two sides are produced (engine-side tap, already in the tree):

    STRATA_LOGITS_DUMP=out_q8.bin strata.exe ... --tokens-file P --max-new 32 ...
    STRATA_LOGITS_DUMP=out_q4.bin strata.exe ... --latent-dense <q4 table> ...

The dump is one record per verify window: {int32 magic "LOGS", int32 T, int32 n_vocab, int32 record},
int64 position, then T * n_vocab float32 rows.  Row 0 is the window's first position.

Acceptance (measured on the shipped FP16 GEMM path, which the engine already runs while documenting it as "not
bit-identical to BF16"): rel-L2 ~2.1e-4 is the deviation the engine already tolerates between backends.  A
table change is a bigger step than that, so this harness prints the numbers and does not invent a threshold:
the pass mark is the owner's call, and the defaults below are the two that matter (top-1 agreement, mean KL).

Usage:
    python kl_gate.py a.bin b.bin [--top1-min 0.99] [--kl-max 0.01]
"""
import argparse
import struct
import sys

import numpy as np

MAGIC = 0x53474F4C  # "LOGS"


def read_records(path):
    """[(position, rows[T, n_vocab])] in file order."""
    out = []
    with open(path, "rb") as f:
        while True:
            hdr = f.read(16)
            if len(hdr) < 16:
                break
            magic, t, nv, _rec = struct.unpack("<iiii", hdr)
            if magic != MAGIC:
                raise SystemExit(f"{path}: bad magic 0x{magic:08x} at record {len(out)}")
            (pos,) = struct.unpack("<q", f.read(8))
            rows = np.frombuffer(f.read(4 * t * nv), dtype="<f4")
            if rows.size != t * nv:
                raise SystemExit(f"{path}: truncated record {len(out)}")
            out.append((pos, rows.reshape(t, nv)))
    if not out:
        raise SystemExit(f"{path}: no records")
    return out


def log_softmax(x):
    m = x.max(axis=-1, keepdims=True)
    z = x - m
    return z - np.log(np.exp(z).sum(axis=-1, keepdims=True))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a", help="baseline dump (current table)")
    ap.add_argument("b", help="candidate dump (new table)")
    ap.add_argument("--top1-min", type=float, default=0.99)
    ap.add_argument("--kl-max", type=float, default=0.01)
    ap.add_argument("--worst", type=int, default=5, help="print the N worst positions")
    args = ap.parse_args()

    a, b = read_records(args.a), read_records(args.b)

    # ALIGN BY POSITION, NOT BY RECORD.  A verify window's T depends on how many draft tokens were accepted, so
    # two runs split the same token stream into different records (observed: (3, V) vs (4, V)).  Each record
    # supplies T consecutive positions starting at `position`, so flatten to position -> row and compare the
    # intersection.  That is also what makes the gate usable after a table change, where the generated tokens may
    # themselves diverge once the distributions differ.
    def by_pos(recs):
        m = {}
        for pos, rows in recs:
            for t in range(rows.shape[0]):
                m[pos + t] = rows[t]
        return m

    ma, mb = by_pos(a), by_pos(b)
    common = sorted(set(ma) & set(mb))
    print(f"records {len(a)} vs {len(b)}; positions {len(ma)} vs {len(mb)}; common {len(common)}")
    if not common:
        raise SystemExit("no positions in common")
    nv = a[0][1].shape[1]
    if any(ma[p].shape[0] != nv for p in common) or any(mb[p].shape[0] != nv for p in common):
        raise SystemExit("inconsistent vocab size across records")

    kl_rows, top1_rows, worst = [], [], []
    bit_identical = True
    for p in common:
        ra, rb = ma[p], mb[p]
        if ra.shape != rb.shape:
            raise SystemExit(f"position {p}: shape {ra.shape} vs {rb.shape}")
        if not np.array_equal(ra, rb):
            bit_identical = False
        la, lb = log_softmax(ra), log_softmax(rb)
        kl = float((np.exp(la) * (la - lb)).sum())
        kl_rows.append(kl)
        top1_rows.append(1.0 if ra.argmax() == rb.argmax() else 0.0)
        worst.append((kl, p, int(ra.argmax()), int(rb.argmax())))

    kl = np.asarray(kl_rows)
    t1 = np.asarray(top1_rows)
    worst.sort(reverse=True)

    print(f"positions compared {kl.size}, vocab {nv}")
    print(f"bit-identical logits:      {'yes' if bit_identical else 'no'}")
    print(f"KL(a||b) mean / max:       {kl.mean():.6e} / {kl.max():.6e}")
    print(f"top-1 agreement:           {t1.mean() * 100:.3f} %  ({int(t1.sum())}/{t1.size})")
    print("worst positions (KL, pos, argmax_a, argmax_b):")
    for w in worst[:args.worst]:
        print(f"   {w[0]:.6e}  pos {w[1]:5d}  {w[2]:6d}  {w[3]:6d}")

    ok = (t1.mean() >= args.top1_min) and (kl.mean() <= args.kl_max)
    print(f"VERDICT: {'PASS' if ok else 'FAIL'} "
          f"(top-1 >= {args.top1_min:.2%}, mean KL <= {args.kl_max:g})")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
