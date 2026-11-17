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
    if len(a) != len(b):
        print(f"records differ: {len(a)} vs {len(b)} (the two runs took different windows)")
    n = min(len(a), len(b))
    if n == 0:
        raise SystemExit("no comparable records")

    kl_rows, top1_rows, worst = [], [], []
    bit_identical = True
    for i in range(n):
        pa, ra = a[i]
        pb, rb = b[i]
        if ra.shape != rb.shape:
            raise SystemExit(f"record {i}: shape {ra.shape} vs {rb.shape}")
        if pa != pb:
            print(f"  record {i}: position {pa} vs {pb}")
        if not np.array_equal(ra, rb):
            bit_identical = False
        la, lb = log_softmax(ra), log_softmax(rb)
        kl = (np.exp(la) * (la - lb)).sum(axis=-1)          # KL(a || b) per position
        agree = (ra.argmax(axis=-1) == rb.argmax(axis=-1)).astype(np.float64)
        kl_rows.append(kl)
        top1_rows.append(agree)
        for t in range(ra.shape[0]):
            worst.append((float(kl[t]), pa + t, int(ra[t].argmax()), int(rb[t].argmax())))

    kl = np.concatenate(kl_rows)
    t1 = np.concatenate(top1_rows)
    worst.sort(reverse=True)

    print(f"records {n}, positions {kl.size}, vocab {a[0][1].shape[1]}")
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
