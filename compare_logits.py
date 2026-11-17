"""Byte-exact comparison of STRATA_NATIVE_LOGITS dumps.

Format (generate.cpp:193-218): "NLG1", int32 vocab, int32 reserved, then per row
int32 position, int32 vocab, vocab float32 logits.
"""
import math
import struct
import sys


def read_rows(path):
    with open(path, "rb") as f:
        blob = f.read()
    if blob[:4] != b"NLG1":
        raise SystemExit(f"{path}: bad magic {blob[:4]!r}")
    vocab, reserved = struct.unpack_from("<ii", blob, 4)
    off = 12
    rows = {}
    while off + 8 <= len(blob):
        pos, nv = struct.unpack_from("<ii", blob, off)
        off += 8
        if nv != vocab or off + 4 * nv > len(blob):
            raise SystemExit(f"{path}: truncated row at {pos}")
        vals = struct.unpack_from(f"<{nv}f", blob, off)
        off += 4 * nv
        rows[pos] = vals
    return vocab, rows


def main(a_path, b_path):
    va, ra = read_rows(a_path)
    vb, rb = read_rows(b_path)
    print(f"A: vocab={va} rows={len(ra)} positions={sorted(ra)}")
    print(f"B: vocab={vb} rows={len(rb)} positions={sorted(rb)}")
    if va != vb:
        raise SystemExit("VOCAB MISMATCH")
    common = sorted(set(ra) & set(rb))
    if not common:
        raise SystemExit("NO COMMON ROWS")
    for pos in common:
        a, b = ra[pos], rb[pos]
        diff = [i for i, (x, y) in enumerate(zip(a, b)) if struct.pack("<f", x) != struct.pack("<f", y)]
        mx = max(a)
        se = sum(math.exp(float(x) - mx) for x in a)
        lse = mx + math.log(se)
        # KL(A || B) over the A distribution, float64, with the same normaliser.
        kl = 0.0
        for x, y in zip(a, b):
            p = math.exp(float(x) - lse)
            if p > 0.0:
                kl += p * (float(x) - float(y))
        ia = max(range(len(a)), key=lambda i: a[i])
        ib = max(range(len(b)), key=lambda i: b[i])
        print(f"pos {pos}: bitwise_diff={len(diff)}/{len(a)} top1 A={ia} B={ib} "
              f"KL(A||B)={kl:.9g} max_abs={max(abs(float(x)-float(y)) for x, y in zip(a, b)):.9g}")
        if diff:
            print("  first diffs:", [(i, a[i], b[i]) for i in diff[:8]])


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
