#!/usr/bin/env python3
"""Verify the raw3k transcode reproduces every Q3_K value bit-for-bit.

decode_old uses the ggml/production-kernel permutation (see raw3k_oracle.py),
NOT the naive v//4 / v//8 indexing the first converter mistakenly used.
"""
import os
import sys
import numpy as np
from raw3k_pack import BLOB, GU_BYTES, N_ROWS, ROWS_PER_BLOB, BLOCK, convert_gu
from raw3k_oracle import ggml_codes


def decode_old(b):
    """b: (n,640,5,110) -> codes (n,640,5,256) int16 in [-4,3]."""
    hm = b[:, :, :, 0:32]
    qs = b[:, :, :, 32:96]
    v = np.arange(256, dtype=np.int32)
    half = v >> 7
    vv = v & 127
    qb = 32 * half + (vv % 32)
    sh2 = 2 * (vv // 32)
    hmb = vv % 32
    hb = 4 * half + (vv // 32)
    lo = (qs[:, :, :, qb].astype(np.int16) >> sh2) & 3
    hi = (hm[:, :, :, hmb].astype(np.int16) >> hb) & 1
    return lo + 4 * hi - 4


def decode_new(b):
    """b: (n,640,5,110) -> codes (n,640,5,256) int16 in [-4,3]."""
    c = b[:, :, :, 0:96]
    bits = np.unpackbits(c, axis=-1, bitorder='big')       # (n,640,5,768)
    bits = bits.reshape(b.shape[0], N_ROWS, ROWS_PER_BLOB, 256, 3)
    code3 = (bits[:, :, :, :, 0].astype(np.int16) << 2) | \
            (bits[:, :, :, :, 1].astype(np.int16) << 1) | \
            bits[:, :, :, :, 2].astype(np.int16)
    return code3 - 4


def main():
    # First: prove decode_old matches the authoritative ggml loop on random blocks.
    rng = np.random.default_rng(7)
    oracle_bad = 0
    for _ in range(64):
        blk = rng.integers(0, 256, size=BLOCK, dtype=np.uint8)
        one = blk.reshape(1, 1, 1, BLOCK)
        oracle_bad += int(np.count_nonzero(decode_old(one)[0, 0, 0] != ggml_codes(blk)))
    print(f'ggml oracle: 64 blocks, mismatches={oracle_bad}')
    assert oracle_bad == 0

    src = sys.argv[1] if len(sys.argv) > 1 else 'E:/Edge0-120B-q3k/experts.bin'
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    with open(src, 'rb', buffering=0) as f:
        data = np.frombuffer(f.read(n * BLOB), dtype=np.uint8).reshape(n, BLOB)
    old = data[:, 0:GU_BYTES].reshape(n, N_ROWS, 5, BLOCK)
    new = convert_gu(data[:, 0:GU_BYTES]).reshape(n, N_ROWS, ROWS_PER_BLOB, BLOCK)

    a = decode_old(old)
    b = decode_new(new)
    bad = np.count_nonzero(a != b)
    print(f'blobs={n} values={a.size} mismatches={bad}')
    print(f'code range old [{a.min()},{a.max()}] new [{b.min()},{b.max()}]')
    keep_ok = np.array_equal(old[:, :, :, 96:110], new[:, :, :, 96:110])
    print(f'scales+d identical: {keep_ok}')
    sys.exit(0 if bad == 0 and keep_ok else 1)


if __name__ == '__main__':
    main()
