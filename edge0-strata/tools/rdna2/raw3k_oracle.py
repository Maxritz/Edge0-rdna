#!/usr/bin/env python3
"""Authoritative check: ggml dequantize_row_q3_K code extraction vs the index
formula used by the production fused kernel (lat_gu_decode) and vs the naive
v//4 / v//8 formula that the first converter mistakenly used.

The ggml loop is transcribed verbatim from include/strata/artifact/dequant.hpp
(dequantize_q3_K), which itself is transcribed from ggml-quants.c.  Only the
code values are compared - scales/d are copied unchanged by the repack.
"""
import numpy as np


def ggml_codes(block):
    """block: 110 bytes -> codes[256] in [-4,3], from the ggml reference loop."""
    hm = block[0:32]
    q = block[32:96]
    codes = np.empty(256, dtype=np.int16)
    y = 0
    m = 1
    for _half in range(2):
        shift = 0
        for _j in range(4):
            for l in range(16):
                codes[y] = ((int(q[l]) >> shift) & 3) - (0 if (int(hm[l]) & m) else 4)
                y += 1
            for l in range(16):
                codes[y] = ((int(q[l + 16]) >> shift) & 3) - (0 if (int(hm[l + 16]) & m) else 4)
                y += 1
            shift += 2
            m <<= 1
        q = q[32:]
    return codes


def kernel_codes(block):
    """The production lat_gu_decode index formula, vectorised over v."""
    hm = block[0:32]
    q = block[32:96]
    v = np.arange(256, dtype=np.int32)
    half = v >> 7
    vv = v & 127
    qb = 32 * half + (vv % 32)
    sh2 = 2 * (vv // 32)
    hmb = vv % 32
    hb = 4 * half + (vv // 32)
    lo = (q[qb].astype(np.int16) >> sh2) & 3
    hi = (hm[hmb].astype(np.int16) >> hb) & 1
    return lo + 4 * hi - 4


def naive_codes(block):
    """The WRONG formula the first converter used - kept to show the mismatch."""
    hm = block[0:32]
    q = block[32:96]
    v = np.arange(256, dtype=np.int32)
    lo = (q[v >> 2].astype(np.int16) << 0) >> ((v & 3) << 1)
    lo = lo & 3
    hi = (hm[v >> 3].astype(np.int16) >> (v & 7)) & 1
    return lo + 4 * hi - 4


def main():
    rng = np.random.default_rng(20261118)
    mism_kernel = mism_naive = 0
    for trial in range(200):
        block = rng.integers(0, 256, size=110, dtype=np.uint8)
        ref = ggml_codes(block)
        mism_kernel += int(np.count_nonzero(ref != kernel_codes(block)))
        mism_naive += int(np.count_nonzero(ref != naive_codes(block)))
    print(f'200 random blocks: kernel-formula mismatches={mism_kernel}, naive-formula mismatches={mism_naive}')
    assert mism_kernel == 0, 'production kernel formula disagrees with ggml - unexpected'
    assert mism_naive > 0, 'naive formula unexpectedly agrees - nothing to fix'
    print('CONFIRMED: converter must use the kernel/ggml permutation, not v//4 / v//8.')


if __name__ == '__main__':
    main()
