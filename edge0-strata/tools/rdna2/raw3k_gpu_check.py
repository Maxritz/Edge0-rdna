#!/usr/bin/env python3
"""Check the full chain: ggml code -> converter -> GPU lat_gu_decode_raw3k
bit extraction (the nat[0..3] / __vsub4 expressions in lat_q3k_fused.cu).

For each 16-value sub-block s the GPU reads 6 bytes at blk+6*s and builds four
uint32 words nat[0..3], each byte c being the code of value 16*s+4*m+c minus 4.
"""
import numpy as np
from raw3k_pack import convert_gu, BLOCK, GU_BYTES
from raw3k_oracle import ggml_codes


def gpu_codes(block):
    """block: 110-byte NEW-format super-block -> codes[256] in [-4,3]."""
    codes = np.empty(256, dtype=np.int16)
    for s in range(16):
        B = [int(block[6 * s + i]) for i in range(6)]
        B0, B1, B2, B3, B4, B5 = B
        nat = [0, 0, 0, 0]
        nat[0] = (B0 >> 5) | (((B0 >> 2) & 7) << 8) | \
                 ((((B0 & 3) << 1) | (B1 >> 7)) << 16) | (((B1 >> 4) & 7) << 24)
        nat[1] = ((B1 >> 1) & 7) | ((((B1 & 1) << 2) | (B2 >> 6)) << 8) | \
                 (((B2 >> 3) & 7) << 16) | ((B2 & 7) << 24)
        nat[2] = (B3 >> 5) | (((B3 >> 2) & 7) << 8) | \
                 ((((B3 & 3) << 1) | (B4 >> 7)) << 16) | (((B4 >> 4) & 7) << 24)
        nat[3] = ((B4 >> 1) & 7) | ((((B4 & 1) << 2) | (B5 >> 6)) << 8) | \
                 (((B5 >> 3) & 7) << 16) | ((B5 & 7) << 24)
        for m in range(4):
            for c in range(4):
                byte = (nat[m] >> (8 * c)) & 0xFF
                codes[16 * s + 4 * m + c] = byte - 4   # __vsub4(nat[m], 0x04040404)
    return codes


def main():
    rng = np.random.default_rng(9182736)
    bad = 0
    for trial in range(512):
        old = rng.integers(0, 256, size=BLOCK, dtype=np.uint8)
        ref = ggml_codes(old)
        gu = np.zeros(GU_BYTES, dtype=np.uint8)
        gu[0:BLOCK] = old
        new = convert_gu(gu.reshape(1, -1))[0, 0:BLOCK]
        got = gpu_codes(new)
        bad += int(np.count_nonzero(ref != got))
    print(f'512 random super-blocks, GPU-extraction mismatches={bad}')
    assert bad == 0


if __name__ == '__main__':
    main()
