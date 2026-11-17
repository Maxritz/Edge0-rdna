#!/usr/bin/env python3
"""Spot-check the WRITTEN raw3k pack against the source q3k pack at scattered
blob offsets, using the ggml/production decode on the source and the natural
3-bit decode on the converted file.  Guards against chunking / I/O mistakes.
"""
import numpy as np
from raw3k_pack import BLOB, GU_BYTES, N_ROWS, ROWS_PER_BLOB, BLOCK
from raw3k_verify import decode_old, decode_new

SRC = 'E:/Edge0-120B-q3k/experts.bin'
DST = 'E:/Edge0-120B-raw3k/experts.bin'
OFFSETS = [0, 1, 17, 511, 4096, 12345, 20000, 24574, 24575]


def read_blob(path, idx):
    with open(path, 'rb', buffering=0) as f:
        f.seek(idx * BLOB)
        return np.frombuffer(f.read(BLOB), dtype=np.uint8)


def main():
    total_bad = 0
    for idx in OFFSETS:
        a = read_blob(SRC, idx)[0:GU_BYTES].reshape(1, N_ROWS, ROWS_PER_BLOB, BLOCK)
        b = read_blob(DST, idx)[0:GU_BYTES].reshape(1, N_ROWS, ROWS_PER_BLOB, BLOCK)
        ca = decode_old(a)
        cb = decode_new(b)
        bad = int(np.count_nonzero(ca != cb))
        tail_same = np.array_equal(a[:, :, :, 96:110], b[:, :, :, 96:110])
        print(f'blob {idx:>6}: values={ca.size} mismatches={bad} scales+d identical={tail_same}')
        total_bad += bad
        assert tail_same
    print(f'TOTAL mismatches across {len(OFFSETS)} blobs: {total_bad}')
    assert total_bad == 0


if __name__ == '__main__':
    main()
