#!/usr/bin/env python3
"""Lossless natural-order 3-bit repack of the Edge0 latent Q3_K expert blobs.

The Q3_K gate/up super-block stores 256 codes as hmask[32] (1 high bit/value)
+ qs[64] (2 low bits/value) + scales[12] + d[2] = 110 B.  The fused A stage must
recombine hmask and qs through funnel shifts (the 110 B pitch is 2-byte
misaligned) and honour ggml's PERMUTED layout: value v lives in qs byte
32*(v//128) + (v%128)%32 at 2-bit field (v%128)//32, with the high bit in hmask
byte (v%128)%32 at bit 4*(v//128) + (v%128)//32.  This is exactly the index
formula in lat_q3k_fused.cu::lat_gu_decode and include/strata/artifact/dequant.hpp
(dequantize_q3_K); raw3k_oracle.py verifies it against the ggml reference loop.

This converter writes the SAME 256 values as 256 plain 3-bit codes in natural
order: codes[96] + the SAME scales[12] + d[2] = 110 B.  Every decoded value is
bit-identical (code = qs + 4*hmask - 4, stored as code+4 in 0..7), so the GPU
decode arm can be gated byte-exactly against the original pack.  The IQ4_NL down
section is copied unchanged.

Usage:
    python tools/rdna2/raw3k_pack.py E:/Edge0-120B-q3k E:/Edge0-120B-raw3k
"""
import os
import shutil
import sys
import time
import numpy as np

BLOB = 1_164_800
GU_BYTES = 704_000          # (640 gate + 640 up) rows x 5 blocks x 110 B
N_ROWS = 1280
ROWS_PER_BLOB = 5
BLOCK = 110
CHUNK_BLOBS = 128


def convert_gu(gu):
    """gu: (n, 704000) uint8 -> (n, 704000) uint8 with natural 3-bit codes.

    ggml value v = 128*half + 32*t + r (half 0..1, t 0..3, r 0..31):
      low 2 bits  = qs[32*half + r] >> 2*t
      high bit    = hm[r] >> 4*half + t
    so the 8 (half,t) groups each cover 32 consecutive values - 8 vectorised
    slices instead of 256 fancy-index gathers, then a shift-based 3-bit pack.
    """
    n = gu.shape[0]
    b = gu.reshape(n, N_ROWS, ROWS_PER_BLOB, BLOCK)
    qs = b[:, :, :, 32:96].reshape(n, N_ROWS, ROWS_PER_BLOB, 2, 32)
    hm = b[:, :, :, 0:32]

    codes = np.empty((n, N_ROWS, ROWS_PER_BLOB, 256), dtype=np.uint8)
    for half in range(2):
        for t in range(4):
            lo = (qs[:, :, :, half, :].astype(np.uint8) >> (2 * t)) & 3
            hi = (hm.astype(np.uint8) >> (4 * half + t)) & 1
            codes[:, :, :, (128 * half + 32 * t):(128 * half + 32 * t + 32)] = lo + 4 * hi

    # Pack 8 codes -> 24 bits (MSB first), then split into 3 bytes.  No 768-bit
    # intermediate; c[0] occupies bits 21..19, c[7] bits 2..0.
    c8 = codes.reshape(n, N_ROWS, ROWS_PER_BLOB, 32, 8).astype(np.uint32)
    packed24 = np.zeros((n, N_ROWS, ROWS_PER_BLOB, 32), dtype=np.uint32)
    for i in range(8):
        packed24 |= c8[:, :, :, :, i] << (21 - 3 * i)
    packed = np.empty((n, N_ROWS, ROWS_PER_BLOB, 96), dtype=np.uint8)
    packed[:, :, :, 0::3] = (packed24 >> 16) & 0xFF
    packed[:, :, :, 1::3] = (packed24 >> 8) & 0xFF
    packed[:, :, :, 2::3] = packed24 & 0xFF

    out = np.empty_like(b)
    out[:, :, :, 0:96] = packed
    out[:, :, :, 96:110] = b[:, :, :, 96:110]   # scales + d unchanged
    return out.reshape(n, GU_BYTES)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(2)
    src_dir, dst_dir = sys.argv[1], sys.argv[2]
    src = os.path.join(src_dir, 'experts.bin')
    dst = os.path.join(dst_dir, 'experts.bin')
    total = os.path.getsize(src)
    assert total % BLOB == 0, (total, BLOB)
    n_blobs = total // BLOB
    os.makedirs(dst_dir, exist_ok=True)
    print(f'{n_blobs} blobs, {total/2**30:.2f} GiB -> {dst}', flush=True)

    t0 = time.time()
    with open(src, 'rb', buffering=0) as fi, open(dst, 'wb', buffering=0) as fo:
        done = 0
        while done < n_blobs:
            n = min(CHUNK_BLOBS, n_blobs - done)
            raw = np.frombuffer(fi.read(n * BLOB), dtype=np.uint8).reshape(n, BLOB)
            out = np.empty((n, BLOB), dtype=np.uint8)
            out[:, 0:GU_BYTES] = convert_gu(raw[:, 0:GU_BYTES])
            out[:, GU_BYTES:] = raw[:, GU_BYTES:]     # IQ4_NL down, unchanged
            fo.write(out.tobytes())
            done += n
            if done % 1024 == 0 or done == n_blobs:
                el = time.time() - t0
                print(f'  {done}/{n_blobs} blobs, {el:.0f}s, '
                      f'{done*BLOB/2**20/el:.0f} MiB/s', flush=True)
    print(f'done in {time.time()-t0:.0f}s', flush=True)

    # Copy every small sidecar file unchanged (manifest, scales, labels, dense, model.json, ...).
    # Sub-directories (e.g. mtp/) are NOT needed: --mtp/--native/--ple-gguf still point at the q3k
    # pack in the A/B harness.  Only experts.bin changes; native_experts.txt stays byte-identical.
    for name in os.listdir(src_dir):
        s = os.path.join(src_dir, name)
        d = os.path.join(dst_dir, name)
        if os.path.isfile(s) and name != 'experts.bin' and not os.path.exists(d):
            shutil.copy2(s, d)


if __name__ == '__main__':
    main()
