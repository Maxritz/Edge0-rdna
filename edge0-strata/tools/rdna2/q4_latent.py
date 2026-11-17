#!/usr/bin/env python3
"""latent_dense_q8.bin -> the same tables as Q4_0 (fmt 4), for the latent D/U lever.

Layout, from latent_dense.cpp (d_bytes/u_bytes) and the kernels' own block readers:

  per layer, D then U, both block-major:
    D rows = G*ELL, blocks a row = H/32
    U rows = G*H,   blocks a row = ELL/32
    Q8_0 block (34 B): fp16 d, then 32 int8          -> value = d * q
    Q4_0 block (18 B): fp16 d, then 16 bytes         -> byte j low nibble = element j,
                                                        high nibble = element j+16,
                                                        value = d * (nib - 8)
                                                       (lat_blk_dot_q4 in latent.cu, exactly)

The quantiser is ggml's Q4_0: d = max_signed / -8 where max_signed is the element of largest magnitude with its
sign, then nib = min(15, trunc(x/d + 8.5)).  That is not the textbook nearest-scale choice, but it is what the
engine's Q4 path is written against and what every other Q4_0 table in a GGUF was built with, so matching it is
the point.

Usage:  python q4_latent.py latent_dense_q8.bin latent_dense_q4.bin [--groups 8] [--ell 1792] [--embd 2560]
Then:   set STRATA_LATENT_DENSE=<out>  and  set STRATA_LATENT_DENSE_FMT=4
"""
import argparse
import os
import sys

import numpy as np

Q8_BLK, Q4_BLK = 34, 18


def q8_to_q4(buf):
    """(nblocks, 34) uint8 Q8_0 -> (nblocks, 18) uint8 Q4_0, ggml's quantiser."""
    scale = np.ascontiguousarray(buf[:, :2]).view("<f2").reshape(-1).astype(np.float32)
    q = np.ascontiguousarray(buf[:, 2:]).view(np.int8).astype(np.float32)
    x = q * scale[:, None]                       # dequantized values, fp32
    amax = np.argmax(np.abs(x), axis=1)
    mx = x[np.arange(x.shape[0]), amax]          # the signed value of largest magnitude
    d = mx / -8.0
    nz = d != 0.0
    idv = np.zeros_like(d)
    idv[nz] = 1.0 / d[nz]
    nib = np.trunc(x * idv[:, None] + 8.5)       # ggml: (int) (x*id + 8.5)
    np.clip(nib, 0.0, 15.0, out=nib)
    nib = nib.astype(np.uint8)
    out = np.empty((buf.shape[0], Q4_BLK), np.uint8)
    out[:, :2] = np.ascontiguousarray(d.astype("<f2")).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = nib[:, :16] | (nib[:, 16:] << 4)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("src")
    ap.add_argument("dst")
    ap.add_argument("--groups", type=int, default=8)
    ap.add_argument("--ell", type=int, default=1792)
    ap.add_argument("--embd", type=int, default=2560)
    args = ap.parse_args()

    g, ell, h = args.groups, args.ell, args.embd
    d_rows, u_rows = g * ell, g * h
    d_cols, u_cols = h // 32, ell // 32
    d_q8, u_q8 = d_rows * d_cols * Q8_BLK, u_rows * u_cols * Q8_BLK
    d_q4, u_q4 = d_rows * d_cols * Q4_BLK, u_rows * u_cols * Q4_BLK
    per_q8, per_q4 = d_q8 + u_q8, d_q4 + u_q4

    size = os.path.getsize(args.src)
    if size % per_q8:
        raise SystemExit(f"{args.src}: {size} B is not a whole number of Q8 layers ({per_q8} B each)")
    layers = size // per_q8
    print(f"{args.src}: {layers} layers, {size / 1e9:.2f} GB Q8 -> {layers * per_q4 / 1e9:.2f} GB Q4")

    with open(args.src, "rb") as fi, open(args.dst, "wb") as fo:
        for l in range(layers):
            raw = fi.read(per_q8)
            if len(raw) != per_q8:
                raise SystemExit(f"layer {l}: short read")
            for off, nb in ((0, d_q8), (d_q8, u_q8)):
                q8 = np.frombuffer(raw, dtype=np.uint8, count=nb, offset=off).reshape(-1, Q8_BLK)
                fo.write(q8_to_q4(q8).tobytes())
            print(f"  layer {l + 1}/{layers}", end="\r", flush=True)
    print()
    print(f"wrote {args.dst} ({os.path.getsize(args.dst) / 1e9:.2f} GB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
