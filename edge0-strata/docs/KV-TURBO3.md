# Turbo3 / Rot4 KV - design, what has landed, what remains

Ported from Kraken-x (`third_party/Kraken-x/include/krk/turbo_kv.hpp`, TurboQuant 3-bit,
arxiv 2504.19874). Both are **codecs over the same orthonormal Hadamard rotation the `q4_0` path already
uses**, which is why the attention path does not change at all: the append side quantizes a rotated row, the
gather side dequantizes it back into **FP16 scratch**, and every attention kernel keeps reading that scratch.
This is a codec swap, exactly like `kv_q4` was, not a new pipeline.

## Why 3 bits with a Lloyd-Max codebook

After an orthonormal rotation every coordinate of a head is ~N(0, sigma^2) with the same sigma, so the
distribution is known and an 8-level Lloyd-Max quantizer can put its levels where the density is. Kraken-x
measured that a uniform 4-bit grid has *lower* MSE (0.0089 vs 0.033) and still breaks the model at the second
token, because a max-derived step spreads 15 levels over ±max and leaves the core - where the attention energy
lives - on about five coarse levels. The centroids/boundaries in `kv_turbo.hpp` are the textbook N(0,1) 3-bit
Lloyd-Max solution.

## Memory, at Strata's head_dim = 256 (per cell, per KV head)

| format | bytes/head | vs f16 | vs int8 | vs q4_0 | status |
|---|---|---|---|---|---|
| f16 | 512 | 1.00x | 0.52x | 0.28x | shipped |
| int8 | 264 | 1.94x | 1.00x | 0.55x | shipped, default |
| q4_0 | 144 | 3.56x | 1.83x | 1.00x | shipped, rotated |
| **rot4** | **112** | **4.57x** | **2.36x** | **1.29x** | codec landed |
| **turbo3** | **100** | **5.12x** | **2.64x** | **1.44x** | codec landed |

Layouts:

```
turbo3   [ f32 sigma ][ 32 x 3 B ]                   100 B   (sigma = ||x||/sqrt(d), one per head row)
rot4     [ 8 x ( f16 block scale ][ 4 x 3 B ] ) ]    112 B   (per 32 values, scale = block RMS)
```

`q4_0` already gives 3.56x over f16, so **the context ceiling is already close to turbo3's**: turbo3 only adds a
further 1.44x, i.e. +44% cells for the same KV bytes. It is worth having - it is the default in Kraken-x and it
is what buys the last chunk of context - but it is not a step change, and the honest number to quote for
"largest context" is the `q4_0` one until the wiring below is finished.

## Landed

**Codec** (`include/strata/kernels/kv_turbo.hpp`, `src/kernels/cuda/kv_turbo.cu`) - both formats, the full API
mirroring `kv_q4.hpp`: step, multi-step, prompt batch and gather. One warp per head row: sigma by warp
reduction with 8 codes a lane packed into 3 bytes; rot4 uses the block RMS and a per-group shared-memory pack.
The gather dequantizes into **FP16 scratch**, so no attention kernel changes.

**Plumbing** - every site that names a KV format:

| file | what |
|---|---|
| `qsa_decode_attn.hpp` | `QsaAttnPools` gained `k_turbo`/`v_turbo`/`k_rot4`/`v_rot4` |
| `qsa_decode_attn.cu` | the **fused reader**: `load8_turbo`/`load8_rot4`, `KV_MODE` 4 and 5 in `load8`, mode detection in `attn_kv_mode`, and the 4/5 arms in all five dispatch ladders (chunk, chunk-sel, lane-cell, slots, rows) plus the two scalar-V branches |
| `kv_stream.hpp/.cu` | `KvHostPools` fields, `present()`, `runs_of()` and `kv_block_bytes()` for both (two runs, no separate scale run; 8 rows x 100 B = 800 and x 112 B = 896 both satisfy the copy kernel's 16-byte rule) |
| `layer.cpp` | pool sizing, the elastic and plain offsets, the VRAM and host-copy allocations, the zeroing, `qsa_attn_pools`, the decode append and the gather fallback; `st.kv_rot` is set for both (they are *defined* rotated), and `st.kv_int8` excludes them |
| `prefill.cpp` | the staging-pool rows, `pools_of()`, the prompt batch append (both the staged and the direct path) and the MTP drafter's append |
| `conversation_snapshot.cpp` | bytes-per-cell, and the `+16` "rotated" marker is excluded (intrinsic to the format, as for q4_0) |
| `generate.cpp` | `--kv turbo3` / `--kv rot4` are accepted; the refusal is gone |
| `CMakeLists.txt` | `kv_turbo.cu` in the kernels target; `kv_turbo_parity` registered as a test |

**Parity gate** (`src/kernels/kv_turbo_parity.cpp`) - before this is trusted on a pack it proves:
the stored scale matches a host reference of the same statistic; **the gathered FP16 scratch equals the host's
dequantisation of the same pool bytes, bitwise** (so a pack/unpack mismatch cannot hide); the rms reconstruction
error is inside the 3-bit Lloyd-Max bound; `<q, k_hat>` tracks `<q, k>`; and the byte counts are ordered
turbo3 < rot4 < q4_0 < int8 < f16.

## Using it

```
--kv turbo3      100 B a head at head_dim 256; the smallest, Kraken-x's default
--kv rot4        112 B a head; per-32 scales, the more conservative of the two
```

Both combine with `--kv-resident` (KV streaming) like q4_0, because the block movers know both layouts.

## Remaining

Nothing functional. Worth doing next: a real-model A/B of the KL against f16 KV at the same prompt, the way
`bench/results/2026-09-27-kv-q4` did for q4_0, to decide which of the two becomes the long-context default.

