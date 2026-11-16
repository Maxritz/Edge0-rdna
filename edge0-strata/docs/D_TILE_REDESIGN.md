# The D stage's token tile: 64 → 256 rows, on paper

`latent_dense_kernel<DU, NC>` in `src/prefill/moe_fused_q3k.cu` (the latent MoE D/U projection) is measured as the
`gather` phase: **1,619 ms of a 14,733 ms prompt at 1,455 tokens**, 11 % of that chunk, and it scales with
`ceil(T / kTileRows)` because the layer's whole D weight section is re-read once per token tile.

| tokens | tiles | `gather` |
| --- | ---: | ---: |
| 443 | 7 | 436 ms |
| 1,455 | 23 | 1,619 ms |

Its bandwidth is ~20 GB/s against a 384 GB/s card, and neither a pitched 2-D copy nor the ISA's cache-less load
(`glc+slc`, measured −0.1 % here) moves it: the cost is re-reading 28.12 MiB of weights per layer per tile. So the
lever is the row tile, and this note is the arithmetic of raising it to 256.

## The geometry as it stands

| constant | value | meaning |
| --- | ---: | --- |
| `kTileRows` | 64 | token rows per block item |
| `WROWS` | 256 | output rows per item (`OB`) |
| `LTHREADS` | 512 | threads per block (16 waves of wave32) |
| `LSTAGES` | 2 | double-buffered weight and activation staging |
| `LWPITCH` | 76 | a decoded weight row in smem |
| `LAPITCH` | 80 | a staged activation row in smem |
| `acc[4][2][4]` | 32 floats | accumulators per thread = `TR × OB / LTHREADS` |

Shared memory today: `2·OB·76 + 2·TR·80 + 4·TR` = **48.25 KiB**, one block per WGP (`__launch_bounds__(512, 1)`).

Hard limits that bound every option below: **64 KiB of LDS per workgroup** (RDNA2 ISA, §2.3.1 — the WGP has 128 KiB
but a single workgroup may allocate at most 64), and a **65,536 dword VGPR file per CU** (131,072 per WGP), which at
512 threads caps a block that must fit in one CU at ~128 VGPR/thread.

## Option table

Traffic is per layer: weights = `ceil(T/TR) × 28.12 MiB`; acts = `NFB × ceil(T/TR) × TR × K × 1.125 B`, which is
independent of `TR` (acts per block scale with `TR`, blocks scale with `1/TR`) and is ~85 MB per layer against the
~650 MB of weights at `TR = 64`, mostly L2-resident. Totals are decimal GB over the whole 48-layer prompt (the
`TR = 64` figure reproduces the 32.56 GB measured for this prompt, which is the cross-check on the arithmetic).

| option | TR × OB | acc/thread | regs/thread (acc + ~50) | smem | weight traffic, T=1,455 | `gather` projection |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| **A** (today) | 64 × 256 | 32 | ~82 | 48.3 KiB | 32.6 GB (measured) | 1,619 ms (measured) |
| **B** | 128 × 256 | 64 | ~114 (58 k dwords, fits 1 CU) | **58.5 KiB** | 17.0 GB | ~810 ms |
| **C** | 256 × 128 | 64 | ~114 (fits 1 CU) | **60.0 KiB** | 8.5 GB | ~420 ms, +4.1 GB of acts |
| **D** | 256 × 256 | 128 | ~178 (91 k dwords → needs the whole WGP) | **79.0 KiB — over the cap** | 8.5 GB | ~420 ms |
| **D′** | 256 × 256, acts single-buffered | 128 | ~178 (needs the whole WGP) | 59.0 KiB | 8.5 GB | ~420 ms, act pipeline removed |

**D is not buildable as written**: 79.0 KiB exceeds the 64 KiB workgroup cap. D′ fits only by dropping the
activation double buffer, which exposes activation-load latency in the inner K loop — and acts are the smaller,
cache-resident term, so that is the wrong thing to sacrifice.

## Recommendation: C, then B as the fallback

**C (256 rows × 128 outputs)** cuts weight traffic 4× (32.6 → 8.5 GB) and fits every limit with margin:
- smem 60.0 KiB of 64;
- 64 accumulators + ~50 working ≈ 114 VGPR/thread → 58,368 dwords, inside one CU's 65,536, so occupancy stays as it
  is rather than collapsing to a WGP-spanning single block;
- the act traffic doubles (85 → 177 MB per layer, +4.1 GB over the prompt) but acts are 2.1 MB per layer and
  L2-resident, so that cost is paid in L2 hits, not DRAM.

**B (128 × 256)** is the conservative first step: half the weight traffic, no new pressure at all (acc 64, smem
58.5 KiB, one CU), and the same code path as C with `TR = 128`. If B measures as projected, C follows by changing
two constants.

**D/D′ are rejected** on the register file and the LDS cap, not on arithmetic.

## What actually has to change in the kernel

The good news from reading it: the two staging predicates generalise with no restructuring, because `2·TR ≤ 512` and
`TR ≤ 512` hold for every option (`if (tid < 2 * kTileRows)` for acts with `arow = tid >> 1`, `if (tid < kTileRows)`
for `srow`). The real edits are:

1. **`TR` becomes a template parameter** (`template <bool DU, bool NC, int TR>`), because `ast`, `srow` and
   `lat_smem_bytes()` are all sized from `kTileRows`, and the A-stage kernels (`latent_kernel_q3k`) must keep 64 —
   they share the constant but have a different thread↔row map (`THREADS - kTileRows * 5` activation loader).
2. **Accumulators gain a sub-tile dimension**: `acc[4][2][4]` → `acc[4][2][4][TR / 64]`, indexed by the sub-tile the
   warp group is on. With `TR = 128` that is 64 floats/thread; with 256 it is 128 and the array no longer fits
   comfortably, which is the reason D needs the WGP.
3. **Row offsets and masks**: `base = TR * t` (not `64 * t`), `nrows = min(TR, M - base)`, and the `on0`/`on1`
   predicates (`nb0 < nrows`, `nb0 + 8 < nrows`) must be evaluated per sub-tile against that sub-tile's row base.
4. **`srow` staging**: already `tid < TR`, but the "rows past the tile's end repeat its last one" rule must apply per
   sub-tile so a short last tile does not read a row from the next block's range.
5. **`lat_smem_bytes(TR, OB)`** replaces the constant function, and the `cudaFuncSetAttribute` / occupancy calls move
   to the instantiated variant — otherwise the launch fails at 60 KiB, which is above the 48 KiB default dynamic-smem
   limit that is already overridden today.

## How it gets verified

Row splitting does not change any row's K-order, so the result should be **bit-identical** to the 64-row build on the
same prompt — a much stronger check than drift. Procedure: same prompt, greedy, 16 tokens, compare the full message
object; then the two longer prompts used for the phase table, comparing `gather` and the whole-chunk timeline.
A non-identical result means the sub-tile masks or the `srow` end-of-tile rule are wrong, not that the arithmetic
changed.
