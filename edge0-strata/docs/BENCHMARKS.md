# Benchmarks - verified on this box

Recorded at HEAD `d2b9995`.  Every number here was produced by a run whose log exists in the working tree's history
or in the commit message that introduced it.  Nothing in this file is an estimate.

## The box

| | |
|---|---|
| GPU | AMD Radeon RX 6700 XT, RDNA2 `gfx1031`, 12 GiB, 20 WGPs, wave32 |
| Link | x16 PCIe, probed **28.0-28.1 GB/s** host→device |
| CPU | Ryzen 5 5600X (AVX2, no AVX-512), 5 pool workers on 6 physical cores |
| Model | 120B latent MoE, Q3_K experts streamed from a 26.66 GiB pinned arena, Q4_0 latent tables |
| Prompt | 4063 tokens (`prompt4k.txt`), MTP + speculative decode (`--spec 4 --spec-min-p 0.5`) |

## Shipped state

```
prefill @ --max-context 4096    351.02 / 351.47 tok/s     (borrows the prompt loan, chunk 832 - new default)
prefill @ --max-context 8192    344.02 tok/s              (owns its buffers, chunk 832)
decode   @ 4096                  30.82 / 30.88 tok/s       (spec 6 / min-p 0.3; 27.3-29.5 at spec 4 / 0.5)
512K + MTP                       exit 0, tokens 715 1822 363 313, prefill 64.50, decode 12.31
```

### Decode caveats (measured, not yet shipped as defaults)

- **A 3x intermittent defect:** about one run in six the speculative acceptance collapses to 0.06-0.14 and decode falls
  to ~11 tok/s (observed: 10.88, 12.49, 10.81, 11.60, 11.44). The collapsed runs still use wide T=6 windows and accept
  ~6 of ~65 drafts, so the drafts are genuinely bad for the whole run. Not the MTP side stream, not the expert lane.
  Real expected decode is therefore ~27.6, not 31.
- **`--pcie-frac 0` is +2.8 % at 1k and +25.9 % at 512K**, but it CHANGES the arithmetic (an expert computed on the CPU
  rounds differently from the GPU), so the 512K gate's tokens move from `715 1822 363 313` to
  `248046 198 248045 74455`. Left as an explicit flag: it needs a KL decision and a re-baselined token contract.
- **The window is `wait for rings + pool`, not the max** (45.15 + 22.58 of a 73.7 ms window). The CPU pool is fully
  exposed; overlapping it is worth ~25 ms/window, i.e. ~46 tok/s.

Reference targets for this engine: **~430 tok/s prefill, 62.4 tok/s decode**.  So the shipped state is ~1.4x short on
prefill and ~2.2x short on decode.

## The wins, before → after (all measured, alternating pairs where drift matters)

| # | change | measured |
|---|---|---|
| 1 | **staging ring depth from the cache** (`base = clamp(cache - 640, 64, 512)`, the scan's own lend test) | `wait copy` 3,168/3,159 → **1,757/1,876 ms**; prefill 271.0 → **299.5 tok/s**; identical 62,926 streamed blobs and byte-identical output |
| 2 | **chunk-aware ring** (never spend the chunk to buy ring depth) | 8192-context chunk collapse closed: 160.98 → **248.18/249.43** |
| 3 | **VRAM-cliff guard** (conditional reserve floor; the cache was parked past a cliff) | shipped default **91.03 → 289.51/294.95 tok/s** (a 3.2x collapse that produced CORRECT output, so no output check could see it) |
| 4 | **pcie-frac default 0.55 → 0.40** for HIP native packs | decode 25.72/27.33 → **29.19/29.28** at fixed cost; 512K decode 11.52 → 12.30 |
| 5 | **prompt path owns its buffers above 4k context** (option finalisation) | 8192-context prefill **94.95/90.25 → 265.21/264.59 (+179 %)**; 4k unchanged at 302.5 |
| 6 | **own-buffers chunk = the traffic minimum just under expert saturation (832, not 2048)** | 8192 prefill **265.21/264.59 → 341.43/340.00 (+28.7 %)**; 4k unchanged; chunk 1024 collapses to 213.37 because it saturates all 512 experts per chunk |
| 7 | **speculative window swept** (documented config, not a default) | decode **28.2 → 30.82/30.88 tok/s (+9.5 %)** at `--spec 6 --spec-min-p 0.3`; surface is sharply interactive (spec 4/0.3 is worse; spec 8/0.2 is 11.78) |
| 8 | **single-buffered A-stage staging** (removes all 96 B of spills; `STRATA_A_SINGLE_BUF=0` selects the control) | `gemm gate/up` **4,486 → 4,311 ms (−3.9 %)**, regs 213 → 164, scratch 96 → 0; prefill 4k **301.0 → 302.6**, 8k **341.4 → 343.9**; bit-identical output. Occupancy was chased too and is **dead**: `__launch_bounds__(512,8)` reaches blocks/cu=2, waves 32/64, and measures **−0.2 %** |
| 9 | **the 4k auto-chunk cap is conditional on the cache** (832 at ≥1,000 slots, else the historical 512) | 4k prefill **301.27/303.28 → 351.02/351.47 (+16.1 %)**; 30 % fewer blobs AND the fixed per-layer-chunk HC cost halves 768 → 480 ms; 8k unchanged |

Earlier kernel-level wins that stand (committed before this file):

| change | measured |
|---|---|
| Q4_0 latent D/U tables on fmt-aware tile kernels | `gemm down` 7,465 → **1,139 ms**, `gather` 4,159 → **993 ms**; prefill +4.5 %, decode +4.5 %; KL 7.58e-3 = 6.1x the engine's own 1.25e-3 run-to-run floor, top-1 100 % |
| staging ring 96 → 80 | prefill 165.92 → **231.39 tok/s** |
| CPU-rows U on `latent_u_grouped` | `u: matvec` 14.60 → **11.05 ms** |
| A-stage `unroll 1` + activation-word hoist | `gemm gate/up` 5,386 → **4,614 ms** |
| packed-fp16 dot2 GEMM, 128x128 tile | `gdn` 3,074 → **1,546 ms**; 9.46 TFLOP/s |

## Reproduction

```bat
:: 4k default
strata.exe --pack E:\Edge0-120B-q3k ... --expert-cache auto --prefill auto --max-context 4096 ^
           --spec 4 --spec-min-p 0.5 --mtp E:\Edge0-120B-q3k\mtp --pcie-frac 0.1 --max-new 32 --stats

:: 8k (now the default behaviour at that context - no flags needed beyond the context)
strata.exe ... --max-context 8192

:: never-break gate
powershell -File run_512k_mtp.ps1 -Embed 1      :: expect exit 0 and tokens 715 1822 363 313
```

## How to read these numbers

- **Batch drift is ±5-6 %.**  Only alternating pairs inside one binary are trusted for a delta; absolute numbers move
  between batches.  Several "wins" were withdrawn earlier in the project for exactly this reason (a ring 80 vs 128
  test that was a 60 % change in a 6.4x-too-small range read as "flat", and +6.8 % that did not reproduce).
- **Some changes are silent.**  The VRAM-cliff regression produced byte-correct output at 1/3 the speed; the
  8192-context regression produced correct output at 1/3 the speed.  Output checks cannot catch them - only the
  ledger and the tok/s can.
- **The profiler's slot labels have drifted** (slot 30 "hc1 norm" and slot 32 "D stage" share a start stamp, and
  `STRATA_GR_UP_WIDE=1` collapses the hc1 slot 67x while decode gets *slower*).  Treat a single slot as a label, not
  a kernel, until cross-checked with a variant switch.
- The engine is **not bit-reproducible in its logits** run to run (mean KL ~1.2e-3); top-1 agreement is the reliable
  statistic for any arithmetic change.
