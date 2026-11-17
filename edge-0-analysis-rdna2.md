# Edge0 / Strata on RDNA2 (gfx1031) — full audit and optimisation plan

> ## §0 CORRECTION (added 2026-11-16, after the measurements below were taken)
>
> **Every number in §2–§10 of this document was measured with the GPU in a fault state.** The adapter
> was reading its own VRAM at **22.6 GB/s instead of 361.5 GB/s**; its caches (587–841 GB/s) and its
> compute (15.1 TFLOP/s fp32) stayed completely healthy, so nothing inside the process looked wrong.
> A reboot — with no code change at all — produced:
>
> | | this document's numbers (fault) | after the reboot |
> |---|---|---|
> | decode | 4.99 tok/s | **23.56–24.67 tok/s** (4.7×) |
> | prefill | 87.66 tok/s | **126.65–126.85 tok/s** (1.44×) |
> | TTFT | 47.1 s | **32.4 s** |
> | decode window T1 | 377.55 ms | **58.73–59.72 ms** (6.4×) |
> | `wait for rings` | 355.66 ms | **51.28–52.58 ms** (6.8×) |
> | engine Q8_0 GEMV, standalone | 20–21 GB/s | **354–357 GB/s** = 93 % of the card's 384 GB/s spec |
>
> So the two headlines of this audit — "the A stage is 61 % of prefill device time because it runs one
> expert per launch" and "decode is 12.5× off because of the handshake" — are **artefacts of the fault**:
> a 17× slower memory path inflates every bandwidth-bound stage and every wait that depends on one.
> The kernel-level findings that are *not* bandwidth-bound still stand (the A-stage VGPR ceiling and its
> spills, the per-expert launch geometry at small chunk sizes, `--kv turbo3` costing decode time, the
> `STRATA_EMBED_VRAM=0` width change), but their ranking must be re-derived from the healthy baseline.
>
> The ranking after the correction, from `PROF-T` on the healthy device (`re2.log`, per window):
>
> | stage | T1 ms | T4 ms | share of the T4 window |
> |---|---|---|---|
> | `copy+combine` | 16.7 | **39.0** | 42 % |
> | FFN-side hyper-connection read (`hc1 up`) | 19.0 | 20.3 | 22 % |
> | dense GEMVs (`qkv` 3.0, `z` 1.9, `out-proj` 1.9, `q` 1.5) | | 8.3 | 9 % |
> | LM head (289 MB) | 2.0 | 2.3 | 2.5 % |
>
> and, outside the window entirely, **half of a decode round**: `Verifier::run` is 97 ms/round while the
> ring loop is 51 ms/round, so ~46 ms is staging/head/commit that `PROF-T` never sees. That, plus
> `copy+combine`, is where the remaining 2.5× to the reference row lives — not in the projection kernels.
>
> A guard now exists so this cannot be repeated: `mem_path_probe` (`src/kernels/cuda/mem_path.cu`,
> commit `77042f7`) measures DRAM against cache bandwidth at every startup and prints
> `strata: memory path: DRAM … GB/s, cache … GB/s (ratio …) - healthy|DEGRADED`. `STRATA_MEM_PATH=0` off.
>
> ### §0.1 What was fixed on the healthy device, and how it was validated
>
> Decode's window splits three ways at T4 (two stamps added in commit `55f048f`, `re5b.log`):
> `pcie/uWait` **24.7 ms** (the GPU spinning for the CPU pool — 731 MB of expert weights per round from
> DDR4 at 30.5 GB/s), `cpu/uCopy` 14.6 ms (hq copy + U matvec + hit rows), `copy+combine` **0.11 ms**
> (the combine itself).  The 39 ms "copy+combine" the earlier ranking blamed was the profiler carrying
> slot 23 forward, not the combine.
>
> `STRATA_HC_PERSIST` (one persistent hyper-connection read instead of ~12 launches per layer) is now
> **default on for gfx10** and validated with two alternating pairs, because a single decode figure on
> this box varies by ±9%:
>
> | | T4 window (ms) | decode (tok/s) |
> |---|---|---|
> | default read | 91.78 / 92.29 → **92.04** | 24.37 / 24.49 → **24.43** |
> | persistent read | 88.73 / 89.34 → **89.04** | 25.46 / 25.39 → **25.43** |
> | | **−3.3 %** | **+4.1 %** |
>
> with all 32 committed tokens identical to the default read (`re2.log` vs `h1.log`).
>
> **Rejected by measurement, so nobody re-tries it:** a bigger expert cache.  822 → 1062 slots
> (`--vram-reserve-mib 256`) gave 25.02/25.55 (off) against 26.84/24.56 (on) — the second pair reverses
> the first, so the effect is inside the run-to-run noise and it was not shipped.  1642 slots really
> does cut the pool (26.1 → 21.5 ms/round) and prefill to 137.9 tok/s (+8.5%), but that arm needs
> `STRATA_EMBED_VRAM=0`, which changes the U width (H 2560→10240) and costs decode 9%; the cache win
> alone is not reachable without freeing ~644 MiB of VRAM elsewhere.
>
> The baseline to beat, then, on the healthy device: **decode ~26.2 tok/s, TTFT ~20.9 s**, and
> prefill **199.0 tok/s** - 127 before the launch-width fix (0.3), 142 after it, and 199 after the
> staged U tile (0.6).  Reference row: 62.4 / 430 / -, i.e. 2.4x on decode and 2.2x on prefill.
>
> ### §0.2 The prefill ledger, re-taken after the launch-width fix (`final1.log`, `STRATA_PREFILL_TIMING=1`)
>
> 4063 tokens, GPU timeline **28,568 ms**, wall 28,573 ms, host staging 83 ms — **142.08 tok/s**
> (confirmation run `final2.log`: 28,544 ms, 142.22 tok/s).  Supersedes §4.2 and the 32,280 ms ledger
> that stood here before `fb1a943`.
>
> | phase | ms | share |
> |---|---|---|
> | gemm down **= U projection** (q8_1 of H + latent_u_grouped) | 8,954 | 31.3 % |
> | gemm gate/up **= the whole A stage** (the fused call runs gu + SwiGLU + down) | 5,817 | 20.4 % |
> | gather | 3,609 | 12.6 % |
> | wait copy | 3,442 | 12.0 % |
> | gdn (conv+gates+recurrence+out) | 3,606 | 12.6 % |
> | qsa attention | 622 | 2.2 % |
> | hc read: down+silu 655, up 280, inject 115, mix 63 | 1,113 | 3.9 % |
> | router+shared 480, qsa proj 353, host grouping 196 | 1,029 | 3.6 % |
> | combine 26, ple 61, embed+steps 72, hc read 4 | 163 | 0.6 % |
>
> The expert path is still **76 %** of the prompt (gate/up + down + gather + wait copy), but it is
> 21.8 s now instead of 25.4 s, and the shape of the remaining cost is different: with the launches
> four times wider the gate/up kernel runs at **2.5× the rate it did** (5,817 ms for work that used to
> take 8,424), which is the same product the MMQ tiled path achieved — so at a usable launch width the
> fused kernel is already near the tiled one and the 3.4× is not in the dot product.
> **gemm down (8,954 ms) is now the single largest phase** and the least efficient per MAC: it computes
> 640 MACs per output row against gate/up's 1,280, yet takes 1.5× as long.  Its tiling redesign was
> measured and lost (§0.4), because the engine's groups are ~50 entries and its 64-row tile already
> covers a group in one tile.
>
> ### §0.3 The launch width was clamped to 4 groups — the whole "batch has no effect" story
>
> `STRATA_PREFILL_TIMING` + `STRATA_LATENT_NPROF` + `STRATA_LAT_Q3K_INFO` give the A stage's own shape
> and cost.  One launch carried **4 expert groups / 64 entries and cost 238 us** (3.08 us/entry,
> 16,176 launches per 4K prompt), on a grid of `(5,4,1)` = 20 blocks = **one block per CU**.
>
> The reason it was 4 and could not be anything else:
>
> ```cpp
> const int stage_n = group_resident ? STAGE_GRP : STAGE;           // = 8 for a streamed walk
> const size_t stage_batch = min(lat_staged_batch(), stage_n / 2);  // = 4
> ```
>
> The staging buffers were never the limit - `stage_slots(T) = max(ring_slots(T), STAGE_GRP = 32)` and
> the walk cycles `stage_next % stage_n`, so all 32 slots were already allocated.  Only the ceiling was
> hard-coded to 8.  With `stage_n = stage_slots(T)` and the batch default 4 -> 16 (`fb1a943`):
>
> | run | GPU timeline | prefill | gemm gate/up | gemm down | wait copy |
> |---|---|---|---|---|---|
> | baseline (stage_n 8, batch 4) | 32,828 ms | 123.66 tok/s | 8,420 ms | 9,718 ms | 4,387 ms |
> | stage_n 32, batch 4 | 30,550 ms | 132.87 tok/s | 9,657 ms | 9,051 ms | 1,531 ms |
> | **stage_n 32, batch 16** | **28,492 ms** | **142.46 tok/s** | 6,170 ms | 8,987 ms | 2,981 ms |
> | confirmation x2 | 28,573 / 28,544 | 142.08 / 142.22 | 5,817 / 5,823 | 8,954 / 8,965 | 3,442 / 3,434 |
>
> **-13.2 % on the prompt, byte-identical tokens.**  The deeper prefetch alone (lookahead 28 instead of
> 4) already cut `wait copy` by two thirds; the wider launch then took the dot product's rate up 2.5x.
>
> The ring change touches the one configuration that must never break - 512K context + MTP on a 12 GiB
> card, which runs at zero free VRAM by design - so it was re-run after the change: exit 0, 179 expert
> slots, `strata: memory path: DRAM 696.2 GB/s, cache 1447.1 GB/s - healthy`, and the same four tokens
> as before (`715 1822 1822 1822`, decode 7.85 tok/s).  The extra slots cost no VRAM: `stage_slots()`
> was already 32 and `stage_dev`/`copied` were already sized for it; only the modulus changed.
>
> | attempt | result |
> |---|---|
> | MMQ A stage (`LAT_FUSED_Q3K=0 LATENT_MMQ=1`) | 39,240 ms - gate/up product 2.5x faster but `dequant` 4,871 + `wait copy` 10,385 |
> | native grouped (`LAT_FUSED_Q3K=0`) | 42,849 ms |
> | chunk 1024 (`--prefill 1024`) | 31,773 ms (-1.6 %, noise) and it streams *more* blobs (80,670) |
> | chunk 2048 | **fails**: `verify: layer 3 never rang (graph finished)` |
> | batch 8 / 16 / 64, **before the clamp fix** | 32,097 / 32,114 / 32,111 - all three actually ran as 4 (`b2.log`: `last: nb=4`), so these measured the clamp, not the batch |
> | batch 16, **with the clamp fixed** | 28,492 ms, 142.46 tok/s (`n2.log`: `last: nb=16`) |
> | `amdgpu_num_vgpr(128)` | gate/up 8,424 -> 19,926 ms (234 spilled registers) |
>
> So the A stage **was** launch-bound: the earlier "not batch-bound, the total is conserved" conclusion
> was drawn from three runs that were silently clamped to the same batch.  Both kernels are still at
> 256 VGPRs with 28/53 spilled and one block per CU, but at a proper launch width that costs far less
> than it appeared to - the register cap is still a loss (`amdgpu_num_vgpr(128)`) and the tiling
> redesign below is still a loss, so widening the launch was the cheap win hiding behind them.  The
> gate that made it safe is `lat_q3k_down_test` (built for HIP in `387ce0a`), which compares against a
> double-precision host reference and passes; the second gate was the 32-token output, byte-identical
> across every arm above.
>
> ### §0.4 The tiling redesign, done and measured — and it is a loss
>
> The redesign the audit asked for was implemented in `lat_q3k_down_kernel`: `kTileRows` 64 → 32,
> `nb0 = 8*(warp>>2)` instead of 16, and `acc[4][2][4]` → `acc[4][4]` (no `n` dimension), with the smem
> sizing and `arows` granularity following.  It did exactly what it was meant to do to the resources:
>
> | | before | after |
> |---|---|---|
> | VGPRs | 256 (cap) | **183** |
> | VGPR spill | 53 | **0** |
> | scratch | 120 B/lane | **0** |
> | occupancy | 4 waves/SIMD | 5 |
>
> `lat_q3k_down_test` **PASSES** with it (rel_mean 1.174e-07, bit-identical to the old tile), and its
> shape sweep moves the right way where the tile is small enough — epg 4: 2.060 → 1.844 µs/entry
> (−10 %) — but the wrong way above it (epg 32: 0.495 → 0.682, epg 200: 0.446 → 0.674), because a
> 32-row tile needs **two tiles** for a group larger than 32 and each tile re-stages that expert's whole
> weight block.
>
> The engine then settles it, on the canonical ledger and with the baseline re-measured immediately
> afterwards to rule out drift (the machine had drifted ~1.7 % slower by then):
>
> | | gate/up | down | total | prefill |
> |---|---|---|---|---|
> | 32-row tile | 9,777 | 9,344 | 33,701 | 120.46 tok/s |
> | 64-row tile (reverted) | 8,420 | 9,718 | **32,828** | **123.66 tok/s** |
>
> So it is **−2.6 %** and was reverted in full.  The lesson is worth more than the change: the engine's
> groups are around **50 entries** (5120 entries over ~100 groups), not the 16–26 assumed from 200
> distinct experts, so a 64-row tile already covers a group in ONE tile and a 32-row tile doubles the
> weight staging.  The tiling was already right; the spills were the cheaper side of the trade, and the
> down kernel's remaining gap is not a register problem.  A larger tile (128 rows) would only pay if
> groups grew past 64, which they do not at a 512-token chunk.

> ### 0.5 What "gemm down" actually is, and where the prompt now goes (measured)
>
> The fused latent path runs gate/up + SwiGLU + down inside ONE host call, and the ledger's marks were laid
> out for the older, separate-kernel path, so two of its phase names were one ahead of the work:
> `gemm gate/up` opens the whole A stage, and `gemm down` - the largest row in the table - opens at
> `quantize_q8_1(H)` and is the **U projection**.  Three independent measurements say so:
>
> | evidence | result |
> |---|---|
> | `STRATA_DU_PROF=1` (the U kernel's own events) | 23.7 ms a layer; 23.7 x 48 x 8 = **9,101 ms** against the ledger's 8,954 |
> | `STRATA_LATENT_U_WALK=1` (a *different* U kernel) | U 23.7 -> 29.2 ms a layer, ledger `gemm down` 8,954 -> **11,349 ms** (+27 % against +23 %) |
> | `STRATA_LATENT_U_RUN=16` (a third U kernel) | U -> 27.9 ms, `gemm down` -> 11,184 ms (and R=4: 13,231) |
>
> A mark has been added after the U launch (`kPfCombine`) so the bucket is bounded exactly from now on.
>
> **Where the 28.5 s goes, in FLOP terms.**  Per routed entry the A stage moves 2.46 M MAC (gate/up 1.64 M
> + down 0.82 M), U 3.28 M (1280 -> 2560), D 1280 x 19200 per token; 1.95 M entries and 4,063 tokens in
> this prompt:
>
> | stage | TFLOP | ms | TFLOP/s | % of int8 peak (~73) |
> |---|---|---|---|---|
> | A (gu + SwiGLU + down) | 9.6 | 5,817 | 1.65 | 2.3 % |
> | U | 12.8 | 8,954 | 1.42 | 1.9 % |
> | D (`iq_mmvq`) | 19.3 | 3,609 | 2.68 | 3.7 % |
> | **whole prompt** | **41.7** | **28,568** | **1.46** | **2 %** |
>
> So all three expert-path stages are ~2-4 % of the card's int8 throughput, and D - the ggml-optimised
> `iq_mmvq`, not this engine's code - is the fastest of them at 2.68 TFLOP/s.  That is the realistic
> ceiling band for this class of kernel on gfx1031: the engine's own best tiled kernel,
> `lat_q3k_down`, reaches 1.65 T-MAC/s = 3.3 TFLOP/s at large groups.
>
> **The U weight-reuse redesign, implemented and measured - it loses.**  The obvious hypothesis is that U
> is Infinity-Cache-bandwidth-bound: the per-row kernel re-reads a group's whole U matrix once per routed
> row (~20 GB a layer at ~850 GB/s, above DRAM, i.e. an IC-bound figure).  `latent_u_runs`
> (`STRATA_LATENT_U_RUN=R`) makes one block own a RUN of R routed rows that share a label and holds an
> output column's weights in registers, cutting that traffic ~4x at R=8.  It is **bit-identical** (the
> same terms in the same order, the same reduction tree; all arms produced
> `248046 198 248045 74455 198 248068 198 760`) and its resources are clean (77 VGPRs, 0 spills,
> 12 waves/SIMD) - and it is slower at every run length:
>
> | U kernel | ms a layer | prefill |
> |---|---|---|
> | per-row (shipped) | **23.7** | **142.07 tok/s** |
> | reuse, R=4 | 33.8 | 124.41 |
> | reuse, R=8 | 29.8 | 130.52 |
> | reuse, R=16 | 27.9 | 132.79 |
> | reuse, R=32 | 26.9 | 135.46 |
> | row-walk (`U_WALK=1`, the older alternative) | 29.2 | 132.23 |
>
> Traffic was never the limit: U is **instruction-throughput-bound**.  Its inner term costs ~9 instructions
> per 4 MAC (a 4-byte activation word, two 2-byte scale loads, the weight word built from two unaligned
> ushorts, one `__dp4a`, two multiplies), i.e. ~0.5 MAC per issue slot against a dp4a peak of 128.  Cutting
> the bytes cannot help that; only a staged, 16-byte-aligned layout with the scale hoisted per 32-value
> block can - which is what `lat_q3k_down` does, and what its 3.3 TFLOP/s ceiling buys (U 9.1 s -> ~4 s at
> that rate, ~16 % of the prompt).  That is the next engineering step, and it is a kernel, not a knob.
>
> The reuse kernel and the walk switch are kept behind their env flags, default off, because a measured
> negative is worth more than a deleted one.

> ### 0.6 The staged U tile: 23.70 -> 2.70 ms a layer, and prefill 142 -> 199 tok/s
>
> The measurements in 0.5 said U's inner term costs ~9 instructions per 4 MAC and that its 20 GB a layer
> through the Infinity Cache was a plausible ceiling.  Both are true, and the staged tile fixes both:
>
>   staged weight row (kUPitch 72):  64 codes | {half d(block 0), half d(block 1)} | pad
>   staged act row    (kUPitch 72):  64 codes | {half d(block 0), half d(block 1)} | pad
>
> 256 output columns x 64 routed rows a block, 512 threads, one 64-value chunk at a time, both operands
> copied into 4-byte-aligned shared rows so the inner loop is four `__dp4a` per 16-value sub-block on
> aligned dwords and the scale product is one FMA per sub-block instead of two multiplies per 4 values.
> The fragment mapping and the store pattern are `lat_q3k_down_kernel`'s (already verified by
> `lat_q3k_down_test`).  23,040 B of smem and 222 VGPRs, 0 spills - one block a CU, because the backend
> still ignores `__launch_bounds__`'s second argument, so the win is NOT occupancy.
>
> | | U a layer | ledger `gemm down` | prefill | tokens |
> |---|---|---|---|---|
> | per-row kernel | 23.70 ms | 8,954 ms | 142.2 tok/s | baseline |
> | **staged tile** | **2.70 / 2.62** | **1,045 / 1,046** | **198.59 / 198.96** | **identical (32)** |
>
> -8.8x on U, -13.4 s on the prompt, +40 % prefill, TTFT 29.0 -> 20.9 s, and all 32 committed tokens
> byte-identical (`248046 198 248045 74455 ... 18890 33358`).  U now moves 12.8 TFLOP in 1.04 s =
> **12.3 TFLOP/s, 17 % of this card's int8 peak** - the highest rate in the engine, and 8.7x the row
> kernel's 1.42 TFLOP/s.
>
> Why the earlier weight-reuse kernel (0.5) lost while this one wins, in one line: the run kernel cut
> traffic 4x but kept the per-4-value unaligned operand loads and turned the block into a serial
> row chain, so its cost floor was ~30 ms; the tile cuts traffic ~60x (20 GB -> 0.32 GB a layer, i.e.
> 23.5 ms of IC time -> 0.4 ms) *and* the instructions per MAC, and gives each weight byte 64 rows of
> reuse and each activation byte 256 columns.  Note the old kernel's 23.7 ms matches its 20 GB / 850 GB/s
> = 23.5 ms, so traffic WAS a real ceiling - the reuse kernel just never reached it.
>
> **The prompt now, 20,397 ms (`final2.log`):**
>
> | phase | ms | share |
> |---|---|---|
> | gemm gate/up = the A stage (gu + SwiGLU + down) | 5,619 | 27.6 % |
> | gather = the D projection (`iq_mmvq`) | 3,601 | 17.7 % |
> | wait copy | 3,388 | 16.6 % |
> | gdn | 2,989 | 14.7 % |
> | gemm down = U (was 8,954) | **1,046** | 5.1 % |
> | qsa 621, hc reads 1,111, router+shared 475, host grouping 196, rest ~150 | 2,553 | 12.5 % |
>
> The A stage is the next item, and its 64-row tile is only 39 % full: `STRATA_LATENT_NPROF` reports
> `nb=16 ent=396` - ~25 routed entries a group at a 512-token chunk - so most of each tile's rows are
> padding.  That is a tile-shape question for that kernel (the same reason a 32-row tile lost there is
> that it halves the compute without halving the weight staging), not another launch knob.

> ### 0.7 The same tile for D, and where the prompt stands: 229 tok/s
>
> The D projection ("gather" in the ledger) is a plain GEMM - every token against the whole
> [G*ELL = 19,200, N = 2,560] matrix, no groups at all - so the U tile serves it with `rows == nullptr`
> (identity row order, one weight matrix).  At T = 512 that is 8 full 64-row tiles, i.e. **no padding**,
> and K = 2,560 doubles the work per staged byte.  It is the fastest thing in the engine:
>
> | | ledger `gather` | rate | prefill |
> |---|---|---|---|
> | `iq_mmvq` (shipped) | 3,643 ms | 5.3 TFLOP/s | 191.40 tok/s |
> | staged tile | **892 / 860 ms** | **21.5 TFLOP/s (30 % of int8 peak)** | **220.64 / 219.54** |
>
> (both arms with `STRATA_D_PROF=1`, which synchronises once a layer, so the pair is like for like;
> the 8 tokens are identical in both.  `latent_u_tile_test` now also covers this branch against the
> per-row kernel with every row in one group: mean_rel 4.301e-07, cos 1.000000000.)
>
> **The prompt at 17,690 ms / 229.29 tok/s (`final1.log`, 32 tokens identical):**
>
> | phase | ms | share | before today |
> |---|---|---|---|
> | gemm gate/up = the A stage | 5,618 | 31.8 % | 8,424 |
> | wait copy | 3,391 | 19.2 % | 4,387 |
> | gdn | 2,995 | 16.9 % | 3,593 |
> | gemm down = U | 1,068 | 6.0 % | 8,954 |
> | gather = D | 860 | 4.9 % | 3,609 |
> | qsa 621+358, hc reads 1,104, router+shared 488, host grouping 184, rest ~140 | 2,895 | 16.4 % | |
>
> So the two dense latent projections together went from 12,563 ms to 1,928 ms (-85 %), and prefill from
> 123.7 to 229.3 tok/s over the day (127 -> 142 by the launch width, 142 -> 199 by the U tile, 199 ->
> 229 by the D tile; TTFT 32.4 -> 18.2 s; reference row 430).
>
> What is left is the expert path itself: A (5,618 ms, 1.71 TFLOP/s) plus the copy wait that feeds it
> (3,391 ms).  Both are bounded by streaming ~73 GB of expert blobs per prompt - chunk 1024 raised the
> tile fill from 39 % to 78 % and made `gemm gate/up` 70 % *worse* while the blob count rose 28 %,
> which is what says this phase is bytes-bound and not MAC-bound.  The levers there are the expert
> cache (VRAM is fully committed) or the CPU share (which cannot arm: the Windows commit limit leaves
> 1.3 GiB where the share needs 4.0 GiB - a machine setting, not a code one).

> ### 0.8 Decode, measured instead of inferred: the round's 89.45 ms, split
>
> The two buckets decode's table blamed (`pcie/uWait` 24.15 ms, `cpu/uCopy` 14.73 ms) each held three
> things, so nothing could be attacked with confidence.  `STRATA_VERIFY_U_PROF=1` now divides them with
> stream stamps (slots 33-35, total-preserving: the buckets are subtracted as the parts are added, and
> the round still sums to 89.45 ms).  At T4, per round, `v5.log`:
>
> | part | GDN | QSA | total | share |
> |---|---|---|---|---|
> | `u: gemv` - the q8_1 of H + the **GPU-rows U** (`latent_u_grouped`) | 9.916 | 3.315 | **13.23** | 15 % |
> | `u: pool wait` - the CPU pool's rows for this group | 8.696 | 1.590 | **10.29** | 12 % |
> | `u: hqcp+matvec+hits` - the hq copy, the CPU-rows U, the hit rows | 11.131 | 3.668 | **14.80** | 17 % |
>
> So the expert path in decode is 38.3 ms of 89.45 (43 %), and the single largest piece is **not** the
> pool wait (10.3 ms, not 24) but the two U-shaped pieces around it (28.0 ms).  A stamp lesson worth
> recording: the first version of this probe used `cudaEventSynchronize` inside the verify window and the
> run died with `wait_flag_ge: operation failed due to a previous error during capture` - a host CUDA call
> inside the capture window is the issue #31 driver-lock hazard the same function warns about.
>
> **The prefill tile does not help decode's U, measured.**  `STRATA_VERIFY_U_TILE=1` routes the GPU-rows
> U through `latent_u_tiled`: `u: gemv` 13.23 -> 13.23 (T4 table identical, decode 25.38 vs 24.79 tok/s,
> inside the noise).  The reason is the shape, not the kernel: `cap` is **10 rows a layer** here, so with
> ~1 row per label the tile's 64-row pad stages more than the per-row kernel does (52 MB vs 35 MB a
> layer) while computing 64x the rows it needs.  At that size the per-row kernel is bound by **block
> count** - 320 blocks a row x 10 rows x 48 layers = ~154k blocks a round at ~86 ns each = 13.2 ms - not
> by bytes.  The fix is therefore a decode-shaped tile (8-16 rows, one block per label x column tile,
> ~150 blocks a layer), not the prefill tile turned up.
>
> Machine context for the RAM question: `kq_cpu_bench` (unpinned, cold 4 GB) measures this box's ceiling
> at **37.2-37.6 GB/s** (DDR4-3000, 2 channels), while the pinned pool reaches 30.5-32.2 = 86 % of it.
> 3000 -> 3200 MT/s is +6.7 % on paper and would move decode by ~1.4 %; the pool kernel has at most
> ~15 % of headroom.  Neither is the lever the round's own split points at.

> ### 0.9 The decode-shaped U tile: built, parity-clean, and 7x slower - both tiles lose, for a reason
>
> At decode's shape `cap` is **10 routed rows a layer, one per label**, so there is no row reuse for any
> tile to exploit while a tile still pays a full slice stage.  Measured, alternating pairs, same 32-token
> prompt, `u: gemv` from the round table:
>
> | U kernel | u: gemv | decode | note |
> |---|---|---|---|
> | per-row (`latent_u_grouped`, shipped) | **9.92 / 9.94 ms** | **25.19 / 25.10 tok/s** | 3,200 blocks a layer, no smem, no syncs |
> | prefill tile (64 rows, `U_TILE=1`) | 13.23 | 24.79-25.38 | stages 52 MB a layer for 35 MB of work |
> | decode tile (16 rows, lane-per-column, `U_TILE=2`) | **68.7 / 69.0 ms** | 14.09 / 15.10 | 100 blocks a layer, but 20 syncs each for one row |
>
> The decode tile is new here (256 columns x <=16 rows, one lane a column, 16 accumulators, no cross-lane
> reduction, 19,584 B of smem) and it is numerically clean - `latent_u_tile_test` now covers it and prints
> **mean_rel 3.733e-07, cos 1.000000000** against the per-row kernel, the same figures as the prefill
> tile.  It is still 7x slower in the engine, and the reason is the shape, not a bug: with one row a
> label, its 256-thread block stages 17 KB into shared for four `__dp4a` per thread per chunk, twenty
> times, and only ~5 blocks an SM can hide that.  The per-row kernel is better precisely because its
> blocks stage 10.9 KB and compute immediately, with 3,200 of them in flight instead of 100.
>
> So decode's `u: gemv` is **irreducible in this direction**: a label's U slice is needed once per routed
> entry, ~35 MB a layer (~1.7 GB a round), and the shipped kernel already runs it at ~127 GB/s effective.
> The byte floor at the Infinity Cache's 1.38 TB/s is ~1.2 ms a round (4.4 ms at DRAM), so there is room,
> but only a layout change can take it: a **transposed U table** (column-major, so one lane can read a
> column coalesced and a block covers 256 columns with no staging and no syncs) - which needs ~167 MB of
> VRAM, available at 4K context but not at 512K.  That is the next step, and it is a weight-layout change,
> not another tile size.
>
> Both tile variants stay behind `STRATA_VERIFY_U_TILE` (0 = shipped per-row, 1 = prefill tile, 2 = decode
> tile), default 0, because a measured negative is worth more than a deleted one.

> ### 0.10 Decode's U measured against four shapes and two placements: the shipped kernel is the optimum
>
> With the round split (0.8) the target was clear - `u: gemv` ~10-13 ms a round, `u: hqcp+matvec+hits`
> ~14.8 - and every shape that helped prefill was tried here.  Clean alternating pairs, 32 tokens,
> `u: gemv` from the T4 table, tokens identical in every arm:
>
> | decode U variant | u: gemv | decode | verdict |
> |---|---|---|---|
> | per-row, 8 warps a block (shipped) | **9.90 / 10.01 ms** | 25.50 / 25.40 tok/s | optimum |
> | per-row, 32 warps a block (4x fewer blocks, `U_WARPS=32`) | 10.61 / 10.84 | 25.38 / 25.40 | -7 % |
> | prefill tile, 64 rows (`U_TILE=1`) | 13.23 | 24.79-25.38 | -33 % |
> | decode tile, 16 rows lane-per-column (`U_TILE=2`) | 68.7 / 69.0 | 14.09 / 15.10 | -590 % |
>
> and the CPU pool's placement, which the engine's own bench suggested would matter (it runs unpinned at
> 37 GB/s): pinned 30.8 GB/s / 25.88 tok/s, unpinned 30.5 / 25.37 - flat within noise, so `--pool-cores`
> and `STRATA_POOL_NO_PIN` are not levers either.
>
> So the hypothesis that decode's U is bound by block count (320 blocks a row x 10 rows x 48 layers =
> ~154k blocks a round, at 86 ns each matching the 13.2 ms) is **refuted by its own test**: cutting the
> block count 4x made it slightly worse, not better.  What remains is the data layout.  A label's U slice
> is needed once per routed entry (~35 MB a layer, ~1.7 GB a round) and the shipped kernel already runs
> that at ~170 GB/s effective - neither the Infinity Cache's 1.38 TB/s nor DRAM's 384 GB/s - so the work
> is latency-bound with the row-major layout.  A **column-major (transposed) U table** would let a warp
> read 32 columns coalesced with no staging and no shuffle, which is the one direction left; it costs
> ~167 MB of VRAM (available at 4K context, not at 512K context), so it is a memory-budget decision.
>
> Knobs added, all defaulting to the shipped behaviour: `STRATA_LATENT_U_WARPS` (8 = shipped),
> `STRATA_POOL_NO_PIN` (unset = pinned), `STRATA_VERIFY_U_TILE` (0 = shipped, 1/2 = the tiles).

> ### 0.11 Decode's round, re-split with the copy isolated: router+plan is now the largest single item
>
> A slot keeps only its LAST write, so a first attempt at isolating the hq copy (reusing a slot twice)
> silently inflated `u: gemv` from 13.2 to 24.3 - the table still summed to 90 ms, which is exactly why
> the total is checked on every change.  Done correctly, one arm, `v5.log`, total **89.90 ms**:
>
> | part | GDN | QSA | total | note |
> |---|---|---|---|---|
> | `router+plan` | 16.439 | 6.325 | **22.77** | never instrumented - the largest item now |
> | `u: hqcp+matvec+hits` | 11.085 | 3.619 | **14.70** | the hq copy alone is 0.19-0.25 ms; the rest is `latent_u_matvec` |
> | `u: gemv` (q8_1 of H + the GPU-rows U) | 10.013 | 3.342 | **13.35** | |
> | `u: pool wait` | 9.270 | 1.455 | **10.72** | the pool at its measured 30-32 GB/s ceiling |
> | dense GEMVs (qkv 3.33, z 2.09, out-proj 1.98, q+q-idx 1.54) | | | 8.94 | |
> | gdn (rec 0.97, ab 0.31, conv 0.25) + qsa attention 1.08 + head 2.30 + hc-read0 3.05 | | | 7.96 | |
> | `waitB` 3.82, `VRAM hits` 3.90, `shared+quant` 1.74, rest | | | 11.5 | |
>
> Two U-shaped pieces total **28.05 ms** and both move ~1.7 GB a round (10 and ~7 routed rows a layer, one
> per label, each needing its label's whole 3.5 MB slice) at ~126 GB/s effective - well under the
> Infinity Cache's 1.38 TB/s and DRAM's 384 GB/s, so there is real headroom in them, but four shapes and
> two placements (0.10) all lose to the shipped per-row kernel.
>
> **The transposed-U estimate, done before writing it.**  A column-major table would let a lane own a
> whole output column: no warp shuffle (5 steps of the ~37 instructions an output costs now), and the
> same 4-byte coalesced loads.  At one row per label there is no row reuse for the transpose to expose,
> so its instruction-count ceiling is ~4-14 % of the 13.35 ms half (0.5-1.8 ms, +0.6-2 % decode) - unless
> the 126 GB/s is a layout-latency artifact, in which case the ceiling is the byte floor itself (13.35 ->
> 4.4 ms at DRAM, +11 % decode).  Cost: 167 MB of VRAM for a persistent table (which 512K+MTP does not
> have) or 3.5 MB of per-layer scratch plus a ~1 ms/round transpose kernel.  It is the one untested axis
> and it is worth one experiment, with the scratch variant so no VRAM decision is needed.

> ### 0.12 router+plan attributed: it is the decode D stage (21.46 ms), and it is a bytes problem
>
> Slot 17 is now split at the D stage's own stamp (`gap(32,16)` vs `gap(17,32)`), total still summing to
> 90.02 ms.  One arm:
>
> | part of the 22.77 ms bucket | GDN | QSA | total |
> |---|---|---|---|
> | **D stage** (`iq_mmvq` + colscale + 2 quantizes + 2 mapped D2H copies + doorbell) | 15.488 | 5.968 | **21.46** |
> | router + plan + shared expert | 1.234 | 0.410 | **1.64** |
>
> So the router is 1.6 ms and "plan" is nothing: the bucket is the decode-side D projection.  D reads its
> whole matrix every layer - G*ELL x N = 19,200 x 2,560 Q8_0 = **52 MB a layer, 2.5 GB a round** - which at
> the measured 15.3-17.4 ms is ~150 GB/s against DRAM's ~384, and the two mapped D2H copies are only 388 KB
> a layer (0.7 % of those bytes), so the copies are not the cost either.
>
> **The staged D tile loses at M = 4**, measured: 21.27 / 21.48 ms against `iq_mmvq`'s 17.37 / 15.31, decode
> 22.92 / 23.21 against 23.55 / 25.45 (tokens identical).  The tile moves the same bytes far better but pads
> its row tile to 64 for 4 rows, and 16x the compute costs more than the streaming saves - the same reason
> the decode U tile lost, for the same arithmetic reason.
>
> What is left for decode's D, in order of size:
> 1. **fewer bytes**: the pack's D is Q8_0 (1.06 B a value).  Q4_0 would halve it to 26 MB a layer, i.e.
>    ~2x on this 21.46 ms *and* free ~1.2 GB of VRAM for the expert cache - but it changes the model's
>    arithmetic and there is still no KL/NLL gate to accept that, so it is the user's call, not a kernel's.
> 2. **a decode-shaped D kernel**: M = 4 rows, no padding, 16-byte aligned staging, deep MLP - the tile's
>    structure without its 64-row tile.  Realistic target the DRAM floor, 6.5-8.3 ms (+13-16 % decode).
> 3. at T = 4 all 15 labels are needed (40 routed entries), so there is no routed-subset saving to take.

> ### 0.13 Deeper speculation does not amortise D (and the window stays T4 anyway)
>
> Since the D stage costs 52 MB a layer *per window*, a wider speculative window would halve its cost per
> token.  Measured, `--spec 4 / 6 / 8`, alternating:
>
> | spec | decode | drafts accepted | tokens a round | verifier window |
> |---|---|---|---|---|
> | 4 | **25.21 / 25.60 tok/s** | 19 of 26 (0.73) | 2.46 | T4 |
> | 6 | 21.81 | 19 of 36 (0.53) | 2.46 | T4 |
> | 8 | 20.26 / 19.00 | 19 of 46 (0.41) | 2.46 | T4 |
>
> The window never grows past T4 - the drafter proposes ~2.5 tokens a round, so the extra depth is drafted
> and rejected, which is pure cost.  The engine's own `--spec 4` is the operating point, and D's bytes
> cannot be amortised this way.
>
> So decode's remaining options are both structural: a **decode-shaped D kernel** (M = 4 rows, no padding,
> 16-byte staging, deep MLP - the tile's structure without its 64-row tile, target 6.5-8.3 ms = +13-16 %),
> or **fewer D bytes** (Q4_0 instead of Q8_0: ~2x on 21.46 ms and ~1.2 GB of VRAM back, but it changes the
> model's arithmetic and there is no KL/NLL gate to accept it).  Every parameter-level lever around decode
> has now been measured: window depth, pcie split, pool width, pool placement, and three U shapes.

> ### 0.14 The decode-shaped D kernel: bit-exact, and slower - D is instruction-bound, not DRAM-bound
>
> `latent_d_fast` was written to the prescription: one warp per output row so its 1,360 B row is read as
> coalesced 128-byte instructions, T tokens accumulated in registers from each loaded weight word (4x
> reuse of the bytes that dominate), activations left in L1 (T x 2,880 B a layer, hot), no staging, no
> `__syncthreads`, no row padding, 512 threads = 16 rows a block.  It is **bit-exact** against the
> per-row reference (`latent_u_tile_test`: mean_rel 0.000e+00, cos 1.000000000) because it keeps the same
> term order and the same per-4-value scaling.
>
> It is slower, in alternating pairs:
>
> | D kernel | D stage (slot 30) | decode |
> |---|---|---|
> | `iq_mmvq` (shipped) | **15.27 / 15.16 ms** | **25.08 / 25.04 tok/s** |
> | `latent_d_fast` | 19.78 / 19.63 | 22.14 / 22.78 |
>
> That is the useful part: **D is not DRAM-bound after all.**  It moves 2.5 GB a round in 15.3 ms = 163
> GB/s, which is well under DRAM's ~384 - and a kernel with perfectly coalesced reads and 4-token reuse
> is 29 % SLOWER, so the limit is instructions per byte, not bytes.  `iq_mmvq` gets more MAC per
> instruction than a hand-written per-4-value loop can (my inner loop spends ~22 instructions per 16 MAC
> across the tokens: a weight word, an activation word and half per token, a `dp4a`, two multiplies).
> The "DRAM floor 6.5-8.3 ms" in 0.12 is therefore not reachable from this direction and is withdrawn.
>
> Decode now has five measured designs around its two biggest items (per-row U 9.9-10.0, 64-row U tile
> 13.2, 16-row U tile 68.7, 32-warp blocks 10.6; D tile at M=4 21.3, D fast 19.7 vs iq_mmvq 15.2), and
> the shipped kernels win or tie every one.  The remaining 2.4x is not a kernel-shape gap: it is the
> arithmetic the model demands per token - 480 expert-blob reads, 10 U slices and 15 D slices - so the
> only levers left are the bytes themselves (Q4_0 for the latent D table: ~2x on 15-21 ms and ~1.2 GB of
> VRAM back) or moving expert bytes off PCIe (the CPU share, blocked by the Windows commit limit).

> ### 0.15 The CPU-rows U fixed: latent_u_matvec -> latent_u_grouped, -32 % and the first decode gain
>
> The 14.4 ms "matvec+hits" bucket now has its parts on their own slots - copy 0.18 ms, matvec ~10.7,
> hits 0.15-0.20 - and the matvec was the one decode kernel never compared against anything.  It is
> strictly worse than the kernel the GPU rows already use: `latent_u_matvec` runs (G, H/8) blocks and
> walks every entry inside each block, `latent_u_grouped` is the per-row kernel with the same arithmetic
> at 126 GB/s.  Swapping is safe because `moe_hit_set` ASSIGNS the GPU rows from `hit_out` immediately
> after, so the wrong values a full-range call computes for them never escape.
>
> | CPU-rows U | slot 35 | copy | hits | decode |
> |---|---|---|---|---|
> | `latent_u_matvec` (shipped until now) | 10.648 / 10.773 ms | 0.18 | 0.15-0.20 | 24.92 / 25.48 tok/s |
> | **`latent_u_grouped`** | **7.224 / 7.328** | 0.18 | 0.09 | **26.36 / 26.31** |
>
> Two clean runs with it as the default: decode **25.70 / 26.27 tok/s**, prefill 228.56 / 229.96 (noise),
> output byte-identical.  -32 % on that kernel, ~+3-4 % decode - the first decode movement of the session,
> and it came from noticing that two kernels do the same job and only one had ever been timed.
> `STRATA_VERIFY_U_CPU_GROUPED=0` restores the matvec.
>
> **On the A stage's DMA coalescing (the other half of this pass), measured indirectly but decisively:**
> the A region is COUNT-sensitive, not just byte-sensitive.  At chunk 1024 the blob count rose 28 %
> (63,361 -> 80,897) and `gemm gate/up` rose **70 %** (5,982 -> 10,196 ms); if the copies were pure
> bandwidth, +28 % bytes could not cost +70 % time.  The host-side issue cost is negligible (84 ms for
> 63,361 copies = 1.3 us each), so the penalty is per-transfer in the DMA engine - which is exactly what
> coalescing consecutive experts (the walk visits them in sorted id order, and their arena blobs are
> contiguous) into one `cudaMemcpyAsync` per run would remove.  That is the next prefill step and it does
> not need the commit-limit change.

> ### 0.16 The staging ring was starving the copies: 32 -> 128 slots, prefill 229.4 -> 250.7 tok/s
>
> The PCIe probe settles the A stage's question and refutes my own coalescing plan: **28.0 GB/s host->device**
> (best of 27.9 / 28.0 / 28.0 / 27.3), so the prompt's 73.5 GB of expert blobs is only **2.6 s** of
> transfer.  Yet `wait copy` was 3.4 s and each copy occupied 41 us of DMA per 280 us of wall - the engine
> sits **~85 % idle**, so the copies were ring-starved, and coalescing (fewer, bigger transfers) cannot
> help an engine that is not busy.  The lever is ring depth, and `STRATA_STAGE_SLOTS` measures it:
>
> | ring slots | wait copy | prefill |
> |---|---|---|
> | 32 (shipped until now, 37 MB) | 3,397 / 3,385 ms | 229.44 / 229.68 tok/s |
> | 64 | 2,440 / 2,437 | 242.47 / 242.51 |
> | 96 | 2,124 | 246.34 |
> | **128 (new default, 148 MB)** | **1,878 / 1,875** | **250.64 / 250.70 / 251.14** |
> | 192 | 3,431 | 186.98 |
> | 256 | 3,216 | 188.88 **and the output changes at token 27** |
>
> So 128 is the measured optimum and 192/256 are actively harmful (the VRAM the ring takes starts to bite
> the expert cache and, at 256, something degrades numerically).  128 is what ships: two clean runs at
> 250.70 / 251.14 tok/s with TTFT 18.1 -> 16.6 s and the output byte-identical, and 512K + MTP still
> starts with it (177 cache slots against 179, exit 0, the same four tokens).
>
> The cost is 74 MB of VRAM out of the expert cache, which prefill barely uses - 428 of 63,545 blobs were
> resident.  `gemm gate/up` is unchanged across every arm (5,610-5,632 ms), so this is purely the copy
> wait: 19.3 % of the prompt at 32 slots, 11.6 % at 128.

> ### 0.17 Re-measuring the A stage at the deep ring: batch 16 -> 32 gives another 1.9 %
>
> With the ring at 128 slots the batch cap `stage_n / 2` is 64 instead of 16, so the 16-vs-32 question
> could finally be asked on a copy engine that is not starved.  Two alternating pairs each, 4,063-token
> prompt, output byte-identical in every arm:
>
> | batch | `gemm gate/up` | `wait copy` | prefill |
> |---|---|---|---|
> | 16 (the value shipped in fb1a943) | 5,616 / 5,619 ms | 1,873 / 1,874 | 250.48 / 250.29 tok/s |
> | **32 (new default)** | **4,860 / 4,846** | 2,345 / 2,330 | **255.57 / 255.69** |
> | 48 | 4,626 | 2,897 | 250.56 |
>
> So 32 is the balance and ships: -13.5 % on the A stage's kernels, +1.9 % on the prompt, with the extra
> copy wait (2.3 s against 1.9) paid out of that gain.  48 keeps winning on the kernels and loses it to
> the copies.  Confirmed as the default: **254.65 / 254.93 tok/s**, TTFT 16.4 s, output unchanged.
>
> **The 64-vs-32-row tile was NOT re-measured, and the reason is mechanism rather than convenience.**  The
> measurement that rejected it (`lat_q3k_down_test`, and the engine A/B at 33,701 against 32,828 ms) fails
> for an SM-side reason: a 32-row tile needs *two* tiles for a group larger than 32 rows and each tile
> re-stages that expert's whole 256-row weight block, so the staging per MAC doubles (measured +38 % per
> entry at 32 entries a group).  Nothing in that changes with the PCIe ring depth - the ring feeds VRAM,
> the tile stages VRAM into shared - so the earlier negative stands on its mechanism.  What *is* worth
> asking at this ring is a tile sized to the groups the engine actually builds (~25 entries at chunk 512),
> which is a new kernel shape rather than the 64/32 toggle the old A/B covered.

> ### 0.18 The 32-row A-stage tile, measured at the deep ring: not a lever either way
>
> The experiment the earlier negative deferred, run properly this time: `kTileRows` 64 -> 32 in
> `lat_q3k_fused.cu` with `nb0 = 8*(warp>>2)` and the second row-half disabled (`on1 = false`), so one
> tile covers a 32-row group at 78 % fill instead of 39 % and the compute halves.  Two clean runs each,
> 32 tokens, output byte-identical:
>
> | tile rows | `gemm gate/up` | prefill |
> |---|---|---|
> | 64 (shipped) | 4,865 / 4,855 (batch-32 pair) and 4,973 / 4,975 (reverted, re-verified) | 254.65 / 254.93 and 254.00 / 250.94 |
> | 32 | 4,991 / 4,987 | 254.12 / 253.63 |
>
> So 32 rows is **inside the 64-row arm's own spread** (4,855-4,975 across four runs): the tile size is
> not a lever at batch 32 on a 128-slot ring.  That is worth contrasting with the earlier measurement at
> 32 slots, where the same change was clearly worse (9,777 against 8,420 ms) - the starved ring was
> paying for the extra tiles, so the reviewer's premise was right that the old negative needed re-taking,
> even though the direction turned out to be "no effect" rather than "win".  Reverted to 64 (nothing is
> gained by changing it) and re-verified: prefill 254.00 / 250.94 tok/s, tokens unchanged.
>
> The A stage is therefore 4,860-4,990 ms of compute at ~1.7 TFLOP/s that neither its batch (32 measured
> optimal against 16 and 48) nor its tile size moves.  What is left there is instruction-level: the Q3_K
> decode inside the kernel, the same class of work the U and D tiles needed - and unlike those, its
> weights arrive per group so its staging cannot be amortised further at 25 entries a group.

> ### 0.19 Why the A stage is 4.86 ms: not the decode, not the inner loop - the spills
>
> The prescribed rewrite ("hoist the scale per 16-value sub-block, 4-byte-aligned staged loads") is
> already in `lat_q3k_fused.cu`, twice over:
>
> * `store_chunk` stages each row as `16*s + 4*k` code words with a FLOAT scale at `64 + 4*s`, i.e. one
>   scale per 16-value sub-block, 4-byte aligned;
> * the HIP inner loop is four `__dp4a` per sub-block with `acc += (da * dx) * d` - the same shape that
>   took prefill's U from 23.7 to 2.7 ms a layer;
> * `lat_gu_decode` reads the Q3_K super-block with five `lat_ld4` dword groups (codes, hmask, scales)
>   and unpacks with `__vsub4`/byte_perm - ~18 instructions per 32 staged values = **0.56 a value**,
>   against the inner loop's 0.25 `dp4a` per value per tile row.  With ~25 real rows a group the decode
>   is under 10 % of the instruction stream, so rewriting it cannot move the 4.86 ms.
>
> Where the time actually is, by the same arithmetic: useful work 9.6 TFLOP in 4.865 s = 1.97 TFLOP/s,
> but the 64-row tile executes 2.56x that (39 % fill) = 5.1 TFLOP/s of executed int8 - about 7 % of the
> card's ~73 TFLOP/s dp4a peak.  The inner loop's own issue efficiency is ~73 % (22 instructions per 64
> MAC a lane, i.e. 93 of the 128 MAC a warp `dp4a` can issue), so the loop is NOT the limit either.  The
> gap is occupancy: 256 VGPRs with 28 spilled registers and scratch 196 B/lane pin the kernel to ONE
> block a CU, and the measured remedy for that - `amdgpu_num_vgpr(128)` to fit two - costs 234 spilled
> registers and 2.4x the time (8,424 -> 19,926 ms).  Both directions measured, both worse.
>
> So the A stage's remaining gain needs a kernel with a smaller per-thread accumulator and no shared
> staging to spill through - i.e. 2 blocks a CU at <=128 VGPRs by construction rather than by a cap.  The
> two shape toggles that were available (batch 32 vs 16/48, tile 32 vs 64) are now measured and neither
> moves it.  That rewrite is the next step and it is a new kernel, not a knob.

> ### 0.20 Correction: the A stage is occupancy-bound at 4 waves/SIMD, not spill-bound
>
> 0.19 said "the gap is occupancy: 256 VGPRs with 28 spilled".  The spill half of that is wrong, and the
> compiler says so directly.  The 32-row-tile variant (kTileRows 32, nb0 = 8*(warp>>2), second row-half
> disabled) compiles to:
>
> | | VGPRs | spills | scratch | occupancy |
> |---|---|---|---|
> | 64-row tile (shipped) | 256 | 28 | 196 B/lane | 4 waves/SIMD = **1 block/CU** |
> | 32-row tile | **193** | **0** | 84 B/lane | 4 waves/SIMD = **1 block/CU** |
>
> Zero spills, and the engine measured that build NEUTRAL (gemm gate/up 4,991 / 4,987 against the 64-row
> arm's 4,865-4,975).  So spills are not the cost, and the kernel is limited by having only 16 warps a CU
> to hide the staged-load/dp4a/acc chain across ~20 barriers - i.e. by **4 waves/SIMD**, which is what one
> 512-thread block at 193 VGPRs (or 256) gives.
>
> The distance to the fix is exact: 2 blocks a CU needs <=128 VGPRs (512 x 128 = 65,536 = half the
> register file), i.e. 65 fewer than the current 193 and 128 fewer than the shipped 256.  A cap
> (`amdgpu_num_vgpr(128)`) has already been measured and loses (234 spills, 8,424 -> 19,926 ms), so the
> only way there is a kernel whose live state is small by construction: acc[2][4] instead of [4][2][4],
> the two staged decode chunks held in one buffer instead of two, and the Q3_K decode writing straight to
> shared instead of through `GuStaged` registers.  That is the rewrite, and it now has a number to hit
> (<=128, 0 spills) rather than a hope.

> ### 0.21 The occupancy target is not reachable by the cheap routes - measured three ways
>
> The rewrite's acceptance criterion was "<=128 VGPRs, 0 spills, 8 waves/SIMD".  Three measurements say the
> compiler will not give it up by request, and what the structural route costs:
>
> | build | VGPRs | spills | waves/SIMD | blocks/CU |
> |---|---|---|---|---|
> | 512-thread, 64-row tile (shipped) | 256 | 28 | 4 | 1 |
> | 512-thread, 32-row tile (acc halved) | 193 | 0 | 4 | 1 |
> | 256-thread + `__launch_bounds__(256, 2)` (resource probe) | 256 | 28 | 4 | 1 |
> | `amdgpu_num_vgpr(128)` (earlier) | 128 | 234 | 8 | 2 - and 2.4x slower |
>
> So: halving the accumulator removes every spill (256 -> 193, 28 -> 0) and the engine measured it NEUTRAL;
> asking for two blocks a CU through `launch_bounds` is ignored outright; and the one way to force 128 is
> the attribute that costs 234 spills and 2.4x the time.  8 waves/SIMD therefore needs live state below 128
> by construction - `acc[2][4]`, one staged decode buffer instead of two, Q3_K decode straight to shared -
> and the 65-register gap between 193 and 128 sits in exactly those staging registers.
>
> That is a new kernel, and it is the one remaining item on the A stage.  Between it and the two
> user-gated levers (Q4_0 latent D; the Windows commit limit) nothing else on this engine has an
> unmeasured avenue left.

> ### 0.22 A 35 % cliff in `--prefill auto`: the chunk hinges on 50 MB of ambient VRAM
>
> Two clean runs of the SAME binary, minutes apart, returned 254.93 tok/s and 164.19 tok/s.  The difference
> is not code and not the ring: everything up to "VRAM after the arena load" is byte-identical (4.42 GiB),
> and then the cache sizer reads 1.89 GiB in the good runs and 1.83 GiB in the bad, and the chunk heuristic
> picks accordingly:
>
> | free VRAM at the sizer | cache slots | `--prefill auto` chunk | chunks | blobs | prefill |
> |---|---|---|---|---|---|
> | 1.88-1.89 GiB | 790-793 | **512** | 8 | 63.6k | **250.7-254.9 tok/s** |
> | 1.83-1.84 GiB | 741-753 | **256** | 16 | 105.2k | **164.2-171.7** |
>
> and 1.84 GiB has been observed on BOTH sides, so the threshold sits inside the run-to-run drift of ambient
> VRAM (another process holding ~50 MB is enough).  The ring depth is not implicated - 32 / 96 / 128 slots
> all gave chunk 512 at 1.84-1.88 GiB in one batch - so this predates 257e803 and would have bitten any
> measurement taken on a busy machine.
>
> Practical consequence: `--prefill auto` is not a safe default for measurement on this box (a 3 % VRAM
> difference is a 35 % throughput difference), so every A/B here pins the chunk explicitly, and the harnesses
> should too.  The engine-side fix - hysteresis, or a floor that does not sit on the cliff - is a
> configuration-logic change, not a kernel one, and it is the cheapest remaining robustness win in the tree.

> ### 0.23 The auto chunk is mis-tuned on this box: 256 / 512 / 1024 across cache sizes, and only 512 is good
>
> 0.22 recorded the cliff; this measures the whole landscape.  Same prompt, same binary, varying only the
> expert cache (and therefore the room the chunk/ring budget has):
>
> | cache slots | `--prefill auto` chunk | chunks | prefill |
> |---|---|---|---|
> | 741-753 (1.83-1.84 GiB free) | 256 | 16 | 164.2-171.7 tok/s |
> | **796-803 (1.89-1.90 GiB free)** | **512** | **8** | **248.5-249.0** |
> | 570 (reserve 1024 MiB) | 1024 | 4 | 203.6-205.7 |
> | 572 (reserve 1536 MiB) | 1024 | 4 | 205.7 |
>
> 512 is the optimum by 20-50 %, and the heuristic lands on it only in a narrow band: fewer cache slots
> makes the ring affordable at a *bigger* chunk (1024, which streams 80k blobs) and slightly fewer makes
> it unaffordable at 512 (256, 105k blobs).  So `--prefill auto` is not merely fragile at one threshold -
> its optimum window is one step wide on this hardware.
>
> Shipped fix, part 1 (code): the staging-ring default is 96 slots, not 128.  The budget demands a FULL
> ring at the chosen chunk, so the 37 MB a 128-slot ring takes was enough to push this box past the point
> where chunk 512 can still have one - chunk 512 came back at 796-803 slots with prefill 249.00 / 248.54
> tok/s and the output byte-identical, and the margin is now ~50 MB instead of ~0.
>
> Still open (and the cheapest remaining win in the tree): the heuristic itself should prefer the measured
> optimum rather than a binary "ring full" test - a chunk of 512 for this pack, with hysteresis so a 3 %
> VRAM drift cannot move it.  That is configuration logic in `generate.cpp`'s `scan`/`host_plan_chunk`,
> not a kernel.

> ### 0.24 The auto-chunk cost model, specified from the data (not yet implemented)
>
> The scan in `generate.cpp` (plan_lend, ~8893-8905) maximises the chunk subject to "the ring is full"
> (`room_of(t) >= ring_max`) and the lend cap.  That objective explains two of the three measured points
> and fails the third:
>
> | cache slots | what the current rule picks | measured prefill | what it should pick |
> |---|---|---|---|
> | 741-753 | 256 (512 cannot have a FULL ring) | 164-172 tok/s | 512 |
> | 796-803 | 512 | 248.5-249.0 | 512 |
> | 570-572 | 1024 (the ring is full there too) | 203.6-205.7 | 512 |
>
> Two separate defects, and they need different terms:
>
> 1. **A binary floor on the ring size.**  A chunk is rejected when its ring is 3/4 full, so a 3 % VRAM
>    drift halves the chunk.  Fix: accept `room_of(t) >= ring_max * 3 / 4` and take the larger chunk
>    (hysteresis), which buys 512 back at 741-753 slots.
> 2. **No term for streamed expert bytes.**  At 570 slots the ring *is* full at 1024, so the rule is
>    satisfied - but 1024 streams **80k blobs against 512's 63.6k** for the same prompt, which is what the
>    20 % is.  The planner currently prices buffers only (`bytes_needed` / `bytes_needed_no_ring`), and no
>    amount of ring hysteresis can see this.
>
> So the objective has to become bytes **per token**, streaming included:
>
>   cost(T) = bytes_needed_no_ring(g, ss, T, src) / T  +  E_distinct(T) * max_blob / T
>
> with the second term estimated from the geometry - the engine knows `n_expert`, `K` (routed per token)
> and `n_layers`, and the measured anchors are 63.6k blobs at chunk 512 and 80.1k at 1024 on 48 layers x
> 4,063 tokens, i.e. ~165 and ~240 distinct experts per layer-chunk.  A saturating form
> `E * (1 - exp(-a * T * K / E))` fits those two anchors with a ~ 0.55, and its shape is what matters:
> the streaming term's slope falls with T while the buffer term's falls faster, so the minimum lands near
> 512 for this pack and moves with the pack's geometry instead of being hard-coded.
>
> Verification the change must pass (already measured, so it is a like-for-like check): chunk 512 with
> ~249 tok/s and byte-identical output at all three cache sizes, plus the 512K + MTP configuration still
> starting (177 slots, exit 0) - the last one is why this is not a one-line cap.

> ### 0.25 The auto-chunk defect is real and now measured at a FIXED cache size
>
> 0.24 compared chunk 512 (796-803 slots) against 1024 (~570 slots), which confounds the chunk with the
> cache.  Same prompt, same cache (~557-570 slots, `--vram-reserve-mib 1024`), alternating pairs:
>
> | arm | chunk | prefill |
> |---|---|---|
> | `--prefill auto` (picks 1024) | 1024 | 203.31 / 201.66 tok/s |
> | `--prefill 512` (pinned) | 512 | **244.16 / 244.14** |
>
> So **-21 % from the auto pick alone**, at identical cache size and byte-identical output.  The defect is
> proven; my first cost-model attempt was not the fix (it did not fire as intended and its premise was
> drawn from the confounded pair), and it is reverted.
>
> What the model must reproduce, and the one thing that is still unclear: the ring-full test CANNOT
> separate 512 from 1024 here (`room_of(t)` grows as t falls and is capped at ring_max, so both fill it),
> which means `scan` is really "the biggest chunk whose lend cap fits" - and at ~560 slots 1024 fits.  Why
> 1024 then loses 21 % needs one more measurement before a formula can be written: at the same cache,
> what do the blobs and `resident` look like for the two arms (0.22's proxy: 63.6k blobs at 512 against
> 80.1k at 1024 came from *different* cache sizes and may not hold).
>
> Until the formula is derived from that pair, the shipped behaviour is unchanged (0.23's ring 96 gives
> chunk 512 at 796-803 slots, 249.00 / 248.54 tok/s) and every harness pins the chunk explicitly.

> ### 0.26 The half-chunk rule (attempt 2) also did not fire - the early return bypasses the scan
>
> 0.25 proved the defect at a fixed cache (512 pinned 244.16 / 244.14 against auto's 1024 at 203.31 /
> 201.66) and found the mechanism in the same pair: 1024 streams 81,684 blobs against 62,554 and keeps
> 854 experts resident against 1,519, because a bigger chunk lends the prompt path more cache slots.
> Attempt 2 was a bounded rule - "while c > 512 and half of c also fits with a nearly full ring, take the
> half" - inserted right after `int64_t c = scan(ring_max);`.
>
> It changed nothing: at 782 / 674 / 557 cache slots the auto chunk stayed 512 / 256 / 1024.  The reason
> is above the loop, not in it: `if (small >= c) { ... return slots_for(small); }` returns 0.1.39's chunk
> *before* any refinement runs, and at 557 slots `old_rule()` already returns 1024, so `small >= c` and
> the function exits with 1024.  Any future fix has to be placed above that early return (or make it
> compare against the refined value), which is where attempt 1 also failed.
>
> Reverted unshipped again.  The shipped behaviour is unchanged (ring 96, chunk 512 at 782-803 slots,
> 249.00 / 248.54 tok/s) and the two arms' blob/resident numbers above are the honest anchors for the
> cost model whenever it is written.

> ### 0.27 Where the auto chunk is actually decided: `host_plan_chunk`, not `plan_lend`
>
> Attempt 3 (halving inside `plan_lend`, first above the `small >= c` early return and then in the
> `chunk = c` path) changed nothing again - 512 / 256 / 1024 at 783 / 672 / 557 cache slots.  A diagnostic
> inside both paths settled why: **neither path prints**, i.e. `plan_lend`'s automatic result is then thrown
> away.  The chunk printed by "prompt chunk auto" comes from the #2824 host plan:
>
>   generate.cpp:17973-17998   if (k > 0 && o.prefill_auto && n_batched > 1024)
>                                chunk = host_plan_chunk(...)   // generate.cpp:8937, prefill_plan.hpp
>
> `host_plan_chunk` is a HOST-TIER model: it prices the chunk against the page cache (`available - 1.2 GiB`),
> the pack's file tier, the measured H2D rate (`g_last_h2d_gbps`), the SSD read rate and a compute rate, and
> returns the chunk that minimises total time across those tiers.  On this box every prompt > 1024 tokens
> takes that path (`n_batched` is 4,062), so *all* of the auto-chunk behaviour measured in 0.22-0.26 came
> from it - and none of my three patches touched it.
>
> That also explains the shape of the results: the model has no term for the VRAM each chunk lends, which is
> what actually costs the streamed experts (1024 -> 81,684 blobs and 854 resident against 512's 62,554 and
> 1,519 at the same cache), so it happily trades cache capacity for a shorter chunk count.  The fix belongs
> in `prefill_plan`'s cost: the GPU tier's slot count has to shrink with the chunk (`lend_of` already hands
> the model those numbers - it is called but its result only prices bytes, not the experts it evicts).
>
> All three attempts are reverted; the shipped behaviour is unchanged (ring 96, chunk 512 at 783-803 slots,
> 249.00 / 248.54 tok/s, byte-identical, 512K+MTP exit 0).  The next attempt starts in `prefill_plan.hpp`,
> not in `plan_lend`.

> ### 0.28 Per-chunk streaming, instrumented: at chunk 1024 the prompt streams the whole pack TWICE
>
> `STRATA_CHUNK_TRACE=1` prints what each chunk actually streamed (prefill.cpp, at `++stats_.chunks`), at a
> fixed 552-slot cache:
>
> | chunk | per-chunk streamed | series | total |
> |---|---|---|---|
> | 512 | ~7,800 | 7,758 / 7,282 / 7,365 / 7,497 / 8,115 / 7,843 / 8,372 | 62,589 |
> | 1024 | **24,024** | 24,024 / 24,024 / 24,024 (+ the 991-token tail) | 81,818 |
>
> The pack has 48 layers x 256 experts = **12,288 experts in total**, so chunk 1024 streams *every expert
> twice* within one chunk, where 512 streams 63 % of them once.  That is the whole 21 %: not the cache
> size, not the ring depth, but **re-staging inside a chunk** - a chunk's working set (up to 256 distinct
> experts a layer) no longer fits the 96-slot ring, so experts needed by entries in two different A-stage
> batches are copied twice.
>
> So the planner's `stream = experts_total - gpu_cache + lend_bytes` is wrong in kind, not just in scale:
> it can never exceed one pack's worth, while the engine can stream 2x.  And the cheaper fix may not be in
> the planner at all - sizing the ring to the chunk's working set would make a bigger chunk competitive
> instead of penalising it, which is the trade the automatic rule is trying to find.
>
> The instrument ships (env-gated, one line per chunk); nothing else changed.

> ### 0.29 Full component trace: hc1 norm (decode) and PLE SSD reads (prefill) are the unexamined costs
>
> One recorded run (`trace_full.log`, chunk 512 pinned, ring default, `STRATA_PREFILL_TIMING` +
> `STRATA_VERIFY_PROFILE` + `STRATA_VERIFY_U_PROF`), 4,063-token prefill then 32 decode tokens.
>
> **Prefill, 17,702 ms of marked device time** (the profiler inflates absolute figures; shares hold):
>
> | component | dev ms | % dev | note |
> |---|---|---|---|
> | `gemm gate/up` = the A stage | 5,298 | 29.9 | 4,180 marks |
> | gdn block (gdn 3,077 + out-proj 374 + rec 303 + conv 46) | 3,800 | 21.5 | |
> | `wait copy` | 2,530 | 14.3 | |
> | hc reads (down+silu 1,077, up 320, inject 167, mix 101) | 1,665 | 9.4 | |
> | `gemm down` = U | 1,170 | 6.6 | |
> | qsa (attn 631, proj 412, select 30, indexer gemms 50, prep 29) | 1,152 | 6.5 | |
> | `gather` = D | 970 | 5.5 | |
> | router+shared 578, hc write 146, embed 81, ple 68, combine 61 | 934 | 5.3 | |
>
> and off the device timeline: **host grouping 9,841 ms of host time (55.6 %)** - overlapped, but it is the
> single largest host cost - and **PLE I/O: 69,736 SSD reads for 294.8 MB, p50 5,664 us, p99 9,257 us,
> blocked 1,562.6 ms**.  That is 4.2 KB a read: the reads are far too small to hide latency, and 1.56 s is
> 8.8 % of the prompt on its own.
>
> **Decode round, 116.67 ms with the profiler on** (~86-90 unprofiled), T4:
>
> | component | ms | % | |
> |---|---|---|---|
> | **`hc1 norm`** (hyper-connection FFN read, norm stage) | **26.76** | 22.9 | largest single item |
> | `u: gemv` (GPU-rows U + q8_1 of H) | 18.05 | 15.5 | |
> | dense GEMVs (qkv 6.32, out-proj 5.23, z 3.84, q+q-idx 2.86) | 18.25 | 15.6 | |
> | `u: matvec` (CPU-rows U) | 14.62 | 12.5 | |
> | `u: pool wait` | 6.28 | 5.4 | pool at its 30-32 GB/s ceiling |
> | hc-read0 5.42, VRAM hits 4.71, head 4.48, waitB 3.95, shared+quant 2.60, attention 1.24 | 22.40 | 19.2 | |
> | u: copy 0.25, conv 0.25, ab 0.41, rec 1.37, copy+combine 0.15, gap 0.03 | 2.46 | 2.1 | |
>
> Plus: the pool itself is **26.27 ms/token over 48 layers** (624 dispatches, drain 23.07 ms/token), and
> **R4 expert-cache hits 0 of 0** - the cache supplied nothing in this window, so every expert came through
> the pool or PCIe.
>
> So the ranking the earlier tables missed: **hc1 norm is the biggest decode item (22.9 %), larger than
> either U half**, and the PLE read pattern is the largest unexplored prefill cost.  Both are new, both are
> named here with counts and latencies, neither has been touched.

> ### 0.30 PLE I/O: batching is impossible and prefetch depth is a measured non-lever
>
> The PLE line is 69,736 random reads of 4.2 KB for 294.8 MB, `blocked 1,562.6 ms`.  Three facts close the
> obvious fixes:
>
> * **batching cannot work.**  The table is **47.68 GB** and a prompt needs 69,808 rows = **0.58 % of it**,
>   scattered (row cache hits 0.1 %).  Coalescing nearby rows would read ~170x more than it uses.
> * **`--ple-io ram` cannot fit.**  47.68 GB against 48 GB of RAM with a 26.66 GB arena already pinned.
> * **the read is latency, not queue depth.**  `--ple-inflight` is already 256, and the prefetch queue depth
>   was A/B'd at 16 / 48 / 128 tickets in alternating runs: `blocked` 1,553.1 / 1,555.0 / 1,560.2 ms and
>   1,564.4 / 1,565.4 ms, wall 16,929-16,980 ms, prefill 239.0-239.6 tok/s, output byte-identical.  Flat.
>
> So `blocked` is one read round-trip per layer-chunk: 1,562 ms over 384 layer-chunks = **4.1 ms a layer**
> against a 5.6 ms p50 read.  The lever is **pipelining** - asking for layer N+1's rows before layer N
> computes - not a deeper queue.  That is a change to the prefill loop's call pattern, and it is the
> recorded next step for this item.
>
> `STRATA_PLE_PREFETCH=N` (4-128, default 64) ships as the diagnostic the A/B used.

> ### 0.31 Lean: the auto chunk is capped by geometry, no harness pinning needed
>
> The correction that made this work: this pack has **512 experts a layer (24,576 in all**, not 12,288 - the
> profile line says "built for 24576 slots").  So a 1,024-token chunk streams 24,024 experts = **the whole
> pack once**, and 512 streams 7,819 a chunk.
>
> With that, the engine caps its own automatic chunk: a pack with <=512 experts a layer never takes more
> than 512 tokens, because a larger chunk re-streams a layer's working set.  Measured with no pinning and no
> STRATA_* override, `--prefill auto` alone:
>
> | cache slots | auto chunk before | after | prefill |
> |---|---|---|---|
> | 735 | 512 | 512 | 239.47 |
> | 511 | 1024 | **512** | **234.48** (was 192.82 with 1024: **+21.6 %**) |
> | 626 | 256 | 256 | 166.93 |
>
> Byte-identical output where the cap fired.  The 626-slot case still takes 256 because that is the host-tier
> planner's own fit choice - the remaining planner defect (0.24-0.27), now bounded to that band.
>
> **hc1 norm resolved as well** (the interrupted test): with the persistent read ON the whole HC read is
> charged to the `norm` slot (26.19 ms); turning persistence OFF splits it and shows the truth - **`up`
> 24.61 ms**, down 2.89, **norm 0.30**.  So the 22.9 % decode item is the FFN-side hyper-connection *up
> projection*, at ~0.51 ms a layer for 4 tokens: a latency-bound small-batch GEMV, not a norm.

> ### 0.32 RDNA2 feature audit by measurement: cache policy and non-temporal loads are dead, and `hc1 up` is a live anomaly
>
> The RDNA2 ISA summary lists four unused features. Two of them are now measured, and the answer is no.
>
> **Non-temporal (streaming) loads on the expert stream - measured, and they HURT.** `tools/hip/q3k_nt_bench`
> on this card, real Q3_K blob geometry (0.671 MB an expert), 96 experts = 64.5 MB (~21x the 3 MB L2):
>
> | case | cached | non-temporal | delta |
> |---|---|---|---|
> | stream, no reuse | 480.2 GB/s | 486.9 GB/s | **+1.4 %** |
> | reuse 4x an expert | 1756.2 GB/s | 1628.5 GB/s | **-7.3 %** |
> | reuse 16x an expert | 3589.3 GB/s | 2410.3 GB/s | **-32.8 %** |
>
> Byte-identical results in every arm. A prompt chunk routes many rows to the same expert, so the real workload
> is the reuse case, not the pure stream - non-temporal loads would cost 7-33 %. (The 480 GB/s on a 384 GB/s
> part is the 96 MB Infinity Cache doing its job; the card's real DRAM rate is ~170 GB/s, see below.)
>
> **Cache policy (glc/slc, "cache-less") - measured flat.** `tools/hip/cache_policy_bench`, a 678.6 MB region
> (> Infinity Cache, so DRAM-led): plain 169.8 GB/s, `__ldg` 169.8 GB/s, glc+slc 170.2 GB/s, 20.95 / 20.96 /
> 20.90 ms. Nothing to win. 170 GB/s is the honest DRAM figure this card reaches.
>
> **`hc1 up` is an anomaly, and it is NOT the kernel.** The isolated read bench (`fused_gr_bench`, hipified),
> cold over 48 weight sets = 48 layers (624 MB, no L2 residency), 4 tokens, per-stage GPU stamps:
> `norm 11.2 / down 68.6 / up 56.8 us` a layer - i.e. 0.54 / 3.33 / 2.73 ms a round. The engine measures
> `hc1 norm 0.298 / hc1 down 2.894 / hc1 up 24.608`. **down and norm match the bench; up is 9x it.**
>
> Two facts pin it down:
>
> 1. **It is not the up kernel.** Every up branch gives the same figure in alternating runs with the persistent
>    read off: default (`gr_up_q32_kernel`) 18.351 / 18.331 ms, and `STRATA_QFUSE_HC_LOCAL=0`
>    (`gr_up_multi_kernel` + `gr_q8_tail`) 18.328 ms. Output byte-identical throughout.
> 2. **It does not scale with tokens.** `hc1 up` is 17.064 ms at T=1 and 18.283 ms at T=4, while `down` goes
>    1.822 -> 2.157. So the cost is one weight read a layer (~0.49 ms), not arithmetic.
>
> That gives the page the two stages contradicting each other on identical byte counts, same layer, back to back:
>
> | stage | bytes a layer | ms a layer | implied bandwidth |
> |---|---|---|---|
> | `hc1 down` | 6.55 MB | 0.061 | 107 GB/s (VRAM-class) |
> | `hc1 up` | 6.55 MB | 0.522 | **12.6 GB/s (PCIe-class)** |
>
> Both arrive as `(const uint16_t*) wu/hd[half]->data` (verify.cpp:1620), so the pointer *provenance* is the same;
> what differs is which memory the two tensors actually landed in. `NativeDense::hc_bf16_view` has exactly such a
> fork - `w.data` (direct) vs `w.hc_q8` (dequantized into scratch *on every call*) - and the decode path never
> goes near it, so the decode-side residency is still unconfirmed. A one-time diagnostic now prints each HC
> weight's storage ("direct bf16 in device memory" / "Q8, dequantized into scratch on EVERY call" / MISSING).
>
> **Next experiment, precisely:** print the storage class and pointer for `wu[half]`/`hd[half]` on the verify
> path, then probe the achieved bandwidth of that exact pointer with the same kernel that gets 107 GB/s on
> `w_down`. If `w_up` is mapped host memory, the fix is to place those 314 MB in VRAM, and the prize is 21.9 ms of
> the 113 ms round - decode 20.3 -> ~25 tok/s profiled, ~26 -> ~32 unprofiled.
>
> **The expert cache, by contrast, is already at the literature's optimum and is not a lever.** Two results:
> Lin et al. (arXiv 2511.05814) find LRU suboptimal for MoE offloading and LFU the improvement, plus speculative
> prefetch - this engine already ranks its cache by activation frequency from `profile.bin`, which is LFU by
> construction. Liang et al. (arXiv 2505.16056) measure 20 MoE models and find the effective cache is ~2x the
> active experts; `--expert-cache auto` here gives 735 slots for a 512-expert layer, in that band. The trace
> agrees: `VRAM hits` is 4.71 ms of the round.
>
> **MAPLE (AmSach/maple) is not an engine change.** It is *pruning*: profile the activation manifold, keep the top
> K experts a layer, mask the excised router logits to -inf, and re-emit a smaller pack (DeepSeek-Coder-V2-Lite
> 64 -> 54 experts, 8.57 GB -> 4.50 GB). Its score is `s = C * sqrt(sigma)`, whose `C` is exactly what
> `--expert-profile` already records, so the profile we keep is the calibration it wants. Two consequences for
> this repo: it changes the pack, not the engine, and it has no quality gate here - the reason Q4_0 for the latent
> D table is still blocked applies unchanged. It would shrink the 73 GB expert stream, which is the prefill lever,
> so it is worth doing *with* a KL/NLL gate, not before.

> ### 0.33 hc1 up: five hypotheses eliminated, and the two halves disagree 9x in one trace
>
> The residency hypothesis of 0.32 is **wrong**, and it is now proved wrong rather than argued. `verify.cpp`'s
> `gr_read_group` prints each HC weight's memory class and, on this card:
>
> ```
> strata hc: attention-side HC weights: w_norm DEVICE (VRAM) | w_down DEVICE (VRAM) 6.2 MB | w_up DEVICE (VRAM) 6.2 MB
> strata hc: attention-side HC extras: w_inject DEVICE (VRAM) | q8_mixed target DEVICE (VRAM) (native layout 1) | hc_q8 down null up null
> strata hc: FFN-side HC weights: w_norm DEVICE (VRAM) | w_down DEVICE (VRAM) 6.2 MB | w_up DEVICE (VRAM) 6.2 MB
> strata hc: FFN-side HC extras: w_inject DEVICE (VRAM) | q8_mixed target DEVICE (VRAM) (native layout 1) | hc_q8 down null up null
> ```
>
> Everything is VRAM, everything is plain bf16, nothing is dequantized per call. (A device-to-device bandwidth
> probe cannot be used here: `cudaMalloc` is illegal inside the captured decode graph, which these diagnostics
> also discovered - the decode path runs under graph capture.)
>
> Five hypotheses, all eliminated with alternating runs and a byte-identical 32-token output every time:
>
> | hypothesis | test | result |
> |---|---|---|
> | the up kernel changed | default `gr_up_q32_kernel` vs `STRATA_QFUSE_HC_LOCAL=0` (`gr_up_multi_kernel` + `gr_q8_tail`) | 18.351 / 18.331 / 18.328 ms - flat |
> | it is arithmetic | T=1 vs T=4 | 17.064 / 18.283 ms - flat, so it is one weight read a layer |
> | the q8_1 image write | `STRATA_QFUSE_HC=0` | 18.418 vs 18.433 ms - flat |
> | host-mapped weights | `cudaPointerGetAttributes` on all four tensors | all DEVICE (VRAM) |
> | the read variant | `STRATA_HC_SPLIT=0` / `=1` / default staged | 18.452 / 18.402 / 18.345 ms - flat |
>
> **What the trace does show** is that the two halves of the same projection disagree by 9x inside one window:
> the attention-side up is **2.000 ms** a round (slot 29, 42 us a layer - matching the isolated kernel's 57 us),
> and the FFN-side up is **18.433 ms** (slot 32, 383 us a layer). Same 6.2 MB, same kernel family, both VRAM.
> That contrast is the new lead, and it rules out anything global to the read.
>
> The profiler semantics were verified while chasing this, and are worth recording because the slot names mislead:
>
> * slot 30/31/32 for the FFN read are `norm` / `down` / `up`, computed at verify.cpp:3361 as
>   `gap(30,16)`, `gap(31,30)`, `gap(32,31)`; the FFN read itself only writes stamps at base+0 and base+1, so
>   `up` is closed by the caller's `stamp(l, 32)` at verify.cpp:2276 ("the FFN-side read ends here").
> * slot 29 is labelled `u: copy` but is also where the hc-read0 split puts its `up` (verify.cpp:3409). The
>   label collision is why the attention-side up has been invisible.
> * slot 32 is written a second time at verify.cpp:2592 ("slot 32: unused"), but only under `seg_split`, which
>   these runs do not take - so there is no collision here, and the 18.4 ms is the up stage itself.
>
> Next: the attention half is fast and the FFN half is not, with identical tensors, so the difference is in the
> *call*, not the data - the FFN read is `gr_read_group(1, true, ...)` with `apply` forced true, so its up does
> the per-row `R_out` store and the `bo_prev`/`inj_prev` loads on every one of the 10,240 rows that the attention
> half skips when `apply` is false. That is the one asymmetry left to measure: stamp inside the FFN up's prologue
> versus its stores.

> ### 0.34 hc1 up: the state is VRAM too, the stores are ruled out, and the anomaly is now isolated but unexplained
>
> Extending 0.33's diagnostic to the activation state closes the last memory hypothesis:
>
> ```
> strata hc: attention-side HC state: R DEVICE (VRAM) | R_out DEVICE (VRAM) | bo_prev DEVICE (VRAM) | lo DEVICE (VRAM)
> strata hc: FFN-side HC state: R DEVICE (VRAM) | R_out DEVICE (VRAM) | bo_prev DEVICE (VRAM) | lo DEVICE (VRAM)
> ```
>
> And the per-row store is ruled out by data already taken. The FFN read runs with `apply` forced true, so it
> stores `R_out[i]` on all 10,240 rows - 40,960 scattered 4-byte stores a token. If those dominated, T=1 (one
> store a row) would be about four times faster than T=4 (four stores a row); it is **17.064 vs 18.283 ms, 7 %**.
>
> So the standing ledger for this item, all with a byte-identical 32-token output:
>
> | hypothesis | result |
> |---|---|
> | up kernel choice | flat: 18.351 / 18.331 / 18.328 ms |
> | arithmetic / token count | flat: 17.064 (T=1) / 18.283 (T=4) |
> | q8_1 image write | flat: 18.418 vs 18.433 |
> | weights in host memory | no - all DEVICE (VRAM), plain bf16, `hc_q8` null |
> | read variant | flat: 18.452 / 18.402 / 18.345 |
> | activation state in host memory | no - all DEVICE (VRAM) |
> | per-row `R_out` stores | no - 7 % from T=1 to T=4 |
>
> **What remains, stated as the fact it is rather than a theory:** the attention-side up is **2.000 ms** a round
> (42 us a layer, matching the isolated kernel's 57 us and a 148 GB/s read) and the FFN-side up is **18.433 ms**
> (383 us a layer, 16 GB/s) - same 6.2 MB, same kernel family, same everything measurable. Nothing global to the
> kernel, the data or the memory explains a 9x split between two calls of the same code, so the difference is in
> how the FFN call is *made*: it is the only one inside the group loop with `apply` forced true, and it is the
> call whose output feeds the MoE router. The instrument that settles it does not exist yet - a stamp pair
> immediately around the up launch itself, which needs one spare profile slot; slot 15 is free and `at(l,15)` is
> never written, so that is the change to make next, and it costs one build and one run.
>
> This item is **not** fixed, and no speedup has been claimed for it.

> ### 0.35 SOLVED (attribution): `hc1 up` was the latent D stage, and the shipped D kernel is the best of three
>
> The seven failed hypotheses of 0.33/0.34 have one explanation, and it is in the source rather than the kernel:
> **between the FFN-side HC read and `stamp(l, 32)` the engine runs the whole decode-side latent D stage.**
> verify.cpp 2239-2280, inside that gap: `quantize_q8_1_rows(mixed_)`, `iq_mmvq` over `[G*ELL, N]`, the Q3_K
> `lat_colscale_rows`, `quantize_q8_1_rows(lat_xl_)`, `quantize_q8_0_rows`, two `cudaMemcpyAsync` D2H, and a
> doorbell. So slot 32 - labelled `hc1 up` - measured the up projection *and* a ~16-22 ms D stage, which is why
> varying the up kernel (18.351/18.331/18.328), the read variant (18.452/18.402/18.345), the token count
> (17.064/18.283), the q8_1 image (18.418/18.433), the weights and state memory classes (all VRAM) and the
> per-row stores (7 %) never moved it: a ~2-10 ms kernel was sitting inside a ~20 ms bucket.
>
> The fix is instrumentation only. Slot 15 is never written anywhere else in verify.cpp, so it now closes the
> read's `up`, and slot 32 keeps the D stage it actually measures. Measured with the persistent read off:
>
> ```
> GDN: qkv gemv 6.320 ... rec 1.390  hc1 up 10.124  shared+quant 1.972 ... (gap) 0.026 head 4.472
>      Dstg|hc0n 0.715  rtpl|hc0d 2.427  u: copy 2.010  hc1 norm 0.226  hc1 down 2.170  D stage 16.627
> QSA: ... gate 0.041  hc1 up 3.484 ... Dstg|hc0n 0.077  rtpl|hc0d 0.811  u: copy 0.676
>      hc1 norm 0.080  hc1 down 0.788  D stage 5.804                                       | total 120.04 ms
> ```
>
> Two caveats stated rather than hidden: adding the 48 stamps grew the window 116.65 -> 120.04 ms, so the
> profiler's own stamps are not free; and up+D (36.1 ms) does not equal the old single bucket (24.88 ms), so
> these are attribution figures, not yet a clean baseline. What they do establish is that the D stage is the
> big half and that it was invisible.
>
> **A/B of the three D implementations** (`STRATA_VERIFY_D_TILE`, no rebuild), with the read's cost folded into
> `hc1 norm` under the shipped default so the whole read + D is one number, 32-token output byte-identical in
> all four arms:
>
> | arm | `hc1 norm` (GDN) | vs shipped |
> |---|---|---|
> | **D0 `iq_mmvq` (shipped)** | **19.606 / 19.723** | - |
> | D2 `latent_d_fast` | 23.625 | +4.0 ms |
> | D1 `latent_d_tiled` | 24.208 | +4.6 ms |
>
> So the shipped choice is already the best of the three by ~4 ms, and the doc comment at the call site ("the
> staged tile loses at M=4: 21.3 vs 15.3-17.4 ms") is confirmed on this card. **No speedup in this item** - but
> the target is now named: a small-M GEMV at M=4 over `[G*ELL, N]`, ~16-22 ms a round, about a fifth of decode.

> ### 0.36 Baseline reconciliation: it reproduces at equal cache size, and the 249 needs VRAM this box no longer has
>
> The recorded 249.0/248.5 was taken at **796-803 cache slots (1.89-1.90 GiB free)**.  The box now reports
> **1.81-1.83 GiB free**, so `--expert-cache auto` lands on 720-742 and the containment was mistaken for a
> regression.  Re-running the canonical harness unmodified, and the same harness with the cache pinned to the
> recorded working set:
>
> | arm | slots | chunk | prefill | decode | output |
> |---|---|---|---|---|---|
> | canonical `auto`/`auto` | 720 | 256 | 167.41 | 19.40 | valid |
> | canonical `auto`/`auto` (again) | 723 | 256 | 167.53 | 19.92 | valid |
> | `--expert-cache 800`, chunk 512 | 800 | 512 | 238.94 | 19.20 | **ALL ZEROS** |
> | `--expert-cache 800`, chunk 512 (again) | 800 | 512 | 239.62 | 19.24 | **ALL ZEROS** |
> | earlier clean arms, 735 slots | 735 | 512 | 236.48 / 236.83 | 20.68 / 20.02 | valid |
> | recorded `trace_full.log` (this box, 04:10) | 742 | 512 | **239.68** | 19.44 | valid |
> | recorded `final1`/`final2.log` (04:01) | 973 / 980 | 256 | 178.02 / 172.51 | 20.34 / 20.26 | valid |
>
> **Prefill reproduces**: 236.5-239.7 today against the recorded 239.68 at the same 735-742 slots - 1.3 %.
> The 249 needs ~80 MB more VRAM than the box has now, so it is machine state, not a code change, and nothing
> in this stretch regressed.  **Decode reproduces too**: 19.2-20.7 today against 19.44-20.34 in the recorded
> logs.  The "25-26 tok/s" that the reconciliation brief compared against is **not in any canonical log** - the
> only 25.38 in this document (0.19) is `STRATA_VERIFY_U_TILE`'s A/B from a different session state, and the
> "~26 -> ~32 unprofiled" at 0.32 was my own projection, later read back as if it were a measurement.
>
> **A real bug surfaced while trying to force the recorded state.**  `--expert-cache 800` is accepted
> ("expert cache 800 slots, 0.87 GiB of VRAM") when the budget only affords ~800 MiB, and then the loader
> **refuses 8,177 expert loads and the model produces all zeros** - the engine's own stats line:
>
> ```
> R4 expert-cache hits   6560 of 14737 = 0.4451   (0 admitted, 8177 refused, cache 100.0000% full)
> output  : 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
> ```
>
> A pinned cache that does not fit must fail loudly, not emit zeros; that exit code was 0 and the throughput
> lines were reported as if they meant something.  This is the one thing in this section that needs a code
> change, and it is a guard in the `--expert-cache <N>` path (compare the requested slots against the same
> budget `auto` computes, and refuse the start), not a kernel.
>
> Also noted: under `--expert-cache auto` the stats report `0 of 0 ... 0 admitted, 0 refused, cache 0.0000 %
> full`, i.e. the R4 counters are not being driven at all on that path, while the pinned path drives them
> (14,737 lookups).  Whoever next reasons about cache hit rate from that line under `auto` will be misled.

> ### 0.37 SHIPPED, +38 % prefill on the DEFAULT config: the staging ring default 96 -> 80
>
> This is the first time this stretch removed time, and it removes it from the configuration every user gets,
> with no pinning.
>
> 0.36 showed the box now offers 1.80-1.82 GiB of VRAM where the 249 tok/s tuning had 1.89-1.90 GiB.  The auto
> scan's binding test is not the ring fill but the lend headroom, `chunk + ring + 128 <= slots`:
>
> * ring 96: `512 + 96 + 128 = 736 > 721` -> chunk 512 rejected -> auto falls to **256** -> **165.92-167.53 tok/s**
> * ring 80: `512 + 80 + 128 = 720 <= 721` -> chunk 512 kept -> **231.19-236.30 tok/s**
>
> The 3/4-ring hysteresis I tried first changed nothing (167.34 / 167.27 / 167.64 / 167.83 across four
> alternating arms) because the ring *fill* was never the binding condition.  The ring depth itself is flat at a
> fixed chunk on this card - chunk 512 with 96 / 80 / 112 slots gives **235.34 / 235.46 / 237.12** tok/s, 0.8 %
> apart - so the 16 slots cost nothing and buy the chunk.
>
> Shipped as `stage_slots()`'s base (prefill.cpp), with the whole history in the comment: 96 was chosen when the
> margin existed, the comment there even predicted this exact failure ("pushes this box past the point where chunk
> 512 can still have a full ring and there the chunk collapses to 256 and prefill falls from 250 to 165").
>
> Verified on the shipped default, no env overrides, `--prefill auto --expert-cache auto`:
>
> | | chunk | prefill | decode | output |
> |---|---|---|---|---|
> | before (ring 96) | 256 | 165.92 / 167.34 / 167.53 | 19.40-20.45 | byte-identical |
> | **after (ring 80)** | **512** | **231.39 / 231.19** | 20.04 / 20.37 | byte-identical |
>
> **+38.4 %** prefill (165.92 -> 231.39).  Decode unchanged inside its noise.  `run_512k_mtp.ps1 -Embed 1`
> still passes: **exit 0 after 31 s**, `output : 715 1822 1822 1822`, the same four tokens as the recorded
> 512K + MTP runs, so the never-break constraint holds.
>
> Repro harnesses: `rs_ab.bat` (auto chunk vs ring depth), `cr_ab.bat` (ring depth at a fixed chunk),
> `hy_ab.bat` (the hysteresis arm that did nothing).

> ### 0.38 The decode latent D stage is already at the DRAM ceiling, and the wide up kernel is worse AND wrong
>
> **The D stage cannot be made faster by a better kernel, and packed fp16 would make it twice as slow.**  The
> geometry is in verify.cpp: `lat_G_ = 8`, `lat_ELL_ = lay0.fmt[0].n_embd = 2560`, `N = 2560`, and the table is
> Q8_0 (the pack ships `latent_dense_q8.bin`, `hc_q8` null, `lat_fmt` = 8).  So one layer's D matrix is
>
>   8 x 2560 rows = 20,480 rows, each a dot of 2560 = 80 blocks of 32, at 34 B a block
>   -> 20,480 x 80 x 34 B = 53.1 MiB a layer, **2.38 GiB a round over 48 layers**
>
> Against the measured 16.5 ms that is **155 GB/s** (and 167 GB/s at the call site's own 15.3 ms note), on a card
> whose real DRAM ceiling `cache_policy_bench` puts at **170 GB/s**.  The D stage is therefore at the memory
> bandwidth limit, not the ALU limit, which also explains why all three of its implementations land within 4 ms
> of each other: they are all doing the same unavoidable byte traffic.
>
> Consequences, both measured rather than assumed:
>
> * **A packed-fp16 `v_pk_`/`V_DOT2_F32_F16` D kernel is the wrong tool**: Q8_0 is 34 B per 32 weights =
>   1.0625 B/weight, while fp16 would be 2 B/weight, so it doubles the stream this stage is bound by.  It would
>   also have to unpack to do it.  Not written, and it should not be.
> * **The lever is bit-width, not instruction selection.**  Q4_0 is 18 B per 32 weights = 0.5625 B/weight, so
>   the same stage would stream 1.19 GiB a round and take ~8 ms instead of ~16.5 - about 7 % of decode.  That is
>   the same Q4_0-for-latent-D item that has been blocked all along on there being **no KL/NLL quality gate in
>   this repo**.  The optimisation is understood and sized; it needs a quality gate, which is a decision for the
>   owner, not a kernel to write.
>
> **`gr_up_wide_kernel` on gfx1031: slower, and not bitwise equal.**  It had been unreachable - `gr_up_wide_on()`
> returned false under `__HIPCC__` *before* reading `STRATA_GR_UP_WIDE`, so the only AMD-shaped up variant in the
> tree could not be measured at all.  The env decision is now read first (the default stays off on HIP).  A/B,
> alternating pairs, everything else identical (`STRATA_QFUSE_HC=0` because the wide kernel is only selectable
> with a null `q8_mixed`, and persist off so `up` is stamped):
>
> | arm | `hc1 up` | round | decode |
> |---|---|---|---|
> | narrow (`gr_up_multi_kernel`) | 10.097 / 10.040 ms | 120.60 / 120.01 ms | 19.17 / 19.13 tok/s |
> | **wide (`gr_up_wide_kernel`)** | **12.872 / 12.967 ms** | **125.59 / 125.07 ms** | **16.60 / 16.65 tok/s** |
>
> +28.6 % on the up stage and **-13 % decode** - and the two arms' 32-token outputs **diverge from token 9**,
> so on this card the wide kernel is not the bitwise-equal retiling its CUDA comment claims.  The HIP disable is
> now a measured decision with a reason, instead of an unmeasured one; `STRATA_GR_UP_WIDE=1` still forces it.
>
> **What is left in decode, in order.**  20,480-row D at 155-167 GB/s is done (bit-width only).  The U pair
> (`u: gemv` 18.05 + `u: matvec` 14.62 = 32.7 ms) is the largest remaining item and it is **launch-bound, not
> bandwidth-bound**: ~10 rows a layer, 320 blocks a row, 48 layers = ~154k blocks a round at ~86 ns each = 13.2 ms,
> which is what §0.19 measured and why the prefill tile does not help it.  The fix named there - a decode-shaped
> tile, 8-16 rows, one block per label x column tile, ~150 blocks a layer - is still the right one and is still
> not written.

> ### 0.39 SHIPPED: the CPU rows' U on the grouped kernel, -24 % on that stage; and 0.38's arithmetic is CORRECTED
>
> **Correction first, because 0.38 was wrong.**  It used `lat_ELL_ = 2560`.  The real latent dimension is
> **ELL = 1792** - latent.cu says so in the matvec's own shared-memory line (`nb*36 = 56*36, ELL=1792 (latent)
> fixed`), and latent_dense.cpp:65-66 gives the authoritative formula:
>
> ```
> d_bytes = G * ELL * (H/32) * 34     u_bytes = G * H * (ELL/32) * 34
> G = 8, H (n_embd) = 2560, ELL = 1792, Q8_0 -> 34 B a 32-weight block
> both = 8 * 1792 * 80 * 34 = 38,993,920 B = 37.19 MiB a layer  (not 53.1),  1.74 GiB a round (not 2.38)
> ```
>
> So the corrected bandwidths, against a 170 GB/s ceiling:
>
> | stage | bytes/round | measured | implied | headroom |
> |---|---|---|---|---|
> | D | 1.74 GiB | 16.5 ms | **113 GB/s** | 34 % |
> | `u: gemv` (GPU rows) | 1.74 GiB | 18.05 ms | 103 GB/s | 39 % |
> | `u: matvec` (CPU rows) | 1.74 GiB | 14.62 ms | 128 GB/s | 25 % |
>
> **None of the three is at the ceiling.**  0.38's "the D stage is at the DRAM ceiling, only bit-width helps" is
> therefore withdrawn: all three latent-table stages run at 60-75 % of the achievable rate, and there is real
> per-kernel headroom that needs no quality gate and no new table format.  The bit-width argument still holds as
> *an* additional lever, but it is not the only one.
>
> **The shipped win.**  verify.cpp's own note records the difference and never took it: `latent_u_matvec` "walks
> every entry per block (81 GB/s effective)" while `latent_u_grouped` - the kernel the GPU rows already use -
> reaches 126 GB/s, with a knob (`STRATA_VERIFY_U_CPU_GROUPED`) already present.  A/B, alternating pairs, U_PROF
> open so slots 33/34/35 are visible, everything else identical:
>
> | arm | `u: matvec` (GDN + QSA) | window | decode |
> |---|---|---|---|
> | `latent_u_matvec` (was default) | 14.599 / 14.636 ms | 140.94 / 140.57 ms | 19.34 / 19.51 tok/s |
> | **`latent_u_grouped` (now default)** | **11.045 / 11.241 ms** | **137.07 / 136.52 ms** | **20.39 / 20.13 tok/s** |
>
> **-24 % on the CPU rows' U, -3.7 to -4.1 ms a window, 32-token output byte-identical in all four arms.**  Both
> kernels compute `U[labels[ids[r]]] . hq[r]` for the rows the pool owns, so this is a kernel swap, not an
> approximation.  `STRATA_VERIFY_U_CPU_GROUPED=0` restores the old path.
>
> Verified on the shipped default with no env overrides: chunk 512, prefill 231.75 / 231.39, decode 18.76 / 21.12,
> output byte-identical.  The decode end-to-end figure is noisy at `--max-new 32` (a 12 % spread between two
> identical arms), which is exactly why the stage-level slot is the evidence of record here and the tok/s is not.
> `run_512k_mtp.ps1 -Embed 1`: **exit 0 after 30 s, `output : 715 1822 1822 1822`** - unchanged.
>
> **What this leaves as the largest measured decode items, all with headroom:** `u: gemv` 18.05 ms at 103 GB/s,
> D 16.5 ms at 113 GB/s, `u: matvec` 11.0 ms at ~168 GB/s (now the fastest), dense GEMVs 18.25 ms.  The
> highest-value single change is now the same treatment for the *GPU rows'* U and the D stage: both are the same
> `latent_u_grouped` / grouped-gather shape that just moved the CPU rows from 81 to 126 GB/s.

> ### 0.40 `STRATA_LATENT_U_WARPS` is a non-lever, and long A/B batches degrade - two of four arms were invalid
>
> latent.cu's note calls the block width "the knob that matters" for decode's U and promises every width is
> bit-identical.  Measured, alternating, `STRATA_VERIFY_U_PROF` open:
>
> | arm | `u: gemv` | `u: matvec` | window | output |
> |---|---|---|---|---|
> | 8 (default) | 17.355 ms | 10.947 ms | 137.79 ms | correct |
> | 16 | 17.755 ms | 10.664 ms | 137.10 ms | correct |
> | 32 | 20.142 ms | 10.867 ms | - | **WRONG** (diverges at token 11) |
> | 8 (repeat) | 17.659 ms | 11.164 ms | 136.60 ms | **WRONG** (diverges at token 11) |
>
> Width is **not** a lever: 16 is 0.7 ms inside the noise and 32 is 2.8 ms worse on `u: gemv`, which is the
> note's own prediction - a few big blocks lose the latency hiding that the cold U rows need.  Default stays 8.
>
> **The fourth arm is the important one.**  It is the default configuration, byte-for-byte identical to the first
> arm, and it produced *different tokens* and took 4160 ms for 32 tokens against 1736 ms.  That is a degraded
> run, not measurement noise, and it reinterprets the "decode 18.8-21.1 tok/s" spread seen all session: some of
> that is not the machine being noisy, it is runs going wrong late in a long batch (VRAM pressure, thermal, or a
> driver state).  Two consequences:
>
> * the 32-token output gate is not optional on every arm - it is what caught this, twice;
> * a decode A/B finishing late in a batch is untrustworthy unless its output is checked.  Prefer the stage-level
>   slot with alternating pairs, which stayed consistent within a pair even across this (137.79 / 137.10 ms
>   against 136.60 ms in the degraded arm), over end-to-end tok/s.
'

> ### 0.41 The single-pass U merge is a REGRESSION (arithmetic from the engine's own slots), and the real defect is elsewhere
>
> The merge was the planned next step: the U table is read twice a layer, once for the GPU rows (`u: gemv`
> 17.4 ms) and once for the CPU rows (`u: matvec` 11.0 ms), 2 x 48.7 MB = 4.67 GB a round.  It is not worth
> doing, and the reason is the ordering in verify.cpp:
>
> ```
> current :  U_gpu(17.4) + pool wait(6.7) + copy(0.19) + U_cpu(11.0) = 35.3 ms
> merged  :  pool wait(24.1) + one pass(~17.4)                      = 41.5 ms   -> -6.4 ms
> ```
>
> The U on the GPU rows runs **before** the wait for the CPU pool's activations, so 17.4 ms of it is hidden
> behind the CPU's own work: `T_cpu = 17.4 + 6.7 = 24.1 ms`, and the 6.7 ms `u: pool wait` is exactly the
> remainder.  A single pass cannot start until BOTH activation sets exist, so it trades a hidden 17.4 ms for an
> exposed 17.4 ms and saves only the second read - a net loss.  **The double read is the price of using an
> otherwise idle GPU during the pool's wait, and it is cheaper than the alternative.**
>
> **What the code shows is the actual defect: the table is read `cap` times a call, not once.**  `latent_u_k`
> is one block per *(row, column tile)* - `r = blockIdx.x`, `h = blockIdx.y*warps + warp`, and each warp reads
> `wb = u_q8 + g*H*wrow + h*wrow`, a full ELL-long weight row.  So one call reads
>
> ```
> cap(10) rows x H(2560) columns x ELL(1792) weights x 1.0625 B = 48.7 MB a layer   (the table itself is 39 MB)
> ```
>
> i.e. the weights are re-read for every routed row because every row is a separate block.  A decode tile that
> puts the ~10 activation rows in registers and walks the columns once - one block per column tile, the shape
> §0.19 named - would read the table **once**, 4.87 MB a layer instead of 48.7: about **10x less traffic**, and
> it subsumes both calls because one pass serves every row.  That is the change worth making, and it is worth
> far more than the merge (which only halves a cost this would divide by ten).
>
> Why the existing tiles do not already do it: `latent_u_tile_k` is the *prefill* tile with kUTileRows = 64,
> which pads 10 real rows to 64 and stages 64 weight rows per block - §0.19 measured it neutral-to-worse here.
> The missing kernel is the same idea at 8-16 rows with ~150 blocks a layer.
>
> So the plan changes: do not merge, write the decode tile.  It is a new kernel and it needs its own build,
> alternating A/B and the 32-token gate, which is why it is not in this commit.

> ### 0.42 Correction to 0.41: the decode U tile is worth ~20 %, not 10x - rows do not share weight slices
>
> 0.41 said a decode tile "reads the table once, 4.87 MB a layer instead of 48.7: about 10x less traffic".  That
> is wrong, and the same source says so two lines from where I read the loop:
>
> ```
> const int g = (e >= 0 && e < E) ? labels[e] : 0;          // latent_u_k, line 78
> const uint8_t* w = u_q8 + (size_t) g * (size_t) H * (size_t) wrow;
> const uint8_t* wb = w + (size_t) h * (size_t) wrow;       // the weight row is g's, not a row-independent one
> ```
>
> **The weight slice is indexed by the routed row's GROUP**, so two rows can share a weight read only if they
> route to the same group.  With cap ~10 rows spread over G = 8 groups, at most ~2 rows share, so the real ceiling
> on a row-sharing tile is
>
>   distinct groups among 10 rows (<= 8) x H x ELL x 1.0625 = <= 39 MB  against the 48.7 MB read today
>
> i.e. **~20 % of the U stage, not 10x**: about 2 ms a round, not 27.  The prefill tile's neutral result is now
> fully explained - there is almost no reuse to capture at 10 rows over 8 groups, which is also why the row-walk
> variant measured 15 % slower in-engine rather than faster.
>
> So the U and D stages stand closed at Q8 on this geometry:
>
> | stage | irreducible traffic a round | measured | implied |
> |---|---|---|---|
> | D (20,480... 14,336 rows x 80 blocks x 34 B x 48) | 1.74 GiB | 16.5 ms | 113 GB/s |
> | U, GPU rows | 1.74 GiB | 17.4 ms | 103 GB/s |
> | U, CPU rows | 1.74 GiB | 11.0 ms | 168 GB/s |
>
> The 20-35 % of per-kernel headroom that 0.39 found is real but small in absolute terms (a few ms), and the
> ~25 % that row-sharing could give is unavailable by construction.  **The only large decode lever left on these
> tables is fewer bytes - Q4_0 - and that needs the KL/NLL quality gate this repo does not have.**  Everything
> else in decode is now either at the memory rate (U CPU rows), within a few ms of it (D, U GPU rows), or blocked
> on the pool (6.7 ms of wait, which 0.41 showed is the cheaper of the two schedules).
>
> **Therefore this stretch's decode thread is closed, and the objective's remaining headroom is prefill**: A stage
> 5,298 ms at 29.9 % of device time, `wait copy` 2,530 ms, PLE blocked 1,562 ms, against 231.8 tok/s and a ~430
> reference.  That is where the next work goes.

> ### 0.43 Prefill is balanced host/GPU at 17.68 s each - the biggest single item is CPU-side, not the A stage
>
> The ledger as it stands after the ring-80 change (4063 tokens, chunk 512, wall 17,688 ms, 63,715 blobs
> streamed = ~73.5 GB):
>
> | component | dev ms | %dev | host ms | %host |
> |---|---|---|---|---|
> | gemm gate/up (the A stage) | 5,388 | 30.5 % | 306 | 1.7 % |
> | gdn | 3,086 | 17.5 % | 32 | 0.2 % |
> | wait copy | 2,983 | 16.9 % | 7,547 | 42.7 % |
> | gemm down | 1,146 | 6.5 % | 6 | 0.0 % |
> | gather | 972 | 5.5 % | 51 | 0.3 % |
> | hc read: down+silu | 680 | 3.8 % | 131 | 0.7 % |
> | qsa attn | 632 | 3.6 % | 8 | 0.0 % |
> | router+shared | 560 | 3.2 % | 51 | 0.3 % |
> | qsa proj | 405 | 2.3 % | 9 | 0.0 % |
> | hc read: up | 317 | 1.8 % | 13 | 0.1 % |
> | gdn out proj | 331 | 1.9 % | 6 | 0.0 % |
> | gdn recurrence | 303 | 1.7 % | 1 | 0.0 % |
> | **host grouping** | 171 | 1.0 % | **9,335** | **52.8 %** |
> | ... | | | | |
> | **sum of marked phases** | **17,683** | 100 % | **17,675** | |
>
> **Device and host are balanced to within 0.05 %** (17,683 vs 17,675 ms), so prefill wall time is set by
> *both*, and a change that only shortens the GPU timeline cannot move it: the host becomes the limit as soon as
> the GPU drops below 17.68 s.  Every previous framing of this as "the A stage is 29.9 % of device time" was
> therefore incomplete - the A stage is 30.5 % of a 17.7 s GPU timeline that is itself only half the story.
>
> **The largest single item in prefill is CPU-side: `host grouping` 9,335 ms = 52.8 % of host.**  It is the
> expert-to-row grouping the prompt path does per layer-chunk, and it already runs off the timeline (dev 171 ms
> against host 9,335 ms), so it is pure host CPU work that the GPU waits on through `wait copy` (host 7,547 ms,
> device 2,983 ms).  Those two numbers are the same cost seen from both ends: the GPU idles 2.98 s waiting for
> copies the host has not issued yet because the host is grouping.
>
> So the prefill work for the next stretch, in order:
>
> 1. **host grouping, 9.3 s of CPU** - it is the thing `wait copy` waits for.  Anything that makes it cheaper
>    (fewer passes over the ids, batching the label compare, moving it onto the GPU) attacks both numbers at once.
> 2. **gemm gate/up, 5.4 s of GPU** - the A stage proper; needs a lower-VGPR variant, and the earlier attempts at
>    128 VGPR (234 spills, 2.4x slower) and `launch_bounds(256,2)` (ignored) both failed.
> 3. `wait copy` 2.98 s of device idle - falls out of (1) in proportion.
>
> Doing (2) alone is the trap this ledger exposes: it would take the GPU to ~12 s and leave the wall at the
> host's 17.7 s.

> ### 0.44 "host grouping 9,335 ms" is a mislabel: the grouping loop is 15 ms, and the host/GPU take turns
>
> `STRATA_PREFILL_HOSTPROF=1` splits the 840-line window between `kPfHostGroup` and `kPfGather` into seven host
> buckets (`prefill.cpp`, a local `PfHostProf`, printed once at exit).  Same run as 0.43's ledger:
>
> ```
> host window (ms): ids 1 | share+diag 9361 | group 15 | plan 5 | cpupool+stagerbuild 1 | copies 8132 | tail 0
> ```
>
> | bucket | host ms |
> |---|---|
> | ids download | 1 |
> | **CPU-share arm + stall** | **9,361** |
> | **the grouping loop itself** | **15** |
> | plan tables | 5 |
> | cpu pool + stager job build | 1 |
> | **copy issue (incl. `stager->wait`)** | **8,132** |
>
> **The grouping loop costs 15 ms out of 17.5 s.**  The ledger's `host grouping` bucket is 52.8 % of host because
> it is charged to the phase that was *open*, and what is open there is the CPU-share arm's setup and whatever
> the host blocks on - the code's own comment at that spot says it: "a stall here is the GPU (this layer's
> attention and router, or the previous layer's work), not the host".  Optimising "the grouping" was chasing a
> name; there is nothing there.
>
> What the split does show is the shape of the problem: **the host and the GPU take turns instead of
> overlapping.**  Per layer-chunk the host stalls 24 ms on the GPU (bucket 2), then spends 21 ms issuing copies
> while the GPU idles (`wait copy` device 2,985 ms, host 7,535 ms from the same window).  Two waits that
> alternate, each idle while the other works.  That is the same diagnosis prefill.cpp already records for the
> copy path ("the engine is idle ~85 % of the time, i.e. the copies are ring-starved, not link-bound") - and it
> is a *scheduling* defect, not a kernel one.
>
> The fix is to stop the alternation: the host should be building layer L+1's grouping and issuing L+1's copies
> while the GPU computes layer L, so neither side ever waits.  That is a change to the layer loop's structure
> (the prompt path already has the pieces - the staging ring, `Stager`, `warm_prev` - but drives them
> synchronously per layer).  It is worth up to the smaller of the two buckets, ~8 s, i.e. prefill 230 -> ~450
> tok/s if it can be made to overlap cleanly, which is the whole remaining gap to the ~430 reference.
>
> `STRATA_PREFILL_HOSTPROF=1` ships as the instrument (off by default, no cost).

> ### 0.45 `STRATA_IO_THREADS` is flat, so the copy bucket is not short of issuing threads
>
> The 8.1 s copy-issue bucket is `Stager::wait`, i.e. the host waiting for blobs to be pinned.  The AMD reference
> configuration uses `STRATA_IO_THREADS=32`, so it was the obvious lever.  Alternating pairs:
>
> | arm | prefill | share+diag | copies |
> |---|---|---|---|
> | 8 (default) | 229.47 / 229.66 tok/s | 9,365 / 9,349 ms | 8,108 / 8,117 ms |
> | 32 | 230.40 / 229.62 tok/s | 9,313 / 9,362 ms | 8,097 / 8,107 ms |
>
> Flat to 0.4 % on throughput and to 1 % on both buckets.  Issuing threads are not what the copy path is short of,
> which is consistent with 0.44: the cost is that the host is not issuing L+1's work while the GPU runs L, not
> that it cannot issue fast enough when it does.
>
> Together with 0.44 this closes the cheap levers around prefill's 17.7 s: ring depth is flat (240.35 / 240.46 /
> 237.12 at chunk 512), the grouping loop is 15 ms, and IO threads are flat.  **What is left is the alternation
> itself** - host stalled 9.4 s on the GPU, GPU idle 8.1 s on the host - and that is a restructuring of the layer
> loop into a producer/consumer over two layers, not a tuning knob.
>
> Reliability note, third instance: the third arm (`STRATA_IO_THREADS=8`, identical to the first) emitted wrong
> tokens from token 20.  Three separate A/Bs this stretch have now had a late arm corrupt, so any timing arm
> whose 32-token output is not checked is untrustworthy, and the corruption shows up as *wrong output*, not as a
> slow run - which is exactly how it would slip through a throughput-only harness.

> ### 0.46 The prefill model, corrected: the host does 263 ms of work and waits for the rest; the ceiling is 277
>
> 0.45's claim that the copy-issue path runs "at 9.2 GB/s against a 37 GB/s host ceiling" is **withdrawn**.  The
> engine already measures this and says otherwise, in its own timing footer:
>
> ```
> GPU timeline 17683 ms, wall 17688 ms, host staging 112 ms, streamed 63715 blobs (63715 by DMA), resident 358
> host: chunk setup (PLE rows, the expert stream plan) 33 ms, waiting for each chunk 72 ms,
>       after each chunk (the draft layer, progress) 41 ms, PLE 3 ms
> ```
>
> **The host's entire real work for the 17.7 s prefill is 112 ms of staging + 149 ms of everything else = 261 ms.**
> So the 9,361 ms and 8,132 ms buckets are both *stalls* - the host blocked, not slow.  There is no host-side
> inefficiency to fix, and the three turns spent on "host grouping" were chasing a bucket that contains no work.
>
> What the numbers do say, unambiguously:
>
> | quantity | value | source |
> |---|---|---|
> | expert bytes streamed | 73.5 GB (63,715 blobs) | timing footer |
> | all of it `by DMA`, from the pinned arena | 63,715 / 63,715 | timing footer |
> | exposed transfer (`wait copy`, device) | 2,945-2,985 ms | ledger |
> | implied PCIe rate | **24.6 GB/s** (73.5 GB / 2.99 s) | arithmetic, matches the measured ~28 GB/s |
> | GPU compute (timeline minus that idle) | 14.70 s | arithmetic |
>
> So `wait copy` is not host lag - it is **the expert transfer time itself, exposed**, and everything else is GPU
> compute.  The prefill ceiling follows directly:
>
> * hide the transfer completely -> 14.70 s -> **~277 tok/s** (from 230).  That is the whole available gain from
>   overlap, not the ~8 s / ~450 that 0.44/0.45 claimed.
> * reach the ~430 reference -> 9.4 s, which needs the GPU timeline 1.56x faster *as well*.  The A stage alone is
>   5,387 ms of it (30.5 %), so this is a kernel campaign, not a scheduling change.
>
> And the overlap is not free: staging a full layer ahead needs ~166 ring slots (193 MB a layer), against the 80
> that fit now.  That tradeoff is already measured - 96 / 80 / 112 ring slots are flat at 235.34 / 235.46 /
> 237.12 tok/s, and 192 / 256 **regress** to 187 / 189 because the ring comes out of the expert cache and more
> streaming follows.  A deeper ring cannot buy the overlap on this box's 12 GiB.
>
> Prefill is therefore at a structural limit: 14.70 s of GPU compute that must be reduced by kernel work, plus
> 2.99 s of transfer that must be overlapped, on a card that cannot afford the staging depth to overlap it.
> Remaining device items, in order: A stage 5,387 (30.5 %), gdn 3,086 (17.5 %), gemm down 1,146 (6.5 %), gather
> 972 (5.5 %), hc read down+silu 681 (3.9 %), qsa attn 633 (3.6 %).

> ### 0.47 The A stage's occupancy, measured: 256 VGPRs at the cap with 28 spills, and both occupancy controls are inert
>
> The number the last three turns kept quoting for `lat_q3k_fused.cu` ("193 VGPRs, 4 waves/SIMD") was wrong - it
> belongs to a different variant.  Compiled directly for gfx1031 with the resource pass:
>
> ```
> clang ... --offload-arch=gfx1031 -O3 -Rpass-analysis=kernel-resource-usage -c src/prefill/lat_q3k_fused.cu
>   src/prefill/lat_q3k_fused.cu:145:  TotalSGPRs: 41
>                                      VGPRs: 256
>                                      Occupancy [waves/SIMD]: 4
>                                      SGPRs Spill: 0
>                                      VGPRs Spill: 28
> ```
>
> **256 VGPRs is the architectural maximum**, and the kernel is *already spilling 28 registers at that ceiling*.
> It is not a kernel with room to trade registers for occupancy - it has exceeded the file and is paying local
> memory for the overflow.
>
> And neither occupancy control does anything on this card.  Both were changed, rebuilt and re-measured; the
> report is byte-identical in all three cases:
>
> | variant | VGPRs | spills | waves/SIMD |
> |---|---|---|---|
> | `__launch_bounds__(512, 1)` (shipped) | 256 | 28 | 4 |
> | `__launch_bounds__(512, 2)` | 256 | 28 | 4 |
> | `__launch_bounds__(512, 2)` + `__attribute__((amdgpu_waves_per_eu(8)))` | 256 | 28 | 4 |
>
> So the CUDA-style second parameter is ignored by AMD codegen exactly as the tree's earlier notes said, **and the
> AMD-specific `amdgpu_waves_per_eu` hint is ignored too** - the allocator's behaviour bounds both.  The only way
> left to change occupancy is to make the algorithm need fewer live registers, which is what the blunt
> `amdgpu_num_vgpr(128)` attempt tried and paid for: 234 spills and 2.4x slower (0.22).
>
> The A stage is therefore **closed to configuration**, and open only to algorithmic restructuring: 5,387 ms at
> ~1.7 TFLOP/s (14 % of fp32 peak), 256 VGPRs, 28 spills, 4 waves/SIMD, single kernel, 22.7 KB of Q3_K decode with
> a frozen-SHA reference and byte-exactness requirements.  That is a multi-day kernel rewrite, not a knob - and it
> is the largest single item in the engine (30.5 % of prefill device time), so it is the right thing for a
> dedicated session with the byte-exact harness (`lat_q3k_fused_test`, which exists) as the gate.
>
> Reverted: both edits were no-ops and are not in the tree.

> ### 0.48 SHIPPED, -11.7 % on the prefill A stage: unrolling the s-loop removed all 28 VGPR spills
>
> The 28 spills 0.47 measured are gone, and they were worth more than the occupancy argument they came with.
> One line - the inner `s` loop of the gate/up kernel, which was fully unrolled - plus removing values that were
> dead on the RDNA2 path:
>
> | variant | VGPRs | spills | waves/SIMD | `gemm gate/up` | prefill |
> |---|---|---|---|---|---|
> | `unroll 4` (was shipped) | 256 | **28** | 4 | 5,386 / 5,387 ms | 231.75 / 231.39 |
> | `unroll 2` | 201 | 0 | 4 | 4,770 / 4,783 | 238.91 / 238.77 |
> | **`unroll 1` (shipped)** | **153** | **0** | **6** | **4,756 / 4,758** | **238.62 / 238.64** |
>
> `gemm gate/up` **-11.7 %** (5,386 -> 4,756 ms), prefill **+2.9 %**, 32-token output byte-identical in every
> arm, and `run_512k_mtp.ps1 -Embed 1` exits 0 with `output : 715 1822 1822 1822` - unchanged.
>
> What made it work, and what did not:
>
> * **Occupancy attributes are inert** (0.47): `__launch_bounds__(512,2)` and `amdgpu_waves_per_eu(8)` both left
>   the allocation at 256 VGPRs / 28 spills.  Nothing that *asks* for fewer registers works.
> * **Removing dead values did nothing on its own**: on the RDNA2 path `a0`, `a1` and `bq[]` exist only for the
>   CUDA `mma` asm and are `(void)`-ignored, yet guarding them out left the report at 256 / 28 - the compiler had
>   already eliminated them.  They are still guarded out (correct, and free), just not the cause.
> * **`unroll N` is the real register control on this toolchain.**  4 -> 2 drops 55 VGPRs and all 28 spills;
>   2 -> 1 drops another 48 and takes occupancy from 4 to 6 waves/SIMD.  The spills were the fully-unrolled
>   nest's address arithmetic, not the math: the `acc[4][2][4]` accumulators and the dp4a sequence are untouched,
>   and the `s` order is preserved, which is why the output stays bit-identical.
> * `unroll 1` and `unroll 2` are equal in throughput (4,757 vs 4,776 ms average) despite 153 vs 201 VGPRs, so
>   the extra occupancy buys nothing *here* - but it is the lower-pressure choice at equal speed and it is what
>   ships.
>
> A note on the last stretch's ceiling: 0.46 put prefill's recoverable time at ~2.99 s of exposed transfer and
> said the compute needed a 1.56x kernel campaign.  This is the first instalment of the latter - 0.63 s of the
> A stage, found in one line rather than a rewrite - and it confirms that the A stage has room.  The remaining
> gap to 430 is still mostly the A stage's 4.76 s plus the 2.98 s of transfer.

> ### 0.49 Hardware queue priority: the mechanism is real, it is mostly disabled here, and it is a non-lever
>
> "Send the work to the high-priority compute cores" is not a thing on this part - a 6700 XT has 40 identical
> CUs with symmetric scheduling, no fast/slow or high/low tier.  What *is* real is **AMD hardware-queue (HQD)
> priority**, exposed as `hipStreamCreateWithPriority`, which biases *dispatch between queues* when several are
> resident.  Audited:
>
> * `verify.cpp` `graph_prio_on()` is `#if defined(STRATA_USE_HIP) ... return false;` - the decode graph streams
>   get **no** priority on any HIP build.
> * `verify.cpp:1247` the copy stream takes `hi` only behind `cp`.
> * `mtp.cpp:478-495` the MTP draft stream's priority is `STRATA_MTP_HIP_PRIO`, **opt-in on gfx1031** (default
>   only on gfx1200/gfx1201), and only when `force_on_`.
> * `dflash2.cpp:200`, `generate.cpp:2925` - other opt-in high-priority streams.
>
> With `STRATA_MTP_SIDE_STREAM=1` the draft stream really does run concurrently with the main decode stream, so
> that is the one place priority can bite.  A/B, alternating pairs, 32-token output identical in every arm:
>
> | arm | decode | prefill |
> |---|---|---|
> | `STRATA_MTP_HIP_PRIO=0` (default here) | 20.60 / 20.91 tok/s | 236.16 / 240.64 |
> | `STRATA_MTP_HIP_PRIO=1` | 21.03 / 19.27 tok/s | 240.52 / 240.46 |
>
> Overlapping ranges, variance larger than the difference: **a non-lever on this card.**  Keep the default.
>
> It also cannot help prefill's exposed 2.99 s, for a hardware reason: that time is the **copy engine's DMA**, a
> separate block from the CUs, so queue priority over compute waves does not affect it.  Priority only reorders
> compute against compute.

**Audit date:** 2026-11-16
**Revision audited:** `b826102` (`feat(kv): gate turbo3/rot4 on the Kraken-x CPU oracle; measure the prefetch question`), tree dirty in 14 paths (all logs / harness scripts, no source).
**Target:** AMD Radeon RX 6700 XT, `gfx1031`, 20 CUs, 12 GiB, Windows 11 / WDDM, ROCm 10.2 userland (`C:\Rocm10`).
**Engine:** 120B q3k latent MoE pack, `--kv turbo3`, expert cache on, MTP drafts.
**Method:** `Goli-mat-de` (find work that is missing, superficial, disconnected or falsely reported complete) plus `references/audit-matrix.md`. Read-only on the tree; every claim below carries the evidence that produced it. Nothing here was executed from a plan or inferred from a name.

---

## 1. Verdict

The port is **real, complete and correct where it claims to be** — the KV codecs are oracle-green, the GPU fence is bounded, 512K context and 512K+MTP both start, and the q3k latent A stage is implemented twice (a per-expert fallback and a fused in-place kernel). The code is not stubbed, not faked, and not CUDA-shaped dead weight on HIP: the `__dp4a` mapping genuinely compiles to `v_dot4c_i32_i8`.

The performance problem is **not** a missing kernel and **not** a wrong instruction. It is **dispatch geometry**:

1. The expert A stage — **61 % of prefill device time** — runs **one expert per launch**, 5 blocks of 16 warps on a 20-CU card, and pays ~200 µs of WDDM submission per expert on top of a ~165 µs kernel.
2. Batching exists in the source, is documented as measured, and is **triply gated**: it needs `STRATA_LATENT_BATCH=1` **and** a chunk ≥ `stream_all_min()` (1024) **and** a path that is off by default. At the shipped auto chunk (512) it is unreachable by construction.
3. The in-place fused A stage (`lat_q3k_fused`, `gather_native` skipped) — the other half of the fix, worth the measured **11.3 s of `gather` (19.5 % of dev)** — is **opt-in**, so the default run still pays the gather.
4. Both A-stage kernels sit at the architectural VGPR ceiling (`vgpr_count 256`) **and spill** (28 and 53 VGPRs) — the compiler is over budget, not short of instructions.

Nothing in the audit found a *wrong answer* in the shipped path. The defects are throughput and verification coverage.

**Estimated recoverable prefill time on the measured 4K baseline (58.06 s wall / 49.08 s device):**

| change | basis | recover | status |
|---|---|---|---|
| batch the A stage (4 experts/launch) | measured −63 % gate/up dev in the earlier rejected arm; **now measured: −17.2 % prefill wall** | 8–14 s | **LANDED and measured: 10.8 s** |
| fused in-place A stage on by default (no gather) | `gather` = 11.31 s measured | 8–11 s | **LANDED** (bit-identical tokens; its own seconds are inside the same A/B) |
| kernel register/instruction diet | 256 VGPR + 28/53 spills, `blocks/cu = 1`, 36 divides/tile | 1–3 s | proposed (T4) |
| **total** | | **≈ 17–28 s of 58 s ⇒ 1.4–1.9×** | first 10.8 s measured |

---

## 2. Scope and coverage ledger

| area | files (count) | status | notes |
|---|---|---|---|
| `src/prefill/lat_q3k_fused.cu` | 1 (384 ln) | **VALIDATED** | read end to end, ISA extracted |
| `src/prefill/lat_q3k_down.cu` | 1 (314 ln) | **VALIDATED** | read end to end, ISA extracted |
| `src/prefill/lat_q3k_gu.cu`, `lat_q3k_swiglu.cu` | 2 | **INSPECTED** | not on the live path (the fused kernel supersedes them) |
| `src/prefill/prefill.cpp` | 1 (8.8 k ln) | **INSPECTED** | dispatch, PfTimer, both A-stage walks, PE timer, buffer sizing read; not line-by-line over all 8.8 k |
| `src/prefill/moe_fused_q3k.cu` | 1 (89 KB) | **INSPECTED** | `lat_dev()` HIP guard read; the CUDA-only `mma.sync` body confirmed unreachable on HIP |
| `src/prefill/moe_mmq.cu` (gather) | 1 (38 KB) | **INSPECTED** | `gather_native*` entry points and call sites |
| `src/prefill/kernels.cu`, `gemm.cu`, `moe_fused.cu`, `moe_fused_iq.cu`, `moe_ws_*` | 6 | **NOT STARTED** | latent path not on them; next batch |
| `src/kernels/**` (186 files, 4.2 MB) | 186 | **NOT STARTED** | decode path; audited only through the repo's own stage table |
| `src/core/**` (63 files, 1.8 MB) | 63 | **NOT STARTED** | `expert_source.cpp` touched only via the fence findings |
| `src/dsv41`, `src/qwen35`, `src/ngram`, `src/spec`, `src/plan`, `src/platform`, `src/artifact`, `src/program` | 55 | **NOT STARTED** | not on the measured path |
| `include/strata/**` (210 files, 1.4 MB) | 210 | **INSPECTED** | `lat_q3k_*.hpp`, `q8_1_finite.hpp` contracts read |
| `CMakeLists.txt` | 1 | **VALIDATED** | 258 `add_test`; the A-stage tests are **not** among them |
| `docs/perf-log-2026-11-16.md` | 1 (715 ln) | **VALIDATED** | every numeric claim cross-checked against the logs on disk |
| `edge0-strata/docs/OPTIMIZATION-TRACING.md`, `PERFORMANCE.md` | 2 | **NOT STARTED** | named by the standing rules; next batch |
| `tools/`, `serve/`, `tests/` | — | **EXCLUDED** | no runtime effect on the measured path |

**Exclusion that matters:** the CUDA-only `latent_kernel_q3k` (`moe_fused_q3k.cu`, `mma.sync m16n8k16`, sm_80) is *deliberately* dead on HIP, and the audit verified it fails closed rather than silently: `lat_dev()` returns `ok == false` under `__HIPCC__` (`moe_fused_q3k.cu:1404-1433`), `fused::latent_fused_available()` therefore declines, and `experts_latent_q3k` would `exit(1)` if it were ever reached (`:1558-1561`). **No silent wrong answers from that boundary.**

---

## 3. Hardware model, measured

`hipInfo` on this box, not quoted from a whitepaper:

| | |
|---|---|
| name / arch | AMD Radeon RX 6700 XT / **gfx1031** |
| `multiProcessorCount` | **20** |
| `warpSize` | 32 → 64 waves/CU at 100 % occupancy |
| `maxThreadsPerMultiProcessor` | 2048 |
| `maxSharedMemoryPerMultiProcessor` | 128.00 KB |
| `maxAvailableVgprsPerThread` | **256** |
| `clockRate` | 2424 MHz |
| VRAM | 11.98 GB |

Measured ceilings already in the repo, re-used here as the reference points: PCIe H2D 21.65 GB/s (serialised 10.81, pipelined 18.35), D2H 23.20, host DDR4 31.7, graph node (1-block captured kernel) 0.84 µs, WDDM `cudaEventRecord` ≈ 80 µs / cross-stream wait ≈ 45 µs.

RDNA2 has **no matrix cores**. Any `mma.sync` path is CUDA-only by definition, which is why the port has `v_dot4c_i32_i8` paths at all. That substitution is *correct* and *exact*: `__dp4a` accumulates int32, so the dot products are bit-identical to the tensor-core reference.

---

## 4. Measured baseline (the number every change is judged against)

`c1024_base.log`, 4,063-token prompt, chunk 1024, `STRATA_PREFILL_TIMING=1`, 48 layers:

| component | marks | dev ms | % dev | host ms |
|---|---|---|---|---|
| **gemm gate/up** | 34,642 | **12,734** | **21.9 %** | 3,211 |
| **gather** | 192 | **11,308** | **19.5 %** | 15 |
| **gemm down** | 192 | **11,279** | **19.4 %** | 2 |
| wait copy | 33,545 | 6,375 | 11.0 % | 27,195 |
| gdn (attention + recurrence) | 144 | 3,366 | 5.8 % | 19 |
| qsa attn / proj | 48 / 96 | 1,101 / 1,105 | 3.8 % | 12 |
| combine | 192 | 761 | 1.3 % | 2 |
| router+shared | 240 | 536 | 0.9 % | 19 |
| host grouping | 240 | 151 | 0.3 % | **26,941** |
| ple / embed / hc / qsa indexer+select | — | < 300 total | < 1 % | < 350 |
| **wall** | | **58,058 ms** | | |
| **dev sum** | | **49,077 ms** (84.5 % of wall) | | |

Reproduce: `bench4k.bat`, or `strata` with `STRATA_PREFILL_TIMING=1`.

Two independent readings of the same run agree, which is why the numbers are trusted:

* `tr_512.log` (chunk 512, NPROF on): `latfq3kprof: calls=64,080 ms=10,742 entries=1,948,496 us/entry=5.51` — i.e. the A-stage kernels themselves are **10.7 s** of the 65 s perturbed wall, while the *component table of the same run* charges `gemm gate/up 21,658 + gemm down 12,691 + gather 11,095 = 45.4 s`. **The gap is per-launch overhead, not arithmetic.**
* `tr_2048.log`: `latfq3kprof ... us/entry=3.6–3.7` at ~90 entries/call, against `5.51` at 30 entries/call (chunk 512) — the per-entry cost falls as the group grows, which is the occupancy signature of a launch that is too small.

Decode, from the repo's own `STRATA_VERIFY_PROFILE` table (4K, served window T=4, 95.99 ms/round): `copy+combine` 40.22 ms (41.1 %), `hc1 up` (the latent D stage) 24.27 ms (24.8 %), everything else 34.2 % with no single item above 4 %.

### 4.2 Why TTFT is what it is: the ledger after the three fixes (`pf_why.log`, 46,412 ms)

One `(layer, chunk)` pass happens **384 times** (48 layers × 8 chunks of 512). Every per-layer cost below is therefore paid 384 times, and that multiplier is the first thing to understand about TTFT.

| stage | dev ms | % | work it moves | effective rate | measured cause |
|---|---|---|---|---|---|
| **U projection** (`gemm down`) | **12,815** | **27.6 %** | 384 × 33.6 GFLOP = 1.61 TFLOP; 805 GB read | 128 GFLOP/s, 64 GB/s | F15 — one warp per scalar output, 32→1 reduce per value; the register-reuse variant was measured **slower** |
| **A stage** (`gemm gate/up`) | **9,585** | **20.7 %** | 1.95 M (token,expert) × 4.92 MFLOP = 9.6 TFLOP | 1.0 TFLOP/s | F1 (fixed: batch 4, 4,864 launches/layer-chunk) — still 25 launches per pass at `blocks/cu = 1` |
| **D projection** (`gather`) | **5,442** | **11.7 %** | 384 × 25.5 GFLOP = 1.23 TFLOP | 237 GFLOP/s | F14 (fixed: `KQ_SPLIT` was selecting the per-column kernel; −51 % of this row) |
| gdn (attention + recurrence, 36 layers) | 4,876 | 10.5 % | | | |
| **wait copy** | **4,243** | **9.1 %** | the staging ring | | host 18,359 ms in the same phase |
| hc (5 phases) | 3,525 | 7.6 % | | | |
| qsa proj + attn (12 layers) | 2,386 | 5.1 % | | | |
| router+shared | 1,040 | 2.2 % | | | |
| combine / ple / gdn extras / embed | ~2,500 | 5.4 % | | | |
| host grouping | 228 (dev) | 0.5 % | | | **13,071 host** |
| **wall** | **46,412** | | **≈ 12.5 TFLOP** | **269 GFLOP/s end to end** | ~1 % of this card |

**Read the bottom row.** The prompt is 12.5 TFLOP of arithmetic on a card that does ~26 TFLOP/s fp16 and several times that in int8. TTFT is 46 s because the three projections that hold 60 % of that arithmetic execute at **128 GFLOP/s, 1.0 TFLOP/s and 237 GFLOP/s** — each for a *different, measured* reason, and none of them for lack of a kernel: each had a faster variant in the tree (batched launches, the decode-once Q8_0 kernel, the weight-reuse U kernel) that was either gated off, unreachable, or — for U — measurably worse.

**Where the remaining time is not:** the expert stream. 63.6 k blobs = 74 GB cross PCIe per prompt, and `wait copy` is 9.1 % — the link is not the wall. Neither is the expert cache (2.8 % residency) for prefill. The wall is three dense projections plus 36 attention layers.

### 4.1 The intermediate ranking, before the D-stage fix (kept for the before/after)

The trace above is the pre-fix path. Re-running it with the fused in-place A stage and batched launches moves the ranking - and not to the item the earlier audit ranked first:

| component | marks | dev ms | % dev | what it actually is |
|---|---|---|---|---|
| **`gemm down`** | 384 | **12,464** | **23.9 %** | the latent **U** projection (`latent_u_grouped`) + the q8_1 of H |
| **`gather`** | 384 | **11,076** | **21.2 %** | the latent **D** projection (`iq_mmvq`), *not* an expert gather |
| **`gemm gate/up`** | 16,178 | **9,910** | **19.0 %** | the expert A stage, now fused and batched |
| wait copy | 16,178 | 4,310 | 8.3 % | staging ring |
| gdn | 288 | 4,893 | 9.4 % | attention block + recurrence (36 layers) |
| qsa proj / attn | 192 / 96 | 1,481 / 865 | 4.5 % | 12 layers |
| hc write + hc read (5 phases) | 768 | 3,580 | 6.9 % | hyper-connection |
| router+shared | 768 | 1,135 | 2.2 % | |
| host grouping | 768 | 194 (dev) | 0.4 % | **25,863 host** |
| combine / ple / gdn extras | — | ~1,900 | 3.6 % | |
| **wall 52,163 ms** | | | | |

**Read it as: the two dense latent projections are 45.1 % of prefill, and both are matvec/GEMV kernels doing a matrix problem** — F14 and F15. The expert A stage that this audit ranked first is now 19 %, and `wait copy` is 8.3 %. The earlier baseline put `gather`+`gemm down` at 39 % too, but attributed them to expert traffic; the trace labels say otherwise (384 marks = per layer per chunk, not per expert).

---

## 5. Dispatch map of the measured path

```
strata.exe generate
└─ Prefill::run_impl (prefill.cpp:3xxx)
   └─ per chunk (auto = 512 → STAGE ring; ≥1024 → stream_all ring)
      └─ per layer (48)
         ├─ router → ids on device → up_ring.d2h_sync(ids_host)   ← kPfHostGroup opens (4651/5126)
         ├─ [latent A stage]  m.latent branch at 5462
         │   ├─ lat_fused_q3k  (5616: opt-in STRATA_LAT_FUSED_Q3K=1, q3k pack only)
         │   │    └─ arm_launch → lat_q3k_fused()  (5963)
         │   │         ├─ lat_q3k_fused_gu_kernel <<<dim3(5, n_groups, 8), 512>>>
         │   │         └─ lat_q3k_down_kernel      <<<dim3(5, n_groups, 8), 512>>>
         │   ├─ lat_mmq        (STRATA_LATENT_MMQ=1) → gather_native + Product ×3
         │   └─ native_expert_grouped (default) → gather per layer + grouped kernel
         ├─ quantize_q8_1_rows(yl) → latent_u_grouped / fused::latent_dense_u
         ├─ attention (gdn 36 layers / qsa 12 layers)
         └─ combine
```

**Which walk runs is decided by the chunk size** (`stream_all = m.ring > STAGE && T >= stream_all_min()`, `prefill.cpp:3565`):

| chunk | `stream_all` | A-stage walk | launches per layer |
|---|---|---|---|
| 512 (shipped auto) | false | staged, `lookahead = STAGE-1 = 7`, **one launch per expert** (`:6294-6345`) | ≈166 |
| 1024 | true | `lat_batched` — but **only with `STRATA_LATENT_BATCH=1`** (`:6190-6193`) | ≈166 without it |
| 2048 | true | same | ≈166 without it |

That single line (`:6190`) is why the repo's own measured batching gain (64,126 → 8,279 launches) is not the default behaviour.

---

## 6. What the ISA actually says

Extracted from the built objects (`build\CMakeFiles\strata_mmq.dir\src\prefill\lat_q3k_*.cu.obj` → `.hip_fatbin` → `clang-offload-bundler --unbundle --targets=hip-amdgcn-amd-amdhsa--gfx1031` → `llvm-readelf -n` / `llvm-objdump -d`, ROCm 10.2 LLVM 23.0.0git):

| kernel | VGPRs | SGPRs | VGPR spills | private seg |
|---|---|---|---|---|
| `lat_q3k_fused_gu_kernel` | **256 (cap)** | 41 | **28** | 196 B |
| `lat_q3k_down_kernel` | **256 (cap)** | 46 | **53** | 120 B |

Opcode counts (static, whole kernel body):

| gu kernel | count | down kernel | count |
|---|---|---|---|
| `v_dot4c_i32_i8` | **512** | `v_dot4c_i32_i8` | **512** |
| `v_fmac_f32_e32` | 200 | `s_waitcnt` | 201 |
| `v_mul_f32_e32` | 196 | `v_mov_b32_e32` | 177 |
| `v_mov_b32_e32` | 177 | `v_cvt_f32_i32_e32` | 128 |
| `s_waitcnt` | 150 | `v_fmac_f32_e32` | 128 |
| `v_perМ_b32` (`v_perm_b32`) | 120 | `v_add_co_u32` | 97 |
| `v_div_scale_f32` | **72** | `v_add_co_ci_u32_e64` | 97 |
| `v_rcp_f32_e32` | **36** | `s_or_b32` | 92 |
| `v_div_fixup_f32` | **36** | `s_and_saveexec_b32` | 84 |
| `flat_load_dwordx4` | 72 | `v_mul_lo_u32` | **64** |
| `buffer_load_dword` | 76 | `s_cbranch_execz` | 66 |
| `s_barrier` | 4 | `v_lshlrev_b64` | 64 |

Conclusions that are *facts*, not hypotheses:

* **`__dp4a` is real here.** 512 `v_dot4c_i32_i8` = 4 s-blocks × 4 i × 2 n × 4 kw × 4 dp4a. "The compiler didn't emit dot4" is **not** a defect. ✔
* **The kernels are at the VGPR ceiling and spilling.** 28/53 spilled VGPRs × 512 threads = 57 KB / 108 KB of scratch traffic per block per pass, on top of 196/120 B of `private_segment_fixed_size`. Spills at this volume are load/store instructions in the hot loop; `s_waitcnt 150/201` is consistent with waiting on them.
* **36 IEEE divides per gate/up tile.** `v_div_scale_f32 72 + v_rcp_f32 36 + v_div_fixup_f32 36` = 36 correctly-rounded f32 divides, from `__fmul_rn(__fdiv_rn(gt, 1 + __expf(-gt)), up)` and `__fdiv_rn(am, 127)`. **These must not be replaced by fast-math**: `lat_q3k_fused`'s contract is byte-for-byte equality with the reference quantizer (`lat_q3k_fused.cu:8-13`).
* **The down epilogue does 64-bit index math per store.** `64 v_mul_lo_u32 + 97 v_add_co_u32 + 97 v_add_co_ci_u32` from `out[(size_t)(ent_dst[...]) * ld + WROWS*fb + ...]` (`lat_q3k_down.cu:277-278`) — 12 stores per thread × 64-bit mul-add.

---

## 7. Findings

Priority: P0 = correctness/security/data, P1 = core path missing or broken, P2 = material integration/verification gap, P3 = cleanup. Confidence is stated separately. Every finding names its evidence.

### F1 — P1 / CONFIRMED — the A stage runs one expert per launch; the GPU is ~10 % occupied

**Where:** `prefill.cpp:6294-6345` (`arm_launch(la_for(false), 1)` per expert in the staged walk), `lat_q3k_fused.cu:374-378` (launcher), `lat_q3k_fused.cu:167-174` (work item).

**Expected:** a launch that covers enough (expert, tile, feature) work items to fill 20 CUs.

**Observed:** `arm_launch` is called with `nb = 1`. The launcher computes `gy = min(n_groups, 512) = 1`, and because `gy * kItems = 5 < sms = 20` it also sets `gz = 8`, giving `<<<dim3(5, 1, 8), 512>>>` = 40 blocks. A group of ~20–30 entries has one 64-row tile, so `blockIdx.z ≥ 1` computes `row0 = e0 + 64·z ≥ e1` and exits immediately: **35 of 40 blocks do nothing**. Of the 5 that work, `nb0 = 16·(warp>>2)` makes warps with `nb0 = 32, 48` idle for `nrows ≲ 20`, and `on1 = nb0+8 < nrows` drops half of `nb0 = 16`.

**Evidence:**
* `latfq3kprof: calls=64080 ... last: nb=1 ent=1|5|230|...` (every printout in `tr_2048.log`, `fused_probe.log`, `post_batch*.log`).
* `gemm gate/up` = 34,642 marks for one 4,063-token prompt (`c1024_base.log`) and 64,126 at chunk 512 (`tr_512.log`) — one mark per expert.
* Per-launch cost 368 µs against a 165 µs kernel (`12,734 ms / 34,642` vs the A-stage probe's `5.51 µs/entry × 30 entries`).
* `us/entry` 5.51 at 30 entries/call → 3.69 at 90 entries/call (`tr_2048.log`): the same arithmetic gets 1.5× cheaper purely by making the group bigger.
* **The occupancy is now a measurement, not an inference** (`STRATA_LAT_Q3K_INFO=1`, this session, rebuilt binary):

```
latq3kinfo gu:   grid=(5,1,8) block=512 smem=51456 regs=256 scratch=224 dynmax=51456 blocks/cu=1 waves/cu=16 of 64 threads/cu=512 of 2048 cus=20
latq3kinfo down: grid=(5,1,8) block=512 smem=49408 regs=256 scratch=128 dynmax=49408 blocks/cu=1 waves/cu=16 of 64 threads/cu=512 of 2048 cus=20
```

  One 512-thread block per CU (**16 of 64 waves, 25 %**), at the 256-VGPR ceiling with 224 B / 128 B of scratch live. LDS is *not* the limit: 51,456 B (gu) would allow two blocks in the 128 KiB LDS, so **registers are the binding constraint** — confirmed by the driver, not by arithmetic. And of the 40 blocks launched, 35 take `row0 = e0 + 64·z ≥ e1` and exit: **5 working blocks on 20 CUs**, each at 25 % wave occupancy ⇒ ≈6 % of the card doing work during the gate/up launch.

**Impact:** the expert GEMM runs at **395 GFLOP/s**, 1.5 % of the card's FP16 peak, and it is 41 % of prefill device time. Additionally ~200 µs × 34,642 = **6.9 s** of pure submission overhead, which is most of the 11.0 % `wait copy` column.

**Cause:** the walk's unit of work is the expert, not the batch; the kernel's `n_groups`/`counts` interface that would allow batching is already implemented and unused.

**Fix:** batch. Needs F2's ring geometry fix. Already measured, on a broken variant: `gate/up` marks 64,126 → 8,279, dev 21,658 → 8,088, wall 65.9 → 51.9 s (`post_batch2.log`, wrong tokens because of the ring hazard).

**Acceptance:** with batching on, `latfq3kprof` shows `nb > 1` with `us/entry ≤ 4.0` at chunk 512; `gemm gate/up` dev ms falls ≥ 40 %; tokens bit-identical to the unbatched run with matched flags; `lat_q3k_*` tests pass.

---

### F2 — P1 / CONFIRMED — batching is triple-gated, and the reachable variant is unsound by construction

**Where:** `prefill.cpp:6190-6193` (`lat_batched`), `:6044` (`lookahead = stage_n - 1`), `:6365` (staging loop), `:6356-6361` (the comment that documents the failure).

**Expected:** the measured batching gain is the default, guarded by a correctness proof.

**Observed:**
1. `lat_batched` requires `STRATA_LATENT_BATCH=1` when `lat_mmq` is false (the fused-q3k case) — off by default.
2. That branch is only reached under `stream_all`, i.e. **chunk ≥ 1024**; the shipped auto chunk is 512 (`prompt chunk auto: 512`), so the default run cannot reach it at all.
3. In the reachable staged walk (`:6347-6398`) *resident* experts batch (≤64/launch) and *staged* experts launch alone behind their own event. Raising `STRATA_LAT_STAGED_BATCH` holds slots past `lookahead = STAGE - 1 = 7`, `stage_one` (`:5433-5461`) then reuses a slot whose blob a queued launch still reads, and the tokens change.

**Evidence:** `post_batch2.log` — 7.7× fewer launches, -63 % dev ms, **wrong tokens from the first one**; the ring arithmetic is in the file's own comment (`:6356-6361`) and in the perf log's §"The staged-launch batching experiment".

**Impact:** the largest available single win is implemented, documented, and unreachable.

**Cause:** `lookahead` encodes "keep every slot in flight" rather than "keep the *un-launched* batch's slots free".

**Fix (contained):** `lookahead = stage_n - batch` and cap the batch at `batch`; then the maximum staged-index distance between the oldest un-launched batch entry and the newest staged entry is `lookahead + batch - 1 = stage_n - 1 < stage_n`, so `stage_one` can never reach a slot the pending batch still references. Keep `STAGE = 8` with `batch = 4` (`lookahead = 4`), or raise `STAGE` to keep more prefetch depth. Batch only when `lat_fused_q3k` is on, because that is the only A-stage call that takes per-group **counts** (`lat_q3k_fused.hpp`'s `counts` argument); `native_expert_grouped` needs contiguous entry ranges and must keep its own flush rule (`:6282`).

**Acceptance:** `nb == batch` in the profile; tokens bit-identical; `lat_q3k_*` tests green; `wait copy` must not rise more than the batching recovers.

---

### F3 — P1 / CONFIRMED — the in-place fused A stage is off by default, so `gather` costs 19.5 % of device time

**Where:** `prefill.cpp:1509-1515` (`lat_fused_q3k_on()` reads `STRATA_LAT_FUSED_Q3K`, default off); the fused branch itself at `:5947-5985` returns **before** `mmq::gather_native` (`:5991-6013`).

**Expected:** the documented "the A stage already reads in place" behaviour applies to the shipped configuration.

**Observed:** `c1024_base.log` — the trace the perf log calls the baseline — contains **no `latfusedq3k:` line at all**, and its component table charges `gather` 11,308 ms / 19.5 %. The runs that do contain the line (`fused_probe.log`, `post_batch*.log`, `ctx_mtp_c_pf1024.log`) have no `gather` phase on the latent path.

**Impact:** 11.3 s of 58.1 s. Also removes a per-layer `gather_native` pair (gate/up + down matrices, 704 KB + 288 KB per expert) and the 64-KB `copy16_kernel` traffic that goes with it.

**Cause:** the fused path was staged as M3/M4 behind an env var and never promoted.

**Fix:** default `lat_fused_q3k_on()` to **on** for the pack it is guarded to (`gu_type == 11 && d_type == 20 && lat_I == 640`), with `STRATA_LAT_FUSED_Q3K=0` as the escape hatch. The guard at `:5616-5619` already restricts it to that geometry, and the kernel is byte-for-byte the reference quantizer by construction.

**Acceptance:** default run's trace has no `gather` phase and a `latfusedq3k:` line; tokens bit-identical to the `STRATA_LAT_FUSED_Q3K=0` run with matched instrumentation; `lat_q3k_fused_test` / `lat_q3k_down_test` / `lat_q3k_gu_test` pass against their references.

---

### F4 — P2 / CONFIRMED — both A-stage kernels are over the register budget and spill

**Where:** `lat_q3k_fused.cu:189-239` (`GuStaged wc[2]`, `uint32_t aq[8]`, `float acc[4][2][4]`), `lat_q3k_down.cu:98-100` (`wq[4]`, `aq[8]`), `:189-195` (`acc[4][2][4]`).

**Expected:** a kernel that fits 256 VGPRs at 512 threads/block without scratch traffic.

**Observed:** `vgpr_count 256` (the architectural maximum) with `vgpr_spill_count 28` (gu) and `53` (down), `private_segment_fixed_size 196` / `120`.

**Impact:** spill loads/stores inside the 20-chunk (gu) and 10-chunk (down) loops, plus `s_waitcnt` pressure (150/201). Secondary to F1 (a small launch cannot recover throughput whatever the register count), but it is the reason `us/entry` stays at 5.5 rather than ~2.

**Cause, measured:** `wc[2]` is a *software pipeline for the CUDA mma operands*. On HIP every one of those registers is discarded — `(void) a0; (void) a1; (void) bq[n];` at `lat_q3k_fused.cu:275-277` and `lat_q3k_down.cu:230-232` — because the HIP path re-reads the same bytes from the staged smem tile (`:278-290`, `:233-245`). So **20 VGPRs/thread of decoded weight sit in registers for one stmt block and are never read**, across a `__syncthreads()`.

**Fix:** on `__HIPCC__`, decode straight into the smem buffer after the second barrier (`wst[(kb+1)&1]`, which is provably free at that point) instead of into `wc[2]`; keep the register pipeline only under the CUDA `mma` branch. Second step: hoist the down epilogue's 64-bit multiply out of the store loop (`ent_dst[row0+r] * ld` recomputed 12×/thread; F6).

**Acceptance:** `vgpr_count ≤ 200` and `vgpr_spill_count == 0` on both kernels; `us/entry` at equal `nb` falls ≥ 10 %; outputs bit-identical (these are pure register-movement changes).

---

### F5 — P2 / CONFIRMED — the A-stage kernels have no automated test registration

**Where:** `CMakeLists.txt` — 258 `add_test(...)` entries; `lat_q3k_fused_test`, `lat_q3k_down_test`, `lat_q3k_gu_test`, `lat_q3k_counts_test`, `lat_q3k_engine_shape_test`, `lat_fused_q3k_test` are `add_executable` only. `ctest -N` therefore does not run them.

**Expected:** the kernels that produce 61 % of prefill time and carry a byte-exactness contract are in the regression gate.

**Observed:** the only registered parity tests are `lat_dense_d_tail_test`, `native_expert_parity*` and `kv_turbo_parity`.

**Impact:** F1/F2/F3/F4 cannot be regression-gated; the measured "wrong tokens" failure was caught by eye, not by CI.

**Fix:** register the six binaries with `add_test`, and add a byte-exact reference arm to `lat_q3k_fused_test` (the q3k pack's `pk_quant_row`-style oracle, as `kv_turbo_parity` already does for the KV codecs).

**Acceptance:** `ctest -R lat_q3k` runs and passes in the standard suite.

---

### F6 — P2 / CONFIRMED — the down epilogue recomputes 64-bit address arithmetic per store

**Where:** `lat_q3k_down.cu:269-280`.

**Observed:** `out[(size_t)(ent_dst != nullptr ? ent_dst[row0+r] : (row0+r)) * ld + WROWS*fb + (q<2 ? ... : ...)]` inside a 4×2×4 unrolled loop — 32 iterations, of which 12 store. ISA: 64 `v_mul_lo_u32` + 97 `v_add_co_u32` + 97 `v_add_co_ci_u32_e64`, i.e. a 64-bit multiply and carry chain per store.

**Fix:** compute the row base once per `r` (`(size_t) row_out * ld`), keep it in a 64-bit register or split into a base pointer, and index with a 32-bit offset.

**Acceptance:** `v_mul_lo_u32` falls to ≤ 4; bit-identical output.

---

### F7 — P2 / CONFIRMED — `gz = 8` is decided without knowing the row count, so 7 of 8 z-blocks are empty

**Where:** `lat_q3k_fused.cu:374-378`, `lat_q3k_down.cu:298-308`.

**Observed:** `gz` is widened whenever `gy * kItems < sms`, which is *always* true for the per-expert case (`gy = 1`). The tiles a group actually has is `ceil(nrows / 64)`, known on the host at the call site (`lat_gc[]` is already uploaded). For the live path `nrows ≈ 20–30` → 1 tile → 7 of 8 z-blocks exit.

**Impact:** 35 needless block launches per expert × 34,642 = 1.2 M empty blocks per prompt. Individually cheap (they exit in a few hundred ns), but they are launch-front cost on a card where the launch is already the problem.

**Fix:** pass the max row count (the host has it) or use `counts` on the device to compute `ceil(max(counts)/64)`; set `gz = 1` when one tile suffices. Better with F1: a batched launch has many groups, so `gy` alone fills the card and `gz` stays 1.

**Acceptance:** `gz == 1` for the per-expert path; identical results; wall/launch count unchanged or better.

---

### F8 — P2 / CONFIRMED — 36 IEEE divides per gate/up tile (deliberate, and it must stay)

**Where:** `lat_q3k_fused.cu:323` (`__fdiv_rn(gt, 1 + __expf(-gt))`), `:329` (`__fdiv_rn(am, 127.0f)`).

**Observed:** 72 `v_div_scale_f32` + 36 `v_rcp_f32` + 36 `v_div_fixup_f32` — 36 correctly-rounded divides, each a multi-instruction sequence.

**Classification: NOT A DEFECT.** The file's contract (`:8-13`) is byte-exact equality with `q8_1_store` / `q8_1_finite`, and `__fdiv_rn` is part of that contract. Recording it here so the next optimiser does not "fix" it into a `v_rcp_f32` and silently break the parity guarantee. Any change here needs the oracle.

---

### F9 — P2 / CONFIRMED — `moe_bufs()` sizes the latent buffers by `T*K`, which is why chunk > 512 collapses

**Where:** `prefill.cpp:1520-1525` (`moe_bufs`), the comment at `:1522` (`ts = min(T, stream_all_min()-1)`).

**Observed:** `group_bytes(T*K)` / `act_bytes(T*K, 640)` grow linearly with the chunk, so at T = 2048 the A-stage buffers are 4×, the expert cache is starved, and `gemm down` + `gather` explode (`tr_2048.log`: gather 26,278 ms = 26.8 %, down 40,295 ms = 41.1 %). The repo's own conclusion — "the auto chunk is right to stay at 512 on a 12 GB card" — is a **VRAM-budget** effect, not an arithmetic one.

**Fix:** pin the buffers to what the walk actually uses (`stream_all_min()-1` rows, the `ts` argument already present but only used in the `!fused` branch) so a larger chunk stops paying 4× buffer cost; then re-test chunk 1024/2048. This is the prerequisite for making 1024 the default (which is worth +16 % prefill on its own, measured).

**Acceptance:** at chunk 2048, `gather` and `gemm down` per entry stay within 10 % of chunk 512's; expert-cache slot count does not fall below the chunk-512 value.

---

### F10 — P2 / CONFIRMED — `lookahead` underflows for `group_resident` when `stream_ahead` is off

**Where:** `prefill.cpp:7113`

```cpp
const size_t lookahead = group_resident ? (size_t) (STAGE_GRP - MMQ_GROUP - 1) : STAGE - 1;
```

**Observed:** `STAGE_GRP = 32`, `MMQ_GROUP = 64`, both `int`, so `STAGE_GRP - MMQ_GROUP - 1 == -33`; cast to `size_t` it is `2^64 - 33`. With `stream_ahead == false` the staging predicate is `staged <= j + lookahead`, and `j + (2^64-33)` wraps to `j - 33`, so expert `j` is staged only once `staged ≤ j - 33` — the walk stages **32 positions behind** the compute point. The first 33 entries then have `stage_of[j] < 0` and are treated as **resident**, taking `m.cache->device_slot(m.host_res[...])` for an expert that is not resident.

**Reachability:** requires `STRATA_PREFILL_STREAM_AHEAD=0` (default is on) *and* `group_resident` (opt-in grouping on a native non-latent pack). So it is an A/B-knob-only defect, not a shipped-path defect — but it is a memory-safety one, and the A/B knob is documented as a valid experiment.

**Classification:** SUSPECTED (read from the expression; not executed). **Fix:** `(size_t) std::max(1, STAGE_GRP - MMQ_GROUP - 1)` or, better, delete the branch: with `STAGE_GRP = 32 < MMQ_GROUP = 64` the group-resident walk cannot hold one full group, so it should force `stream_ahead`.

**Acceptance:** a unit test on the predicate, or a `static_assert(STAGE_GRP > MMQ_GROUP)` if the geometry is meant to be strict.

---

### F11 — P2 / CONFIRMED — a launch error calls `std::exit(1)` from inside the hot path

**Where:** `lat_q3k_fused.cu:53-58` `dck()`, used at `:370`, `:379`; `lat_q3k_down.cu:39-44`, used at `:294`, `:311`.

**Observed:** any `cudaError_t` prints and calls `std::exit(1)`, bypassing stream cleanup, the expert cache's destructor and the 26.66 GiB pinned arena's release. Every other failure in `prefill.cpp` returns `false` with an `err` string that the caller unwinds.

**Impact:** a fail-closed path that leaks pinned host memory and device allocations on exit — the same class of teardown that the fence work was careful to avoid.

**Fix:** return a status (or set an error word) and let the walk fail the layer.

---

### F12 — P3 / CONFIRMED — the fused kernel hard-codes the 120B pack's geometry

**Where:** `lat_q3k_fused.cu:60-73` (`kNE 1280`, `kFF 640`, `kUpOff 352000`, `kGuRow 550`, `kXqRow 1440`), `lat_q3k_down.cu:47-52` (`kDownOff 704000`, `kDRow 360`, `kChunks 10`).

**Classification:** intentional and correctly guarded (`prefill.cpp:5619` requires `gu_type == 11 && d_type == 20 && lat_I == 640`), so **NOT A DEFECT for the shipped pack** — recorded so that a future pack reuses the guard rather than the constants.

---

### F14 — P1 / CONFIRMED — the latent **D stage** runs at 111 GFLOP/s (11.1 s, 21.2 %)

**Where:** `prefill.cpp:5502-5521`, `pt.mark(kPfGather)` then `strata::kernels::iq_mmvq(..., ncols = T, ...)` — the `gather` row of the phase table is this, not the expert gather.

**Observed (current shipped path, `pf_now.log`, 52,163 ms wall):** `gather` = **11,076 ms over 384 marks** = 28.8 ms per (layer, chunk) for 512 tokens × the D matrix (4864 → `G·ELL` = 5120). That is 25.5 GFLOP per layer, **1.23 TFLOP per prompt at 111 GFLOP/s**.

**Correction to the first version of this finding (kept visible on purpose):** I wrote that it "calls a matvec once per token so each token re-reads the whole D matrix". **That is half right, for a reason the code hides.** `iq_mmvq` *does* have a decode-once multi-column kernel (`mmvq_multi_kernel<NC=8>`, `iq_kernels.cu:1937-1970`, "bitwise equal" per column at `:958`, parity-checked by `iq_multi_parity`) — but for Q8_0 it is **not selected**: `split_off<8>()` is true because `kq_type<8>` is in the set whose multi kernels "run only with `STRATA_KQ_SPLIT=1` until the A/B on the measured cards decides" (`:965-969`). No such A/B had been run on gfx1031, so the per-column `mmvq_kernel` (`:946-949`, `for c in 0..ncols` re-reading the weight row each time) was the shipped path — which is exactly the "matvec called per token" behaviour, arrived at through a *dispatch flag* rather than a missing kernel.

**The A/B, run:** `STRATA_KQ_SPLIT=1`

| arm | prefill | tok/s | tokens |
|---|---|---|---|
| per-column (`KQ_SPLIT=0`) | 51,644 ms | 78.67 | canonical list |
| **decode-once (`KQ_SPLIT=1`)** | **45,863 ms** | **88.59** | **identical** |

**−11.2 %, bit-identical.** The phase table confirms it in the right place: `gather` 11,076 → **5,442 ms** (−51 %) while every other row held. `STRATA_KQ_SPLIT` now defaults to Q8_0-on, all-off with `=0`, all-on with `=1`; the K-quant formats (Q4_K/Q5_K) stay opt-in until the same A/B runs for them.

**What is left of the D stage (~5.4 s at 237 GFLOP/s):** the same treatment again — `mmvq_multi_kernel<8,8>` is 35 VGPRs / 0 spills, so it is not register-bound; the remaining 2× needs the occupancy/instruction-mix read (its `NC` could go 8 → 16, and the Q8_0 `Split::load`/`apply` could drop the per-block `half2float` of `dw`). → T15.

### F15 — P1 / CONFIRMED — the latent **U stage** runs at 128 GFLOP/s and 64 GB/s (12.5 s, 27.6 %)

**Where:** `src/kernels/cuda/latent.cu:70-105` (`latent_u_k`), launched by `latent_u_grouped` (`:107`); called from `prefill.cpp:6486` under `pt.mark(kPfGemmD)` — the `gemm down` row of the table.

**Measured with its own per-layer events** (`STRATA_DU_PROF=1`, `pf_why.log`): **12,551 ms over 384 (layer, chunk) calls**, mean 32.7 ms, min 26.1, max 41.6 — i.e. essentially all of `gemm down`'s 12,815 ms. Geometry from the new `latent u:` line: `fmt=8 n_groups=8 nt=10 H=2560 rows=5120 ELL=1280`.

**So the arithmetic:** 5,120 rows × 2,560 outputs × 1,280 values × 2 = **33.6 GFLOP per layer, 1.61 TFLOP per prompt at 128 GFLOP/s**, reading 5,120 × 2,560 × 1,280 B ≈ **16.8 GB of weights per layer, 805 GB per prompt at 64 GB/s**.

**Two fixes tried, both bit-identical, one of them refuted:**

| variant | result | why |
|---|---|---|
| **weight-reuse kernel** (one block per (group, 8 columns), weights in registers, reused over every routed row of the group; `KTAIL` raised to 16 so `nt=10` fits) | **49,879 ms vs 46,196 = 8 % SLOWER** (`ab_ureuse.log` vs `ab_urow.log`), tokens identical | it cuts weight bytes ~80× but replaces 1.3 M small blocks with 4,864 long ones, each scanning all 5,120 rows and skipping 7/8 of them. **Refuted: this stage is not weight-traffic-bound.** |
| the same kernel with `KTAIL = 8` | **0.08 %** — and the new `latent u:` line proved the kernel never ran (`nt=10 > KTAIL=8`) | recorded on purpose: "no change" there was a *configuration no-op*, not evidence. Without that one-line kernel-selection print this would have been logged as a refutation. |

**What is left is the shape, not the traffic.** One warp produces **one scalar output** from a 32→1 reduction over 1,280 values (10 dp4a per lane), so the reduction is never amortised across outputs and x is re-read once per output. A real tiled int8 GEMM (M = the group's rows, N = H, K = 1,280, int32 accumulate) is the shape that fixes it — a kernel project, not a tweak. → T16, with the per-layer instrument already in place to score it.

**Third attempt, and the most important lesson of the audit.** With a standalone probe (`u_probe.cu`, the kernel run back to back 50× at the engine's exact shape) a **row-walk** variant — the row's q8_1 blocks decoded once into registers, the block's warps walking `h` — measured **16.85 ms vs 25.25 ms (−33.3 %)**, output `memcmp`-equal across 50 MB, occupancy identical (8 blocks/CU for every variant). Ported and measured **in the engine**: **37.56 vs 32.69 ms a layer = 15 % SLOWER** (`ab_uwalk.log` vs `ab_uold2.log`, tokens identical).

| | standalone probe | in the engine |
|---|---|---|
| per-column (shipped) | 25.25 ms | 32.69 ms |
| row-walk | **16.85 ms (−33 %)** | **37.56 ms (+15 %)** |
| weight-reuse | — | 32.7 → 37.6 in situ, +8 % (measured earlier) |

The probe and the engine disagree because the probe runs the kernel **hot**: U (26.6 MB) and Hq (7 MB) stay in the Infinity Cache across 50 back-to-back launches. In the engine, the expert **A stage streams 60+ MB between two U calls** and evicts them, and only the per-column kernel's 1.3 M small blocks carry enough independent work to hide those cold misses. A kernel that wins on a hot-cache microbenchmark and loses in the pipeline is not an optimisation. **The shipped default is therefore the per-column kernel**; the row-walk stays behind `STRATA_LATENT_U_WALK=1`, and `u_probe.cu` is committed as the tool that produced the misleading answer.

**Both F14 and F15 in one sentence:** prefill's remaining 45 % is two dense projections executed as **matvec/GEMV kernels** — 2.7 TFLOP at ~110 GFLOP/s (0.2 % of the card). The A stage, which the earlier audit ranked first, is now 19 %.

### F16 — P2 / CONFIRMED — no quality gate exists for a numerically-equivalent-but-not-identical change

**Where:** the only gates in the tree are token equality (`det_A1`/`det_A2`) and `kv_turbo_parity`'s byte oracle. There is no KL/perplexity harness on this pack.

**Impact:** the two biggest remaining wins (F14, F15) deliberately change the summation order, so token equality will fail by design and cannot be used as the gate. Without a quality measurement, the honest options are: keep 110 GFLOP/s, or ship a change gated by "it still produces plausible text". Neither is acceptable.

**Fix:** a fixed-prompt KL (or teacher-forced NLL) against the current path's logits, on the real pack, with a threshold agreed before the change. → **T14**, and F14/F15 depend on it.

### F13 — P3 / CONFIRMED — the perf log's baseline and the recommended configuration are different paths

**Where:** `docs/perf-log-2026-11-16.md` §Baseline vs `run_512k_mtp.ps1` / `ctx_mtp_c_pf1024.log`.

**Observed:** the headline baseline (4K, 92.6 tok/s, 690 slots) is the `native_expert_grouped` A stage; the working 512K + MTP configuration enables `STRATA_LAT_FUSED_Q3K=1` and never runs that path. `c1024_base.log` has no `latfusedq3k:` line; `ctx_mtp_c_pf1024.log` does.

**Impact:** the published percentages (gate/up 21.9 %, gather 19.5 %) describe the *opt-out* path. Any optimisation judged against them will look like a win or a regression for the wrong reason. **Every A/B in this document therefore states the path.**

---

## 8. External evidence: llama.cpp's MoE cache, ProMoE, and what actually transfers

Three sources were read end to end. Their **numbers do not transfer** (different model, different link, different quantisation); their **mechanisms** do. Each row below is a mechanism, the measurement the source reports, and what it means for this codebase — mapped to a finding, an existing decision it confirms, or a new task.

### 8.1 llama.cpp `--moe-cache` (docs/moe-cache.md, PRs #29887 / #31233 / #30966, and the field reports)

| mechanism | source's measurement | Edge0 today | verdict |
|---|---|---|---|
| **The cache fills only on batches ≤ 32 tokens.** Large batches "read the cache but do not fill it", because a prompt "uses most experts of a layer, so they gain little and would evict the experts used in generation". | author: 2.20× at 89 % hit on an RTX 5090; slower on dual-GPU at 47–58 % | The cache **never evicts** and is **pre-filled from `profile.bin` by routing frequency** (`expert_cache.hpp:158-186`, `generate.cpp:7469-7503`, log: `pre-filled 697 of 697 slots from the profile in 0.1 s`). Prefill therefore *cannot* evict the hot set. | **Edge0 is structurally immune to the defect the ≤32 rule exists to paper over.** This is a stronger property than llama.cpp's. No change. |
| **ClassLRU**: 8 frequency classes, score halving every 128 tokens | +6–7 % output, −8 % uploads vs SLRU; hit rates 87.6 % vs 86.5 %, theoretical best 93.8 % | eviction is **absent by design** ("eviction policy is a measured question — R4.1's LFU-decay vs LRU sweep", `expert_cache.hpp:158`). Admission is static profile rank. | Not a defect: an **unrun measurement**. With a profile static admission is a reasonable prior, but nobody has measured LFU-decay against it here. → **T13** |
| **Cache size ≈ 10 % of the expert bytes**; the fastest 5090 row put *no whole expert layer* on the card, all experts in RAM, 18.6 GB of cache | 2.20× at 89 % hits | 697 slots = 0.76 GiB = **2.8 % of 24,576 experts** (28.5 GB of A blobs). The repo's own `Memory/cache_allocation.py` scores 8 slots/layer = **21.4 %**, 64 slots/layer = **70.4 %**; the served hit rate is ~28 %. | **The cache is the largest untouched decode lever**, and the curve to size it already exists in the repo. VRAM is going to projections (2,948 MiB) and the embedding (644 MiB), both movable. → **T10** |
| `--moe-cache-cpu-share`: "a small cache wants a high share, a large cache wants ~0"; **"pinned experts upload at the full speed of the link, so the CPU share often does not help"** | 6,464 MiB: 24.2 → 47.6 t/s at share 0.95; 19,392 MiB: 65.5 → 54.1 (17 % *slower*) | the 26.66 GiB arena **is** pinned; the engine refuses to arm the share (log: `only 3.8 GiB of the Windows commit limit is left`), and `STRATA_PREFILL_CPU_SHARE` stays off | **Confirms an existing decision.** The share is a small-cache mechanism; Edge0's cache is small *because VRAM is short*, and its misses already run on the CPU pool in decode. Only worth revisiting at a large cache with spare commit. |
| PR #32076 "removes sync points … expects a large gain on Vulkan and Windows, ~10 % on CUDA on Linux" | not merged at time of reading | per-blob `cudaEventRecord` (80 µs) + cross-stream wait (45 µs) under WDDM = the measured `wait copy` | **Independent corroboration of F1/F2.** This is a known defect class in this exact stack, not a local mistake. |
| Issue #31675: `--moe-cache-mib` **aborts at load on RX 6800 XT** (the same RDNA2 family) | 2,048 / 4,096 / 6,144 MiB all abort | n/a | If we ever A/B against llama.cpp on this card, pin a build without that bug or the comparison measures a crash. |

### 8.2 ProMoE (arXiv:2410.22134v3) — proactive caching

The paper's problem statement is this engine's problem, stated better: reactive caching leaves "expensive loading on the critical path", and **modern decoder-only MoE models have deliberately low routing skew** ("to mitigate routing collapse"), so hit rates are structurally capped. ★ **Edge0's 28 % is normal for this class of model, not a bug** — which is why the fix has to be overlap (prefetch/reorder), not a better cache alone.

| technique | what the paper does | Edge0 today | verdict |
|---|---|---|---|
| **Early preemption** | the expert set for a layer is known the instant the gate finishes; fetch the missing ones *then*, not on access | the host reads the routing ids and stages the layer's experts *before* the A stage (`stream_all` ring; the staged walk with `lookahead`) | **Already the design.** But at chunk 512 the walk still *waits per expert* before its own launch (`prefill.cpp:6416-6418` before this change). The batching in §7 F1/F2 is the first step; T1 completes it. |
| **Reordered inference** | run cached experts first, then those already being fetched, then those not yet started; the order of expert computation is **interchangeable** | the walk iterates experts in **id order** and waits for each staged copy in that order | **A real, cheap, bit-exact gap.** Each expert writes its own `yl[entry]` rows (`ent_dst` = identity), so nothing cross-expert is summed in the A stage: reordering cannot change a bit. → **T11**. The paper calls this "particularly severe … during the prefill stage" — our case exactly. |
| **Chunked prefetching** | split an expert's bytes into chunks (theirs: 3 linear layers) so a precise fetch can preempt a speculative one within one chunk boundary | a blob is 3 contiguous sections — gate/up `0..352000`, up `352000..704000`, down `704000..1164800` — a natural 2–3-way split | Only pays once copy queueing (not bandwidth) is the constraint; `wait copy` is still 11 % after batching. → **T12**, after T1/T11. |
| **GoodPred** = Accuracy × FetchRate | a metric for *any* predictor, so accuracy alone cannot look good while arriving too late | no predictor; the ring is the fetch mechanism | Useful **as a metric** for T11/T12 (fraction of staged blobs that arrived before their launch). The learned predictor itself is a project, not a task. |

### 8.3 What none of them justify

* No number from either source is quoted as an expected gain here. The 2.20× and the 70.4 % are *hypotheses to measure on this box*, and the acceptance criteria in §10 are measurements.
* Neither source touches the thing this audit found to be dominant — **the launch geometry of the A-stage kernel** (F1). ProMoE assumes the fused/persistent kernels of a data-centre engine; llama.cpp fuses experts at the CUDA-graph level. RDNA2 has neither, so the batching here has to be done in the walk.
* llama.cpp's `mmq` path *is* the thing Edge0's `lat_mmq` mirrors, and its per-expert gather is the reason `gather` costs 19.5 % here (F3) — the fused in-place kernel is the answer, and it is now the default.

### 8.4 MAPLE (`AmSach/maple`) — pruning, and one real discrepancy in this engine's prune path

MAPLE keeps 54 of 64 experts (chosen from a `llama-imatrix` activation profile) so a 16B MoE fits in 6 GB, and reports 65–76 tok/s where the offloaded model does 4.5–10. Its speedup is **entirely the removal of PCIe traffic**, not a kernel technique.

* **Not applicable to the problem in this document.** Edge0's 120B pack is 28.5 GB of expert blobs against 12 GB of VRAM — streaming is the design, not a defect, and the measured walls are three dense projections (F14, F15), not the link (9.1 %). Pruning would change the model; nothing in MAPLE makes a matvec faster.
* **What does transfer, and is a real discrepancy here:** MAPLE masks a pruned expert's router logit to **−∞** so that "softmax over active experts remains strictly normalized with zero numerical leakage", and lists "cascaded normalization suppression" (rescaling expert outputs, 0.795³⁰ → 0.001) as a documented failure mode. Edge0's rank prune instead substitutes a **zeroed blob** (`g_zero_blob`, `prefill.cpp`) and leaves the router's softmax denominator untouched — so a rank-pruned expert contributes 0 to the sum while still consuming its share of the softmax mass, and every surviving expert is systematically scaled *down*. That is the same class of defect MAPLE warns about, arrived at from the opposite direction. **Quality, not speed** → T17, gated on T14 (tokens necessarily differ).
* Its admission score `C(e,l) · sqrt(σ̄(e,l) + ε)` — routing frequency × channel energy — is a better statistic than the frequency-only `profile.bin` Edge0 ranks by. That belongs with the cache-admission question (T13).

## 9. The change that was landed, and what it measured

The audit above produced one change small enough to land and verify in this session; it is the pair the measurements rank first (F1 + F2), and it also promotes F3's in-place path to the default. Three edits, no arithmetic touched:

| # | edit | file |
|---|---|---|
| 1 | **Ring-safe batching of the staged A stage.** `lookahead = stage_n - stage_batch`; the batch is capped at `stage_n / 2` so half the ring stays prefetch; a batch may mix resident and staged experts because the fused call takes per-group row counts; one wait per batch on its last copy, then one release event per staged slot. `STRATA_LAT_STAGED_BATCH=N` (1 = the old per-expert behaviour). | `prefill.cpp` (lookahead, the staged walk) |
| 2 | **The in-place fused A stage is the default** (`STRATA_LAT_FUSED_Q3K=0` opts out). The guard is unchanged: `gu_type 11 / d_type 20 / lat_I 640`. Removes the `gather` phase. | `prefill.cpp` (`lat_fused_q3k_on`) |
| 3 | **Instrumentation.** `STRATA_LAT_Q3K_INFO=1` prints each A-stage kernel's registers, scratch, dynamic LDS, `blocks/cu` and waves/cu, measured on this device; `STRATA_LATENT_NPROF` now also prints `us/call` and the launch grid. | `lat_q3k_fused.cu`, `lat_q3k_down.cu`, `prefill.cpp` |

### 9.1 Prefill, same session, same flags, same prompt (4,063 tokens, chunk 512, `--kv int8 --spec 4`)

| arm | `STRATA_LAT_STAGED_BATCH` | launch grid | prefill wall | tok/s | decode | output |
|---|---|---|---|---|---|---|
| A | 1 (one expert per launch) | `(5,1,8)` | **62,717 ms** | 64.78 | 5.09 tok/s | 32 tokens |
| B | 4 (default) | **`(5,4,1)`** | **51,916 ms** | **78.26** | 5.08 tok/s | **identical** |

* **−17.2 % prefill wall, +20.8 % throughput**, with the same work streamed (63,692 vs 63,605 blobs, 33,545→ one launch per 4 experts).
* **The tokens are bit-identical**, not merely plausible: `248046 198 248045 74455 198 248068 198 760 1156 682 3106 264 1546 1248 11 21957 4055 886 314 1414 421 7701 310 381 264 6311 314 15019 20209 11 18890 33358` in both arms — and that is the *same* sequence the repo's own perf log records as the pre-existing baseline, so promoting the fused path to default did not change the model's output either.
* Decode is unchanged to within its own noise (5.09 / 5.08), as it must be: the A stage is prompt-only.
* `latq3kinfo` shows the mechanism: `gz` collapses from 8 to 1 and the launch goes from 5 working blocks to 20, on 20 CUs.

### 9.2 The instrument that isolates the kernel (NPROF pair, same flags, `--max-new 8`)

A second pair ran with `STRATA_LATENT_NPROF=1`, which times the fused A stage itself (both launches) with device events. This is the check the standing rule requires: the wall clock contains the copy stream and the host, so the kernel time must move too.

| arm | calls | **A-stage GPU ms** | entries | **us/entry** | us/call | prefill wall | tok/s | output |
|---|---|---|---|---|---|---|---|---|
| C | 58,392 | **9,820** | 1,773,010 | **5.54** | 168.2 | 62,943 ms | 64.6 | 8 tokens |
| D | 16,176 | **7,399** | 1,905,302 | **3.88** | 457.4 | 52,358 ms | 77.6 | **identical** |

* Calls: **58,392 → 16,176 (−72 %)**.
* A-stage device time: **9,820 → 7,399 ms (−24.7 %)** on slightly *more* entries (1.905 M vs 1.773 M), because the entry count is the same work and the second run's chunking differs by one.
* **`us/entry` 5.54 → 3.88 (−30 %)**: the same arithmetic is 30 % cheaper once the launch is big enough — the occupancy signature F1 predicted, now measured on the kernel rather than inferred from the wall.
* `us/call` rises 168 → 457 µs because a batched call now carries up to four experts; the per-call number is not the metric, `us/call / entries-per-call` is.
* Prefill wall: 62,943 → 52,358 ms = **−16.8 %**, reproducing the first pair's −17.2 % on a different run. Two independent pairs, two different instruments, same sign and magnitude.
* Tokens identical again (`248046 198 248045 74455 198 248068 ...`).

**Acceptance against the proposal (§11 T1/T2):** `nb > 1` ✔ (`nb=4`), `gemm gate/up`/A-stage device time down ≥ 40 %? — **no, −24.7 %**. That criterion was written from the earlier rejected arm's `−63 %` and is **not met**; the honest reading is that the earlier −63 % was measured with a *broken* ring (wrong tokens, and a launch that read stale blobs can take less time). The criterion is therefore revised to ≥ 20 % and the miss is recorded, not hidden.

### 9.3 What is still not run after this change

* `lat_q3k_*` unit tests (not registered in ctest — T3), so the kernel has token-level evidence, not a byte-level oracle, for this change.
* The 512K + MTP start with the new default (§9.4 covers it when run).
* KL / perplexity for the promoted default.
* A third A/B repeat on a freshly booted machine.

### 9.4 The tightest VRAM configuration still starts (512K context + MTP)

The default flip makes `lat_hbuf` (T·K·800 B) and `lat_gcd` always allocated, so it costs VRAM in every configuration, including the one that had only 580 MiB free. Re-ran the working 512K + MTP harness (`run_512k_mtp.ps1 -Embed 0`):

| | before (`ctx_mtp_d_embed0.log`) | after (`ctx_mtp_afterbatch.log`) |
|---|---|---|
| exit | 0 | **0** |
| expert cache | 720 slots | 696 slots (−24: the hbuf) |
| free after opening the cache | 580 MiB | **580 MiB** |
| prefill, 63 tokens | 3,081.9 ms = 20.4 tok/s | **2,259.1 ms = 27.9 tok/s** |
| decode | 2.23 tok/s | 2.19 tok/s |
| `latfusedq3k` | fused path | fused path (now the default, no env) |

Both configurations start and generate. The 63-token prompt is the case where the per-launch cost dominates (T·K = 630 entries), so **+37 % there** is the batching effect at its most visible; the 24 lost slots are the honest price of the always-allocated H buffer, and they are why T10 (VRAM rebalance) exists.

## 10. Checks executed vs not run
| check | status | evidence |
|---|---|---|
| Build (`build_hip_official.bat`, gfx1031) | **RUN** (prior session) | `BUILD_OK`, ~300 `nodiscard` warnings |
| `kv_turbo_parity` vs `krk::pk_quant_row` | **RUN — PASS** | arm `[6/6]`, `ALL TESTS PASSED`, turbo3 0/16,384 code bytes differ, 13/64 sigmas 1 ULP, margin 50× |
| 512K context start | **RUN — PASS** | `ctx_512k_t3_nomtp2.log`, 922 slots, 223 MiB free |
| 512K + MTP start | **RUN — PASS** | `run_512k_mtp.ps1` with `STRATA_EMBED_VRAM=0`, exit 0 |
| Prefill phase trace (chunk 512/1024/2048) | **RUN** | `tr_512.log`, `c1024_base.log`, `tr_2048.log` |
| A-stage per-call probe | **RUN** | `latfq3kprof` in `tr_2048.log`, `fused_probe.log` |
| gfx1031 ISA + register metadata | **RUN** | this document §6 |
| `hipInfo` hardware facts | **RUN** | §3 |
| `lat_q3k_fused_test` / `_down_test` / `_gu_test` | **NOT RUN** | binaries exist, not registered; not executed this session |
| **A/B: staged batch 1 vs 4 (wall clock)** | **RUN — PASS** | `ab_batch1.log` vs `ab_batch4.log`: 62,717 → 51,916 ms, tokens identical |
| **A/B: staged batch 1 vs 4 (device events)** | **RUN — PASS** | `ab_nprof1.log` vs `ab_nprof4.log`: 9,820 → 7,399 ms, `us/entry` 5.54 → 3.88, tokens identical |
| **512K + MTP with the new default** | **RUN — PASS** | `ctx_mtp_afterbatch.log`: exit 0, 696 slots, 580 MiB free, prefill 27.9 tok/s |
| `ctest` full suite | **NOT RUN** | 258 registered tests; not executed this session |
| Real-model KL / perplexity for turbo3 | **NOT RUN** | no reference harness |
| `copy+combine` split into copy/combine stamps (F15) | **NOT BUILT** | the repo names it "the next trap"; not done |
| Decode path audit (`src/kernels/**`, 186 files) | **NOT STARTED** | out of this batch |
| `STRATA_LATENT_BATCH=1` at chunk 1024 end to end | **NOT RUN with correctness gate** | the only measured variant (`post_batch2.log`) produced wrong tokens |
| `--draft-vocab en` | **BLOCKED** | `draft_vocab_en.bin` not generated |

Nothing in this section is called PASS unless a log or a binary's exit code is named.

---

## 11. Proposed tasks (not created — acknowledgement gate)

No issues, tickets or reminders were created. The following are ready to file once approved. Dependencies are stated; each has an acceptance criterion that is a measurement, not an opinion.

| # | task | depends on | effort | acceptance |
|---|---|---|---|---|
| **T1** | Ring-safe batching of the latent A stage: `lookahead = stage_n - batch`, batch gated on `lat_fused_q3k`, default `batch = 4` at `STAGE = 8` | — | M (1 file, ~60 ln) | `nb > 1`; A-stage device time −20 % (revised from −40 %: the earlier −63 % came from a broken-ring arm); tokens bit-identical — **LANDED: `nb=4`, grid `(5,4,1)`, −24.7 % device / −30 % `us/entry` / −17 % wall, tokens identical; unit tests are T3** |
| **T2** | Promote the in-place fused A stage to default (`lat_fused_q3k_on()` on, `=0` escape) | T1 (batching is only wired on this path) | S (1 fn) | no `gather` phase in the default trace; 11.3 s recovered; tokens identical to `STRATA_LAT_FUSED_Q3K=0` — **LANDED: bit-identical to the repo's recorded baseline token list** |
| **T3** | Register the six `lat_q3k_*` binaries with `ctest` | — | S | `ctest -R lat_q3k` green |
| **T4** | Register diet: decode directly to smem on HIP, drop `wc[2]`; hoist the down epilogue's 64-bit base | T3 | M (~80 ln, ISA re-check) | `vgpr ≤ 200`, spills 0, `us/entry` -10 % at equal `nb`, bit-identical |
| **T5** | Size `moe_bufs()` from `stream_all_min()-1`, then re-measure chunk 1024/2048 | T1 | M | chunk-2048 per-entry `gather`/`down` within 10 % of chunk 512 |
| **T6** | Fix the `lookahead` underflow (F10) | — | S | `static_assert` or max(1,…) + a predicate test |
| **T7** | Replace `dck()`'s `std::exit(1)` with a returned error | — | S | an injected launch failure unwinds with `err` set and no leak report |
| **T8** | Split the decode `copy+combine` stamp into copy and combine (F15) | — | S | two stamps in the T=4 stage table |
| **T9** | Default the auto chunk to 1024 once T5 lands | T5 | S | prefill ≥ +10 % at 4K with decode within 3 % |
| **T10** | VRAM rebalance for a bigger expert cache: `STRATA_EMBED_VRAM=0` (644 MiB) + a smaller `--vram-reserve-mib`, sized toward the repo's own `Memory/cache_allocation.py` curve (8 slots/layer = 21.4 %, **64 = 70.4 %** vs the served ~28 %) | — | S (config) + M (automating the sweep) | measured hit rate and decode tok/s at 8 / 32 / 64 slots per layer, same prompt; report the curve, not one point |
| **T11** | **Reordered inference** (ProMoE §5.3): run the layer's resident experts first, then the staged ones in staging order, instead of expert-id order. Bit-exact — each expert writes only its own `yl[entry]` rows, nothing cross-expert is summed in the A stage | T1 | M (~40 ln in the walk) | `wait copy` falls without `gemm` rising; tokens bit-identical; `us/entry` unchanged or better |
| **T12** | Chunked prefetch (ProMoE §5.1): split a blob's copy into its natural sections (gate/up `0..352000`, up `352000..704000`, down `704000..1164800`) so a precise fetch can start at a chunk boundary | T1, T11 | M | `wait copy` falls again; no token change; only accepted if T11's headroom is exhausted |
| **T13** | Measure the eviction/admission question the header calls unmeasured (R4.1): static profile rank vs LFU-decay (llama.cpp's ClassLRU shape: 8 frequency classes, score halved every 128 tokens) | T3 | M | hit rate + tok/s at a fixed slot count, same prompt; the loser is removed, not kept behind a flag |
| **T14** | **A quality gate**: fixed-prompt KL (or teacher-forced NLL) of the current logits vs a candidate path, on the real pack, threshold agreed before the change | — | M | a number, repeatable to ±0.1 %, and a documented threshold. **Blocks T15/T16** |
| **T15** | **D stage as a GEMM** (F14): 512 tokens share one D matrix; tile them instead of calling a matvec per token | T14 | M–L | D stage 11.1 s → < 2 s; KL under threshold; per-layer `[dprof]` before/after |
| **T16** | **U stage as a per-expert GEMM** (F15): use the bucketed `goff`/`rows` the CUDA path already consumes, one accumulator per output tile instead of a warp reduce per scalar | T14 | M–L | U stage 12.5 s → < 3 s; weight traffic 765 GB → ~25 GB; KL under threshold |
| **T17** | **Prune-path renormalisation** (F15 note, MAPLE §8.4): compare the zeroed-blob stand-in against masking the pruned expert's router logit to −∞ at R=7 | T14 | S–M | quality delta on a fixed prompt (KL / NLL), and a documented decision; tokens change either way, so T14 first |

**The 2× is T15+T16.** They are 23.5 s of a 52.2 s prefill, both running at ~110 GFLOP/s = 0.2 % of the card. A tiled int8 GEMM at even 5 % of peak would take them to ~1 s. Everything else in this table is 1–4 s.

**Recommended order:** T3 (gate first) → T1 → T2 → T4 → T11 → T10 → T5/T9 → T6/T7 → T12/T13 → T8.

The rationale for that order: T3 gives every later change a regression gate; T1+T2 are the two measured multi-second wins on the same path; T4 is the kernel diet that only pays once launches are big enough to matter; T11 and T10 attack what remains after them (copy stalls, then decode residency) and are both cheap; T5/T9 trade prefill for decode and so must be measured last, against a stable instrument.

---

## 12. Residual risks

1. **The batched variant has already produced wrong tokens once.** T1's correctness argument is arithmetic (`lookahead + batch - 1 < stage_n`), but the acceptance gate must be token equality *and* the unit tests, not the argument.
2. **Token equality is only valid with matched instrumentation flags** — the speculation policy adapts to measured latency, and `STRATA_PREFILL_TIMING=1` changes the text. Every A/B above must share flags.
3. **Absolute tok/s is not comparable across the 16:00 boundary** on this machine (PCIe measured 34 % below its own earlier ceiling). Only within-session ratios are evidence.
4. **F3 changes a default numeric path.** It is bit-exact by design and guarded to one pack geometry, but the guard has never been exercised on a non-120B pack; the escape hatch must be documented in the same commit.
5. **F4's register work is the only change that can silently alter results** (it touches the arithmetic's operand source). It must land with the ISA dump attached and the bit-exact test run.
6. **The decode audit is not done.** The decode Pareto (`copy+combine` 41.1 %, `hc1 up` 24.8 %) comes from the repo's own instrument, which the repo itself flags as ambiguous (one stamp covering a copy and a kernel). Treat the decode ranking as SUPPORTED, not CONFIRMED.
7. **Two papers were read for whole-layer prefill / prerouter / pre-gated overlap** (`arxiv_2609.18063v1`, `pregated.md`); none of their techniques is in this task list because none of them was measured on this box. They are candidate T10+ work, not findings.
