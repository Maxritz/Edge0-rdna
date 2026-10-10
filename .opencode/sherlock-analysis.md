# Sherlock Analysis Log

## [RUN-001] 2026-10-09 — MODE: FULL — TARGET: Edge0 / llama.cpp (ROCm HIP, gfx1201)

### BASELINE
- Hardware: AMD Radeon RX 9070 XT, gfx1201 (RDNA4), 16 GiB VRAM, ReBAR on.
  CPU 16 threads; 95.9 GiB RAM.
- Engine: pinned llama.cpp (b11100) + Edge0 patch band, HIP build (`wt/win/build-hip`).
- Model: Qwen3.5-35B-A3B (MXFP4 experts + Q6_K dense), 32k ctx, n_slots=4.
- Plan: `-ngl 99 --n-cpu-moe 16`.
- Workload: prompt eval + 128-1481 token decode.

Recorded metrics (engine logs + sweeps + test-backend-ops):

| Metric | Value | Source |
|---|---|---|
| decode (tg) | 36.6-38.1 tok/s | engine log `print_timing` |
| prompt eval (cold, 18 tok) | 39 tok/s | engine log |
| prompt eval (warm, 5560 tok) | 440 tok/s | manual log |
| prompt eval (warm, 3439 tok) | 543 tok/s | manual log |
| model load | ~7.0 s | engine log |
| f16 GEMM (m=4096,n=512,k=14336) | **8.44 TF** | test-backend-ops |
| q4_K GEMM (same shape) | 75.1 TF | test-backend-ops |
| mxfp4 GEMM (same shape) | 86.2 TF | test-backend-ops |
| q6_K GEMM (same shape) | 39.2 TF | test-backend-ops |
| tilelang f16 GEMM (4096^3) | 99 TF | tilelang bench |
| decode device-time share | 88.2% | perf.rs trace |

### FINDINGS
| Rank | Component | Cost / evidence | Status |
|------|-----------|-----------------|--------|
| 1 | hipBLASLt Tensile data not staged | engine log: `rocblaslt error: Cannot read TensileLibrary_lazy_gfx1201.dat`; engine bin has `rocblas\library\{gfx1200,gfx1201}` but **no `hipblaslt\library`**; SDK has `hipblaslt\library\{gfx1200,gfx1201}`. `ggml-hip.dll` imports `hipblas.dll`; arch dispatch (rocBLAS vs hipBLASLt) happens inside hipBLAS. **RDNA2 (gfx1031) -> rocBLAS; RDNA4 (gfx1201) -> hipBLASLt.** `Copy-HipRuntime` in `windows/scripts/vendor-build.ps1` copies rocblas only, never hipblaslt | CONFIRMED |
| 2 | f16 dense GEMM 8.44 TF vs 99 TF peak | test-backend-ops vs tilelang | SUSPECTED (linked to #1) |
| 3 | decode bandwidth-bound | decode 88.2% of device time; sweep shows decode falls with more CPU experts (40->26 tok/s as n_cpu_moe 12->31) | CONFIRMED |
| 4 | mmap + CPU tensor overrides | engine log: "tensor overrides to CPU are used with mmap enabled - consider using --load-mode none" | CONFIRMED (warning) |
| 5 | q6_K GEMM only 39 TF (~40% peak) | test-backend-ops | CONFIRMED |

### HYPOTHESES
| ID | Claim | For | Against | Test | Cost | Status |
|----|-------|-----|---------|------|------|--------|
| H1 | Staging `hipblaslt\library\<arch>` (RDNA3/4) fixes the BLASLt error and lifts f16/dense GEMM; RDNA2 targets need `rocblas\library\gfx1030` staged instead | error + missing dir; arch split confirmed | fallback may still be slow | extend `Copy-HipRuntime`, re-run test-backend-ops f16 | low | PENDING |
| H2 | `--load-mode none` improves the CPU-expert path | engine warning | mmap is usually fine for CPU experts | A/B load-mode | low | PENDING |
| H3 | decode is bandwidth-bound, not compute | 88.2% dev share; sweep slope | no per-op split yet | phase-scoped GPU util + rocprof 1-step | med | PENDING |
| H4 | q6_K GEMM can be improved toward peak | 39 TF vs 75+ for q4_K | may be inherent to q6_K | tilelang q6_K dequant-GEMM A/B | med | PENDING |

### ACTIONS
- [ ] Extend `Copy-HipRuntime` in `windows/scripts/vendor-build.ps1` to stage `hipblaslt\library\<arch>` (and the rocblaslt dll) for RDNA3/4 targets, and `rocblas\library\gfx1030` for RDNA2; re-run `test-backend-ops perf -o MUL_MAT` (H1).
- [ ] A/B `--load-mode none` vs mmap on the 32k plan (H2).
- [ ] Phase-scope the perf.rs resource sampler to the generation window so GPU util is meaningful for decode (H3).
- [ ] tilelang q6_K dequant-GEMM at m=4096,n=512,k=14336, A/B vs ggml 39 TF (H4).
- [ ] Re-verify: decode tok/s, prompt tok/s, f16 GEMM TF after each change.

### NOTES
- test-backend-ops ran from the same staged bin as the engine, so its numbers reflect
  the engine's runtime, including the BLASLt gap.
- tilelang f16 GEMM (99 TF) is the workload-relevant ceiling for dense prefill on this GPU.
- Nothing changed yet; this RUN is baseline + diagnosis only.

## [RUN-002] 2026-10-09 — MODE: HIGH — TARGET: same

### DELTA from RUN-001
- H1 CONFIRMED (kernel level): staged `hipblaslt\library\{gfx1200,gfx1201}` from the SDK
  into the engine bin (`xcopy`). The `rocblaslt error` / `hipModuleLoad failed` lines are gone.
- f16 large-N GEMM (m=4096, n=512, k=14336): **8.42 TF -> 96.33 TF (+11.4x)**.
- f16 small-n (decode) unchanged (~610 GFLOPS) - bandwidth-bound, as expected.

### PREFILL / DECODE DATA (Qwen3.5-35B, -ngl 99 --n-cpu-moe 16, -fa auto, pp512/tg128)
| Metric | BEFORE | AFTER | Δ |
|---|---:|---:|---:|
| pp512 t/s | 504.27 ± 78.45 | 512.46 ± 83.43 | ~noise (+1.6%) |
| tg128 t/s | 40.49 ± 0.17 | 41.87 ± 0.96 | +3.4% (near noise) |
| f16 GEMM (4096x512x14336) | 8.42 TF | 96.33 TF | +11.4x |
| rocblaslt error | present | gone | fixed |

### ANALYSIS (why the 11x kernel win does not move this model)
- The 35B GGUF has **no f16 (type 1) tensors**: types are MXFP4(39), Q6_K(14), F32(0),
  Q8_0(8), Q4_K(12), Q5_K(13). So the f16 hipBLASLt GEMM path is **unused by this model**,
  and its prefill is bounded by the **CPU-expert GEMM** (n_cpu_moe 16) plus attention, not
  by f16 GEMM.
- The staging fix still matters: it removes error spam and restores f16/bf16 dense GEMM for
  any model that actually carries f16/bf16 tensors (and is a prerequisite for the tilelang
  f16-GEMM comparison to be meaningful).

### UPDATED FINDINGS
| Rank | Component | Cost / evidence | Status |
|------|-----------|-----------------|--------|
| 1 | hipBLASLt staging | f16 GEMM 96 TF, errors gone | FIXED |
| 2 | real-model prefill | pp512 ~512 t/s, flat vs f16 fix | CONFIRMED (CPU-expert bound) |
| 3 | decode bandwidth-bound | tg128 ~41 t/s, unchanged by f16 fix | CONFIRMED |

### NEW HYPOTHESES
| ID | Claim | For | Against | Test | Cost | Status |
|----|-------|-----|---------|------|------|--------|
| H5 | Prefill is CPU-expert bound | pp512 flat despite 11x GPU f16 GEMM | - | sweep n_cpu_moe 0/8/16 prefill | med | PENDING |
| H4 | q6_K GEMM (39 TF) improvable | q4_K 75, mxfp4 86 | may be inherent | tilelang q6_K A/B | med | PENDING |

### ACTIONS
- [x] Stage `hipblaslt\library\<arch>` (done for gfx1200/gfx1201); script fix applied to `Copy-HipRuntime`.
- [x] Retest kernel (f16 GEMM) and record prefill (pp512/tg128).
- [ ] Sweep `n_cpu_moe` for prefill to confirm the CPU-expert bound (H5).
- [ ] Rebuild via `vendor-build.ps1` so the staging fix is baked into the build output (this run staged manually).

## [RUN-003] 2026-10-09 — MODE: HIGH — TARGET: same — prefill sweep

### WORKLOAD
llama-bench, Qwen3.5-35B-A3B (MXFP4+Q6_K), `-ngl 99 -fa auto -p 512,2048 -n 128 -r 3`,
sweep `--n-cpu-moe`. hipBLASLt already staged.

### DATA
| n_cpu_moe | pp512 | pp2048 | tg128 |
|---:|---:|---:|---:|
| 0  | 759.6 | 763.2 | 33.8 |
| 8  | 429.5 | 453.4 | **44.1** |
| 16 | 481.3 | 536.2 | 40.7 |
| 24 | 339.9 | 384.2 | 30.4 |
| 31 | 267.0 | 304.9 | 25.8 |
| 40 | 219.1 | 247.3 | 21.9 |

### FINDINGS
| Rank | Component | Cost / evidence | Status |
|------|-----------|-----------------|--------|
| 1 | Prefill is CPU-expert bound | pp2048 rises monotonically as experts move to GPU: 247 (40) -> 763 (0), ~3.1x | CONFIRMED (H5) |
| 2 | Decode has a sweet spot at n_cpu_moe=8 | tg128 peaks 44.1 at 8; 33.8 at 0 (spills), 21.9 at 40 | CONFIRMED |
| 3 | Prefill vs decode tension | prefill wants 0 CPU experts, decode wants ~8 | CONFIRMED |

### ANALYSIS
- At `n_cpu_moe=0` all experts target the 16 GiB card; the model (18.32 GiB) does not fully
  fit, so it spills to system RAM over PCIe. Prefill amortizes weight reads across the batch
  (fast, 763 t/s); decode reads weights every token (bandwidth-bound, drops to 33.8).
- The current default plan (`--n-cpu-moe 16`) is a compromise: pp512 481, tg128 40.7.
  `--n-cpu-moe 8` trades prefill (429, -11%) for decode (44.1, +8.4%).
- pp512 non-monotonicity (8 < 16) is within the ±78 noise band; pp2048 is the cleaner signal.

### UPDATED HYPOTHESES
| ID | Claim | Status |
|----|-------|--------|
| H5 | Prefill is CPU-expert bound | CONFIRMED |
| H3 | Decode is bandwidth-bound | CONFIRMED |

### ACTIONS
- [ ] Pick plan per workload: prefill-heavy -> low n_cpu_moe; decode-heavy -> 8.
- [ ] Re-run on the 12 GiB target profile (n_cpu_moe higher) to get its curve.

## [RUN-004] 2026-10-09 — MODE: HIGH — TARGET: expert pool type coverage

### FIX
`windows/serve/prefetch.cc` `tt_bytes()`: replaced the hand table (F32/F16/BF16/Q4_1
only) with `ggml_blck_size` / `ggml_type_size`, so every ggml type is sized and the
per-expert stride is exact. Unknown types still return 0 (fail closed). Removed the
now-unused `is_q4_1` / `plain_esz`.

### VALIDATION (fast, no full rebuild)
`examples/pool_types.rs` loads `ggml-base.dll` and checks the real
`ggml_type_size`/`ggml_blck_size` per type + per-expert bytes for the target shapes.
Result: **PASS**. Every expert type sizes to an exact integer stride:
Q2_K 84/256, Q4_K 144/256, Q5_K 176/256, Q6_K 210/256, IQ2_XXS 66/256,
MXFP4 17/32, NVFP4 36/64, Q8_0 34/32. Old code returned 0 (skipped) for all but
F32/F16/BF16/Q4_1 — and mis-sized BF16 (assumed 2 B/elem; this fork's BF16 is
blocked 18/32).

### LIVE POOL TEST (35B, pool-only: E0_POOL_MB=4096, no E0_PREROUTER)
```
[pref-trace] POOL2 init: tt=120 caps hot=4096MB obs=0MB resolver=on   # was tt=0
[pref-trace] POOL2 key1=blk.0.ffn_down_exps.weight per=557056         # MXFP4
[pref-trace] POOL2 fill#1024 blk.2.ffn_down_exps.weight per=860160    # Q6_K
[pref-trace] POOL2 hit=327680 fillD=4074 bypass=0 wait=60377 hot=2285MB
```
- tt=120 (40 layers x 3) registered; per-expert bytes exact; `bypass=0` (resolver
  never rejected a size); pool fills and hits accumulate. MXFP4 + Q6_K experts are
  now pooled — previously skipped. `map=FAIL` is expected in pool-only mode (the
  read-only view is only mapped for head/selfcheck; fills use the file handle).

### STATUS
- Pool type coverage: FIXED. Unblocks MXFP4 (gpt-oss, 35B-UD), NVFP4, K-quants
  (Qwen3.8), IQ2_XXS+Q2_K (DeepSeek-V4).
- Remaining: wire `--pool-mb` into the local-GGUF load path (currently Edge0-tier only);
  measure pool hit rate + disk throughput on a real >32 GiB model (DeepSeek / gpt-oss).

## [RUN-005] 2026-10-09 — MODE: HIGH — gpt-oss-120b pool-only + --pool-mb wiring

### CHANGES
- `engine.rs`: local GGUF loads now pass `--pool-mb` (`pool_for_gguf()`: `E0_POOL_MB`
  override, else RAM-clamped default; 0 disables). Build clean.
- `windows/README.md` §1.4 updated (local GGUFs now use the pool).

### WORKLOAD
gpt-oss-120b Q8_0 (63.4 GB, MXFP4 experts, 36 layers x 128 experts, K=4),
`-ngl 99 --n-cpu-moe 32 -c 4096 -fa auto`, pool-only (no E0_PREROUTER), 27-token prompt
+ 96 decode tokens.

### DATA
| pool | tt | fillD | hits | bypass | coverage | hot | decode | prefill |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 GiB  | 108 | 974  | 81920  | 576535 | 12.4% | 4093 MB  | 10.48 tok/s | 2.75 tok/s |
| 6 GiB  | 108 | 1462 | 147456 | 504121 | 22.6% | 6143 MB  | 10.98 tok/s | 4.52 tok/s |
| 8 GiB  | 108 | 1949 | 212992 | 453076 | 32.0% | 8190 MB  | 10.84 tok/s | 4.39 tok/s |
| 16 GiB | 108 | 3898 | 442368 | 216691 | 67.1% | 16380 MB | 10.70 tok/s | 2.85 tok/s |

Coverage ≈ pool_size / ~24 GiB working set (linear until it saturates): 4/6/8/16 GiB ->
12/23/32/67%. The effective distinct-expert working set touched is ~24 GiB, so a 6-8 GiB
pool (the product-tier range) covers only a quarter to a third of gpt-oss expert reads.

- tt=108 = 36 layers x 3 MXFP4 expert tensors registered (per=4406400 exact, matches
  `pool_types`); resolver accepts (per==nb02).
- Coverage = hits/(hits+bypass): fraction of CPU-expert accesses served by the pool vs
  mmap. Scales strongly with pool size.

### FINDINGS
| Rank | Component | Cost / evidence | Status |
|------|-----------|-----------------|--------|
| 1 | Pool coverage scales with size | 4->16 GiB: 12.4% -> 67.1% (-360k mmap bypasses) | CONFIRMED |
| 2 | Pool has no eviction | slots fill once; once at cap (~930 experts @4 GiB, ~3898 @16 GiB) remaining experts bypass to mmap permanently | CONFIRMED (by design) |
| 3 | gpt-oss expert working set ~18 GiB | 32 CPU layers x 128 experts x 4.4 MB; needs ~18 GiB pool to fully cover | CONFIRMED |
| 4 | Decode unchanged by pool size | 10.5 vs 10.7 tok/s; this box has 96 GiB RAM so experts are page-cache resident (no disk pressure) | CONFIRMED |
| 5 | Single-fill waits | wait=68629 (threads sleeping on a slot being filled) | SUSPECTED tax |

### ANALYSIS
- The MXFP4 fix works on gpt-oss (tt=108, exact strides). The limiter is pool *capacity*
  vs working set, not type support.
- The no-eviction design (fill-once, no recycle) is safe but wrong for working sets
  larger than the pool: it fills with the first-touched experts and cannot adapt. On a
  32 GiB machine the gpt-oss 18 GiB expert set cannot be fully pooled, so a large
  fraction stays on the mmap path — the reload churn the pool is meant to remove.
- Decode here is not a disk test (96 GiB RAM holds the whole 63 GiB file in page cache);
  the pool's value shows up under real RAM pressure, which this machine cannot reproduce.

### ACTIONS
- [ ] Pool eviction (LRU / age-based) so a small pool adapts to the hot set instead of
      filling once. This is the change that makes disk streaming real for >pool models.
- [ ] Re-test under an enforced RAM cap (Job Object) to see disk-throughput impact.
- [ ] Consider an explicit "stream experts" mode (force `--n-cpu-moe` = all, pool on)
      for models that would otherwise fit, per the disk-tier intent.

## [RUN-006] 2026-10-10 — MODE: HIGH — pool step-boundary eviction

### CHANGE
`serve/prefetch.cc`: per-slot `lastuse`/`sreg`; `e0_pool_step()` decommits slots not
served within `E0_POOL_KEEP_STEPS` steps, gated on `g_pool_full` (only when the working
set exceeds the pool). Runs from `edge0_pref_on_step` after `llama_synchronize`.
Slot stride page-aligned (`stride = align_up(per, 4096)`) so commit/decommit of one
slot never touches a neighbour (gpt-oss `per`=4406400 is NOT page-aligned; 35B's is).

### BUGS FOUND AND FIXED (both would corrupt/crash)
1. Evicting without `llama_synchronize` decommitted pages a compute thread was mid-read
   on -> 0xC0000005 access violation in ggml-cpu. Fixed: `e0_pool_step` runs after sync.
2. Non-page-aligned slot stride made `VirtualAlloc`/`Free` overlap adjacent slots ->
   corruption/AV (only on models whose per-expert size is not a 4096 multiple, e.g.
   gpt-oss MXFP4 4406400). Fixed: page-aligned stride.

### VERIFICATION (correctness)
`E0_POOL_SELFCHECK=1` byte-compares every fill against the mmap view:
| run | fills | sc ok | sc bad |
|---|---:|---:|---:|
| 35B fits (no evict) | 4887 | 4887 | **0** |
| gpt-oss, evict on | 7414 | 7414 | **0** |
Zero mismatches -> eviction + refill serve correct bytes.

### DATA (gpt-oss-120b, MXFP4, `--n-cpu-moe 32`, pool 8 GiB)
| variant | evict | fills | coverage | hot | decode |
|---|---:|---:|---:|---:|---:|
| keep=1000000 (no evict) | 0 | 1949 | 35.0% | 8190 MB | 9.95 tok/s |
| keep=16 (evict on) | 5883 | 7414 | 49.7% | 6434 MB | 4.64 tok/s |

### HONEST FINDING
- Eviction works and bounds pool RAM (hot 6.4 GB < 8 GB cap) and lifts "coverage", BUT on
  this 96 GiB box it is **net-negative**: eviction causes refills (7414 vs 1949) and decode
  drops 9.95 -> 4.64 tok/s. With ample RAM the OS page cache holds the file and mmap is
  effectively free; the pool's explicit `ReadFile` is slower than a page-cache-resident
  mmap fault.
- The pool (9.95 tok/s) is itself **slower than mmap-only** (~12.9 tok/s) on this box.
  The pool only wins under real RAM pressure, which this machine cannot reproduce.
- "Coverage" (hits/(hits+bypass)) is the wrong KPI: a bypass is a fast page-cache hit, and
  an evicted+refilled slot is a slow disk read. Disk reads / refills is the metric that
  matters, and eviction increases them.
- Conclusion: eviction is correct and necessary for RAM-constrained machines, but must be
  tuned (or off by default) where RAM is plentiful. Needs a real 32 GiB test to justify.

## [RUN-007] 2026-10-10 - MODE: HIGH - split-precision FP32 GEMM (tilelang, gfx1201)

BASELINE: 1.5 TF @ 1024x1024x2048 (incl H2D+D2H); ggml ROCm0 f16 = 96.3 TF, f32 = 12.4 TF
FINDINGS:
| Rank | Component | Cost | Evidence | Status |
| 1 | occupancy | dominant | 6 shared buf = 64 KB/block (2 fp32 + 4 fp16) -> ~1 CTA/SM on RDNA4 | CONFIRMED |
| 2 | 3x MMA count | 3x FLOPs | hi*hi+hi*lo+lo*hi by design | CONFIRMED (inherent) |
| 3 | vectorization | minor | "T.vectorized extent 8 lowered as serial" | SUSPECTED |
VERDICT: even fixed, 3x-MMA caps at ~32 TF (96/3); only worthwhile if fp32-accurate prefill is
required AND occupancy is restored. ggml f32 already 12.4 TF -> max upside ~2.6x prefill, not free.
Separately the kernel is WRONG (abs err 39.9 vs f64, plain fp16 0.024) - prime suspect the
swizzled-Ah residual read-back (gen_split.py); PROBE NEEDED, see ponytail-diag.

## [RUN-008] 2026-10-10 - MODE: HIGH - MTP self-speculative decode

WORKLOAD: Tiel-Coder-35B-A3B-MTP-APEX (qwen35moe, Q5_K, embedded MTP head, 26.7 GB),
-ngl 99 --n-cpu-moe 24 -c 4096 -fa auto, 160 decode tokens, greedy. Pin flags present:
--spec-type draft-mtp, --spec-draft-n-max, --spec-draft-p-min, --spec-draft-ngl.
DATA:
| variant | acceptance | mean len | decode tok/s |
| baseline (no MTP) | - | - | 20.74 |
| MTP n-max 6, p-min 0.6 | 0.696 | 2.84 | 8.82 |
| MTP n-max 4, p-min 0.5, draft-ngl 99 | 0.639 | 3.00 | 17.39 |
FINDING: MTP works but is net-negative (-16% best case) on expert-offloaded MoE. With experts
on CPU the verification batch CPU matmul scales with batch, so extra tokens are NOT free in the
target; spec decode only wins when the target is GPU-resident/bandwidth-bound. --spec-draft-ngl 99
mattered (8.82 -> 17.39). Not a lever for the CPU-expert regime.

## [RUN-009] 2026-10-10 - MODE: FULL - RDNA2 (gfx1031) vs RDNA4 (gfx1201)

### BASELINE
- Hardware A: RX 6700 XT, gfx1031 (RDNA2), 12 GiB VRAM, ~320 GB/s.
- Hardware B: RX 9070 XT, gfx1201 (RDNA4), 16 GiB VRAM, ~640 GB/s (RUN-003).
- Engine (A): our patch band built on the remote box, `D:\edge0\wt\win\build-hip\bin`,
  build eb0ef8074, ROCm at D:\Rocm10. Same `llama-bench -p 512 -n 128 -ngl 99 -fa auto`.
- Models on C:\x (fast disk; D: is slow). VRAM: 12272 MiB reported.

### DATA - Qwen3.5-35B-A3B Q4_K (18.32 GiB) `--n-cpu-moe` sweep
| n_cpu_moe | RDNA2 pp512 | RDNA2 tg128 | RDNA4 pp512 | RDNA4 tg128 |
|---:|---:|---:|---:|---:|
| 0  | 435 | 26.75 | 760 | 33.8 |
| 8  | 292 | 28.86 | 430 | 44.1 |
| 16 | 245 | 31.36 | 481 | 40.7 |
| 24 | 319 | **34.64** | 340 | 30.4 |
| 31 | 259 | 29.27 | 267 | 25.8 |
| 40 | 210 | 24.20 | 219 | 21.9 |

Other MoE (RDNA2, n_cpu_moe 16): Laguna-XS.2 IQ4_XS 461 pp / **46.21** tg; GLM-4.7-Flash-APEX
215 pp / 29.25 tg; L3.2-8X3B (llama arch) Q8_0 466 pp / **11.90** tg.

### FINDINGS
| Rank | Component | Cost / evidence | Status |
| 1 | VRAM ceiling (12 GiB) shifts the decode optimum | RDNA4 peaks at n_cpu_moe 8 (44.1); RDNA2 peaks at 24 (34.6). At 0 the 18.32 GiB model+KV exceeds 12 GiB -> spills -> 26.75. | CONFIRMED |
| 2 | RDNA2 decode = 0.79x RDNA4, prefill = 0.57x | decode peak 34.6 vs 44.1; prefill 435 vs 760 (matches ~0.5 bandwidth + no WMMA on RDNA2 -> DP4A) | CONFIRMED |
| 3 | Best RDNA2 config | Laguna-XS.2 IQ4_XS 16.84 GiB, 46.2 tok/s (fits fully, GDN/hybrid) | CONFIRMED |
| 4 | Q8_0 MoE is bandwidth-bound | L3.2-8X3B Q8_0 18.21 GiB -> 11.9 tok/s (8.5 bpw) | CONFIRMED |

### ANALYSIS
- "Uses all 12 GB": at n_cpu_moe 0 the model does not fit; the optimum is the smallest
  n_cpu_moe that keeps VRAM near-full (24 here) -> decode 34.6. Same shape as the 16 GiB box
  (which fits at 8). VRAM capacity, not just bandwidth, sets the decode sweet spot.
- RDNA2 vs RDNA4 decode gap (0.79x) is smaller than the prefill gap (0.57x): decode is
  bandwidth-bound (weights read once) so it tracks the ~0.5 BW ratio plus CPU-expert share;
  prefill is compute-bound and RDNA2 lacks WMMA (DP4A path).
- Disk (D:) is slow on this box; models moved to C:\x. Engine binary stays on D: (small).

### ACTIONS
- [ ] Record RDNA2 numbers in benchmarks.md (done below).
- [ ] Fixed cross-machine comparison captured for the pcie/bandwidth model.

## [RUN-010] 2026-10-10 - MODE: FULL - decode bottleneck (batch-1 GEMV efficiency)

### BASELINE
- Model Qwen3.5-35B-A3B Q4_K: 40 layers, 256 experts, top-8, embd 2048.
  file 19.69 GB; expert bytes 18.04 GB; dense trunk = 19.69 - 18.04 = 1.65 GB;
  KV 81920 B/token (f16). Per expert 1.76 MB; per layer top-8 = 14.1 MB/token.
- Decode (tg128) measured: RDNA4 gfx1201 n_cpu_moe 8 = 44.1 tok/s (22.7 ms/tok);
  RDNA2 gfx1031 n_cpu_moe 24 = 34.6 tok/s (28.9 ms/tok).

### DECOMPOSE - bytes moved per decode token (batch 1)
| n_cpu_moe | GPU layers | VRAM bytes/token | CPU bytes/token |
|---:|---:|---:|---:|
| RDNA4 @ 8 | 32 | dense 1.65 GB + 32*14.1 MB = 2.10 GB | 8*14.1 MB = 113 MB |
| RDNA2 @ 24 | 16 | dense 1.65 GB + 16*14.1 MB = 1.88 GB | 24*14.1 MB = 338 MB |

### INVESTIGATE - observed vs hardware ceiling
| GPU | peak VRAM BW | achieved BW (VRAM bytes * tok/s) | utilisation |
|---|---:|---:|---:|
| RDNA4 gfx1201 | ~640 GB/s | 2.10 GB * 44.1 = 92.6 GB/s | **14%** |
| RDNA2 gfx1031 | ~320 GB/s | 1.88 GB * 34.6 = 65.0 GB/s | **20%** |

- Implied ceiling if decode saturated VRAM BW: RDNA4 ~305 tok/s, RDNA2 ~170 tok/s.
  Measured is 6-7x below that -> decode is NOT bandwidth-bound at the achieved rate.
- The CPU-expert share of the 22.7 ms: 113 MB from RAM at ~50 GB/s ~= 2.3 ms ~= 10%
  (RDNA4); the remaining ~90% is GPU time running the MMVQ/GEMV path at ~14% BW.

### FINDINGS
| Rank | Component | Cost / evidence | Status |
| 1 | batch-1 GEMV path runs at 14-20% of VRAM BW | achieved 92.6/65.0 GB/s vs ~640/320 peak | CONFIRMED |
| 2 | decode is op/launch/occupancy-bound, not BW-bound | 6-7x gap to the BW ceiling; earlier test-backend-ops f16 m4096 n1 k14336 = 611 GFLOPS (tiny) | CONFIRMED |
| 3 | CPU-expert share ~10% at n_cpu_moe 8 | 113 MB/token RAM read | SUSPECTED |
| 4 | VRAM capacity shifts the optimum | RDNA2 n_cpu_moe 24 vs RDNA4 8 (RUN-009) | CONFIRMED |

### HYPOTHESES
| ID | Claim | For | Against | Test | Cost | Status |
| H1 | larger decode batch (spec-decode/verify) lifts BW util | BW-bound only when many rows amortise weight reads | MTP measured net-negative on CPU-expert MoE (RUN-008) | re-measure decode util with n=4/8 rows | low | PENDING |
| H2 | MMVQ kernel occupancy is the limiter | 14-20% BW at batch 1 | no per-kernel counters yet | rocprof on 1 decode step | low | PENDING |
| H3 | CPU expert matmul cost dominates at high n_cpu_moe | 338 MB/token + ~3B active CPU GEMM | decode peaks mid-sweep, not at 0 | phase-scoped perf.rs (CPU vs GPU window) | med | PENDING |
| H4 | GPU-resident dense trunk (1.65 GB) dominates at low n_cpu_moe | 1.65 GB is 79% of the 2.10 GB/token at n_cpu_moe 8 | - | sweep shows decode falls at 0 then rises | low | PENDING |

### ANALYSIS
- "Bandwidth-bound" was too loose for batch-1. The MMVQ/GEMV path reads each weight
  row exactly once but processes one activation column, so it saturates well below
  peak BW unless the batch is wide. The lever is not "read fewer bytes" alone; it is
  "process more columns per weight read" (verification batches) or a better GEMV kernel.
- This is consistent across both GPUs (RDNA2 20% > RDNA4 14% because RDNA2 peak BW is
  lower, so the same op time is a larger fraction).
- RDNA2's DP4A (no WMMA) affects prefill (0.57x) more than decode (0.79x): decode
  under-utilises compute so the tensor-core gap is not the decode limit.

### ACTIONS
- [ ] Trap (Dispatch/Dependency): per-kernel counters with rocprof on one decode step
      (grid dims, occupancy, BW) to name the exact limiter (H2).
- [ ] Trap (Transfer): phase-scope perf.rs resource sampler to the decode window to
      split CPU-expert vs GPU time (H3).
- [ ] Verify H1 by measuring decode BW util at batch 4-8 (verification-style) vs 1.
- [ ] Keep the RDNA2 engine for a matched rocprof run (RPCS provider over SSH).

### NEXT RUN
- Execute the Dispatch trap (rocprof) on both GPUs to convert rank-1 from
  "14-20% BW" to the named kernel + occupancy number.

## [RUN-011] 2026-10-10 - MODE: FULL (trace) - RDNA2 (gfx1031) app trace + pool-default regression

### BASELINE
- Hardware: RX 6700 XT gfx1031, 11.98 GiB VRAM, 47.9 GiB RAM, ROCm D:\Rocm10, ReBAR off.
- Engine: our patch band built on the box (D:\edge0\wt\win\build-hip\bin), ctx 32768,
  n_slots 4, threads 6, model C:\x\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf (19.69 GB).
- Tool: `trace_demo.exe` built on the box (cargo, 4m56s), EDGE0_BIN_DIR -> RDNA2 engine.

### TRACE (component table, pool DEFAULT on = buggy path)
| component | scope | ops | %dev | dev us | host us |
|---|---|---:|---:|---:|---:|
| engine.decode | engine | 128 | 83.5% | 58635.23 | 0.00 |
| engine.prompt-eval | engine | 18 | 16.5% | 82263.33 | 0.00 |
| engine.launch | app | 1 | - | - | 9785168.00 |
| engine.start_gguf | app | 1 | - | - | 11907057.00 |
| engine.wait_ready | app | 1 | - | - | 9699499.00 |
| gguf.detect_gpu | app | 2 | - | - | 1571786.50 |
| gguf.inspect_for_gpu | app | 1 | - | - | 2121841.00 |
- instrumentation floor: 0.86 us host each (256 empty ops).
- decode 58.6 ms/tok = 17.06 tok/s; prompt 82.26 ms/tok. resources: gpu 15.5%, vram 9.59/11.98, cpu 44%.

### INVESTIGATE - why 17 tok/s when llama-bench got 31-34 at the same n_cpu_moe
True/false tests (all back-to-back, same box, ctx 32768, n_cpu_moe 26):
| test | claim | result |
|---|---|---|
| T1 threads | server uses fewer threads than bench | FALSE - bench -t 6 == default (31.85 vs 31.93) |
| T2 sampling | sampling dominates server decode | FALSE - temp 0.2 (30.53) vs greedy (28.44) = ~6% |
| T3 ctx size | 32768 vs 8192 is slower | FALSE - 8192/16384/32768 all ~28.3 tok/s |
| T4 warmup/clocks | cold GPU on first run | FALSE - reproduces warm (17.4, 17.9) |
| T5 spawn method | CREATE_NO_WINDOW / job / cwd | FALSE - spawn_probe plain/nowin/job/full/nocwd all 27.9-28.8 |
| T6 pool default | app passes --pool-mb, bench does not | **TRUE** |

A/B/A (identical exe + args): manual 28.26, 28.39, 28.49 vs trace_demo 16.92, 15.6, 17.4 tok/s.
Slow log is full of `[pref-trace] POOL2 ... bypass=176229 evict=12896 wait=80556`; fast log has none.
`start_gguf` -> `pool_for_gguf()` defaulted 2048 MB (`--pool-mb 2048`).

### FINDINGS
| Rank | Component | Cost / evidence | Status |
| 1 | app default `--pool-mb 2048` for local GGUF | 58.6 vs 34.8 ms/tok decode (1.69x); pool counters show thrash | CONFIRMED |
| 2 | pool is net-negative when the model fits in RAM | 19.69 GB model << 47.9 GB RAM; mmap page cache wins | CONFIRMED |
| 3 | decode still op-bound (not BW) at 28.8 tok/s | 34.8 ms/tok at n_cpu_moe 26 | CONFIRMED |

### HYPOTHESES
| ID | Claim | For | Against | Test | Cost | Status |
| H1 | RAM-aware pool default removes the regression | E0_POOL_MB=0 -> 28.8 tok/s | none | change default, re-trace | low | **CONFIRMED** |
| H2 | pool helps only when model > RAM | theory + 96 GB finding (RUN-006) | not measured on a <=20 GB box | low-RAM box test | med | PENDING |

### OPTIMISE
`pool_for_gguf(model_path)`: enable the pool only when `model_bytes + 4 GB headroom >`
physical RAM; `E0_POOL_MB` remains an explicit override. Unit test
`engine::tests::pool_default_is_ram_aware_and_overridable` covers both directions + override.

### VERIFY (RDNA2, same box, same model, rebuilt)
| metric | before (pool 2048) | after (RAM-aware off) | delta |
|---|---:|---:|---:|
| decode ms/tok | 58.6 | 34.8 | -40.6% |
| decode tok/s | 17.1 | 28.8 | +68% (1.69x) |
| prompt ms/tok | 82.3 | 51.6 | -37% |
- Matches the manual mmap baseline (28.4). Reproduced across runs. `cargo test --lib` 13/13.

### ACTIONS
- [x] Implement RAM-aware pool default (H1) + unit test.
- [x] Rebuild + re-trace on RDNA2 -> decode restored to 28.8 tok/s.
- [ ] Test H2 on a <=20 GB RAM machine (pool expected to help only there).

### NEXT RUN
- If a <=20 GB RAM box is available, validate H2; otherwise next-highest gap is the
  batch-1 GEMV efficiency (RUN-010) via a dispatch/instruction trap.

## [RUN-012] 2026-10-10 - MODE: FULL (trace) - RDNA2 decode optimum + arch instruction map

### BASELINE
- Box: Ryzen 5 5600X (6c/12t), 47.9 GiB RAM, RX 6700 XT gfx1031 11.98 GiB, ROCm D:\Rocm10.
- Model Qwen3.5-35B-A3B Q4_K (19.69 GB), ctx 32768, -fa auto, n_predict 256.
- Method: llama-server + GPU/CPU PDH sampling during decode, and llama-bench (r=3).

### DATA - decode curve (ctx 32768, llama-server)
| n_cpu_moe | pp tok/s | tg tok/s |
|---:|---:|---:|
| 16 | 23.0 | 32.37 |
| 20 | 21.9 | **35.05** |
| 24 | 20.6 | 31.33 |
| 26 (planner pick) | 19.6 | ~29.5 |
| 28 | 19.1 | 28.30 |
| 32 | 18.2 | 25.75 |
| 36 | 17.6 | 23.55 |
| 40 | 15.8 | 21.93 |

Decode curve (ctx 8192) with resource sampling:
| n_cpu_moe | 0 | 4 | 8 | 12 | 16 | 20 | 24 | 28 | 32 | 36 | 40 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tg tok/s | 26.27 | 26.10 | 26.58 | 28.68 | 30.17 | **34.76** | 31.06 | 27.73 | 25.30 | 23.46 | 21.71 |
| gpu med % | 95.4 | 83.1 | 71 | 63.7 | 52.9 | 54 | 52.9 | 55.4 | 55.4 | 54.7 | 54.7 |
| cpu med % | 95.8 | 82.3 | 70.8 | 63.9 | 51.5 | 54.1 | 53 | 56.3 | 55.9 | 55.8 | 55 |

llama-bench p32/n256 (r=3): n16 31.37+-0.26, n20 **37.60+-0.47**, n24 33.69+-0.43.

### FINDINGS
| Rank | Component | Cost / evidence | Status |
| 1 | decode is non-monotonic in n_cpu_moe; peak ~20 | 35.05 vs 29.5 at the planner's 26 (+19-21%), reproducible both ctx and bench | CONFIRMED |
| 2 | planner picks the smallest-fitting n (26), not the fastest (20) | plan args --n-cpu-moe 26; budget_gib 7.85 = vram 11.85 - 1.5 headroom - 2.5 KV | CONFIRMED |
| 3 | GPU and CPU util move in lockstep | both 95% at n0, both ~54% at n>=20 | CONFIRMED |
| 4 | neither GPU nor CPU is saturated at the peak | 54%/54% at n20 -> latency, not throughput, bound | CONFIRMED |

### ARCH INSTRUCTION MAP (verified in wt/win/ggml/src/ggml-cuda/common.cuh)
| feature | gfx1031 RDNA2 | gfx1201 RDNA4 | evidence |
|---|---|---|---|
| int8 dot | `__builtin_amdgcn_sdot4(a,b,c,false)` | `__builtin_amdgcn_sudot4(true,a,true,b,c,false)` | common.cuh:722-726 |
| v_dot2_f32_f16 | yes | yes | common.cuh:769-775 (V_DOT2_F32_F16_AVAILABLE) |
| WMMA (f16/bf16/iu8/iu4/fp8) | none | yes | common.cuh:279-281 (AMD_WMMA_AVAILABLE = RDNA3/RDNA4) |
| FP8 e4m3 | no | no (CDNA3 only) | common.cuh:857 |
- Consequence: RDNA2 prefill runs the DP4A (sdot) path (no tensor cores) -> prefill 0.57x
  RDNA4 (RUN-009). Decode is weight-streaming bound on both -> 0.79x. Consistent.
- MMQ/ROCmFP4 port target on gfx1201: `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12`
  (and iu8). On RDNA2 no WMMA exists, so the same quant needs a v_dot/dp4a MMVQ path.

### HYPOTHESES
| ID | Claim | For | Against | Test | Cost | Status |
| H3 | optimum is a CPU/GPU balance point (latency-bound) | both ~54% util at peak | - | sweep confirms shape | low | **CONFIRMED** |
| H5 | planner should target the measured peak, not smallest-fit | +19-21% decode at n20 | risk if model/box differs | planner change + re-verify | med | PENDING |
| H2 | pool helps only when model > RAM | RAM 47.9 >> model 19.7 -> pool off correct | needs <=20 GB box | low-RAM box test | med | PENDING |

### OPTIMISE / ACTIONS
- [ ] H5: prefer the decode-optimal n_cpu_moe, not the smallest that fits. Options:
      (a) planner budget accuracy (it under-counts GPU headroom -> over-offloads), and/or
      (b) an opt-in measured auto-tune of n_cpu_moe over a small candidate set.
- [ ] Port plan updated: target wmma ...iu4/iu8 on gfx1201; v_dot/dp4a MMVQ on RDNA2.

### NEXT RUN
- Decide H5 (planner accuracy vs auto-tune) and verify the chosen n reproduces >=35 tok/s
  and still fits at ctx 32768.

### RUN-012 QUANT PATH MAP (port target per format)
| operation | gfx1031 (RDNA2) | gfx1201 (RDNA4) | correctness requirement |
|---|---|---|---|
| signed INT4 dot | pack 8 nibbles/32-bit; supported integer-dot (v_dot8/sdot/dp4a) | integer dot or IU4 WMMA | correct sign extension + accumulation |
| unsigned INT4 dot | packed unsigned dot | IU4 WMMA, unsigned input controls | correct zero-point handling |
| asymmetric INT4 | integer dot + scale/zero-point correction | IU4 WMMA + scale/zero-point correction | correct zero-point compensation |
| NVFP4 E2M1 | decode packed FP4, apply block scales, FP32/F16 arithmetic or proven integer approx | decode+scale; test native IU4 WMMA or other native path | preserve NVFP4 exponent/mantissa/scale semantics |
| custom FP4 (integer-friendly) | custom decoder + packed integer dot | custom decoder + IU4 WMMA | quant/dequant conventions must match |
| FP8 E4M3 / BF8 E5M2 | software convert + arithmetic fallback | native FP8/BF8 WMMA | match format, scaling, saturation, rounding |
| two-level / dual-scale FP4 | decode and apply both scales | fuse scale application into the matmul/accumulation kernel | exact scale granularity and rounding |

### RUN-012 TODOS
- [ ] T1 (H5): stop the planner choosing the slowest-fitting n_cpu_moe. Either fix the
      budget (it under-counts GPU headroom -> over-offloads to CPU) or add an opt-in
      measured auto-tune of n_cpu_moe over a small candidate set. Verify the chosen n
      reproduces >=35 tok/s at ctx 32768 and still fits.
- [ ] T2 (port): ROCmFP4/TurboQuant HIP kernels. gfx1201: `__builtin_amdgcn_wmma_i32_16x16x16_iu4_w32_gfx12`
      (+iu8). gfx1031: no WMMA -> v_dot8/v_dot4 (dp4a) MMVQ path. Enum+sizes+dequant+MMVQ+MUL_MAT_ID+KV dtype.
- [ ] T3 (H2): validate the pool on a <=20 GB RAM box (pool should help only there).
- [ ] T4 (RUN-010): batch-1 GEMV efficiency dispatch/instruction trap (deferred; needs rocprof).

## [RUN-013] 2026-10-10 - MODE: HIGH - autotune n_cpu_moe (fixes RUN-012 T1/H5)

### BASELINE
- RDNA2 box, Qwen3.5-35B-A3B Q4_K, ctx 32768. Planner picks n_cpu_moe 26 -> decode 28.8 tok/s.
- RUN-012 measured the true peak at n_cpu_moe 20 (35.05 tok/s).

### CHANGE
- New `windows/app/src-tauri/src/autotune.rs`: `llama-bench -o json` sweep over a candidate
  set (planner pick, +/- step, and the ~0.6x peak region), parses `avg_ts` per n_cpu_moe,
  picks the measured best (tie -> smaller n), caches per (model, size, ctx, gpu) in
  `state/autotune.json`. No estimates: only `llama-bench` numbers are used.
- `engine::apply_tuned_ncpu_moe`: rewrites `--n-cpu-moe N` to the cached/swept best.
  Default OFF; `EDGE0_AUTOTUNE=1` runs the sweep once. New command `engine_autotune`.

### BUG FOUND + FIXED DURING VERIFY
- First probe returned `measured: []` — `llama-bench` has no `--ctx-size` flag (it sizes KV
  itself; flags are -fitc/-fit-target). Passing it made every run error. Removed; ctx is now
  only a cache key. (Compiler-free bug, caught by the empty result, not assumed away.)

### VERIFY (RDNA2, real run)
| stage | planned | applied | decode ms/tok | tok/s |
|---|---:|---:|---:|---:|
| before (planner) | 26 | 26 | 34.7 | 28.8 |
| autotune sweep measured | 26 | best=20 (tg 35.02) | - | - |
| app after cache | 26 | **20** | 29.3 | **34.1** |
- Sweep output: 12->28.99, 16->29.07, 20->35.02, 22->33.15, 26->29.77, 30->27.04.
- App log: `[autotune] n_cpu_moe 26 -> 20`; decode 34.1 tok/s (+18%), matches bench 35.0.
- `cargo test --lib` 20/20 (8 new autotune tests).

### ARCH ASYNCCOPY/WAIT NOTE (compiler-verified)
| builtin | gfx1031 | gfx1201 | evidence |
|---|---|---|---|
| `__builtin_amdgcn_s_wait_asynccnt` | REJECTED (`needs gfx1250-insts`) | REJECTED (same) | local compile |
| `__builtin_amdgcn_sched_barrier(0)` | emits `s_waitcnt vmcnt(0) expcnt(0) lgkmcnt(0)` | emits `s_wait_loadcnt_dscnt` + granular waits | local compile |
- Use `sched_barrier(0)` for a coarse drain; `s_wait_vscnt`/`s_wait_dscnt` only at real
  data-dependency boundaries. `cp.async.wait_group` has no direct AMD equivalent here.

### TODOS (remaining, honest)
- [ ] T2 ROCmFP4/TurboQuant HIP port (gfx1201 wmma iu4/iu8; gfx1031 v_dot/dp4a).
- [ ] T3 pool-on-<=20-GB-RAM validation.
- [ ] T4 batch-1 GEMV dispatch/instruction trap (needs rocprof; not installed).
- [ ] RUN-001 H2 `--load-mode none` A/B; H4 q6_K tilelang GEMM A/B; phase-scope perf sampler.

## [RUN-014] 2026-10-10 - MODE: HIGH - --load-mode none vs mmap (fixes RUN-001 H2)

### BASELINE
- RDNA2 box, Qwen3.5-35B-A3B Q4_K, n_cpu_moe 20. Engine warns:
  "tensor overrides to CPU are used with mmap enabled - consider using --load-mode none".

### DATA (llama-bench, r=3)
| test | mmap (default) | --load-mode none | delta |
|---|---:|---:|---:|
| pp32 | 34.88 | 152.26 | 4.4x |
| tg256 | 37.63 | 38.40 | +2% |
| pp2048 (realistic) | 433.89 | 772.43 | 1.78x |

### FINDING
| Rank | Component | Cost / evidence | Status |
| 1 | mmap demand-pages CPU-expert weights during prefill | pp2048 434 -> 772 tok/s with RAM-resident load | CONFIRMED |
| 2 | prefill (not just pp32) gains in absolute terms | +338 tok/s at p2048 | CONFIRMED |
| 3 | decode unaffected | 37.6 -> 38.4 within noise | CONFIRMED |

### FIX
- `engine::load_mode_for_gguf(path)`: `--load-mode none` when model + 4 GiB <= RAM, else None
  (mmap needed to demand-page). Wired into `start_gguf`.
- Unit test folded into `ram_derived_defaults_are_ram_aware_and_overridable`.

### VERIFY (RDNA2, real trace_demo run, cache + load-mode both on)
- Newest engine log: **no mmap warning** (flag applied). decode 27.9 ms/tok = 35.8 tok/s.
- Both RAM-derived defaults (pool off, load-mode none) now active together.

### TODOS (remaining, honest)
- [ ] T2 ROCmFP4/TurboQuant HIP port (gfx1201 wmma iu4/iu8; gfx1031 v_dot/dp4a).
- [ ] T3 pool-on-<=20-GB-RAM validation.
- [ ] T4 batch-1 GEMV dispatch/instruction trap (needs rocprof; not installed).
- [ ] H4 q6_K tilelang GEMM A/B; phase-scope perf sampler to the decode window.

## [RUN-015] 2026-10-10 - MODE: HIGH - session close: resource peaks + remaining-item status

### DONE THIS SESSION
- RUN-011 fix: RAM-aware pool default (17.1 -> 28.8 tok/s, +68%).
- RUN-013 fix: measured n_cpu_moe autotune (26 -> 20, +18%; new `autotune.rs`).
- RUN-014 fix: RAM-resident `--load-mode none` (pp2048 434 -> 772 tok/s, 1.78x).
- This run: perf report now carries window-peak cpu/gpu/vram (fixed the 15.5% false
  util reading that misled RUN-010). All 21 lib tests pass; full release build clean.

### VERIFY (final RDNA2 trace_demo, all fixes active)
- decode 26.0 ms/tok = 38.5 tok/s (was 58.6 ms/tok at session start - >2x).
- engine log: no mmap warning; [autotune] n_cpu_moe 26 -> 20; no POOL2 lines.
- resources: gpu_util_max 64.2, cpu_pct_max 54.5 (peak, not last sample).

### REMAINING (honest status)
| item | status | blocker |
|---|---|---|
| T2 ROCmFP4/TurboQuant HIP port | NOT DONE | multi-stage kernel work + format-matched fixtures |
| T3 pool on <=20 GB RAM box | NOT DONE | no such machine; process cap is not a RAM cap |
| T4 batch-1 GEMV dispatch trap | BLOCKED | rocprof absent on both boxes (rechecked) |
| H4 q6_K tilelang GEMM A/B | NOT DONE | tilelang kernel work |

### NOTE
- T4 cannot be executed without rocprof; per-kernel counters are unavailable. The util/latency
  evidence (RUN-012) stands in for it until a profiler is present.

## [RUN-016] 2026-10-10 - MODE: FULL - ROCmFPX type bring-up (T2 stage 1) on gfx1031

### BASELINE
- Fixtures: ornith-1.0-9b-ROCmFPX-STRIX_LEAN (qwen35, types 100/101), gemma-4-E2B
  ROCMFP4 (gemma4, type 100), ornith-1.0-35B-Q3_0 (qwen35moe, types 101/102/104).
- Before: all three "failed to load model" (unknown tensor types).

### CHANGE (patch 0007, replayed into wt/win)
- ggml types ROCMFP4/4_FAST/6/8/3/2 (enum 43..48, COUNT 49); block structs + traits;
  CPU dequant/quant + generic vec_dot; CUDA dequant to_fp32/to_fp16; GGUF code remap
  (100/101/102/103/104/107 -> the new ids; 105/106 turbo are KV, excluded).
- Routing so the types take dequant-to-f16 + cuBLAS: added to supports_op MUL_MAT gate,
  and excluded in should_use_mmvq + should_fuse_mul_mat_vec_q (else dispatch aborts).

### FINDINGS (each a real defect caught by testing, not assumed)
| Rank | Component | Evidence | Status |
| 1 | load path | three fixtures now load + generate (were hard-fail) | FIXED |
| 2 | cpu vec_dot rounded the scaled weight | gemma-E2B output `<unusedN>`; roundf(0.5)=0 | FIXED (float accumulate) |
| 3 | prototypes in wrong header | remote build error: 12 undeclared in ggml-cpu.c | FIXED (quants.h) |
| 4 | MMVQ dispatch abort | `mmvq.cu:1428 fatal error` on generate | FIXED (should_use_mmvq + fuse guard) |
| 5 | build band strips wt/win edits | git reset --hard wipes direct edits | NOTE (use patches/) |

### VERIFY (ornith-9B, gfx1031)
| metric | before | after |
|---|---:|---:|
| load | fails | loads, ready ~4 s |
| output | n/a | coherent ("Thinking Process: ...") |
| decode tok/s | 2.30 (bad vec_dot) | 4.02 |
| prefill tok/s | 2.72 | 14.63 |

### REMAINING (honest)
- Decode 4 tok/s is far below native: the dequant-to-f16 path reloads f16 from
  ggml-common each token. Needs a real ROCmFPX MMVQ kernel (dequant inline, dp4a on
  RDNA2 / wmma iu4 on RDNA4) -> stage 2.
- turbo3/turbo4 KV not implemented.

### ACTIONS
- [ ] Stage 2: ROCmFPX MMVQ kernel (gfx1031 dp4a; gfx1201 iu4 wmma).
- [ ] Test all three fixtures incl. the 35B MoE (types 101/102/104) end to end.
- [ ] Autotune/lb/ub + GGML_OP_OFFLOAD_MIN_BATCH prefill levers (MoE offload guide).
- [ ] Research digest written: docs/research-references.md (DFlash, MoE-Infinity, guide).

## [RUN-018] 2026-10-10 - MODE: FULL (trace) - per-op device timing + all-model sweep

### BASELINE
- gfx1031, all 11 models in C:\x, engine = this fork. Plan chosen by gguf_tool.py.
- New: EDGE0_CUDA_OP_TIMING=1 -> per-node HIP event pairs in ggml_backend_cuda_graph_compute,
  aggregated by ggml_op_name (disables CUDA graphs + MoE fusion while profiling).

### PER-OP COMPONENT TABLE (Qwen3.5-35B-A3B, n_cpu_moe 22, decode window)
| component | ops | dev ms | %dev |
|---|---:|---:|---:|
| MUL_MAT | 25963 | 1906.75 | 32.5% |
| ADD | 28538 | 704.99 | 12.0% |
| MUL | 18661 | 596.16 | 10.1% |
| RMS_NORM | 12689 | 437.69 | 7.4% |
| UNARY | 11294 | 388.74 | 6.6% |
| GET_ROWS | 6709 | 248.96 | 4.2% |
| MUL_MAT_ID | 3561 | 215.17 | 3.7% |
| CPY | 3988 | 168.82 | 2.9% |
| SCALE | 4168 | 156.63 | 2.7% |
| ARGSORT/SOFT_MAX/CLAMP/SUM_ROWS/DIV/GATED_DELTA_NET/SSM_CONV/ROPE/FLASH_ATTN_EXT | - | <2% each | - |

### FINDINGS
| rank | component | evidence | status |
| 1 | device time is NOT the decode wall | total_dev 5875 ms over 1536 graphs ~= 3.8 ms/graph, but decode is ~130 ms/token | CONFIRMED |
| 2 | MUL_MAT dominates device time (dense projections) | 32.5%; MUL_MAT_ID (experts) only 3.7% because most experts run on CPU | CONFIRMED |
| 3 | decode is CPU-expert + scheduling bound, not GPU-kernel bound | device 3.8 ms/graph vs 130 ms/token: >95% is outside the timed op window | CONFIRMED |
| 4 | Qwen3.5 MoE already uses hybrid linear attention | GATED_DELTA_NET + SSM_CONV present in the graph | CONFIRMED |

### IMPLICATION
- Per-op GPU timing alone cannot see the CPU-expert cost; the wall is the CPU-side
  MUL_MAT_ID for offloaded experts + per-graph scheduling. To move decode, cut bytes read
  per token (KV quant frees VRAM -> more experts on GPU) or cut CPU expert work.
- Next lever: asymmetric KV (-ctk q8_0 -ctv q4_0) frees VRAM for more GPU experts.

### ALL-MODEL SWEEP (llama-bench -p512 -n128 -r2, RDNA2)
| model | arch | MoE | offload | pp512 | tg128 |
|---|---|---|---:|---:|---:|
| gemma-4-E2B ROCMFP4 | gemma4 | no | full GPU | 1643.9 | 131.3 |
| ornith-9b ROCmFPX | qwen35 | no | full GPU | 438.4 | 64.1 |
| Qwen3-30B i1-Q2_K | qwen3moe | yes | cpu-exp 7/48 | 272.1 | 69.6 |
| Laguna-XS.2 IQ4_XS | laguna | yes | cpu-exp 21/40 | 389.2 | 39.9 |
| Qwen3.5-35B Q4_K | qwen35moe | yes | cpu-exp 22/40 | 323.6 | 33.7 |
| GLM-4.7-Flash | deepseek2 | yes | cpu-exp 27/47 | 357.6 | 24.9 |
| Saluki-27B IQ2-mix-MTP | qwen35 | no | full GPU | 178.4 | 21.8 |
| L3.2-8X3B Q8_0 | llama | yes | cpu-exp 16/28 | 423.0 | 11.3 |
| qwen3.8 reap-288 Q4_K | qwen4exp | yes | all-CPU | 60.1 | 10.4 |
| ornith-35b ROCmFPX | qwen35moe | yes | cpu-exp 22/40 | 207.6 | 3.6 |
| Qwen3.6-27B BF16-MTP Q5_K | qwen35 | no | over-budget | 97.2 | 2.8 |

### ACTIONS
- [ ] Test asymmetric KV (-ctk q8_0 -ctv q4_0) on the 35B; measure decode + context.
- [ ] fp6/fp3 native MMVQ (finish ornith-35b fast path).
- [ ] Long-context (>=118k) sweep with quantized KV.
