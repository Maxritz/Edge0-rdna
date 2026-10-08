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
