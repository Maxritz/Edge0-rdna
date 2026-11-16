# Edge0 / Strata on RDNA2 (gfx1031) — full audit and optimisation plan

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
