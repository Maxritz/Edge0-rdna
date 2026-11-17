# RDNA2 optimisation — what we know, and what it was to what it is

Engine: Strata (Edge0, 120B Q3_K latent MoE).  Target: **gfx1031 / RX 6700 XT, 12 GiB, Windows HIP**.
Every number below is measured on that box unless it says otherwise.  The chronological record with the raw
reasoning is `edge-0-analysis-rdna2.md` at the repo root (sections 0.1-0.59); this document is the distilled,
reusable part: the hardware as it actually behaves, the wins with before/after, the things that were tried and
measured **not** to work, the ISA that is used / unused / inert, and the traps that cost real time.

---

## 1. The hardware, as measured (not from the spec sheet)

| quantity | measured | how |
|---|---|---|
| multiprocessors | **20** (i.e. 20 WGPs / 40 CU) | `hipGetDeviceProperties`, `hipInfo` |
| wavefront | **32** (native wave32) | `warpSize` |
| L2 | **3 MB** | `l2CacheSize` — one layer's 6.2 MB HC weights do not fit |
| VRAM | 11.98 GiB usable, ~1.8 GiB free at run time | `totalGlobalMem`, engine's arena log |
| clock | 2424 MHz | `clockRate` |
| **DRAM read** | **~170 GB/s** | `tools/hip/cache_policy_bench`, 678.6 MB region (beyond Infinity Cache) |
| **Infinity-Cache-resident read** | **~480 GB/s** | `tools/hip/q3k_nt_bench`, 64.5 MB (21x L2, fits the 96 MB IC) |
| fp32 peak (derived) | 12.3 TFLOP/s | 40 CU x 64 lanes x 2 x 2.4 GHz |
| **fp16 peak (derived)** | **24.6 TFLOP/s** | same, x2 for packed half2 |

Consequences that drove real decisions: the 3 MB L2 means a layer's weights are always cold, so anything that
assumes residency is wrong; the 96 MB Infinity Cache is why the expert stream reaches 480 GB/s and why
non-temporal loads buy nothing (§3); 24.6 TFLOP/s is the denominator for every "x % of peak" below.

---

## 2. Shipped wins — what it was, what it is

All of these are byte-identical output unless marked, and all but the last keep
`512K + MTP exit 0, tokens 715 1822 1822 1822`.

| # | what changed | what it was → what it is | measured |
|---|---|---|---|
| 1 | **Staging ring default 96 → 80** (`prefill.cpp`, `stage_slots`) | auto-chunk **256 → 512** | prefill **165.92 → 231.39 tok/s (+39.4 %)** |
| 2 | **CPU-rows U on `latent_u_grouped`** instead of `latent_u_matvec` (`verify.cpp`) | per-stage | `u: matvec` **14.60 → 11.05 ms (-24.3 %)**, window -3.7 to -4.1 ms |
| 3 | **`unroll 1` on the A-stage `s` loop** (`lat_q3k_fused.cu`) | 256 VGPR / 28 spills / 4 waves → **153 / 0 / 6** | `gemm gate/up` **5,386 → 4,756 ms (-11.7 %)** |
| 4 | **Activation words hoisted out of the `i` loop** (`lat_q3k_fused.cu`) | 64 → 40 LDS per (s,n) | `gemm gate/up` **4,738 → 4,614 ms (-2.6 %)** |
| 5 | **Packed-fp16 dense GEMM** (`dot2_gemm.cu`, `V_DOT2_F32_F16`) replacing rocBLAS on gfx103x, **128x128 tile** | qkv call **6.94 → 2.84 ms (2.44x)**; 4.09 → **9.46 TFLOP/s** | `gdn` **3,074 → 1,546 ms (-49.7 %)**, prefill **231.4 → 254.7 tok/s (+10.1 %)**, profiler off, chunk 512 auto |

Cumulative prefill on the default config: **165.92 → 261.86 tok/s (1.58x)**.  Decode: **~20 tok/s**, unchanged by
all of the above (see §6 for why).

Win #5 changes the arithmetic (different GEMM accumulation order) and therefore the produced tokens; that was
explicitly approved.  `hip_gemm_f16_io_parity` stays green with rel-L2 **2.1e-4** against FP64 — the same bound
the engine's shipped FP16 path already met.  `STRATA_GEMM_DOT2=0` reverts to the library.  The 512K+MTP run
still exits 0; its tokens change (`715 1822 363 313`), which is the approved consequence, not a break.

### Why #1 was worth 39 %

The auto-chunk scan's binding test is **lend headroom**, not ring fill: `chunk + ring + 128 <= cache slots`.
At 721 slots, `512 + 96 + 128 = 736 > 721` so chunk 512 was rejected and the planner fell to 256.  With ring 80:
`512 + 80 + 128 = 720 <= 721`.  Ring depth itself is flat at a fixed chunk (96/80/112 → 235.34/235.46/237.12)
and **192/256 regress** (187/189), because the ring comes out of the expert cache.  A 3/4-ring hysteresis changed
nothing — the fill was never the binding condition.

### Why #5 was invisible for two turns

The kernel was in the tree and reported "ON", but **never launched**: see §5, trap 1.  Two full A/Bs
(§0.55, §0.56) therefore compared rocBLAS against itself and concluded "no faster, and tiling doesn't matter".
Both conclusions were artefacts of the bug.

---

## 3. Measured non-levers — do not spend time here again

| lever | measurement | verdict |
|---|---|---|
| **Non-temporal / streaming loads** on the expert stream (`q3k_nt_bench`) | stream **+1.4 %**; reuse 4x **-7.3 %**; reuse 16x **-32.8 %** | worse where it matters (a chunk reuses experts) |
| **Cache policy** `glc`/`slc` (`cache_policy_bench`) | 169.8 / 169.8 / 170.2 GB/s | flat |
| **`STRATA_LATENT_U_WARPS`** 16 / 32 | `u: gemv` 17.76 / 20.14 vs 17.36 ms | flat then worse |
| **`STRATA_IO_THREADS`** 8 vs 32 | prefill 229.5/229.7 vs 230.4/229.6; host buckets within 1 % | flat |
| **Single-pass U merge** (one table read for GPU+CPU rows) | arithmetic from the engine's own slots: 35.3 → 41.5 ms | **regression** — the double read hides the pool wait |
| **`gr_up_wide_kernel`** on gfx1031 | `hc1 up` 10.05 → 12.92 ms, decode -13 %, output diverges | slower *and* not bitwise equal |
| **Occupancy attributes** `__launch_bounds__(512,2)`, `amdgpu_waves_per_eu(8)` | 256 VGPR / 28 spills / 4 waves, byte-identical report | inert on this toolchain |
| **`kTileRows` 64 → 32** | 153 VGPR, 0 spills, 6 waves — identical allocation | no effect (and would strand half the warps) |
| **`i`-loop `unroll 2`** | 153 → 160 VGPR | worse |
| **Removing 53 spills in the down kernel** | 256 → 144 VGPR, 4 → 7 waves | `gemm down` 1,144 → 1,145/1,166 ms — **nothing** |
| **Stream priority** (`STRATA_MTP_HIP_PRIO=1`) | decode 20.60/20.91 vs 21.03/19.27 | inside noise |
| **`STRATA_PF_GEMM=1` / `STRATA_WMMA_GEMM=1`** on gfx103x | `gdn` 3,079 / 3,084 vs 3,075/3,082 | both decline (need the gfx11 WMMA product) |
| **dot2 tile 128x128** (64 acc, 4:1 vs 2:1) | **RETESTED after the launch bug was fixed**: qkv 3.17 → **2.84 ms**, gate 1.96 → 1.81; `gdn` 1,702 → **1,545 ms (-9.2 %)**, prefill 249.7 → 251.9 (profiler on), output byte-identical in all four arms | **wins — 128x128 is the shipped tile.**  The earlier "flat" reading was an artefact of §5.1 (the kernel was not launching), which is exactly why a measurement taken through a suspected-broken code path must be re-run |

Two lessons from the bottom rows: **spill count is not predictive** (removing 53 bought nothing, while the
A-stage's 28-spill removal had come with an unroll change that was the actual cause), and **occupancy is not the
limit** for either Q3_K kernel (4 waves and 6 waves measure the same).

---

## 4. RDNA2 ISA — used, unused, inert

Audited by grep over `src/` + `include/` (72 `__builtin_amdgcn_*` occurrences in ~15 files).

**Used, and correct for gfx103x:**

| construct | where | note |
|---|---|---|
| `V_DOT4_I32_I8` / `__builtin_amdgcn_sdot4` | int8 expert kernels | gated on `__gfx1030/1031/1032__` |
| `V_DOT2_F32_F16` (`v_dot2_f32_f16` asm) | `native_bf16.cu` (decode GEMVs), **`dot2_gemm.cu` (dense GEMM)** | the native packed-fp16 form |
| `V_CVT_PKRTZ_F16_F32` | activation packing | RTZ, matches the scalar path's contract |
| `V_PERM_B32` / `__builtin_amdgcn_perm` | quant unpacking | |
| `V_PERMLANEX16`, `V_MOV_DPP` | attention lane movement | |
| `s_waitcnt`, `s_barrier`/wave barrier, fences | staging order | |
| `S_MEMREALTIME` + `S_SLEEP` | persistent HC barrier timeout | |
| wave32 | default | `warpSize == 32` |
| Infinity Cache | implicit | 480 GB/s on a resident stream |

**Unused and worth knowing:** `v_pk_*` packed arithmetic in general (only `v_dot2` is used); explicit
`ds_read*`/`ds_write*` and LDS bank swizzle (LDS is used implicitly through `__shared__`); GDS append/consume.

**Inert on RDNA2 — do not gate on them:** `v_wmma_*` (RDNA3/RDNA3.5 only; the entire gfx11 dense-GEMM path
compiles to nothing here), `__builtin_amdgcn_sudot4`, fp8 tensor ops, mfma.

---

## 5. Traps that cost real time

### 5.1 Device arch macros are invisible to host code — four separate bugs

`__gfx1031__` is defined only in the **device** compile pass.  A host function guarded on it compiles to
`return false`, and the launch/kernel/constants inside it simply do not exist for the host.  This bit four times
in one file:

1. the launcher body guarded on `__gfx1031__` → always `return false`;
2. the tile-size constants inside the same guard → undeclared for host;
3. the **kernel definitions** inside the guard → host cannot reference the symbols;
4. the **launch statement** inside the guard → host compiles it out.

Symptom in every case: the code reports "ON", runs, and produces *the old path's* result — the parity harness
reported rocBLAS's numbers unchanged to six digits.  **Rule: arch-state the decision at runtime, keep every
host-visible symbol and the launch outside any `#if __gfx…`, guard only device function bodies.**

### 5.2 `hipDeviceProp_t::gcnArchName` is empty on Windows HIP

A `strncmp(p.gcnArchName, "gfx103", 6)` test silently refuses.  Use `major`/`minor` (`10.3` = gfx103x) and print
the decision once so it can never be invisible again.

### 5.3 Long A/B batches corrupt late arms

Four separate instances: an arm identical to an earlier one produced **different tokens AND ran 2–3x slower**
(e.g. decode 7.95 vs 20.5 tok/s; `gdn` timings unchanged).  It is not a kernel bug and not ordinary noise.
**Rules: check the 32-token output on every arm; treat a late arm's timing as untrustworthy if its output
diverged; prefer alternating pairs, and for compile-time changes rebuild and re-run the baseline immediately
adjacent.**

### 5.4 Batch-to-batch drift is ±6 % on the A stage

The same binary measured `gemm gate/up` at 4,462 / 4,514 / 4,756 / 4,758 ms across batches.  A separated A/B
"finds" anything from -6 % to +6 %.  Adjacent rebuild-and-rerun is mandatory for small effects (the -2.6 %
hoist was only resolvable that way).

### 5.5 The fp16 parity harness cannot distinguish accumulation orders

`hip_gemm_f16_io_parity` returns rel-L2 ≈ **2.1e-4** for every conforming kernel, because that is dominated by
the **fp16 output format**, not by the sum order.  Two completely different GEMMs report the same six digits.
Use it to bound the format, never to claim a kernel matched.

### 5.6 HIP graph capture forbids allocation and copies

The decode window is captured; `cudaMalloc`/`cudaMemcpy` inside it fails with *"operation failed due to a
previous error during capture"*.  This killed a bandwidth probe mid-investigation.  Do such probes outside the
window, or use pointer metadata only.

### 5.7 The profiler is not free, and its slots collide

Each `gpu_stamp` costs ~17 µs of serialized GPU time; adding 48 stamps per window grew it 116.65 → 120.04 ms.
Slot names also mislead: `verify.cpp` slot 29 is labelled `u: copy` but is where the attention-side HC **up**
lands, and slot 32 was charged the **latent D stage** as if it were `hc1 up` — which is why that item read
18.4 ms for three turns while the up projection was ~2 ms.

### 5.8 Two GPU engines, not one

The copy engine's DMA is separate silicon from the CUs.  Stream/queue priority reorders compute against compute
and cannot touch DMA time — so a high-priority "compute" queue cannot fix exposed transfer time.

---

## 6. Remaining levers, sized and gated

| item | measured now | what it needs |
|---|---|---|
| **A stage** `gemm gate/up` | ~4,600 ms (~27 % of device) | 153 VGPR / 0 spills / 6 waves; NOT register-, spill- or occupancy-limited, and 37 % fewer LDS bought only 2.6 %.  Needs real kernel work, not tuning. |
| **exposed DMA** | **2.98 s** of `wait copy` = 73.5 GB at 24.6 GB/s | staging ≥ 1 layer ahead needs ~166 ring slots (193 MB) vs the 80 that fit on 12 GiB.  Measured trade, not a knob. |
| **decode D/U tables** | ~4.7 GB/round at **103-168 GB/s** of a 170 GB/s ceiling | **Q4_0** halves the bytes → ~20 → ~26 tok/s.  **Approved by the owner; the KL/NLL quality gate does not exist and must be built first.** |
| dot2 GEMM beyond 9.46 TFLOP/s | qkv 2.84 ms at 128x128, 38 % of fp16 peak | the tile scan is not exhausted (256x128, deeper k-slab for fewer barriers, 8-16 accumulators more per thread); each step is one constant and one alternating pair |
| decode dense GEMVs | 18.25 ms | same dot2 machinery applies (decode is a different kernel shape). |

Decode is at the memory rate for its byte volume: hc read ~26.7 ms, latent D/U ~4.7 GB/round, and the pool wait
is already the cheaper of two schedules.  Nothing there moves without fewer bytes.

---

## 7. The method that actually worked

1. **Measure the component, never infer it from an aggregate.**  Every wrong conclusion this stretch came from a
   ratio (FLOPs/time, bytes/time); every right one came from a bucket.  On `gdn` alone, three consecutive
   inferences — instruction, tiling, dequant — were each refuted by the next measurement.
2. **Instrument inside the window.**  `STRATA_PREFILL_TIMING` (ledger), `STRATA_PREFILL_HOSTPROF` (7 host
   buckets), `STRATA_GEMM_SPLIT` (dequant / matmul / widen), `fused_gr_bench`, `q3k_nt_bench`,
   `cache_policy_bench`.  The host window's "52.8 % host grouping" turned out to be 15 ms of grouping and 9.4 s
   of stalls.
3. **Alternate, and check the output every arm.**  Two arms is not enough when drift is ±6 %; separated batches
   are not comparable at all.
4. **A null result is a deliverable.**  Half the levers here are closed with numbers, which is why the next
   session does not have to try them.
5. **State the withdrawal when you were wrong.**  Two of this stretch's claims were withdrawn by their own
   authors (D "at ceiling" from `ELL=2560` instead of 1792; spills as the cause of the A-stage gain, refuted by
   the down-kernel control), and the dot2 "no faster" verdict was an artefact of §5.1.
