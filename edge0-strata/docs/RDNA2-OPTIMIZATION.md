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
| **Stream priority** (`STRATA_MTP_HIP_PRIO=1`, and now `STRATA_MAIN_PRIO` on the main compute stream) | MTP draft: decode 20.60/20.91 vs 21.03/19.27. Main queue high-priority: prefill 254.61 vs 253.98/255.10, decode 20.64 vs 20.86/20.11 | inside noise.  **Mechanism**: HQD priority arbitrates *between* compute queues; prefill has one, so there is nothing to outrank.  It only matters where two compute queues compete, and that case measured neutral too.  Kept on by default (free; can only help under desktop contention, which a clean bench never has) |
| **`STRATA_PF_GEMM=1` / `STRATA_WMMA_GEMM=1`** on gfx103x | `gdn` 3,079 / 3,084 vs 3,075/3,082 | both decline (need the gfx11 WMMA product) |
| **dot2 tile 256x128** (512 threads, 64 acc, half the blocks) | qkv per call **3.28** vs 2.85 (128x128); gate 2.09 vs 1.80 | **worse** — halving the block count loses more parallelism than the staging saves. 128x128 stays; opt-in via `STRATA_GEMM_DOT2_TILE=256` |
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

**Measured, not assumed - the SHIPPED device ELFs dumped and counted.** `tools/rdna2/isa_extract.py` pulls the
`.hip_fat` section (COFF truncates `.hip_fatbin` to 8 bytes) and splits the clang offload bundle into its 79
gfx1031 ELFs; `tools/rdna2/isa_mix.py` histograms a dump per kernel.

| kernel | instructions | **VOPD** | **v_pk_/v_dual** | dp4a | notes |
|---|---|---|---|---|---|
| `latent_u_tile_k` (U *and* the tile D) | 2,509 | **0** | **0** | 512 (20 %) | ~300 VALU on 64-bit address math (99 `v_add_co_u32` + 99 `v_add_co_ci_u32` + 97 `v_lshlrev_b64`) |
| `lat_q3k_fused_gu_kernel` (A stage, 4,562 ms) | 2,115 | **0** | **0** | 128 | 312 scalar FP (`v_fma_f32` 124 + `v_fmac_f32_e32` 88 + `v_mul_f32_e32` 100) and **72 `v_div_scale_f32`** |
| `latent_u_tile_s_k` | 692 | **0** | **0** | 16 | 309 scalar `s_*`, incl. 64 `s_nop` |
| `bf16_f32_dot2_*` (decode GEMVs) | 166-1,171 | **0** | **0** | 1-8 | |

RDNA2's two "more work per instruction" features are **entirely unused** engine-wide - and both are available to
this target: hand-written asm compiled for gfx1031 emitted cleanly,

```
v_dot8_i32_i4   v1, v1, v2, v3     <- 8 int4 MACs in one instruction: the Q4_0 nibble layout DIRECTLY
v_pk_fma_f32    v2, v1, v1, v3     <- two packed fp32 FMAs in one instruction
v_dual_fmac_f32 v3, v2, v1         <- the VOPD dual-issue form itself
```

`v_dot8_i32_i4` ranks first: the Q4 path unpacks 16 nibble bytes to 32 s8 codes and then runs `v_dot4c_i32_i8`
(4 MACs), so it pays for the unpack *and* runs at half density.  dot8 consumes the nibbles as-is at 8 MACs a
instruction - ~2x density, no unpack, on a kernel now on both the latent U and D hot paths.  Expect 25-30 % fewer
instructions; whether that becomes time depends on issue-bound vs the memory rate (latent D/U measures 103-168
GB/s of a 170 GB/s ceiling).

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

### 5.9 `clang -S` does NOT validate inline asm

It writes the asm text into the .s file verbatim; the AMDGPU assembler runs at `-c`/link time.  A check that
compiled each dot mnemonic with `-S` therefore reported `v_dot4_i32_i4` as valid when that instruction does not
exist on this target - a GPU probe would have found it only by crashing on an illegal opcode.  Validate inline asm
with `-c`, always.

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

### 6a. The KL/NLL gate for a table change, and the engine's own noise floor

`tools/rdna2/kl_gate.py` compares the model's own next-token distribution under two latent tables on the same
tokens, via the engine's `STRATA_LOGITS_DUMP` tap.  It aligns by **position, not by record** (a verify window's T
depends on how many draft tokens were accepted: the first version died on "record 2: shape (3, V) vs (4, V)").

Validated with the *same* table twice:

```
bit-identical logits:  no
KL(a||b) mean / max:   1.193e-03 / 1.121e-02
top-1 agreement:       100.000 %  (16/16)
```

**The engine is not bit-reproducible run to run at the logits** - 1.2e-3 mean KL between two identical runs,
top-1 agreeing everywhere.  So the gate's floor is ~1.2e-3, not 0; a table change has to clear that, and top-1
agreement is the more robust statistic at this level.  Likely sources are the scheduling-dependent ones (the CPU
pool's partial contributions to the combine, float atomics) - and it is worth noting that it also means every
"byte-identical output" gate this stretch used was testing the **tokens**, not the arithmetic.

The dump captures the decode windows (positions past the prompt), which is exactly where a decode table lever
lives.

### 6b. The gate, validated and calibrated (the acceptance bound for a table change)

Two runs of the **same** Q8 latent table, `--max-new 128`, compared on the coherent prefix:

| sample | positions | KL mean | p50 | p95 | max | top-1 |
|---|---|---|---|---|---|---|
| `--max-new 16` | 16 | 1.193e-03 | 8.6e-05 | 5.58e-03 | 1.12e-02 | **100 %** |
| `--max-new 128` | 87 | 1.248e-03 | 2.38e-04 | 6.03e-03 | 1.81e-02 | **100 %** |

Two independent samples agreeing on the mean is what makes this a calibration.  The engine reproduces its
*tokens* exactly (87/87 top-1) but is **not bitwise reproducible in its logits** (~1.2e-3 mean KL run to run).

Two requirements for the harness, both learned the hard way:

* **truncate at the first argmax divergence.**  The two runs are separate *generations*; once they pick different
  tokens the contexts differ and later positions score different token streams.  Without truncation this run
  reported mean 4.31e-02 / max 5.285 from a first flip at position 4151 — meaningless.  With it, 87 positions and
  stable numbers.
* `--max-context` must exceed prompt + `--max-new` (4096 cannot hold 4063 + 128), hence 8192 here.

**Acceptance bound for Q4_0:** top-1 agreement is the reliable statistic; KL below ~1.2e-3 mean (6e-3 p95) is the
engine's own noise and cannot distinguish a table change.  A Q4 latent-D table has to clear that, measured on the
same truncated-prefix basis.

### 6c. Q4_0 latent D/U: built, gated, and a measured net loss - with the cause attributed

`tools/rdna2/q4_latent.py` converts the tables (2.49 GB -> 1.42 GB, exact 18/34).  Gate on the Q8 baseline vs the
Q4 table, same prompt, `--max-context 8192 --max-new 128`:

```
first argmax divergence at position 4079   (the Q8-vs-Q8 calibration pair: 4150)
KL mean 9.475e-03  p50 9.18e-04  p95 3.91e-02  max 1.02e-01
top-1 agreement 100.000 % (16/16 on the coherent prefix)      VERDICT PASS
```

**7.6x the calibrated 1.25e-3 floor** - measurable and bounded; the shorter coherent prefix is the honest signal
that behaviour changed.

A/B, alternating pairs, chunk pinned 512:

| arm | decode | prefill |
|---|---|---|
| **Q4** | **21.67 / 21.92 tok/s (+6.3 %)** | 170.08 / 170.10 (-35.4 %) |
| Q8 | 20.62 / 20.40 | 262.76 / 261.82 |

**Decode moves for the first time in the whole stretch** - the diagnosis (decode is bound by latent D/U bytes) is
confirmed.  But prefill loses 35 %, so it stays opt-in.

Prefill ledger, same binary, only the table changed:

| component | fmt=4 | fmt=8 | delta |
|---|---|---|---|
| **gemm down** | 7,465 | 1,144 | **+6,321** |
| **gather** | 4,159 | 973 | **+3,186** |
| gemm gate/up | 4,767 | 4,630 | +137 |
| wait copy | 1,858 | 3,055 | **-1,197** |
| **total** | **23,970** | **15,533** | **+8,437** |

The regression lands in the **expert-stream phases**, not in a latent row, and `wait copy` *falls* - the shape of
a stall charged to whichever phase is open, not of extra expert work.

**CONFIRMED from the code, not inferred.**  The Q4 table silently loses the fast U path:

* `latent_u_tiled` / `latent_u_tiles`: `if (n <= 0 || fmt != 8 || ...) return;` - they refuse fmt 4 outright;
* `latent_u_tile_k` (the tiled kernel they launch): `const size_t wrowb = (size_t)(ELL / 32) * 34;  // a Q8_0
  weight row` - **the block stride is hardcoded to Q8 and the kernel takes no `fmt` parameter at all**;
* the `fmt == 4 ? 18 : 34` branches exist only in the *per-column* kernels (latent.cu:80 and :788), not in the
  tile path.

So with `STRATA_LATENT_DENSE_FMT=4` prefill drops to `latent_u_grouped` at 5,120 rows instead of the tiled kernel -
which is exactly what the two runs' own log line said (`rows=5120` vs `rows=10`; the log prints once per process,
so they had reached *different kernels*).  The fix is to give `latent_u_tile_k` a `fmt` and a Q4 block unpack
(16 nibble bytes -> 32 int8 codes, `(b&0xF)-8` and `(b>>4)-8`, staged into the same int8 layout) and relax the two
guards - not to add a fourth product.  That is a contained kernel change that needs its own verification cycle:
the Q8 path's byte layout must stay untouched, `kl_gate.py` (1.25e-3 floor) is the quality gate, and the 32-token
output plus `run_512k_mtp.ps1` are the regression gates.

It also corrected the geometry a third time: **ELL = 1280**, not 1792 (48 x 55,705,600 B = the file exactly, and
the kernel logs `ELL=1280`), so D/U traffic is **2.67 GB a round**, not 4.7.

### 6d. Q4_0 latent D/U, SHIPPED - and the second win it unlocked

The three edits diagnosed in 6c landed, plus one more hardcode the first three would have left behind:

| defect | where | fix |
|---|---|---|
| `latent_u_tiled` / `latent_u_tiles` refuse fmt 4 | `latent.cu` guards | accept 4 or 8 |
| prefill's U dispatch requires fmt 8 | `prefill.cpp:6621` | `lat_fmt == 8 \|\| lat_fmt == 4` |
| `latent_u_tile_k` row stride `(ELL/32)*34` | `latent.cu:330` | `(ELL/32) * (fmt == 4 ? 18 : 34)` |
| **`latent_u_tile_k` within-row block stride `b * 34`** | `latent.cu:376` | `b * (fmt == 4 ? 18 : 34)` |
| `latent_u_tile_s_k` two-block stride 68 | `latent.cu:577` | `kb * (fmt == 4 ? 36 : 68)` |
| `latent_d_tiled` + prefill's D dispatch require fmt 8 | `latent.cu:727`, `prefill.cpp:5622` | accept 4 or 8 |

The Q4 nibble unpack stages into the **same int8 layout the Q8 path already uses** - byte j gives element
`j = (b & 0xF) - 8` and element `j + 16 = (b >> 4) - 8`, one scale per 32-value block - so the compute path
is untouched.

A/B, alternating pairs, chunk pinned 512, output compared every arm:

| arm | decode tok/s | prefill tok/s |
|---|---|---|
| **F4** | **21.55 / 21.42** | **270.41 / 263.22** |
| F8 | 20.52 / 20.59 | 259.10 / 251.50 |

decode **+4.5 %**, prefill **+4.5 %** - no trade.  Ledger, the two rows the diagnosis named:

| component | fmt=4 now | fmt=4 before | fmt=8 |
|---|---|---|---|
| `gemm down` | **1,139** | 7,465 | 1,175 |
| `gather` | **993** | 4,159 | 1,030 |
| `wait copy` | 2,573 | 1,858 | 3,030 |
| `gemm gate/up` | 4,562 | 4,767 | 4,659 |

Gate: KL mean **7.58e-3** (p50 2.46e-3, p95 2.71e-2), top-1 **100 %** (16/16 coherent prefix) = **6.1x** the
calibrated 1.25e-3 floor.  PASS.

**The second win, and it is bigger at long context.**  Halving the table returns ~1.3 GiB of VRAM, and the expert
cache is sized from free VRAM:

```
Q8: 1.67 GiB free ->   597 expert-cache slots -> chunk auto 256 -> 16 chunks, 105,756 experts streamed
Q4: 2.97 GiB free -> 1,793 slots               -> chunk auto 512 ->  8 chunks,  62,739 experts streamed
```

So the lever pays twice: 18/34 of the D/U bytes, and a larger auto-chunk because the latent table no longer
competes with the expert cache - **-40.7 % expert fetches**.  At `--max-context 8192`: prefill 180.58 -> 264.19
tok/s (+46.3 %), decode 21.93 -> 24.93 (+13.7 %).  Any lever that frees VRAM should now be evaluated on this
axis too, not just on its own stage.
### 6e. The A stage is NOT a data-type problem: fp16 v_dot2 measured 0.54x of int8 dp4a

The A stage (lat_q3k_fused_gu_kernel, 4,631 ms = 31.2 % of prefill) sits at 1.79 TFLOP/s of a 24.6 peak.  Its
ISA (2,115 instructions) is 128 _dot4c_i32_i8 against 128 _cvt_f32_i32, 124 _fma_f32, 100 _mul_f32,
120 _perm_b32 and 72 division instructions - so it looked like int8 spends 4 of every 5 instructions on
conversion and scaling, and that switching the expert gate/up to packed fp16 (_dot2_f32_f16, the instruction the
dense GEMM uses to reach 9.46 TFLOP/s) would remove that overhead.

**Prototyped both inner loops on this card, same work, uild/probe_f16gu.cpp:**

`
int8 dp4a + cvt/scale :  10.911 / 10.924 / 10.929 ms   6006 / 5999 / 5996 Gvalue/s
fp16 v_dot2 (baked sc):  20.339 / 20.311 / 20.357 ms   3222 / 3227 / 3219 Gvalue/s
=> fp16 speedup 0.54x   (stable to 0.1 %)
`

**fp16 is 1.85x SLOWER.**  The reason is instruction density, not overhead: _dot4_i32_i8 is **4 MACs** a
instruction while _dot2_f32_f16 is **2**, so 64 values cost 16 instructions either way at best - and fp16 needs
32.  The int8 path's cvt/fma scaling is already included in that comparison and it still wins.

**And the dot instruction is not the A stage's bottleneck at all.**  Two corrections to the first version of this
section, both from the same probe harness:

1. `v_dot4_i32_i4` and `v_dot4_u32_u4` DO NOT EXIST - the assembler rejects them ("invalid instruction, did you
   mean v_dot4c_i32_i8?").  The dot4 family is int8-only by design; int4 exists only in dot8.  (The first check
   that reported them as valid used `clang -S`, which copies inline asm into the .s VERBATIM and never runs the
   ISA assembler.  Only `-c` validates - trap 5.9.)
2. A rate comparison that holds ILP constant reverses the dot8 result.

```
ILP sweep, 64 values per accumulator per iteration, 40 blocks x 128 threads, gfx1031:
  dp4a  ILP1   7,728 Gvalue/s   0.530 ns / 64 values
        ILP2  12,374            0.331
        ILP4  17,938            0.228
        ILP8  19,700            0.208
        ILP16 21,768            0.188
  dot8  ILP8  22,315            0.184
        ILP16 23,605            0.174
```

**Independent accumulators are worth 2.8x** (7.7k -> 21.8k): the dot instructions are latency-bound at low ILP, and
only at ILP16 does dot8's 8 MACs a instruction finally pay - and then only 1.08x.  So dot8 buys 8 % and would need
int4 activations: not worth a precision change.

**The real finding is the ratio.**  The A stage achieves ~1.1-1.8 TFLOP/s = ~0.9 T MAC/s against a **21.8 T MAC/s**
instruction ceiling: **about 4 %**.  The shipped kernel already carries `acc[4][2][4]` = 32 accumulators, i.e. it is
already at or past ILP16, so the dots are NOT what makes it 4,631 ms.  What is: the Q3_K unpack around them -
`v_perm_b32` 120, `v_lshrrev_b32` 59, `v_alignbit_b32` 58 of 2,115 instructions - plus the LDS staging and the two
`__syncthreads` per 64-value chunk.  **A first reading of this - "so it is an unpack problem" - was then measured and OVERTURNED.**  The Q3_K
unpack (`qs` 2-bit field + hmask + the staged byte order) costs, in isolation, `probe_unpack.cpp`:

```
unpack ILP1 0.080 ns / 64 values    ILP4 0.072    ILP8 0.071    (903 Gvalue/s)
dp4a   ILP16                                        9,932 Gvalue/s   (probe_ilp3, template, corrected accounting)
dot8   ILP16                                       15,443 Gvalue/s   -> dot8 is 1.55x dp4a at equal ILP
```

The unpack is ~11x the dot PER VALUE - and it still cannot be the 4,631 ms, because the unpacked weight tile is
reused by all 64 tokens of the tile: per staged weight value there are 64 tokens of dots, so the unpack is about
**one eighth** of the inner work, not 86 %.  (The earlier `dot8 1.08x` figure was a 4x accounting error - an
accumulator receives NINS x 4 = 16 values, not 64 - corrected above.)

So the 4,631 ms is neither the dots (~4 % of the instruction ceiling) nor the unpack (amortised 64x): it is the
staged-tile traffic and its ordering - two `__syncthreads` per 64-value chunk (40 barriers a tile), LDS bandwidth,
and the five 128-row work items a 1,280-row expert splits into.  **The next measurement must time a WHOLE tile
iteration**, not its parts: four pieces of this stage have now been measured in isolation and each one, multiplied
out, indicts one of the other three.

Probes kept in `tools/rdna2/probe_*.cpp` (f16 vs dp4a, dot8 rate, dot variants, ILP sweep).

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

### 5.9 `clang -S` does NOT validate inline asm

It writes the asm text into the .s file verbatim; the AMDGPU assembler runs at `-c`/link time.  A check that
compiled each dot mnemonic with `-S` therefore reported `v_dot4_i32_i4` as valid when that instruction does not
exist on this target - a GPU probe would have found it only by crashing on an illegal opcode.  Validate inline asm
with `-c`, always.

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

### 6a. The KL/NLL gate for a table change, and the engine's own noise floor

`tools/rdna2/kl_gate.py` compares the model's own next-token distribution under two latent tables on the same
tokens, via the engine's `STRATA_LOGITS_DUMP` tap.  It aligns by **position, not by record** (a verify window's T
depends on how many draft tokens were accepted: the first version died on "record 2: shape (3, V) vs (4, V)").

Validated with the *same* table twice:

```
bit-identical logits:  no
KL(a||b) mean / max:   1.193e-03 / 1.121e-02
top-1 agreement:       100.000 %  (16/16)
```

**The engine is not bit-reproducible run to run at the logits** - 1.2e-3 mean KL between two identical runs,
top-1 agreeing everywhere.  So the gate's floor is ~1.2e-3, not 0; a table change has to clear that, and top-1
agreement is the more robust statistic at this level.  Likely sources are the scheduling-dependent ones (the CPU
pool's partial contributions to the combine, float atomics) - and it is worth noting that it also means every
"byte-identical output" gate this stretch used was testing the **tokens**, not the arithmetic.

The dump captures the decode windows (positions past the prompt), which is exactly where a decode table lever
lives.

### 6b. The gate, validated and calibrated (the acceptance bound for a table change)

Two runs of the **same** Q8 latent table, `--max-new 128`, compared on the coherent prefix:

| sample | positions | KL mean | p50 | p95 | max | top-1 |
|---|---|---|---|---|---|---|
| `--max-new 16` | 16 | 1.193e-03 | 8.6e-05 | 5.58e-03 | 1.12e-02 | **100 %** |
| `--max-new 128` | 87 | 1.248e-03 | 2.38e-04 | 6.03e-03 | 1.81e-02 | **100 %** |

Two independent samples agreeing on the mean is what makes this a calibration.  The engine reproduces its
*tokens* exactly (87/87 top-1) but is **not bitwise reproducible in its logits** (~1.2e-3 mean KL run to run).

Two requirements for the harness, both learned the hard way:

* **truncate at the first argmax divergence.**  The two runs are separate *generations*; once they pick different
  tokens the contexts differ and later positions score different token streams.  Without truncation this run
  reported mean 4.31e-02 / max 5.285 from a first flip at position 4151 — meaningless.  With it, 87 positions and
  stable numbers.
* `--max-context` must exceed prompt + `--max-new` (4096 cannot hold 4063 + 128), hence 8192 here.

**Acceptance bound for Q4_0:** top-1 agreement is the reliable statistic; KL below ~1.2e-3 mean (6e-3 p95) is the
engine's own noise and cannot distinguish a table change.  A Q4 latent-D table has to clear that, measured on the
same truncated-prefix basis.

### 6c. Q4_0 latent D/U: built, gated, and a measured net loss - with the cause attributed

`tools/rdna2/q4_latent.py` converts the tables (2.49 GB -> 1.42 GB, exact 18/34).  Gate on the Q8 baseline vs the
Q4 table, same prompt, `--max-context 8192 --max-new 128`:

```
first argmax divergence at position 4079   (the Q8-vs-Q8 calibration pair: 4150)
KL mean 9.475e-03  p50 9.18e-04  p95 3.91e-02  max 1.02e-01
top-1 agreement 100.000 % (16/16 on the coherent prefix)      VERDICT PASS
```

**7.6x the calibrated 1.25e-3 floor** - measurable and bounded; the shorter coherent prefix is the honest signal
that behaviour changed.

A/B, alternating pairs, chunk pinned 512:

| arm | decode | prefill |
|---|---|---|
| **Q4** | **21.67 / 21.92 tok/s (+6.3 %)** | 170.08 / 170.10 (-35.4 %) |
| Q8 | 20.62 / 20.40 | 262.76 / 261.82 |

**Decode moves for the first time in the whole stretch** - the diagnosis (decode is bound by latent D/U bytes) is
confirmed.  But prefill loses 35 %, so it stays opt-in.

Prefill ledger, same binary, only the table changed:

| component | fmt=4 | fmt=8 | delta |
|---|---|---|---|
| **gemm down** | 7,465 | 1,144 | **+6,321** |
| **gather** | 4,159 | 973 | **+3,186** |
| gemm gate/up | 4,767 | 4,630 | +137 |
| wait copy | 1,858 | 3,055 | **-1,197** |
| **total** | **23,970** | **15,533** | **+8,437** |

The regression lands in the **expert-stream phases**, not in a latent row, and `wait copy` *falls* - the shape of
a stall charged to whichever phase is open, not of extra expert work.

**CONFIRMED from the code, not inferred.**  The Q4 table silently loses the fast U path:

* `latent_u_tiled` / `latent_u_tiles`: `if (n <= 0 || fmt != 8 || ...) return;` - they refuse fmt 4 outright;
* `latent_u_tile_k` (the tiled kernel they launch): `const size_t wrowb = (size_t)(ELL / 32) * 34;  // a Q8_0
  weight row` - **the block stride is hardcoded to Q8 and the kernel takes no `fmt` parameter at all**;
* the `fmt == 4 ? 18 : 34` branches exist only in the *per-column* kernels (latent.cu:80 and :788), not in the
  tile path.

So with `STRATA_LATENT_DENSE_FMT=4` prefill drops to `latent_u_grouped` at 5,120 rows instead of the tiled kernel -
which is exactly what the two runs' own log line said (`rows=5120` vs `rows=10`; the log prints once per process,
so they had reached *different kernels*).  The fix is to give `latent_u_tile_k` a `fmt` and a Q4 block unpack
(16 nibble bytes -> 32 int8 codes, `(b&0xF)-8` and `(b>>4)-8`, staged into the same int8 layout) and relax the two
guards - not to add a fourth product.  That is a contained kernel change that needs its own verification cycle:
the Q8 path's byte layout must stay untouched, `kl_gate.py` (1.25e-3 floor) is the quality gate, and the 32-token
output plus `run_512k_mtp.ps1` are the regression gates.

It also corrected the geometry a third time: **ELL = 1280**, not 1792 (48 x 55,705,600 B = the file exactly, and
the kernel logs `ELL=1280`), so D/U traffic is **2.67 GB a round**, not 4.7.

### 6d. Q4_0 latent D/U, SHIPPED - and the second win it unlocked

The three edits diagnosed in 6c landed, plus one more hardcode the first three would have left behind:

| defect | where | fix |
|---|---|---|
| `latent_u_tiled` / `latent_u_tiles` refuse fmt 4 | `latent.cu` guards | accept 4 or 8 |
| prefill's U dispatch requires fmt 8 | `prefill.cpp:6621` | `lat_fmt == 8 \|\| lat_fmt == 4` |
| `latent_u_tile_k` row stride `(ELL/32)*34` | `latent.cu:330` | `(ELL/32) * (fmt == 4 ? 18 : 34)` |
| **`latent_u_tile_k` within-row block stride `b * 34`** | `latent.cu:376` | `b * (fmt == 4 ? 18 : 34)` |
| `latent_u_tile_s_k` two-block stride 68 | `latent.cu:577` | `kb * (fmt == 4 ? 36 : 68)` |
| `latent_d_tiled` + prefill's D dispatch require fmt 8 | `latent.cu:727`, `prefill.cpp:5622` | accept 4 or 8 |

The Q4 nibble unpack stages into the **same int8 layout the Q8 path already uses** - byte j gives element
`j = (b & 0xF) - 8` and element `j + 16 = (b >> 4) - 8`, one scale per 32-value block - so the compute path
is untouched.

A/B, alternating pairs, chunk pinned 512, output compared every arm:

| arm | decode tok/s | prefill tok/s |
|---|---|---|
| **F4** | **21.55 / 21.42** | **270.41 / 263.22** |
| F8 | 20.52 / 20.59 | 259.10 / 251.50 |

decode **+4.5 %**, prefill **+4.5 %** - no trade.  Ledger, the two rows the diagnosis named:

| component | fmt=4 now | fmt=4 before | fmt=8 |
|---|---|---|---|
| `gemm down` | **1,139** | 7,465 | 1,175 |
| `gather` | **993** | 4,159 | 1,030 |
| `wait copy` | 2,573 | 1,858 | 3,030 |
| `gemm gate/up` | 4,562 | 4,767 | 4,659 |

Gate: KL mean **7.58e-3** (p50 2.46e-3, p95 2.71e-2), top-1 **100 %** (16/16 coherent prefix) = **6.1x** the
calibrated 1.25e-3 floor.  PASS.

**The second win, and it is bigger at long context.**  Halving the table returns ~1.3 GiB of VRAM, and the expert
cache is sized from free VRAM:

```
Q8: 1.67 GiB free ->   597 expert-cache slots -> chunk auto 256 -> 16 chunks, 105,756 experts streamed
Q4: 2.97 GiB free -> 1,793 slots               -> chunk auto 512 ->  8 chunks,  62,739 experts streamed
```

So the lever pays twice: 18/34 of the D/U bytes, and a larger auto-chunk because the latent table no longer
competes with the expert cache - **-40.7 % expert fetches**.  At `--max-context 8192`: prefill 180.58 -> 264.19
tok/s (+46.3 %), decode 21.93 -> 24.93 (+13.7 %).  Any lever that frees VRAM should now be evaluated on this
axis too, not just on its own stage.
### 6e. The A stage is NOT a data-type problem: fp16 v_dot2 measured 0.54x of int8 dp4a

The A stage (lat_q3k_fused_gu_kernel, 4,631 ms = 31.2 % of prefill) sits at 1.79 TFLOP/s of a 24.6 peak.  Its
ISA (2,115 instructions) is 128 _dot4c_i32_i8 against 128 _cvt_f32_i32, 124 _fma_f32, 100 _mul_f32,
120 _perm_b32 and 72 division instructions - so it looked like int8 spends 4 of every 5 instructions on
conversion and scaling, and that switching the expert gate/up to packed fp16 (_dot2_f32_f16, the instruction the
dense GEMM uses to reach 9.46 TFLOP/s) would remove that overhead.

**Prototyped both inner loops on this card, same work, uild/probe_f16gu.cpp:**

`
int8 dp4a + cvt/scale :  10.911 / 10.924 / 10.929 ms   6006 / 5999 / 5996 Gvalue/s
fp16 v_dot2 (baked sc):  20.339 / 20.311 / 20.357 ms   3222 / 3227 / 3219 Gvalue/s
=> fp16 speedup 0.54x   (stable to 0.1 %)
`

**fp16 is 1.85x SLOWER.**  The reason is instruction density, not overhead: _dot4_i32_i8 is **4 MACs** a
instruction while _dot2_f32_f16 is **2**, so 64 values cost 16 instructions either way at best - and fp16 needs
32.  The int8 path's cvt/fma scaling is already included in that comparison and it still wins.

**And the instruction that WOULD have beaten dp4a on paper does not either.**  _dot8_i32_i4 is 8 int4 MACs a
instruction - twice dp4a - so it was the last instruction-level candidate.  Prototyped the same way
(\uild/probe_dot8_rate.cpp\, int4 codes x int4 activations, same 64 values, same scaling):

\int8 dp4a  (4 MAC/instr) :  11.088 / 10.939 ms   5910 / 5991 Gvalue/s
int4 dot8  (8 MAC/instr) :  17.094 / 17.116 ms   3834 / 3829 Gvalue/s
=> dot8 speedup 0.65x   (its paper ceiling, with this scaling, is 1.33x)
\
**dot8 is 35 % slower than dp4a despite twice the MACs**: on gfx1031 it does not issue at dp4a's rate.  So the
quantized dot product is at its hardware ceiling in the shipped kernel:

| instruction | MACs/instr | vs dp4a |
|---|---|---|
| \_dot4_i32_i8\ (dp4a, shipped) | 4 | **1.00x** |
| \_dot2_f32_f16\ | 2 | 0.54x |
| \_dot8_i32_i4\ | 8 | 0.65x |

**There is no instruction-level lever left in the A stage.**  Its 1.79 TFLOP/s cannot be fixed by changing the
dot product, and the '5-8x headroom' figure is withdrawn as an artefact of comparing an int8 kernel against an
fp16 peak that is not reachable for this data type.  What remains is reuse/shape - a roofline for THIS kernel's
achieved MACs against its staged bytes - and that is a kernel project, not a tweak.

Three further strikes against the fp16 route, each independently fatal:
* Q3_K to fp16 doubles the STAGED bytes (16 int8 codes = 16 B, 16 halves = 32 B), so the LDS tile must halve,
  which cuts the weight reuse the stage depends on;
* the dense GEMM's 9.46 TFLOP/s is not transferable evidence - its weights are fp16 ON DISK, so it never pays a
  dequant pass, and it has 128x128 reuse against this stage's 64 rows;
* the arithmetic would change for no measured gain.

So **the A stage is already on the densest instruction available for its data type.**  Whatever is missing is
reuse/shape, not the data type - and the ceiling for THIS shape has never been measured (a roofline over its
achieved MACs and staged bytes), so the '5-8x headroom' claimed earlier was an inference from a different kernel
and is withdrawn.

### 6f. Decode's per-layer fixed cost is a producer wait, NOT node overhead (measured)

Decode re-baselined on the current binary (`STRATA_VERIFY_PROFILE=1`, 64 generated tokens, 29 rounds):

```
round phases   window 71.178 ms   draft_next 5.467   (76.690 ms/round)
decode         64 tokens in 2334.5 ms -> 27.41 tok/s
window         hc1 norm 12.712 (T2) -> 14.398 (T4)
```

The 22.0 ms / 86.65 ms figures quoted earlier in this session are STALE - three fixes (ring depth, chunk-aware
ring, VRAM-cliff guard) changed residency and the window since.  Fit the slot across window widths:

```
slope      = (14.398 - 12.712) / (4 - 2) = 0.843 ms per extra token
intercept  = 12.712 - 2 x 0.843          = 11.03 ms per window (48 layers) = 0.230 ms/layer FIXED
```

93 % of the slot does not depend on the token count.  A 2,560-element sum-of-squares in 0.23 ms is impossible as
work for ~10 KB.

**What it is NOT**, measured with the engine's own `STRATA_DUMMY_NODES=k` (k empty kernel nodes per layer):

```
nodes 0   window 70.868 ms
nodes 4   window 71.519 ms   -> 192 nodes  = 3.4 us/node
nodes 8   window 71.950 ms   -> 384 nodes  = 2.8 us/node
```

**One dependent kernel node costs ~3 us.**  The 0.230 ms/layer fixed cost is ~25x that, so it is not dispatch or
launch overhead - and fusing the norm into a neighbouring kernel would buy ~9 us/layer, about 4 % of the slot.
It is a genuine WAIT on whoever produces the norm's input (the CPU pool's rows / the doorbell), so the fix is to
make that producer deliver earlier, not to merge kernels.

`qkv gemv` (+12 %) and `hc1 up` (+12 %) show the same sublinear shape, so the same diagnosis applies to at least
three decode slots: the window pays ~11 ms per layer of serialisation around ~2 ms of work.

### 6g. The `hc1 norm` slot is an ACCOUNTING ARTEFACT - the target is withdrawn

Four shipped variants, environment only, decode 64 tokens, `STRATA_VERIFY_PROFILE=1`.  `hc1 norm` appears twice per
window-width row (the GDN and QSA brackets); both are listed:

| arm | `hc1 norm` (GDN + QSA) | decode |
|---|---|---|
| default | 11.5+5.2 -> 14.8+5.9 ms | 27.09 tok/s |
| `STRATA_QFUSE=0` | 12.6+5.5 -> 14.9+6.0 | 27.09 |
| `STRATA_QFUSE_HC_LOCAL=0` | 12.6+5.5 -> 14.9+6.0 | 27.14 |
| **`STRATA_GR_UP_WIDE=1`** | **0.200+0.067 -> 0.192+0.069 (67x less)** | **25.62 (SLOWER)** |

The slot collapses 67x while total decode gets *slower*.  So the ~18 ms is not that kernel's work: the label captures
whatever sits at that point in the layer sequence, and changing the kernel moves the LABEL, not the time.  This is the
third stacked-slot problem in this profiler (slot 32 was once charged the latent D stage; slot 30 and slot 32 both
begin at stamp 16, so `hc1 norm` can contain the D stage outright).

Consequences, all of which should have been applied before spending a turn instrumenting it:
  * "hc1 norm is 18-21 ms of a 71 ms window" describes a LABEL, not a kernel - optimising it was never going to
    recover that time, exactly as fusing it was measured to be worth only ~4 % (0.230 ms vs a 3 us node);
  * the null arm is clean, so the window time itself is real: with the profiler ON the window measures 72.914 /
    72.325 ms and with it OFF 74.167 / 70.241 - the instrument is innocent;
  * what remains unexplained is a per-layer fixed cost (~0.23-0.36 ms) that is not node dispatch (3 us), not the CPU
    pool (moving CPU experts 4.5x leaves the slot flat), not LDS (halving it was neutral), and not the named kernel
    (this probe).  The best-supported remaining candidate is host-visible synchronisation per layer - `wait_flag_ge`
    is `<<<1,1>>>` spinning on a host-mapped word (`verify.cpp:2504`, `:2640`), i.e. uncached PCIe polls, and a
    cross-stream wait was measured at ~45 us.

**Next measurement for it (no engine edit):** `STRATA_HC_SPLIT=1` brackets the read's own phases, and
`wait_flag_ge_stamped` already records t_in/t_out on the `set_pcie_balance` path - read per-wait durations there
before touching the protocol.  The candidate fix is `hipStreamWaitValue32` (HIP-FIRST-AUDIT.md section 2), which
needs an edit and therefore a real dev loop.


## 8. 2026-11-17 - three-way debate, reversed roofline, and the chunk optimum

Three independent analyses (RDNA2 ISA implementer, memory/scheduling, architecture principles) produced nine
recommendations.  Four died on measurement inside the same session.  This section records what survived, what the
hardware arithmetic says the targets actually are, and the two measured wins that came out of chasing it.

### 8a. The prefill ledger CLOSES - the "missing 20 %" was a wrapped log

The memory analysis reported 10,739 ms of named stages against a 13,400 ms timeline and made "reconcile the ledger"
its top item.  The table is printed across four wrapped lines; reading all of them, the 25 stages sum to EXACTLY
13,400 ms.  Nothing is hidden.  What it does show is a larger D/U chain than the notes carried:

    gather 908 + gemm down 1086 + down+silu 654 + up 295 = 2,943 ms (22.0 %)

not the ~2,000 previously attributed to it, so the latent MoE chain is gate/up 4,426-4,675 + D/U 2,943 = **7.4-7.6 s
of 13.4 s (55-57 %)**.  Reading a wrapped log is the same trap class as the profiler labels in section 6g.

### 8b. "Move fewer bytes" is REFUTED as a prefill lever

With the loan held constant (8k, both arms owning), fewer bytes measured SLOWER:

    8k chunk  512: 60,338 blobs -> 287.98 / 296.26 / 301.97 tok/s
    8k chunk 2048: 48,710 blobs -> 263.99 / 264.69

Traffic and time are not monotonically related, because the chunk also sets the prompt reserve
(`160 + chunk*680/1024` MiB) and therefore the expert cache.  A byte argument that ignores the cache is incomplete.

### 8c. The chunk optimum is a TRAFFIC MINIMUM just under expert saturation - 832

8k, own buffers, 4063-token prompt, Q4 table, alternating pairs:

| chunk | cache slots | blobs | prefill |
|---|---|---|---|
| 256 | 1,292 | 99,117 | 205.69 |
| 384 | 1,226 | 75,783 | 246.18 |
| 512 | 1,141-1,149 | 60,338 | 287.98 / 296.26 |
| 640 | 1,064 | 54,320 | 311.91 / 312.43 |
| 704 | 1,026 | 48,863 | 323.76 |
| 768 | 973-988 | 48,171 | 325.73 / 318.88 |
| **832** | **930-950** | **42,514** | **341.17 / 341.43 / 341.72 / 341.75 / 340.00** |
| 896 | 911 | 42,606 | 339.21 |
| 960 | 873 | 41,837 | 332.08 |
| 1024 | 833 | 80,785 | **213.37** |

The 1024 collapse is NOT a cache cliff - 833 slots still function at other chunks.  It is **expert saturation**: at
1024 tokens a chunk routes nearly every one of the 512 experts, so distinct-per-chunk jumps and blobs nearly double
(41.8k -> 80.8k).  The minimum therefore sits just under saturation (~832), and it wins because it improves BOTH
terms at once: 30 % fewer blobs than 512 AND 1.9x the cache slots of 2048.  Shipped as the own-buffers default
(generate.cpp option finalisation); 8k prefill **265.21 / 264.59 -> 341.43 / 340.00 (+28.7 %)**.

### 8d. "Dual submission" / deeper pipelining is NEUTRAL at the chunk optimum

At chunk 832 the exposed DMA is already only 585-623 ms (4.4 %), against 1,360 ms at chunk 512:

| arm | ring | `wait copy` | prefill |
|---|---|---|---|
| baseline | 310 | 623 | 339.01 / 341.75 |
| `--prefill-pipe 2` | 309 | 588 | 341.42 |
| `--prefill-pipe 3` | 299 | 603 | 340.78 |
| `STRATA_PREFILL_RING=1024` | 308 | 585 | 341.61 |
| `STRATA_PREFILL_RING=2048` | 308 | 594 | 341.98 |

All within 1 %.  The ring will not exceed ~310 slots at this chunk regardless of the override, and the chunk has
already absorbed the exposure a second in-flight chunk was meant to hide.  Two chunks in flight has nothing to buy.

### 8e. The A stage: measured geometry, and why occupancy is at most a fifth of the deficit

`STRATA_LAT_Q3K_INFO=1` reports the truth the ISA analysis could not know:

    latq3kinfo gu: grid=(5,32,1) block=512 smem=51456 regs=213 scratch=96 blocks/cu=1 waves/cu=16 of 64

* the grid is `kItems(5) x n_groups(32) x 1` = **160 blocks**, block = **512** threads (not 128 - the "20 waves /
  12.5 % fill" claim assumed 128-thread blocks and a 1-group grid, so its 1.5-3x prediction is unanchored);
* `blocks/cu = 1` because `lat_smem_bytes() = 2*256*80 (weights, 80 %) + 2*64*80 + 256 = 51,456 B`, and 2x that
  exceeds the 64 KiB CU LDS -> **16 waves of 64 = 25 %**;
* the cliff CANNOT be crossed by tuning: `WROWS=256` weight rows x 2 threads/row IS `LTHREADS=512`, so halving both
  leaves waves/CU at 16, and 32 waves needs 512 weight rows = 92 KB of LDS.  It needs a tile redesign;
* `scratch=96` = real spills;
* arithmetic: 3.19 T MAC in 4,509 ms = **0.71 T MAC/s = 3.3 %** of the measured 21.8 T MAC/s dp4a probe.  At 25 %
  occupancy the issue ceiling predicts 5.45 T MAC/s, so **occupancy explains at most 1/7.7 of the deficit** and the
  rest is per-tile dependent latency (spills + the LDS chain), not fill.

Staging pattern is flat and stays closed: 4,509 control / 4,444 `STRATA_A_RDNA=1` / 4,495 `STRATA_A_NAT=0` / 4,489
control.

### 8f. Decode: the speculative window is a real multiplier, and a fragile one

Acceptance is 0.740-0.800, but the adaptive controller picks a window of ONE token in 9-12 of ~30 rounds, throwing the
multiplier away.  Swept (decode 64, prompt64, min-p = the draft acceptance floor):

| arm | decode | tokens/round |
|---|---|---|
| spec 4 / min-p 0.5 (the harness base) | 27.16 / 27.29 / 27.33 / 27.31 / 27.38 / 29.38 / 29.46 / 29.47 | 2.23-2.29 |
| spec 5 / 0.2 | 28.85 | 2.71 |
| spec 6 / 0.2 | 30.20 / 30.24 / 30.25 | 3.14 |
| **spec 6 / 0.3** | **30.82 / 30.88** | **3.00** (accept 0.721) |
| spec 6 / 0.35 | 26.73 | 2.72 |
| spec 6 / 0.4 | 25.02 | 2.29 |
| spec 6 / 0.0 | 26.40 | 3.05 |
| spec 7 / 0.2 | 29.02 | 3.30 |
| spec 8 / 0.2 | 11.78 | 2.21 |
| spec 4 / 0.3 | 26.48 / 26.37 | 2.31 |

So **spec 6 / min-p 0.3 = 30.82-30.88 against 28.2 mean base (+9.5 %)** - the first decode gain in several turns,
and it is configuration, not a kernel.  It is NOT shipped as a default: `0.3` helps spec 6 and HURTS spec 4
(26.4 vs 29.5), and `0.2` guts spec 8 (11.78), so the surface is sharply interactive; and the 512K + MTP gate pins
`--spec 2`, which any global floor would move.  Documented as the decode configuration instead.

**Instability to know about:** the same flags produced 10.88 tok/s once, with acceptance 0.103 (8 of 78) instead of
0.721.  That is the spec path landing in a degenerate state, not timing noise - a separate defect worth chasing.

### 8g. Reversed roofline - what 1000 tok/s prefill and 400 tok/s decode would require

Prefill 1000 tok/s = 4063 tokens in 4.063 s (today 11.9 s at chunk 832):

* traffic at the measured minimum (42,514 blobs x 1.16 MB = 49.3 GB) -> **17.2 GB/s = 61 % of the 28.0 GB/s link**;
  a true single pass (48 x 512 x 1.16 MB = 28.5 GB) would be 7.0 GB/s = 25 %;
* compute 3.19 T MAC at 21.8 T MAC/s = **146 ms**, i.e. 6 % of the dot peak over 4 s;
* so 1000 tok/s is **not physically blocked** - it needs every one of the 25 stages to shrink 3.4x.  Both ceilings sit
  far above the requirement, which is why the deficit is scheduling/tile shape, not arithmetic.

Decode 400 tok/s = 2.5 ms/token:

* one token routes 48 layers x 8 experts x 1.16 MB = **445 MB of weights**;
* 445 MB / 2.5 ms = **178 GB/s sustained**, against a measured **28.0 GB/s** link and **372 GB/s** DRAM;
* the link supplies 6.3 % of that, so **93.7 % must be VRAM-resident = 25.0 GiB** of the 26.66 GiB expert set, on a
  **12 GiB** card.  **400 tok/s is arithmetically impossible here by ~2.1x**, regardless of kernel quality.

The ladder that IS reachable:

| decode | requires | status |
|---|---|---|
| 27-30 | today, link at ~45 % duty | shipped |
| ~45 | remove the per-layer expert-blob latency tail | open, edit |
| **62.9** | the link fully saturated at 0 % residency - **this is exactly the 62.4 reference target** | roof of the current design |
| ~114 | 45 % residency (11.9 GiB of 26.66) + a saturated link | hardware roof |

So the reference targets are not arbitrary: **430 prefill is ~12 % of the prefill roof, and 62.4 decode is the link
ceiling of the current residency design.**  Decode beyond ~114 needs a card with more VRAM or a faster link; prefill
beyond ~1000 needs the A-stage redesign in 8e plus uniform 3.4x.

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
