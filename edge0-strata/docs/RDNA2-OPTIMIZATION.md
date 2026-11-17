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


### 8h. A stage: the spills were the cost, occupancy is measured DEAD - single-buffered staging is the default

The A-stage tile double-buffered BOTH its weight staging (`2*256*80 = 40,960 B`, 80 % of the 51,456 B dynamic smem)
and its activation staging.  Single-buffering them (`lat_q3k_fused_gu_kernel<SINGLE>`, `STRATA_A_SINGLE_BUF`) leaves
`WROWS`, `kItems`, `ra`, `rb`, the dp4a body and the accumulation order untouched - only where the staged bytes sit
changes - so the product is bit-identical (verified: the 32-token output string is identical across both arms and both
repeats).

    arm                                regs  scratch   smem    blocks/cu  gate/up ms        prefill tok/s
    two buffers (control)              213     96     51,456       1       4,498 / 4,474     301.23 / 300.69
    single buffer  <- THE DEFAULT      164      0     25,856       1       4,317 / 4,304     302.51 / 302.64
    single + __launch_bounds__(512, 8) 128     96     25,856       2       4,476 / 4,474     301.41 / 301.27

Three conclusions, in order of how much they should change future work:

1. **The gain is the spills, not the occupancy.**  Single-buffering removes all 96 B of scratch (213 -> 164 regs) and
   measures `gemm gate/up` **-3.9 %** (4,486 -> 4,311 ms mean).  That is the whole win and it is what shipped.
2. **Occupancy is measured DEAD.  Do not chase it again.**  `__launch_bounds__(LTHREADS, 8)` on the single-buffer form
   reaches **blocks/cu = 2 and waves/cu = 32 of 64 - a genuine doubling** - and measures 4,476 / 4,474 ms and
   301.41 / 301.27 tok/s, i.e. **-0.2 %, inside noise**.  The register cap that buys the second block re-creates the
   spills (scratch back to 96) and hands the entire spill win back.  Combined with the earlier "regs 153 vs 256 and
   4 vs 6 waves identical" and "-6.2 % instructions = 0 % time", the A stage is now triply shown not to be
   fill/occupancy/issue-count limited at the margins available.
3. **The LDS-cliff hypothesis was WRONG, and this is the measurement that says so.**  The claim was that 51,456 B of
   smem blocks a second block per CU, so dropping under 32 KiB would double occupancy.  Measured: smem 30,976 B
   (2 x that = 61,952 <= 65,536) and then **25,856 B, 39 % of the CU's 64 KiB - and blocks/cu was STILL 1.**  Reducing
   LDS never buys a block; the binding term is the register budget, and it only moves when registers are capped.  So
   the "reduce the live weight rows to cross the cliff" redesign would have crossed nothing.

What is left for this block (~4,300 ms = 32 % of prefill) is therefore the dependent LDS chain and MAC density:
2,115 instructions carry 128 dp4a = **6 % MAC density**, and at 0.71 T MAC/s the stage runs 3.3 % of the measured
21.8 T MAC/s.  Neither fill, occupancy, registers, staging pattern (`STRATA_A_RDNA`/`STRATA_A_NAT` flat at
4,444-4,509) nor bytes explains that; an instruction-mix change (hoisting the per-4-value rescale to 16-value blocks,
and attributing the 72 `v_div_scale_f32` in an int8 body) is the untested head of the fishbone.

Gates for the shipped default: 4k `301.23/300.69 -> 302.51/302.64`; 8k chunk 832 `341.43/340.00 -> 343.94/343.31`;
512K + MTP exit 0, tokens `715 1822 363 313`, cache 872 slots; 32-token output identical in every arm.


### 8i. The D/U chain, attributed - and a WITHDRAWN claim about its two "hc read" rows (see 8k)

> **SUPERSEDED IN PART - see 8k.**  This section's label mapping and the D/U pricing stand.  Its conclusion that the
> two hyper-connection rows are "a FIXED per-layer cost, independent of tokens" is **wrong**: it came from reading the
> ledger's *marks* column as milliseconds.  The rows are ordinary GEMMs that scale with M and run at ~48 % of the fp16
> peak.

The last large block with no measured hypothesis.  First the labels, because three of the four were wrong:

| ledger row | real phase | what it actually brackets |
|---|---|---|
| `gather` | `kPfGather` | `quantize_q8_1_rows(mixed)` + the **D** projection |
| `gemm down` | `kPfGemmD` | `quantize_q8_1(H)` + the **U** projection (the source comment says so) |
| `hc read: down+silu` | `kPfHcDown` | the HC **down projection** D->LR + silu (not a read) |
| `hc read: up` | `kPfHcUp` | the HC **up projection** LR->D + mix (not a read) |

Geometry: `N=2560, HC=4, D=N*HC=10240, LR=320, K=10, NE=512`.  `STRATA_D_PROF` and `STRATA_DU_PROF` give the
projections their own event pairs (both only exist on the non-fused `STRATA_LATENT_FUSED_DU=0` path, which measures the
same as the default: `gemm down` 1,079 vs 1,080, `gather` 906 both).

    block                     MAC         ms     T MAC/s   % of 21.8 T MAC/s peak
    D projection            5.15 T       850       6.06          27.8 %
    U projection            6.44 T     1,056       6.10          28.0 %
    HC down                 0.64 T       768       0.84           3.8 %
    HC up                   0.64 T       768       0.84           3.8 %
    A stage (gemm gate/up)  3.19 T     4,311       0.74           3.4 %

**D and U are the best-run blocks in the engine - 28 % of the dp4a peak, i.e. 8x the A stage's MAC rate.**  They are
NOT the problem and should be left alone; U is also labelled `gemm down`, which is why 1,080 ms of real, efficient
work kept being read as a candidate.

**The HC pair is the real surprise: it is a FIXED cost per layer-chunk.**  Same config, three chunk sizes:

    chunk    layer-chunks    hc down ms    hc up ms    per layer-chunk
    256          768           1,536        1,536        2.0 / 2.0
    512          384             768          768        2.0 / 2.0
    1,024        192             384          384        2.0 / 2.0

A 4x change in tokens per chunk leaves them EXACTLY unchanged.  So they cannot be GEMMs scaling with M (the down
projects D=10240 -> LR=320, up the reverse).  They are weight-stream / grid-fill bound: 10240 x 320 = 3.28 M weights
per layer-chunk, and a 64-row x 128-col tile gives a grid of only ~20 blocks - one per CU, 16 waves.  1,536 ms a
prefill at chunk 512, and it HALVES for every doubling of the chunk, which is part of why chunk 832 beat 512.

**Both S23 fusions are gfx11-only and refuse on gfx1031** - `STRATA_PF_HCDOWN=1` fails hard with "could not run the
hyper-connection projections", and `STRATA_HC_UPMIX=1` changes nothing (768/768 identical).  So there is no shipped
switch that touches this on RDNA2; it is an RDNA2 gap, not a tuning miss.

**Reframing the A stage (32 % of prefill).**  It is NOT 29x from its ceiling on the terms that were being used.  It
must pull 60,338 expert blobs x 1.16 MB = **70 GB at chunk 512, which is 2.5 s at the measured 28.0-28.1 GB/s link**,
against 4,311 ms measured - i.e. **58 % link duty**.  At chunk 832 it is 49.3 GB = 1.76 s against ~4.3 s, 41 % duty.
So the stage is a streaming problem within ~2x of its floor, which is exactly why fill, occupancy (32/64 waves
measured, -0.2 %), registers, smem and instruction mix all measured flat: they are not what it is waiting on.

**What this leaves, ranked by measured size and by what is now actually known:**

1. **A stage non-overlap ~1.8 s** - the bytes need 2.5 s, it takes 4.31 s, and the visible `wait copy` is only 623 ms,
   so the stream stalls *inside* the A stage (2,163 launches, each waiting on its own 32 experts).  Deeper ring/pipe
   was measured neutral at chunk 832, so the fix is launch/consumption order, not depth.
2. **HC down+up 1,536 ms at 2.0 ms/layer-chunk** - needs grid parallelism (K-split or a 32-column N tile to turn ~20
   blocks into ~80), or the gfx11 fusion ported down.  It is the cleanest bounded target left: the work is fixed and
   the current grid is 20 blocks on 20 CUs.
3. D and U: leave them.  28 % of peak is not where the 3.9 s lives.

Neither 1 nor 2 is a constant or a flag; both need a kernel edit, which is why this section stops at the attribution.


### 8j. The 4k chunk cap was 15 % slow - the cap is only right for the cache it was measured at

The 0.28 cap forced the automatic 4k chunk to 512 tokens.  Its own comment records why: at a **552-slot** cache a
1,024-token chunk streamed every expert of the pack twice (24,024 blobs) against 512's 62,558, i.e. 203.31 / 201.66
versus 244.16 / 244.14 tok/s.  That reasoning is sound - for a 552-slot cache.  This box now gets **1,472 slots** at
4k on the borrowing path, and there the optimum is just under expert saturation:

    4k, borrowing, cache 1,472, 4063-token prompt, alternating pairs
    chunk   cache slots   blobs    hc-down ms   prefill tok/s
    512       1,472      62,313      768        302.80 / 302.75 / 303.28 / 301.27
    576       1,472      59,447      768        300.45
    640       1,472      56,105      672        319.52
    704       1,472      50,461      576        332.12 / 332.43
    768       1,472      49,683      576        334.21
    832       1,472      43,774      480        349.95 / 351.02 / 351.47
    896       1,472      43,851      480        352.90
    960       1,472      43,068      480        346.14

Bigger wins on BOTH terms at once: **30 % fewer blobs** and the fixed per-layer-chunk hyper-connection cost **halves**
(768 -> 480 ms, from section 8i it is 2.0 ms a layer-chunk and does not scale with tokens).  The cache is not hurt
because the borrowing path keeps it at full size.  832 is within 0.8 % of 896 and equals the own-buffers value, so both
regimes now agree on one number.

The cap is therefore conditional on the cache it was measured against: `832` when the cache has >= 1,000 slots, `512`
below that (which preserves the original 552-slot result).  4k default: **301.27 / 303.28 -> 351.02 / 351.47 (+16.1 %)**;
8k unchanged at 344.02; 512K + MTP exit 0 with tokens `715 1822 363 313`.

This is the third time the largest available win was a *cache-vs-chunk* admission decision rather than a kernel
(sections 8a-8h: ring depth, the VRAM cliff, the borrow flag, the own-buffers chunk, now the borrowed chunk).  The
pattern to remember: on this box the expert cache is worth more than the bytes, and the chunk is the variable that
trades one for the other - so it must be chosen against the cache size that will actually be granted, never as a
constant.


### 8k. CORRECTION to 8i - the "1,536 ms hyper-connection pair" was MARKS, not milliseconds; the target is withdrawn

Section 8i claimed the two hyper-connection projections are "a FIXED 2.0 ms per layer-chunk, independent of the token
count".  **That is wrong**, and the error is worth recording because it is a tooling trap, not a modelling one.

The ledger prints two formats.  The component table is

    component                 marks   dev ms    %dev   host ms   %host
    hc read: down+silu          480      658     5.7%      93.1     0.8%

and the compact one-line summary prints `hc read: down+silu 658 (5.7%)`.  A regex of the form
`hc read: down\+silu (\d+)` matches the **marks** column in the table (480) and the **ms** in the compact line (658).
Every "fixed 2.0 ms per layer-chunk" figure in 8i came from that regex, so the numbers were mark counts:

    chunk    layer-chunks    "hcDown" read    marks/layer-chunk
    64          3,072            6,144              2.0
    128         1,536            3,072              2.0
    256           768            1,536              2.0
    512           384              768              2.0
    1,024         192              384              2.0
    2,048          96              192              2.0

A mark count is proportional to the number of layer-chunks *by construction* - 2 marks per layer-chunk - so the
"exactly 2.0 at every chunk size" that looked like a smoking gun was arithmetic on a count, not a measurement.
The task built on it ("the 1,536 ms pair") is therefore also void: **1,536 was 768 layer-chunks x 2 marks.**

The real times, from the dev-ms column:

    chunk    M      layer-chunks   hc down   hc up   sum    per layer-chunk
    256     256         768          775      335    1,110      1.45 ms
    512     512         384          650      300      950      2.47 ms
    832     832         240          657      256      913      3.80 ms
    1,024   1024        192          679      242      921      4.80 ms
    2,048   2048         96          946      280    1,226     12.77 ms

It scales with M exactly as a GEMM should, and at the shipped chunk it is **913 ms = 8 % of prefill**, doing
2 x 832 x 320 x 10240 = 5.45 G MAC in 913 ms = **5.97 T MAC/s**.  Against the 24.6 TFLOP/s fp16 peak (12.3 T MAC/s)
that is **~48 % of peak**; against the best fp16 GEMM ever measured on this box (9.46 TFLOP/s = 4.73 T MAC/s) it is
*above* it.  So the hyper-connection projections are the best-utilised kernels in the engine, and replacing them with a
hand tiled kernel is not a win - the planned ~80-block rewrite was NOT built, deliberately.

### 8l. The denominator was wrong for every bf16/fp16 GEMM in this document

Everything above has been priced against one number, "21.8 T MAC/s", which is the **int8 dp4a** probe.  The dense
projections are bf16/fp16.  Re-priced against the fp16 rate (24.6 TFLOP/s = 12.3 T MAC/s):

    block                     T MAC/s   % of int8 dp4a peak   % of fp16 peak
    D projection               6.06            27.8 %              49 %
    U projection               6.10            28.0 %              50 %
    HC down+up (chunk 832)     5.97            27.4 %              48 %
    A stage gate/up            0.74             3.4 %               not bf16 (int8 dp4a)

**Every dense GEMM in this engine runs at ~48-50 % of the fp16 peak, and the only block below it is the int8 dp4a A
stage at 3.4 % of ITS peak.**  That is the whole story of the prefill gap in one line: there is exactly one anomalously
slow block, and it is the one that uses the int8 dot product.  The dense path is not a target under any denominator.

Consequence for the work list: the 4,063-token prefill at chunk 832 is 11.5 s, of which `gemm gate/up` is 3.74 s (32 %)
and everything else is either at ~50 % of its own peak or is a fixed per-chunk cost already harvested by the chunk
choice.  The remaining 1.22x to 430 tok/s lives in the A stage, not in the dense projections.


### 8m. Decode, attacked - the tail theory is DEAD, the real cost is pool+rings being serial, and there is a 3x intermittent defect

Decode is 30.82-31.38 tok/s against a 62.9 link ceiling.  Three things measured this turn, in order of how much they
matter.

**(1) The per-layer expert-blob latency tail is REFUTED by its own falsifier.**  The tail theory said `pcie/uWait`
(8-17 ms a window) is the PCIe expert copy chain.  `--pcie-frac 0` removes that lane entirely, so uWait must collapse:

    arm                 pcie experts/layer   cpu experts/layer   pcie/uWait ms   waitB ms   window ms   decode
    pcie-frac 0.40           4.83                  8.82            2.56 - 3.43    5.83       73.7      28.4
    pcie-frac 0.10           0.84                 12.23            8.39           0.91       70.7      27.6
    pcie-frac 0              0                    13.25            9.36 - 9.45    0.18       71.4      29.2

With ZERO PCIe experts uWait nearly TRIPLES.  It is therefore not the copy chain - it is the generic wait for whichever
lane supplies the expert, and it grows when the CPU pool is asked to do more.  **The slot that does track the lane is
`waitB`** (5.83 -> 0.91 -> 0.18, monotone with the share).  One-layer-ahead PCIe prefetch, which the tail theory
prescribed, would have optimised the wrong slot - it was not built.

**(2) The real decode cost is that the CPU pool and the ring wait are SEQUENTIAL, not overlapped.**

    arm            window   wait for rings   pool      sum
    0.40           73.73        45.15        22.58    67.7
    0.10           70.74        41.12        24.02    65.1
    0             71.67        41.26        24.76    66.0

rings + pool ~= 66 of a 72 ms window, so the round is the SUM of the two, not the max.  The 22-25 ms of CPU pool
(13.3 ms gate/up + 8.2 ms down, at 24.7-34.5 GB/s i.e. the DDR4 ceiling) is fully exposed every window.  Overlapping it
with the GPU's 41-45 ms of ring consumption would take the window toward max(41, 25) ~= 45 ms: 29.2 -> ~46 tok/s,
which is most of the way to the 62.9 ceiling.  That is the decode target, and it is an architecture change, not a flag.

**(3) There is a 3x intermittent defect, and it is the most damaging thing in decode.**  In about one run in six the
speculative acceptance collapses to 0.06-0.14 and decode falls from ~31 to ~11 tok/s.  Observed this session: 10.88,
12.49, 10.81, 11.60, 11.44 - five occurrences.  Mechanism, from the per-round tallies:

    run                 window choices                          drafts accepted        tok/s
    good (ns1)   T1:6  T2:2 T3:2 T5:1 T6:11            0:8  1:3 2:3 3:2 4:1 5:5      30.91
    bad  (ns5)   T1:39 T2:4 T3:2 T6:13                 0:54 1:2 2:2                 11.44
    bad  (s6f0a) T1:38 T2:10 T3:1 T6:11                0:57 2:1                      10.81

The collapsed runs DO run T=6 windows (11-13 of them) and accept ~6 drafts out of ~65 across those rounds, so the drafts
themselves are near-useless for the WHOLE run - the controller is right to fall back to T=1, and T=1 then feeds the
estimator its own zero.  It is a run-level draft-head defect (state/init), not window tuning.  It is NOT the MTP side
stream (5 good + 1 bad with `STRATA_MTP_SIDE_STREAM=0`) and NOT the expert lane (occurred at pcie-frac 0.40 AND 0).
This costs more than every other decode item combined: the expected decode is ~0.83*31 + 0.17*11 = 27.6, not 31.

**(4) `--pcie-frac 0` is INERT - the +2.8 % / +25.9 % it first appeared to buy were measurement artefacts.  WITHDRAWN.**
First pass said 0.40 -> 28.28 / 28.37 / 28.41 / 28.47 against `0` -> 29.14 / 29.17 / 29.19 / 29.23 at 1k (+2.8 %, ranges
not overlapping), and 12.14 -> 15.28 tok/s at 512K (+25.9 %).  Both failed to reproduce under a cleaner design, same
single binary, alternating, 2 runs per arm each at 1k and at 512K:

    1k, 64 tokens        default(=0) 29.22 / 29.24  (accept 0.783)     --pcie-frac 0.4  29.19 / 29.20  (accept 0.735)
    512K, 12 tokens      default(=0) 15.41 / 15.47  (9 rounds)         --pcie-frac 0.4  15.38 / 15.33  (9 rounds)

Within 0.2 % at both contexts, and the 512K OUTPUT IS BYTE-IDENTICAL across all four arms
(`715 1822 363 313 198 262 514 307 283 220 16 15`).  Two specific errors produced the first pass: the 1k difference was
**batch drift between arms**, and the 512K "+25.9 %" compared a single decode round (the gate emits 4 tokens = one
round) at 12.14 against one at 15.28 - noise, not a signal.  A KL measurement taken during that first pass read mean
1.84e-3 / top-1 100 % (35 positions) with a bit-identical 0.40-vs-0.40 control; since the clean runs are byte-identical
there is no arithmetic difference to gate either.  **So no default was changed**: pcie-frac stays 0.40, and the
"+2.8 % / +25.9 %" figures in the first draft of this section are retracted.

**What this changes about the method:** the 512K gate is not a throughput measurement - 4 tokens is one round.  Any
decode claim at long context must run >= ~10 tokens so there are several rounds, and any claim needs the acceptance
ratio printed beside it, because acceptance moved 0.735 <-> 0.783 between these arms while throughput did not move at all.


### 8n. Decode: the pool is already ~44 % overlapped, and the window cost is INDEPENDENT of the token count

Section 8m left one instruction: overlap the CPU pool with the GPU's ring consumption for ~25 ms a window
(29.2 -> ~46 tok/s).  Both halves of that arithmetic are wrong, and a single falsifier shows it.

**Falsifier: starve the pool and see whether decode notices.**  `--pool-workers 1` makes the pool ~5x slower.  If the
pool were exposed, the window would grow by the pool's growth; if it were fully hidden, the window would not move.

    arm                    decode      accept   pool ms   rings ms   window ms
    default                28.61       0.800     23.15     44.00      73.03
    --pool-workers 1       26.83       0.800     33.21     39.84      78.62
    --pool-workers 1 (2)   26.71       0.800     33.59     39.79      78.92

**+10.0 ms of pool bought only +5.6 ms of window** - so 4.4 of every 10 ms of pool is ALREADY hidden behind the GPU, and
only ~56 % (about 13 ms of the 23 ms) is exposed.  A perfect overlap is therefore worth ~13 ms a window, i.e.
73 -> ~60 ms and 28.6 -> **~35 tok/s, not 46**.  (Note also `rings` FALLING 44.0 -> 39.8 as `pool` rises: that is the
overlap moving, and it is the signature of a partially hidden pool.)

**And the window cost barely depends on the token count at all.**  The collapsed run below runs mostly a ONE-token window
and still pays 81.8 ms:

    run        rounds   accept          tok/round   window ms   window sizes
    good          28    0.800 (36/45)      2.29       73.03     T1:9 T2:2 T3:8 T4:9
    collapsed     63    0.059 (1/17)       1.02       81.78     T1:52 T2:7 T3:2 T4:2

A T=1 window must still run all 48 layers, each with ~8 distinct experts, so 81.8 ms of layer chain buys one token
instead of 2.3.  That is why the collapse is a 3x loss rather than a 2x one, and it reframes the whole decode target:
**62.4 tok/s needs either 4.6 tokens a round at today's ~73 ms, or ~33 ms rounds at today's 2.3** - the per-layer chain
is the wall, not the pool, and not the PCIe lane.

**What the collapse is, precisely.**  It is NOT drafts being generated and rejected.  The collapsed run makes only
**17 drafts in 63 rounds** because the controller latches at T=1 (52 of 63 rounds), so almost nothing is speculated at
all; the 1-of-17 that were drafted were also wrong, so the draft head really is bad for that run.  `DIAGHQ` is present
with `nan=0` in every run and is not the signal.  It is run-level draft-head state, it occurs in roughly 1 run in 3-6,
and it costs 3x.  Nothing in this session has isolated it.

**So neither of 8m's two decode targets survives inspection:** the tail was a misread slot, the pool overlap is already
4/10 done and worth 22 % not 57 %.  What is left for decode is either the per-layer chain (48 layers x ~1.5 ms) or the
draft head's run-level corruption, and both are deep.


### 8o. Context capability (the actual requirement) and WHY long-context prefill falls behind

**Context is verified, not estimated.**  Every row is `run_512k_mtp.ps1 -Embed 1` with turbo3 KV, MTP on,
`--kv-resident 65536`, reserve 300:

    context    result                                    KV pinned   VRAM free after session   expert cache
    256K       exit 0, tokens 715 1822 363 313              -              -                  1,259 / 1.37 GiB
    384K       exit 0, tokens 715 1822 363 313              -              -                  1,056 / 1.15 GiB
    512K       exit 0, tokens 715 1822 363 313           2.34 GiB        4,864 MiB             852 / 0.92 GiB
    512K rot4  exit 0, tokens 248046 198 248045 74455     -              -                  1,049 / 1.14 GiB
    768K       exit 0, tokens 715 1822 363 313           3.52 GiB        2,688 MiB             670 / 0.73 GiB
    896K       CRASH, exit -529697949 = 0xE06D7363      4.10 GiB        2,456 MiB               -
    1M         exit 1, "0 MiB free", MapViewOfFile 1450     -                0                    -

So **768K (786,432) is the verified maximum with VRAM still free (2,688 MiB)**, 512K runs with 4,864 MiB free, and the
requirement "at least 512K with turboquant and roto" is met on both KV modes.  Two defects fall out of the sweep:
**896K dies with an unhandled C++ exception (0xE06D7363) instead of a clean refusal**, and 1M exhausts RAM
(`MapViewOfFile` win32 error 1450 = ERROR_NO_SYSTEM_RESOURCES) after reporting 0 MiB free.  Both bounds are
pinned-KV/RAM, not VRAM - VRAM is still 2.4-2.7 GiB free when it fails.

**Why our long-context prefill gets SLOWER while the reference gets FASTER.**  The reference (16 GiB VRAM + 20 GiB RAM,
Q3_K) reports 430 tok/s at 4K and **778 at 16K**.  Ours, measured this session on a 16,255-token prompt:

    prompt   tok/s    streamed blobs   cache slots
    4,063    351.52      43,774           1,472
    16,255   280.47     243,225             810
    16,255   245.00     347,582           1,017   (chunk 512 instead of 832)
    32,511   273.93     534,109             608

The traffic is the whole story: **4x the tokens costs 5.6x the blobs** (43,774 -> 243,225) and the cache shrinks
1,472 -> 810 at the same time.  At chunk 832 a 16K prompt needs 20 chunks, and each chunk re-streams ~253 of the layer's
512 experts (243,225 / (20 x 48) = 253), so the total is set by tokens x chunks rather than by the expert set.  The
reference's 16 GiB card can use a much larger chunk, so it streams each expert closer to once and its per-token cost
FALLS with length.

The one-pass alternative is priced and impossible here: the prompt buffers cost `160 + chunk*680/1024` MiB, so
chunk = 16,384 would need ~11 GB of a 12 GiB card.  **The long-context prefill gap is a VRAM-size boundary, not a
kernel defect** - 1.22x short at 4K (351.5 vs 430) but 2.77x short at 16K (280 vs 778), and the widening is exactly the
cache/traffic effect above.

**Where the two shortfalls really live, in one line each:**
  * 4K prefill 1.22x: the int8 A stage at 3.4 % of the dp4a peak and ~48 % PCIe duty (8h), everything dense at 48-50 %
    of its fp16 peak (8l) - i.e. about one block, on a card where the expert stream and the expert cache share 12 GiB.
  * decode 2.0x: the per-window layer chain is ~73 ms and nearly INDEPENDENT of T (8n), so 62.4 tok/s needs 4.6 tokens
    a round against the 2.3-3.0 achieved.  The reference's 62.4 is, to within 1 %, this box's own PCIe link ceiling
    (62.9) at zero residency - so the reference saturates its link in decode and we run it at ~48 % duty.


### 8p. The draft-head collapse is a DESYNC at one round, not a slow drift - and the p1 trace proves it

Section 8m said the collapse (3x decode loss, ~1 run in 5) was "run-level draft-head state, nothing has isolated it".
`STRATA_SPEC_TRACE=<file>` writes one line per round - `T accepted sfx round_ms p1..pN` - and diffing a good run against
a collapsed one from the same binary (6 trials, 1 collapsed) locates it exactly.

**The two runs are identical for 19 rounds, then split at one event.**  First 19 window choices, in order:

    good (tr2)  1 1 1 1 1 1 1 1 2 4 3 3 3 4 3 3 4 4 3 | 2 4 4 4 3 3 4 4
    bad  (tr5)  1 1 1 1 1 1 1 1 2 4 3 3 3 4 3 3 4 4 3 | 1 1 1 1 1 1 1 4 1 4 1 1 1 3 1 4 ... 1

At round 19 both runs chose the SAME window (T=3) with the same-looking draft probability (p1 0.701 vs 0.723); the good
run accepted 2 rows and the bad run accepted 0.  That is the whole divergence - after it the runs have committed
different token sequences and never re-converge.

**And after that split the draft head is confidently WRONG, which is the diagnostic.**  The bad run does keep choosing
wide windows whenever p1 clears the 0.5 gate, and accepts **zero** rows on every one of them:

    round   27    29    33    35    37    40    42
    T        4     4     3     4     3     2     4
    p1    0.899 0.596 0.823 0.725 0.759 0.889 0.989
    accepted 0     0     0     0     0     0     0

    wide-round acceptance:   good 1.89/round (36 of 19)     bad 0.89/round (16 of 18), 0 on every wide round after 19

A draft head that reports p1 = 0.99 for a token the target then rejects is not tunned badly - it is conditioning on the
wrong state.  Since the two runs agree for 19 rounds and differ only after an accepted-count mismatch, the mechanism is
a **drafter K/V desync at the first round where the accepted count differs**, after which every draft is conditioned on a
stale prefix.  The `T=1` latch that makes it cost 3x is a *symptom*: with drafts useless, p1 sits below the 0.5 gate in
29 of 47 rounds, so no drafting happens at all and each round buys one token at a full T=1 window price.

**The engine already has the recovery path, and it is the thing to test:**
`STRATA_MTP_CATCHUP_ALL=1` - "catches the drafter's K/V up for the whole verified window, rejected rows included"
(mtp.cpp:74).  If the default catch-up skips rejected rows and that is what desyncs the drafter, enabling it should
either remove the collapse or measurably cut its rate.

**An A/B of that was attempted and is INVALID, recorded so it is not mistaken for a result:** 8 runs per arm were
launched while the context sweep still held the GPU, and all 16 died at
`cudaMalloc(1538625024) ... failed (out of memory): 10676 MiB of 12272 MiB VRAM free` during the weight arena - i.e.
VRAM contention from a concurrent engine, not the flag.  The experiment is queued for when the GPU is free, and must be
scored as a RATE (collapses per N runs), not a single pair, because the defect is stochastic.


### 8q. CATCHUP_ALL does NOT fix the desync (rate test), and what HipKittens-RDNA is actually worth here

**8p's proposed fix is refuted.**  `STRATA_MTP_CATCHUP_ALL` (catch the drafter's K/V up over the whole verified window,
rejected rows included) was run as a RATE test - 10 runs per arm, interleaved, one binary, spec 4 / min-p 0.5:

    arm                collapses (<20 tok/s)    good-run mean
    CATCHUP_ALL off        2 of 10                 23.45 tok/s
    CATCHUP_ALL on         2 of 10                 23.26 tok/s

Identical.  Collapsed runs looked the same in both arms (62-63 rounds, accept 0.062; 48-53 rounds, accept 0.38-0.40).
So the drafter's K/V catch-up over rejected rows is NOT the desync path.  The diagnosis from 8p stands - after the
divergence the draft head reports p1 0.6-0.99 while accepting 0 - but the mechanism lies elsewhere, and this flag is
closed.  (A rate test was the right instrument: a single pair would have "confirmed" either answer.)

**HipKittens-RDNA.**  Worth taking seriously because the fork carries `include/rdna2` explicitly for **gfx1031** - this
exact chip.  But read what it contains before planning around it:

  * upstream HipKittens targets **CDNA3/4/5** (gfx942 / gfx950 / gfx1250) and its tile primitives are *sized to the
    tensor-core units*.  gfx1031 has **no matrix cores**, which is exactly why this engine is `v_dot4`-based.  So there
    is no drop-in GEMM to lift, and no CDNA tile shape to copy.
  * the RDNA2 path is a real ops layer (`register`, `shared`, `memory`, `sync`, `group` + `types`), and its `memory`
    utilities are where the transferable value is:
      - `make_srsrc()` builds a **buffer resource** (with the 13:0 row-stride and the swizzle-enable bit) and loads go
        through `llvm.amdgcn.raw.buffer.load.b128`.  A buffer load carries its own address descriptor, so it sidesteps
        the generic-address computation a `flat_load` needs.
      - a `move<T>` trait exists specifically "to prevent generic addressing" (`ds_`/`global_`/`buffer_` rather than
        `flat_`).
      - the `*_async` helpers are not DMA engines - they are the same instruction with **`s_waitcnt` deferred**, so N
        loads can be issued before one wait.  That is the classic RDNA latency-hiding idiom.
  * there is no `ds_read_b128`; it uses `b64`.

**Why that matters to OUR bottleneck.**  The A stage's body is 2,115 instructions carrying 128 `dp4a` = **6 % MAC
density**, and its weight path is `lat_ld4()`: for every 16 values it does **four 32-bit loads plus four
`__funnelshift_r`** to realign a 550-byte row with 2-byte misalignment.  That is address/realignment work, which is
precisely what `make_srsrc` + `buffer_load_dwordx4` collapses into **one** instruction - and what 8h's flat scatter of
fill/occupancy/registers/smem/staging could never explain, because none of those change the instruction count of the
load path.  The honest status is "strong lead, unmeasured": the right instrument is a standalone probe comparing
`lat_ld4` against a buffer-load form on the same misaligned 550-byte layout, at rows = 1/8/16/31/64, with the engine's
own `isa_extract.py`/`isa_mix.py` counting the body.  That is a probe build, not an engine patch, so it carries no risk
to the shipped binary.


### 8r. MEASURED: the buffer-resource load is 1.7-3.0x faster than the A stage's flat load path

8q's lead is now measured, standalone, with no engine change.  `build/probe_addr.cpp` reads the SAME bytes two ways from
550-byte rows with the engine's 2-byte misalignment, at rows = 1/16/64, 20 blocks x 64 threads (the card filled):

    rows   lat_ld4 (flat/generic)   buffer resource (dword)   speedup
    1        52.71 Gvalues/s            91.37                  1.73x
    16       33.68                      90.91                  2.70x
    64       30.25                      91.60                  3.03x

`lat_ld4` is the shipped path verbatim: five overlapping 32-bit loads + four `__funnelshift_r` per 16 values.  The
buffer form does **exactly the same five dword loads and the same four funnel shifts** - only the addressing mode
differs (`make_srsrc` + `llvm.amdgcn.raw.buffer.load.i32`).  The shape of the result matters as much as the ratio:
the buffer path is **flat across working-set size (90-92 Gvalues/s)** while the flat path degrades 52.7 -> 30.3, i.e.
the generic-address cost scales with the working set, and the descriptor removes it.

**Two false starts, recorded because both would have been reported as a refutation:**
  1. the first probe was folded away - with the row loop iteration-invariant, `ld4` measured 0.001 ms and 243,809
     Gvalues/s, which is not a rate.  The offset now varies with the iteration in BOTH kernels.
  2. the second run had the buffer base at `rows + r*550`, which violates the **256-byte base-alignment** rule for an
     AMD buffer descriptor; that measured the buffer form 38-75x SLOWER.  With the base at the aligned allocation and
     the row as a flat byte `voffset`, the same code is 3x faster.  **On this hardware an unaligned srsrc is not a
     slow path, it is a nonsense path** - the first number was measuring undefined behaviour.

**What this does and does not prove.**  It proves the addressing mode change is worth up to 3x *on this load pattern*.
It does NOT prove the A stage gains 3x: the stage is 2,115 instructions carrying 128 `dp4a`, so the load path is only
part of the body, and the stage is also ~48 % PCIe-duty bound, so a faster load path can be partly hidden.  Between
those two bounds the honest expectation is a fraction of 3x on `gemm gate/up` (3,738-4,311 ms), and it is the first
measured, structural lead on that block after fill, occupancy, registers, smem, staging and bytes all came back flat.

**Port plan (not yet done, and it is a real engine edit):** the kernel receives `blob[g]` per expert; a buffer
descriptor needs the aligned ARENA base plus a byte `voffset`, so `lat_q3k_fused` must also receive the arena base and
its range, and the row offset becomes `blob[g] - base + row*kGuRow + ...` in 32-bit form.  Then `lat_ld4`'s five loads
become five `buf_load_b32` calls.  The gate is the usual one: alternating pairs, the 32-token output check, and
`run_512k_mtp.ps1 -Embed 1` expecting exit 0 with tokens `715 1822 363 313`.


### 8s. The buffer-resource port was ATTEMPTED, REJECTED on correctness, and reverted

8r measured 1.7-3.0x on the load pattern, so it was ported into `lat_ld4` (`STRATA_A_BUF`, default on, both paths in one
binary via a `__device__` flag like `g_a_nat`).  It BUILT and it was WRONG:

    arm                    gemm gate/up      prefill     32-token output
    buffer resource        7,894 / 7,484     101.9 / 113.4   DIFFERS from the baseline string
    STRATA_A_BUF=0 (flat)  8,447 / 8,000     107.9 / 113.2   matches the baseline

The buffer path changes the output, i.e. it reads the wrong bytes, so it was reverted with
`git checkout HEAD -- src/prefill/lat_q3k_fused.cu`.  The shipped single-buffer A stage is intact.

**The timing in that table is worthless and must not be quoted.**  Both arms ran ~3x slow, and so did the REVERTED build
(`gemm gate/up` 7,476 ms, 110.69 tok/s) on IDENTICAL work - cache 1,472 slots, ring 64, 43,774 blobs, the same numbers
as the 351.52 tok/s run (`l4k.log`).  Same work, same cache, 3.2x slower, no competing process, CPU at 2 %: the GPU was
clock-throttled for that whole window.  (I briefly read the flat arm's 8,447 ms as a regression from my branch; the
reverted build's 7,476 ms on identical work shows that reading was wrong.  Nothing in the engine had regressed.)

**What is actually settled:** the descriptor formulation as written is incorrect - most likely the 256-byte page base
plus a nominal 512-byte range is not a valid descriptor for this access pattern (the working probe used the allocation
base and a full-allocation range, which is 4096-aligned and large).  The next attempt should keep the probe's exact
formulation: a 4096-aligned base and a range covering the blob, rather than a per-page descriptor.

**And a hard constraint learned:** no performance number can be trusted on this box without a clock sanity check.  A
canonical run (`--prefill 832`, 4k, 43,774 blobs) is **351.5 tok/s**; anything near 110 on the same workload means the
card is throttled and the window must be discarded.


### 8t. WITHDRAWN: 8r's "buffer-resource loads are 1.7-3.0x faster" was measuring ZERO-FILL, not data

`tools/rdna2/probe_bufcheck.cpp` reads the same bytes two ways and compares the VALUES (no timing, so clock state
cannot corrupt it):

    words compared 8800   differing 8800
    FIRST DIFF at word 0: flat=5e3914ef   srsrc=00000000
    VERDICT: FORMULATION WRONG

**The descriptor path returns zero for every single word.** A buffer load returning 0 for an in-range offset is the
out-of-bounds result, so the descriptor as I built it is not addressing the memory at all.  Which means:

  * 8r's 91.60 vs 30.25 Gvalues/s at 64 rows was not "3x faster loads" - it was **loading nothing**.  Zero-fill is
    cheap, and the accumulated XOR of zeros is a perfectly optimisable loop, which is exactly why the "buffer" arm
    looked flat in working-set size: it never touched the working set.
  * the engine port's WRONG 32-token output (8s) is fully explained: it was multiplying by zeros.
  * the 4096-aligned-base + 1 MiB-range correction does NOT rescue it; the failure is in the descriptor itself.

**So the buffer-resource lead is withdrawn in full.**  The rule that caught it is worth keeping: a timing probe must
compare VALUES against the reference path, not just report a rate.  `probe_bufcheck.cpp` now exists for that and would
have saved two builds and a wrong engine edit.

What would be needed to re-open it: a known-good srsrc example for gfx1031 (the `config` word and the base encoding are
the suspects - `0x110000` was taken from HipKittens without verifying what its OOB_SELECT/stride bits do on RDNA2), and
`probe_bufcheck` must report IDENTICAL before any timing is looked at again.

**This leaves the A stage with no live lead.**  Everything measured on `gemm gate/up` (3,738-4,311 ms, 3.4 % of the
dp4a peak) is flat: fill, occupancy (32/64 waves), registers (164, scratch 0), smem (25,856), staging order, bytes, and
now the addressing mode.  What remains untested there is the instruction MIX itself - 2,115 instructions carrying 128
`dp4a` = 6 % MAC density, with 128 cvt and ~72 `v_div_scale_f32` in an int8 body - which is a different kind of work
from everything above.

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
