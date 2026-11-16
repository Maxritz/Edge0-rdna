# RDNA2 ISA Reference (Strata-relevant inventory)

Source: `rdna2-shader-instruction-set-architecture.txt` + `rdna2_arch_summary.txt`
(gfx103x target). "Used" = already in Strata's kernels; "unused" = present in the
docs but not yet exploited. This doc is what the engine should do next — it is not
a general ISA tutorial.

## 1. Dot-product family (VOP3P, §6.3)

| Opcode | Instruction | What it does | Strata use |
|---|---|---|---|
| 19 | `V_DOT2_F32_F16` | `D = S0.f16[0]*S1.f16[0] + S0.f16[1]*S1.f16[1] + S2` | ✅ MMVF fp8-dot2 kernels (decode GEMVs) |
| 20 | `V_DOT2_I32_I16` | signed short dual dot | ⬜ — int16 quant (q8-scale planes in `shared+quant`) |
| 21 | `V_DOT2_U32_U16` | unsigned short dual dot | ⬜ — unsigned quant activations |
| 22 | `V_DOT4_I32_I8` | signed byte 4-dot | ✅ dp4a via `__builtin_amdgcn_sdot4` (int8 experts) |
| 23 | `V_DOT4_U32_U8` | unsigned byte 4-dot | ⬜ — unsigned activations (e.g. some PLE/quant paths) |
| 24 | `V_DOT8_I32_I4` | 4-bit nibble 8-dot | ⬜ — IQ4_NL experts (`moe`/`iq_kernels`) |
| 17/19 | `V_DOT2C_F32_F16`/`V_DOT4C_I32_I8` | clamped (saturating) dot | ⬜ — clamp into the dot, drop separate clamp epilogues |

NOT available on RDNA2: fp8 tensor-core ops (`V_CVT_F32_FP8`, `V_CVT_PKRTZ_E4M3_F32`),
mfma, `sudot4`. FP8 storage must be unpacked via bit shifts in-register (what the
fp8-dot2 kernels do).

## 2. Conversions & data types

- `V_CVT_PKRTZ_F16_F32` (VOP2 opcode 47): **round-to-zero** pack of two f32 → fp16.
  Used for activations. Same contract as STRATA_PF_F16's doc.
- `V_CVT_{U16,I16}_F16`, `V_CVT_F16_{U16,I16}`: 16-bit int ↔ fp16 lane ops,
  cheap integer path for quant epilogues.
- `V_PK_MAD_U16`, `V_PK_ADD_U16`, `V_MUL_F16/FMAC_F16`: packed 16-bit arithmetic
  for epilogues (norms, RMS) — half-rate work, no accuracy impact.
- FP16 has **no FP8 pack op on RDNA2** (verified: no matches in the ISA doc).
- Mode register (§3.5): FP_DENORM can flush or preserve subnormals; FP16_OVFL
  clamps overflowed fp16 to MAX_FP16 while keeping true INF; DX10_CLAMP forces
  NaN→0. Kernels that depend on NaN semantics should check this register.

## 3. LDS — §2.3.1

- **64 banks × 512 entries × 4 B**, 128 KiB per WGP (shared by CU pair), **up to
  64 KiB per workgroup**.
- Two allocation modes: **CU mode** (LDS split upper/lower; two SIMD32s run in
  parallel — faster, but waves can't see each other's halves) vs **WGP mode**
  (one contiguous block, all 4 SIMD32s share). A workgroup always lands inside
  one CU with CU mode.
- 64 atomic units in LDS.
- Bank conflicts: with 64 banks × 512 deep entries, row-stride reads of powers of
  two cause conflict chains; SWIZZLE (xor with the row index) removes them.
  Strata's `fused_gr`/`gdn_rec`/`moe_fused` kernels stage weights/activations
  in LDS — swizzle is the main un-tapped latency lever for `hc1 up`.

## 4. Wavefront size — §2.1

- **Native wave32** on RDNA2. Wave64 = two passes (low 32, then high 32); a half
  can be skipped if EXEC==0, but never both.
- Kernels compiled for wave64 double the instruction issue vs wave32.
- HIP codegen for gfx103x defaults to wave32; explicit `__attribute__((amdgpu_flat_work_group_size(32)))`
  guards against accidental wave64 in the hot kernels (AGEM analytic kernels).

## 5. Wave control & waits — §3.3/§3.4/§3.6

- EXEC mask: per-thread on/off. Instructions can be skipped wholesale when
  EXEC==0 (VALU only; wave32 memory is never skipped). `CBRANCH` skips code
  when the mask is likely zero — free epilogue wins for tail rows.
- `STATUS` bits (skip/exec state) are readable; `IN_BARRIER` reports barrier
  wait — useful for instrumentation, not for speed.
- Counters: VMCNT (loads outstanding), VSCNT (stores), LGKMCNT (LDS/GDS),
  EXPCNT (exports) — the doc's own wait-counter mechanism (the wait-by-count
  alternative to `s_waitcnt`-style polling).

## 6. Memory hierarchy — §2.4

- Channels: L2 → read-only L1 → L0 (per WGP), then **cache-less load
  instructions** that force DRAM fetch and dedupe overlapping requests inside
  the clause. For streaming weight rows (GEMVs), the compiler route through
  `__ldg` already lowers to BUF_LOAD; L2-typed loads suffice.
- Output: write-combine cache for scatter/store; read/write cache with
  **atomic units** that can return the initial value (saves the read-back pass).
- Prefetch: `S_INST_PREFETCH` and wave-launch prefetch depths (1-3 cachelines)
  — instruction-cache tuning only, not data.

## 7. GDS — §2.3.2

- 64 KiB global data share, 32 banks × 512 × 4 B, 128 B/cycle, 32 atomics.
- Append/consume and domain launch-ordered append/consume **in hardware** —
  compaction and queue construction without global round-trips.
- Preload before dispatch, write back after completion.
- Strata use case: expert-index compaction in `copy+combine`'s planning step
  (the 27 ms leader in the profile) — candidate to replace global-memory sync
  passes. PCIe-bound phases are unaffected.

## 8. Buffer loads (ISA doc, VOPC/BUF formats)

- `BUFFER_LOAD_DWORDx2/x4` with index offsets — one load instruction for 4
  weights per lane. VOPC ALU also has vector compares that can write the
  comparison result to **EXEC** (V_CMPX) for predicate-free epilogues.
- M0 holds buffer descriptors for indirect addressing (`{base,size}`) —
  scalar-indexed indirect loads without a vector gather.

## 9. SALU — §2.1 / §3.8

- Scalar ALU operates on **one value per wavefront** (index math, loop tails,
  barrier setup) — off the vector pipes entirely.
- `SCC` doubles as carry-in for extended-precision integer arithmetic and as
  conditional-move/branch selector.
- `V_ADD_TID`, `V_ADD_DW`-style ops make per-lane offsets near-free in fused
  kernels (`hc1 up` lane striding).

## 10. Traps & error handling — §3.10–3.11

- Memory violations (LDS/memory alignment, out-of-range, read-only surfaces,
  GWS abort) set sticky `TRAPSTS.mem_viol`, jump to the trap handler if
  present, else halt. Buffer instructions flag `BUFFER_OOB`.
- Three address-watch registers and `Xnack` translation errors can be trapped
  — useful for debugging bad weights.
- **Practical note:** kernels with interleave-y indexing into LDS risk
  alignment violations after a swizzle/refactor — trap handlers make such
  bugs loud instead of silent.

## Ranked actions for Strata (doc-grounded)

| # | Doc basis | Target stage | Why |
|---|---|---|---|
| 1 | §3 LDS + CU mode | `hc1 up` 22 ms | bank-swizzle the fused-gr staging arrays; move SALU index math off VGPRs |
| 2 | wave64=2 passes | any REG>… kernels | enforce wave32 on hot kernels; check AGEM/GDN against VGPR dumps |
| 3 | GDS §2.3.2 | `copy+combine` planning | global compaction without global sync passes (not the PCIe part) |
| 4 | VOP3P 21/23 | intra-quant epilogues | unsigned int dot where activations are unsigned |
| 5 | VOP3P 24 | IQ4_NL rows | nibble dot for 4-bit experts |
| 6 | VOP2 47 + F16_OVFL/DX10_CLAMP | dot2 kernels | double-check mode state; clamp-for-free via dot2C |
| 7 | buffer loads (DWx4, offsets) | strided GEMV rows | combine: 4 weights per lane per load |
| 8 | CBRANCH/EXEC skips | tails, epilogues | skip epilogue rows when EXEC=0 |

Explicitly **not** in these docs (RDNA3/4-only, do not pursue): fp8 tensor math,
mfma, `sudot4`.
