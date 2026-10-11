# Edge0 Benchmarks

Measured on: AMD Radeon RX 9070 XT, gfx1201 (RDNA4), 16 GiB VRAM, ReBAR on,
16 threads, 95.9 GiB RAM. Engine: pinned llama.cpp (b11100) + Edge0 patch band,
HIP build `wt/win/build-hip`. All numbers from real runs (no projections).

Reproduce: `llama-bench.exe` / `test-backend-ops.exe` from
`wt/win/build-hip/bin`; tilelang from the local `G:\tilelang-rocm` tree.

## 1. Kernel GEMM (test-backend-ops, ROCm0, m=4096 n=512 k=14336)

| type | before hipBLASLt fix | after fix |
|---|---:|---:|
| f16  | 8.42 TF | **96.33 TF** |
| f32  | 12.44 TF | - |
| bf16 | 7.01 TF | - |
| q4_0 | 82.66 TF | - |
| q8_0 | 80.43 TF | - |
| mxfp4 | 86.19 TF | - |
| q4_K | 75.14 TF | - |
| q6_K | 39.21 TF | - |
| q2_0 | 88.91 TF | - |
| iq4_xs | 87.66 TF | - |
| nvfp4 | 57.87 TF | - |
| q2_K | 2.61 TF | - |

Card fp16 peak ~96 TF. ggml's quantized GEMM is already near peak; the fix
recovered the f16/bf16 path (was falling to a slow fallback with no hipBLASLt
Tensile data). Small-n (decode, bandwidth-bound) unchanged: f16 m4096 n1 k14336
= 611 GFLOPS.

## 2. tilelang GEMM (local tree, gfx1201 WMMA), f16

| shape | ms | GFLOP/s |
|---|---:|---:|
| 4096^3 | 1.385 | 99,250 |
| 8192^3 | 12.356 | 88,990 |
| 512x4096x4096 | 0.234 | 73,530 |
| 1x4096x4096 (GEMV) | 0.127 | 264 |

tilelang f16 hits card peak; matches ggml's quantized GEMM class. No headroom
over ggml for the types we run.

## 3. Qwen3.5-35B-A3B (MXFP4 experts + Q6_K dense) - n_cpu_moe sweep

`llama-bench -p 512,2048 -n 128 -ngl 99 -fa auto`, 16 GiB profile:

| n_cpu_moe | pp512 | pp2048 | tg128 |
|---:|---:|---:|---:|
| 0  | 759.6 | 763.2 | 33.8 |
| 8  | 429.5 | 453.4 | **44.1** |
| 16 | 481.3 | 536.2 | 40.7 |
| 24 | 339.9 | 384.2 | 30.4 |
| 31 | 267.0 | 304.9 | 25.8 |
| 40 | 219.1 | 247.3 | 21.9 |

Prefill is CPU-expert bound (pp2048 247 -> 763, ~3.1x as experts move to GPU).
Decode peaks at n_cpu_moe 8 (44.1); 0 spills to PCIe (33.8).

## 4. Hardware profile targets (same 35B, `-p 512 -n 128 -ngl 99 -fa auto`)

Planner (`gguf_tool.py inspect --vram-gb V --ram-gb 32 --ctx 32768`) picks
`n_cpu_moe`; measured:

| profile | n_cpu_moe | gpu est | cpu est | pp512 | tg128 |
|---|---:|---:|---:|---:|---:|
| 16 GiB / 32 GiB | 16 | 7.9 GiB | 10.4 | 481 | 40.7 |
| 12 GiB / 32 GiB | 25 | 7.93 GiB | 10.39 GiB | 344 | 29.9 |
| 8 GiB / 32 GiB  | 35 | 3.66 GiB | 14.67 GiB | 252 | 23.0 |

## 5. lazy-mode A/B (35B, n_cpu_moe 16)

| mode | pp512 | tg128 |
|---|---:|---:|
| off | 506.9 | 40.88 |
| on  | 498.4 | 40.65 |

No effect on this model (no arch-marked lazy tensors; MoE experts are not marked).
`--lazy-mode on` targets PLE/engram embedding tensors (qwen4exp), not experts.

## 6. Disk-tier candidate plans (12 GiB VRAM / 32 GiB RAM / 32k ctx)

| model | size | arch | experts | plan |
|---|---:|---|---:|---|
| Qwen3.5-35B-A3B Q4 | 18.3 GiB | qwen35moe | 256 | fits, n_cpu_moe 25 |
| Nemotron-3.5-Lightning-30B NVFP4 | 19.8 GiB | nemotron_h_moe | 128 | fits, n_cpu_moe 30 |
| gpt-oss-120b Q8 | 63.4 GB | gpt-oss | 128 | cpu 52 GiB > 32 GiB RAM -> disk |
| DeepSeek-V4-Flash IQ2XXS | 57.5 GB | deepseek4 | 256 | does-not-fit -> disk |
| qwen3.8-flash-next-reap-288 Q4 | 83.8 GB | qwen4exp | 288 | does-not-fit -> disk |

The 32 GiB RAM ceiling means the >32 GiB models must keep experts on disk
(mmap / expert pool), not in RAM.

## 7. Performance trace (perf.rs, 35B 32k, plan n_cpu_moe 16)

- decode = 88.2% of device time, ~26844 us/token (~37 tok/s)
- prompt-eval = 11.8%
- app spans: launch ~7.6 s, wait_ready ~7.5 s, inspect_for_gpu ~0.47 s
- resources at load: cpu ~52%, gpu ~27%, ram 33.7/95.9 GiB, vram 14.5/15.9 GiB

## 8. Notes

- gpt-oss / DeepSeek / qwen4exp expert types are MXFP4 / IQ2_XXS+Q2_K / Q4_K —
  all need the `tt_bytes` -> `ggml_type_size` pool fix to be pool-eligible.
- Qwen3.8-Flash-Next (reap-288, Swift) carry PLE tensors inline
  (`ple_conv1d/key/value/norm`, `per_layer_token_embd`); the standalone
  `*-ngram-embeddings-*` GGUF is an alternative PLE asset, not required.

## 9. Expert pool type coverage (tt_bytes fix)

`tt_bytes()` now defers to `ggml_blck_size`/`ggml_type_size`, so the pool covers
every ggml type with an exact per-expert stride. Verified with
`examples/pool_types.rs` against the real `ggml-base.dll`:

| type | size/blck | bits/wt | old pool |
|---|---:|---:|---|
| Q2_K | 84/256 | 2.625 | skipped |
| Q4_K | 144/256 | 4.500 | skipped |
| Q5_K | 176/256 | 5.500 | skipped |
| Q6_K | 210/256 | 6.562 | skipped |
| IQ2_XXS | 66/256 | 2.062 | skipped |
| MXFP4 | 17/32 | 4.250 | skipped |
| NVFP4 | 36/64 | 4.500 | skipped |
| Q8_0 | 34/32 | 8.500 | skipped |

Per-expert bytes exact (35B MXFP4 557056, 35B Q6_K 860160, gpt-oss MXFP4 4406400,
DSV4 IQ2_XXS 2162688 / Q2_K 2752512). Live pool test (35B, `E0_POOL_MB=4096`,
pool-only): `POOL2 init: tt=120` (was 0), `bypass=0`, hits 327680+, hot 2.3 GB.

## 10. gpt-oss-120b pool coverage vs pool size (pool-only, MXFP4)

gpt-oss-120b Q8_0 (63.4 GB, 36 layers x 128 experts, K=4), `-ngl 99 --n-cpu-moe 32
-c 4096 -fa auto`, no prerouter. Coverage = pool hits / (hits + mmap bypasses).

| pool | tt | fills | hits | bypass | coverage | hot | decode tok/s |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 4 GiB  | 108 | 974  | 81920  | 576535 | 12.4% | 4093 MB  | 10.5 |
| 6 GiB  | 108 | 1462 | 147456 | 504121 | 22.6% | 6143 MB  | 11.0 |
| 8 GiB  | 108 | 1949 | 212992 | 453076 | 32.0% | 8190 MB  | 10.8 |
| 16 GiB | 108 | 3898 | 442368 | 216691 | 67.1% | 16380 MB | 10.7 |

Coverage is ~pool_size / 24 GiB (the distinct-expert working set), linear until it
saturates. A 6-8 GiB pool (product tiers) covers ~1/4 to 1/3 of expert reads. Decode is
flat across pool sizes here because this machine (96 GiB RAM) keeps the whole 63 GB file
in page cache; the pool's value is under real RAM pressure. The pool has no eviction
(fill-once), so a pool smaller than the working set permanently bypasses the overflow to
mmap — an LRU/eviction policy is the next change needed.

## 11. Multi-model `--n-cpu-moe` sweep (RX 9070 XT 16 GiB, 96 GiB RAM, t=16)

`llama-bench -p 256 -n 64 -ngl 99 -fa auto`, decode tok/s (tg64) per `--n-cpu-moe`:

| model | size | arch | experts | 0 | 16 | 24 | 31 |
|---|---|---:|---:|---:|---:|---:|---:|
| Laguna-XS-2.1 IQ3_XXS | 13.0 | laguna | 256 | **112.0** | 44.4 | 33.1 | 27.6 |
| laguna-xs2 Q4_K_M | 20.3 | laguna | 256 | **43.6** | 36.2 | 30.6 | 25.8 |
| GLM-4.7-Flash Q4_K_M | 18.1 | deepseek2 | 64 | **41.9** | 31.1 | 25.3 | 21.0 |
| Qwen3.5-35B-A3B Q4_K_XL | 19.7 | qwen35moe | 256 | 35.0 | **39.1** | 29.3 | 24.5 |
| gpt-oss-120b Q8_0 | 63.4 | gpt-oss | 128 | - | 12.9 | - | - |

Prefill (pp256) follows the same shape — 600-700 tok/s when experts fit on GPU, falling
with `--n-cpu-moe`. Two regimes:
- **Fits on the 16 GiB card** (Laguna-XS-2.1 13 GB, laguna-xs2 20 GB, GLM 18 GB): decode
  peaks at `--n-cpu-moe 0` (all experts on GPU). Laguna-XS-2.1 IQ3_XXS hits **112 tok/s**
  — the fastest config measured on this card.
- **Barely over budget** (Qwen3.5-35B 19.7 GB, 256 experts): `--n-cpu-moe 0` spills and
  drops; a small offload (`--n-cpu-moe 8-16`) is the sweet spot.

gpt-oss-120b thread sweep (n_cpu_moe 32): t=8 59.2 pp / 13.0 tg, t=16 56.5 / 12.9,
t=24 56.5 / 12.7 — thread count barely matters; ~8 threads is fine.


## 12. MTP self-speculative decode (Tiel-Coder-35B-A3B-MTP-APEX)

`qwen35moe` Q5_K, embedded MTP head, `-ngl 99 --n-cpu-moe 24 -c 4096 -fa auto`, 160 greedy
tokens, llama-server. Flags: `--spec-type draft-mtp --spec-draft-n-max N --spec-draft-p-min P
[--spec-draft-ngl 99]`.

| variant | draft acceptance | mean accepted len | decode tok/s |
|---|---:|---:|---:|
| baseline (no MTP) | - | - | **20.74** |
| MTP n-max 6, p-min 0.60 | 0.696 | 2.84 | 8.82 |
| MTP n-max 4, p-min 0.50, draft-ngl 99 | 0.639 | 3.00 | 17.39 |

MTP is correct and accepts well, but is **net-negative** for expert-offloaded MoE: the
verification batch's CPU matmul cost scales with batch size, so the extra verified tokens are
not free. Spec decode only pays when the target is GPU-resident (weights read once per batch).
`--spec-draft-ngl 99` (GPU draft) recovers most of the loss but not to baseline.

## 16. Dense Qwen3.8-27B (qwen35) + MoE engine-path runs (RX 9070 XT 16 GiB)

Correction: `Qwen3.8-27B-WebGGUF-Q4_0` and `Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp` are **dense**
(`qwen35`, 65 layers, 0 experts, 27.32B) - not MoE. `Muse-Glimmer-30B-UD-Q8_K_XL` is also
**dense** (`muse-glimmer`, 0 experts, 32.3 GB). The MoE Qwen3.8s are `qwen4exp` (Flash-Next).

### Dense 27B, full offload vs `-ngl` sweep (llama-bench, `-p 512 -n 128 -fa auto -t 16`)
| model | size | ngl=0 (iGPU/CPU) | ngl=32 (~8 GiB) | ngl=48 (~12 GiB) | ngl=99 (full) |
|---|---:|---:|---:|---:|---:|
| Qwen3.8-27B WebGGUF Q4_0 | 14.63 GiB | 1.68 / 47 | 3.21 / 64 | 5.42 / 150 | **11.09 / 249** |
| Qwen3.8-27B GSQ-RCO IQ3_S-mtp | 11.28 GiB | 2.10 / 42 | 3.49 / 66 | 6.55 / 130 | **29.81 / 755** |

(tg128 tok/s / pp512 tok/s). Dense 27B is memory-bound and needs near-full GPU residency to be
usable; the smaller IQ3_S (11.3 GiB) fits cleanly and is ~2.7x the Q4_0 (14.6 GiB, spill). The
IQ3_S-mtp file carries an embedded MTP head (`blk.N.nextn.eh_proj/enorm/hnorm/shared_head_norm`).

### MoE through the ENGINE path (llama-server + planner n_cpu_moe + --pool-mb 2048 + --flash-attn auto)
| model | arch | experts | quant | n_cpu_moe | decode |
|---|---|---:|---|---:|---:|
| Tiel-Coder-35B-A3B-MTP-APEX | qwen35moe | 256 | Q5_K | 20 | 10.03 |
| Unsloth-Ornith-1.5-35B-A3B-UD-Q4_K_XL | qwen35moe | 256 | Q4_K | 15 | 11.09 |
| ornith-35b-Q8_0 | qwen35moe | 256 | Q8_0 | 26 | 6.98 |

(128 greedy tokens, single run.) These carry the 2 GB pool, which is a net loss on this 96 GiB box
(page-cache-resident mmap is faster than explicit pool reads), so they read below the bare sweeps.

### Simulated memory targets (planner, ctx 4096)
- 8 GiB VRAM / 16 GiB RAM: gpt-oss-120b -> `n_cpu_moe 34` (GPU 5.3, CPU 53.7 GiB).
- 8 GiB VRAM / 10 GiB RAM: `n_cpu_moe` unschedulable; CPU needs ~54 GiB > 10 GiB -> disk-bound
  worst case (the pool/streaming target; not faithfully reproducible on this host, see section 15).
- 12 GiB VRAM / 16 GiB RAM: same class; dense 27B needs ~full residency, so it runs at the
  ngl=48-ish rates above (~5-7 tok/s), MoE offloads experts.

## 19. RDNA2 (gfx1031, RX 6700 XT 12 GiB) - cross-machine

Our patch-band engine, build eb0ef8074, ROCm at D:\Rocm10, models on C:\x, `llama-bench -p 512 -n 128 -ngl 99 -fa auto`.

Qwen3.5-35B-A3B Q4_K (18.32 GiB) `--n-cpu-moe` sweep:

| n_cpu_moe | pp512 | tg128 |
|---:|---:|---:|
| 0 | 435 | 26.75 |
| 8 | 292 | 28.86 |
| 16 | 245 | 31.36 |
| 24 | 319 | 34.64 |
| 31 | 259 | 29.27 |
| 40 | 210 | 24.20 |

Other MoE (n_cpu_moe 16): Laguna-XS.2 IQ4_XS 461 pp / **46.21** tg; GLM-4.7-Flash-APEX (deepseek2, Q6_K) 215 pp / 29.25 tg; L3.2-8X3B (llama arch, Q8_0) 466 pp / 11.90 tg.

RDNA2 vs RDNA4 (Qwen3.5-35B): decode peak 34.6 vs 44.1 (0.79x); prefill 435 vs 760 (0.57x). The 12 GiB ceiling shifts the decode optimum from `n_cpu_moe` 8 (RDNA4, 16 GiB) to 24 (RDNA2, 12 GiB). Best RDNA2 config measured: Laguna-XS.2 IQ4_XS at 46.2 tok/s.

### 19b. RDNA2 app trace + pool-default regression (RUN-011)

Traced the real app path (`trace_demo` -> `engine::start_gguf`) on the gfx1031 box (ctx 32768, n_slots 4, threads 6, Qwen3.5-35B-A3B Q4_K, n_cpu_moe 26):

| metric | pool default 2048 (buggy) | RAM-aware default (fixed) | manual mmap |
|---|---:|---:|---:|
| decode ms/tok | 58.6 | 34.8 | 35.2 |
| decode tok/s | 17.1 | 28.8 | 28.4 |
| prompt ms/tok | 82.3 | 51.6 | 49.8 |
| VRAM | 9.59/11.98 GiB | 9.60/11.98 GiB | - |

Root cause: `pool_for_gguf()` defaulted `--pool-mb 2048`, so the app always used the expert prefetch pool. On this 47.9 GiB box the 19.69 GB model fits in RAM, so the pool thrashed (`bypass=176229 evict=12896`) while mmap page cache stayed hot. Fix: `pool_for_gguf(model)` enables the pool only when `model_bytes + 4 GB > RAM`; `E0_POOL_MB` still overrides. Verified: +68% decode on RDNA2, matches llama-bench. This extends the RUN-006 correction (mmap beats the pool at every budget when RAM holds the weights).

## 20. Prefill / KV levers + ROCmFPX 35B breakdown (RDNA2, RUN-019)

35B-A3B Q4_K, n_cpu_moe 20, `llama-bench -p 512`:

| config | pp512 |
|---|---:|
| default (-b2048 -ub512) | 308.9 |
| -b4096 -ub4096 | 329.1 |
| + GGML_OP_OFFLOAD_MIN_BATCH=512 | 356.6 |

KV cache at n_cpu_moe 20 (llama-bench, short ctx): f16 37.9 tg vs q8_0/q4_0 37.2 tg
(neutral - KV is tiny at 512 tokens; the win is at long context).

ROCmFPX ornith-1.0-35B byte breakdown (why 3.6 tok/s): fp3=8.95 GB, Q5_K=5.17 GB,
fp6=4.38 GB, fp4_fast=0.54 GB. The native MMVQ kernel covers fp4_fast only (0.54 GB);
fp3+fp6 = 13.3 GB run the dequant-to-f16 fallback. fp6/fp3 MMVQ is the next kernel.

## 21. Long-context KV lever validated (RDNA2, 35B, ctx 32768)

`llama-server --ctx-size 32768`, Qwen3.5-35B-A3B Q4_K, decode (n=128):

| KV | n_cpu_moe | decode tok/s |
|---|---:|---:|
| f16 | 26 (planner fit) | 26.68 |
| q8_0/q4_0 | 26 | 28.17 |
| q8_0/q4_0 | 20 | **33.10** |
| q8_0/q4_0 | 16 | 27.12 (spills) |

At 32k the f16 KV is ~2.7 GB vs ~0.7 GB quantized (q8_0/q4_0). Freeing that VRAM lets
`n_cpu_moe` drop 26 -> 20 (6 more expert layers on the GPU), lifting decode +24%
(26.68 -> 33.10 tok/s). n16 regresses: the model spills. So the recommended long-context
config is **`-ctk q8_0 -ctv q4_0` with n_cpu_moe from the planner recomputed against the
quantized KV budget**. The KV quant itself (same n) is +5.6%.

## 22. RAM-capped benchmark - tiered expert residency is dead as wired (RDNA2, 16 GiB SYS RAM)

**Box:** RX 6700 XT gfx1031, 12 GiB VRAM, **16,305 MB (~15.9 GiB) SYS RAM** (user pulled 32 GiB
from 48), ROCm 10 at `C:\Rocm10\bin`, build-hip, patch band 0001-0009.

**Model:** `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` (18.63 GiB - overruns 15.9 GiB RAM by 2.7 GiB).
`-ngl 99 --n-cpu-moe 20 -fa auto --ctx-size 4096`, 48 decode tokens, thinking off
(`--reasoning-format none --no-reasoning-preserve`), Qwen3.8 non-thinking sampling.

| arm | env | decode tok/s | ms/tok |
|---|---|--:|--:|
| A - no tiers | (none) | **11.55** | 86.61 |
| B - RAM L1 pool | `--pool-mb 2048` | 8.36 | 119.61 |
| C - RAM + VRAM hot | `--pool-mb 2048` + `E0_VRAM_MB=1024` | 8.36 | 119.61 |

Arm C trace over the run: `hit=6292 miss=2533 evict=1293 fills L1=0 GGUF=2533` =
**71.7% hit rate while 28% SLOWER**. Every fill came from the GGUF, never from the resident
expert tensor.

**Why it cannot pay off:** `ggml_cuda_mul_mat_id` only runs on the GPU backend, so `src0` is
necessarily already VRAM-resident when the hook fires. The pool duplicates data that is
already in VRAM and pays NVMe reads + H2D copies + evictions to serve "D2D hits" that
replace a VRAM read with a VRAM read.

| expert tensor lives in | does the hook fire? | result |
|---|---|---|
| host RAM (`-cmoe` / `--n-cpu-moe`) | never - CPU backend runs the matmul | dead |
| VRAM (`-ngl`) | yes | dead and 28% slower: data already resident |

## 23. 78 GiB 288-expert model - pure cold-tier ceiling (RDNA2, 16 GiB)

`qwen3.8-flash-next-reap-288-Q4_K_M.gguf` (78.06 GiB), `-cmoe` (all experts on host).
Architecture verified from arXiv:2608.30320: 125B total / **6B activated per token**, and
"51B parameters of n-gram embedding tables held off the accelerator" that are "prefetched
from host memory" - the tiered-residency design is built into the model. Token mix is 3 GDN
+ 1 QSA attention per 4; n-gram embedding at Layer 2.

| metric | value |
|---|---:|
| load | 76 s |
| prompt eval (12 tok) | 78,114 ms (0.15 tok/s) |
| decode (32 tok) | 1,143,445 ms (36,885 ms/tok, **0.03 tok/s**) |

78 GiB demand-paged through a 16 GiB box thrashes the page cache on every expert miss. This
is the number the VRAM/RAM/NVMe tiering exists to fix, and it is the reason the negative in
S22 matters.

`-ngl 99` without `-cmoe` on this model cannot load at all:
`alloc_tensor_range: failed to allocate ROCm buffer of size 48239767808` (48.2 GiB requested
on 12 GiB VRAM) - llama.cpp offloads expert tensors all-or-nothing per tensor.

## 24. VRAM hot-tier routing defect (fixed) and profiling tooling

Two defects kept the tier from running at all; both fixed in patch 0009:

1. `edge0_pref_init` parsed `E0_VRAM_MB` 88 lines *after* the early-return gate that tests
   `g.vram_mb` (prefetch.cc:997 vs :1085), so the env var alone was a no-op.
2. `ggml_cuda_mul_mat_id` takes a fused fast path (`mul_mat_vec_q` at ggml-cuda.cu:1990) and
   **returns** before reaching the per-expert loop where the base is substituted.
   `ggml_cuda_mul_mat_id_needs_sync` only feeds an *assert* at line 2015, so patching it
   changed nothing. Measured proof: `hit=0 miss=0 refused=0 bypass=0` over 4 steps.

After 0009 the hook engages - steps 1 to 4 show hits 12 -> 36 -> 56 -> 77 with `fills GGUF`
tracking misses. Identification and hot top-N accounting are working.

**Profiling:** rocprof/bprofiler are absent, but the `hip_tracer` proxy DLL from
`G:\AMD-Ai\hip_tracer` is now wired on the box. Two deployment constraints, both learned the
hard way:

- Launching via `hip_tracer_launcher.exe` does NOT work. Windows DLL search starts with the
  directory of the child executable, and `Copy-HipRuntime` already put the real
  `amdhip64_7.dll` there, so PATH ordering is never consulted. The proxy must be copied
  INTO `build-hip/bin/` with the real DLL renamed alongside it as `amdhip64_7_real.dll`
  (the proxy forwards via the PE forwarder mechanism to a file in its own directory).
- The forward target must be the remote's own ROCm (`C:\Rocm10\bin\amdhip64_7.dll`,
  identical bytes under `D:\Rocm10\bin`), not a local copy - the two differed
  (17,503,240 vs 16,577,024 bytes).

Result: clean capture, 12 s load, 9.80 tok/s, 10,290 HIP calls, 934 KB CSV.

**Honest limitation:** kernel names do not resolve for this engine - 99% of the rows come
back as `stub@ggml-hip.dll` - so it does **not** yet give per-kernel attribution, which was
the point of wiring it. The memcpy shapes are readable and show the expert traffic (a
stream of repeated 142.6 MB blocks, plus 10.37 GB and 417 MB transfers).

## 25. Full `C:\x` model sweep (RDNA2, RX 6700 XT 12 GiB, 48 GiB RAM)

Every `.gguf` in `C:\x` (11 files, 227 GB). `llama-bench`, **`-t 8 -p 512 -n 128 -r 2 -o csv`**,
12 GiB VRAM / 48 GiB host RAM, engine build `c838aa844` (build 11111). Auto-fit is used for
all rows (omit `-ngl`, default `-1`) with `--load-mode none`; the 78 GiB model is the exception
(explicit `-ngl 99 -ncmoe 99`, mmap). Columns are **Prompt Process (pp512 tok/s)** and
**Token Gen (tg128 tok/s)**.

| model | arch | size GiB | params | ncmoe 0 pp/tg | ncmoe 2 pp/tg | ncmoe 6 pp/tg |
|---|---|---:|---:|---:|---:|---:|
| gemma-e2b-rocmfp4 | gemma4 E2B (dense) | 3.13 | 4.65 B | 1633 / 132.2 | 1643 / 132.0 | **1644 / 132.3** |
| ornith9b | qwen35 9B (dense) | 4.62 | 8.95 B | 439.8 / 64.5 | 440.7 / 64.6 | **440.9 / 64.6** |
| Underdog-Saluki-27B IQ2_XS | qwen35 27B (dense) | 7.78 | 27.3 B | **179.7 / 22.1** | 179.5 / 22.1 | 179.6 / 22.1 |
| Qwen3-30B-A3B Q2_K | qwen3moe 30B.A3B | 10.48 | 30.5 B | **300.8 / 115.2** | 297.6 / 99.1 | 292.2 / 77.1 |
| Laguna-XS.2 IQ4_XS | laguna 30B.A3B | 16.84 | 33.4 B | 577.3 / 39.5 | 557.6 / 38.2 | **849.3 / 39.9** |
| GLM-4.7-Flash-APEX Q6_K | deepseek2 30B.A3B | 17.89 | 29.9 B | 421.9 / 23.0 | 408.4 / 22.2 | **701.2 / 24.4** |
| ornith35b | qwen35moe 35B.A3B | 17.96 | 34.7 B | 263.9 / 29.1 | **421.1 / 24.3** | 253.8 / 15.1 |
| Qwen3.5-35B-A3B Q4_K_M | qwen35moe 35B.A3B | 18.34 | 34.7 B | 415.9 / 28.4 | **929.5 / 28.2** | 196.8 / 30.4 |
| Qwen3.6-27B-AEON Q5_K_M | qwen35 27B (dense) | 18.19 | 27.3 B | 78.0 / 3.27 | **79.2 / 3.25** | 76.6 / 3.30 |
| L3.2-8X3B-MOE Q8_0 | llama (MoE) | 18.22 | 18.4 B | 432.6 / 11.3 | 397.0 / 11.3 | **606.0 / 11.8** |
| qwen3.8-flash-next-reap-288 Q4_K_M | qwen4exp A3B | 78.05 | 124.1 B | 50.9 / 10.6 (`-ngl 99 -ncmoe 99`) | - | - |

Best measured per model in bold. Raw CSVs: `D:\edge0\b1.csv`..`b11.csv` on the box.

### 25b. Findings

- **Two dense subtypes.** Small dense models run fast (`gemma-e2b` pp 1644 / tg 132;
  `ornith9b` 441 / 64.6). Dense 27B is memory-bound: the IQ2_XS file (7.78 GiB) fits mostly in
  VRAM -> 180 / 22.1, but the Q5_K_M file (18.19 GiB) cannot -> **78 / 3.27**, a 6.7x decode
  collapse for 2.3x more bytes. `-ncmoe` is a no-op on all dense models (no MoE tensors).
- **MoE `-ncmoe` is model-dependent, not monotonic.** Prefill improves sharply once experts fit
  (`Qwen3.5-35B` pp 415.9 -> **929.5** at ncmoe 2; `Laguna` 577 -> **849** at 6; `GLM` 422 -> **701**
  at 6), but over-offloading past the fit collapses it (`Qwen3.5-35B` ncmoe 6 -> 196.8;
  `ornith35b` ncmoe 6 -> 253.8). The optimum is the smallest `-ncmoe` that stops VRAM spill.
- **When the model already fits VRAM, `-ncmoe` only hurts.** `Qwen3-30B-A3B Q2_K` (10.48 GiB
  < 12 GiB) peaks at ncmoe 0 (**tg 115.2**); pushing experts to CPU drops decode to 77.1.
- **`L3.2-8X3B Q8_0` decode is CPU-matmul bound** (~11.3 tg) regardless of `-ncmoe`; prefill
  still scales with GPU expert residency (432 -> 606).
- **The 78 GiB / 124 B-A3B model runs**, but only with all experts on host: `--load-mode none`
  cannot allocate 78 GiB in 48 GiB RAM (`failed to load model`), auto-fit (`-ngl -1`) also fails,
  and `-ngl 99 -ncmoe 0` requests a 48.2 GiB ROCm buffer. Working config = **`-ngl 99 -ncmoe 99`**
  (mmap): pp 50.9 / tg 10.6 - decode is usable, prefill is host-bound.

### 25c. Cross-check vs S19 (RDNA2, `-ngl 99`, n_cpu_moe 16)

S19 measured n_cpu_moe 16: Laguna 461 / 46.2, GLM 215 / 29.3, L3.2 466 / 11.9. The S25 rows use
auto-fit with ncmoe 0/2/6, so they are not the same operating point (ncmoe 0 leaves more experts
on GPU than n_cpu_moe 16, and Laguna ncmoe 6 shows the prefill peak that S19's fixed 16 missed).
Numbers agree in shape: L3.2 decode ~11-12 tg on both; Laguna decode 39-46; GLM 23-29.

