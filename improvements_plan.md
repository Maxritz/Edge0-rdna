# Edge0 Improvements Plan

Durable findings from the tracing / tuning / pooling investigation, plus a
prioritised roadmap. Numbers here are measured on this machine unless marked
otherwise. This file is the working memory for the engine work; update it rather
than re-deriving.

## 1. Baseline: hardware, engine, models

### Machine (current)
- GPU: AMD Radeon RX 9070 XT, `gfx1201` (RDNA4), 16 GiB VRAM. ReBAR **on**
  (`HKLM\...\Class\{4d36e968-...}\0000\KMD_RebarControlMode = 0x1`).
- CPU: 16 threads visible. RAM: 95.9 GiB.
- Target profile we must also satisfy: **12 GiB VRAM / 32 GiB RAM**.

### Engine
- Pinned upstream llama.cpp `7ab4ee7` (b11100) in `vendor/llama.cpp` (never
  patched in place). Platform worktrees replay patch bands:
  `wt/win` = common band (6) + windows band (2).
- Windows product copies `windows/serve/*` into `wt/win/src/edge0` (CMake glob,
  patch #3). These are our own files:
  - `prefetch.cc` - advisory expert prefetch + self-managed L1 expert pool +
    three-way overlap accounting (`[pref-trace]` lines).
  - `mem_budget.cc` - hard working-set cap (`--mem-budget-mb`).
- Patch surface #7 (`patches/llama.cpp/common/0006`) routes the three expert-base
  computations (`ggml-cpu.c` generic `mul_mat_id`, `iqp.cpp`, `repack.cpp`) through
  `ggml_edge0_mmid_base(src0, id, nb02)`. NULL resolver = exact upstream math.
  The pool installs a resolver here to serve expert bytes from private pages.
- CLI surface: `--pool-mb` (env `E0_POOL_MB`), `--mem-budget-mb` (env
  `E0_MEM_BUDGET_MB`), plus env gates `E0_POOL`, `E0_POOL_OBS_MB`, `E0_PREROUTER`,
  `E0_PREFETCH`, `E0_PREF_*`.
- Engine binary: `wt/win/build-hip/bin/llama-server.exe` (HIP/ROCm build).

### Performance tracer (added this session, committed `35a4722`)
- `windows/app/src-tauri/src/perf.rs`: opt-in (`--profileperf` /
  `EDGE0_PROFILEPERF=1`), zero-cost when off, RAII spans, PDH + registry resource
  sampling, engine-log phase parsing, report table + JSON.
- Wired into `lib.rs` (`perf_set` / `perf_report` / `perf_reset`) and a
  Performance Trace panel in the UI.
- Examples: `trace_demo.rs` (load + generate + report), `tune_sweep.rs`
  (measured config sweep, env-driven).

### Models
- Primary: `Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf` (19.69 GB), arch `qwen35moe`.
  Experts are **MXFP4 (ggml type 39)** and **Q6_K (type 14)** - NOT Q4_1/Q4_K.
  40 layers, 256 experts, 4 used, 262144 max ctx, KV ~81920 B/token (f16).
- `gpt-oss-120b-Q8_0.gguf` (63.4 GB): 36 layers, 128 experts, 4 used, experts
  **MXFP4**, expert bytes ~61.1 GB, KV 73728 B/token.

## 2. Measured performance (baseline)

### Trace demo (35B, 32k ctx, plan `-ngl 99 --n-cpu-moe 16`)
- `engine.decode` 88.2% of device time, ~26844 us/token (~37 tok/s).
- `engine.prompt-eval` 11.8%. App spans: `engine.launch` ~7.6 s,
  `engine.wait_ready` ~7.5 s, `gguf.inspect_for_gpu` ~0.47 s.
- Resources at load: cpu ~52-58%, gpu ~26-28%, ram 33.7/95.9 GiB,
  vram 14.5/15.9 GiB, rebar on.

### Tuning sweep - 32k ctx (decode tok/s / prefill tok/s)
| n_cpu_moe | kv | decode | prefill | vram |
|---:|---|---:|---:|---:|
| 8  | f16  | 40.2 | 434 | 15.1 |
| 12 | f16  | 40.6 | 362 | 14.9 |
| 16 | f16  | 40.2 | 429 | 14.4 |
| 20 | f16  | 34.7 | 342 | 12.7 |
| 24 | f16  | 31.4 | 281 | 11.1 |
| 31 | f16  | 26.1 | 208 |  8.2 |
| 12 | q8_0 | 40.2 | 356 | 14.9 |
| 16 | q8_0 | 38.6 | 420 | 14.1 |
| 24 | q8_0 | 30.8 | 277 | 10.8 |

Findings: decode peaks ~40 tok/s for `n_cpu_moe` 8-16 and falls off steeply
beyond; KV quantization buys no decode speed at 32k (it buys capacity).

### Tuning sweep - 128k ctx (the config we actually need, >=118k)
| n_cpu_moe | kv | decode | prefill | vram |
|---:|---|---:|---:|---:|
| 24 | q8_0 | 30.8 | 272 | 12.0 |
| 28 | q8_0 | 27.2 | 231 | 10.4 |
| 31 | q8_0 | 25.3 | 207 |  9.1 |
| 34 | q8_0 | 23.4 | 191 |  7.8 |
| 40 | q8_0 | 21.3 | 166 |  5.2 |
| 31 | f16  | 26.2 | 210 | 10.2 |
| 40 | f16  | 21.5 | 166 |  6.3 |

Findings: at 128k, f16 KV edges out q8_0 at equal `n_cpu_moe`; the real value of
KV quantization is VRAM freed for GPU experts (lower `n_cpu_moe`), not raw speed.

## 3. Expert pool: what it is and the one fix it needs

- The pool (`prefetch.cc`) moves opaque expert bytes from disk into committed
  private pages, keyed by tensor NAME, served via patch surface #7. Design is
  byte-blob agnostic (arena, `ReadFile`, slot state machine). Pool-only mode =
  demand fill of only the routed experts (no sidecar / no `E0_PREROUTER`).
- The ONLY quant-aware line is `tt_bytes()` (hand table for F32/F16/BF16/Q4_1).
  MXFP4 (39) and Q6_K (14) size to 0, so `e0_pool_scan` skips them. Both target
  models use MXFP4 experts, so **neither pools today**.
- Fix (about 10 lines, one function): size experts via ggml's own arithmetic
  instead of a hand table:

```cpp
size_t tt_bytes(const TT & x) {
    const int64_t bs = ggml_blck_size((ggml_type) x.type);
    const size_t  es = ggml_type_size((ggml_type) x.type);
    if (!bs || !es) return 0;                 // unknown -> not poolable, graceful
    size_t n = 1;
    for (int i = 0; i < 4; ++i) n *= (size_t) x.dim[i];
    if (n % (size_t) bs) return 0;
    return es * (n / (size_t) bs);
}
```

This covers Q4_1, Q4_K, Q5_K, Q6_K, MXFP4, NVFP4, F16, BF16, F32, IQ* in one
line and deletes `is_q4_1`/`plain_esz`. `prefetch.cc` already includes
`llama.h` -> `ggml.h`, which declares both functions.
- Cost is the **rebuild** (`windows/scripts/vendor-build.ps1 -Backend hip`), not
  the code.
- Hook coverage: MXFP4 has a repack specialization that routes through the
  hooked `forward_mul_mat_id`; NVFP4 uses the generic/IQP path. Both hit #7.
- `E0_POOL_MB` / `--pool-mb` enables it; without `E0_PREROUTER` it is pure
  demand fill.

## 4. KV cache

- Accepted KV types at the pin (`common/arg.cpp`): `f32, f16, bf16, q8_0, q4_0,
  q4_1, iq4_nl, q5_0, q5_1`. **No q6**; "q5" is only legacy `q5_0/q5_1` (no K-quant).
- Constraints (`src/llama-context.cpp`): quantized V requires flash attention;
  block size must divide head dim; MLA / DeepSeek4 require K type == V type.
- Our numbers show KV quant is a *capacity* lever, not a speed lever. TurboQuant
  turbo3/turbo4 (ROCmFPX) is the low-bit-KV path our pin lacks.

## 5. Attention variants

- Plain MHA = most KV-expensive; not an optimization.
- GQA (our model) already cuts KV ~4x vs MHA; MQA is more aggressive.
- MLA (DeepSeek/Kimi) compresses KV to a latent; engine supports `is_mla`. This
  is the biggest attention-side lever for our long-context ceiling, gated on
  running an MLA model (DeepSeek-V4-Flash / Kimi files on disk).
- Flash attention already present and required for quantized V.
- Sparse/data-dependent attention (paper #4) not present (only positional SWA).

## 6. Decode speed

- Decode is bandwidth-bound: ~40 tok/s at 32k is set by bytes moved (quant size,
  how many experts on CPU, KV size), not FLOPs.
- Biggest available lever: **MTP / NextN self-speculative decoding** (~1.5x on
  Qwen3.6-35B-A3B in ROCmFPX tests, lossless at greedy). Requires a model with an
  MTP/NextN head (gpt-oss does not have one).
- Our pin already has speculative infra (`common/speculative.*`) and NextN tensor
  arch entries, but not the full draft-mtp decode path ROCmFPX advertises.

## 7. Kernel layer (compute / prefill)

- Our HIP backend: RDNA2 (`gfx1030`) = DP4A minimum, no WMMA; RDNA3 (`gfx1100`)+ =
  WMMA; RDNA4 (`gfx1200`) = WMMA. Flash attention already branches this way.
- gfx1201 -> WMMA/rocWMMA; gfx1031 -> DP4A/SDOT4 (int8). No FP4 matrix path on
  RDNA2, so MXFP4 experts there would fall to a slow dequant path (prefer
  DP4A-friendly int8/int4 quants on gfx1031).
- These paths help **prefill** and the GPU-vs-CPU expert decision; they do NOT fix
  batch-1 decode.

### tilelang port (`G:\tilelang-rocm`)
- Local port with real RDNA targets: `tilelang/rocm/op/gemm/gemm_wmma.py` (RDNA3/4)
  and `gemm_fma.py` (RDNA2 fallback), `src/rocm/target_utils.cc` detection, a
  `perfgunner` profiler. Configured against `G:/ROCM10RT-gfx1201`.
- Relevant examples: `gdn` (our qwen35moe recurrent layers), `deepseek_mla/amd`,
  `flash_decoding` (gqa_decode), `dequantize_gemm` (mxfp4 / w4a8), `fusedmoe`,
  `grouped_gemm`.
- Use as a **kernel lab** (author + autotune, then port the winner into ggml) or as
  the kernel layer of an own-runtime engine. Not a drop-in for ggml.

## 8. ggml custom-op bridge (assessment)

- ggml's custom-op surface (`ggml_map_custom1/2/3`, `GGML_OP_CUSTOM`) is a **CPU
  callback** (`ggml_custom_op_t(dst, ith, nth, userdata)`), not a GPU dispatch.
- HIP dispatch (`ggml-cuda.cu:2067 ggml_cuda_compute_forward`) has cases for
  `MUL_MAT`/`MUL_MAT_ID` etc. and **no generic custom case**.
- Real per-kernel bridge cost (all C/C++): op enum + constructor (`ggml.h`), a
  dispatch case (`ggml-cuda.cu`), a new `ggml-cuda/<kernel>.cu` linking the
  tilelang `extern "C"` wrapper on ggml's stream, graph-builder wiring in
  `src/llama-graph.cpp`, and a parity test. Rust is not in this path.
- tilelang emits an `extern "C"` wrapper per kernel (`codegen_hip.cc`), so the
  FFI launch itself is trivial; the hard part is graph/layout/stride/quant-format
  matching and correctness.

### Measured: the FFI bridge works (2026-10-09)
Built and ran the bridge end-to-end against the local tilelang tree
(`G:\tilelang-rocm`), no fetch, no pip install (dev-root import loads
`build\lib\tvm_*.dll`). Chain: tilelang kernel -> C-ABI launcher -> Rust.

- tilelang compiled the quickstart matmul to HIP for gfx1201 and used the RDNA4
  intrinsic `__builtin_amdgcn_wmma_f32_16x16x16_f16_w32_gfx12`.
- Launcher `launcher.cpp` exposes `e0_matmul_f16` (host f16 A/B -> host f16 C =
  relu(A@B), owning the device round-trip) and `e0_hip_device_name`.
- Built: `hipcc -x hip --offload-arch=gfx1201 -std=c++17 -O3 -shared`
  (rocwmma headers need C++17). Runtime needs `amdhip64_7.dll` on PATH
  (`G:\ROCM10RT-gfx1201\bin`).
- Rust `examples/tilelang_bridge.rs` loads it with `LoadLibraryA`/`GetProcAddress`
  (raw Win32, no crates) and calls it.
- Result (1024^3, f16 in / f32 accum / relu): device "AMD Radeon RX 9070 XT",
  534363/1048576 nonzero (relu), max abs err 6.25e-2, max rel err 4.88e-4,
  1.66 ms for H2D+kernel+D2H (~1293 GFLOP/s). **PASS.**

Conclusion: the *launch* half of "call a tilelang kernel" is trivial and language
-agnostic (done from Rust). The remaining cost for a ggml integration is entirely
C++ (op enum, `ggml-cuda.cu` dispatch case, graph-builder wiring, parity), not the
FFI. So a Rust runner over tilelang kernels is easy; a llama.cpp custom-op is C++.

### tilelang dev-tree notes
- Import works from the source tree: `PYTHONPATH=G:\tilelang-rocm`; it prints
  "Loading tilelang libs from dev root" and uses `build\lib`.
- Version string `0.1.15+cuda.git76c95fe1` is a **build tag** (CUDA stubs ON, TVM
  `USE_CUDA OFF`); generated device code is HIP and torch is the ROCm build
  (`torch.version.cuda=None`, `torch.version.hip=7.16`), so "cuda" is naming only.

## 9. External assessments

### Kraken spec (`kraken_compressed_expert_tiered_residency_spec.md`)
- MoE **expert-weight** architecture: offline W8/W6/W4/ternary quant, hot/warm/cold
  residency VRAM/RAM/NVMe, predictive prefetch, CPU fallback, grouped HIP GEMM.
- Not about KV (the word appears only as budget line items).
- Most of it already exists in simpler form in llama.cpp: W4 experts, `--n-cpu-moe`
  CPU fallback, mmap warm tier. New parts are a from-scratch engine (months, high
  risk); its own text warns smaller representation is not faster if the kernel
  spends time unpacking.

### ROCmFPX (`github.com/charlie12345/ROCmFPX`)
- llama.cpp fork adding AMD weight formats (ROCmFP2/3/4/6/8, ROCmI4), MTP
  self-speculative decode, and TurboQuant turbo3/turbo4 KV.
- Relevant pieces: MTP spec-decode (decode speed), TurboQuant KV (our missing
  capacity lever), low-bit expert formats (less disk/RAM/PCIe per expert).
- Adoption = engine merge (replay our patch band onto their tree, or port their
  kernels into our pin). Their validated numbers are mostly `gfx1151`.

### Papers (`kraken/docs/implementing-papers.md`) vs our engine
- Help: #5 DeepSeek-V4.1-Flash KV compression (capacity), #1 Tail-Replay (hybrid
  prefix reuse / TTFT), #6 REA (chat prompt layer), #11 Sample-Guided Exact Top-K
  (router / sparse attention), #12 layer-ordering correctness prior.
- Big bet, defer: #4 Self-Indexing Attention (sparse long-context; new op + index).
- Not ours: #2 RadixMLP (multi-request batching), #3 Feather+CHT (datacenter
  scheduler), #7 YOCO (architecture), #8 SPLASH (multi-GPU), #9 VitaLLM (hardware;
  only leading-one top-K and radix-4 Booth are software-transferable), #10 CodeTD
  (needs attention-map export).

## 10. Prioritised roadmap

Cheapest, highest-confidence first:

1. **`tt_bytes` -> ggml_type_size fix** (unblocks the pool for MXFP4/NVFP4/Q6_K,
   i.e. both target models). Then verify with gpt-oss-120b pool-only on a capped
   profile.
2. **KV experiments** at 128k: add q4_0 / iq4_nl rows, push `n_cpu_moe` lower, find
   the free-VRAM-for-experts sweet spot.
3. **MTP self-speculative decode** (decode speed; needs an MTP-head model).
4. **Low-bit KV (TurboQuant turbo3/turbo4)** or paper #5 - the capacity lever.
5. **Tail-Replay** for hybrid prefix reuse (TTFT on long prompts).
6. **Router top-K** (paper #11) and **layer-order logging** (paper #12) - small,
   self-contained.
7. **Kernel lab**: tilelang for prefill / GPU-expert compute; bridge only when a
   kernel is proven worth integrating.
8. **Long-term**: MLA model bring-up; sparse attention (#4); own-runtime engine
   (Kraken-style) if the above hit a wall.

## 11. Open questions / blockers
- 32 GiB / 12 GiB test profile is not this machine (95.9 GiB / 16 GiB); needs a
  commit-capped harness to reproduce.
- MTP needs a model with an MTP/NextN head.
- tilelang bridge integration is per-kernel C/C++ work.

## 12. Build/runtime staging fix: hipBLASLt (2026-10-09)

- Symptom (engine + llama-bench logs): `rocblaslt error: Cannot read
  "TensileLibrary_lazy_gfx1201.dat"` + `hipModuleLoad failed: Kernels.so-000-gfx1201.hsaco`.
- Cause: the runtime staging in `windows/scripts/vendor-build.ps1` `Copy-HipRuntime`
  copied `rocblas\library` but never `hipblaslt\library`. Arch dispatch is inside
  hipBLAS (`ggml-hip.dll` imports only `hipblas.dll`): **RDNA2 (gfx1031) uses rocBLAS,
  RDNA3/4 (gfx1100/gfx1200/gfx1201) use hipBLASLt**. Without the Lt library data, the
  RDNA4 f16 GEMM fell to a slow fallback.
- Fix: `Copy-HipRuntime` now stages `hipblaslt\library\<arch>` for each target (keeps the
  rocblas path for RDNA2).
- Measured (test-backend-ops, f16, m=4096 n=512 k=14336): **8.42 TF -> 96.33 TF (+11.4x)**,
  errors gone. Decode small-n unchanged (bandwidth-bound).
- Real model (Qwen3.5-35B, no f16 weights): pp512 504 -> 512 t/s, tg128 40.5 -> 41.9 t/s
  (near noise). The model has no f16 tensors, so the f16 GEMM path is unused; its prefill
  is CPU-expert bound. The fix matters for f16/bf16 models and removes error spam.
- NOTE: this run staged the libs manually; a `vendor-build.ps1` rebuild bakes it in.

## 13. Split-precision FP32 GEMM (tilelang) - attempted, does NOT validate

Technique borrowed from the "AI agents write CUDA kernels" article: split each fp32 into
hi+lo fp16, 3 tensor-core MMAs (hi*hi + hi*lo + lo*hi), fp32 accumulate, skip the
negligible lo*lo. Goal: fp32-accurate GEMM at fp16 tensor-core speed.

Chain: `gen_split.py` (tilelang, gfx1201) -> `launcher_split.cpp` (C ABI) ->
`examples/split_gemm.rs` (Rust; f64 reference + fp16 baseline). All under
`%TEMP%\opencode` except the Rust example.

Result: FAIL.
- Accuracy: max abs err vs f64 = 39.9, vs plain fp16 = 0.024 -> the split kernel is
  wrong (worse than plain fp16) and identically wrong across kernel variants, so the
  shared->shared cast/residual lowering is at fault, not the math.
- Speed: ~1.5 TF (1024x1024x2048) vs ggml ROCm0 f16 96.3 TF / f32 12.4 TF. Six shared
  buffers (2 fp32 + 4 fp16) = 64 KB shmem/block -> ~1 block/SM on RDNA4.

Conclusion: technique sound, this tilelang implementation is neither correct nor
competitive. Not shipped (no fake success). If revisited: split in fragments, shrink
shared footprint, and validate against ggml f32 before timing.

## 14. ninfer-offload borrows (H:\ports\ninfer-offload)

CUDA fork (device-slot cache over page-locked host banks); its code is not portable, its
hot-expert POLICY is. Implemented in `windows/serve/prefetch.cc`:

- **Decayed routing frequency** per slot (`freq[]`), decay `2^(-1/16)` per step (half-life 16
  steps, per ninfer), incremented on each hit/fill. Also a total-count `hits[]`.
- **Frequency-based eviction** (opt-in `E0_POOL_FREQ=1`): when the pool is full, evict the
  lowest-frequency stale slots, capped at `E0_POOL_MAX_SWAPS` (default 256) per step. Default
  stays recency-based (no regression); frequency is the better policy by design but unproven here.
- **Routing-count dump** (`E0_POOL_STATS=<file>`): one whitespace line of per-expert counts per
  tensor, ninfer's `write_routing_counts`, for seeding the pool / feeding the prerouter next run.

### Result (gpt-oss-120b, MXFP4, --n-cpu-moe 32, pool 8 GiB, selfcheck on)
| policy | evict | fillD | coverage | selfcheck bad |
|---|---:|---:|---:|---:|
| recency (default) | 5883 | 7414 | 49.7% | 0 |
| frequency (E0_POOL_FREQ=1) | 7554 | 9011 | 50.0% | 0 |

Both correct (0 mismatches). Frequency is a wash on this box (page-cache absorbs the disk reads),
and slightly more evictions here; its value is for a RAM-constrained host, untestable here.
Kept opt-in and documented, not made the default.

### What we did NOT take (deliberately)
- The device-slot / page-locked-host-bank design and the CUDA replacement protocol (unmap ->
  wait 2 requests -> copy on a side stream -> map): CUDA-specific. Our analog is decommit after
  `llama_synchronize`. Different cache level: ninfer caches GPU<->host; we cache disk<->RAM.
- No predictor in ninfer; we have the prerouter (predict-ahead). The two compose: predict (admit
  early) + decayed frequency (retain).

## 15. RAM-constrained test (--mem-budget-mb): mmap beats the pool

Attempted to reproduce the 16 GB / 8 GB-VRAM product target on this box with the process RAM
cap (`--mem-budget-mb`). gpt-oss-120b, plan `-ngl 99 --n-cpu-moe 34` (GPU 5.3 GiB, CPU 53.7 GiB),
120 greedy tokens.

| budget | config | decode tok/s | RSS MB |
|---|---|---:|---:|
| 20 GiB | mmap | 8.02 | 20291 |
| 20 GiB | pool 4 GiB | 4.84 | 20283 |
| 20 GiB | pool 4 GiB freq | 4.56 | 20281 |
| 16 GiB | mmap | 5.22 | 16210 |
| 16 GiB | pool 3 GiB | 3.72 | 16199 |
| 16 GiB | pool 3 GiB freq | 3.80 | 16197 |

Conclusion: **mmap wins at every budget**, and the pool is net-negative. Reason (now understood):
`--mem-budget-mb` caps the process WORKING SET, not the OS **file/standby cache**, which is
system-wide and unbounded. On a 96 GiB host the 63 GB file stays cached in RAM regardless of the
per-process cap, so mmap faults hit RAM; the pool's explicit `ReadFile` does the same read PLUS a
copy into private pages, and also thrashes its own arena. The pool only pays off when the weights
are genuinely not resident anywhere, i.e. a machine whose PHYSICAL RAM cannot hold the file.

Implication: the 16 GiB/8 GiB product target cannot be faithfully reproduced with a process cap
on this box. A real test needs a VM with 16 GiB RAM (or a cold-cache / limited-page-cache harness).
Until then, treat the pool as unproven on its target and keep it off by default (as it is).

## 17. Model triage + ROCmFPX PR #42 + kraken expert index

### Dense vs MoE (checked against raw tensors, not the planner's metadata)
| model | arch | FFN tensors | verdict |
|---|---|---|---|
| Qwen3.8-27B-WebGGUF-Q4_0 | qwen35, 65L | ffn_gate/up/down.weight + ssm_* | **dense** hybrid (attn+SSM), 0 experts |
| Qwen3.8-27B-GSQ-RCO-IQ3_S-mtp | qwen35, 65L | ffn_* .weight + nextn.* | **dense** + embedded MTP head |
| Muse-Glimmer-30B-UD-Q8_K_XL | muse-glimmer, 52L | ffn_* .weight | **dense** |
| Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX | qwen35moe, 41L | ffn_*_exps.weight + gate_inp | **MoE (256e)** but expert types 107/102 = ROCmFPX custom, NOT in pinned ggml enum -> **unloadable by our pin** |
| Qwen3-30B-A3B-abliterated-erotic.i1-Q2_K | qwen3moe | ffn_*_exps.weight Q2_K + gate_inp | **MoE, loadable** |
| Qwen2.5-Coder-32B-Instruct-3MPER0RR | qwen2, 32L | ffn_gate/up/down.weight | **dense** |

Note: a real MoE has `ffn_{gate,up,down}_exps.weight` (3D, [hid,inter,n_exp]) AND a router
`ffn_gate_inp.weight`. Dense has only the `ffn_*.weight` triple. `qwen35` (65L, ssm_*) is a
dense hybrid; the MoE sibling is `qwen35moe`.

### ROCmFPX PR #42 (charlie12345) - "vulkan: add ROCmFP2 Q8_1 decode kernels"
- Lesson: ROCmFP2 was SMALLER than ROCmFP4 (2.628 vs 4.293 BPW) but did NOT decode faster until
  the **routed (MUL_MAT_ID) Q8_1 integer-dot matrix-vector** path was added; decode had fallen
  back to dequantize-and-dot. After: ROCmFP2 90.3 tok/s vs ROCmFP4 76.2 (Strix Halo, +18.5%).
- Borrow: (1) for the DISK tier, a smaller expert format (ROCmFP2 ~2.6 BPW vs our MXFP4 4.25)
  reduces bytes per miss ~1.6x; (2) for DECODE, a dedicated ROUTED mmvq (MUL_MAT_ID) with
  integer dot is the unlock - generic dequant+dot is the slow fallback. Applies to our engine
  (ggml mmvq exists; the routed specialization is the lever).

### kraken expert index (sidecar `<model>.gguf.krakenexperts.json`)
- kind "kraken-expert-index", mode "full-forward": per layer, per expert `mass` + `hits`
  (activation mass + routing count), 40 layers x 256 experts, 502 positions scanned.
- This is EXACTLY the routing-frequency file we wanted (ninfer write_routing_counts / our
  E0_POOL_STATS): a precomputed hot-expert ranking. Seeds the pool hot-set and the prerouter.
- The file we have is for the ROCmFPX model (unloadable), but the format is what our stats dump
  should emit/consume.

## 18. ROCmFPX models in our folders (custom quant types; unloadable by the pin)

Scanned G:\More-models + H:\OLLAMA-Models\GGUF for custom (non-ggml) quant type ids.
Custom ids observed: 100, 101, 102, 104, 107 (ROCmFP4/FP2/FP3 family), 142 (ternary/PQ2).

### MoE with ROCmFPX types (both UNLOADABLE by our pinned engine)
| model | arch | size | custom types |
|---|---|---:|---|
| Qwen3.8-Distill-35B-A3B-Coder-Abliterated-Q2KXL_ROCMFPX.gguf | qwen35moe | 12.3 GB | 102x229, 107x214 |
| ornith-1.0-35B-Q3_0_ROCMFPX.gguf | qwen35moe | 19.3 GB | 101x2, 102x40, 104x268 |

### Dense with ROCmFPX types (also unloadable)
granite-4.1-3b-Q4_0_ROCMFP4_COHERENT (100); Spark-X2.5-4B-Q4_0_ROCMFP4_STRIX_LEAN (100,101);
gemma-4-E2B-it-Q4_0_ROCMFP4_COHERENT (100); Ornith-1.0-9b-ROCmFPX-STRIX_LEAN (100,101);
Ternary-Bonsai-2-27B-PQ2_0 (142).

### Spark-X2.5-4B (spark2_5), the case that proves the point
- Q4_0_ROCMFP4_STRIX_LEAN: types 100/101, 2.26 GB -> llama-bench "failed to load model".
- Q8_0 (standard): loads; pp512 6981 tok/s, tg128 95.9 tok/s (dense 4B, fully on GPU).
- Card is right: stock/our llama.cpp cannot load the ROCmFP4 file.

### Takeaway
Any model whose tensors use ROCmFPX quant enums (100-107, 142) needs the ROCmFPX llama.cpp
fork (or an enum+kernel port into our pin). Our pin tops out at ggml's MXFP4(39)/NVFP4(40).
The loadable equivalents on disk are the standard-quant siblings (e.g. Spark Q8_0, Q4_K_M).

NOTE: the scanner bug that spiked RAM to 89 GB was a string-array skip in the GGUF KV parser
(seeked 8*n instead of skipping each length-prefixed string); fixed with a per-element skip and
a length guard.

## 17b. ROCmFPX PR #42 diff (Vulkan-only) - specifics

- Scope: `ggml-vulkan.cpp` pipeline registration + GLSL shaders ONLY. PR states "no HIP,
  CUDA, GGUF, or quantizer-format change". So NOT portable to our HIP build as-is.
- Adds `GGML_TYPE_Q2_0_ROCMFPX` (FP2) to: `matmul` (prefill), `matmul_id` (routed/MoE),
  `mul_mat_vec_*` (MMVQ), and `mul_mat_vec_id_*` (routed MMVQ), each with a Q8_1 integer-dot
  variant. Decode only sped up once the ROUTED `mul_mat_vec_id` + Q8_1 path existed; before
  that decode fell back to dequantize-and-dot.
- Key trick: branchless SWAR codebook unpack (exhaustively tested over all 256 packed bytes):
    codes = (packed | packed<<12) & 0x000f000f;
    codes = (codes | codes<<6)    & 0x03030303;
    high  = (codes >> 1)          & 0x01010101;
    lanes = 0xfcfcfcfc + 3*codes - high - (high<<8);   // -> {-4,-1,1,4}
  No table lookup, no branches: expands a packed byte to the codebook in-register.
- Portable to us as PRINCIPLE: a dedicated ROUTED mmvq (MUL_MAT_ID) with native packed
  decode + integer dot. We run HIP, not Vulkan, so the GLSL is not reusable, but the same
  specialization applied to ggml-cuda (HIP) mmvq would be.

## 19. ik_llama.cpp (H:\LLAMA-bins\ik_llama.cpp) - NOT ROCmFPX; it is IQK

Checked: no ROCmFPX anywhere in H:\LLAMA-bins (recursive *rocmfp* = none). ik_llama.cpp's ggml
enum custom types: Q4_0_4_4(31), Q4_0_4_8(32), Q4_0_8_8(33), I2_S(36), MXFP4(39),
Q1_0_G128(41), Q6_0(133). No SPARK arch (the "spark" hits were "DSpark"/DeepSeek4). So the
ROCmFPX models (types 100-107,142) load ONLY in charlie12345's fork.

What ik_llama.cpp IS useful for: **IQK**, the hand-optimized quantized GEMM/GEMV family for
AMD, built for HIP (`build-rocm/` present):
  ggml/src/iqk/iqk_gemm_{kquants,iquants,iqk_quants,ktquants,1bit,floats,legacy_quants}.cpp
  ggml/src/iqk/iqk_mul_mat.cpp (+ iqk_topk_moe), iqk_flash_attn.cpp, iqk_kda.cpp
  custom 4x4 / 8x8 matmul layouts (Q4_0_4_4/4_8/8_8)

Relevance: this is the AMD/HIP analogue of what ROCmFPX PR#42 added on Vulkan - a dedicated,
hand-tuned quantized matmul (incl. routed/MoE topk) rather than generic dequant-and-dot. For
our engine's MoE decode on gfx1201/gfx1031, IQK is the reference to mine (kernel structure,
packed layouts, expert handling), not the ROCmFPX GLSL.

Decision: ROCmFPX = charlie's fork only (not portable, Vulkan here). IQK = study for a fast
routed quantized matmul in our HIP engine; its 4x4/8x8 layouts are the analog of the
"dedicated routed mmvq" idea.

## 17c. ROCmFPX quant type codes (public enum) - exact map

From charlie12345/ROCmFPX `ggml/include/ggml.h` (enum ggml_type):
    100 GGML_TYPE_Q4_0_ROCMFP4        (UE4M3 scales + packed AMD FP4)
    101 GGML_TYPE_Q4_0_ROCMFP4_FAST   (single-scale speed layout)
    102 GGML_TYPE_Q6_0_ROCMFPX        (6-bit UE4M3-scale)
    103 GGML_TYPE_Q8_0_ROCMFPX        (8-bit UE4M3-scale)
    104 GGML_TYPE_Q3_0_ROCMFPX        (3-bit UE4M3-scale)
    105 GGML_TYPE_TURBO3_0            (TurboQuant 3-bit KV-cache, 3.5 bpw)
    106 GGML_TYPE_TURBO4_0            (TurboQuant 4-bit KV-cache, 4.5 bpw)
    107 GGML_TYPE_Q2_0_ROCMFPX        (2-bit S40 codebook + dual UE4M3)
    108 GGML_TYPE_Q4_0_ROCMI4         (native signed 4-bit + UE4M3)
    109 GGML_TYPE_COUNT

Maps our files:
- Qwen3.8-Distill-35B-A3B-Q2KXL_ROCMFPX: 102 (Q6_0) + 107 (Q2_0)
- ornith-1.0-35B-Q3_0_ROCMFPX: 101 + 102 + 104 (Q3_0)
- Spark-X2.5-4B / granite-4.1-3b / gemma-4-E2B / Ornith-9b ROcmFP4: 100, 101
- Ternary-Bonsai-2-27B-PQ2_0: type 142 -> NOT in this enum (max 108); different fork, not ROCmFPX.

Key consequences:
1) The codes are public. Loading these in OUR engine = a port: add the enum values + the block
   structs (ggml-common.h) + ggml_type_size/ggml_blck_size entries + dequant/MMVQ/GEMV kernels
   to the HIP backend. Bounded, fully specified, not a mystery. NOTE: our pool's tt_bytes now
   defers to ggml_type_size, so registering the enum+sizes alone makes the pool size these
   experts; the kernels are what actually run them.
2) TURBO3_0/TURBO4_0 (105/106) are the low-bit KV-cache quant we flagged as our missing
   capacity lever (our pin only has f16/q8_0/q4_0/q5_0/iq4_nl). Same repo -> the KV-compression
   path is available as types 105/106 if we port them.
3) GGML_TYPE_COUNT=109 (not 143), so any file with type >=109 (e.g. Ternary-Bonsai 142) comes
   from a different quantizer fork.

## 20. Porting TurboQuant KV + ROCmFP4 into our HIP engine - spec + staged plan

Scope reality: this is a multi-stage kernel port into the vendored ggml HIP backend, not a
flag. Each type needs: enum + block struct (ggml-common.h) + ggml_type_size/ggml_blck_size +
traits/name table (ggml.c) + a dequant kernel + an MMVQ (GEMV) path + (for MoE) a MUL_MAT_ID
path + (for KV) KV-cache dtype integration. Our vendor tree is never patched in place
(patch band). No stubs: a type is either fully loadable+runnable or not registered.

### TurboQuant KV (types 105 TURBO3_0 / 106 TURBO4_0) - data model (byte-exact, from source)
    block_turbo3_0 { ggml_half d; uint8_t qs[12]; }  sizeof=14  -> 32 vals, 3.5 bpw
    block_turbo4_0 { ggml_half d; uint8_t qs[16]; }  sizeof=18  -> 32 vals, 4.5 bpw
    d = FP16 L2-norm; qs = packed 3-bit (turbo3) / 4-bit (turbo4) codebook indices.
    QK=32, QR=2. (Codebook tables + the actual dequant formula still to pull: the header only
    defines the block; the kvalue table + to-from-float live in ggml-quants.c.)

### ROCmFP4 (types 100/101) - data model
    Block structs are defined in the same ggml-common.h but were in the truncated half of the
    fetch. Still to retrieve before implementing. Known: UE4M3 scale(s), packed FP4 codes,
    32-element blocks; _FAST = single-scale layout.

### Staged plan (each stage independently verifiable; stop if a stage cannot be completed clean)
    Stage A  TurboQuant KV (105/106): enum + block struct + size/blck table + ggml_type_name +
             dequant(block->f32) in ggml-quants + ggml_get_rows/vec_dot + KV-cache accept path.
             Verify: quantize a vector, dequant, assert round-trip vs reference; then KV at ctx.
    Stage B  ROCmFP4 (100/101): same skeleton + the repack/MMVQ kernel for decode; the MoE
             MUL_MAT_ID path if a routed model is the target.
    Stage C  (optional) ROCmFP2/FP3/FP6/FP8 + the SWAR FP2 unpack for the low-bit disk tier.

Note: our pool's tt_bytes already defers to ggml_type_size, so registering the enum+sizes makes
the pool SIZE these experts automatically; the kernels are what actually run them. For the KV
lever specifically (Stage A) no MoE kernel is needed - it is a KV cache dtype + dequant.
