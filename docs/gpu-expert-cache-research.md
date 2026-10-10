# GPU-resident expert caching: running a >VRAM MoE entirely on GPU

Research notes + integration plan. Goal: run an 18 GB MoE fully on GPU on a 12 GB card by
keeping a **hot-expert pool in VRAM** and copying experts D2D, instead of (a) computing
experts on the CPU or (b) reading them over PCIe per token.

## The core idea
MoE activates a tiny fraction of experts per token. Keep non-expert weights + KV + a
**VRAM expert pool** resident; experts live in RAM (or disk) and are pulled into the pool
only when routed. On a cache hit the expert is copied into the GPU compute buffer via
**D2D (~500 GB/s)** rather than PCIe H2D (~25 GB/s) or CPU compute — ~20x per byte.
Working set = active experts per token, which scales with *active* params, not total.

## Public implementations (verified, both real code)

### FATE — ongunm/llama-moe-cache (closest to our fork)
~500 LoC C++ extension to llama.cpp, 5 files. Dual license AGPL-3.0 / MIT.
- `src/llama-fate.{h,cpp}` (new): GPU pool, prefetcher, prediction state.
- `ggml/src/ggml-backend.cpp`: expert-hook callback in the ggml scheduler (intercepts
  per-expert copies for caching).
- `ggml/src/ggml-cuda/ggml-cuda.cu`: prefetch stream, async H2D, GPU-side barriers.
- Prediction: **cross-layer** (experts at layer N predict N+1) + **temporal** (layer N
  last token predicts layer N this token). Zero-shot heuristic; no training/learning.
- Measured: Qwen3-30B-A3B Q4_K_M (18.6 GB) on RTX 4070 Ti **12 GB**: 33.74 -> **64.45
  t/s** (1.91x), **99.50% hit** (75690/384). Mixtral-8x7B 99.94%.
- Limitation: prefill slower (experts processed individually); `--no-mmap` needed for
  pinned DMA; LRU only.

### moe-l2 — yalun753/moe-l2 (Apache-2.0, larger, Python proxy + CUDA patches)
- on-demand pin (lazy mmap, zero VRAM) + host-pinned `cudaHostRegister` so the GPU reads
  experts directly via PCIe DMA; **A3 LRU expert cache** in VRAM (up to 32768 slots);
  selective pin via a router map (top-K experts/layer). Cold-page eviction caps RSS.
- Measured (RTX 3060 12G): Qwen3.6-35B-A3B UD-IQ2_M (11.5 GB) 25-29 t/s @256K ctx, VRAM
  8.7 GB. 4090: DS-V2-Lite 139-154 t/s. DeepSeek-V4-Flash (157B, 85 GB file) on 10-11 GB.
- **Key finding (directly relevant):** cache benefit = **expert size x hit rate**. DS-V2-Lite
  (1.55 MB/expert, top-6) wins big; Qwen (~1 MB) neutral; Mixtral (252 MB, top-2) never.
  => a VRAM pool only pays when expert size is non-trivial and routing is skewed.
- Windows-native binaries exist; but the docs here are NVIDIA/CUDA.

## What this means for *our* models (working-set math)
Per-token expert bytes = K * per-expert-size * layers.

| model | expert bytes | layers | experts | K | per-expert | working set/token |
|---|---:|---:|---:|---:|---:|---:|
| ornith-1.0-35B Q3_0 (18 GB) | 17.85 GB | 40 | 256 | 8 | 1.74 MB | **~557 MB** |
| Qwen3.5-35B Q4_K (19.7 GB) | 18.04 GB | 40 | 256 | 8 | 1.76 MB | **~563 MB** |
| Qwen3-30B i1-Q2_K | ~7 GB | 48 | 128 | 8 | ~0.5 MB | ~200 MB |

So a **1-2 GB VRAM expert pool** holds the hot set with margin; dense (~1.6 GB) + KV +
pool fits well inside 12 GB. Expert sizes (1.7 MB) are in the FATE regime where a pool
pays (unlike Mixtral's 252 MB). Temporal+cross-layer hit should approach FATE's 99%.

## Why our fork is unusually well-positioned
Two primitives already exist:
- **`ggml_edge0_mmid_base(src0, id, nb02)`** expert-base hook (patch band #6) — the exact
  interception point an expert cache needs; today it serves bytes from the CPU L1 pool.
- **`windows/serve/prefetch.cc`** — an L1 expert pool with frequency eviction, step-boundary
  re-touch, and a selfcheck. Today it pools *pages in RAM*; the same machinery retargeted to a
  **VRAM pool** with D2D copy is the FATE design.

## Integration plan (staged, no stubs)
1. **VRAM pool + D2D copy (Stage A, the win).** Allocate a persistent VRAM pool; on an
   expert access, if present, point the compute at the pooled copy (D2D) else H2D and
   insert. Hook = extend the existing mmid path on the GPU backend (CUDA), mirroring
   FATE's `ggml-backend.cpp` scheduler hook. Evict by frequency (reuse our decayed-freq
   score). Validation: cache hit %, decode tok/s vs the current CPU-expert and auto-fit
   baselines on ornith-35b / Qwen3.5-35B.
2. **Temporal + cross-layer prediction (Stage B).** Prefetch predicted experts on a
   dedicated stream to overlap H2D with compute (FATE). Reuse `prefetch.cc`'s teacher
   routing observation as the signal.
3. **Pinned host + `--no-mmap` (Stage C).** Enable direct async DMA (FATE's noted fix).
4. **Prefill bypass.** Don't per-expert-cache during prefill (FATE's known slowdown) —
   keep the batch copy path for prompt tokens.

## Honest risks
- The hook is in the CUDA backend (device expert copy), not the CPU path we use today; that
  is the integration effort, not the algorithm.
- A pool only pays for skewed routing and non-trivial expert size — true for our 35B
  (top-8, 1.7 MB) but must be measured, not assumed (moe-l2: Qwen ~1 MB was neutral).
- moe-l2/FATE are NVIDIA-first; our target is HIP/RDNA2+RDNA4 — porting the copy/stream
  code to HIP is required (the D2D/H2D primitives exist in HIP).

## References
- FATE: github.com/ongunm/llama-moe-cache ; FATE paper arXiv:2502.12224; HOBBIT arXiv:2411.01433
- moe-l2: github.com/yalun753/moe-l2
- REAP (expert pruning, complimentary): arXiv:2510.13999

## KV-type wiring map (verified against the tree)

The flash-attention path only uses a factory for KV types that has a **vec-kernel
instance**. Adding one KV type touches 5 places (traced 2026-10):

1. `ggml_cuda_fattn_kv_type_supported(type)` — `ggml/src/ggml-cuda/fattn.cu:513`.
2. `get_vec_dot_KQ<type_K,D,nthreads>()` — `fattn-common.cuh:620` (K side, `Q·K`).
3. `get_dequantize_V<type_V,T,ne>()` — `fattn-common.cuh:643` (V side).
4. `EXTERN_DECL_FATTN_VEC_CASES` list + `EXTERN_DECL_FATTN_VEC_CASES(D,type_K)` —
   `fattn-vec.cuh:578`.
5. a `template-instances/fattn-vec-instance-<K>-<V>.cu` pair (generated file:
   `generate_cu_files.py`; add by hand or rerun the generator).

Already wired (works today): `F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0`. That is why
`-ctk q8_0 -ctv q4_0` runs and gives the +24% decode at 32k (RUN-019).

Remaining:
- **FP4/MXFP4 KV** — wireable like Q4_0 (non-rotated). Add `vec_dot_fattn_vec_KQ_mxfp4`
  (codebook `kvalues_mxfp4`, per-32 `e` scale) and `dequantize_V_mxfp4`, plus the 4 hooks
  + 7 `.cu` files (F16/BF16/Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 as the V partner set). Alignment:
  32-value blocks; K row length must be a multiple of 32. This is the DeepSeek-V4.1-Flash
  FP4-KV direction (arXiv:2609.19969), bounded and reuses an in-tree codebook.
- **TurboQuant (turbo3/turbo4)** — NOT wireable like Q4_0. Its per-128 FWHT rotation
  couples K and Q: `vec_dot_fattn_vec_KQ_*` computes `Q·K` directly, so a rotated K needs
  **Q rotated into the same basis first** (a Q-side path change) or an inverse-FWHT before
  the dot. The type foundation (patch 0007: types 49/50, block structs, FWHT quant/dequant,
  gcc round-trip verified) is done; the attention wiring is a separate, larger effort.

