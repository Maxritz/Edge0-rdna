# Native RDNA2 HIP Backend for RX 6700 XT

## Method-level replacement specification

**Target GPU:** AMD Radeon RX 6700 XT  
**LLVM target:** `gfx1031`  
**Architecture:** RDNA2 / GFX10.3  
**Primary programming model:** HIP C++ with AMDGPU-native wave, LDS, VALU, SALU, DPP and packed dot-product instructions  
**Scope:** Replace NVIDIA-derived CUDA execution methods with an AMD-first implementation for the GGML operator surface in `LLAMA_CUDA-FUNCTION-TABLE.md`  
**Design objective:** A native AMD backend for RDNA2, not the CUDA backend compiled through HIP compatibility headers  
**Software baseline under investigation:** User's ROCm/HIP 7.16 Windows environment; exact package and component support must be verified on the target machine  
**Document date:** 17 November 2026

---

## 1. Purpose and non-goals

The objective is to implement a backend whose algorithms, work partitioning, data movement, memory layout, arithmetic, synchronisation and dispatch are designed for RDNA2 `gfx1031`.

This is not a request to mechanically translate:

- CUDA runtime calls into HIP runtime calls;
- cuBLAS calls into rocBLAS calls;
- CUDA warp intrinsics into similarly named AMD intrinsics;
- CUDA shared-memory tiles into LDS tiles without redesigning the data layout;
- NVIDIA asynchronous-copy assumptions into ordinary HIP copies;
- Tensor Core matrix fragments into a software imitation on hardware that has no RDNA WMMA facility.

HIP is the host/device programming and runtime interface. The hot-path compute methods must be AMD-native and selected according to the actual `gfx1031` hardware, workload shape, data type and numerical contract.

RDNA2 is not a smaller RDNA4. The absence of WMMA, SWMMAC and GFX12 transpose-load operations changes the matrix and attention implementation. The design must exploit RDNA2's packed VALU and dot-product instructions instead of pretending that matrix-accelerator instructions exist.

### 1.1 Explicit non-goals

- Do not retain `ggml-cuda` source files as the main HIP backend implementation.
- Do not make CUDA compatibility headers the foundation of the backend.
- Do not make rocBLAS mandatory for every matrix operation.
- Do not assume rocWMMA works on `gfx1031`. The current rocWMMA supported-target list excludes `gfx103x`.
- Do not implement a fake WMMA layer around ordinary VALU instructions and present it as hardware matrix acceleration.
- Do not change GGUF quantisation formats or reinterpret packed quantised tensors as ordinary INT4 or INT8 matrices.
- Do not silently replace FP32 reference execution with FP16, BF16, integer-dot or another lower-precision path.
- Do not claim an operation is GPU accelerated merely because the backend accepts it. Prove that the intended `gfx1031` kernel executes.
- Do not silently fall back to CPU for an operator marked as supported by the backend.
- Do not infer full ROCm library support from successful HIP runtime device enumeration.

rocBLAS may be used as an optional comparison where a compatible build is available. It must not dictate the architecture of the hot path. Native HIP kernels are the primary implementation target.

---

## 2. RX 6700 XT hardware model

The Radeon RX 6700 XT is an RDNA2 device with LLVM target `gfx1031`. AMD's GPU specifications list 40 compute units, 12 GiB VRAM, wavefront sizes of 32 or 64, 128 KiB LDS capacity, and 96 MiB Infinity Cache. AMD lists up to 384 GB/s external memory bandwidth and 13.21 TFLOP/s peak FP32 vector performance for this product. These are hardware specifications, not expected LLM performance.

| Property | RX 6700 XT reference value | Engineering implication |
|---|---:|---|
| Architecture | RDNA2 | Use GFX10.3 instructions and features only |
| LLVM target | `gfx1031` | Compile and inspect specifically for this target |
| Compute units | 40 | Measure wave and workgroup distribution across the device |
| Stream processors | 2,560 | VALU throughput is a major compute resource |
| VRAM | 12 GiB GDDR6 | Residency and model placement are first-class concerns |
| Memory interface | 192-bit | Coalescing and transaction efficiency matter |
| External memory bandwidth | Up to 384 GB/s | Theoretical ceiling, not a guaranteed application result |
| Infinity Cache | 96 MiB | Reuse and working-set behaviour must be profiled |
| LDS capacity | 128 KiB reported by AMD's GPU architecture table | Per-workgroup allocation and occupancy must be queried and measured |
| Wavefront modes | Wave32 and Wave64 | Select by kernel family and shape; do not impose one mode globally |
| Peak FP32 vector throughput | Up to 13.21 TFLOP/s | Theoretical reference only |
| Peak FP16 vector throughput | Up to 26.43 TFLOP/s | Packed FP16 VALU capability, not a matrix-unit throughput claim |

Record actual clocks, memory state, power limits, driver, operating system, HIP package, compiler version and GPU properties for every benchmark. Board-partner clocks and power limits may differ.

### 2.1 RDNA2 compute capabilities that matter

RDNA2 has useful packed arithmetic and dot-product ALU instructions. The AMD RDNA2 ISA describes the following inference-relevant instruction families:

| Instruction family | Operation | Backend use |
|---|---|---|
| `V_DOT2_F32_F16` | Two packed FP16 products accumulated into FP32 | FP16 dot-product and selected low-precision GEMV/GEMM paths |
| `V_DOT2C_F32_F16` | Accumulating form of packed FP16 dot product | Repeated FP16 dot accumulation where compiler lowering is suitable |
| `V_DOT2_I32_I16` | Two signed INT16 products accumulated into INT32 | Packed integer dot experiments |
| `V_DOT2_U32_U16` | Two unsigned INT16 products accumulated into UINT32 | Unsigned packed integer dot experiments |
| `V_DOT4_I32_I8` | Four signed INT8 products accumulated into INT32 | INT8 quantised dot kernels |
| `V_DOT4C_I32_I8` | Accumulating signed INT8 dot form | INT8 accumulation where generated ISA is beneficial |
| `V_DOT4_U32_U8` | Four unsigned INT8 products accumulated into UINT32 | Unsigned INT8 dot kernels |
| `V_PK_FMA_F16` and related packed operations | Two FP16 arithmetic lanes in one VALU instruction | Packed FP16 arithmetic |

LLVM's GFX10.3 instruction reference also lists INT4 dot8 instructions for the `gfx1030` instruction family. Treat `V_DOT8_I32_I4` and `V_DOT8_U32_U4` as candidate `gfx1031` optimisations: compile a minimal kernel for `gfx1031`, inspect the emitted ISA and run correctness tests before making them a required backend dependency.

These are packed vector ALU operations. They are not WMMA, MFMA or Tensor Core-style matrix engines. A matrix multiplication kernel still has to organise the work, load operands, distribute values among lanes, handle accumulation and reuse data explicitly.

### 2.2 Wavefront policy

Do not globally force every kernel to wave32 or wave64.

- Wave32 is a natural starting point for many independent vector tasks and short reduction paths.
- Wave64 may help some reduction, memory and compute kernels by increasing work per wave, but can also change register pressure and scheduling behaviour.
- Packed dot-product instructions are per-lane arithmetic operations. They do not turn a wave into a matrix-fragment engine.
- Benchmark wave32 and wave64 for GEMV, quantised dot, reductions, attention and memory kernels where the compiler supports both modes.
- Record the actual generated wave mode, VGPR count, SGPR count, LDS allocation, spills, active waves and kernel time.

The RX 6700 XT is `gfx1031`, not `gfx1030`. Shared ISA-family documentation is useful, but the installed compiler must accept the target and emit the expected instructions for `gfx1031` itself.

---

## 3. AMD-native execution primitives

The backend should expose an internal abstraction over AMD-specific primitives. The abstraction should describe semantics and data layouts, not imitate CUDA names.

| RDNA2 mechanism | Intended use | Design rule |
|---|---|---|
| Wave32 / Wave64 | Lane-level work partition | Select per kernel family and shape |
| VALU | Per-lane arithmetic, unpacking, reductions and accumulation | Primary arithmetic engine for dense and quantised kernels |
| SALU / SGPR | Wave-uniform control, offsets, descriptors and metadata | Keep uniform decisions uniform where possible |
| LDS | Workgroup-local reuse, staging, reductions and exchange | Use only when reuse or communication savings exceed staging cost |
| DPP | Selected lane permutations and reductions | Use only for supported patterns and inspect generated ISA |
| `ds_bpermute` | Cross-lane data exchange through LDS datapath | Compare against DPP and conventional LDS exchange |
| `ds_swizzle` | Structured lane/data rearrangement | Use for layouts that match its supported patterns |
| Wave ballot and lane masks | Compaction, masks and predicated work | Preserve exact active-lane semantics |
| `mbcnt` | Lane indexing and compacted-position calculation | Use where it simplifies a measured hot path |
| `V_DOT2_F32_F16` | Packed FP16 dot product | Candidate for eligible FP16 arithmetic, never a silent FP32 replacement |
| `V_DOT4_I32_I8` | Packed INT8 dot product | Candidate for INT8 quantised operations after signedness and scale handling |
| `V_DOT8_I32_I4` | Packed INT4 dot product | Experimental until compiler and ISA validation pass on `gfx1031` |
| `V_PK_FMA_F16` | Two packed FP16 fused multiply-add operations | Use in FP16 paths where precision is acceptable |
| `s_waitcnt` | Wait for outstanding memory or export operations | Place according to the actual dependency and ISA rules |
| Workgroup barriers | LDS producer/consumer ordering | Use only where there is cross-wave or LDS dependency |
| HIP streams and events | Submission and dependency management | Use explicit dependencies and measure actual overlap |
| HIP Graphs | Reusable launch sequences | Adopt only when replay reduces measured CPU submission cost on the installed runtime |

### 3.1 DPP and lane-exchange strategy

Use AMD wave operations for:

- row and column reductions;
- quantisation scale reductions;
- short scans and prefix sums;
- small transposes;
- dot-product operand preparation;
- attention score reductions;
- top-k and argmax stages;
- routing metadata movement.

Build at least three implementations for important reduction patterns:

1. DPP or wave-intrinsic implementation;
2. LDS exchange implementation;
3. direct VALU implementation with compiler-generated lane movement where applicable.

Compare output correctness, instruction count, dependency chains, LDS traffic, VGPR use and kernel time. A shorter source implementation is not evidence of a faster kernel.

### 3.2 LDS strategy

LDS is workgroup-shared memory. It is not a direct asynchronous global-memory transfer engine.

Candidate tile pipeline:

```text
Global memory
    |
    v
Coalesced vector loads
    |
    v
VGPR staging / unpack / layout transform
    |
    +------------------------------+
    |                              |
    v                              v
Direct VGPR operand path        LDS staging
    |                              |
    |                         Workgroup barrier
    |                              |
    +---------------+--------------+
                    |
                    v
             VALU / packed dot
                    |
                    v
                Accumulate
```

Test direct VGPR consumption against LDS staging. For every tile shape, record:

- global bytes read;
- LDS bytes written and read;
- LDS allocation per workgroup;
- VGPR and SGPR counts;
- spills;
- active waves/workgroups;
- barriers and `s_waitcnt` behaviour;
- total kernel time.

Do not assume that double-buffered LDS is faster. A direct register path may beat a staged path for small or bandwidth-sensitive workloads. Conversely, LDS may pay off when multiple waves reuse a tile or when lane exchange is expensive.

### 3.3 Global-to-LDS and asynchronous movement

Do not design the `gfx1031` backend around NVIDIA `cp.async`, TMA or a presumed direct asynchronous global-to-LDS operation.

The baseline RDNA2 method is:

1. issue coalesced global loads into VGPRs;
2. unpack or rearrange values in VGPRs;
3. write to LDS only when reuse or cross-wave exchange requires it;
4. use the appropriate memory wait and workgroup barrier;
5. consume the tile;
6. reuse the buffer only after its readers have completed.

Host-to-device copy overlap is a separate runtime-level question. Test pinned host buffers, `hipMemcpyAsync`, streams and events on the actual driver. Do not confuse an asynchronous host/device copy with a shader instruction that transfers global memory directly into LDS.

For device-side load/compute overlap, use software pipelining, independent workgroups, multiple resident waves and double/triple buffering where measurements justify the registers, LDS and barriers. Prove overlap with a profiler timeline.

### 3.4 No RDNA4-only matrix or load instructions

The `gfx1031` path must not contain dependencies on:

- RDNA3/RDNA4 WMMA builtins;
- SWMMAC or 2:4 WMMA sparse operations;
- GFX12 global transpose-load builtins;
- GFX12.5 global-load-to-LDS facilities;
- CDNA MFMA or CDNA Data Movement Engine facilities.

For matrix operand transposes, use explicit vector permutation, DPP, `ds_bpermute`, LDS staging or a layout that avoids the transpose. Choose among them by measurement.

---

## 4. Matrix execution: RDNA2 packed VALU and dot-product engine

### 4.1 No WMMA on `gfx1031`

RDNA2 does not provide the RDNA3/RDNA4 WMMA execution facility and is not a supported target for rocWMMA. It also does not provide CDNA MFMA. Do not build this backend around matrix fragments or a hardware matrix tile abstraction.

The replacement is a shape-specialised VALU matrix engine:

- partition output rows/columns across lanes and workgroups;
- keep partial accumulators in VGPRs;
- use packed FP16 or integer dot instructions where the input representation qualifies;
- use ordinary FP32 FMA for FP32 parity;
- stage reusable operand tiles in LDS only when reuse justifies it;
- tune wave mode, K loop, unrolling and tile dimensions independently;
- maintain separate GEMV and GEMM dispatch paths.

This design can still be highly parallel. It simply obtains parallelism from the vector ALUs, packed dot operations, wave scheduling and explicit data reuse rather than a matrix accelerator.

### 4.2 Matrix methods by data type

| Workload/input | RDNA2 method | Restrictions |
|---|---|---|
| FP32 GEMV | Wave-partitioned VALU FMA | Preserve FP32 reference semantics and accumulation order where required |
| FP32 GEMM | Tiled VALU FMA with VGPR accumulators and optional LDS reuse | No WMMA; tile sizes must limit register pressure |
| FP16 GEMV | Packed FP16 dot2 or packed FP16 arithmetic | FP32 accumulation path must be explicitly validated |
| FP16 GEMM | Tiled packed dot2 / packed FP16 FMA | Not a matrix-fragment operation; organise reductions and data reuse explicitly |
| INT8 GEMV/GEMM | `V_DOT4_I32_I8` or unsigned variant | Handle signedness, zero points, scales, tails and overflow exactly |
| INT4 GEMV/GEMM | Candidate `V_DOT8_I32_I4` / unsigned variant | Require successful `gfx1031` compile, ISA inspection and numerical tests |
| BF16 input | Convert/decode to a supported compute representation, usually FP32 or FP16 if allowed | No native RDNA2 BF16 dot instruction in this design; quantify conversion and precision effects |
| FP8/BF8 input | Software decode to a supported compute type | No native RDNA2 FP8/BF8 dot path |
| GGUF Q4/Q5/Q6/IQ | Format-specific unpack, scale and dot/FMA kernel | Never treat the packed GGUF block as a plain contiguous integer matrix |

### 4.3 Quantised matrix execution

The dot instructions operate on packed arithmetic values, not on GGUF quantisation block semantics. Each quantisation format still requires its own decoding rules.

For each supported format:

1. load packed blocks coalescently;
2. decode scales, minimums, zero points and lookup metadata;
3. unpack values into packed operands or VGPR fragments;
4. use integer dot instructions only where signedness and representation match;
5. apply block scales and affine corrections;
6. accumulate in the required precision;
7. handle tails and block boundaries;
8. compare against the trusted CPU dequantise-and-multiply reference.

Do not force every format through the same INT4 path. Q4_0, Q4_1, Q4_K, Q5_K, Q6_K and IQ formats have different packed layouts and scale semantics.

### 4.4 Matrix dispatch policy

Maintain separate dispatch families:

- `gemv_f32`: exact FP32 vector path;
- `gemv_f16_dot2`: FP16 packed dot path;
- `gemv_q8_dot4`: INT8 packed dot path;
- `gemv_q4_dot8`: experimental INT4 dot path;
- `gemv_quant_valu`: generic format-specialised decode and VALU path;
- `gemm_f32_valu`: tiled FP32 VALU GEMM;
- `gemm_f16_dot2`: tiled FP16 dot2 GEMM;
- `gemm_q8_dot4`: tiled INT8 dot4 GEMM;
- `gemm_q4_dot8`: experimental INT4 dot8 GEMM;
- `gemm_quant_valu`: format-specific tiled quantised GEMM.

Dispatch should use shape, batch, K dimension, format, stride, alignment, wave mode and measured kernel performance. Never select an integer-dot kernel merely because its input is described as Q4 or Q8.

---

## 5. Method replacement map for the uploaded GGML CUDA function table

The following map covers the operator and kernel families in `LLAMA_CUDA-FUNCTION-TABLE.md`. It specifies the intended RDNA2 method, not a claim that every implementation already exists.

### 5.1 Main backend, device and memory management

| Existing CUDA area | Native RDNA2 implementation |
|---|---|
| Main CUDA backend context | Independent `ggml-hip-rdna2` backend context |
| Device enumeration | HIP device enumeration plus `gfx1031` validation |
| CUDA stream pool | Small HIP stream set with measured submission and overlap |
| CUDA event dependencies | HIP events and explicit dependency tracking |
| CUDA memory pool | HIP device allocator plus backend-owned suballocator |
| CUDA VMM pool | Use HIP virtual memory only after verifying runtime support on the exact build; otherwise bounded `hipMalloc` arena |
| Pinned host pool | HIP host allocation/registration where supported and measured |
| CUDA host buffer registration | HIP host registration with explicit lifetime tracking |
| CUDA graphs | HIP Graphs as optional launch-overhead optimisation, not a mandatory execution model |
| Device capability table | Runtime `gfx1031` capability record, ISA-probe results and kernel registry |
| Multi-device all-reduce | RCCL only if a compatible package and topology are available; otherwise out of initial single-GPU scope |

### 5.2 Matrix operations

| Existing CUDA area | Native RDNA2 implementation |
|---|---|
| cuBLAS GEMM fallback | Native shape-specialised VALU GEMM; optional rocBLAS comparison if compatible |
| MMF FP32 | Tiled FP32 VALU FMA with register accumulators |
| MMVF FP32 vector | Wave-partitioned FP32 GEMV |
| MMVQ quantised vector | Format-specialised unpack plus dot2/dot4/dot8 or VALU accumulation |
| MMQ quantised batch | Tiled quantised VALU/dot-product GEMM with optional LDS reuse |
| MMID / expert matrix multiply | Grouped expert GEMV/GEMM, compacted token lists and format-specific kernels |
| Outer product | Dedicated VALU outer-product kernel |
| Batched / strided GEMM | Separate dispatch for small-batch and large-batch shapes; avoid a generic kernel for all sizes |

### 5.3 Quantisation, unpacking and conversion

| Existing CUDA area | Native RDNA2 implementation |
|---|---|
| Device dequantisation helpers | Format-specific VGPR unpack and scale reconstruction |
| Q4 family | Nibble unpack, block scale and zero/min handling, then integer dot or VALU |
| Q5 family | High-bit reconstruction plus format-specific scale handling |
| Q6_K | Dedicated packed decode and dot/FMA path |
| Q8 family | Packed byte loads and signed/unsigned dot4 where the representation matches |
| IQ formats | Lookup/table decode or specialised bit extraction, then VALU accumulation |
| F16/F32/BF16 conversion | Dedicated vector conversion kernels; preserve the original data type and conversion rules |
| Quantise | Per-format quantisation with explicit scale calculation, rounding and clipping tests |
| Temporary dequantisation | Avoid full-tensor expansion; decode tiles immediately before arithmetic |

### 5.4 Attention and KV cache

| Existing CUDA area | Native RDNA2 implementation |
|---|---|
| Flash Attention MMA | Replace with online-softmax attention using VALU dot products and wave/LDS reductions; no WMMA path |
| Flash Attention tile | Workgroup-tiled QK/softmax/PV implementation with explicit LDS layout |
| Flash Attention vector | Wave32/wave64 vector attention for small head dimensions and decode |
| Flash attention dispatcher | Dispatch by head dimension, query length, context length, mask, type and supported layout |
| Attention support query | Report only combinations with a registered and validated RDNA2 kernel |
| QK dot product | FP32 FMA or packed FP16 dot2 where precision contract permits |
| PV multiply | VALU accumulation with staged reuse; use packed dot2 only for eligible FP16 paths |
| Softmax | Stable online max and denominator accumulation using wave reductions and LDS across waves |
| Causal mask | Fused score-mask path with exact boundary semantics |
| KV cache write | Coalesced stores, optionally fused with RoPE when layout and dependencies permit |
| Paged KV cache | Page-table lookup, coalesced traversal and bounded page metadata |
| KV gather/scatter | Dedicated kernels with ownership, bounds and duplicate-index rules |

Build three independent attention families:

**Decode attention**

- Single-query-token path.
- Paged KV traversal.
- Low launch overhead.
- Wave reductions for max and sum.
- Avoid materialising the full attention score matrix.
- Prioritise cache locality and KV bytes read.

**Prefill attention**

- Tiled QK dot products using FP32 FMA or packed FP16 dot2 where allowed.
- Online softmax.
- Tiled PV accumulation.
- LDS only where operand reuse or cross-wave reduction benefits justify it.
- Do not emulate WMMA fragments.

**Long-context attention**

- Streaming over KV pages/tiles.
- Online maximum and denominator accumulation.
- Bounded scratch storage.
- Explicit page and context-boundary validation.

Attention acceptance requires output comparison against a trusted reference, causal-mask tests, variable-length context tests, odd dimensions, page-boundary tests and actual kernel-dispatch telemetry.

### 5.5 Fused arithmetic and normalisation

| Existing area | Native RDNA2 method |
|---|---|
| Add/subtract/multiply/divide and broadcasting | Coalesced VALU kernels; fuse compatible graph chains |
| Fused add/multiply | Shape-specialised fused VALU kernels |
| Unary activations | Fused vector kernels for GELU, SiLU, ReLU, sigmoid, tanh and related operations |
| GLU / SwiGLU / GeGLU | Fused activation-gate-multiply kernels |
| RMSNorm / LayerNorm | Wave partial reductions plus controlled LDS/workgroup reduction; fuse adjacent operations where correct |
| Softmax | Stable max-subtract-exp-sum-normalise implementation using wave/LDS reduction hierarchy |
| Scale / softcap | Fuse with adjacent pointwise activation when register pressure remains acceptable |
| RoPE | Fused pairwise rotation and KV write when graph dependencies and data layout permit |
| RMSNorm + MUL + RoPE | Dedicated fused path with reference-equivalent intermediate semantics |
| Copy/duplicate/contiguous | Coalesced vector copy; avoid redundant copies through graph aliasing where safe |

A fusion is accepted only when end-to-end execution improves and numerical results remain within the defined contract. Do not fuse solely to reduce kernel count if register pressure, spills or occupancy make the fused kernel slower.

### 5.6 Reductions, indexing and tensor movement

| Existing area | Native RDNA2 method |
|---|---|
| Sum / sum rows / mean | Wave reduction, then LDS or second-stage reduction as required |
| Argmax / top-k / argsort | Hierarchical candidate reduction and compaction |
| Cumsum | Wave scan plus workgroup carry propagation |
| Get rows | Coalesced gather with index-locality analysis |
| Set rows | Scatter with duplicate-index semantics tested explicitly |
| Add ID | Indexed update kernel with an explicit duplicate-write policy |
| Repeat / repeat-back | Stride-aware broadcast and reduction kernels |
| Concat / pad / upscale / roll | Coalesced copy/transform kernels, fused when graph adjacency makes it beneficial |
| Arange / timestep embedding | Launch-minimised VALU generation kernels |
| Diagonal / triangular / fill / clamp | Lightweight elementwise or structured kernels |
| Count equal | Hierarchical comparison and reduction |
| Solve triangular | Dedicated algorithm only where used and tested; otherwise report unsupported rather than silently falling back |

### 5.7 Convolution and image-like operations

| Existing area | Native RDNA2 method |
|---|---|
| Conv2D | Direct tiled convolution initially; compare im2col plus VALU GEMM only for qualifying shapes |
| Depthwise Conv2D | Dedicated channel/spatial kernel, not generic GEMM |
| ConvTranspose1D/2D | Scatter/gather-aware tiled kernels with boundary tests |
| Im2col / Im2col3D | Fused or tiled lowering; avoid full materialisation when possible |
| Col2im1D | Coalesced scatter/accumulation with correct overlapping-window semantics |
| Pool1D/Pool2D | Direct local reduction kernels |
| Reflect padding | Boundary-specialised indexing and coalesced interior path |

These operations are part of the uploaded CUDA operator surface even if they are not first-priority LLM inference kernels. They must be included in the backend support matrix.

### 5.8 MoE and specialised model operators

| Existing area | Native RDNA2 method |
|---|---|
| TopK-MoE | Router top-k, expert grouping, prefix-sum compaction and grouped expert dispatch |
| MoE weighted reduction | Expert-output reduction using wave/LDS hierarchy |
| Add ID | Indexed token/expert scatter with deterministic duplicate handling |
| Gated Delta Net | Dedicated recurrent state-update kernels; fuse snapshot/cache scatter only after correctness proof |
| GLA | Dedicated gated linear attention state-update and scan kernels |
| RWKV WKV6/WKV7 | Dedicated recurrence kernels with state-layout and sequence-length tuning |
| SSM convolution / scan | Dedicated convolution and scan kernels; fuse activation only when beneficial |
| Lightning indexer | Dedicated index/scoring kernel and reduction; validate exact semantics |
| DSV4 HC pre/comb/post | Separate kernels initially, then measured fusion where dependency and register use permit |
| FWHT | Wave/LDS butterfly transform with size-specific stages |
| Cross entropy forward/backward | Stable reduction-based kernels with exact masking and reduction semantics |
| Optimiser steps AdamW/SGD | Dedicated elementwise/state-update kernels; training support is separate from inference acceptance |

### 5.9 Backward and training-only surface

The uploaded function table includes backward operations and optimiser kernels. Do not silently omit them from the support matrix. Separate the milestones:

- inference-complete backend;
- training/backward-complete backend.

Backward normalisation, backward activation, cross-entropy backward and optimiser steps need their own numerical tests. They should not block the first inference milestones unless the consuming application requires training.

---

## 6. Workload-specific methods

### 6.1 Decode engine

Batch-one decode is usually dominated by weight movement, cache behaviour, quantised dot throughput and launch/synchronisation overhead. RDNA2 has no WMMA path, so optimise the data path and packed arithmetic directly.

Priorities:

1. Keep weights in their original GGUF quantised representation in VRAM.
2. Read packed quantisation blocks coalescently.
3. Decode values and scales directly into VGPRs or the smallest practical working tile.
4. Use INT8 dot4 or validated INT4 dot8 only when packed operands match the instruction semantics.
5. Use FP16 dot2 only for a permitted FP16 compute path.
6. Fuse decode with dot-product accumulation.
7. Avoid large temporary dequantised tensors.
8. Reuse static metadata and scale data where possible.
9. Keep KV reads coalesced and page-aware.
10. Reduce CPU submission gaps and unnecessary synchronisation.

Measure each model layer separately where possible. Report bytes read per generated token, achieved memory bandwidth, Infinity Cache sensitivity, arithmetic time, CPU launch time, inter-token latency and tokens/s.

### 6.2 Prefill engine

Prefill offers more parallelism, but it cannot use RDNA3/RDNA4 WMMA on this GPU. The matrix engine must be built from VALU FMA, packed FP16 dot2 and integer dot instructions.

Priorities:

1. Tiled FP32 VALU GEMM for the exact FP32 path.
2. Packed FP16 dot2 GEMM for eligible lower-precision workloads.
3. Format-aware INT8/INT4 quantised GEMM for supported packed representations.
4. LDS reuse for operands shared across waves or output tiles.
5. Tiled attention with online softmax.
6. Fused QKV projection/RoPE/KV writes where valid.
7. Multi-token and multi-request batching.
8. Avoid launch-per-row and launch-per-small-fragment designs.
9. Tune tiles against VGPR, LDS, wave mode, instruction issue and achieved bandwidth.

Report tokens/s at fixed prompt lengths and batch sizes, and measure time-to-first-token separately from total prefill throughput.

### 6.3 MoE engine

Initial exactness requirement: preserve the same router outputs, top-k expert IDs, routing weights and combination semantics as the reference.

Execution plan:

```text
Token activations
      |
      v
Router and top-k
      |
      v
Build expert/token worklists
      |
      v
Prefix-sum / compact token rows
      |
      v
Check expert tile residency
      |
      +------> Schedule required tile transfers
      |
      v
Grouped expert GEMV/GEMM
      |
      v
Weighted expert combination
      |
      v
Output activations
```

Do not initially alter expert selection to improve cache hit rate. First optimise how the already-selected experts execute. Grouped expert GEMM should use the RDNA2 VALU/dot engine, not an assumed WMMA path.

### 6.4 KV-cache engine

Implement:

- fixed-size page allocation;
- per-request ownership;
- per-page generation and reference counters;
- page-table lookup;
- coalesced page traversal;
- attention-compatible layout;
- safe asynchronous page movement;
- bounded eviction;
- telemetry for hits, misses, page faults and bytes moved.

Do not assume cross-layer KV sharing is valid for conventional transformer models.

---

## 7. Backend architecture

The backend should be independently implemented under a distinct namespace and source tree, while exposing the required GGML backend interface.

```text
GGML graph
   |
   v
RDNA2 graph analyser
   |-- op and tensor metadata
   |-- shape / stride / quantisation inspection
   |-- fusion eligibility
   |-- decode vs prefill classification
   |-- memory residency requirements
   |
   v
RDNA2 dispatch planner
   |-- quantised GEMV dispatch
   |-- quantised GEMM dispatch
   |-- FP32 VALU GEMM dispatch
   |-- FP16 dot2 dispatch
   |-- INT8 dot4 dispatch
   |-- experimental INT4 dot8 dispatch
   |-- attention dispatch
   |-- MoE dispatch
   |-- pointwise/reduction dispatch
   |-- explicit unsupported-op reporting
   |
   v
Native HIP execution layer
   |-- HIP device/context
   |-- allocator and buffer manager
   |-- streams and events
   |-- kernel launch layer
   |-- optional graph replay
   |-- profiling and trace hooks
   |
   v
RDNA2 kernel families
   |-- VALU / SALU
   |-- wave32 / wave64
   |-- DPP / lane exchange
   |-- VGPR / LDS
   |-- packed FP16 dot2
   |-- INT8 dot4
   |-- experimental INT4 dot8
   |-- quantised decode and accumulation
   |-- attention / KV / MoE / recurrence
   |
   v
RX 6700 XT (gfx1031)
```

### 7.1 Suggested source layout

```text
ggml/src/ggml-hip-rdna2/
  CMakeLists.txt
  ggml-hip-rdna2.cpp
  ggml-hip-rdna2.h
  backend/
    device.cpp
    context.cpp
    allocator.cpp
    buffer.cpp
    stream_scheduler.cpp
    graph_replay.cpp
    dispatch.cpp
    capability_probe.cpp
    telemetry.cpp
  kernels/
    common/
      layouts.h
      quant_traits.h
      wave_ops.h
      reduction.h
      validation.h
    elementwise/
      binary.hip
      unary.hip
      activation.hip
      norm.hip
      softmax.hip
      rope.hip
    matvec/
      f32_gemv.hip
      f16_dot2_gemv.hip
      q8_dot4_gemv.hip
      q4_dot8_gemv.hip
      quant_valu_gemv.hip
    matmul/
      f32_valu_gemm.hip
      f16_dot2_gemm.hip
      q8_dot4_gemm.hip
      q4_dot8_gemm.hip
      quant_valu_gemm.hip
    attention/
      decode.hip
      prefill.hip
      online_softmax.hip
      kv_cache.hip
    moe/
      router.hip
      compact.hip
      expert_gemm.hip
      weighted_reduce.hip
    tensor_ops/
      gather.hip
      scatter.hip
      copy.hip
      reduce.hip
      topk.hip
    recurrent/
      gdn.hip
      gla.hip
      rwkv.hip
      ssm.hip
  tests/
    device_smoke/
    isa_probes/
    operator_correctness/
    quant_formats/
    attention/
    moe/
    model_parity/
  benchmarks/
    dot2/
    dot4/
    dot8/
    gemv/
    gemm/
    lds_vs_vgpr/
    wave32_vs_wave64/
```

The INT4 dot8 directory and dispatch path remain experimental until the exact installed `gfx1031` compiler emits the required instructions and the kernels pass tests.

### 7.2 Runtime kernel registry

Each registered kernel should declare:

- GGML operation and sub-operation;
- supported input and output types;
- quantisation formats;
- shape, stride and alignment constraints;
- wave mode;
- required ISA features;
- LDS and VGPR expectations;
- numerical tolerance;
- whether the kernel is exact, lower precision or approximate;
- benchmark status for `gfx1031`;
- fallback policy.

Dispatch must reject unsupported combinations explicitly. A kernel compiled for `gfx1030` or another target is not accepted as proof of a valid `gfx1031` implementation.

---

## 8. No-silent-fallback policy

An operation marked as supported by the RDNA2 backend must execute a registered `gfx1031` kernel.

Every dispatch should emit optional diagnostic telemetry:

```text
operation
selected kernel
input types
quantisation format
shape and strides
wave mode
ISA feature path
GPU launch count
GPU execution time
fallback reason, if any
```

Rules:

1. Unsupported operator: return an explicit unsupported status or use a clearly reported configured fallback.
2. Supported operator: launch the registered HIP kernel.
3. Kernel launch failure: report an error, do not silently rerun on CPU.
4. Experimental dot8 path unavailable: choose the validated dot4 or generic VALU path and record the reason.
5. Runtime target mismatch: fail capability validation.
6. Benchmark mode: fail the test if an intended GPU operator did not execute on the GPU.

A CPU fallback may be retained as a correctness reference. It must not be represented as GPU execution.

---

## 9. Numerical contract and correctness

### 9.1 FP32 parity path

The default correctness path must preserve the original FP32 semantics.

- Use FP32 VALU arithmetic and explicit accumulation rules.
- Do not route FP32 inputs through FP16 dot2 for speed without an explicit opt-in mode.
- Do not change quantisation or dequantisation semantics to make a kernel fit a packed dot instruction.
- Record maximum absolute error, maximum relative error and RMS error.
- Compare logits and greedy next-token decisions on deterministic prompt sets.

### 9.2 Quantised formats

For each GGUF format:

- compare device dequantised values with the trusted CPU implementation;
- test all block types and scale/minimum variants;
- test tails and dimensions not divisible by the kernel tile;
- test negative values and scale extremes;
- compare dot/GEMM output against a high-precision reference;
- verify integer accumulator bounds and signedness;
- verify no packed nibble/byte is read outside the block;
- preserve the original quantisation format in model storage.

The availability of `V_DOT4_I32_I8` or candidate `V_DOT8_I32_I4` does not establish that a GGUF tensor can be fed to it directly. Format unpack and scale application are mandatory.

### 9.3 Integrated model tests

For every supported model family:

- compare logits at selected layers;
- compare final logits;
- compare greedy next-token choice;
- measure output divergence over fixed prompts;
- test different context lengths;
- test batch-one decode and batched prefill separately;
- capture all CPU fallbacks and skipped kernels;
- retain a fixed benchmark model/prompt manifest.

---

## 10. Benchmark and profiling methodology

### 10.1 Required baselines

Every experiment must compare against:

- a simple native HIP implementation;
- the previous native RDNA2 kernel, where one exists;
- the existing project implementation;
- rocBLAS only where an officially or locally compatible build exists and the shapes are comparable;
- CPU reference for correctness, not necessarily performance.

Do not use CUDA performance as the definition of a good AMD algorithm. The baseline is the measured implementation on the same RX 6700 XT and the same software stack.

### 10.2 Microbenchmark protocol

- Warm up the GPU before timing.
- Use GPU events for device execution timing.
- Use a monotonic host clock for end-to-end timing.
- Synchronise at the correct boundary for asynchronous work.
- Run at least 30 repetitions for microbenchmarks.
- Report median, p95 and standard deviation.
- Record GPU clocks, temperature, power state and memory state where available.
- Avoid comparing runs with materially different thermal or power conditions.
- Save raw measurements and generated ISA for instruction-specific experiments.
- Record wave mode, VGPR, SGPR, LDS and spill counts.

### 10.3 Required metrics

| Category | Metric |
|---|---|
| Throughput | Tokens/s |
| Latency | TTFT, inter-token latency p50/p95/p99 |
| Memory | Bytes read/written, achieved bandwidth, cache behaviour |
| Residency | Hit rate, miss rate, eviction count |
| Scheduling | Queue wait, CPU submission time, launch count |
| Concurrency | Transfer/compute overlap from timeline evidence |
| Compute | Kernel time, operations/s where meaningful |
| Occupancy | Active waves/workgroups, VGPR/LDS use, spills |
| Instruction use | Actual emitted dot, DPP, LDS and FMA instructions |
| Quality | Maximum absolute error, relative error, RMS error, output agreement |
| Stability | Failures, hangs, data corruption and long-run drift |

### 10.4 Separate decode and prefill

Decode and prefill must never be collapsed into one headline number.

- Decode: report batch size, context length, model, quantisation, tokens/s and inter-token latency.
- Prefill: report prompt tokens, batch size, tokens/s and time-to-first-token.
- MoE: report active experts, tokens per expert, routing time, expert compute time and expert residency misses.
- Attention: report head dimension, query length, KV length, mask, cache layout and scratch bytes.

---

## 11. Implementation phases and acceptance criteria

### Phase 0 - Target and toolchain validation

Deliver:

- Windows build and runtime compatibility report for the exact installed HIP 7.16 package;
- `gfx1031` device enumeration evidence;
- compiler version and supported target evidence;
- smoke test for allocation, H2D/D2H copy, kernel execution and exact result;
- ISA probe results for FP16 dot2, INT8 dot4 and candidate INT4 dot8.

Exit only when the test binary compiles specifically for `gfx1031`, runs on the RX 6700 XT and the generated ISA has been inspected.

### Phase 1 - Core runtime and no-fallback dispatch

Deliver:

- independent `ggml-hip-rdna2` backend;
- device and context management;
- buffer allocator;
- stream/event handling;
- kernel registry;
- explicit unsupported-op reporting;
- dispatch telemetry.

Exit only when the backend can prove which GPU kernel executed for every supported operation.

### Phase 2 - Quantised decode

Deliver:

- Q4, Q5, Q6, Q8 and required IQ format unpack kernels;
- FP16 dot2 path;
- INT8 dot4 path;
- experimental INT4 dot8 path if `gfx1031` compiler/ISA tests pass;
- generic VALU fallback for every supported quant format;
- batch-one GEMV benchmarks.

Exit only when all enabled formats pass CPU-reference tests and the measured path is GPU-resident.

### Phase 3 - Prefill matrix engine

Deliver:

- FP32 VALU GEMM;
- FP16 packed dot2 GEMM;
- INT8 dot4 GEMM;
- experimental INT4 dot8 GEMM;
- format-aware quantised GEMM;
- tile-size and LDS/VGPR comparison suite.

Exit only when prefill performance improves over the existing project baseline without changing the exact FP32 reference path.

### Phase 4 - Attention and KV cache

Deliver:

- decode attention;
- tiled prefill attention;
- online softmax;
- KV cache allocation and write;
- paged KV reads;
- dispatch coverage and parity tests.

Exit only when attention executes on GPU for supported shapes and no hidden CPU fallback remains.

### Phase 5 - MoE and specialised operators

Deliver:

- routing and top-k;
- expert/token compaction;
- grouped expert compute;
- weighted reduction;
- GDN, GLA, RWKV and SSM kernels required by target models.

Exit only when routing decisions and expert outputs match the reference contract.

### Phase 6 - Fusion, scheduling and residency

Deliver:

- fused pointwise/normalisation/RoPE paths;
- decode/prefill/background scheduling;
- bounded VRAM residency manager;
- asynchronous prefetch and eviction;
- measured launch-overhead reduction.

Exit only when end-to-end model execution improves and the profiler confirms the intended overlap.

### Phase 7 - Full operator coverage

Deliver:

- support matrix covering all required operations in the uploaded CUDA function table;
- explicit inference/training coverage distinction;
- correctness suite for all registered kernels;
- no-silent-fallback test suite;
- complete RDNA2 benchmark report.

Exit only when every operation is either implemented and tested or explicitly reported as unsupported. No operation may be silently skipped.

---

## 12. Initial experiment backlog

| ID | Experiment | Question to answer | Acceptance evidence |
|---|---|---|---|
| R2-00 | `gfx1031` smoke and ISA probe | Does the exact installed compiler generate and run the target instructions? | Build logs, ISA dump and exact correctness |
| R2-01 | Wave32 versus wave64 | Which mode wins per kernel family? | Resource counts and repeatable timing |
| R2-02 | DPP versus LDS reductions | Which exchange method is fastest for each reduction shape? | Correctness, ISA and timing |
| R2-03 | FP16 dot2 | Does `V_DOT2_F32_F16` improve eligible GEMV/GEMM? | ISA proof and FP32-reference error report |
| R2-04 | INT8 dot4 | Does `V_DOT4_I32_I8` improve Q8-like paths after unpack and scale work? | ISA proof, format tests and end-to-end timing |
| R2-05 | INT4 dot8 | Is the GFX10.3 dot8 path available and profitable on this `gfx1031` toolchain? | Successful target compile, ISA and full quant tests |
| R2-06 | VGPR versus LDS operands | When does staging pay for itself? | Bytes moved, register/LDS counts and kernel time |
| R2-07 | Quantised GEMV | Which quant format dominates decode cost? | Per-layer and full-token measurements |
| R2-08 | Quantised GEMM | Which tile and wave layout wins for prefill? | Shape sweep and end-to-end prefill |
| R2-09 | Online attention | Can attention avoid score-matrix materialisation and preserve output? | Numerical parity and bandwidth evidence |
| R2-10 | KV paging | Does page size and layout improve long-context reads? | Context sweep, cache metrics and latency |
| R2-11 | Fused QKV/RoPE/KV write | Does fusion reduce traffic without harmful register pressure? | End-to-end time and ISA/resource inspection |
| R2-12 | MoE grouping | Does grouping selected experts reduce redundant loads and launch cost? | Identical router selections and model output |
| R2-13 | Stream overlap | Does copy/compute overlap occur on the installed driver? | Profiler timeline, not stream count |
| R2-14 | HIP Graph replay | Does replay reduce CPU submission cost? | Host timing and launch timeline |
| R2-15 | Full-model parity | Are all supported operations actually executed on GPU? | Dispatch trace, logits and fixed-prompt test |

---

## 13. Build and verification policy

### 13.1 Exact target

Use `gfx1031` for the RX 6700 XT. Do not silently compile for `gfx1030` or another RDNA2 target and assume the result is equivalent.

Illustrative Linux/ROCm command, where that compiler installation supports the target:

```bash
hipcc --offload-arch=gfx1031 -O3 smoke.cpp -o smoke
```

Illustrative Windows PowerShell capture:

```powershell
Get-ComputerInfo | Select-Object WindowsProductName, WindowsVersion, OsBuildNumber
hipconfig --full
hipcc --version
rocminfo
```

The exact CMake generator, HIP compiler path and linker configuration depend on the installed HIP 7.16 package. The backend build must record those values rather than assume a standard Linux ROCm directory layout.

### 13.2 Windows support caveat

AMD's current Windows HIP SDK hardware table distinguishes runtime enablement from full HIP SDK support. It lists the RX 6700 XT (`gfx1031`) as runtime-supported but not HIP-SDK-supported. This does not invalidate a locally working ROCm/HIP 7.16 build or custom compiler/runtime configuration. It does mean that prebuilt HIP SDK libraries and tools must not be assumed to be officially supported for this GPU on Windows.

For the user's installed environment, record separately:

- HIP runtime enumerates and launches on `gfx1031`;
- HIP compiler accepts `--offload-arch=gfx1031`;
- required headers and device libraries exist;
- each library links against the same runtime/toolchain;
- each required profiler/debugger works with the GPU and OS;
- each kernel executes correctly and produces the expected ISA.

A successful runtime smoke test is not proof that rocBLAS, rocWMMA, RCCL, profiling or debugger components are supported. rocWMMA is not a valid matrix path for `gfx1031` in any case.

### 13.3 ISA verification

For every instruction-specific optimisation:

1. compile a minimal isolated kernel for `gfx1031`;
2. disassemble the code with tools matching the installed compiler;
3. confirm the intended instruction appears;
4. confirm no unsupported instruction or unexpected software expansion appears;
5. run exact input/output tests;
6. benchmark both isolated and integrated workloads.

A source-level builtin call is not proof of a particular instruction being emitted.

---

## 14. Definition of done

The RDNA2 backend is complete when:

- it is an independent HIP backend, not a CUDA source tree compiled through compatibility headers;
- all hot-path matrix methods are designed for VALU and RDNA2 packed dot instructions;
- FP32 parity remains a first-class path;
- all required GGUF quantisation formats have format-aware GPU kernels or are explicitly marked unsupported;
- decode and prefill have separate execution strategies;
- attention and KV operations execute on GPU for their advertised shapes;
- MoE routing preserves the reference's selected experts and weights;
- the backend has explicit support and fallback reporting;
- the generated ISA is inspected for performance-critical kernels;
- all registered kernels pass correctness and stress tests;
- end-to-end benchmarks are repeatable and include raw measurements;
- no CUDA runtime, cuBLAS or NVIDIA execution assumptions are required by the hot path.

Success is not a promised token-per-second number. Success is a traceable improvement over the existing implementation on the RX 6700 XT, with numerical behaviour preserved and no false GPU-acceleration claims.

---

## 15. Technical references

References checked for this specification. Documentation is versioned and may change; use the installed compiler and its matching headers as the final authority for emitted code.

### AMD RDNA2 architecture and ISA

1. AMD, **RDNA 2 Shader Instruction Set Architecture**. Includes RDNA2 dot-product ALU operations and packed FP16 instructions.  
   https://www.amd.com/content/dam/amd/en/documents/radeon-tech-docs/instruction-set-architectures/rdna2-shader-instruction-set-architecture.pdf
2. AMD, **Radeon RX 6700 XT specifications**. 40 CUs, 12 GB GDDR6, 192-bit interface, up to 384 GB/s bandwidth and 96 MB Infinity Cache.  
   https://www.amd.com/en/products/graphics/desktops/radeon/6000-series/amd-radeon-rx-6700-xt.html
3. AMD ROCm, **GPU architecture specifications**. RDNA2 target, compute-unit count, wave modes and LDS specifications.  
   https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html
4. LLVM, **AMDGPU GFX10.3 instruction reference**. Instruction syntax for the `gfx1030` ISA family. Validate all candidate instructions against `gfx1031` locally.  
   https://rocmdocs.amd.com/projects/llvm-project/en/latest/LLVM/llvm/html/AMDGPUAsmGFX1030.html
5. LLVM, **AMDGPU usage and intrinsic reference**. Target-specific instruction and intrinsic availability.  
   https://llvm.org/docs/AMDGPUUsage.html

### ROCm and Windows support

6. AMD ROCm, **HIP SDK Windows system requirements**. The GPU table separates Runtime and HIP SDK support; current table lists RX 6700 XT as Runtime supported and HIP SDK unsupported.  
   https://rocm.docs.amd.com/projects/radeon-ryzen/en/latest/docs/shared/hipsdk/reference/system-requirements.html
7. AMD ROCm, **HIP performance guidelines**.  
   https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/performance_guidelines.html
8. AMD ROCm, **HIP programming model**.  
   https://rocm.docs.amd.com/projects/HIP/en/latest/understand/programming_model.html
9. AMD ROCm, **HIP asynchronous execution**.  
   https://rocm.docs.amd.com/projects/HIP/en/latest/how-to/hip_runtime_api/asynchronous.html

### Matrix and operator-library compatibility

10. AMD ROCm, **rocWMMA API reference and supported GPU architectures**. The supported RDNA list includes `gfx1100`, `gfx1101`, `gfx1102`, `gfx1200` and `gfx1201`; `gfx103x` is excluded.  
    https://rocm.docs.amd.com/projects/rocWMMA/en/docs-7.14.0/api-reference/api-reference-guide.html
11. ROCm/TheRock, **GFX target build configuration**. The `gfx1031` target excludes rocWMMA and several other libraries in the current target configuration.  
    https://github.com/ROCm/TheRock/blob/main/cmake/therock_amdgpu_targets.cmake

---

## 16. Source inventory and scope notes

This specification is based on:

- the uploaded `LLAMA_CUDA-FUNCTION-TABLE.md`, which defines the CUDA operator and kernel families to replace;
- the uploaded `RetroCompute_RDNA2_ROCm_7.16_Research_and_Test_Plan.md`, which defines the original RDNA2 and ROCm 7.16 research context;
- the separate `RDNA4_Native_HIP_Backend_Methods_RX9070XT.md`, used as a structural companion only. RDNA4-only methods have deliberately not been copied into this RDNA2 design;
- AMD RDNA2 ISA, AMD RX 6700 XT specifications, AMD ROCm GPU architecture tables, LLVM GFX10.3 instruction references and current ROCm Windows/rocWMMA support documentation.

### Scope distinction

This is an engineering specification, not a claim that the listed kernels have been implemented or benchmarked. The following are confirmed architectural planning points:

- target GPU is RX 6700 XT / `gfx1031`;
- RDNA2 supports packed FP16 and integer dot-product ALU instructions;
- RDNA2 is not a rocWMMA-supported architecture;
- RDNA2 has no RDNA3/RDNA4 WMMA or SWMMAC execution path;
- the current AMD Windows support table distinguishes RX 6700 XT runtime support from full HIP SDK support.

The following remain mandatory local validation items:

- exact ROCm/HIP 7.16 package compatibility on the user's Windows installation;
- `gfx1031` compiler and linker behaviour;
- availability and emitted ISA for each builtin, especially INT4 dot8;
- profiler/debugger and library compatibility;
- actual performance of each wave, tile, LDS and packed-dot strategy;
- numerical parity for each quantisation format and integrated model.

No performance improvement is assumed until measured on the RX 6700 XT.
