# RetroCompute: which historical mechanisms this engine has already measured

A catalogue of 50-100 architectures is a research programme. A catalogue of *mechanisms* is six things, and this
engine has already run most of them on gfx1031 with numbers. This file exists so that no one re-implements, in a
new project, something that is already in the tree and already measured here.

Read it as: **status → evidence → verdict**. "Open" means nobody has measured it on this card.

---

## 1. The mechanisms that are already implemented, with the numbers

| mechanism | historical source | where it lives | measured on gfx1031 |
|---|---|---|---|
| **Binning / deferred visibility** | PowerVR Kyro TBDR | router → host grouping → experts bucketed by label, executed once per label | chunk change alone moved **105,756 → 62,739** expert fetches (`kg8.log` / `kg4.log`) |
| **Working-set tile budget** | Kyro / PowerVR "fit the tile on chip" | the auto-chunker's geometry rule | measured **non-monotonic**: chunk 512 → **278.0** tok/s, 1024 → 208.4, 2048 → 253.2. 512 is the optimum; the rule is right |
| **Explicit DMA + local staging** | Cell SPE local store + DMA | tile kernels stage weights to LDS, `latent_u_tile_k` (256 cols x 64 rows) | latent U **23.70 → 2.70 ms a layer (-8.8x)** |
| **Prefetch depth / double buffering** | Cell DMA, DB2 prefetch queues | the staging ring (`stage_slots`) | **ring 80 → 128: prefill 243.59/251.01 → 264.96/263.19 (+6.8 %), `wait copy` 3,306/3,162 → 2,410/2,522 ms (-24 %), tokens byte-identical** |
| **Buffer pool + pinning + predicted residency** | DB2 buffer pools | expert cache slots sized from free VRAM; `pre-filled 1910 of 1910 from the profile`; lend/refill | table change alone: **598 → 1,910 slots**, 105 → 955 at 512K |
| **Demand-paged tiering (VRAM / RAM / NVMe)** | OS/2 demand loading, DB2 | expert cache → page-locked arena → SSD with keep-alive | arena 26.66 GiB loaded at 1.73 GiB/s; VRAM 4.42 GiB free after it |
| **Vector chaining / fusion** | Cray chaining | gate/up + SwiGLU + down in ONE call; D→q8_1 fused; U→combine fused | why the ledger has 2,163 `gemm gate/up` entries rather than 6,489 |
| **Compressed weights, decode on the fly** | S3 Savage texture compression | Q3_K experts, Q4_0 latent tables, packed-fp16 GEMM | Q4 tables: prefill **+4.5 %**, decode **+4.5 %**; dot2 GEMM **3,074 → 1,546 ms** |
| **Packet classification / routing queues** | Intel IXP, IBM PowerNP | router + `latent_dense_u_tables` (prefix-sum compaction on device) | host grouping **365 ms** of a 14,617 ms prompt |
| **Parallel compositing** | PixelFlow | MoE combine of GPU rows + CPU-pool rows | combine **59 ms** |
| **Hardware multithreading to hide latency** | Denelcor HEP, Niagara | 6 waves of wave32 at 153 VGPR / 0 spills (measured) | A stage occupancy is the measured operating point |
| **Cache-resident working set** | memory interleaving / NUMA | resident expert streams hit the Infinity Cache | DRAM **372 GB/s** vs cache-resident **1,419 GB/s** (ratio 0.26) |

## 2. Mechanisms that do not transfer, and why

| proposed mechanism | source | why it does not apply here |
|---|---|---|
| **Page cleaners** | DB2 | A page cleaner exists to write DIRTY pages back. Model weights are immutable and read-only: there is nothing to write back, so the maintenance cost that DB2 pays is exactly zero here. What is left of the idea is prefetch, which is already the ring. |
| **Transaction logging / recovery** | DB2, mainframes | Nothing mutates. KV state is per-request and recomputed. There is no log to replay. |
| **On-chip framebuffer tiling** | Kyro | The tile fits on chip because a framebuffer tile is a few KB. Our working set is 26.7 GiB of experts against 64 KB of LDS per workgroup: six orders of magnitude. Tiling by *reuse* is what we do; tiling by *residency* is impossible. |
| **Skip arbitrary dense weight blocks** | Kyro visibility | Changes the mathematics. Valid only where sparsity is architectural - and here it is (MoE top-k), which is where it is already exploited. |
| **Out-of-order execution / branch prediction** | DEC Alpha | In-order SIMT hardware; no control over issue order beyond ILP, occupancy and ILP-friendly code shape. |
| **More independent dispatches** | BeOS, Transputer | The measured cost goes the WRONG way: an ungrapped launch is **28-31 us** against **0.805 us** inside a graph, i.e. ~8x. More services as separate launches would pay that 8x more often. Graph capture is the prerequisite, not more services. |
| **Vulkan backend** | "separate backend" | This engine is HIP. Every measured win that touches instructions (`v_dot2_f32_f16`, `dp4a`, the `v_dot8_i32_i4` candidate) depends on inline AMDGCN asm, which Vulkan's SPIR-V path does not expose. A Vulkan port would be a second engine with strictly less control - a portability project, not an optimisation. |
| **Wave64** | SIMD arrays | wave32 is the measured default and the occupancy readout (153 VGPR / 6 waves) is wave32. Wave64 halves scheduling granularity; nothing has measured it as better here. |

## 3. What is genuinely open (nobody has measured it on this card)

| item | why it is open | expected size |
|---|---|---|
| **Ship the ring at 128 adaptively** | measured +6.8 % today, but the default is still 80. Hardcoding 128 would break the Q8 path (`chunk + ring + 128 <= slots`: 762 > 721 collapses the chunk to 256, which was measured at -35 %). Needs the ring depth chosen by the SAME scan that chooses the chunk. | +6.8 % prefill, in hand, unshipped |
| **`v_dot8_i32_i4` in the Q4 path** | the ISA supports it (verified: it assembles for gfx1031); the shipped kernel uses `v_dot4c_i32_i8` after unpacking nibbles to s8. | ~25-30 % fewer instructions in `latent_u_tile_k`; whether that is time depends on issue- vs byte-bound |
| **SALU control plane** | measured: `latent_u_tile_k` spends ~300 of 2,509 instructions on 64-bit address math for wave-uniform pointers. | ~10 % of that kernel's issue slots |
| **Packed fp32 + the A stage's divisions** | measured: 312 scalar-FP instructions and **72 `v_div_scale_f32`** in a 4,562 ms kernel; 0 packed ops emitted. | unknown until the division chain is located; the biggest single block |
| **Eviction policy comparison (LRU / LFU / predicted)** | the profile prefill is a static predicted set, not an adaptive policy. | low today: slots (1,910) exceed the touched set (max 512 a layer), so there is no steady-state eviction to optimise |
| **Graph capture of the decode loop** | identified (28-31 us vs 0.805 us) and never attempted. | the largest host-side item; affects decode, which is latency-bound |

## 4. The rule this session keeps re-learning

Every mechanism above is worth exactly what it measures on this card, and the sign is not predictable: ring depth
was documented as "flat" until the VRAM to pay for it appeared; a bigger chunk was the obvious way to spend spare
cache slots and it lost 25 %; VOPD and packed math are free instruction slots that no one spends. **Measure in
alternating pairs, on identical artefacts, and keep the negative results.** For a mechanism that has no measurement
on gfx1031, the correct status is "unknown", not "promising".
