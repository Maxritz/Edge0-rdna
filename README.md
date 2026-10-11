# edge0 · RDNA

**Streaming MoE inference for AMD Radeon (RDNA2 / RDNA4), built on a pinned llama.cpp with the Edge0 patch band.**

English | [中文](README_zh.md) | [日本語](README_ja.md) | [Español](README_es.md) | [Français](README_fr.md)

This is a fork of [Edge0-AI/edge0](https://github.com/Edge0-AI/edge0) focused on running
large sparse-MoE models on consumer **AMD Radeon** GPUs — RDNA2 (`gfx1031`, RX 6700 XT)
and RDNA4 (`gfx1201`, RX 9070 XT) — through the Windows HIP (ROCm) engine, with dense
compute on the GPU and MoE expert weights served from host memory.

## Engine

The engine is upstream llama.cpp at a pinned commit (`vendor.llama.pin`) plus a replayable
patch band in `patches/llama.cpp/`. `windows/scripts/vendor-build.ps1` clones the pin into
`vendor/llama.cpp`, replays the band (`git am --3way`) into an isolated build worktree at
`wt/win`, and builds `llama-server.exe` for HIP (ROCm) or Vulkan. The vendor tree is never
patched in place — direct edits in `wt/win` are wiped on the next build; add a patch to the
band instead.

Patch band (`patches/llama.cpp/common/`):

| patch | what it adds |
|---|---|
| 0001–0006 | fork-private surfaces: LoRA wiring in the main graph, router-input capture, CMake glob of `edge0/*.cc`, prerouter advisory hook, `edge0_apply_mem_budget` P-tier budget, `mul_mat_id` expert-base resolver |
| 0007 | **ROCmFPX low-bit weight types** — GGML types ROCMFP4 / 4_FAST / 6 / 8 / 3 / 2 (enum 43–48), GGUF codes 100–107, with CPU dequant/quant and CUDA dequant + `MUL_MAT` routing |
| 0008 | VRAM hot-expert base hook |
| 0009 | route pooled expert tensors to the per-expert loop |

ROCmFPX loads and generates correctly on `gfx1031` (verified with `ornith-1.0-9b` and
`gemma-E2B`); a native MMVQ/MMQ kernel is the next stage for decode speed. See
[`patches/llama.cpp/README.md`](patches/llama.cpp/README.md).

Backend dispatch: hipBLAS sends RDNA2 to **rocBLAS** and RDNA3/4 to **hipBLASLt**, so the
build stages both `rocblas\library` and `hipblaslt\library\<arch>` next to `llama-server.exe`.
On RDNA4 a missing hipBLASLt library dir drops the dense f16 GEMM to ~8 TF (vs ~96 TF).

## Benchmarks

**RDNA2 — RX 6700 XT (`gfx1031`), 12 GiB VRAM, 48 GiB RAM.** `llama-bench -fa auto -p 512
-n 128 -r 2 -t 8` with the per-row `--n-cpu-moe`; the 78 GiB row uses explicit
`-ngl 99 --n-cpu-moe 99` over mmap. Columns are **Prompt Process (pp512)** and
**Token Gen (tg128)** tok/s. All 11 models in `C:\x`.

| Model | arch | MoE | n_cpu_moe | pp512 | tg128 |
|---|---|---|--:|--:|--:|
| gemma-4-E2B-it Q4_0 ROCMFP4 | gemma4 | no | 6 | 1643.7 | 132.3 |
| ornith-1.0-9b ROCmFPX | qwen35 | no | 6 | 440.9 | 64.6 |
| Qwen3-30B-A3B-abliterated i1-Q2_K | qwen3moe | yes | 0 | 300.8 | 115.2 |
| Laguna-XS.2 IQ4_XS | laguna | yes | 6 | 849.3 | 39.9 |
| Qwen3.5-35B-A3B UD-Q4_K_XL | qwen35moe | yes | 2 | 929.5 | 28.2 |
| ornith-1.0-35B Q3_0 ROCmFPX | qwen35moe | yes | 2 | 421.1 | 24.3 |
| Underdog-Saluki-27B IQ2-mix-MTP | qwen35 | no | 0 | 179.7 | 22.1 |
| GLM-4.7-Flash-APEX-I | deepseek2 | yes | 6 | 701.2 | 24.4 |
| L3.2-8X3B-MOE Q8_0 | llama | yes | 6 | 606.0 | 11.8 |
| Qwen3.6-27B-AEON BF16-MTP Q5_K_M | qwen35 | no | 2 | 79.2 | 3.25 |
| qwen3.8-flash-next-reap-288 Q4_K_M | qwen4exp | yes | 99 | 50.9 | 10.6 |

**RDNA4 — RX 9070 XT (`gfx1201`), 16 GiB VRAM.** Full sweeps in
[`benchmarks.md`](benchmarks.md): Qwen3.5-35B-A3B decode peaks at `--n-cpu-moe 8`
(**44.1 tok/s**, pp2048 763); Laguna-XS-2.1 IQ3_XXS fits the card and hits **112 tok/s**
— the fastest config measured on that GPU.

Full per-type, round-by-round data: [`benchmarks.md`](benchmarks.md). Investigation log:
[`.opencode/sherlock-analysis.md`](.opencode/sherlock-analysis.md).

## Build (Windows / HIP)

Prerequisites: Visual Studio 2022 (C++, MSVC), CMake ≥ 3.21, AMD HIP SDK (ROCm) ≥ 6.1,
Ninja, Vulkan SDK (fallback backend), Rust + Node.js (app shell), Python 3.10+ (converter
and benches).

```powershell
git clone https://github.com/Maxritz/Edge0-rdna
cd Edge0-rdna/windows
pwsh -File scripts/vendor-build.ps1                      # HIP (ROCm) for RDNA2 + RDNA4
pwsh -File scripts/vendor-build.ps1 -Backend vulkan      # Vulkan fallback (any GPU vendor)
pwsh -File scripts/vendor-build.ps1 -GpuTargets "gfx1030;gfx1031;gfx1200;gfx1201"
```

Output: `wt\win\build-hip\bin\llama-server.exe` (HIP) or
`wt\win\build-vk\bin\Release\llama-server.exe` (Vulkan). See
[`windows/README.md`](windows/README.md) for the full build guide, backend notes, and the
Tauri app.

## Repository layout

| Path | Contents |
|---|---|
| `windows/` | Windows app + engine (C++ + HIP/ROCm and Vulkan); `scripts/vendor-build.ps1` |
| `patches/llama.cpp/` | the llama.cpp patch band, replayed into `wt/win` at build time |
| `vendor.llama.pin` | pinned upstream llama.cpp commit |
| `benchmarks.md` | measured RDNA2 / RDNA4 numbers |
| `docs/` | architecture, MoE, SSD streaming, prerouter, RDNA ISA notes |
| `.opencode/sherlock-analysis.md` | investigation log (RUN-001…) |
| `python/` · `macos/` · `ios/` · `android/` | other platform engines, retained from upstream edge0 |

## Upstream

The base framework and the other platform engines come from
[Edge0-AI/edge0](https://github.com/Edge0-AI/edge0). This fork concentrates on the AMD RDNA
bring-up and the llama.cpp patch band.

## License

Apache-2.0, including vendored third-party code (see [NOTICE](NOTICE)).
