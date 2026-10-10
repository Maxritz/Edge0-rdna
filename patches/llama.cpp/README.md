# patches/llama.cpp — patch band ledger

Bands are replayed into `wt/win` by `windows/scripts/vendor-build.ps1` (git am --3way
after `git reset --hard <pin>`). Edits made directly in `wt/win` are wiped each build;
add a patch here instead.

## common/
- 0001..0006 — fork-private surfaces (LoRA wiring, router capture, CMake glob,
  prerouter, mem-budget, mul_mat_id expert-base resolver).
- **0007-edge0-private-ROCmFPX-low-bit-weight-types-types-100.patch** — ROCmFPX low-bit
  weight formats. Adds GGML types ROCMFP4/4_FAST/6/8/3/2 (enum 43..48), block structs +
  traits, CPU dequant/quant + generic vec_dot, CUDA dequant (to_fp32/to_fp16) and the
  MUL_MAT/supports_op + should_use_mmvq + should_fuse_mul_mat_vec_q routing so the types
  take the dequant-to-f16 + cuBLAS path. GGUF codes 100/101/102/103/104/107 are remapped
  in gguf.cpp. turbo3/turbo4 (105/106) are KV-cache formats and are NOT handled here.

Status: loads and generates correctly on gfx1031 (verified ornith-1.0-9b, gemma-E2B).
Decode ~4 tok/s (dequant-per-matmul), prefill ~14.6 tok/s. Native MMVQ/MMQ kernel is the
next stage for speed.
