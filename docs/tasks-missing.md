# Task Ledger — MISSING / PARTIAL vs the tiered-cache goal

Source: `/completeness audit` (2026-10-11). Goal: three-tier expert residency
(VRAM hot / RAM warm / NVMe cold) + intelligent prefetch. Windows/HIP engine
(`windows/serve/`) has RAM+NVMe; the hot tier and predictor are missing.

## MISSING (Windows/HIP engine)
- [x] **T1 — VRAM hot-expert tier.** Port the MLX `hot`/`staged` design
      (`python/src/edge0/streaming/`, `docs/streaming.md`) into `windows/serve/prefetch.cc`:
      a VRAM (device) expert pool with residency bitmap, D2D/H2D fill, served via the
      patch-#7 `ggml_edge0_mmid_base` resolver. Owner: agent-pool.
- [x] **T2 — Hot top-N residency + adaptive eviction.** Decayed-count top-N per layer
      (`hot_per_layer`, `hot_window`), ARC (recency+frequency) instead of plain
      recency/freq. Owner: agent-pool.
- [ ] **T3 — Intelligent cross-layer-gate prefetch.** Predict layer i+1 experts from
      layer i's gate input (FATE arXiv:2502.12224: 78.8% base, 97.15% at 75th-pct
      confidence). CPU-side, parallel with GPU. New module. Owner: agent-predict.
- [ ] **T4 — Low-bit KV turbo3/4 wired into attention** (currently types-only). Owner:
      agent-kv.

## PARTIAL
- [ ] **T5 — KV 128k sweep** (q4_0/iq4_nl rows, push n_cpu_moe lower). Measurement only.
- [x] ROCmFPX weight formats — load + run (9B 64, 35B ~30 tok/s).
- [x] planner / autotune / RAM-aware pool / load-mode.

## NEGATIVE — re-evaluate
- [ ] **T6 — MTP / spec-decode.** Measured net-negative on CPU-expert MoE (RUN-008),
      BUT spec-decode is a first-class feature (model-native MTP head, or a separate
      drafter). Re-evaluate with `/debate-3`: is the negative real, or a config artifact?

## Verification rule
Each coding task is DONE only when it builds on the HIP engine AND a measured
number on the RDNA2 box shows the intended effect (or an honest negative).
