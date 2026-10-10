# Research references and how they map to Edge0

Collected during the RDNA2/ROCmFPX work. Each entry: what it is, and the concrete
applicability verdict for this engine.

## DFlash: Block Diffusion for Flash Speculative Decoding
Chen, Liang, Liu (UC San Diego). arXiv 2602.06036v2. Code: github.com/z-lab/dflash
(MIT). Models: hf.co/collections/z-lab/dflash and /dflash-2. llama.cpp PR: #27342.

Method: a lightweight 5-layer (8 for Coder) **block-diffusion** drafter generates a
block of tokens in ONE parallel forward pass, conditioned on target-model hidden
features injected as extra **KV entries** into every draft layer. Draft shares the
target's frozen token embedding + LM head. Trained per-target against the frozen
target; loss weighted exp(-(k-1)/gamma) to favor early block positions; anchors +
masked blocks sampled randomly.

Numbers: up to 6.1x on Qwen3-8B; ~4.9x greedy avg; ~2.2-2.5x over EAGLE-3.
Qwen3-Coder-30B-A3B (MoE) tau 6.4-8.1; Qwen3.5-35B-A3B tau 5.4-7.9 speedup 1.7-2.4x;
GPT-OSS-120B tau 3.7-5.4 speedup 1.3-1.7x; Qwen3.5-27B tau 5.5-9.1 speedup 2.5-3.9x.
Without target conditioning: only 2.7-3.7x. Long-context needs light fine-tune.

Applicability to Edge0:
- Target-specific. We have checkpoints for our exact models (Qwen3.5-35B-A3B,
  GPT-OSS-120B, GLM 5.1; DFlash-2: Qwen3.8-27B, Muse-Glimmer-30B).
- Speedup is real when T_verify is cheap (GPU-resident target). Our bottleneck is
  CPU-expert MoE offload: verify cost scales with block size through the same CPU
  experts -> T_verify grows -> the L=(T_draft+T_verify)/tau win shrinks.
- Consistent with our MTP result (RUN-008): self-spec was net-negative on CPU-expert
  MoE (20.7 -> 17.4). DFlash is a better drafter but hits the same verify wall when
  experts are on CPU.
- Idea worth building (user): a load-time probe that loads the drafter, measures
  acceptance tau on a short run, and enables spec-decode only if it beats the verify
  cost. Cheap, measured; mirrors autotune.rs.
- Integration is a real chunk: load drafter, KV-inject target features, sparse block
  attention, verify loop (PR #27342).

## MoE-Infinity: Offloading-Efficient MoE Model Serving
Xue, Fu, Lu, Mai, Marina (Edinburgh). arXiv 2401.14361. Code: github.com/TorchMoE/MoE-Infinity.

Method: request-level **Expert Activation Matrix (EAM)** + **EAMC** collection trace
expert activation per request (skewed reuse, group activation). Activation-aware
multi-layer prefetching: match current iteration EAM against EAMC to predict next
experts; prioritize by layer proximity (1-(i-l)/L); prefetch across layers. Caching
holds initial-layer experts (later layers benefit more from prefetch). Reports 2-20x
over llama.cpp / DeepSpeed-Inference / Mixtral-Offloading / BrainStorm.

Key facts: expert activation ratio is low (Switch-128x0.2B 1.2% @bs1; Arctic 1.5%;
Mixtral-8x7B 25% @bs1). Skewed reuse (a few experts reused heavily within a request)
becomes uniform across many requests -> request-level tracing is required, model-level
counts wash out. EAMC cost < 1% of inference latency.

Applicability to Edge0:
- This is the algorithm our `serve/prefetch.cc` L1 pool approximates (frequency-based
  eviction). The missing pieces: request-level EAM, cross-layer group prefetch with
  layer-proximity priority, and caching among many layers rather than a single pool.
- Our honest negative (RUN-006/011): on a box whose RAM holds the weights, mmap wins
  because the file stays page-cached. The pool/EAM only pays when the model exceeds
  RAM and experts must stream from disk over PCIe. That is exactly MoE-Infinity's
  regime. So an EAM-style pool is a lever for the <=20 GB RAM profile, not this box.

## Doctor-Shotgun: llama.cpp MoE offload guide
hf.co/blog/Doctor-Shotgun/llamacpp-moe-offload-guide (Jan 2026).

- Keep attention + dense FFN + shared expert on GPU; route experts to CPU with
  `-ot "exps=CPU"` or `--cpu-moe`; use `--n-cpu-moe N` to keep N trailing MoE
  layers' experts on GPU. `--n-cpu-moe` counts from the highest-numbered layers.
- Prompt processing: GPU-offload prefill triggers when a batch exceeds a threshold
  (default 32 tokens; `GGML_OP_OFFLOAD_MIN_BATCH` overrides). Raise `-b`/`-ub`
  (e.g. 4096) so prompts batch large enough to be worth copying CPU weights to GPU.
- NUMA: bind to one socket when the model fits one node; `--numa distribute` helps
  when spread. `--no-direct-io` sometimes faster on load distribution.

Applicability to Edge0:
- Confirms our `--n-cpu-moe` approach. We do NOT currently tune `-b/-ub` or
  `GGML_OP_OFFLOAD_MIN_BATCH` -> concrete, cheap prefill levers to test.
- NUMA matters only on multi-socket; our boxes are single-socket.

## KV-cache papers (DeepSeek-V2/MLA, PiKV, TransMLA, KV survey)

Verified: DeepSeek-V2 arXiv:2405.04434, PiKV arXiv:2508.06526, TransMLA
arXiv:2502.07864, KV survey arXiv:2603.20397. All resolve on arXiv.

**DeepSeek-V2 / MLA** (arXiv:2405.04434). Multi-head Latent Attention compresses the
KV cache into a low-rank latent vector: -93.3% KV vs DeepSeek 67B, up to 5.76x
generation throughput. This is an *architecture* choice baked into DeepSeek-family
weights, not a runtime toggle. Our engine already handles `deepseek2` (GLM-4.7) via the
engine's MLA KV path.

**TransMLA** (arXiv:2502.07864). Converts a GQA pretrained model into MLA post-training
(93% KV compression, 10.6x speedup @8K on LLaMA-2-7B), then fine-tune ~6B tokens to
recover quality. Requires retraining per model; needs a target model we control.

**PiKV** (arXiv:2508.06526). Expert-sharded KV storage + MoE-routing-aware KV partition
across GPUs. Targets multi-GPU datacenter serving; single-GPU laptop is out of scope.

**KV Cache Optimization Strategies** survey (arXiv:2603.20397). Taxonomy of eviction,
compression, hybrid memory, attention variants, combinations; maps techniques to
deployment scenarios including edge devices. Good menu; no new mechanism.

Applicability to Edge0 (verdict: **use the survey as a menu; the portable lever is KV
quantization**):
- **KV quantization is the immediately actionable one.** Our per-op trace (RUN-018)
  shows decode is CPU-expert bound while VRAM holds KV. Asymmetric quantization
  (`-ctk q8_0 -ctv q4_0` / `-ctv q6_0`) shrinks the KV bytes/token, freeing VRAM to move
  more expert layers onto the GPU -> directly raises the decode-favorable `n_cpu_moe`.
  Already cheaply testable on our engine.
- **MLA/TransMLA**: only for DeepSeek-family weights (MLA native) or a GQA model we are
  willing to retrain. Not a runtime toggle.
- **PiKV / streaming-eviction (H2O/StreamingLLM)**: multi-GPU or endless-agent-loop
  concerns; not our current bottleneck.
- **Note on the pasted summary**: specific claims there ("139 tok/s", "200k context",
  "-ctv q6_0") are not in the abstracts I verified; treat them as unverified pointers,
  not measured facts. This repo's `benchmarks.md` records only our own measurements.
