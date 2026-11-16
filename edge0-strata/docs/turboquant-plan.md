# TurboQuant + RotoQuant plan (Strata)

Status: rotation implemented in-tree (2026-11-15, env STRATA_KV_ROT); the
quantizer ladders below are follow-ups. Companion to
`docs/lmcache-integration-plan.md` and `docs/rdna2-isa-reference.md`.

**In-tree fact (12,000-line verify/layer/prefill/mtp files):** the engine
ALREADY rotates KV with FWHT-256 behind `STRATA_KV_ROT=1`:
- `fwht256_inplace_cuda` applied to K/V at store (`verify.cpp:2008`,
  `layer.cpp:1273`, `prefill.cpp`, `mtp.cpp`) and to queries
  (`<Hq,Hk> = <q,k>`, `verify.cpp:1964`, `layer.cpp:1315`)
- attention output de-rotated back (`H^-1 = H`, `verify.cpp:2176`,
  `layer.cpp:1344`)
- K8V4 rotates only V; Q4_0 always rotates; int8 only with
  `STRATA_KV_ROT=1` (`qsa_set_kv_int8_rotate`, `generate.cpp:3903`)
- the config (`serve/config-rdna2.json`: `--kv int8`) does **not** set the
  env flag — so rotation is available but unused on the current RDNA2 run.

`rotoquant.cu` (an independent sign-diagonal WHT implementation written
before that fact was discovered) has been removed — one rotation
implementation, one convention.

## 1. TurboQuant (Google, 2025) — the technique

Google's 2025 TurboQuant paper ("TurboQuant: Online Vector Quantization with
Near-optimal Distortion Rate and Compressive Uncertainty in High Dimensions")
quantizes vectors by **(a) rotating them with a random orthogonal transform,
(b) applying a scalar quantizer designed for the Gaussian distribution**
the rotation asymptotically induces, and **(c) estimating inner products
with a 1-bit Quantized-JL (QJL) sketch of the query**.

Why the rotation is the quality saver: un-rotated activations/weights have
amplitude outliers (esp. one straggling large channel), which crush scalar
quantizers. A random orthogonal matrix spreads energy across every
dimension (central-limit ⇒ coordinates ⇒ standard Gaussian), so a shallow
scalar quantizer reaches the theoretical rate-distortion optimum. The
rotation costs a matrix multiply; TurboQuant shows it's worth paying,
and is in fact necessary for stable low-bit operation.

**RotoQuant** = rotation-then-quantize in general. In our engine: apply the
same idea to the KV cache (and to weights already quantized: the fp8 dot2
conversion can rotate first, then quantize the rotated vector).

## 2. Why it matters for Strata (quality first)

- Current KV path is `--kv int8` per-block-scaled — int8 on un-rotated data
  is where the outliers live; the fp8 dot2 fix kept quality by keeping
  amplitudes faithful, not by flattening them.
- With rotation, int8 → (fp8 or int4) at the same output error: **memory
  halves again** (0.5 B/elem at int4) without the quality cliff that plain
  low-bit quant has. That is exactly "preserving quality" through rotation.
- LMCache's chunk store can then keep chunks at int4/fp8 (with a version
  tag on the rotation) and decode them losslessly back to f32 — the chunk
  store's storage cost shrinks ~2× vs int8 KV.

## 3. What remains to be built (the ladder on top of the rotation)

The rotation is done (+fused: `<Hq, Hk> = <q, k>`, de-rotated output). What
TurboQuant adds on top is the **quantizer ladder** that makes the rotated
data cheap to store and fast to touch:

1. **int4 / fp8 ladder under rotated int8**: today rotation only feeds the
   existing int8/q4 quantizers. Under rotation, the int4 nibble path
   (`V_DOT8_I32_I4`, ISA doc VOP3P 24) becomes quality-safe at 0.5 B/elem —
   the follow-up to implement (`V_DOT8_I32_I4` version of the q4 pool with
   the existing WHT-256 in front). Est. KV VRAM halves again.
2. **QJL-style score path** (1-bit random-sign sketch for attention scores):
   replaces the fp16 dot at scores time. Est. win: QSA attention stage
   (0.31 ms) plus the indexer top-k; the sketch is cheaper than the dot.
3. **Stencil decision needed**: `STRATA_KV_ROT=1` vs adaptive per-K/V
   selection. The current gate is global (qsa_set_kv_int8_rotate) — make it
   per-tensor if control is wanted, not per-request.

## 4. Numerics / quality protocol

- Parity: rotated-int8 vs bf16 reference — max rel error per output and
  argmax stability on the serve evaluation.
- TurboQuant's guarantee: distortion ≈ rate-distortion optimum for the
  Gaussianized data; empirically 1-bit QJL adds ≤ 2–3% attention-score
  noise at n=128.
- The config exposes the ladder (fp8/int8/int4) per tensor class so the
  memory/quality tradeoff is a flag, not a recompile.

## 5. Phases

| Phase | Deliverable | Effort |
|---|---|---|
| 0 | rotation (FWHT-256) | landed |
| 1 | env-gate + pairing (`STRATA_KV_ROT=1`) | landed |
| 2 | int4 nibble ladder under rotation | 2–4 d |
| 3 | QJL score path | 3–5 d |
| 4 | parity + KL/perf validation on the serve evaluation | 0.5 d |

## 6. Interaction with LMCache

LMCache chunk storage holds the rotated, quantized KV bytes with a rotation
version tag (WHT-256, no sign matrix — the in-tree rotation has no sign:
FWHT-256 is a pure Hadamard with 1/√256 normalization; deterministic so
must also stay deterministic); expanding a chunk means de-rotating and
dequantizing to f32 before pasting into the state. Rotation must be a
constant across the cache, so a chunk produced by an older build is
rejected by the version tag, never silently mis-read.
