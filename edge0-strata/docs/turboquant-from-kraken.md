> **Status (2026-11-15): the port was removed from the tree.** `include/strata/kernels/kv_turbo.hpp`,
> `src/kernels/cuda/kv_turbo.cu`, `src/kernels/kv_turbo_parity.cpp` and the CMake target `kv_turbo_parity` are
> deleted, so nothing unverified ships. This file stays as the specification for a later attempt.
>
> **Verified before removal:** the rotation matched the host reference bit for bit, rotate->unrotate returned the
> input to 7e-7, dot products were preserved to 1.5e-5, and the codebook quality was right: normalised MSE
> **0.03395** (turbo3, 52 B/row) and **0.03261** (rot4, 56 B/row) against Kraken's own 0.033.
>
> **Never fixed:** the scales. The device produced sigma ~25x too small (0.0386 where the host got 0.979), so every
> packed row disagreed with the host and the dot/decode checks failed. Two reduction rewrites went in and neither was
> re-verified: `__shfl_xor_sync` is **wrong on this part** (a wave is 64 lanes, so a 32-lane reduction smears one
> KV block's scale into the next), and the replacement serial thread-0 sum was never run.
>
> **Why it was removed rather than left in place:** a debug run carrying a device-side `printf` in that kernel
> faulted the GPU and Windows put the RX 6700 XT into an `Error` device state that only a reboot cleared. A
> half-working kernel that can wedge the card does not belong in the tree, wired to the engine or not. If this is
> attempted again: no device `printf`, no `__shfl_*` reductions, run the parity binary before anything links it into
> `strata_kernels`.

# TurboQuant, ported reference: Kraken-x (`third_party/Kraken-x`)

Reference implementation to lift into Strata: `include/krk/turbo_kv.hpp`
(3-bit Lloyd-Max KV), `docs/KQUANT-KERNELS.md` (Q2_K/Q3_K int8 dot kernels, with
measured numbers), `tools/turbo_kv_selftest.cpp` (round-trip test).

## The codec, exactly (so the port is mechanical)

**3-bit Lloyd-Max for N(0,1)** — 8 centroids and 7 boundaries:

```
centroids:  -2.15194566  -1.34390926  -0.75600525  -0.24509420
             0.24509420   0.75600525   1.34390926   2.15194566
boundaries: -1.74793313  -1.05003882  -0.50062380   0.0
             0.50062380   1.05003882   1.74793313
```

**Rotation** = sign diagonal (deterministic splitmix64 per (layer, kv head),
refreshed every 64 signs) then a **per-chunk normalized Hadamard** (chunks are
the largest power-of-two runs of `d`, each scaled by `1/sqrt(chunk_len)`).

> **Critical:** `rotate = S then H`, `unrotate = H then S`. They are **not**
> the same operation — H and S do not commute, and applying rotate twice is
> *not* the identity. Kraken's header records that assuming otherwise produced
> MSE 2.0 instead of 0.03. Strata's KV rotation is a bare FWHT-256 with no sign
> diagonal, so it *is* self-inverse; that is exactly why the `STRATA_KV_ROT`
> resume bug must be tracked down as a *pairing* error, not a maths error.

**Row layout** (`d` = one KV head's head_dim):
- turbo3: `ceil(d/8)*3` packed code bytes + one f32 scale, `sigma = ||x||/sqrt(d)`
  (per-vector standard deviation of a rotated coordinate).
- rot4: per 32-value block, 2 B f16 scale + 12 B (32 × 3-bit) = 14 B/block;
  `d=128 → 56 B/row`; `d=256 → 112 B`.

Scales: **block RMS**, not block max. A max-derived scale is inflated by
`E[max|N|]/sigma ≈ 2.35` and coarsens every level — the defect that made the
uniform 4-bit grid unusable.

**Scores** need no unrotation: `H` is orthogonal and symmetric, so
`<q,k> = <Hq,Hk>`; rotate `q` once per head and dot against the stored codes
(`turbo_dot` / `rot4_dot`). Values accumulate in the rotated basis and are
rotated back once per output.

## Measured lessons (their numbers, quoted)

1. Sub-8-bit KV **works with Lloyd-Max and fails with a uniform grid**: "this
   started as Q4_0's uniform 16-level grid … which measured *better* MSE than
   turbo3 (0.0089 vs 0.033) and still broke the model at the second token."
   MSE is not the metric that predicts text; the codebook's shape is.
2. 3-bit row = 100 B vs f16's 512 at d=256 → **5.1×**; d=128 → 52 vs 256 → 4.9×.
   rot4 at d=256 = 112 B → 4.57×.
3. Q2_K/Q3_K **int8 dot kernels** vs dequantise-then-multiply: **2.1× (Q2_K)**
   and **4.5× (Q3_K)** median of six paired reps on CPU; both are the same
   blocked-int8 recipe as Q4_K/Q6_K/Q8_0, and the win is largest for Q3_K
   because 3.4 bits/weight expanded to 32 leaves the SIMD unpack with almost
   nothing to do.
4. Two measured traps in the int8 kernels: reducing lanes **through memory**
   (store-to-load forwarding stall: 24.1 vs 36.1 tok/s) and **scalar
   unpacking** (Q3_K 23.8 vs 28.0 tok/s). Extract 16 values at a time with
   `_mm_srl_epi16`-class shifts.
5. Correctness metric: error **relative to Σ|term|**, not to the result —
   random-block fixtures cancel hard and dividing by the result measures the
   fixture's conditioning.

## What this means for Strata

**KV size is not our context limiter.** Our `--kv int8` cell is
`n_head_kv*head_dim*2 + n_head_kv*(head_dim/64)*2*2 = 528 B` at
`n_head_kv=2, head_dim=128`:
2048 cells = **1.0 MiB**, 32768 cells = **16.5 MiB**. Replacing int8 with rot4
(224 B/cell, 2.36× smaller) would save **7.0 MiB at 32k context** — real, but it
is not what bounds `--max-context` here; the expert cache and the window
buffers are.

So the ordering is:

1. **Find the actual context limiter first** (measured, not assumed):
   `--vram-reserve-mib 1024` with ~450–800 MiB free at request start means the
   binding constraint is the expert cache + prefill chunk buffers competing for
   the same VRAM, not KV. Grow `--max-context` in a temp config and read the
   startup report (`MiB of VRAM free with everything loaded`) — that, not the KV
   maths, gives the number.
2. **Q3_K int8 dot on the CPU expert pool** is the highest-value port from this
   reference if our CPU experts dequantise Q3_K today (our pack *is* q3k, and
   the CPU pool runs AVX-2): a 4.5× kernel win on the CPU share shows up
   directly in the 11 ms/window "CPU experts" and in prefill's gather.
3. **turbo3/rot4 KV** after that: 2.36× KV compression at equal-or-better
   quality than int8 (their text-correctness A/B), which buys headroom for
   `--max-context` and for the park/VRAM slot path.

## Port checklist (KV codec)

- [ ] `kTurboCentroid/kTurboBoundary` + `turbo_nearest` (device-side: 8 compares).
- [ ] `turbo_rotate`/`turbo_unrotate` as **one pair**, with the sign seed per
      (layer, kv head) and the per-chunk 1/sqrt normalization; share the sign
      table between the append and the read (Strata's FWHT-256 path can stay
      for the mean-preserving form, but a seeded sign diagonal must be added if
      we adopt `rot4`).
- [ ] Row bytes helper + the scale (RMS for rot4, per-vector sigma for turbo3).
- [ ] `pk_dot(rotated_q, row)` used by the QSA score kernel; `pk_decode_row`
      for the value accumulation and for any path that needs f32.
- [ ] Format tag in `conversation_snapshot.cpp` (there is already a `+16`
      rotated discriminator; a new enum value, not another bit, is the clean
      way), plus the same tag on the prompt checkpoint so a resumed session can
      never read codes it did not write.
- [ ] Parity test against the host reference on random rows **and** on real
      weights, with the rel-to-gross metric.
