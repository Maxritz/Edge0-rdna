# Turbo3 / Rot4 KV - audit ledger (Goli-mat-de matrix)

Scope: the `--kv turbo3` / `--kv rot4` implementation on branch `edge0-rdna`, commit state of 2026-11-16 15:5x.
Files inspected in full or in the changed regions: `kv_turbo.hpp`, `kv_turbo.cu`, `kv_turbo_parity.cpp`,
`kv_stream.hpp`, `kv_stream.cu`, `qsa_decode_attn.hpp`, `qsa_decode_attn.cu` (loaders, both scalar-V branches,
`attn_kv_mode`, five dispatch ladders), `qsa_prompt_attn.cu` (dispatcher only), `qsa_pattn.cu` (guard only),
`layer.cpp` (KV state/pools/append/gather/zeroing), `layer.hpp`, `prefill.cpp` (staging, `pools_of`, prompt
append, MTP append), `conversation_snapshot.cpp`, `generate.cpp` (CLI), `CMakeLists.txt`.
Uninspected: `qsa_prompt_attn.cu` kernel bodies (1433+ lines) - not needed, the format is refused there;
`mtp.cpp` ring internals beyond the append call; Metal/CPU backends (not built here).

## Findings

| ID | Sev | Location | Requirement / invariant | Evidence | Status |
|---|---|---|---|---|---|
| A1 | P2 | `kv_stream.cu:175` (`copy_kernel`) with the new runs from `runs_of` | every run length must be a multiple of 16, because the copy moves `len/16` `uint4`s and **silently drops the remainder** | only the int8 *scale* run is asserted (`kv_stream.cu:218`). turbo3's run is `n_head_kv * page_size * 100`; page_size 4 gives 800 (fine), but any page_size not a multiple of 4 gives e.g. 200 -> 12 bytes lost. rot4's 112 B/head is always safe. | **LATENT** - unreachable at this model's page_size 4 |
| A2 | P2 | `kv_stream_parity.cpp:60-91,118,242`; `kv_window_parity.cpp:82-106,142,327` | a format the engine can select must be exercised by the mover parity tests | both enumerate `kKvF16/kKvInt8/kKvQ4/kKvHybrid` only, and their append dispatch ends in `else` = fp16, so adding turbo to a loop without a branch would test the *wrong* codec | **NOT RUN** - `--kv-resident` + turbo3/rot4 is implemented and unvalidated |
| A3 | P3 | `conversation_snapshot_test.cpp:67` | new formats round-trip through a snapshot | loop sets `kv_int8/kv_q4/kv_hybrid` only; production compares `image.format != l.format` (`conversation_snapshot.cpp:394`) which works for 4/5 | NOT RUN |
| A4 | P3 | `qsa_prompt_attn.cu:2417-2465` | prefill must not read a turbo3 pool as another format | dispatcher branches on `k_q4`/`k_q`/`k_pool`; turbo3 leaves all three null and reaches `if (!pools.k_pool || !pools.v_pool) return false;` (2465) -> `prefill.cpp:4380` runs `qsa_decode_attn_batch` (extended). Measured prefill 91.4/92.7/91.6/88.5 tok/s vs int8 93.0/92.5/87.5/91.2 - indistinguishable, so the fallback costs nothing at 4K | PASS by measurement, but the refusal is *implicit* (null-ness), not an explicit format check - fragile if a future format keeps `k_pool` set |
| A5 | P3 | `generate.cpp` CLI | a new storage format should have a stated accuracy gate | only the synthetic parity test and 6 matching greedy tokens on a random-token prompt | UNVALIDATED on a real model |
| A6 | P3 | decode tok/s as an A/B metric | a measurement must isolate the variable | turbo3 decode was 19.55 / 24.42 / 19.73 / 19.85 tok/s for the *same* config: spec acceptance on a random-token prompt is random, so per-token tok/s is not usable here; the fixed-width window (`PROF-T T=4`) is 95.99 (int8) vs 94.58 (turbo3) ms | risk recorded; claim narrowed to the fixed-width window |

## Verified (executed, not inferred)

| check | command | result |
|---|---|---|
| codec parity | `build-tests\kv_turbo_parity.exe` | **PASS** - turbo3 scale rel 1.19e-07, **0** gathered values differ from the pool bytes, rms 0.374; rot4 scale rel 0, **0** mismatches, rms 0.359; byte order turbo3 100 < rot4 112 < q4_0 144 < int8 264 < f16 512 |
| engine builds | `build_hip_official.bat` | BUILD_OK, `strata.exe` 15:26 |
| runs and serves | `bench4k.bat --kv turbo3` | prefill 91.4-92.7 tok/s, decode 19.6-24.4 tok/s, **0 expired waits** |
| like-for-like cost | `STRATA_VERIFY_PROFILE=1` | T=4 window 94.58 ms (turbo3) vs 95.99 ms (int8); attention 1.375 vs 1.077, append 0.279 vs 0.134, copy+combine 28.87 vs 30.32 (more cache slots) |
| context | `scan_ctx.bat turbo3 N` | 32768 / 65536 / 98304 / 131072 **RUN**; 196608 and 262144 refuse on VRAM |

## Corrections to earlier claims in this session

- I first reported the CPU expert pool at "~2.2 GB/s" (derived from the 32K decode rate). **Measured: 30.6 GB/s**
  (`pool multi ... 30.6 GB/s over the rows phases`), i.e. at DDR4-3200's achievable bandwidth. The CPU path is
  not the defect; that estimate was wrong and is retracted.
- A single turbo3 run reported 19.55 tok/s, and I nearly reported a regression. It is spec variance (A6).

## Recommended corrective actions, in order

1. **A2**: extend both mover parity tests to turbo3/rot4 and make an unknown format *fail* rather than fall to
   fp16. This is the gate for `--kv-resident`, which is the only path past ~200K context on this card.
2. **A1**: assert `bytes % 16 == 0` for every run in `runs_of`, not just int8's scale run.
3. **A4**: replace the implicit null-ness refusal with an explicit `qsa_kv_format` check in the prompt-attn
   dispatcher.
4. **A5**: real-model KL against f16 KV at a fixed prompt, as `bench/results/2026-09-27-kv-q4` did for q4_0.
5. **A3**: add the formats to the snapshot round-trip loop.
