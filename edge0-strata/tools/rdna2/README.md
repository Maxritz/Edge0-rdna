# RDNA2 latent-layout experiment tools

These tools implement the **lossless natural-order Q3_K repack** (`raw3k`) of the
Edge0-120B expert blobs and the checks used to validate it. The repack was
**measured and rejected** on 2026-11-18: it is bit-exact in isolation, but the
decode path uses a CPU expert pool that has no raw3k-aware Q3_K decoder (see
`docs/perf-log-2026-11-18.md`), it is 40-50 ms slower in gate/up, and it saves
zero bytes. The runtime arm was reverted; the canonical pack is untouched.

Kept as reproducible evidence, not as a shipping path:

- `raw3k_pack.py` - converter writing 256 plain 3-bit codes + the original scales/d.
- `raw3k_oracle.py` - ggml reference decoder used to prove the converter's permutation.
- `raw3k_verify.py` - written-pack check against 128 real blobs (209,715,200 values).
- `raw3k_gpu_check.py` - super-block check through the exact GPU decode expression.
- `raw3k_decode_test.cu` - HIP test with the engine's full define set.
- `raw3k_check_written.py`, `symbol_copy_test.cu` - diagnostic helpers.

Do not use these to regenerate a pack for serving. See the perf log for the
A/B tables and root cause.
