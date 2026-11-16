# LMCache Integration Plan (Strata / RDNA2)

Status: planning (2026-11-15). Nothing here is implemented yet.

## 1. What LMCache is

AMD's KV-cache management layer for LLM serving. It turns KV cache from
ephemeral state into reusable, persistent cache: chunks are keyed by content,
stored persistently (CPU RAM, local disk, or remote storage: Redis, S3…),
reused across requests and across engine instances. It is vendor-neutral and
extracts the KV layer from the engine. Value delivered: TTFT reduction
(3×–10× in reported multi-turn/RAG workloads) by prefix + **non-prefix**
reuse (matching chunks at any position in the prompt).

Official engine integrations today: **vLLM, SGLang**; TRT-LLM "coming soon"
per docs (docs.lmcache.ai/developer_guide/integration.html, retrieved
2026-11-15). It ships two connector patterns for those engines:
- **In-process** (`LMCacheConnectorV1`): the LMCacheEngine runs inside the
  engine process (Python-only; not a wire boundary).
- **MP/multiprocess** (`LMCacheMPConnector`): a standalone `lmcache server`
  owns KV storage; engines connect over **ZMQ (DEALER↔ROUTER)** with a
  **msgpack** control plane and an **IPC-based data plane** (CUDA-IPC; on
  ROCm a VMM/hipIPC-style import — being active work per
  LMCache#6626 / vllm#67577).

Chunking: LMCache keys chunks with its **own blake3 rolling hash,
chunk_size = 256 tokens**, independent of vLLM's sha256 block hashes. Control
messages ride msgpack; the one-time cache registration carries a pickle blob.
(The vllm.cpp "lmcache-cpp-client-connector" spec documents this exact
interaction for a C++ engine.)

ROCm: the published wheel targets **gfx942 (MI300X/MI325X) and gfx950
(MI350X)** fat binaries, ABI-matched to the upstream vllm/vllm-openai-rocm
image (torch 2.13, Py3.12). **gfx103x (RDNA2) is NOT covered.** A HIP source
build is possible (`BUILD_WITH_HIP=1 CXX=hipcc`), but the wheel's fat
binaries won't load. LMCache's GDS backend supports **hipFile**
(direct GPU↔storage I/O) — `LMCACHE_GDS_BACKEND=hipfile` or `gds_backend:
hipfile` in the config (rocm.docs.amd.com hipFile integrations list).

## 2. What this means for Strata

Strata is a custom C++/HIP engine (no torch/vLLM dependency). LMCache cannot
drop in as a wheel; there are two real paths:

### Path A — LMCache server + Strata C++ client (the real library)
Run `lmcache server` (built from source for hip/gfx103x) and implement a
Strata-side client speaking the MP protocol:
- ZMQ DEALER↔ROUTER, msgspec/msgpack control plane.
- Data plane via hipIPC import (RDNA2 + gfx103x: the kernel destination
  memory must be resident device memory; the server's MMU-mode port).
- Token hashing per LMCache rules (blake3 chunks, 256 tokens) — identical
  keys, so chunks deduplicate with other LMCache deployments on the same
  server.
- One-time cache registration contains the token-hash list + lane info
  (like vLLM's connector init).

Pros: identical cache semantics/hashes as the vLLM ecosystem; cross-engine
sharing for free. Cons: biggest surface (server + protocol + IPC data
plane), gfx103x source build is untested territory, and Strata's KV layout
must match LMCache's chunk export shape (`mp` renders chunks in a
pattern-compatible way: contiguous KV pages per token, transferable by IPC
fd).

### Path B — LMCache-style layer inside Strata (no server)
Implement the same semantics natively in C++/HIP inside the engine:
- Token-chunked KV store (256-token chunks, blake3 hashes),
- CPU-RAM + disk tiers modeled on LMCache's StorageManager (a
  `MemoryChunkStore` + `LocalDiskChunkStore` analog),
- prefix + non-prefix reuse,
- HIP-side dump/restore kernels for KV chunks (device↔pinned↔disk).

Pros: no Python, no server, no dependency on LMCache's IPC stack; Strata
already does conversation/prompt reuse ("55 reused tokens on the last
request") — this is the natural next step of that same machinery. Cons: no
cross-engine ecosystem, pagination semantics to design.

## 3. Recommendation

**Path B first, with the hash/chunk semantics borrowed from LMCache**, i.e.
a native Strata KV-reuse layer that stays LMCache-*compatible* at the hash
level (same token hashing, chunk size 256) so a future Path A client can
point at the same data. Rationale:
- The official wheel does not cover gfx103x; Path A requires a from-source
  server build + implementing a C++ MP client against an undocumented IPC
  surface, with no RDNA2 measurements possible until the build lands.
- Strata's engine already keeps conversation/prompt-prefix reuse; the
  marginal PR value of the server (sharing across vLLM hosts) is zero
  today.
- Path B exercises exactly the TTFT win on the RDNA2 target: multi-turn
  follow-ups stop re-prefilling the prompt; the prefill chunking work the
  ISA doc (§7 of the prefill-GEMM work) also becomes instantiable per-chunk.

## 4. Strata sketch (Path B)

Core shape, mirroring LMCache semantics:
1. **Chunker**: token window → chunk ids = `blake3(tokens[i:i+256])`
   (chained: chunk k hash builds on chunk k−1 hash, so identifiers are
   position-independent but history-bound — same as LMCache's).
2. **Store**: `unordered_map<chunk_hash → (locator, tier, size)>` with tiers
   (device → pinned RAM → disk/NVMe). Device chunk buffers are slices of a
   dedicated KV-reuse arena (allocated at init; sizes from cache plan).
3. **Reuse path** (prefill entry): walk the prompt in 256-token chunks;
   on hit, copy KV into the state's KV slots directly (no recompute) and
   mark the chunk's tokens as immediately "generated" for the state.
4. **Store path**: after each verified window/chunk commit, register the
   completed chunks (sync from device if the arena holds them, or copy-out
   through CUDAMemcpyAsync + the checksum path).
5. **Eviction**: LRU over chunk registry with tier-aware budgets; a hard
   budget driven by the serve knobs (`--kv-cache-mib` analog; the current
   `--conversation-cache-mib`/`lmcache_bytes` knob is the closest existing
   precedent).

Given ROCm: memory pool + hipMemcpyAsync for tier movement (no hipFile for
v1 — LMCache's GDS/hipFile integration is a later phase), MUCH cheaper than
the total re-prefill on a hit.

## 5. Risks / unknowns

- **Chunk layout compatibility for sharing** with vLLM exports: Strata's KV
  layout (GDN/QSA tensors) must match whatever LMCache's MP path expects;
  only matters for Path A.
- **Sequestered KV tensors** (GDN state is recurrent, not plain KV) →
  only the attention KV is directly chunkable; GDN's recurrent state needs
  either its own (small) chunk representation or must be excluded from
  store (re-prefill the recurrent part, reuse only attention KV) — design
  decision.
- **Correctness of approximate reuse** over prefix-only: any non-prefix hit
  must be audited against the attention's position dependence (position
  embeddings recomputed or adjusted on paste).
- **Interaction with the expert cache**: a reused-KV request skips the
  prefill, which changes cache residency pressure; the cache planner must
  price the reuse arena.

## 6. Phased roadmap (effort estimates, RDNA2 target)

| Phase | Deliverable | Effort |
|---|---|---|
| 0 | chunker + registry + blake3 hashing (Strata-side, LMCache-identical keys) | 1–2 d |
| 1 | **device↔pinned reuse**: prefix hits restore KV from host memory | 2–3 d |
| 2 | non-prefix reuse with position handling | 3–5 d |
| 3 | disk tier + restart-safe store (optional cross-session sharing) | 3–5 d |
| 4 | (later) Path A: MP client against an lmcache server built for gfx103x | 1–2 wk |

Phase 1 already converts Strata's repeated-prompt work (the "55 reused"
case generalizes from conversation-level reuse to any prefix of the prompt).
