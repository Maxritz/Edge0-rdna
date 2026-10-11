// serve/vram_pool.h - edge0 VRAM hot-expert tier (device-resident L0 above the RAM L1).
//
// The RAM L1 pool (prefetch.cc, E0_POOL_MB) serves expert bytes to the *CPU* matmul
// path through patch #7's ggml_edge0_mmid_base(src0, id, nb02) resolver. That resolver
// is only ever called by ggml/src/ggml-cpu/* and its return is dereferenced as a HOST
// pointer, so it can never hand back a device address. The HIP backend computes the
// expert base itself, as `src0->data + expert*nb02` inside device code
// (ggml/src/ggml-cuda/mmvq.cu mul_mat_vec_q_moe), so the VRAM tier attaches to a
// separate hook on that backend (patch band #8, ggml_edge0_set_vram_hooks) and is only
// consulted when the expert matmul actually runs on the GPU.
//
// Residency model (the Fate / moe-l2 three-tier picture, docs/gpu-expert-cache-research.md):
//   L0 VRAM  = this pool (hot experts, D2D on hit)      E0_VRAM_MB
//   L1 RAM   = prefetch.cc private-page pool            E0_POOL_MB
//   L2 NVMe  = whole-block GGUF read                    (always available)
// An L0 hit is a device read with no PCIe/NVMe traffic. An L0 miss fills one slot from
// the L1 host source when one is registered, else from the GGUF (H2D).
//
// Data model
//   VramTensor : one *_exps.weight tensor the pool may cache
//     name        GGUF tensor name (stable key; the ggml_tensor* changes every forward)
//     foff        absolute file offset of expert 0 (g.data_base + tt.off)
//     per         bytes per expert (on-disk size == kernel stride nb02)
//     nexps       tt.dim[2]
//     layer       consumer layer index for hot top-N grouping (-1 = ungrouped)
//   Slot (device): slot_bytes = align_up(max per, 4096); nslots = cap / slot_bytes
//     key         (tensor_id<<20 | expert) of the resident expert, or unused
//     list        ARC membership: T1 recent, T2 frequent, PIN hot; B1/B2 are ghosts
//     freq/ts     decayed routing score + last-step stamp (mirrors PoolTT)
//   Index: key -> slot. Eviction selected by E0_VRAM_EVICT (arc default, freq = the
//   legacy E0_POOL_FREQ recency behaviour, kept so the two stay A/B testable).
//
// All device calls go through function pointers resolved at runtime from the loaded HIP
// runtime, so this translation unit links with no extra import library and is inert
// (edge0_vram_init returns false) unless E0_VRAM_MB > 0.
//
// Gate: E0_VRAM_MB (default 0 = off, existing behaviour untouched).
#pragma once

#include <cstddef>
#include <cstdint>

struct ggml_tensor;

#ifdef _WIN32
#  define E0_VRAM_API __declspec(dllexport)
#else
#  define E0_VRAM_API __attribute__((visibility("default")))
#endif

// One poolable expert tensor (a *_exps.weight whose dim[2] == the expert count).
struct e0_vram_tensor {
    const char * name;    // GGUF tensor name
    uint64_t     foff;    // absolute file offset of expert 0
    size_t       per;     // bytes per expert
    size_t       nexps;   // expert count
    int          layer;   // owning/consumer layer for hot grouping (-1 = none)
};

// Optional L1->L0 promotion source: host bytes for (tensor, expert), or null. Supplies
// the private-page RAM pool once it holds the expert, so a VRAM fill can skip the file.
typedef const void * (*e0_vram_host_src_fn)(void * user, const char * tensor_name, int64_t id, size_t * len);

// Reserve a bounded device buffer and register the tensor inventory. gguf is the read
// handle prefetch.cc already owns (whole-block H2D fills); it must stay valid for the
// session. Returns false when the tier is disabled or unusable (caller keeps the host path).
E0_VRAM_API bool edge0_vram_init(int mb, const e0_vram_tensor * tensors, size_t n,
                                 void * gguf, const char * gguf_path);

// Register the optional L1 host source (call before or after init; null clears it).
E0_VRAM_API void edge0_vram_set_host_src(e0_vram_host_src_fn fn, void * user);

// Hot top-N refresh + ARC/freq idle pruning. Called from edge0_pref_on_step after
// llama_synchronize, so no GPU kernel is mid-read and slots may be recycled safely.
E0_VRAM_API void edge0_vram_step(void);

// Final accounting dump (called from ~Pref).
E0_VRAM_API void edge0_vram_trace_final(void);

// True once edge0_vram_init has installed the tier.
E0_VRAM_API bool edge0_vram_active(void);
