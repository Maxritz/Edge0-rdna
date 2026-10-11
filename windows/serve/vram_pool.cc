// serve/vram_pool.cc - edge0 VRAM hot-expert tier (device L0 over the RAM L1 / NVMe tiers).
//
// Why this is a backend hook and not the patch #7 resolver
// -------------------------------------------------------
// prefetch.cc already serves expert bytes to the CPU matmul through patch #7's
// ggml_edge0_mmid_base(src0, id, nb02). That hook is only called by
// ggml/src/ggml-cpu/* (ggml-cpu.c, iqp.cpp, repack.cpp) and its `const char*`
// return is dereferenced by CPU vec_dot kernels as a HOST pointer, so it can never
// deliver a device address. The HIP backend computes the expert base itself inside
// device code (`src0->data + expert*stride`) and does not call the resolver at all.
// Therefore the VRAM tier attaches to a separate HIP-backend hook (patch band #8,
// ggml_edge0_set_vram_hooks) and is consulted only when the expert matmul runs on
// the GPU.
//
// Attach point
// ------------
// ggml/src/ggml-cuda/ggml-cuda.cu : ggml_cuda_mul_mat_id already copies the routing
// ids to the host and iterates the used experts one at a time (the FATE path), setting
// `src0_slice.data = (char *) src0->data + i02*nb02` and calling ggml_cuda_mul_mat for
// that single expert. That per-expert assignment is the copy point a VRAM cache needs:
// when expert i02 is resident we substitute the pooled DEVICE base, so the expert
// matmul reads VRAM (D2D) instead of PCIe/NVMe. Patch #8 makes that path the one taken
// whenever a pooled tensor is present (ggml_edge0_vram_has), and records the residency
// so the CUDA-graph path stays off (the fallback synchronizes the stream).
//
// Residency model (docs/gpu-expert-cache-research.md, Fate arXiv:2502.12224, moe-l2):
//   L0 VRAM  = this pool (hot experts, D2D on hit)       E0_VRAM_MB
//   L1 RAM   = prefetch.cc private-page pool             E0_POOL_MB
//   L2 NVMe  = whole-block GGUF read                     always available
// An L0 hit is a device read with no PCIe/NVMe traffic. An L0 miss fills one slot,
// preferring the L1 host source (registered callback), then a whole-block GGUF read.
//
// Concurrency / correctness
// -------------------------
//   * Fills happen on the compute thread inside the resolver (a hit is a pure host
//     lookup). When no free slot exists, the resolver evicts exactly one clear victim
//     and reuses that slot; because ggml_backend_sched_compute_splits synchronizes a
//     backend before overwriting a tensor it fed (ggml-backend.cpp:1687) and then runs
//     that backend's split, no enqueued-but-unexecuted kernel can still read the victim.
//     If nothing is evictable the fill is refused and the backend falls back to the
//     original bytes (graceful degradation, like the CPU pool's budget refusal).
//   * Hot top-N pinning and idle pruning run only from edge0_vram_step, which
//     prefetch.cc calls at the step boundary after llama_synchronize, so no GPU kernel
//     is mid-read when a slot is recycled.
//   * One process-wide mutex serializes index/fill/evict accounting. Decode is a
//     single compute stream, so contention is negligible; misses (not hits) are the
//     only critical-path cost.
//
// The llama target is built as a unity build, so this file and prefetch.cc share one
// translation unit: every internal lives in the named namespace e0_vram_impl so its
// symbols cannot collide with prefetch.cc's anonymous-namespace state.
//
// Device API is resolved at runtime from the loaded HIP runtime DLL, so this unit
// links with no extra import library and is inert unless E0_VRAM_MB > 0. Default OFF.
#include "vram_pool.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

// ── Backend hook declaration (patch band #8, exported by ggml-hip.dll, C decoration) ──
// Global scope: a dllimport declaration may not sit inside a namespace.
typedef const void * (*ggml_edge0_vram_base_fn)(const struct ggml_tensor * src0, int64_t id, size_t nb02);
typedef bool         (*ggml_edge0_vram_has_fn)(const struct ggml_tensor * src0);
#ifdef _WIN32
extern "C" __declspec(dllimport) void ggml_edge0_set_vram_hooks(ggml_edge0_vram_base_fn base_fn, ggml_edge0_vram_has_fn has_fn);
#else
extern "C" void ggml_edge0_set_vram_hooks(ggml_edge0_vram_base_fn base_fn, ggml_edge0_vram_has_fn has_fn);
#endif

// The llama target is a unity build: this file and prefetch.cc share one translation
// unit, and prefetch.cc pulls in windows.h, whose min/max macros would rewrite
// std::max/std::min. Undef them and use the local helpers below.
#ifdef min
#  undef min
#endif
#ifdef max
#  undef max
#endif

namespace e0_vram_impl {

template <typename T> constexpr T e0_max(T a, T b) { return a < b ? b : a; }
template <typename T> constexpr T e0_min(T a, T b) { return a < b ? a : b; }

// ── Runtime HIP entry points (no import library; resolved from the loaded runtime) ──
typedef int    e0_hipError_t;
enum { E0_H2D = 1, E0_D2H = 2, E0_D2D = 3 };

struct HipApi {
    e0_hipError_t (*MemGetInfo)(size_t *, size_t *);
    e0_hipError_t (*Malloc)(void **, size_t);
    e0_hipError_t (*Free)(void *);
    e0_hipError_t (*Memcpy)(void *, const void *, size_t, int);
    const char *  (*GetErrorString)(e0_hipError_t);
    bool ok = false;
};
HipApi g_hip;

bool e0_hip_load() {
    static const char * dlls[] = { "amdhip64.dll", "amdhip64_7.dll", "amdhip64_6.dll", "amdhip64_5.dll", "libamdhip64.so" };
    HMODULE m = nullptr;
    for (const char * d : dlls) { m = GetModuleHandleA(d); if (m) break; }
    if (!m) for (const char * d : dlls) { m = LoadLibraryA(d); if (m) break; }
    if (!m) return false;
    g_hip.MemGetInfo     = (decltype(g_hip.MemGetInfo))     (void *) GetProcAddress(m, "hipMemGetInfo");
    g_hip.Malloc         = (decltype(g_hip.Malloc))         (void *) GetProcAddress(m, "hipMalloc");
    g_hip.Free           = (decltype(g_hip.Free))           (void *) GetProcAddress(m, "hipFree");
    g_hip.Memcpy         = (decltype(g_hip.Memcpy))         (void *) GetProcAddress(m, "hipMemcpy");
    g_hip.GetErrorString = (decltype(g_hip.GetErrorString)) (void *) GetProcAddress(m, "hipGetErrorString");
    g_hip.ok = g_hip.Malloc && g_hip.Free && g_hip.Memcpy && g_hip.MemGetInfo;
    return g_hip.ok;
}

constexpr size_t kGranule = 4096;

struct VramTensor {
    std::string name;
    uint64_t foff  = 0;      // absolute file offset of expert 0
    size_t   per   = 0;      // bytes per expert
    size_t   nexps = 0;
    int      layer = -1;     // consumer layer for hot grouping
};

enum : uint8_t { SLOT_FREE = 0, SLOT_RESIDENT = 1 };
enum : uint8_t { LIST_NONE = 0, LIST_T1, LIST_T2, LIST_PIN };
struct Slot {
    uint64_t key   = UINT64_MAX;   // (tensor<<20)|expert of the resident data
    uint8_t  state = SLOT_FREE;
    uint8_t  list  = LIST_NONE;
    double   freq  = 0.0;          // decayed routing score (freq mode + hot scoring)
    uint64_t ts    = 0;            // last step served (idle retention)
};

struct Pool {
    bool        on = false;
    int         mode_arc = 1;      // 1 = ARC (default), 0 = legacy freq eviction
    void *      dev = nullptr;
    size_t      cap = 0;
    size_t      slot_bytes = 0;
    int         nslots = 0;
    size_t      used_bytes = 0;
    std::vector<Slot> slots;

    std::vector<VramTensor> tens;
    std::unordered_map<std::string, int> name2t;   // tensor name -> index
    std::unordered_map<uint64_t, int> index;       // key -> slot (resident)

    e0_vram_host_src_fn host_src = nullptr;
    void *      host_user = nullptr;
    void *      gguf = nullptr;                    // prefetch.cc read handle (L2 fills)
    std::vector<uint8_t> stage;                    // GGUF read staging buffer

    // ARC state (keys; resident keys have a slot, B1/B2 are ghosts that keep the key)
    std::vector<uint64_t> t1, t2, b1, b2;
    int p = 0;                                     // adaptive target for T1
    int c = 0;                                     // non-pinned capacity

    // hot top-N
    int    hot_n = 0;
    double hot_decay = 0.75;
    int    hot_update = 4;
    std::vector<std::vector<double>> counts;       // tensor -> expert decayed count

    int    keep_steps = 16;
    int    max_swaps  = 256;
    int    trace_every = 32;
    std::string stats_path;

    long   calls = 0, hits = 0, miss = 0, bypass = 0, refused = 0, evicts = 0;
    long   fill_l1 = 0, fill_gguf = 0, pin_fills = 0;
    uint64_t step = 0;
    uint64_t evict_bytes = 0;

    std::mutex mtx;
};

Pool g;

uint64_t e0_key(int tid, int64_t id) { return ((uint64_t) (uint32_t) tid << 20) | (uint64_t) (id & 0xFFFFF); }

void e0_list_del(std::vector<uint64_t> & v, uint64_t k) {
    for (size_t i = 0; i < v.size(); ++i) if (v[i] == k) { v.erase(v.begin() + (long) i); return; }
}
void e0_free_slot(int s) { g.slots[s].state = SLOT_FREE; g.slots[s].list = LIST_NONE; g.slots[s].key = UINT64_MAX; }

// Copy one expert into a slot. Prefers the host L1 source, else a whole-block GGUF read.
bool e0_fill(int slot, int tid, int64_t id) {
    const VramTensor & t = g.tens[tid];
    const void * src = nullptr;
    if (g.host_src) { size_t len = 0; src = g.host_src(g.host_user, t.name.c_str(), id, &len); if (src && len < t.per) src = nullptr; }
    bool from_gguf = false;
    if (!src) {
        if (g.stage.size() < t.per) g.stage.resize(t.per);
        LARGE_INTEGER off; off.QuadPart = (LONGLONG) (t.foff + (uint64_t) id * t.per);
        if (!g.gguf || !SetFilePointerEx((HANDLE) g.gguf, off, nullptr, FILE_BEGIN)) return false;
        DWORD got = 0;
        if (!ReadFile((HANDLE) g.gguf, g.stage.data(), (DWORD) t.per, &got, nullptr) || got != (DWORD) t.per) return false;
        src = g.stage.data();
        from_gguf = true;
    }
    void * dst = (char *) g.dev + (size_t) slot * g.slot_bytes;
    e0_hipError_t e = g_hip.Memcpy(dst, src, t.per, E0_H2D);
    if (e != 0) {
        if (g_hip.GetErrorString) fprintf(stderr, "[vram-trace] H2D failed: %s\n", g_hip.GetErrorString(e));
        return false;
    }
    if (from_gguf) ++g.fill_gguf; else ++g.fill_l1;
    return true;
}

void e0_recount_cap() {
    int pinned = 0;
    for (const Slot & s : g.slots) if (s.list == LIST_PIN) ++pinned;
    g.c = g.nslots - pinned;
}

// ── slot/target selection (ARC) ──
// Returns a reclaimable slot, or -1 when everything is pinned. Ghosts hold no data and
// are released first; then an LRU T1/T2 victim is freed to its ghost list.
int e0_take_slot() {
    int free_s = -1;
    for (int s = 0; s < g.nslots; ++s) if (g.slots[s].state == SLOT_FREE) { free_s = s; break; }
    if (free_s >= 0) return free_s;
    if (!g.b1.empty()) { e0_list_del(g.b1, g.b1.front()); return e0_take_slot(); }
    if (!g.b2.empty()) { e0_list_del(g.b2, g.b2.front()); return e0_take_slot(); }
    if (!g.t1.empty() && (g.t1.size() > (size_t) g.p || g.t2.empty())) {
        uint64_t k = g.t1.front(); g.t1.erase(g.t1.begin());
        auto it = g.index.find(k);
        if (it != g.index.end()) { int s = it->second; g.index.erase(it); e0_list_del(g.b1, k); g.b1.push_back(k); ++g.evicts; g.evict_bytes += g.slot_bytes; e0_free_slot(s); return s; }
        return e0_take_slot();
    }
    if (!g.t2.empty()) {
        uint64_t k = g.t2.front(); g.t2.erase(g.t2.begin());
        auto it = g.index.find(k);
        if (it != g.index.end()) { int s = it->second; g.index.erase(it); e0_list_del(g.b2, k); g.b2.push_back(k); ++g.evicts; g.evict_bytes += g.slot_bytes; e0_free_slot(s); return s; }
        return e0_take_slot();
    }
    return -1;   // every slot pinned: refuse the fill
}

void e0_insert(int s, int tid, int64_t id, bool promote_t2) {
    const uint64_t k = e0_key(tid, id);
    if (g.mode_arc) {
        e0_list_del(g.b1, k); e0_list_del(g.b2, k);
        if (promote_t2) { g.t2.push_back(k); g.slots[s].list = LIST_T2; }
        else            { g.t1.push_back(k); g.slots[s].list = LIST_T1; }
    } else {
        g.slots[s].list = LIST_T2;
    }
    g.slots[s].key = k; g.slots[s].state = SLOT_RESIDENT; g.slots[s].ts = g.step; g.slots[s].freq = 1.0;
    g.index[k] = s;
}

// ── legacy frequency eviction (E0_VRAM_EVICT=freq; parity with prefetch.cc E0_POOL_FREQ) ──
void e0_freq_prune() {
    struct Cand { int slot; double f; };
    std::vector<Cand> cand;
    for (auto & kv : g.index) {
        int s = kv.second;
        if (g.slots[s].list == LIST_PIN) continue;
        if (g.step - g.slots[s].ts <= (uint64_t) g.keep_steps) continue;
        cand.push_back({ s, g.slots[s].freq });
    }
    size_t cap = e0_min<size_t>((size_t) g.max_swaps, cand.size());
    if (cap) {
        std::partial_sort(cand.begin(), cand.begin() + cap, cand.end(), [](const Cand & a, const Cand & b) { return a.f < b.f; });
        for (size_t i = 0; i < cap; ++i) {
            auto it = g.index.find(g.slots[cand[i].slot].key);
            if (it != g.index.end()) { e0_free_slot(it->second); g.index.erase(it); ++g.evicts; g.evict_bytes += g.slot_bytes; }
        }
    }
}

// ── hot top-N pinning (mirrors LayerOptions.hot_per_layer / hot_decay / hot_update_interval) ──
void e0_hot_update() {
    if (g.hot_n <= 0) return;
    for (size_t t = 0; t < g.tens.size(); ++t) {
        if (g.counts[t].empty()) continue;
        const VramTensor & vt = g.tens[t];
        int n = e0_min<int>((int) vt.nexps, g.hot_n);
        std::vector<int> ord((size_t) vt.nexps);
        for (size_t i = 0; i < ord.size(); ++i) ord[i] = (int) i;
        std::partial_sort(ord.begin(), ord.begin() + n, ord.end(),
                          [&](int a, int b) { return g.counts[t][a] > g.counts[t][b]; });
        for (auto & kv : g.index)   // return this tensor's previous pins to T2
            if ((int) (kv.first >> 20) == (int) t && g.slots[kv.second].list == LIST_PIN) {
                g.slots[kv.second].list = LIST_T2; g.t2.push_back(kv.first);
            }
        for (int i = 0; i < n; ++i) {
            const int64_t id = ord[i];
            if (g.counts[t][id] <= 0.0) break;
            const uint64_t k = e0_key((int) t, id);
            auto rit = g.index.find(k);
            if (rit != g.index.end()) { g.slots[rit->second].list = LIST_PIN; e0_list_del(g.t1, k); e0_list_del(g.t2, k); continue; }
            int s = e0_take_slot();
            if (s < 0) break;
            if (!e0_fill(s, (int) t, id)) { e0_free_slot(s); break; }
            g.slots[s].key = k; g.slots[s].state = SLOT_RESIDENT; g.slots[s].list = LIST_PIN; g.slots[s].ts = g.step;
            g.index[k] = s; ++g.pin_fills;
        }
    }
    e0_recount_cap();
}

// ── Resolver callbacks handed to the HIP backend (patch band #8) ──
const void * e0_vram_base_cb(const ggml_tensor * src0, int64_t id, size_t nb02) {
    if (!g.on || !src0 || !src0->name) return nullptr;
    std::lock_guard<std::mutex> lk(g.mtx);
    ++g.calls;
    auto nit = g.name2t.find(src0->name);
    if (nit == g.name2t.end()) { ++g.bypass; return nullptr; }
    const int tid = nit->second;
    if (nb02 != g.tens[tid].per) { ++g.bypass; return nullptr; }

    if (g.hot_n > 0) g.counts[tid][(size_t) id] += 1.0;

    const uint64_t k = e0_key(tid, id);
    auto rit = g.index.find(k);
    if (rit != g.index.end()) {                                  // resident hit: D2D
        int s = rit->second;
        g.slots[s].ts = g.step;
        if (g.hot_n > 0) g.slots[s].freq += 1.0;
        if (g.mode_arc) {
            if (g.slots[s].list == LIST_T1) { e0_list_del(g.t1, k); g.t2.push_back(k); g.slots[s].list = LIST_T2; }
            else if (g.slots[s].list == LIST_T2) { e0_list_del(g.t2, k); g.t2.push_back(k); }
        }
        ++g.hits;
        return (const char *) g.dev + (size_t) s * g.slot_bytes;
    }

    bool promote_t2 = false;
    if (g.mode_arc) {                                            // ghost hit adapts the T1 target
        auto in_b1 = std::find(g.b1.begin(), g.b1.end(), k) != g.b1.end();
        auto in_b2 = std::find(g.b2.begin(), g.b2.end(), k) != g.b2.end();
        if (in_b1) { int d = (int) e0_max<size_t>(1, g.b2.size() / e0_max<size_t>(1, g.b1.size())); g.p = e0_min(g.c, g.p + d); e0_list_del(g.b1, k); promote_t2 = true; }
        else if (in_b2) { int d = (int) e0_max<size_t>(1, g.b1.size() / e0_max<size_t>(1, g.b2.size())); g.p = e0_max(0, g.p - d); e0_list_del(g.b2, k); promote_t2 = true; }
    }
    int s = e0_take_slot();
    if (s < 0) { ++g.refused; return nullptr; }                  // not evictable: fall back
    if (!e0_fill(s, tid, id)) { e0_free_slot(s); ++g.refused; return nullptr; }
    e0_insert(s, tid, id, promote_t2);
    ++g.miss;
    return (const char *) g.dev + (size_t) s * g.slot_bytes;
}

bool e0_vram_has_cb(const ggml_tensor * src0) {
    if (!g.on || !src0 || !src0->name) return false;
    std::lock_guard<std::mutex> lk(g.mtx);
    return g.name2t.find(src0->name) != g.name2t.end();
}

bool init(int mb, const e0_vram_tensor * tensors, size_t n, void * gguf) {
    if (g.on) return true;
    if (mb <= 0 || !tensors || n == 0) return false;
    if (!e0_hip_load()) { fprintf(stderr, "[vram-trace] HIP runtime not found => tier off\n"); return false; }

    std::vector<VramTensor> ts;
    size_t maxper = 0;
    for (size_t i = 0; i < n; ++i) {
        if (!tensors[i].name || !tensors[i].per || !tensors[i].nexps) continue;
        VramTensor vt;
        vt.name = tensors[i].name; vt.foff = tensors[i].foff; vt.per = tensors[i].per;
        vt.nexps = tensors[i].nexps; vt.layer = tensors[i].layer;
        ts.push_back(vt);
        maxper = e0_max(maxper, vt.per);
    }
    if (ts.empty()) return false;

    g.mode_arc = 1;
    if (const char * ev = getenv("E0_VRAM_EVICT")) g.mode_arc = (strcmp(ev, "freq") != 0);
    if (const char * hn = getenv("E0_VRAM_HOT_N")) g.hot_n = atoi(hn);
    if (const char * hd = getenv("E0_VRAM_HOT_DECAY")) { double d = atof(hd); if (d > 0.0 && d < 1.0) g.hot_decay = d; }
    if (const char * hu = getenv("E0_VRAM_HOT_UPDATE")) { int v = atoi(hu); if (v > 0) g.hot_update = v; }
    if (const char * ks = getenv("E0_VRAM_KEEP_STEPS")) { int v = atoi(ks); if (v >= 0) g.keep_steps = v; }
    if (const char * ms = getenv("E0_VRAM_MAX_SWAPS")) { int v = atoi(ms); if (v > 0) g.max_swaps = v; }
    if (const char * tn = getenv("E0_VRAM_TRACE_N")) { int v = atoi(tn); if (v > 0) g.trace_every = v; }
    if (const char * sp = getenv("E0_VRAM_STATS")) g.stats_path = sp;

    g.slot_bytes = (maxper + kGranule - 1) / kGranule * kGranule;
    if (!g.slot_bytes) return false;
    g.cap = (size_t) mb << 20;
    g.nslots = (int) (g.cap / g.slot_bytes);
    if (g.nslots <= 0) { fprintf(stderr, "[vram-trace] E0_VRAM_MB=%d too small for %zu B slots => off\n", mb, g.slot_bytes); return false; }

    size_t free_v = 0, total_v = 0;
    if (g_hip.MemGetInfo(&free_v, &total_v) != 0) { fprintf(stderr, "[vram-trace] hipMemGetInfo failed => off\n"); return false; }
    size_t want = (size_t) g.nslots * g.slot_bytes;
    while (g.nslots > 0 && want > free_v) { --g.nslots; want = (size_t) g.nslots * g.slot_bytes; }
    if (g.nslots <= 0) { fprintf(stderr, "[vram-trace] no free VRAM for tier (free=%zuMB) => off\n", free_v >> 20); return false; }

    if (g_hip.Malloc(&g.dev, want) != 0 || !g.dev) { fprintf(stderr, "[vram-trace] hipMalloc(%zuMB) failed => off\n", want >> 20); return false; }

    g.tens = std::move(ts);
    for (size_t i = 0; i < g.tens.size(); ++i) g.name2t[g.tens[i].name] = (int) i;
    g.counts.assign(g.tens.size(), {});
    for (size_t i = 0; i < g.tens.size(); ++i) g.counts[i].assign(g.tens[i].nexps, 0.0);
    g.slots.assign((size_t) g.nslots, Slot{});
    g.gguf = gguf;
    g.used_bytes = want;
    g.c = g.nslots;
    g.on = true;

    ggml_edge0_set_vram_hooks(e0_vram_base_cb, e0_vram_has_cb);   // attach to patch surface #8

    fprintf(stderr, "[vram-trace] INIT: slots=%d slotB=%zu poolMB=%zu tensors=%zu evict=%s hotN=%d hotDecay=%.3f hotUpd=%d keep=%d\n",
            g.nslots, g.slot_bytes, want >> 20, g.tens.size(), g.mode_arc ? "arc" : "freq",
            g.hot_n, g.hot_decay, g.hot_update, g.keep_steps);
    fflush(stderr);
    return true;
}

void step() {
    if (!g.on) return;
    std::lock_guard<std::mutex> lk(g.mtx);
    ++g.step;
    if (g.hot_n > 0 && (g.step % (uint64_t) g.hot_update) == 0) {
        for (auto & cv : g.counts) for (auto & c : cv) c *= g.hot_decay;   // decayed counts (LayerOptions.hot_decay)
        e0_hot_update();
    }
    if (!g.mode_arc) e0_freq_prune();
    else {
        // Idle ARC entries are released in bounded batches so the next accesses can
        // repopulate; the scan runs only here, never on the compute path.
        for (int swaps = 0; swaps < g.max_swaps; ++swaps) {
            int victim = -1; uint64_t oldest = UINT64_MAX;
            for (auto & kv : g.index) {
                int s = kv.second;
                if (g.slots[s].list == LIST_PIN) continue;
                if (g.step - g.slots[s].ts <= (uint64_t) g.keep_steps) continue;
                if (g.slots[s].ts < oldest) { oldest = g.slots[s].ts; victim = s; }
            }
            if (victim < 0) break;
            uint64_t k = g.slots[victim].key;
            bool from_t2 = g.slots[victim].list == LIST_T2;
            auto it = g.index.find(k);
            if (it == g.index.end()) break;
            g.index.erase(it);
            e0_list_del(g.t1, k); e0_list_del(g.t2, k);
            (from_t2 ? g.b2 : g.b1).push_back(k);
            e0_free_slot(victim);
            ++g.evicts; g.evict_bytes += g.slot_bytes;
        }
    }
    if (g.trace_every && (g.step % (uint64_t) g.trace_every) == 0)
        fprintf(stderr, "[vram-trace] step=%llu hit=%ld miss=%ld refused=%ld bypass=%ld evict=%ld | fills L1=%ld GGUF=%ld pin=%ld\n",
                (unsigned long long) g.step, g.hits, g.miss, g.refused, g.bypass, g.evicts,
                g.fill_l1, g.fill_gguf, g.pin_fills);
}

void trace_final() {
    if (!g.on) return;
    std::lock_guard<std::mutex> lk(g.mtx);
    long tot = g.hits + g.miss;
    fprintf(stderr, "[vram-trace] FINAL hit=%ld/%ld=%.1f%% miss=%ld refused=%ld bypass=%ld evict=%ld evictMB=%llu fills L1=%ld GGUF=%ld slots=%d slotB=%zu poolMB=%zu\n",
            g.hits, tot, tot ? 100.0 * g.hits / tot : 0.0, g.miss, g.refused, g.bypass, g.evicts,
            (unsigned long long) (g.evict_bytes >> 20), g.fill_l1, g.fill_gguf, g.nslots, g.slot_bytes, g.used_bytes >> 20);
    if (!g.stats_path.empty()) {
        FILE * f = fopen(g.stats_path.c_str(), "w");
        if (f) {
            for (size_t t = 0; t < g.tens.size(); ++t) {
                fprintf(f, "%s", g.tens[t].name.c_str());
                for (size_t i = 0; i < g.tens[t].nexps; ++i) fprintf(f, " %.0f", g.counts.empty() ? 0.0 : g.counts[t][i]);
                fprintf(f, "\n");
            }
            fclose(f);
            fprintf(stderr, "[vram-trace] stats written: %s\n", g.stats_path.c_str());
        }
    }
    fflush(stderr);
}

}  // namespace e0_vram_impl

// ── Public entry points (global scope; C++ linkage matches the header) ──────────────
E0_VRAM_API bool edge0_vram_active(void) { return e0_vram_impl::g.on; }

E0_VRAM_API void edge0_vram_set_host_src(e0_vram_host_src_fn fn, void * user) {
    std::lock_guard<std::mutex> lk(e0_vram_impl::g.mtx);
    e0_vram_impl::g.host_src = fn; e0_vram_impl::g.host_user = user;
}

E0_VRAM_API bool edge0_vram_init(int mb, const e0_vram_tensor * tensors, size_t n,
                                 void * gguf, const char * gguf_path) {
    (void) gguf_path;
    std::lock_guard<std::mutex> lk(e0_vram_impl::g.mtx);
    return e0_vram_impl::init(mb, tensors, n, gguf);
}

E0_VRAM_API void edge0_vram_step(void) { e0_vram_impl::step(); }

E0_VRAM_API void edge0_vram_trace_final(void) { e0_vram_impl::trace_final(); }
