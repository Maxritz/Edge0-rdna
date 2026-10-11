// serve/prefetch.cc — advisory expert-prefetch router + self-managed L1 pool + accounting.
// Depends only on llama public/near-public APIs (declared inline, linked from llama.lib)
// and parses GGUF/safetensors headers by hand; llama internal headers are not touched.
// The real routing gate is never modified: predictions only warm the page cache. A wrong
// prediction costs latency, never quality.
// Three-way overlap accounting (printed as [pref-trace]):
//   head   = pred@t-1 ∩ teacher@t        teacher = host-side recompute of gate·m_in (gate is F32 in our GGUF)
//   sticky = teacher@t-1 ∩ teacher@t     (the free win the OS page-cache LRU already provides)
//   random = uniform sample of K ∩ teacher@t   (popularity floor)
#include "prefetch.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#include <psapi.h>

#include <atomic>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

// Near-public staging API (src/llama-ext.h). NOTE: that header has **no extern "C"
// wrapper** — its symbols are C++-decorated, so you must include the real header rather
// than hand-write declarations. This unit is globbed into the llama target by the patch
// band (patches/llama.cpp/common #3), which puts src/ on the PRIVATE include path;
// llama.h is reachable via ../include PUBLIC.
#include "llama.h"
#include "llama-ext.h"

// Device-resident hot-expert tier (E0_VRAM_MB). The RAM pool above serves the CPU
// matmul through patch #7; this tier serves the GPU matmul through the HIP backend hook
// (patch #8). Its fill source callback below promotes experts already resident in the
// RAM L1 pool, so a VRAM fill can skip the file read when the RAM tier has the block.
#include "vram_pool.h"


namespace {

// Pool final telemetry (defined in the pool section below) — fwd-declared for ~Pref.
static void e0_pool_trace_final();

// ── Shapes (pinned for the 35B GGUF: E=256, H=2048, feat=2560, K=4, owners 6..38 → consumers 7..39) ──
constexpr int   EK = 256, HH = 2048, FE = 2560, KDIM = 4, NH = 33;
constexpr int   OWNER0 = 6;                       // first owner
std::vector<int> consumers;                        // 7..39

// ── GGUF/safetensors header parsing ──
std::string rd_str(FILE * f) {
    uint64_t n = 0;
    if (fread(&n, 8, 1, f) != 1) return {};
    std::string s((size_t) n, '\0');
    if (n && fread(s.data(), 1, (size_t) n, f) != n) return {};
    return s;
}
struct GGUFType { uint32_t t; };
size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
// GGUF KV value types: 0u8 1i8 2u16 3i16 4u32 5i32 6f32 7bool 8str 9arr 10u64 11i64 12f64
size_t scalar_size(uint32_t t) { static const size_t s[] = {1,1,2,2,4,4,4,1,0,0,8,8,8}; return t <= 12 ? s[t] : 0; }
void skip_val(FILE * f, uint32_t t) {
    if (t == 8) { rd_str(f); return; }
    if (t == 9) {
        uint32_t et; uint64_t n;
        if (fread(&et, 4, 1, f) != 1 || fread(&n, 8, 1, f) != 1) return;
        for (uint64_t i = 0; i < n; ++i) skip_val(f, et);
        return;
    }
    size_t s = scalar_size(t);
    if (s) fseek(f, (long) s, SEEK_CUR);
}
// Tensor dtype is the GGML_TYPE enum (ggml.h). Size via ggml's own block arithmetic so
// every type is covered (F32/F16/BF16, Q4_1, the K-quants, MXFP4, NVFP4, IQ*) instead of
// a hand table. Returns 0 for anything ggml cannot size -> the tensor is simply not pooled.
struct TT {
    std::string name;
    uint32_t type = 0;
    uint64_t dim[4] = {1, 1, 1, 1};
    uint64_t off = 0;
};
size_t tt_bytes(const TT & x) {
    const int64_t bs = ggml_blck_size((ggml_type) x.type);
    const size_t  es = ggml_type_size((ggml_type) x.type);
    if (bs <= 0 || es == 0) return 0;
    size_t n = 1;
    for (int i = 0; i < 4; ++i) n *= (size_t) x.dim[i];
    if (n % (size_t) bs) return 0;
    return es * (n / (size_t) bs);
}

// IEEE half → float (subnormals included)
float f16_to_f32(uint16_t h) {
    uint32_t sign = (uint32_t) (h & 0x8000) << 16;
    uint32_t e = (h >> 10) & 0x1F, m = h & 0x3FF;
    uint32_t bits;
    if (e == 0) {
        if (m == 0) bits = sign;
        else {                                        // subnormal: renormalize
            e = 127 - 15 + 1;
            while (!(m & 0x400)) { m <<= 1; e--; }
            m &= 0x3FF;
            bits = sign | (e << 23) | (m << 13);
        }
    } else if (e == 0x1F) {
        bits = sign | 0x7F800000u | (m << 13);
    } else {
        bits = sign | ((e + 127 - 15) << 23) | (m << 13);
    }
    float r;
    memcpy(&r, &bits, 4);
    return r;
}

// safetensors: { "<key>": {"dtype":"F16","shape":[..],"data_offsets":[a,b]} }
struct SEnt { long long a = -1, b = -1; std::vector<long long> shape; };
bool st_load(const char * path, std::map<std::string, SEnt> & out) {
    FILE * f = fopen(path, "rb");
    if (!f) return false;
    uint64_t hn = 0;
    if (fread(&hn, 8, 1, f) != 1) { fclose(f); return false; }
    std::string hdr((size_t) hn, '\0');
    bool ok = fread(hdr.data(), 1, (size_t) hn, f) == hn;
    fclose(f);
    if (!ok) return false;
    size_t i = 0;
    while ((i = hdr.find("\"layers.", i)) != std::string::npos) {
        size_t ke = hdr.find("\":", i + 1);
        size_t vo = hdr.find('{', ke);
        size_t vc = ke == std::string::npos || vo == std::string::npos ? std::string::npos : hdr.find('}', vo);
        if (vc == std::string::npos) break;
        std::string key = hdr.substr(i + 1, ke - i - 1);
        std::string o = hdr.substr(vo, vc - vo + 1);
        SEnt e;
        size_t doff = o.find("\"data_offsets\"");
        size_t lb = doff == std::string::npos ? std::string::npos : o.find('[', doff);
        size_t rb = lb == std::string::npos ? std::string::npos : o.find(']', lb);
        if (rb == std::string::npos) { i = vc; continue; }
        e.a = strtoll(o.c_str() + lb + 1, nullptr, 10);
        const char * p = o.c_str() + lb + 1;
        while (*p && *p != ',') p++;
        if (*p == ',') e.b = strtoll(p + 1, nullptr, 10);
        size_t sp = o.find("\"shape\""), slb = sp == std::string::npos ? std::string::npos : o.find('[', sp),
               srb = slb == std::string::npos ? std::string::npos : o.find(']', slb);
        if (srb != std::string::npos) {
            std::string nums = o.substr(slb + 1, srb - slb - 1);
            const char * q = nums.c_str();
            while (*q) {
                char * r; long long v = strtoll(q, &r, 10);
                if (r != q) e.shape.push_back(v);
                q = r; while (*q == ',' || *q == ' ') q++;
            }
        }
        out[key] = e;
        i = vc;
    }
    return !out.empty();
}

bool grab_f16(FILE * f, uint64_t data_base, const SEnt & e, std::vector<float> & dst) {
    size_t n = 1;
    for (auto d : e.shape) n *= (size_t) d;
    dst.resize(n);
    if (_fseeki64(f, (long long) (data_base + e.a), SEEK_SET) != 0) return false;
    std::vector<uint16_t> raw(n);
    if (fread(raw.data(), 2, n, f) != n) return false;
    for (size_t i = 0; i < n; ++i) dst[i] = f16_to_f32(raw[i]);
    return true;
}

// ── Runtime state ──
struct Head { std::vector<float> fc1, fc2, lin; };   // [512][2560] / [256][512] / [256][2560]
struct Hist { std::vector<float> cur, prev; };        // owner-layer two-step onehot (E)

struct Pref {
    bool active = false;
    std::map<int, Head> heads;                        // owner → head triple
    std::map<int, std::vector<float>> gate;           // consumer → [E][H]
    struct EXP { std::vector<TT> t; };               // consumer → up/gate/down exps metadata
    std::map<int, EXP> exps;
    std::map<int, Hist> hist;                         // layers 6..39 (39 keeps teacher onehot only)
    std::map<int, std::vector<int>> pred_prev;        // consumer → previous-step head top-K
    std::map<int, std::vector<int>> pred_ext_prev;    // consumer → previous-step FATE percentile set
    std::map<int, std::vector<int>> exec_prev;        // consumer → previous-step teacher (sticky)
    uint64_t data_base = 0;
    HANDLE fh = INVALID_HANDLE_VALUE, mh = NULL;
    void * view = nullptr;
    bool do_prefetch = false;
    int  trace_every = 32;
    int  stride = 1;          // full head/teacher recompute period (E0_PREF_STRIDE); skipped steps only re-issue prefetch
    bool async = false;       // E0_PREF_ASYNC=0 falls back to synchronous (tax-splitting A/B)
    // Predicted-range VirtualLock: pinned bytes are not trimmed while under budget;
    // lock failure once the cap is hit = natural ceiling. NOTE: VirtualLock normally
    // requires SeLockMemoryPrivilege, absent on consumer machines — treat this as best-effort.
    bool   do_pin = false;
    size_t pin_cap = 3ull << 30, pin_bytes = 0;
    std::map<int, std::vector<std::pair<size_t, size_t>>> pinned;   // consumer → currently pinned (file_off,len)
    // Self-managed L1 over the mmap view (E0_POOL): re-touch at step boundaries, in-table
    // age-based LRU; "evict" = drop from table (stop re-touching; reference-bit aging lets
    // the OS reclaim to standby — a free soft layer, no privileged APIs involved).
    int  pool = 0;                                    // 0=legacy advisory 1=sticky 2=sticky+observation
    struct Pin { size_t off, len; int age; uint8_t reg; };
    std::map<int, std::vector<Pin>> pins;             // consumer → pin table (age = steps since last hit)
    size_t reg_bytes[2] = {0, 0}, reg_cap[2] = {0, 0};   // 0=hot zone (teacher) 1=observation zone (head)
    long pool_hit = 0, pool_tot = 0, lock_fail = 0, evicts = 0, shrinks = 0;
    // Async: m_in row-snapshot mailbox ((NH+1)×H floats, contiguous); worker runs full steps.
    std::vector<float> snap;
    std::atomic<bool> busy{false}, quit{false};
    std::mutex wmtx;
    std::condition_variable cv;
    std::thread worker;
    long hits = 0, sticky = 0, rnd = 0, total = 0, steps = 0;
    long long pool_mb = 0;                        // private-page pool hot-zone MB (>0 enables; E0_POOL_MB)
    long long vram_mb = 0;                        // device hot-expert pool MB (>0 enables; E0_VRAM_MB)
    // ── FATE cross-layer gate prefetch (T3, arXiv:2502.12224) ──
    // Training-free: layer c's routing is predicted by feeding the previous layer's captured
    // input through c's own gate matrix. CPU-only, runs on the existing worker thread.
    bool  xlayer = false;          // E0_XLAYER=1 enables the FATE predictor (replaces the head)
    float x_pct = 0.75f;           // E0_XLAYER_PCT: confidence percentile cutoff (paper: 75th)
    int   x_ext_max = 64;          // E0_XLAYER_EXT_MAX: cap on the percentile set (25% of E = 64)
    int   x_pf_cap = 96;           // E0_XLAYER_PF_TOK_MAX: per-layer popularity cap during prefill
    int   x_pf_tok = 64;           // E0_XLAYER_PF_TOK: tokens aggregated per prefill ubatch
    bool  prefill = false;         // current worker snapshot is a multi-token (prefill) ubatch
    int   pf_tok = 0;              // tokens captured in the prefill snapshot
    std::vector<float> snap_pf;    // prefill snapshot: pf_tok*(NH+1)*HH floats (token-major)
    long  x_hits = 0, x_tot = 0;   // FATE top-K overlap vs teacher (accuracy proof)
    long  x_ext_hits = 0;          // FATE percentile-set overlap vs teacher (per-K)
    long  x_issued = 0, x_res = 0; // FATE experts issued / already resident (pool check)
    uint32_t lcg = 2463534242u;
    uint32_t next() { lcg ^= lcg << 13; lcg ^= lcg >> 17; lcg ^= lcg << 5; return lcg; }
    ~Pref() {
        quit = true; cv.notify_all();
        if (worker.joinable()) worker.join();
        if (!total) return;
        fprintf(stderr, "\n[pref-trace] head   overlap = %ld/%ld = %.1f%%\n", hits, total, 100.0 * hits / total);
        fprintf(stderr, "[pref-trace] sticky overlap = %ld/%ld = %.1f%%\n", sticky, total, 100.0 * sticky / total);
        fprintf(stderr, "[pref-trace] random overlap = %ld/%ld = %.1f%%\n", rnd, total, 100.0 * rnd / total);
        fprintf(stderr, "[pref-trace] head−sticky = %.1f pp | head−random = %.1f pp\n",
                100.0 * (hits - sticky) / total, 100.0 * (hits - rnd) / total);
        if (x_tot)
            fprintf(stderr, "[pref-trace] FATE cross-layer: top-K overlap = %ld/%ld = %.2f%% | ext-set = %ld/%ld = %.2f%% | issued=%ld resident=%ld\n",
                    x_hits, x_tot, 100.0 * x_hits / x_tot, x_ext_hits, x_tot, 100.0 * x_ext_hits / x_tot, x_issued, x_res);
        e0_pool_trace_final();
        edge0_vram_trace_final();
        if (pool)
            fprintf(stderr, "[pref-trace] POOL(%s) hit = %ld/%ld = %.1f%% | pins hot=%lluMB obs=%lluMB (cap %llu/%llu) evict=%ld lockfail=%ld shrink=%ld\n",
                    pool == 2 ? "sticky+obs" : "sticky", pool_hit, pool_tot, pool_tot ? 100.0 * pool_hit / pool_tot : 0.0,
                    (unsigned long long) (reg_bytes[0] >> 20), (unsigned long long) (reg_bytes[1] >> 20),
                    (unsigned long long) (reg_cap[0] >> 20), (unsigned long long) (reg_cap[1] >> 20),
                    evicts, lock_fail, shrinks);
        fflush(stderr);
    }
};
Pref g;

float gelu_erf(float x) { return 0.5f * x * (1.f + erff(x * 0.7071067811865475f)); }
float dotf(const float * __restrict a, const float * __restrict b, size_t n) {
    float s = 0.f;
    for (size_t i = 0; i < n; ++i) s += a[i] * b[i];   // /arch:AVX2 auto-FMA reduction; the bottleneck is weight streaming, not ALU
    return s;
}
void topk(const float * v, int n, int k, std::vector<int> & out) {
    out.clear();
    std::vector<char> used(n, 0);
    for (int j = 0; j < k; ++j) {
        int best = -1;
        for (int i = 0; i < n; ++i) if (!used[i] && (best < 0 || v[i] > v[best])) best = i;
        if (best < 0) break;
        used[best] = 1; out.push_back(best);
    }
}
// FATE (arXiv:2502.12224 §4.2): gate logits are compared positionally (Gate_in_{i+1} fed to
// layer i+1's gate), never normalized - softmax is monotonic so top-K is identical, and the
// percentile keeps its positional meaning on the raw logits. The "confidence score" cutoff is
// a percentile over the E logits, so it selects E*(1-pct) experts regardless of scale (never a
// fixed threshold, which is the paper's point when it uses a distribution-relative cutoff).
void topk_pct_cut(const float * v, int n, float pct, int kmax, std::vector<int> & out, float & thr) {
    std::vector<float> s(v, v + n);
    std::sort(s.begin(), s.end());
    int ord = (int) std::lround((double) pct * (n - 1));
    if (ord < 0) ord = 0; if (ord > n - 1) ord = n - 1;
    thr = s[ord];
    out.clear();
    for (int i = 0; i < n && (int) out.size() < kmax; ++i) if (v[i] >= thr) out.push_back(i);
}
int ovlp(const std::vector<int> & a, const std::vector<int> & b) {
    int n = 0;
    for (int x : a) for (int y : b) if (x == y) { ++n; break; }
    return n;
}

}  // namespace

// ══ Private-page expert pool (E0_POOL_MB): L1 with fill-once slots ══
// Design: decode reads expert bytes **directly from pool pages** via the expert-base
// resolver (patch surface #7), bypassing the view page-fault path entirely — a hit is a
// plain read with no kernel involvement. Slots fill once and never recycle (no eviction
// ⇒ no read-vs-overwrite race, correctness by construction); once the budget is
// exhausted, new experts pass through to the mmap view = graceful degradation to the
// un-pooled behavior. Three acquisition channels:
//   demand = synchronous whole-block ReadFile on the compute thread (first touch)
//   hot    = worker prefill from the teacher set
//   obs    = dedicated quota for head predictions
// Whole-block reads are also fragmentation-proof (1×1MB sequential vs a 256×4K fault
// storm — an order of magnitude apart on consumer SSDs).
static_assert(sizeof(LONG) == 4, "slot state atoms");
namespace {

struct PoolTT {                       // one 3D experts tensor (40 slots measured across all layers)
    uint8_t * arena = nullptr;        // MEM_RESERVE, MEM_COMMIT on demand
    volatile LONG * st = nullptr;     // slot state: 0=empty 1=filling 2=ready 3=not-poolable
    CRITICAL_SECTION * cvlk = nullptr;// per-slot pair: CS lock + CV (filler owns it, readers sleep until woken)
    CONDITION_VARIABLE * cv = nullptr;
    size_t per = 0;                   // bytes per expert (on-disk size)
    size_t stride = 0;                // page-aligned slot size (per rounded up) so commit/decommit of one slot never touches a neighbour
    size_t nexps = 0;                 // number of expert slots (== tt.dim[2])
    uint64_t foff = 0;                // file offset of expert 0
    uint64_t * lastuse = nullptr;     // per-slot step number when last served (0 = never)
    uint8_t  * sreg = nullptr;        // per-slot region the slot was committed in (0=hot 1=obs)
    double   * freq = nullptr;        // per-slot decayed routing frequency (hot-expert score)
    uint64_t * hits = nullptr;        // per-slot total routing count (for the stats dump)
};
static volatile LONG g_p_sc_ok = 0, g_p_sc_bad = 0;      // byte-level self-check (fill vs view memcmp sampling)
static bool g_pool_sc = false;                           // E0_POOL_SELFCHECK=1 enables
static void e0_pool_selfcheck(uint8_t * slot, uint64_t foff, size_t per) {
    if (!g_pool_sc || !g.view) return;
    const uint8_t * v = (const uint8_t *) g.view + foff;
    int bad = 0;
    for (size_t i = 0; i < per; i += 4096) if (memcmp(slot + i, v + i, 64) != 0) { bad = 1; break; }
    if (bad) InterlockedIncrement(&g_p_sc_bad); else InterlockedIncrement(&g_p_sc_ok);
}
// ⚠ The tensor *pointer* changes every forward (same tensor registered multiple times,
// measured) ⇒ key by tensor NAME (table built at init, read-only afterwards → lock-free).
std::map<std::string, PoolTT *> g_ptt_by_name;                  // GGUF name → PTT
std::map<int, std::vector<PoolTT *>> g_ptt_by_c;                // consumer → 3 PTTs (for prefill)
std::mutex g_pmtx;                                              // fill/accounting lock (fills are ms-scale, not a spin hotspot)
size_t g_pool_cap[2] = {0, 0};                                  // 0=hot 1=obs (E0_POOL_MB/E0_POOL_OBS_MB)
size_t g_pool_bytes[2] = {0, 0};
volatile LONG g_p_calls = 0;
bool g_pool_on = false;
volatile LONG g_p_hits = 0, g_p_miss_d = 0, g_p_bypass = 0, g_p_waits = 0, g_p_fills = 0;
// Step-boundary eviction: the pool keeps slots used within the last g_keep_steps decode
// steps and decommits (frees physical pages for) the rest, so resident RAM tracks the hot
// set instead of the whole working set. Eviction runs only from on_step (after
// llama_decode returns = no compute thread is mid-read), so there is no read/overwrite race.
volatile LONG g_p_step = 0;
volatile LONG g_p_evicts = 0;
size_t g_p_evict_bytes = 0;
int g_keep_steps = 16;
int g_max_swaps = 256;             // per-step eviction cap (churn control; ninfer: 2/layer/call)
const double g_freq_decay = 0.9576; // 2^(-1/16): half-life 16 decode steps, per ninfer
std::string g_stats_path;          // E0_POOL_STATS: dump per-expert routing counts on exit
bool g_pool_freq = false;          // E0_POOL_FREQ=1: frequency-based eviction (else recency)
volatile LONG g_pool_full = 0;   // set when a fill is refused for budget: the working set exceeds the pool

// Expert-base resolver hook from the patch band (exported by ggml.dll, C decoration).
// Decls mirror the patch header's extern "C" block (C language linkage on the fn-pointer type).
extern "C" {
typedef const char * (*ggml_edge0_mmid_base_fn)(const struct ggml_tensor * src0, int64_t id, size_t nb02);
__declspec(dllimport) void ggml_edge0_set_mmid_resolver(ggml_edge0_mmid_base_fn fn);
}

// Resolver callback (patch surface #7, runs on ggml compute threads): returns the base
// address of expert `id`'s data.
extern "C" {
static const char * e0_mmid_base_res(const ggml_tensor * s, int64_t id, size_t nb02) {
    LONG c = InterlockedIncrement(&g_p_calls);
    if (c <= 2) { fprintf(stderr, "[pref-trace] POOL2 call#%ld name=%s nb02=%zu\n", c, s->name ? s->name : "?", nb02); fflush(stderr); }
    const char * orig = (const char *) s->data + (size_t) id * nb02;
    auto it = g_ptt_by_name.find(s->name ? s->name : "");
    if (it == g_ptt_by_name.end() || !it->second->st || it->second->per != nb02) {
        LONG b = InterlockedIncrement(&g_p_bypass);
        if (b <= 6)
            fprintf(stderr, "[pref-trace] POOL2 miss#%ld name=%s nb02=%zu in_table=%d table=%zu\n",
                    b, s->name ? s->name : "?", nb02, (int) (it != g_ptt_by_name.end()), g_ptt_by_name.size());
        return orig;
    }
    PoolTT * pt = it->second;
    for (;;) {
        LONG st = pt->st[id];
        if (st == 1) {                                        // another thread is filling: sleep until woken (no spinning)
            InterlockedIncrement(&g_p_waits);
            CRITICAL_SECTION & lk = pt->cvlk[id];
            EnterCriticalSection(&lk);
            if (pt->st[id] == 1) SleepConditionVariableCS(&pt->cv[id], &lk, 2000);
            LeaveCriticalSection(&lk);
            continue;
        }
        if (st == 2) {
            pt->lastuse[id] = (uint64_t) g_p_step;             // recency for step-boundary eviction
            if (pt->freq) pt->freq[id] += 1.0;                 // decayed-frequency hot score (benign race)
            if (pt->hits) pt->hits[id] += 1;                   // total count for the stats dump
            LONG h = InterlockedIncrement(&g_p_hits);
            if (h <= 3 || (h & 0x3FFF) == 0)
                fprintf(stderr, "[pref-trace] POOL2 hit=%ld fillD=%ld bypass=%ld wait=%ld evict=%ld sc=%ld/%ld | hot=%lluMB obs=%lluMB\n",
                        h, g_p_miss_d, g_p_bypass, g_p_waits, g_p_evicts, (long) g_p_sc_ok, (long) g_p_sc_bad,
                        (unsigned long long) (g_pool_bytes[0] >> 20), (unsigned long long) (g_pool_bytes[1] >> 20));
            fflush(stderr);
            return (const char *) (pt->arena + (size_t) id * pt->stride);
        }
        if (st == 3) { InterlockedIncrement(&g_p_bypass); return orig; }
        {                                                                   // claim the fill: CAS + exclusive CV section
            CRITICAL_SECTION & lk = pt->cvlk[id];
            if (InterlockedCompareExchange(&pt->st[id], 1, 0) != 0) continue;
            EnterCriticalSection(&lk);
        }
        LONG fs = InterlockedIncrement(&g_p_fills);
        if (fs <= 3 || (fs & 0x3FF) == 0) {
            fprintf(stderr, "[pref-trace] POOL2 fill#%ld name=%s id=%lld per=%zu hits=%ld\n",
                    fs, s->name ? s->name : "?", (long long) id, pt->per, g_p_hits);
            fflush(stderr);
        }
        // Demand fill on this thread (hot budget).
        bool ok = false;
        {
            std::lock_guard<std::mutex> lk(g_pmtx);
            if (g_pool_bytes[0] + pt->stride <= g_pool_cap[0]) {
                uint8_t * dst = pt->arena + (size_t) id * pt->stride;
                if (VirtualAlloc(dst, pt->stride, MEM_COMMIT, PAGE_READWRITE)) {
                    uint64_t fo = pt->foff + (uint64_t) id * pt->per;
                    LARGE_INTEGER off; off.QuadPart = (LONGLONG) fo;
                    ok = SetFilePointerEx(g.fh, off, nullptr, FILE_BEGIN)
                         && ReadFile(g.fh, dst, (DWORD) pt->per, nullptr, nullptr);
                    if (ok) { g_pool_bytes[0] += pt->stride; e0_pool_selfcheck(dst, fo, pt->per); }
                }
            }
            else InterlockedExchange(&g_pool_full, 1);   // budget refused: working set exceeds the pool
            if (ok) InterlockedIncrement(&g_p_miss_d);
        }
        if (ok && pt->lastuse) { pt->lastuse[id] = (uint64_t) g_p_step; pt->sreg[id] = 0; }
        if (ok) { if (pt->freq) pt->freq[id] += 1.0; if (pt->hits) pt->hits[id] += 1; }
        pt->st[id] = ok ? 2 : 3;
        WakeAllConditionVariable(&pt->cv[id]);
        LeaveCriticalSection(&pt->cvlk[id]);
        return ok ? (const char *) (pt->arena + (size_t) id * pt->stride) : orig;
    }
}
}

// Worker prefill (teacher=hot / head=obs): called outside the resolver path; same per-slot CAS pattern.
static void e0_pool_prefill(int c, const std::vector<int> & ids, int reg) {    auto it = g_ptt_by_c.find(c);
    if (it == g_ptt_by_c.end()) return;
    for (PoolTT * pt : it->second)
        for (int id : ids) {
            if (pt->st[id] != 0) continue;
            CRITICAL_SECTION & lk = pt->cvlk[id];
            if (InterlockedCompareExchange(&pt->st[id], 1, 0) != 0) continue;
            EnterCriticalSection(&lk);
            bool ok = false;
            {
                std::lock_guard<std::mutex> lk(g_pmtx);
                if (g_pool_bytes[reg] + pt->stride <= g_pool_cap[reg]) {
                    uint8_t * dst = pt->arena + (size_t) id * pt->stride;
                    if (VirtualAlloc(dst, pt->stride, MEM_COMMIT, PAGE_READWRITE)) {
                        LARGE_INTEGER off = { (DWORD) (pt->foff + (uint64_t) id * pt->per),
                                              (LONG) ((pt->foff + (uint64_t) id * pt->per) >> 32) };
                        ok = SetFilePointerEx(g.fh, off, nullptr, FILE_BEGIN)
                             && ReadFile(g.fh, dst, (DWORD) pt->per, nullptr, nullptr);
                        if (ok) { g_pool_bytes[reg] += pt->stride; e0_pool_selfcheck(dst, pt->foff + (uint64_t) id * pt->per, pt->per); }
                    }
                }
            }
            if (ok && pt->lastuse) { pt->lastuse[id] = (uint64_t) g_p_step; pt->sreg[id] = (uint8_t) reg; }
            if (ok) { if (pt->freq) pt->freq[id] += 1.0; if (pt->hits) pt->hits[id] += 1; }
            pt->st[id] = ok ? 2 : 3;
            WakeAllConditionVariable(&pt->cv[id]);
            LeaveCriticalSection(&lk);
        }
}

// ── FATE prefetch issue: pool-aware (pool path skips re-filling committed slots) ──
// Resident check: pool mode asks the slot table (st==2 = the expert is in the private pages);
// the advisory path asks the pin table (already in the touch set, no PVM entry needed).
static bool e0_pool_expert_resident(int c, int e) {
    auto it = g_ptt_by_c.find(c);
    if (it == g_ptt_by_c.end()) return false;
    for (PoolTT * pt : it->second) if (e >= 0 && (size_t) e < pt->nexps && pt->st[e] == 2) return true;
    return false;
}
static bool e0_adv_expert_resident(int c, int e) {
    auto it = g.pins.find(c);
    if (it == g.pins.end()) return false;
    for (const auto & p : it->second) {
        auto pc = g.exps.find(c);
        if (pc == g.exps.end()) return false;
        for (const TT & tt : pc->second.t) {
            const size_t per = tt_bytes(tt) / EK;
            if (p.off == (size_t) g.data_base + tt.off + (size_t) e * per) return true;
        }
    }
    return false;
}
static bool e0_expert_resident(int c, int e, int reg) {
    if (g_pool_on) return reg == 1 ? false : e0_pool_expert_resident(c, e);   // obs has no residency table in this build
    return e0_adv_expert_resident(c, e);
}
// FATE issue: pool prefill (skips committed slots) or, in advisory mode, PVM warm-up over the
// non-resident predicted ranges. Miss = latency only: this only warms pages.
static void e0_issue_xlayer(int c, const std::vector<int> & ids, int reg) {
    if (ids.empty()) return;
    auto pc = g.exps.find(c);
    if (pc == g.exps.end()) return;
    if (g_pool_on) {
        std::vector<int> miss;
        for (int e : ids) if (!e0_expert_resident(c, e, reg)) miss.push_back(e);
        g.x_res += (long) (ids.size() - miss.size());
        e0_pool_prefill(c, miss, reg);
        return;
    }
    if (!g.view) return;
    static WIN32_MEMORY_RANGE_ENTRY win[3 * EK];
    int nw = 0;
    for (const TT & tt : pc->second.t) {
        const size_t per = tt_bytes(tt) / EK;
        for (int e : ids) {
            if (nw == 3 * EK) break;
            if (e0_expert_resident(c, e, reg)) continue;
            win[nw].VirtualAddress = (PVOID) ((char *) g.view + g.data_base + tt.off + (size_t) e * per);
            win[nw].NumberOfBytes  = per;
            ++nw;
        }
    }
    if (nw) {
        static decltype(&PrefetchVirtualMemory) pvm =
            (decltype(&PrefetchVirtualMemory)) GetProcAddress(GetModuleHandleA("kernel32"), "PrefetchVirtualMemory");
        if (pvm) pvm(GetCurrentProcess(), nw, win, 0);
    }
}

static void e0_pool_trace_final() {
    if (!g_pool_on) return;
    fprintf(stderr, "[pref-trace] POOL2 hit=%ld fillD=%ld bypass=%ld wait=%ld evict=%ld evictMB=%llu sc=%ld/%ld | hot=%lluMB obs=%lluMB\n",
            g_p_hits, g_p_miss_d, g_p_bypass, g_p_waits, g_p_evicts,
            (unsigned long long) (g_p_evict_bytes >> 20), (long) g_p_sc_ok, (long) g_p_sc_bad,
            (unsigned long long) (g_pool_bytes[0] >> 20), (unsigned long long) (g_pool_bytes[1] >> 20));
    // routing-count dump (borrowed from ninfer's write_routing_counts): one whitespace-separated
    // line of per-expert counts per tensor, for seeding the pool / the prerouter next run
    if (!g_stats_path.empty()) {
        FILE * f = fopen(g_stats_path.c_str(), "w");
        if (f) {
            for (auto & kv : g_ptt_by_name) {
                fprintf(f, "%s", kv.first.c_str());
                PoolTT * pt = kv.second;
                for (size_t id = 0; id < pt->nexps; ++id)
                    fprintf(f, " %llu", (unsigned long long) (pt->hits ? pt->hits[id] : 0));
                fprintf(f, "\n");
            }
            fclose(f);
            fprintf(stderr, "[pref-trace] POOL2 stats written: %s (%zu tensors)\n", g_stats_path.c_str(), g_ptt_by_name.size());
        }
    }
}

// Step-boundary eviction by decayed routing frequency (borrowed from ninfer-offload):
// retain the *frequently routed* experts, not just the recently used ones. Runs from on_step,
// after llama_synchronize, so no compute thread holds an expert pointer. Freed pages drop
// resident RAM; evicted slots return to state 0 and refill on demand. g_pmtx serializes
// against concurrent fills.
static void e0_pool_step() {
    if (!g_pool_on) return;
    const LONG now = InterlockedIncrement(&g_p_step);
    size_t freed = 0;
    long ev = 0;
    {
        std::lock_guard<std::mutex> lk(g_pmtx);
        // Only reclaim when the working set exceeds the pool (a fill was refused). If the
        // set fits, keep every slot (no churn, full coverage); eviction exists to bound RAM.
        if (g_pool_full) {
          if (g_pool_freq) {
            // frequency-based victim selection (ninfer-style): decay then evict the least
            // frequently routed stale slots, capped per step to bound churn.
            for (auto & kv : g_ptt_by_name) {
                PoolTT * pt = kv.second;
                if (!pt->freq) continue;
                for (size_t id = 0; id < pt->nexps; ++id) pt->freq[id] *= g_freq_decay;
            }
            struct Cand { PoolTT * pt; size_t id; double f; };
            std::vector<Cand> cand;
            for (auto & kv : g_ptt_by_name) {
                PoolTT * pt = kv.second;
                if (!pt->lastuse || !pt->sreg) continue;
                for (size_t id = 0; id < pt->nexps; ++id) {
                    if (pt->st[id] != 2) continue;
                    if (now - (LONG) pt->lastuse[id] <= g_keep_steps) continue;
                    cand.push_back({ pt, id, pt->freq ? pt->freq[id] : 0.0 });
                }
            }
            size_t cap = (size_t) g_max_swaps;
            if (cand.size() < cap) cap = cand.size();
            if (cap) {
                std::partial_sort(cand.begin(), cand.begin() + cap, cand.end(),
                                  [](const Cand & a, const Cand & b) { return a.f < b.f; });
                for (size_t i = 0; i < cap; ++i) {
                    PoolTT * pt = cand[i].pt;
                    size_t id = cand[i].id;
                    if (VirtualFree(pt->arena + id * pt->stride, pt->stride, MEM_DECOMMIT)) {
                        const int reg = pt->sreg[id] < 2 ? (int) pt->sreg[id] : 0;
                        g_pool_bytes[reg] = g_pool_bytes[reg] >= pt->stride ? g_pool_bytes[reg] - pt->stride : 0;
                        pt->st[id] = 0;
                        ++ev;
                        freed += pt->stride;
                    }
                }
            }
          } else {
            // recency-based (default): decommit every stale slot
            for (auto & kv : g_ptt_by_name) {
                PoolTT * pt = kv.second;
                if (!pt->lastuse || !pt->sreg) continue;
                for (size_t id = 0; id < pt->nexps; ++id) {
                    if (pt->st[id] != 2) continue;
                    if (now - (LONG) pt->lastuse[id] <= g_keep_steps) continue;
                    if (VirtualFree(pt->arena + id * pt->stride, pt->stride, MEM_DECOMMIT)) {
                        const int reg = pt->sreg[id] < 2 ? (int) pt->sreg[id] : 0;
                        g_pool_bytes[reg] = g_pool_bytes[reg] >= pt->stride ? g_pool_bytes[reg] - pt->stride : 0;
                        pt->st[id] = 0;
                        ++ev;
                        freed += pt->stride;
                    }
                }
            }
          }
        }
        g_p_evict_bytes += freed;
    }
    if (ev) InterlockedExchangeAdd(&g_p_evicts, (LONG) ev);
    if (ev && (g_p_step <= 3 || (g_p_step % 32) == 0))
        fprintf(stderr, "[pref-trace] POOL2 step=%ld evict=%ld freed=%lluMB hot=%lluMB obs=%lluMB (keep=%d)\n",
                now, ev, (unsigned long long) (freed >> 20), (unsigned long long) (g_pool_bytes[0] >> 20),
                (unsigned long long) (g_pool_bytes[1] >> 20), g_keep_steps);
}

static void e0_pool_scan(const std::map<std::string, TT> & tinfo) {   // pool all *_exps tensors (independent of consumer shapes)
    for (auto & kv : tinfo) {
        const std::string & nm = kv.first;
        const TT & tt = kv.second;
        auto suff = [&](const char * sfx) -> size_t {           // exact suffix match (returns length on hit)
            size_t len = strlen(sfx), k = nm.rfind(sfx);
            return (k != std::string::npos && k + len == nm.size()) ? len : 0;
        };
        size_t l1 = suff(".ffn_up_exps.weight"), l2 = suff(".ffn_gate_exps.weight"), l3 = suff(".ffn_down_exps.weight");
        size_t sl = l1 ? l1 : (l2 ? l2 : l3);
        if (!sl || tt.dim[2] < 2 || tt.dim[2] > 4096) continue;
        const size_t per = tt_bytes(tt) / (size_t) tt.dim[2];
        if (!per) continue;
        PoolTT * e = new PoolTT();
        e->per = per;
        e->stride = align_up(per, 4096);
        e->nexps = (size_t) tt.dim[2];
        e->foff = g.data_base + tt.off;
        e->arena = (uint8_t *) VirtualAlloc(nullptr, e->stride * (size_t) tt.dim[2], MEM_RESERVE, PAGE_READWRITE);
        e->st = (volatile LONG *) VirtualAlloc(nullptr, sizeof(LONG) * (size_t) tt.dim[2], MEM_COMMIT, PAGE_READWRITE);
        e->lastuse = (uint64_t *) VirtualAlloc(nullptr, sizeof(uint64_t) * e->nexps, MEM_COMMIT, PAGE_READWRITE);
        e->sreg = (uint8_t *) VirtualAlloc(nullptr, e->nexps, MEM_COMMIT, PAGE_READWRITE);
        e->freq = (double *) VirtualAlloc(nullptr, sizeof(double) * e->nexps, MEM_COMMIT, PAGE_READWRITE);
        e->hits = (uint64_t *) VirtualAlloc(nullptr, sizeof(uint64_t) * e->nexps, MEM_COMMIT, PAGE_READWRITE);
        e->cvlk = new CRITICAL_SECTION[(size_t) tt.dim[2]];
        e->cv = new CONDITION_VARIABLE[(size_t) tt.dim[2]];
        for (size_t z = 0; z < (size_t) tt.dim[2]; ++z) { InitializeCriticalSection(&e->cvlk[z]); InitializeConditionVariable(&e->cv[z]); }
        if (!e->arena || !e->st || !e->lastuse || !e->sreg || !e->freq || !e->hits) { delete[] e->cvlk; delete[] e->cv; e->cvlk = nullptr; e->cv = nullptr; delete e; continue; }
        g_ptt_by_name[nm] = e;
    }
    g_pool_on = !g_ptt_by_name.empty();
    if (g_pool_on) ggml_edge0_set_mmid_resolver(e0_mmid_base_res);   // attach to patch surface #7
    { int k = 0; for (auto & q : g_ptt_by_name) if (k++ < 3) fprintf(stderr, "[pref-trace] POOL2 key%d=%s per=%zu\n", k, q.first.c_str(), q.second->per); }
    fprintf(stderr, "[pref-trace] POOL2 init: tt=%zu caps hot=%lluMB obs=%lluMB resolver=%s\n",
            g_ptt_by_name.size(), (unsigned long long) (g_pool_cap[0] >> 20),
            (unsigned long long) (g_pool_cap[1] >> 20), g_pool_on ? "on" : "FAIL");
    fflush(stderr);
}

// L1 -> L0 fill source for the VRAM tier: if the RAM pool currently holds expert `id`
// of tensor `name`, hand its bytes to the VRAM tier so the fill skips the file read.
// Called on the compute thread from the backend resolver; a slot is only read once its
// state is 2 (ready) and slots are never overwritten in this pool, so the read is safe.
static const void * e0_vram_host_src(void * /*user*/, const char * name, int64_t id, size_t * len) {
    auto it = g_ptt_by_name.find(name ? name : "");
    if (it == g_ptt_by_name.end()) return nullptr;
    PoolTT * pt = it->second;
    if (!pt || (size_t) id >= pt->nexps || pt->st[id] != 2) return nullptr;
    *len = pt->per;
    return (const char *) (pt->arena + (size_t) id * pt->stride);
}

}  // namespace

static void enable_capture(llama_context * ctx) {
    for (int L = OWNER0; L <= OWNER0 + NH; ++L) llama_set_embeddings_layer_inp(ctx, (uint32_t) L, true);
}

// ── Pool re-touch ladder: deterministic accounting (zero prediction = sticky; + observation zone = sticky/obs) ──
// Mechanism: VirtualLock needs SeLockMemoryPrivilege, which consumer machines lack ⇒ pinning
// is NOT the product mechanism. What works unprivileged = demand touch + working-set hard cap
// + the OS clock's natural LRU:
//   "pin"   = worker volatile-reads the range in 4K steps (demands pages into the WS; any
//             trimmed page's fault tax is paid inside the worker, off the critical path);
//   "evict" = drop from the table only (stop re-touching → reference-bit ages → OS trims it
//             into standby = free soft layer);
//   "cap"   = at init, SetProcessWorkingSetSizeEx(MAX = baseline + hot + obs + headroom, HARDWS_MAX_ENABLE).
// Hot zone = re-touch of the previous step's teacher ids (same experts in a row = zero-fault
// hit); observation zone = head-guess quota (a bad guess only ages its own pages — the hot
// zone re-touches every step and never ages = contamination ceiling). Low-memory notification
// ⇒ shrink the tables (hot to 75%, observation to empty).
static volatile char g_touch_sink;                  // prevent DCE (every page is really read)
static void e0_touch(const void * view, size_t off, size_t len) {
    const volatile char * p = (const volatile char *) view + off;
    char s = 0;
    for (size_t i = 0; i < len; i += 4096) s += p[i];
    g_touch_sink = s;
}
static void e0_pool_evict(int R) {                  // evict the oldest entry in region R (age==0 entries are exempt)
    int wc = -1, wi = -1, ba = 0;
    for (auto & pc : g.pins)
        for (int k = 0; k < (int) pc.second.size(); ++k) {
            auto & p = pc.second[k];
            if (p.reg == R && p.age > ba) { ba = p.age; wc = pc.first; wi = k; }
        }
    if (wc < 0) return;
    g.reg_bytes[R] -= g.pins[wc][wi].len; ++g.evicts;
    g.pins[wc].erase(g.pins[wc].begin() + wi);      // dropping from the table = stop re-touching; physical reclaim is the OS clock's job
}
static void e0_issue_pool() {
    if (!g.view) return;
    const int nreg = g.pool == 2 ? 2 : 1;
    std::map<int, std::vector<std::pair<size_t, size_t>>> want[2];   // 0=hot(teacher) 1=observation(head)
    auto ranges = [&](const std::map<int, std::vector<int>> & src, int R) {
        for (auto & pc : src) {
            auto & v = want[R][pc.first];
            for (const TT & tt : g.exps[pc.first].t) {
                const size_t per = tt_bytes(tt) / EK;
                for (int e : pc.second)
                    v.push_back({ (size_t) g.data_base + tt.off + (size_t) e * per, per });
            }
        }
    };
    ranges(g.exec_prev, 0);
    if (nreg > 1) {
        std::map<int, std::vector<int>> preds;
        { std::lock_guard<std::mutex> lk(g.wmtx); preds = g.pred_prev; }
        ranges(preds, 1);
    }
    for (auto & kv : g.pins) for (auto & p : kv.second) ++p.age;     // uniform step-rate aging
    std::vector<std::pair<size_t, size_t>> touch;                     // this step's demand set (≤ table cap)
    for (int R = 0; R < nreg; ++R) {
        for (auto & wv : want[R]) {
            auto & cur = g.pins[wv.first];
            std::vector<char> hit(wv.second.size(), 0);
            for (size_t i = 0; i < wv.second.size(); ++i)
                for (auto & p : cur)
                    if (p.reg == R && p.off == wv.second[i].first) { p.age = 0; hit[i] = 1; if (!R) ++g.pool_hit; break; }
            if (!R) g.pool_tot += (long) wv.second.size();           // denominator = all hot-zone demand this step
            for (size_t i = 0; i < wv.second.size(); ++i) {
                if (hit[i]) continue;
                auto & w = wv.second[i];
                while (g.reg_bytes[R] + w.second > g.reg_cap[R]) {
                    size_t b = g.reg_bytes[R];
                    e0_pool_evict(R);
                    if (g.reg_bytes[R] == b) { ++g.lock_fail; break; }  // nothing evictable ⇒ hard ceiling (refused-pin counter)
                }
                if (g.reg_bytes[R] + w.second > g.reg_cap[R]) { ++g.lock_fail; continue; }
                cur.push_back({ w.first, w.second, 0, (uint8_t) R });
                g.reg_bytes[R] += w.second;
            }
        }
    }
    for (auto & pc : g.pins) for (auto & p : pc.second) e0_touch(g.view, p.off, p.len);  // re-touch the whole table
    typedef HANDLE(WINAPI * QMRN)(LPCWSTR, void *);                   // memory-pressure cooperation (table-side fast shrink)
    typedef BOOL(WINAPI * GMRN_T)(HANDLE, PBOOL);
    static QMRN qmrn = (QMRN) GetProcAddress(GetModuleHandleA("kernel32"), "QueryMemoryResourceNotification");
    static GMRN_T gmrn = (GMRN_T) GetProcAddress(GetModuleHandleA("kernel32"), "GetMemoryResourceNotification");
    static HANDLE hn = qmrn ? qmrn(L"MemoryLow", nullptr) : nullptr;
    if (hn && gmrn) {
        BOOL low = FALSE;
        if (gmrn(hn, &low) && low) {
            ++g.shrinks;
            for (int R = 0; R < nreg; ++R)
                while (g.reg_bytes[R] && (R || g.reg_bytes[R] * 4 > g.reg_cap[R] * 3)) {
                    size_t b = g.reg_bytes[R];
                    e0_pool_evict(R);
                    if (g.reg_bytes[R] == b) break;
                }
        }
    }
}

// ── Advisory issue: PVM soft warm-up + VirtualLock hard retention (lock failure once the
// budget is hit = natural ceiling). Callable from the worker thread or the main thread
// (synchronous / stride-skipped); pred_prev is copied under a short wmtx lock.
static void e0_issue() {
    if (g.pool) { e0_issue_pool(); return; }          // pool accounting takes over (legacy pin/prefetch path retired)
    if (!g.view || (!g.do_prefetch && !g.do_pin)) return;
    std::map<int, std::vector<int>> preds;
    { std::lock_guard<std::mutex> lk(g.wmtx); preds = g.pred_prev; }
    static WIN32_MEMORY_RANGE_ENTRY win[4 * 3 * NH];
    int nw = 0;
    for (auto & pc : preds) {
        const int c = pc.first;
        std::vector<std::pair<size_t, size_t>> want;
        for (const TT & tt : g.exps[c].t) {
            const size_t per = tt_bytes(tt) / EK;
            for (int e : pc.second) want.push_back({ (size_t) g.data_base + tt.off + (size_t) e * per, per });
        }
        if (g.do_pin) {
            auto & cur = g.pinned[c];
            for (auto & r : cur) {                                   // release stale pins
                bool keep = false;
                for (auto & w : want) if (w.first == r.first) { keep = true; break; }
                if (!keep) { VirtualUnlock((char *) g.view + r.first, r.second); g.pin_bytes -= r.second; }
            }
            for (auto & w : want) {                                  // pin new predictions (skipped once capped)
                bool have = false;
                for (auto & r : cur) if (r.first == w.first) { have = true; break; }
                if (have || g.pin_bytes + w.second > g.pin_cap) continue;
                if (VirtualLock((char *) g.view + w.first, w.second)) g.pin_bytes += w.second;
            }
            cur = want;   // remember current pin state (entries skipped by the cap unlock harmlessly next round)
        }
        if (g.do_prefetch)
            for (auto & w : want) {
                if (nw == 4 * 3 * NH) break;
                win[nw].VirtualAddress = (PVOID) ((char *) g.view + w.first);
                win[nw].NumberOfBytes  = w.second;
                ++nw;
            }
    }
    if (nw) {
        static decltype(&PrefetchVirtualMemory) pvm =
            (decltype(&PrefetchVirtualMemory)) GetProcAddress(GetModuleHandleA("kernel32"), "PrefetchVirtualMemory");
        if (pvm) pvm(GetCurrentProcess(), nw, win, 0);
    }
}

// ── Full-step computation: (1) teacher (2) three-way accounting (3) head → preds + roll history (4) issue ──
// rows = snapshot buffer, row order = layers OWNER0..OWNER0+NH, HH floats per row.
// This function has no ctx dependency ⇒ it can run on the worker thread (keeps the ~90ms
// head GEMV off the critical path).
static void e0_full_step(const float * rows) {
    std::map<int, std::vector<int>> teacher;
    for (int c : consumers) {
        const float * x = rows + (size_t) (c - OWNER0) * HH;
        std::vector<float> tl(EK);
        const float * gw = g.gate[c].data();
        for (int e = 0; e < EK; ++e) tl[e] = dotf(gw + (size_t) e * HH, x, HH);
        topk(tl.data(), EK, KDIM, teacher[c]);
    }
    // Three-way settlement: pred copy (short lock) vs teacher @ this snapshot.
    std::map<int, std::vector<int>> preds_copy, ext_copy;
    { std::lock_guard<std::mutex> lk(g.wmtx); preds_copy = g.pred_prev; ext_copy = g.pred_ext_prev; }
    for (int c : consumers) {
        auto pp = preds_copy.find(c);
        if (pp != preds_copy.end()) {
            g.hits += ovlp(pp->second, teacher[c]); g.total += KDIM;
            if (g.xlayer) {
                g.x_hits += ovlp(pp->second, teacher[c]); g.x_tot += KDIM;
                auto xe = ext_copy.find(c);
                if (xe != ext_copy.end()) g.x_ext_hits += ovlp(xe->second, teacher[c]);
            }
            auto sp = g.exec_prev.find(c);
            if (sp != g.exec_prev.end()) g.sticky += ovlp(sp->second, teacher[c]);
            std::vector<int> r;
            for (int tries = 0; tries < 64 && (int) r.size() < KDIM; ++tries) {
                int v = (int) (g.next() % (uint32_t) EK);
                bool dup = false; for (int y : r) if (y == v) { dup = true; break; }
                if (!dup) r.push_back(v);
            }
            g.rnd += ovlp(r, teacher[c]);
        }
        g.exec_prev[c] = teacher[c];
    }
    if (g_pool_on) for (auto & kv : g.exec_prev) e0_pool_prefill(kv.first, kv.second, 0);   // teacher → hot zone
    // head: owner o features = [m_in|cur_oh|prev_oh] → new pred (consumer = o+1)
    std::map<int, std::vector<int>> new_preds;
    std::map<int, std::vector<int>> new_ext;
    if (g.xlayer) {
        // FATE: predict consumer c from owner o = c-1's captured gate input. Gate_in_{o+1} is
        // used to predict layer o+1, the paper's exact scheme; c == o+1 here.
        for (int o = OWNER0; o < OWNER0 + NH; ++o) {
            const int c = o + 1;
            const float * x = rows + (size_t) (o - OWNER0) * HH;
            const float * gw = g.gate[c].data();
            std::vector<float> lg(EK);
            for (int e = 0; e < EK; ++e) lg[e] = dotf(gw + (size_t) e * HH, x, HH);
            std::vector<int> tk, ext;
            topk(lg.data(), EK, KDIM, tk);
            float thr;
            topk_pct_cut(lg.data(), EK, g.x_pct, g.x_ext_max, ext, thr);
            new_preds[c] = tk;                                    // top-K: contract for pred_prev/accounting
            new_ext[c]   = ext;
            e0_issue_xlayer(c, ext, g_pool_cap[1] ? 1 : 0);       // top-K + percentile set
            g.x_issued += (long) ext.size();
        }
    } else {
    std::vector<float> feat(FE), act(512), lg(EK);
    for (int o = OWNER0; o < OWNER0 + NH; ++o) {
        const int c = o + 1;
        auto & h = g.hist[o];
        memcpy(feat.data(),          rows + (size_t) (o - OWNER0) * HH, HH * 4);
        memcpy(feat.data() + HH,     h.cur.data(),  EK * 4);
        memcpy(feat.data() + HH + EK, h.prev.data(), EK * 4);
        const Head & hd = g.heads[o];
        for (int j = 0; j < 512; ++j) act[j] = gelu_erf(dotf(hd.fc1.data() + (size_t) j * FE, feat.data(), FE));
        for (int e = 0; e < EK; ++e)
            lg[e] = dotf(hd.fc2.data() + (size_t) e * 512, act.data(), 512)
                  + dotf(hd.lin.data() + (size_t) e * FE, feat.data(), FE);
        topk(lg.data(), EK, KDIM, new_preds[c]);
    }
    }
    if (g_pool_on && g_pool_cap[1]) for (auto & kv : new_preds) e0_pool_prefill(kv.first, kv.second, 1);  // head → observation zone
    { std::lock_guard<std::mutex> lk(g.wmtx); g.pred_prev = std::move(new_preds); g.pred_ext_prev = std::move(new_ext); }
    // roll the onehot history (teacher→cur, cur→prev)
    for (auto & kv : g.hist) {
        auto t = teacher.find(kv.first);
        if (t == teacher.end()) continue;
        kv.second.prev = kv.second.cur;
        kv.second.cur.assign(EK, 0.f);
        for (int e : t->second) kv.second.cur[e] = 1.f;
    }
    e0_issue();
    if (g.trace_every && g.steps % g.trace_every == 0 && g.total) {
        fprintf(stderr, "[pref-trace] step %ld: head %.1f%% sticky %.1f%% rand %.1f%% pinned=%lluMB (n=%ld)\n",
                g.steps, 100.0 * g.hits / g.total, 100.0 * g.sticky / g.total,
                100.0 * g.rnd / g.total, (unsigned long long) (g.pin_bytes >> 20), g.total);
        if (g.pool)
            fprintf(stderr, "[pref-trace]  POOL hit=%.1f%% (n=%ld) pins %llu+%lluMB evict=%ld fail=%ld shrink=%ld\n",
                    100.0 * g.pool_hit / (g.pool_tot ? g.pool_tot : 1), g.pool_tot,
                    (unsigned long long) (g.reg_bytes[0] >> 20), (unsigned long long) (g.reg_bytes[1] >> 20),
                    g.evicts, g.lock_fail, g.shrinks);
        fflush(stderr);
    }
}

// FATE prefill (arXiv:2502.12224 §4.2/§4.3): a prefill ubatch activates many experts per
// layer. Aggregate token->expert assignments and issue in descending popularity, so the
// experts serving the most tokens become visible to the OS first. Popularity = the number of
// tokens that select the expert in their top-K (the paper's frequency count).
static void e0_xlayer_prefill() {
    if (g.pf_tok <= 0) return;
    for (int o = OWNER0; o < OWNER0 + NH; ++o) {
        const int c = o + 1;
        const float * gw = g.gate[c].data();
        auto & ss = g.snap_pf;
        std::vector<int> cnt(EK, 0);
        for (int t = 0; t < g.pf_tok; ++t) {
            const float * x = ss.data() + ((size_t) t * (NH + 1) + (size_t) (o - OWNER0)) * HH;
            std::vector<float> lg(EK);
            for (int e = 0; e < EK; ++e) lg[e] = dotf(gw + (size_t) e * HH, x, HH);
            std::vector<int> tk;
            topk(lg.data(), EK, KDIM, tk);
            for (int e : tk) cnt[e] += 1;
        }
        std::vector<float> pop(EK);
        for (int e = 0; e < EK; ++e) pop[e] = (float) cnt[e];
        int k = g.pf_tok * KDIM; if (k > EK) k = EK; if (g.x_pf_cap < k) k = g.x_pf_cap;
        std::vector<int> ids;
        topk(pop.data(), EK, k, ids);
        e0_issue_xlayer(c, ids, 0);
        g.x_issued += (long) ids.size();
    }
    g.pf_tok = 0;
}

static void e0_worker() {
    for (;;) {
        {
            std::unique_lock<std::mutex> lk(g.wmtx);
            g.cv.wait(lk, [&] { return g.quit.load() || g.busy.load(); });
            if (g.quit) return;
        }
        if (g.busy.load()) {
            if (g.prefill) e0_xlayer_prefill();
            else           e0_full_step(g.snap.data());
            g.busy = false;
        }
    }
}

bool edge0_pref_init(llama_model * model, llama_context * ctx, int pool_mb, const char * gguf_path) {
    (void) model;
    if (g.active) { if (!g.gate.empty()) enable_capture(ctx); return true; }   // hot reload (only head mode needs capture)
    const char * side = getenv("E0_PREROUTER");
    const char * gg   = (gguf_path && *gguf_path) ? gguf_path : getenv("E0_GGUF");   // primary CLI channel = model path
    if (pool_mb > 0) g.pool_mb = pool_mb;                                        // --pool-mb formal CLI
    else if (const char * pmb = getenv("E0_POOL_MB")) g.pool_mb = atoll(pmb);    // env fallback (harness/debug); runnable without the sidecar
    // Must be parsed HERE, not at its later use site: the early return below tests g.vram_mb, so
    // reading it after the gate made E0_VRAM_MB alone a no-op (the tier only came up with --pool-mb).
    if (const char * vm = getenv("E0_VRAM_MB")) g.vram_mb = atoll(vm);
    g.xlayer = getenv("E0_XLAYER") != nullptr;                                   // FATE (T3): training-free cross-layer gate prefetch
    if (const char * xp = getenv("E0_XLAYER_PCT")) { float v = (float) atof(xp); if (v > 0.f && v < 1.f) g.x_pct = v; }
    if (const char * xe = getenv("E0_XLAYER_EXT_MAX")) { int v = atoi(xe); if (v > 0) g.x_ext_max = v; }
    if (const char * xc = getenv("E0_XLAYER_PF_TOK_MAX")) { int v = atoi(xc); if (v > 0) g.x_pf_cap = v; }
    if (const char * xt = getenv("E0_XLAYER_PF_TOK")) { int v = atoi(xt); if (v > 0) g.x_pf_tok = v; }
    // FATE needs the F32 gate + the layer-inp capture, exactly like the sidecar head, but without
    // the safetensors heads. treat gate-mode as "headson" for the plumbing and skip the sidecar below.
    const bool headson = (side && *side) || g.xlayer;
    if ((!headson && !g.pool_mb && !g.vram_mb) || !gg || !*gg) return false;

    if (consumers.empty()) for (int c = OWNER0 + 1; c <= OWNER0 + NH; ++c) consumers.push_back(c);  // 7..39

    // ── GGUF header: skip-scan KVs (for general.alignment) + build the tensor table ──
    FILE * f = fopen(gg, "rb");
    if (!f) return false;
    struct { uint32_t magic, ver; uint64_t nt, nkv; } h;
    if (fread(&h, sizeof h, 1, f) != 1 || memcmp(&h.magic, "GGUF", 4)) { fclose(f); return false; }
    size_t alignment = 32;
    for (uint64_t i = 0; i < h.nkv; ++i) {
        std::string k = rd_str(f);
        uint32_t t; if (fread(&t, 4, 1, f) != 1) { fclose(f); return false; }
        if (k == "general.alignment" && t == 4) fread(&alignment, 4, 1, f);
        else skip_val(f, t);
    }
    std::map<std::string, TT> tinfo;
    for (uint64_t i = 0; i < h.nt; ++i) {
        TT x; x.name = rd_str(f);
        uint32_t nd; fread(&nd, 4, 1, f);
        for (uint32_t d = 0; d < nd && d < 4; ++d) fread(&x.dim[d], 8, 1, f);
        fread(&x.type, 4, 1, f); fread(&x.off, 8, 1, f);
        tinfo[x.name] = x;
    }
    g.data_base = align_up((uint64_t) ftell(f), alignment);

    // Register exps metadata (needed by both pool and head) + copy the F32 gate weights host-side (head mode only).
    for (int c : consumers) {
        for (const char * suf : {"ffn_up_exps", "ffn_gate_exps", "ffn_down_exps"}) {
            auto et = tinfo.find("blk." + std::to_string(c) + "." + suf + ".weight");
            if (et != tinfo.end() && et->second.dim[2] == (uint64_t) EK) g.exps[c].t.push_back(et->second);
        }
        if (!headson) continue;
        auto it = tinfo.find("blk." + std::to_string(c) + ".ffn_gate_inp.weight");
        if (it == tinfo.end() || it->second.type != 0) {
            fclose(f); fprintf(stderr, "[pref-trace] gate missing/bad type c=%d => dormant\n", c); return false;
        }
        const TT & gt = it->second;
        if ((uint64_t) gt.dim[0] != (uint64_t) HH || (uint64_t) gt.dim[1] != (uint64_t) EK) {
            fclose(f); fprintf(stderr, "[pref-trace] gate dims %llux%llu != %dx%d c=%d => dormant\n",
                               (unsigned long long) gt.dim[0], (unsigned long long) gt.dim[1], HH, EK, c); return false;
        }
        size_t nb = tt_bytes(gt);
        std::vector<float> buf(nb / 4);
        _fseeki64(f, (long long) (g.data_base + gt.off), SEEK_SET);
        if (fread(buf.data(), 1, nb, f) != nb) { fclose(f); return false; }
        g.gate[c] = std::move(buf);
    }
    fclose(f);

    // sidecar head weights (sidecar mode only; FATE has no trained heads)
    if (side && *side) {
    std::map<std::string, SEnt> st;
    if (!st_load(side, st)) { fprintf(stderr, "[pref-trace] sidecar parse failed => dormant\n"); return false; }
    FILE * sf = fopen(side, "rb");
    if (!sf) return false;
    uint64_t shn = 0; fread(&shn, 8, 1, sf);
    uint64_t sdata = 8 + shn;
    for (int o = OWNER0; o < OWNER0 + NH; ++o) {
        Head hd;
        auto key = [&](const char * w) { return "layers." + std::to_string(o) + "." + w + ".weight"; };
        bool ok = grab_f16(sf, sdata, st[key("fc1")], hd.fc1)
               && grab_f16(sf, sdata, st[key("fc2")], hd.fc2)
               && grab_f16(sf, sdata, st[key("linear_init")], hd.lin);
        if (!ok) { fclose(sf); fprintf(stderr, "[pref-trace] sidecar missing layer o=%d => dormant\n", o); return false; }
        g.heads[o] = std::move(hd);
    }
    fclose(sf);
    }   // headson

    // Dedicated read handle (pool fills via g.fh) + a self-owned read-only mapping (PVM anchor; head mode only).
    if (headson || g.pool_mb) {
        g.fh = CreateFileA(gg, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                           NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        if (g.fh != INVALID_HANDLE_VALUE && headson) {
            g.mh = CreateFileMappingA(g.fh, NULL, PAGE_READONLY, 0, 0, NULL);
            if (g.mh) g.view = MapViewOfFile(g.mh, FILE_MAP_READ, 0, 0, 0);
        }
    }
    g.do_prefetch = getenv("E0_PREFETCH") != nullptr;
    if (const char * tn = getenv("E0_PREF_TRACE_N")) g.trace_every = atoi(tn);
    if (const char * sn = getenv("E0_PREF_STRIDE")) g.stride = atoi(sn) > 0 ? atoi(sn) : 1;
    const char * an = getenv("E0_PREF_ASYNC");
    g.async = !(an && *an == '0');                     // async is the default; =0 falls back to synchronous (tax isolation)
    g.do_pin = getenv("E0_PREF_PIN") != nullptr;
    if (const char * pm = getenv("E0_PIN_MAX_MB")) g.pin_cap = (size_t) atoll(pm) << 20;
    // Device-resident hot-expert tier (L0). Independent of the RAM pool and the head: it
    // pools every *.ffn_*_exps.weight and is served by the HIP backend hook (patch #8).
    if (g.vram_mb > 0) {
        // Its own read handle (falls back to g.fh when the RAM pool shares one).
        HANDLE vh = g.fh;
        if (vh == INVALID_HANDLE_VALUE)
            vh = CreateFileA(gg, GENERIC_READ, FILE_SHARE_READ | FILE_SHARE_WRITE | FILE_SHARE_DELETE,
                             NULL, OPEN_EXISTING, FILE_ATTRIBUTE_NORMAL, NULL);
        // Inventory every *_exps.weight whose last dim is the expert count (mirrors e0_pool_scan).
        std::vector<e0_vram_tensor> vt;
        std::vector<std::string> names;   // stable storage; names[i].c_str() assigned below
        for (auto & kv : tinfo) {
            const std::string & nm = kv.first;
            const TT & tt = kv.second;
            auto rsuf = [&](const char * sfx) { size_t len = strlen(sfx), k = nm.rfind(sfx); return k != std::string::npos && k + len == nm.size(); };
            if (!rsuf(".ffn_up_exps.weight") && !rsuf(".ffn_gate_exps.weight") && !rsuf(".ffn_down_exps.weight")) continue;
            if (tt.dim[2] < 2 || tt.dim[2] > 4096) continue;
            size_t per = tt_bytes(tt) / (size_t) tt.dim[2];
            if (!per) continue;
            int layer = -1;
            { int blk = -1; if (sscanf(nm.c_str(), "blk.%d.", &blk) == 1) layer = blk; }
            names.push_back(nm);
            vt.push_back({ nullptr, g.data_base + tt.off, per, (size_t) tt.dim[2], layer });
        }
        for (size_t i = 0; i < vt.size(); ++i) vt[i].name = names[i].c_str();   // names is final here (no realloc)
        // Prefer the handle the RAM pool opened; else a private one.
        if (vh == INVALID_HANDLE_VALUE) { fprintf(stderr, "[vram-trace] no GGUF handle => tier off\n"); }
        else {
            edge0_vram_set_host_src(e0_vram_host_src, nullptr);   // L1 -> L0 promotion (no-op until the pool is live)
            edge0_vram_init((int) g.vram_mb, vt.data(), vt.size(), (void *) vh, gg);
        }
    }
    if (const char * pl = getenv("E0_POOL")) {                      // sticky=zone1  obs=zone1+2
        if (!strcmp(pl, "obs")) g.pool = 2;
        else if (*pl && strcmp(pl, "0")) g.pool = 1;
    }
    if (g.pool) {
        g.reg_cap[0] = g.pin_cap;                                   // hot zone = E0_PIN_MAX_MB (default 3G)
        const char * om = getenv("E0_POOL_OBS_MB");
        g.reg_cap[1] = (size_t) (om ? atoll(om) : 1024) << 20;      // observation zone default 1G
        g.do_pin = false;                                           // the advisory pin path does not run in parallel
        // Working-set hard cap = startup baseline WS + pool quotas + headroom (the residency contract without SeLockMemory).
        PROCESS_MEMORY_COUNTERS_EX pmc = {};
        pmc.cb = sizeof(pmc);
        if (K32GetProcessMemoryInfo(GetCurrentProcess(), (PROCESS_MEMORY_COUNTERS *) &pmc, sizeof(pmc))) {
            const char * hm = getenv("E0_POOL_WSHEAD_MB");
            size_t head = (size_t) (hm ? atoll(hm) : 1024) << 20;
            SIZE_T wmax = (SIZE_T) pmc.WorkingSetSize + g.reg_cap[0] + g.reg_cap[1] + head;
            SIZE_T qmin = 0, qmax = 0; DWORD qf = 0;      // the Ex variant rejects min=-1 (error 87) ⇒ read current quotas and keep them
            GetProcessWorkingSetSizeEx(GetCurrentProcess(), &qmin, &qmax, &qf);
            BOOL r = SetProcessWorkingSetSizeEx(GetCurrentProcess(), qmin, wmax,
                                                QUOTA_LIMITS_HARDWS_MIN_DISABLE | QUOTA_LIMITS_HARDWS_MAX_ENABLE);
            fprintf(stderr, "[pref-trace] POOL ws-cap: base=%lluMB max=%lluMB set=%d err=%lu\n",
                    (unsigned long long) (pmc.WorkingSetSize >> 20), (unsigned long long) (wmax >> 20),
                    (int) r, r ? 0UL : GetLastError());
        }
    }
    g.snap.assign((size_t) (NH + 1) * HH, 0.f);

    // Open the layer-input capture channel (same staging API as speculative decode; head mode only).
    if (headson) {
        enable_capture(ctx);
        for (int L = OWNER0; L <= OWNER0 + NH; ++L) g.hist[L] = Hist{ std::vector<float>(EK, 0.f), std::vector<float>(EK, 0.f) };
    }
    if (g.pool_mb) {
        g_pool_cap[0] = (size_t) g.pool_mb << 20;
        if (const char * pom = getenv("E0_POOL_OBS_MB")) g_pool_cap[1] = (size_t) atoll(pom) << 20;
        if (const char * ks = getenv("E0_POOL_KEEP_STEPS")) { int v = atoi(ks); if (v >= 0) g_keep_steps = v; }
        if (const char * ms = getenv("E0_POOL_MAX_SWAPS")) { int v = atoi(ms); if (v > 0) g_max_swaps = v; }
        if (const char * sp = getenv("E0_POOL_STATS")) g_stats_path = sp;
        g_pool_freq = getenv("E0_POOL_FREQ") != nullptr;
        g_pool_sc = getenv("E0_POOL_SELFCHECK") != nullptr;
        if (g_pool_sc && !g.view) {      // self-check needs the view: map it even in pool-only mode
            g.mh = CreateFileMappingA(g.fh, NULL, PAGE_READONLY, 0, 0, NULL);
            if (g.mh) g.view = MapViewOfFile(g.mh, FILE_MAP_READ, 0, 0, 0);
        }
        e0_pool_scan(tinfo);
        for (int c : consumers)
            for (const TT & tt : g.exps[c].t) {
                auto it = g_ptt_by_name.find(tt.name);
                if (it != g_ptt_by_name.end()) g_ptt_by_c[c].push_back(it->second);
            }
    }
    g.active = headson || g_pool_on || edge0_vram_active();
    if (g.async) g.snap_pf.assign((size_t) g.x_pf_tok * (NH + 1) * HH, 0.f);   // FATE prefill snapshot
    if (g.active && !headson) { g.async = false; }        // pool-only mode: no worker (demand is self-sufficient)
    if (g.active && headson && g.async) g.worker = std::thread(e0_worker);
    fprintf(stderr, "[pref-trace] INIT ok: heads=%zu gates=%zu exps=%zu map=%s prefetch=%s pin=%s(async=%d) stride=%d K=%d E=%d pool=%d(hot=%lluMB obs=%lluMB) vram=%s\n",
            g.heads.size(), g.gate.size(), g.exps.size(), g.view ? "ok" : "FAIL",
            g.do_prefetch ? "on" : "off", g.do_pin ? "on" : "off", (int) g.async, g.stride, KDIM, EK,
            (int) g_pool_on, (unsigned long long) (g_pool_cap[0] >> 20), (unsigned long long) (g_pool_cap[1] >> 20),
            edge0_vram_active() ? "on" : "off");
    return g.active;
}

void edge0_pref_on_step(llama_context * ctx, int64_t last_row) {
    if (!g.active) return;
    // Step boundary: the GPU work for this step has completed. The RAM pool decommit and
    // the VRAM ARC/hot-pin maintenance both recycle memory the GPU may have read, so they
    // run only here, after a synchronize, and never on the compute path.
    if (g_pool_on) { llama_synchronize(ctx); e0_pool_step(); }
    if (edge0_vram_active()) { llama_synchronize(ctx); edge0_vram_step(); }
    if (g.gate.empty()) return;                 // pool-only mode: no capture channel, nothing to account — demand is autonomous
    llama_synchronize(ctx);
    ++g.steps;

    if (g.stride > 1 && (g.steps % g.stride) != 0) {   // downsampled: reuse existing preds, just re-issue (µs scale)
        if (!g.async && !g.pool) e0_issue();            // async=worker already issued on cadence; pool=skipped steps neither age nor evict
        return;
    }
    // The capture buffer holds one row per ubatch token; the wire-up passes last_row =
    // batch_view.n_tokens - 1, so n_tokens = last_row + 1 (multi-token => prefill).
    const size_t ntok = (size_t) (last_row + 1);
    const bool is_prefill = g.xlayer && ntok > 1;
    if (g.async) {
        if (g.busy.load()) return;                     // worker busy: drop this snapshot (preds already valid, lossless)
        if (is_prefill) {
            int take = (int) ntok; if (take > g.x_pf_tok) take = g.x_pf_tok;
            for (int t = 0; t < take; ++t) {
                float * dst = g.snap_pf.data() + (size_t) t * (NH + 1) * HH;
                for (int L = OWNER0; L <= OWNER0 + NH; ++L, dst += HH) {
                    float * buf = llama_get_embeddings_layer_inp(ctx, (uint32_t) L);
                    if (!buf) return;
                    memcpy(dst, buf + (size_t) t * HH, (size_t) HH * 4);
                }
            }
            g.pf_tok = take; g.prefill = true;
        } else {
            float * dst = g.snap.data();
            for (int L = OWNER0; L <= OWNER0 + NH; ++L, dst += HH) {
                float * buf = llama_get_embeddings_layer_inp(ctx, (uint32_t) L);
                if (!buf) return;                      // channel not ready
                memcpy(dst, buf + (size_t) last_row * HH, (size_t) HH * 4);
            }
            g.prefill = false;
        }
        { std::lock_guard<std::mutex> lk(g.wmtx); g.busy = true; }
        g.cv.notify_one();
        return;
    }
    // Synchronous path: snapshot + full step inline (reference/fallback).
    float * dst = g.snap.data();
    for (int L = OWNER0; L <= OWNER0 + NH; ++L, dst += HH) {
        float * buf = llama_get_embeddings_layer_inp(ctx, (uint32_t) L);
        if (!buf) return;
        memcpy(dst, buf + last_row * HH, (size_t) HH * 4);
    }
    e0_full_step(g.snap.data());
}
