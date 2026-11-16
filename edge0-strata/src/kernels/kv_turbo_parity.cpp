// src/kernels/kv_turbo_parity.cpp - Parity test for the Turbo3 and Rot4 KV codecs (kv_turbo.hpp).
//
// What it proves, and why each check is here:
//  1. APPEND writes the layout the header documents: the row's sigma (turbo3) / per-32 scale (rot4) matches a host
//     reference of the same statistic, and the codes decode to the rotated input.
//  2. GATHER is CONSISTENT WITH APPEND, bitwise: dequantising the pool bytes on the host reproduces the FP16
//     scratch the kernel gathered, exactly. A pack/unpack mismatch (an off-by-one in a group offset, a wrong bit
//     shift, the wrong byte order) fails here even when it is symmetric enough to look plausible.
//  3. DISTORTION is the codec's and not garbage: the reconstruction error against the rotated input stays inside
//     the 3-bit Lloyd-Max bound, and <q, k_hat> tracks <q, k>.
//  4. The FORMAT IS SMALLER than the formats it replaces, at the byte counts the engine sizes pools from.
//
// Run it before enabling --kv turbo3 / --kv rot4 on a new pack: a codec that is merely wrong produces plausible,
// finite, wrong attention, which no other check in this repo would catch.

#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_turbo.hpp"
#include "strata/kernels/qsa.hpp"

// The independent implementation, and therefore the only real oracle available. Everything in this file up to
// now compares the GPU kernel against a host twin written from the same header, which cannot catch "both wrong
// the same way". Kraken-x's codec is a different codebase, and `third_party/Kraken-x/tools/turbo_kv_selftest.cpp`
// proves its codebook by re-solving Lloyd-Max rather than trusting the table.
//
// It is optional at CONFIGURE time, not at run time: when the vendored tree is absent the oracle arm prints
// "NOT RUN" and the test exits 77 (ctest SKIPPED), because a check that quietly passes by finding nothing to
// compare is the failure mode this project keeps re-learning.
#if defined(__has_include)
#  if __has_include("krk/turbo_kv.hpp")
#    include "krk/turbo_kv.hpp"
#    define STRATA_HAVE_KRK_ORACLE 1
#  endif
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;

void ck(cudaError_t e, const char* w) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", w, cudaGetErrorString(e));
        std::exit(2);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 64), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 64), "memset");
    return p;
}

// The KV append entry points take DEVICE pointers. On this stack a host pointer is not rejected - the GPU
// dereferences it through the PCIe window, reads whatever the host has there, and returns plausible garbage -
// so the mistake has to be caught by the caller. This test exists to catch codec bugs, and a harness that
// silently quantises the wrong bytes produces false ones.
void need_device_ptr(const void* p, const char* what) {
    if (p == nullptr) return;   // a null plane is legal (single-plane append)
    cudaPointerAttributes a{};
    const cudaError_t e = cudaPointerGetAttributes(&a, p);
    if (e != cudaSuccess || a.type != hipMemoryTypeDevice) {
        std::fprintf(stderr,
                     "FAIL %s: %p is not device memory (attrs error '%s', type %d). The append kernels\n"
                     "      dereference this pointer on the GPU; a host pointer here reads host RAM and returns\n"
                     "      wrong bytes instead of failing. Copy it to a device buffer first.\n",
                     what, p, cudaGetErrorString(e), (int) a.type);
        std::exit(2);
    }
}

// The same Sylvester FWHT-256 the q4_0 parity test uses, so the rotation under test is the engine's.
void fwht256_host(float* x) {
    constexpr int n = 256;
    for (int s = 1; s < n; s *= 2)
        for (int i = 0; i < n; i += 2 * s)
            for (int j = 0; j < s; ++j) {
                const float u = x[i + j], v = x[i + j + s];
                x[i + j] = u + v;
                x[i + j + s] = u - v;
            }
    for (int i = 0; i < n; ++i) x[i] *= 1.0f / 16.0f;
}

int nearest(float t) {
    int idx = 0;
    while (idx < k::kTurboLevels - 1 && t > k::kTurboBoundary[idx]) ++idx;
    return idx;
}

// Host reference for one turbo3 row: ONE f32 sigma at offset 0, then 8 codes a group in 3 bytes at 4 + 3g.
// (Kraken-x puts the scale last; this port puts it first so the f32 is 4-aligned, and kv_turbo.hpp documents
// that. The kernel writes it first - this reference must agree with the kernel, not with Kraken-x.)
void quantize_turbo_host(const float* x, int d, std::vector<uint8_t>& row) {
    row.assign((size_t) (d / 8) * 3 + 4, 0);
    double n2 = 0.0;
    for (int i = 0; i < d; ++i) n2 += (double) x[i] * x[i];
    const float sigma = (float) (std::sqrt(n2) / std::sqrt((double) d));
    const float inv = sigma > 0.0f ? 1.0f / sigma : 0.0f;
    std::memcpy(row.data(), &sigma, 4);
    for (int g = 0; g < d / 8; ++g) {
        uint32_t acc = 0;
        for (int j = 0; j < 8; ++j) acc |= (uint32_t) nearest(x[g * 8 + j] * inv) << (3 * j);
        const size_t off = 4 + (size_t) 3 * g;
        row[off + 0] = (uint8_t) (acc & 0xFFu);
        row[off + 1] = (uint8_t) ((acc >> 8) & 0xFFu);
        row[off + 2] = (uint8_t) ((acc >> 16) & 0xFFu);
    }
}

// Host reference for one rot4 row: one f16 scale per 32-value block (the block RMS), four groups of eight codes.
void quantize_rot4_host(const float* x, int d, std::vector<uint8_t>& row) {
    row.assign((size_t) (d / 32) * 14, 0);
    for (int b = 0; b < d / 32; ++b) {
        double m2 = 0.0;
        for (int i = 0; i < 32; ++i) m2 += (double) x[b * 32 + i] * x[b * 32 + i];
        const float s = (float) std::sqrt(m2 / 32.0);
        const float inv = s > 0.0f ? 1.0f / s : 0.0f;
        const uint16_t sh = k::f16_from_f32(s);
        std::memcpy(row.data() + (size_t) b * 14, &sh, 2);
        for (int g = 0; g < 4; ++g) {
            uint32_t acc = 0;
            for (int j = 0; j < 8; ++j) acc |= (uint32_t) nearest(x[b * 32 + g * 8 + j] * inv) << (3 * j);
            const size_t off = (size_t) b * 14 + 2 + 3 * g;
            row[off + 0] = (uint8_t) (acc & 0xFFu);
            row[off + 1] = (uint8_t) ((acc >> 8) & 0xFFu);
            row[off + 2] = (uint8_t) ((acc >> 16) & 0xFFu);
        }
    }
}

// Dequantise one row from its bytes - the reference the GATHER must reproduce BITWISE.  The gather stores
// fp16, so every value is rounded through fp16 here too: comparing fp16 output against an unrounded float
// reports a mismatch on almost every element for no reason.
void dequant_turbo_host(const std::vector<uint8_t>& row, int d, float* y) {
    float sigma = 0.0f;
    std::memcpy(&sigma, row.data(), 4);
    for (int g = 0; g < d / 8; ++g) {
        const size_t off = 4 + (size_t) 3 * g;
        const uint32_t acc = (uint32_t) row[off] | ((uint32_t) row[off + 1] << 8) | ((uint32_t) row[off + 2] << 16);
        for (int j = 0; j < 8; ++j)
            y[g * 8 + j] = k::f32_from_f16(k::f16_from_f32(sigma * k::kTurboCentroid[(acc >> (3 * j)) & 7u]));
    }
}

void dequant_rot4_host(const std::vector<uint8_t>& row, int d, float* y) {
    for (int b = 0; b < d / 32; ++b) {
        uint16_t sh = 0;
        std::memcpy(&sh, row.data() + (size_t) b * 14, 2);
        const float s = k::f32_from_f16(sh);
        for (int g = 0; g < 4; ++g) {
            const size_t off = (size_t) b * 14 + 2 + 3 * g;
            const uint32_t acc = (uint32_t) row[off] | ((uint32_t) row[off + 1] << 8) | ((uint32_t) row[off + 2] << 16);
            for (int j = 0; j < 8; ++j)
                y[b * 32 + g * 8 + j] =
                    k::f32_from_f16(k::f16_from_f32(s * k::kTurboCentroid[(acc >> (3 * j)) & 7u]));
        }
    }
}

// -------------------------------------------------------------------------------------------------------------
// THE ORACLE: Kraken-x's own codec, on the same vectors, compared byte for byte against the GPU kernel.
//
// SCOPE, stated exactly. The rotation is a DELIBERATE divergence (an unseeded FWHT256 here, a seeded per-(layer,
// head) one there), so the oracle is fed the SAME rotated vector the kernel sees: the test rotates `x` with
// Kraken's own turbo_rotate() and hands that vector to the GPU, while pk_quant_row(x, seed) rotates internally
// to those same bytes. What is compared is then every decision AFTER the rotation - the norm, sigma, the
// standardisation and every 3-bit code. The transform itself is compared separately (the Hadamard arm), so the
// two checks together cover the whole encode.
//
// WHY THIS IS NOT A FUZZY TOLERANCE:
//   * every CODE byte must be identical. A code is the quantizer's decision; one flipped code is a bug, and
//     codes are what attention actually reads back.
//   * the stored SCALE may legitimately differ in its last bits: the kernel reduces ||x||^2 with an f32 warp
//     shuffle tree, while both references accumulate in f64. Rather than hide that behind a tolerance, the
//     margin is measured - `margin_min` is the closest any coordinate came to a decision boundary, `need` is
//     the largest distance a scale difference of the observed size CAN move it. The arm fails unless the margin
//     exceeds the movement by 4x. That is an argument from the numbers, not a hope the sample was lucky.
// -------------------------------------------------------------------------------------------------------------
#ifdef STRATA_HAVE_KRK_ORACLE

// Continuous k-means on N(0,1), ported from Kraken-x tools/turbo_kv_selftest.cpp. This is the paper's Eq. (4)
// solved numerically, so the shipped table is CHECKED against the fixed point instead of trusted.
static void solve_lloyd(double* c, int n) {
    const int S = 200000;
    std::vector<double> xs(S);
    for (int s = 0; s < S; ++s) {
        const double u = (s + 0.5) / S;
        double lo = -6, hi = 6;
        for (int k = 0; k < 48; ++k) {
            const double mid = 0.5 * (lo + hi);
            if (0.5 * (1.0 + std::erf(mid / std::sqrt(2.0))) < u) lo = mid;
            else hi = mid;
        }
        xs[s] = 0.5 * (lo + hi);
    }
    for (int i = 0; i < n; ++i) c[i] = -3.0 + 6.0 * (i + 0.5) / n;
    for (int it = 0; it < 200; ++it) {
        double sum[8] = {0}, wsum[8] = {0};
        for (int s = 0; s < S; ++s) {
            const double x = xs[s];
            int best = 0;
            double bd = 1e30;
            for (int i = 0; i < n; ++i) {
                const double d = (x - c[i]) * (x - c[i]);
                if (d < bd) { bd = d; best = i; }
            }
            sum[best] += x;
            wsum[best] += 1.0;
        }
        for (int i = 0; i < n; ++i) c[i] = wsum[i] > 0 ? sum[i] / wsum[i] : c[i];
    }
}

// A monotone order on f32 bit patterns, so an ULP count is honest across zero (raw integer differences jump).
inline long long ulp_order(uint32_t u) {
    return (u & 0x80000000u) ? -(long long) (u & 0x7FFFFFFFu) : (long long) u;
}
inline long long ulp_diff_f32(float a, float b) {
    uint32_t x, y;
    std::memcpy(&x, &a, 4);
    std::memcpy(&y, &b, 4);
    return (long long) std::llabs(ulp_order(x) - ulp_order(y));
}

struct Oracle {
    long long rows = 0;
    long long coords = 0;      // quantized values compared
    long long code_diff = 0;   // MUST be zero
    long long scale_diff = 0;  // may be non-zero: f32 vs f64 accumulation
    double scale_ulp_max = 0.0;
    double scale_rel_max = 0.0;
    double margin_min = 1e30;  // closest approach to a decision boundary
    double t_abs_max = 0.0;    // largest standardised coordinate
};

// One row: the oracle's bytes vs the kernel's, plus the boundary margin for that row.
template <bool ROT4>
void oracle_row(const uint8_t* got, const uint8_t* ref, const float* y, int D, Oracle& o) {
    ++o.rows;
    o.coords += D;
    if (ROT4) {
        for (int b = 0; b < D / 32; ++b) {
            const size_t at = (size_t) b * 14;
            uint16_t ra = 0, rb = 0;
            std::memcpy(&ra, ref + at, 2);
            std::memcpy(&rb, got + at, 2);
            if (ra != rb) {
                ++o.scale_diff;
                const float fa = k::f32_from_f16(ra), fb = k::f32_from_f16(rb);
                o.scale_ulp_max = std::max(o.scale_ulp_max, (double) ulp_diff_f32(fa, fb));
                const double rel = fa != 0.0f ? std::fabs((double) fb - (double) fa) / std::fabs((double) fa) : 1.0;
                o.scale_rel_max = std::max(o.scale_rel_max, rel);
            }
            for (int i = 2; i < 14; ++i)
                if (got[at + i] != ref[at + i]) ++o.code_diff;
            const float s = k::f32_from_f16(ra);
            const float inv = s > 0.0f ? 1.0f / s : 0.0f;
            for (int j = 0; j < 32; ++j) {
                const float t = y[b * 32 + j] * inv;
                float m = 1e30f;
                for (int q = 0; q < k::kTurboLevels - 1; ++q) m = std::min(m, std::fabs(t - k::kTurboBoundary[q]));
                o.margin_min = std::min(o.margin_min, (double) m);
                o.t_abs_max = std::max(o.t_abs_max, (double) std::fabs(t));
            }
        }
        return;
    }
    // turbo3: the reference writes the codes first and sigma LAST; this port writes sigma first, 4-aligned.
    const size_t nbytes = (size_t) (D / 8) * 3;
    for (size_t i = 0; i < nbytes; ++i)
        if (got[4 + i] != ref[i]) ++o.code_diff;
    float sa = 0.0f, sb = 0.0f;
    std::memcpy(&sa, ref + nbytes, 4);
    std::memcpy(&sb, got, 4);
    if (std::memcmp(&sa, &sb, 4) != 0) {
        ++o.scale_diff;
        o.scale_ulp_max = std::max(o.scale_ulp_max, (double) ulp_diff_f32(sa, sb));
        const double rel = sa != 0.0f ? std::fabs((double) sb - (double) sa) / std::fabs((double) sa) : 1.0;
        o.scale_rel_max = std::max(o.scale_rel_max, rel);
    }
    const float inv = sa > 0.0f ? 1.0f / sa : 0.0f;
    for (int j = 0; j < D; ++j) {
        const float t = y[j] * inv;
        float m = 1e30f;
        for (int q = 0; q < k::kTurboLevels - 1; ++q) m = std::min(m, std::fabs(t - k::kTurboBoundary[q]));
        o.margin_min = std::min(o.margin_min, (double) m);
        o.t_abs_max = std::max(o.t_abs_max, (double) std::fabs(t));
    }
}

template <bool ROT4>
Oracle run_oracle(const k::QsaShapes& s, int pages, uint32_t seed) {
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size;
    const size_t row_bytes = ROT4 ? k::kv_rot4_bytes_per_head(D) : k::kv_turbo_bytes_per_head(D);
    const long long cells = (long long) pages * P;
    std::mt19937 rng(seed ^ 0x9E3779B9u);
    std::normal_distribution<float> nd(0.0f, 3.0f);   // ~N(0,3): real K/V rows are not unit vectors

    const krk::KvPack pack = ROT4 ? krk::KvPack::Rot4 : krk::KvPack::Turbo3;

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = i;
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * sizeof(int32_t), cudaMemcpyHostToDevice), "table H2D");

    const size_t pool_bytes = (size_t) cells * H * row_bytes;
    uint8_t* d_k = dalloc<uint8_t>(pool_bytes);
    uint8_t* d_v = dalloc<uint8_t>(pool_bytes);

    // x = what the reference is given (it rotates internally); y = the same vector after Kraken's rotation,
    // which is what the kernel is given. Both sides therefore quantise y.
    std::vector<float> Kx((size_t) cells * H * D), Ky((size_t) cells * H * D);
    std::vector<float> Vx((size_t) cells * H * D), Vy((size_t) cells * H * D);
    std::vector<uint8_t> want_k((size_t) cells * H * row_bytes), want_v((size_t) cells * H * row_bytes);
    std::vector<float> rot((size_t) D), xrow((size_t) D), yrow((size_t) D);

    for (long long c = 0; c < cells; ++c) {
        for (int h = 0; h < H; ++h) {
            const size_t base = ((size_t) c * H + h) * D;
            for (int j = 0; j < D; ++j) xrow[j] = nd(rng);
            krk::turbo_rotate(xrow.data(), yrow.data(), D, seed);
            std::memcpy(&Kx[base], xrow.data(), sizeof(float) * D);
            std::memcpy(&Ky[base], yrow.data(), sizeof(float) * D);
            for (int j = 0; j < D; ++j) xrow[j] = nd(rng);
            krk::turbo_rotate(xrow.data(), yrow.data(), D, seed);
            std::memcpy(&Vx[base], xrow.data(), sizeof(float) * D);
            std::memcpy(&Vy[base], yrow.data(), sizeof(float) * D);

            const size_t wi = ((size_t) c * H + h) * row_bytes;
            if (ROT4) {
                krk::rot4_quant_row(&Kx[base], D, seed, want_k.data() + wi, rot.data());
                krk::rot4_quant_row(&Vx[base], D, seed, want_v.data() + wi, rot.data());
            } else {
                krk::turbo_quant_row(&Kx[base], D, seed, want_k.data() + wi, rot.data());
                krk::turbo_quant_row(&Vx[base], D, seed, want_v.data() + wi, rot.data());
            }
        }
    }

    // The prompt path, so the batch launch is what is gated (the engine prefills through it).
    //
    // THE K/V INPUT MUST BE A DEVICE COPY. An earlier revision of this file passed Ky.data() - host memory - to
    // the batch entry point, and on this stack that did NOT fault: the GPU read host RAM through the PCIe window,
    // quantised whatever happened to be there, and produced a convincing "99% of the batch output is wrong".
    // The only thing that was wrong was the harness. need_device_ptr() now refuses that call outright.
    float* d_ky = dalloc<float>((size_t) cells * H * D);
    float* d_vy = dalloc<float>((size_t) cells * H * D);
    need_device_ptr(d_ky, "oracle K");
    need_device_ptr(d_vy, "oracle V");
    ck(cudaMemcpy(d_ky, Ky.data(), sizeof(float) * (size_t) cells * H * D, cudaMemcpyHostToDevice), "Ky H2D");
    ck(cudaMemcpy(d_vy, Vy.data(), sizeof(float) * (size_t) cells * H * D, cudaMemcpyHostToDevice), "Vy H2D");
    if (ROT4) k::kv_append_rot4(d_k, d_v, d_table, 0, cells, d_ky, d_vy, s, nullptr);
    else      k::kv_append_turbo(d_k, d_v, d_table, 0, cells, d_ky, d_vy, s, nullptr);
    ck(cudaDeviceSynchronize(), "oracle append");

    std::vector<uint8_t> hk_batch(pool_bytes), hv_batch(pool_bytes);
    ck(cudaMemcpy(hk_batch.data(), d_k, pool_bytes, cudaMemcpyDeviceToHost), "oracle batch K D2H");
    ck(cudaMemcpy(hv_batch.data(), d_v, pool_bytes, cudaMemcpyDeviceToHost), "oracle batch V D2H");

    // The same cells through the PER-STEP path, which arm [4/6] already proves consistent. Appending both ways
    // separates "the codec reads the wrong vector" from "the batch launch writes the wrong row", two failures
    // that look identical from the pool bytes alone.
    ck(cudaMemset(d_k, 0, pool_bytes), "reset k");
    ck(cudaMemset(d_v, 0, pool_bytes), "reset v");
    {
        float* d_kcur = dalloc<float>((size_t) H * D);
        float* d_vcur = dalloc<float>((size_t) H * D);
        int32_t* d_step = dalloc<int32_t>(k::kStepCount);
        for (long long c = 0; c < cells; ++c) {
            int32_t hs[k::kStepCount] = {(int32_t) c, (int32_t) c + 1, 0, 0};
            ck(cudaMemcpy(d_step, hs, sizeof hs, cudaMemcpyHostToDevice), "step");
            ck(cudaMemcpy(d_kcur, &Ky[(size_t) c * H * D], sizeof(float) * H * D, cudaMemcpyHostToDevice), "k H2D");
            ck(cudaMemcpy(d_vcur, &Vy[(size_t) c * H * D], sizeof(float) * H * D, cudaMemcpyHostToDevice), "v H2D");
            if (ROT4) k::kv_append_rot4_step(d_k, d_v, d_table, d_step, d_kcur, d_vcur, s, nullptr);
            else      k::kv_append_turbo_step(d_k, d_v, d_table, d_step, d_kcur, d_vcur, s, nullptr);
            ck(cudaDeviceSynchronize(), "append step");
        }
        cudaFree(d_kcur); cudaFree(d_vcur); cudaFree(d_step);
    }

    std::vector<uint8_t> hk(pool_bytes), hv(pool_bytes);
    ck(cudaMemcpy(hk.data(), d_k, pool_bytes, cudaMemcpyDeviceToHost), "oracle k D2H");
    ck(cudaMemcpy(hv.data(), d_v, pool_bytes, cudaMemcpyDeviceToHost), "oracle v D2H");

    // The two paths must agree BYTE FOR BYTE. The same cell quantised at prefill time and at decode time landing
    // on different bytes would mean attention reads one codec for prompt positions and another for generated
    // ones, which no distortion bound would catch. (Within one path, sigma still differs from the reference in
    // the last bit; across the two paths of the SAME kernel it must not differ at all.)
    long long path_diff = 0, first_bad = -1;
    for (size_t i = 0; i < pool_bytes; ++i)
        if (hk_batch[i] != hk[i] || hv_batch[i] != hv[i]) {
            ++path_diff;
            if (first_bad < 0) first_bad = (long long) i;
        }
    std::printf("  prompt (batch) vs decode (per-step) append: %lld of %zu bytes differ -> %s\n", path_diff,
                2 * pool_bytes, path_diff == 0 ? "identical" : "MISMATCH");
    if (path_diff != 0) {
        std::fprintf(stderr, "  FAIL: the batch and per-step appends disagree; first at byte %lld\n", first_bad);
        ++g_fail;
        // Name the rows so the failure is diagnosable without a debugger: the leading scale, three ways.
        const size_t nbytes = ROT4 ? 0 : (size_t) (D / 8) * 3;
        auto show = [&](const uint8_t* base, bool from_ref) {
            float v = 0.0f;
            if (ROT4) { uint16_t u = 0; std::memcpy(&u, base, 2); v = k::f32_from_f16(u); }
            else { std::memcpy(&v, base + (from_ref ? nbytes : 0), 4); }
            return v;
        };
        int shown = 0;
        for (long long c = 0; c < cells && shown < 8; ++c) {
            for (int h = 0; h < H && shown < 8; ++h) {
                const long long row = (((long long) (c / P)) * H + h) * P + (c % P);
                if (std::memcmp(hk_batch.data() + (size_t) row * row_bytes, hk.data() + (size_t) row * row_bytes,
                                row_bytes) == 0) continue;
                const size_t wi = ((size_t) c * H + h) * row_bytes;
                std::printf("    c=%lld h=%d row=%lld  batch %10.6f  step %10.6f  ref %10.6f\n", c, h, row,
                            show(&hk_batch[(size_t) row * row_bytes], false), show(&hk[(size_t) row * row_bytes], false),
                            show(&want_k[wi], true));
                ++shown;
            }
        }
    }

    Oracle o;
    for (long long c = 0; c < cells; ++c) {
        for (int h = 0; h < H; ++h) {
            const size_t wi = ((size_t) c * H + h) * row_bytes;
            const long long row = (((long long) (c / P)) * H + h) * P + (c % P);
            const size_t base = ((size_t) c * H + h) * D;
            oracle_row<ROT4>(hk.data() + (size_t) row * row_bytes, want_k.data() + wi, &Ky[base], D, o);
            oracle_row<ROT4>(hv.data() + (size_t) row * row_bytes, want_v.data() + wi, &Vy[base], D, o);
        }
    }

    cudaFree(d_table); cudaFree(d_k); cudaFree(d_v); cudaFree(d_ky); cudaFree(d_vy);
    return o;
}

void report_oracle(const char* name, const Oracle& o) {
    const double need = o.t_abs_max * o.scale_rel_max;   // how far the scale difference CAN move a coordinate
    const double ratio = need > 0.0 ? o.margin_min / need : 1e30;
    std::printf("  %-7s %lld rows / %lld coords\n", name, o.rows, o.coords);
    std::printf("          codes  %lld differ (must be 0)  -> %s\n", o.code_diff,
                o.code_diff == 0 ? "IDENTICAL to the Kraken-x reference" : "MISMATCH");
    std::printf("          scale  %lld bytes differ, max %.0f ulp (rel %.1e)\n", o.scale_diff, o.scale_ulp_max,
                o.scale_rel_max);
    std::printf("          margin closest boundary approach %.3e vs %.3e the scale can move it  -> %.0fx\n",
                o.margin_min, need, ratio);
    if (o.code_diff != 0) {
        std::fprintf(stderr, "  FAIL %s: %lld code bytes differ from the Kraken-x reference\n", name, o.code_diff);
        ++g_fail;
    }
    if (o.scale_ulp_max > 4.0) {
        std::fprintf(stderr, "  FAIL %s: the stored scale differs by %.0f ulp, more than f32-vs-f64 accumulation\n",
                     name, o.scale_ulp_max);
        ++g_fail;
    }
    if (ratio <= 4.0) {
        std::fprintf(stderr, "  FAIL %s: the boundary margin (%.3e) is too close to what the scale can move (%.3e)\n",
                     name, o.margin_min, need);
        ++g_fail;
    }
}

#endif  // STRATA_HAVE_KRK_ORACLE

// One codec end to end through a paged pool: append every cell, gather them back, compare.
struct Result {
    double sig_rel = 0.0;      // worst relative difference in the stored scale
    long long gather_bad = 0;  // gathered fp16 vs host dequant of the same bytes
    double max_abs_err = 0.0;  // reconstruction vs the rotated input
    double rms_err = 0.0;
    double dot_rel_mean = 0.0; // <q, k_hat> vs <q, k>
};

template <bool ROT4>
Result run_codec(const k::QsaShapes& s, int pages, uint32_t seed) {
    const int H = (int) s.n_head_kv, D = (int) s.head_dim, P = (int) s.page_size;
    const size_t row_bytes = ROT4 ? k::kv_rot4_bytes_per_head(D) : k::kv_turbo_bytes_per_head(D);
    const long long cells = (long long) pages * P;
    std::mt19937 rng(seed);
    std::normal_distribution<float> nd(0.0f, 1.0f);

    std::vector<int32_t> table(pages);
    for (int i = 0; i < pages; ++i) table[i] = i;
    std::shuffle(table.begin(), table.end(), rng);
    int32_t* d_table = dalloc<int32_t>(pages);
    ck(cudaMemcpy(d_table, table.data(), pages * sizeof(int32_t), cudaMemcpyHostToDevice), "table H2D");

    const size_t pool_bytes = (size_t) cells * H * row_bytes;
    uint8_t* d_k = dalloc<uint8_t>(pool_bytes);
    uint8_t* d_v = dalloc<uint8_t>(pool_bytes);
    float* d_kcur = dalloc<float>(H * D);
    float* d_vcur = dalloc<float>(H * D);
    int32_t* d_step = dalloc<int32_t>(k::kStepCount);

    std::vector<std::vector<float>> in_k(cells), in_v(cells);
    for (long long c = 0; c < cells; ++c) {
        in_k[c].resize(H * D);
        in_v[c].resize(H * D);
        for (auto& x : in_k[c]) x = nd(rng) * 2.0f;
        for (auto& x : in_v[c]) x = nd(rng) * 2.0f;
        for (int h = 0; h < H; ++h) {   // the engine rotates before the append; so does the test
            fwht256_host(in_k[c].data() + h * D);
            fwht256_host(in_v[c].data() + h * D);
        }
    }

    for (long long c = 0; c < cells; ++c) {
        int32_t hs[k::kStepCount] = {(int32_t) c, (int32_t) c + 1, 0, 0};
        ck(cudaMemcpy(d_step, hs, sizeof hs, cudaMemcpyHostToDevice), "step");
        ck(cudaMemcpy(d_kcur, in_k[c].data(), in_k[c].size() * 4, cudaMemcpyHostToDevice), "k H2D");
        ck(cudaMemcpy(d_vcur, in_v[c].data(), in_v[c].size() * 4, cudaMemcpyHostToDevice), "v H2D");
        if (ROT4) k::kv_append_rot4_step(d_k, d_v, d_table, d_step, d_kcur, d_vcur, s, nullptr);
        else      k::kv_append_turbo_step(d_k, d_v, d_table, d_step, d_kcur, d_vcur, s, nullptr);
        ck(cudaDeviceSynchronize(), "append");
    }

    std::vector<uint8_t> hk(pool_bytes), hv(pool_bytes);
    ck(cudaMemcpy(hk.data(), d_k, pool_bytes, cudaMemcpyDeviceToHost), "k D2H");
    ck(cudaMemcpy(hv.data(), d_v, pool_bytes, cudaMemcpyDeviceToHost), "v D2H");

    // The scale actually stored, against the host's own statistic for the same row.
    Result r;
    for (long long c = 0; c < cells; ++c) {
        for (int h = 0; h < H; ++h) {
            const long long page = table[c / P];
            const long long row = (page * H + h) * P + (c % P);
            std::vector<uint8_t> want;
            if (ROT4) quantize_rot4_host(in_k[c].data() + h * D, D, want);
            else      quantize_turbo_host(in_k[c].data() + h * D, D, want);
            const uint8_t* got = hk.data() + row * row_bytes;
            // turbo3: compare the f32 sigma; rot4: every block's f16 scale.
            if (ROT4) {
                for (int b = 0; b < D / 32; ++b) {
                    uint16_t a = 0, b2 = 0;
                    std::memcpy(&a, want.data() + (size_t) b * 14, 2);
                    std::memcpy(&b2, got + (size_t) b * 14, 2);
                    const double wa = k::f32_from_f16(a), gb = k::f32_from_f16(b2);
                    const double rel = wa != 0.0 ? std::fabs(gb - wa) / std::fabs(wa) : std::fabs(gb);
                    r.sig_rel = std::max(r.sig_rel, rel);
                }
            } else {
                float a = 0.0f, b2 = 0.0f;
                std::memcpy(&a, want.data(), 4);
                std::memcpy(&b2, got, 4);
                const double rel = a != 0.0f ? std::fabs(b2 - a) / std::fabs(a) : std::fabs(b2);
                r.sig_rel = std::max(r.sig_rel, rel);
            }
        }
    }

    // Gather, and compare with the host's dequantisation of the SAME stored bytes.
    const int max_ids = (int) std::min<long long>(cells, 64);
    int32_t* d_ids = dalloc<int32_t>(max_ids);
    uint16_t* d_ks = dalloc<uint16_t>((size_t) max_ids * H * D);
    uint16_t* d_vs = dalloc<uint16_t>((size_t) max_ids * H * D);
    std::vector<int32_t> ids(max_ids);
    for (int i = 0; i < max_ids; ++i) ids[i] = (int32_t) (rng() % (unsigned) cells);
    ck(cudaMemcpy(d_ids, ids.data(), max_ids * 4, cudaMemcpyHostToDevice), "ids");
    int32_t hs[k::kStepCount] = {0, 0, 0, max_ids};
    ck(cudaMemcpy(d_step, hs, sizeof hs, cudaMemcpyHostToDevice), "step");
    if (ROT4) k::kv_gather_rot4_step(d_k, d_v, d_table, d_ids, d_step, max_ids, s, d_ks, d_vs, nullptr);
    else      k::kv_gather_turbo_step(d_k, d_v, d_table, d_ids, d_step, max_ids, s, d_ks, d_vs, nullptr);
    ck(cudaDeviceSynchronize(), "gather");

    std::vector<uint16_t> hks((size_t) max_ids * H * D), hvs((size_t) max_ids * H * D);
    ck(cudaMemcpy(hks.data(), d_ks, hks.size() * 2, cudaMemcpyDeviceToHost), "ks D2H");
    ck(cudaMemcpy(hvs.data(), d_vs, hvs.size() * 2, cudaMemcpyDeviceToHost), "vs D2H");

    double sum_sq = 0.0;
    long long n_elem = 0;
    double dot_ref = 0.0, dot_err = 0.0;
    std::vector<float> ref((size_t) D);
    for (int i = 0; i < max_ids; ++i) {
        const long long c = ids[i];
        for (int h = 0; h < H; ++h) {
            const long long page = table[c / P];
            const long long row = (page * H + h) * P + (c % P);
            for (int is_v = 0; is_v < 2; ++is_v) {
                const uint8_t* src = (is_v ? hv.data() : hk.data()) + row * row_bytes;
                if (ROT4) dequant_rot4_host(std::vector<uint8_t>(src, src + row_bytes), D, ref.data());
                else      dequant_turbo_host(std::vector<uint8_t>(src, src + row_bytes), D, ref.data());
                const uint16_t* got = (is_v ? hvs.data() : hks.data()) + ((size_t) i * H + h) * D;
                const float* orig = (is_v ? in_v[c].data() : in_k[c].data()) + h * D;
                for (int d = 0; d < D; ++d) {
                    const double g = k::f32_from_f16(got[d]);
                    if (g != (double) ref[d]) ++r.gather_bad;
                    const double e = std::fabs(g - (double) orig[d]);
                    r.max_abs_err = std::max(r.max_abs_err, e);
                    sum_sq += e * e;
                    ++n_elem;
                }
                // A dot against a fixed query: the quantity attention actually computes.
                double dr = 0.0, dh = 0.0;
                for (int d = 0; d < D; ++d) {
                    const double qv = (double) in_k[(c + 1) % cells][h * D + d];   // any rotated query
                    dr += qv * (double) orig[d];
                    dh += qv * (double) k::f32_from_f16(got[d]);
                }
                dot_ref += std::fabs(dr);
                dot_err += std::fabs(dh - dr);
            }
        }
    }
    r.rms_err = std::sqrt(sum_sq / (double) n_elem);
    r.dot_rel_mean = dot_ref > 0.0 ? dot_err / dot_ref : 0.0;

    cudaFree(d_table); cudaFree(d_k); cudaFree(d_v); cudaFree(d_kcur); cudaFree(d_vcur);
    cudaFree(d_step); cudaFree(d_ids); cudaFree(d_ks); cudaFree(d_vs);
    return r;
}

void report(const char* name, const Result& r, double bytes_per_head, double rms_bound) {
    std::printf("  %-7s %.0f B/head  scale rel %.2e  gather mismatches %lld  max |e| %.4f  rms %.4f  "
                "dot rel %.4f\n", name, bytes_per_head, r.sig_rel, r.gather_bad, r.max_abs_err, r.rms_err,
                r.dot_rel_mean);
    if (r.sig_rel > 1e-3) { std::fprintf(stderr, "  FAIL %s: the stored scale differs from the host statistic\n", name); ++g_fail; }
    if (r.gather_bad != 0) { std::fprintf(stderr, "  FAIL %s: %lld gathered values differ from the pool bytes\n", name, r.gather_bad); ++g_fail; }
    if (r.rms_err > rms_bound) { std::fprintf(stderr, "  FAIL %s: rms error %.4f above the bound %.4f\n", name, r.rms_err, rms_bound); ++g_fail; }
}

}  // namespace

int main() {
    std::printf("=== kv_turbo_parity: Turbo3 and Rot4 KV codecs ===\n");
    k::QsaShapes s = k::qsa_real_shapes();
    if (s.head_dim != 256) {
        std::fprintf(stderr, "FAIL: head_dim %lld, the codecs and the Hadamard transform are defined at 256\n",
                     (long long) s.head_dim);
        return 1;
    }
    const int D = (int) s.head_dim;

    std::printf("[1/6] byte counts (pools are sized from these)\n");
    const double t = (double) k::kv_turbo_bytes_per_head(D);
    const double r4 = (double) k::kv_rot4_bytes_per_head(D);
    const double q4 = 144.0, i8 = (double) D + (double) (D / 64) * 2.0, f16 = (double) D * 2.0;
    std::printf("  f16 %.0f  int8 %.0f  q4_0 %.0f  rot4 %.0f  turbo3 %.0f  (bytes per head)\n", f16, i8, q4, r4, t);
    if (!(t < r4 && r4 < q4 && q4 < i8 && i8 < f16)) {
        std::fprintf(stderr, "FAIL: the formats are not ordered as documented (turbo3 < rot4 < q4_0 < int8 < f16)\n");
        ++g_fail;
    } else {
        std::printf("  -> ordered as documented; turbo3 is %.2fx int8 and %.2fx q4_0\n", i8 / t, q4 / t);
    }

#ifdef STRATA_HAVE_KRK_ORACLE
    std::printf("[2/6] the codebook IS the Lloyd-Max fixed point (re-solved, not trusted)\n");
    {
        double c[8];
        solve_lloyd(c, 8);
        double worst = 0.0;
        for (int i = 0; i < 8; ++i) worst = std::max(worst, std::fabs(c[i] - (double) k::kTurboCentroid[i]));
        std::printf("  max |solved - shipped| %.2e over 8 levels (tolerance 2e-3)\n", worst);
        if (!(worst < 2e-3)) {
            std::fprintf(stderr, "FAIL: the shipped centroid table is not the N(0,1) 3-bit Lloyd-Max optimum\n");
            ++g_fail;
        }
    }

    std::printf("[3/6] the rotation's Hadamard equals Kraken-x's, bit for bit at d=%d\n", D);
    {
        std::mt19937 rng(99);
        std::normal_distribution<float> nd(0.0f, 2.0f);
        std::vector<float> a((size_t) D), b((size_t) D);
        long long bad = 0;
        for (int trial = 0; trial < 64; ++trial) {
            for (int i = 0; i < D; ++i) a[i] = nd(rng);
            b = a;
            fwht256_host(a.data());                    // the engine's transform (unseeded)
            krk::turbo_hadamard(b.data(), D);          // the reference's, without the sign stream
            for (int i = 0; i < D; ++i)
                if (std::memcmp(&a[i], &b[i], 4) != 0) ++bad;
        }
        std::printf("  %d values compared, %lld differ (only the +-1 sign stream is a deliberate divergence)\n",
                    D * 64, bad);
        if (bad != 0) {
            std::fprintf(stderr, "FAIL: the unseeded rotation differs from Kraken-x's turbo_hadamard\n");
            ++g_fail;
        }
    }
#else
    std::printf("[2/6] codebook + rotation vs Kraken-x: NOT RUN (third_party/Kraken-x/include/krk/turbo_kv.hpp absent)\n");
#endif

    std::printf("[4/6] turbo3: append, gather, consistency and distortion\n");
    const Result a = run_codec<false>(s, 8, 1234);
    // 3-bit Lloyd-Max on unit-variance coordinates: the paper's distortion is 0.03 in variance terms, i.e. an
    // rms near sqrt(0.03) * sigma ~ 0.17 sigma, and these vectors are ~2 sigma. 0.6 is a generous bound that a
    // layout bug (a swapped group, a wrong shift) cannot pass.
    report("turbo3", a, t, 0.6);

    std::printf("[5/6] rot4: append, gather, consistency and distortion\n");
    const Result b = run_codec<true>(s, 8, 5678);
    report("rot4", b, r4, 0.6);

    int skipped = 0;
#ifdef STRATA_HAVE_KRK_ORACLE
    // The oracle arm: different codebase, same vectors, byte for byte. This is the check that can catch a port
    // which is self-consistent and wrong, which is the only failure mode the arms above cannot see.
    std::printf("[6/6] ORACLE: Kraken-x krk::pk_quant_row vs the GPU kernel, same rotated vectors\n");
    report_oracle("turbo3", run_oracle<false>(s, 4, krk::turbo_seed(3, 1)));
    report_oracle("rot4", run_oracle<true>(s, 4, krk::turbo_seed(3, 1)));
#else
    std::printf("[6/6] ORACLE: NOT RUN - the vendored Kraken-x tree is absent, so there is nothing independent to\n"
                "      compare against. This test SKIPS rather than reporting a pass it did not earn.\n");
    skipped = 1;
#endif

    if (g_fail == 0 && skipped) {
        std::printf("=== kv_turbo_parity: SKIPPED the oracle arm (exports code 77) ===\n");
        return 77;
    }
    if (g_fail == 0) {
        std::printf("=== kv_turbo_parity: ALL TESTS PASSED ===\n");
        return 0;
    }
    std::fprintf(stderr, "=== kv_turbo_parity: FAILED (%d) ===\n", g_fail);
    return 1;
}
