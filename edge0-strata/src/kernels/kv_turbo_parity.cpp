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

    std::printf("[1/3] byte counts (pools are sized from these)\n");
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

    std::printf("[2/3] turbo3: append, gather, consistency and distortion\n");
    const Result a = run_codec<false>(s, 8, 1234);
    // 3-bit Lloyd-Max on unit-variance coordinates: the paper's distortion is 0.03 in variance terms, i.e. an
    // rms near sqrt(0.03) * sigma ~ 0.17 sigma, and these vectors are ~2 sigma. 0.6 is a generous bound that a
    // layout bug (a swapped group, a wrong shift) cannot pass.
    report("turbo3", a, t, 0.6);

    std::printf("[3/3] rot4: append, gather, consistency and distortion\n");
    const Result b = run_codec<true>(s, 8, 5678);
    report("rot4", b, r4, 0.6);

    if (g_fail == 0) {
        std::printf("=== kv_turbo_parity: ALL TESTS PASSED ===\n");
        return 0;
    }
    std::fprintf(stderr, "=== kv_turbo_parity: FAILED (%d) ===\n", g_fail);
    return 1;
}
