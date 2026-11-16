// src/kernels/cuda/latent_u_tile_test.cpp - the staged U tile against the per-row kernel.
//
// latent_u_tile_k changes HOW U's sum is evaluated (one scale per 16 values instead of two multiplies
// per 4, aligned shared operands, weights reused over 64 rows), so its output is numerically equal to
// latent_u_grouped's but NOT bit-identical.  This is the gate for that: same operands through both
// kernels, then the relative error and cosine of the whole [n, H] result.  It also checks that
// latent_u_runs - the weight-reuse variant, which keeps the per-row kernel's term order IS bit exact,
// because that is the claim its own comment makes.
//
// Shapes: ELL 128 (4 Q8_0 blocks a row, so the 64-value chunk path and its tail both run), H 512
// (two 256-column tiles), G 3 groups, 20 routed rows spread over them (7 / 7 / 6, so a group's last
// tile is partly padding - the case the engine hits with its ~25 entries a group).
#include "strata/kernels/latent.hpp"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <vector>

#define CK(e) do { const cudaError_t r_ = (e); if (r_ != cudaSuccess) { \
    std::printf("FAIL %s: %s\n", #e, cudaGetErrorString(r_)); std::exit(1); } } while (0)

namespace {

constexpr int kELL = 128;
constexpr int kH = 512;
constexpr int kG = 3;
constexpr int kRows = 20;
constexpr int kE = 3;                 // one expert per label group here

uint32_t g_seed = 12345u;
uint32_t rnd() { g_seed = g_seed * 1664525u + 1013904223u; return g_seed >> 8; }
float rnd_scale() { return 0.002f + (float) (rnd() % 1000) * 0.00001f; }

void store_half(uint8_t* p, float v) {
    const __half h = __float2half(v);
    std::memcpy(p, &h, 2);
}

}  // namespace

int main() {
    int dev = 0;
    if (cudaGetDevice(&dev) != cudaSuccess) { std::printf("SKIP: no device\n"); return 77; }
    cudaDeviceProp prop{};
    CK(cudaGetDeviceProperties(&prop, dev));

    const int nb = kELL / 32;                       // Q8_0/q8_1 blocks a row
    const size_t wrow = (size_t) nb * 34;           // {half d; int8 qs[32]}
    const size_t arow = (size_t) nb * 36;           // {half d; half s; int8 qs[32]}
    const size_t wbytes = (size_t) kG * kH * wrow;
    const size_t abytes = (size_t) kRows * arow;
    std::vector<uint8_t> w(wbytes), a(abytes);
    for (size_t i = 0; i < wbytes; i += 34) {        // one Q8_0 block
        store_half(&w[i], rnd_scale());
        for (int k = 0; k < 32; ++k) w[i + 2 + k] = (uint8_t) (int8_t) (int) (rnd() % 255) - 127;
    }
    for (size_t i = 0; i < abytes; i += 36) {        // one q8_1 block
        store_half(&a[i], rnd_scale());
        store_half(&a[i + 2], 0.0f);                 // the sum term: unused by both kernels
        for (int k = 0; k < 32; ++k) a[i + 4 + k] = (uint8_t) (int8_t) (int) (rnd() % 255) - 127;
    }
    std::vector<int32_t> ids(kRows), labels(kE);
    for (int r = 0; r < kRows; ++r) ids[r] = r % kE;
    for (int e = 0; e < kE; ++e) labels[e] = e;

    uint8_t *dw = nullptr, *da = nullptr;
    int32_t *dids = nullptr, *dlab = nullptr;
    float *dref = nullptr, *dtile = nullptr, *drun = nullptr;
    void* scratch = nullptr;
    CK(cudaMalloc(&dw, wbytes));
    CK(cudaMalloc(&da, abytes));
    CK(cudaMalloc(&dids, kRows * 4));
    CK(cudaMalloc(&dlab, kE * 4));
    CK(cudaMalloc(&dref, (size_t) kRows * kH * 4));
    CK(cudaMalloc(&dtile, (size_t) kRows * kH * 4));
    CK(cudaMalloc(&drun, (size_t) kRows * kH * 4));
    CK(cudaMalloc(&scratch, 1 << 20));
    CK(cudaMemcpy(dw, w.data(), wbytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(da, a.data(), abytes, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dids, ids.data(), kRows * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlab, labels.data(), kE * 4, cudaMemcpyHostToDevice));

    std::printf("staged U tile vs per-row: ELL %d H %d G %d rows %d (groups %d/%d/%d)\n", kELL, kH, kG, kRows,
                (kRows + 2) / 3, (kRows + 1) / 3, kRows / 3);

    // the reference: the shipped per-row kernel
    strata::kernels::latent_u_grouped(dw, da, dids, dlab, kE, kRows, kELL, kH, dref, nullptr, 8, kG);
    strata::kernels::latent_u_tiled(dw, da, dids, dlab, kE, kRows, kELL, kH, dtile, scratch, nullptr, 8, kG);
    strata::kernels::latent_u_runs(dw, da, dids, dlab, kE, kRows, kELL, kH, drun, scratch, nullptr, 8, 4, kG);
    CK(cudaDeviceSynchronize());

    std::vector<float> ref((size_t) kRows * kH), tile((size_t) kRows * kH), run((size_t) kRows * kH);
    CK(cudaMemcpy(ref.data(), dref, ref.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(tile.data(), dtile, tile.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(run.data(), drun, run.size() * 4, cudaMemcpyDeviceToHost));

    double max_rel = 0.0, sum_rel = 0.0, dot = 0.0, nr = 0.0, nt = 0.0, max_abs = 0.0;
    size_t bit_exact_run = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double d = std::fabs((double) tile[i] - (double) ref[i]);
        const double rel = d / (std::fabs((double) ref[i]) + 1e-6);
        if (rel > max_rel) max_rel = rel;
        sum_rel += rel;
        if (d > max_abs) max_abs = d;
        dot += (double) ref[i] * (double) tile[i];
        nr += (double) ref[i] * (double) ref[i];
        nt += (double) tile[i] * (double) tile[i];
        if (std::memcmp(&run[i], &ref[i], 4) == 0) ++bit_exact_run;
    }
    const double mean_rel = sum_rel / (double) ref.size();
    const double cos = dot / (std::sqrt(nr) * std::sqrt(nt) + 1e-30);
    std::printf("staged tile : mean_rel %.3e  max_rel %.3e  max|d| %.3e  cos %.9f\n", mean_rel, max_rel, max_abs,
                cos);
    std::printf("run  kernel : %zu of %zu values bit-identical to the per-row kernel\n", bit_exact_run, ref.size());

    bool ok = true;
    if (!(mean_rel < 1e-5)) { std::printf("FAIL: staged tile mean_rel %.3e >= 1e-5\n", mean_rel); ok = false; }
    if (!(cos > 0.9999999)) { std::printf("FAIL: staged tile cos %.9f\n", cos); ok = false; }
    if (bit_exact_run != ref.size()) {
        std::printf("FAIL: the run kernel is not bit-identical (%zu of %zu)\n", bit_exact_run, ref.size());
        ok = false;
    }

    // The D projection rides the same tile with rows == nullptr (one weight matrix, identity row order).
    // Its reference is the per-row kernel with every row in group 0, which is the same arithmetic.
    std::vector<int32_t> ids0(kRows, 0), lab0(1, 0);
    int32_t* dids0 = nullptr;
    int32_t* dlab0 = nullptr;
    CK(cudaMalloc(&dids0, kRows * 4));
    CK(cudaMalloc(&dlab0, 4));
    CK(cudaMemcpy(dids0, ids0.data(), kRows * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dlab0, lab0.data(), 4, cudaMemcpyHostToDevice));
    strata::kernels::latent_u_grouped(dw, da, dids0, dlab0, 1, kRows, kELL, kH, dref, nullptr, 8, 0);
    strata::kernels::latent_d_tiled(dw, da, kRows, kELL, kH, dtile, nullptr, 8);
    CK(cudaDeviceSynchronize());
    CK(cudaMemcpy(ref.data(), dref, ref.size() * 4, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(tile.data(), dtile, tile.size() * 4, cudaMemcpyDeviceToHost));
    double d_max_rel = 0.0, d_sum_rel = 0.0, d_dot = 0.0, d_nr = 0.0, d_nt = 0.0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double dv = std::fabs((double) tile[i] - (double) ref[i]);
        const double rel = dv / (std::fabs((double) ref[i]) + 1e-6);
        if (rel > d_max_rel) d_max_rel = rel;
        d_sum_rel += rel;
        d_dot += (double) ref[i] * (double) tile[i];
        d_nr += (double) ref[i] * (double) ref[i];
        d_nt += (double) tile[i] * (double) tile[i];
    }
    const double d_mean_rel = d_sum_rel / (double) ref.size();
    const double d_cos = d_dot / (std::sqrt(d_nr) * std::sqrt(d_nt) + 1e-30);
    std::printf("D (no groups): mean_rel %.3e  max_rel %.3e  cos %.9f\n", d_mean_rel, d_max_rel, d_cos);
    if (!(d_mean_rel < 1e-5) || !(d_cos > 0.9999999)) {
        std::printf("FAIL: the D tile (identity rows) is not numerically equal\n");
        ok = false;
    }
    std::printf("%s\n", ok ? "PASS latent_u_tile_test" : "FAIL latent_u_tile_test");
    return ok ? 0 : 1;
}
