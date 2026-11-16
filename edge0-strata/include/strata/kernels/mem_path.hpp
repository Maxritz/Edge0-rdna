// include/strata/kernels/mem_path.hpp - the device's memory path, measured at startup.
//
// WHY THIS EXISTS.  On 2026-11-16 this box spent hours with its GPU in a state where every read of
// its own VRAM ran at 22.6 GB/s instead of 361.5 GB/s (a stuck memory path after a TDR: the caches
// stayed fast at 587-841 GB/s and compute stayed at 15 TFLOP/s, so nothing looked wrong from the
// inside).  Every engine measurement taken in that state - 5.04 tok/s decode, 46.3 s prefill - was
// attributed to the code, and the code's own Q8_0 GEMV kernel was in fact running at 93% of the
// card's 384 GB/s spec.  A one-line, always-on check would have answered the question instantly.
//
// The check is self-calibrating, so it needs no spec table for any card:
//
//   cache_gbps : eight passes over a 64 MiB buffer  (Infinity Cache / L2 resident)
//   dram_gbps  : one pass over a 512 MiB buffer     (past every cache)
//   ratio      : dram_gbps / cache_gbps
//
// Healthy cards land at 0.3-0.6 (Navi 22 here: 361/841 = 0.43).  A device whose DRAM path has fallen
// back to PCIe speed lands an order of magnitude lower (22.6/587 = 0.04) while its caches stay fast,
// which is the signature that distinguishes a device state fault from a slow kernel.
//
// Cost: ~20 ms once, at startup.  `STRATA_MEM_PATH=0` skips it; there is no other switch, because a
// measurement you have to remember to ask for is a measurement nobody takes.
#pragma once

namespace strata::kernels {

struct MemPath {
    double dram_gbps = 0.0;    // one streaming pass over 512 MiB
    double cache_gbps = 0.0;   // eight passes over 64 MiB
    double ratio = 0.0;        // dram / cache
    int vram_mib = 0;
    bool ran = false;
};

// Measures the device's read path.  Returns false when it could not run (no device, allocation
// refused); out is still filled with whatever was measured.  Side-effect free: the scratch is freed
// before it returns, so a caller's VRAM budget is unaffected.
bool mem_path_probe(MemPath& out, int device = 0);

// "healthy" or "DEGRADED", with the reason in one sentence (see the ratio above).
const char* mem_path_verdict(const MemPath& m);

// The one line the engine prints, and the one line it prints when the path is degraded.
// Returns false when the path is degraded (so a caller can refuse to measure).
bool mem_path_report(const MemPath& m, void* file);

}  // namespace strata::kernels
