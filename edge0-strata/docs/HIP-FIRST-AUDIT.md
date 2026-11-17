# HIP-first audit: where this engine is shaped like CUDA, and what ROCm would do instead

The engine was written against the CUDA API and compiled through HIP. Most of it is fine — the kernels are already
`gfx103x`-native (`v_dot2_f32_f16`, `dp4a`, wave32, LDS staging, runtime arch dispatch, `s_waitcnt` ordering) — but
a few decisions are CUDA workarounds or CUDA idioms that ROCm answers differently. Ranked by measured cost.

---

## 1. Host→device copies are issued as a KERNEL, because of a CUDA driver lock

`copy_bytes_from_mapped` (`src/kernels/cuda/elementwise.cu:290-306`) moves bytes with a compute dispatch rather
than the copy engine. The reason is in `verify.cpp`, at the call site: *"a copy KERNEL, not cudaMemcpyAsync: a
host-issued H2D inside the verify window (the GPU spinning on the flag it raises) is the issue #31 driver-lock
deadlock"*.

Measurements, decode window, per all-48-layer window, from `STRATA_VERIFY_PROFILE=1`:

| window width | `cpu/uCopy` ms |
|---|---|
| T1 (n=12) | 1.928 |
| T2 (n=2) | 3.602 |
| T3 (n=3) | 4.930 |
| T4 (n=13) | 6.368 |

Two costs that do not exist on the copy engine: it occupies compute CUs (stealing them from the actual GEMMs), and
it is serialised into the compute queue. ROCm answers this with an explicit **async copy on a non-blocking stream**
(SDMA engine) plus an event: the transfer leaves the compute queue entirely and overlaps. **The deadlock being
worked around is a CUDA driver lock** — it must be re-tested on HIP, not inherited. That is the definition of aping
CUDA: keeping a workaround for a bug in a different driver.

Cost if it works: 2-7 % of the decode window, and the CUs back.

## 2. GPU-side doorbell waits are SPIN KERNELS

The engine parks the GPU on a mapped flag by launching a kernel that spins with `S_SLEEP`/`S_MEMREALTIME` (the ISA
table in `RDNA2-OPTIMIZATION.md` §4 records this). ROCm has stream memory operations — `hipStreamWaitValue32/64`
and `hipStreamWriteValue32` — which park the **queue** on a memory value with no kernel resident at all. A spinning
kernel holds a CU (and its registers/LDS) while doing nothing, which is exactly the resource decode needs.

The host can still release it by writing the value, so the engine's `release_gpu_waits()` escape hatch survives.
Untested here; it is the largest structural ROCm-first item.

## 3. The 26.66 GiB expert arena runs on 4 KB pages

```
strata generate: expert arena: cudaHostRegister PORTABLE ok; large pages refused for 28628221952 B
                 (GetLargePageMinimum=2097152, VirtualAlloc error 1314); using 4 KB pages
```

`error 1314` is `ERROR_PRIVILEGE_NOT_HELD`: "Lock pages in memory" is not granted. The CPU pool reads this arena for
every expert it owns — 10.05 distinct experts per layer per window — so TLB pressure lands exactly on decode's CPU
half. Not a code change (it needs the Windows privilege, i.e. an admin decision), but it is worth knowing that the
CPU pool's throughput is partly a page-size artefact rather than an AVX2 limit.

## 4. Already tested and NOT worth revisiting

* **Stream priority** — measured as a non-lever three ways (`STRATA_MAIN_PRIO`). CUDA's priority semantics are not
  ROCm's; both were tried.
* **cudaStreamSynchronize / cudaDeviceSynchronize** — grepped: only in cold paths (conversation snapshots, tests,
  `dflash2` init). Nothing in the decode loop synchronises the device.
* **Wave64** — wave32 is the measured default and the occupancy readout (153 VGPR, 6 waves) is wave32.

## 5. Untested, plausible

* **D2H to mapped host** (`verify.cpp:2280-2285`): `cudaMemcpyAsync` straight into mapped host memory. On ReBAR
  hardware that write goes over PCIe per call; a pinned bounce buffer plus copy engine may beat it. Untested.
* **`gpu_stamp` cost** — ~17 µs per stamp (documented). Off by default; the profiler itself is a CUDA-style
  synchronous timer. A ring of device timestamps read once at the end would be nearly free, but the profiler is
  not on the hot path today.

---

# Where decode's 86.65 ms actually goes (the measurement that ranks all of the above)

One decode round is **96.06 ms** for ~2.13 tokens: `window 86.653 | draft_next 9.360 | choose+draft 0.002 |
verdict+emit 0.045`. The GPU is the bottleneck, **not** the CPU pool and **not** the export cache:

* `wait for rings 58.995 ms/round` is the **host** spinning on `while (*seq < want)` waiting for the GPU to reach
  each layer (`verify.cpp:3724-3762`, `ms_wait += b - a`). It is the host idling while the GPU works, so it is not
  a cost to remove — misreading it as a CPU stall was wrong.
* the CPU pool does 20.1 ms of real work per round, overlapped.

GPU slots, summed over all 48 layers of one window (T1, n=12):

| slot | ms | share of 86.65 |
|---|---|---|
| **hc1 norm** (GDN 15.769 + QSA 6.232) | **22.0** | **25 %** |
| **D stage** (12.447 + 5.129) | 17.6 | 20 % |
| pcie/uWait (expert DMA stalls) | 9.4 | 11 % |
| hc1 up (7.048 + 2.337) | 9.4 | 11 % |
| qkv gemv | 6.1 | 7 % |
| head | 4.2 | 5 % |
| hc-read0 (3.751 + 1.109) | 4.9 | 6 % |
| z / rec / shared+quant / VRAM hits | 3.7 / 0.9 / 1.8 / 1.2 | |

**The single biggest decode item is a normalisation pass, not a GEMM** — 22 ms of 86.65. Any plan that starts with
GEMM work in decode is aimed at the third-largest slot.

## A trap worth recording

`--vram-reserve-mib 256` (2,149 cache slots) vs `512` (1,918) moved decode from 21.97/22.14 to 22.64/22.46 — inside
the noise — and left `wait for rings` unchanged (59.4 vs 59.0 ms). **The cache size also changed the produced
tokens** (`715 1822 363 313 198 262 514 1249 2796 ...` against `... 514 264 283 220 ...`), because the CPU/GPU
expert split changes the accumulation order. So expert-cache size is a *numerical* parameter: decode A/Bs on it may
compare throughput only, never "byte-identical output".
