// serve/prefetch.h — edge0 advisory expert-prefetch router.
// Self-contained unit compiled into the llama target; depends only on llama
// public/near-public APIs (llama.h + the src/llama-ext.h layer-input channel,
// same route as speculative.cpp), linked into llama-server.
// The real routing gate is never touched: the router head predicts, for step t+1,
// the expert set of consumer layers from the owner layer's input (m_in) and a
// two-step history, then issues prefetches for the matching GGUF byte ranges.
// A miss costs latency only, never quality.
//
// Env gates (all dormant by default):
//   E0_PREROUTER=<path to prerouter_edge0_*.safetensors>   enable head + advisory prefetch
//   E0_PREFETCH=0                                          account only, no prefetch (A/B split)
//   E0_PREF_TRACE_N=<k>                                    print [pref-trace] every k steps (default 32)
//   E0_POOL_MB=<n>                                         hot-zone size MB of the private-page
//                                                          L1 pool (served through patch #7's
//                                                          expert-base resolver; can run without
//                                                          the sidecar as a demand pool)
//   E0_POOL_OBS_MB=<n>                                     observation-zone MB (meaningful only with heads on)
//   E0_POOL=sticky|obs                                     route MUL_MAT_ID reads through the L1 pool
//                                                          instead of plain mmap (sticky = re-touch the
//                                                          teacher set at step boundaries; obs = add the
//                                                          head-driven observation quota on top)
//   E0_PIN_MAX_MB=<n>                                      hot-zone byte cap (default 3072)
//   E0_POOL_WSHEAD_MB=<n>                                  headroom for the working-set hard cap
//                                                          (cap = base + hot + obs + headroom; default 1024)
//   E0_XLAYER=1                                            FATE cross-layer gate prefetch (T3,
//                                                          arXiv:2502.12224): predict consumer layer c from
//                                                          owner c-1's captured gate input (no training, no
//                                                          sidecar); replaces the head. Needs the F32 gate +
//                                                          the layer-inp capture channel (patch 0002).
//   E0_XLAYER_PCT=<f>                                       confidence percentile cutoff (default 0.75)
//   E0_XLAYER_EXT_MAX=<n>                                   cap on the percentile prefetch set (default 64)
//   E0_XLAYER_PF_TOK=<n>                                    prefill ubatch tokens aggregated for popularity (default 64)
//   E0_XLAYER_PF_TOK_MAX=<n>                                prefill per-layer popularity cap (default 96)
// [pref-trace] prints head/sticky/random accounting = proof the mechanism is live.
#pragma once

#include <cstdint>

// This unit is compiled into the llama target; when llama is built as a DLL it
// must be exported so llama-server-impl can link against it.
#ifdef _WIN32
#  define E0_PREF_API __declspec(dllexport)
#else
#  define E0_PREF_API __attribute__((visibility("default")))
#endif

struct llama_model;
struct llama_context;

// Called once after server model+ctx are ready. Returns false = not enabled or
// shape mismatch (stays dormant). pool_mb/gguf_path = formal CLI channels
// (--pool-mb / model path); <= 0 / NULL falls back to the env channel (debug/harness).
E0_PREF_API bool edge0_pref_init(llama_model * model, llama_context * ctx,
                                 int pool_mb, const char * gguf_path);

// Called after every llama_decode; last_row = the row index of this ubatch's last token
// in the layer-input capture buffer.
E0_PREF_API void edge0_pref_on_step(llama_context * ctx, int64_t last_row);
