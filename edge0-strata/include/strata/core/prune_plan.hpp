#pragma once
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace strata::core {

/// The MoE rank cut for ONE layer during generation, read from the same two knobs prefill's mask reads:
///   STRATA_PRUNE_APPLY=1 gates the mask (unset/0 = keep every slot),
///   STRATA_PRUNE_PLAN_L="r0,r1,...,r47" (index = layer) is the layer-aware budget, and
///   STRATA_PRUNE_STATS=R is the flat rank used when the plan is unset or shorter than `layer`.
/// A value of R keeps the first R slots and 0 keeps all of them, so prompt and generation can never disagree
/// about a layer: prefill builds its keep[] mask with the identical rule.
///
/// The plan is parsed once per process, lazily, exactly like prefill's static.  Returns 0 = "keep all", which
/// every combine entry point reads as no mask.
inline int64_t prune_rank_for_layer(int64_t layer) {
    static const bool apply = [] {
        const char* v = std::getenv("STRATA_PRUNE_APPLY");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    if (!apply) return 0;
    static const std::vector<int64_t> plan = [] {
        std::vector<int64_t> p;
        const char* pl = std::getenv("STRATA_PRUNE_PLAN_L");
        for (const char* s = pl; s != nullptr && *s != '\0'; ) {
            p.push_back((int64_t) std::atoll(s));
            while (*s != '\0' && *s != ',') ++s;
            if (*s == ',') ++s;
        }
        return p;
    }();
    static const int64_t flat = [] {
        const char* v = std::getenv("STRATA_PRUNE_STATS");
        return v != nullptr ? (int64_t) std::atoll(v) : (int64_t) 0;
    }();
    if (!plan.empty() && layer >= 0 && (size_t) layer < plan.size()) return plan[(size_t) layer];
    return flat;
}

}  // namespace strata::core
