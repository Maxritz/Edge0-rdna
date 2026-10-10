// autotune.rs — measured n_cpu_moe selection. The offline planner picks the smallest
// expert offload that fits the VRAM budget; measurement (RUN-012) shows decode is
// non-monotonic in n_cpu_moe and peaks *below* that fit point (35.05 vs 29.5 tok/s on
// gfx1031/35B), so the fastest config is not the smallest-fitting one.
//
// This module does not guess the peak: it runs `llama-bench -o json` on a small
// candidate set around the planner's choice, reads the real decode rate, and keeps the
// measured best. Results are cached per (model, ctx, gpu) so the cost is paid once.
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::process::{Command, Stdio};

#[derive(Debug, Clone, PartialEq)]
pub struct BenchResult {
    pub n_cpu_moe: u32,
    pub pp: f64,
    pub tg: f64,
}

/// Parse `llama-bench -o json`: an array with one object per test. Prompt tests carry
/// n_prompt>0 and n_gen==0; decode tests carry n_gen>0. Only objects that report a
/// positive average and a n_cpu_moe are kept.
pub fn parse_bench_json(text: &str) -> Vec<BenchResult> {
    let v: Value = match serde_json::from_str(text.trim()) {
        Ok(v) => v,
        Err(_) => return Vec::new(),
    };
    let arr = match v.as_array() {
        Some(a) => a,
        None => return Vec::new(),
    };
    let mut out = Vec::new();
    for o in arr {
        let n_cpu_moe = match o["n_cpu_moe"].as_u64() {
            Some(n) => n as u32,
            None => continue,
        };
        let avg = o["avg_ts"].as_f64().unwrap_or(0.0);
        if !(avg > 0.0) {
            continue;
        }
        let n_gen = o["n_gen"].as_u64().unwrap_or(0);
        let n_prompt = o["n_prompt"].as_u64().unwrap_or(0);
        if n_gen > 0 {
            out.push(BenchResult { n_cpu_moe, pp: 0.0, tg: avg });
        } else if n_prompt > 0 {
            out.push(BenchResult { n_cpu_moe, pp: avg, tg: 0.0 });
        }
    }
    out
}

/// Candidate n_cpu_moe values to probe: the planner's pick plus its neighbours at
/// +/- one step, clamped to [0, layers]. Order is ascending; duplicates removed.
pub fn candidates(current: u32, layers: u32, step: u32) -> Vec<u32> {
    let step = step.max(1);
    let mut set = vec![
        current.saturating_sub(step).min(layers),
        current.min(layers),
        current.saturating_add(step).min(layers),
    ];
    // Also probe the peak region seen in the sweep (roughly 60% of the fit point) when
    // the planner is far from it, so the search is not trapped below the real optimum.
    let peak = (current as f64 * 0.6).round() as u32;
    set.push(peak.min(layers));
    set.push((peak.saturating_sub(step)).min(layers));
    set.push((peak + step).min(layers));
    set.sort_unstable();
    set.dedup();
    set
}

/// Fastest candidate by measured decode tok/s. Ties break toward the smaller n (less
/// CPU work, more VRAM headroom). Returns None when no candidate measured a decode.
pub fn best_of(results: &[BenchResult]) -> Option<BenchResult> {
    results
        .iter()
        .filter(|r| r.tg > 0.0)
        .cloned()
        .reduce(|a, b| if b.tg > a.tg + 0.05 || (b.tg >= a.tg - 0.05 && b.n_cpu_moe < a.n_cpu_moe) { b } else { a })
}

/// Run llama-bench once at one n_cpu_moe and return the decode tok/s (0.0 when the run
/// produced no decode measurement). The bench binary is launched with no console window.
/// `ctx` is part of the sweep's cache key but is not passed to llama-bench, which sizes
/// its KV cache itself (it takes -fitc/-fit-target, not --ctx-size).
pub fn run_bench(
    bench_exe: &Path,
    model: &str,
    ctx: u32,
    n_cpu_moe: u32,
    bench_prompt: u32,
    bench_gen: u32,
    reps: u32,
) -> Result<f64, String> {
    let _ = ctx;
    if !bench_exe.exists() {
        return Err(format!("E-AUTOTUNE-NOBENCH {}", bench_exe.display()));
    }
    let mut cmd = Command::new(bench_exe);
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        cmd.creation_flags(0x0800_0000);
    }
    let out = cmd
        .args([
            "-m", model,
            "-ngl", "99",
            "--n-cpu-moe", &n_cpu_moe.to_string(),
            "-fa", "auto",
            "-p", &bench_prompt.to_string(),
            "-n", &bench_gen.to_string(),
            "-r", &reps.to_string(),
            "-o", "json",
        ])
        .current_dir(bench_exe.parent().unwrap_or_else(|| Path::new(".")))
        .stdin(Stdio::null())
        .output()
        .map_err(|e| format!("E-AUTOTUNE-SPAWN {e}"))?;
    let text = String::from_utf8_lossy(&out.stdout).to_string();
    let results = parse_bench_json(&text);
    Ok(best_of(&results).map(|r| r.tg).unwrap_or(0.0))
}

/// Sweep the candidate set and return the measured best (n_cpu_moe, decode tok/s).
/// Every candidate is measured at least once; a candidate that fails to load is skipped
/// (kept, not fabricated).
pub fn sweep(
    bench_exe: &Path,
    model: &str,
    ctx: u32,
    current: u32,
    layers: u32,
    bench_prompt: u32,
    bench_gen: u32,
    reps: u32,
) -> (Option<BenchResult>, Vec<BenchResult>) {
    let cands = candidates(current, layers, (layers / 10).max(2));
    let mut measured: Vec<BenchResult> = Vec::new();
    for n in cands {
        if let Ok(tg) = run_bench(bench_exe, model, ctx, n, bench_prompt, bench_gen, reps) {
            if tg > 0.0 {
                measured.push(BenchResult { n_cpu_moe: n, pp: 0.0, tg });
            }
        }
    }
    (best_of(&measured), measured)
}

// --------------------------------------------------------------------------- cache

fn cache_path(state_dir: &Path) -> PathBuf {
    state_dir.join("autotune.json")
}

fn cache_key(model: &str, ctx: u32, gpu: &str) -> String {
    let name = Path::new(model)
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| model.to_string());
    let size = std::fs::metadata(model).map(|m| m.len()).unwrap_or(0);
    format!("{name}|{size}|{ctx}|{gpu}")
}

/// Read a cached tuned n_cpu_moe for this (model, ctx, gpu), if one was measured.
pub fn cached(state_dir: &Path, model: &str, ctx: u32, gpu: &str) -> Option<u32> {
    let text = std::fs::read_to_string(cache_path(state_dir)).ok()?;
    let v: Value = serde_json::from_str(&text).ok()?;
    v[cache_key(model, ctx, gpu)]["n_cpu_moe"].as_u64().map(|n| n as u32)
}

/// Persist a measured result so later loads reuse it instead of re-benchmarking.
pub fn store(
    state_dir: &Path,
    model: &str,
    ctx: u32,
    gpu: &str,
    n_cpu_moe: u32,
    tg: f64,
) -> Result<(), String> {
    let _ = std::fs::create_dir_all(state_dir);
    let p = cache_path(state_dir);
    let mut v: Value = std::fs::read_to_string(&p)
        .ok()
        .and_then(|t| serde_json::from_str(&t).ok())
        .unwrap_or_else(|| json!({}));
    if !v.is_object() {
        v = json!({});
    }
    v[cache_key(model, ctx, gpu)] = json!({ "n_cpu_moe": n_cpu_moe, "tg": tg });
    std::fs::write(&p, serde_json::to_string_pretty(&v).unwrap())
        .map_err(|e| format!("E-AUTOTUNE-WRITE {e}"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = r#"[
      {"n_cpu_moe":20,"n_prompt":16,"n_gen":0,"avg_ts":48.596},
      {"n_cpu_moe":20,"n_prompt":0,"n_gen":32,"avg_ts":32.224}
    ]"#;

    #[test]
    fn parses_prompt_and_gen_objects() {
        let r = parse_bench_json(SAMPLE);
        assert_eq!(r.len(), 2);
        let gen = r.iter().find(|x| x.tg > 0.0).unwrap();
        assert_eq!(gen.n_cpu_moe, 20);
        assert!((gen.tg - 32.224).abs() < 1e-6);
        let pp = r.iter().find(|x| x.pp > 0.0).unwrap();
        assert!((pp.pp - 48.596).abs() < 1e-6);
    }

    #[test]
    fn rejects_garbage() {
        assert!(parse_bench_json("not json").is_empty());
        assert!(parse_bench_json("{}").is_empty());
        assert!(parse_bench_json(r#"[{"n_prompt":16,"n_gen":0,"avg_ts":5}]"#).is_empty());
    }

    #[test]
    fn candidates_cover_the_fit_point_and_peak() {
        let c = candidates(26, 40, 4);
        assert!(c.contains(&26), "fit point must be probed");
        assert!(c.contains(&16) || c.contains(&15), "peak region must be probed: {c:?}");
        assert!(c.windows(2).all(|w| w[0] <= w[1]), "ascending");
        assert!(c.iter().all(|&n| n <= 40));
    }

    #[test]
    fn candidates_clamp_at_bounds() {
        assert!(candidates(0, 40, 4).iter().all(|&n| n <= 40));
        let hi = candidates(40, 40, 4);
        assert!(hi.iter().all(|&n| n <= 40));
        assert!(hi.contains(&40));
    }

    #[test]
    fn best_of_picks_fastest_and_breaks_ties_low() {
        let r = vec![
            BenchResult { n_cpu_moe: 16, pp: 0.0, tg: 32.0 },
            BenchResult { n_cpu_moe: 20, pp: 0.0, tg: 35.0 },
            BenchResult { n_cpu_moe: 24, pp: 0.0, tg: 35.0 },
        ];
        let b = best_of(&r).unwrap();
        assert_eq!(b.n_cpu_moe, 20, "tie must break toward the smaller n");
        assert!((b.tg - 35.0).abs() < 1e-9);
    }

    #[test]
    fn best_of_none_when_no_decode() {
        let r = vec![BenchResult { n_cpu_moe: 20, pp: 48.0, tg: 0.0 }];
        assert!(best_of(&r).is_none());
    }

    #[test]
    fn cache_roundtrip() {
        let dir = std::env::temp_dir().join(format!("edge0-at-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let model = dir.join("m.gguf");
        std::fs::write(&model, b"x").unwrap();
        let ms = model.to_str().unwrap();
        assert_eq!(cached(&dir, ms, 32768, "ROCm0"), None);
        store(&dir, ms, 32768, "ROCm0", 20, 35.0).unwrap();
        assert_eq!(cached(&dir, ms, 32768, "ROCm0"), Some(20));
        assert_eq!(cached(&dir, ms, 8192, "ROCm0"), None, "ctx must key the cache");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
