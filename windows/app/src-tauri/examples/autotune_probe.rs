// Runs the measured n_cpu_moe autotune for a model and prints exactly what was
// measured. Uses the real llama-bench (no estimates). Mirrors the engine_autotune
// command so it can be exercised headless on a target box.
//
//   set EDGE0_BIN_DIR=<bin> & set EDGE0_REPO=<windows> & set EDGE0_PY=python3
//   set EDGE0_SMOKE_GGUF=<model> & set EDGE0_DEMO_CTX=32768
//   cargo run --release --example autotune_probe
use edge0_app_lib::{engine, paths, perf};

fn main() {
    perf::set_enabled(false);
    paths::ensure_dirs(None).expect("ensure_dirs");
    let path = std::env::var("EDGE0_SMOKE_GGUF")
        .unwrap_or_else(|_| r"H:\OLLAMA-Models\GGUF\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf".to_string());
    let ctx: u32 = std::env::var("EDGE0_DEMO_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(32768);
    eprintln!("autotune {path} ctx={ctx} bin={:?}", paths::bin_dir());
    match engine::autotune_model(&path, ctx) {
        Ok(v) => println!("{}", serde_json::to_string_pretty(&v).unwrap()),
        Err(e) => {
            eprintln!("autotune failed: {e}");
            std::process::exit(1);
        }
    }
}
