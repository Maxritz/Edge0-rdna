// Headless demonstration of the performance tracer through the real shell code
// paths: enable tracing, load a GGUF via engine::start_gguf (which arms --perf),
// run one real generation, fold the engine log in, print the report, stop the engine.
//
//   cargo run --example trace_demo --release
// Env: EDGE0_SMOKE_GGUF (model path), EDGE0_DEMO_CTX (context, default 32768).
use edge0_app_lib::sink::{NoopSink, Sink};
use edge0_app_lib::{engine, paths, perf};
use serde_json::json;
use std::sync::{Arc, Mutex};

fn main() {
    let path = std::env::var("EDGE0_SMOKE_GGUF")
        .unwrap_or_else(|_| r"H:\OLLAMA-Models\GGUF\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf".to_string());
    let ctx: u32 = std::env::var("EDGE0_DEMO_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(32768);

    perf::set_enabled(true);
    paths::ensure_dirs(None).expect("ensure_dirs");

    let sink: Arc<dyn Sink> = Arc::new(NoopSink);
    let state: engine::EngineState = Mutex::new(None);

    let info = match engine::start_gguf(&sink, &state, &path, ctx) {
        Ok(v) => v,
        Err(e) => {
            eprintln!("start_gguf failed: {e}");
            perf::set_enabled(false);
            std::process::exit(1);
        }
    };
    let base = info["base_url"].as_str().unwrap_or("").to_string();
    let log = info["log"].as_str().unwrap_or("").to_string();
    eprintln!("engine up at {base}  (log {log})");
    eprintln!("plan: {}", serde_json::to_string(&info["plan"]).unwrap_or_default());

    let client = reqwest::blocking::Client::new();
    let body = json!({
        "prompt": "Explain in detail how a mixture-of-experts transformer routes each token to its experts.",
        "n_predict": 128,
        "temperature": 0.2
    });
    let t0 = std::time::Instant::now();
    match client
        .post(format!("{base}/completion"))
        .json(&body)
        .timeout(std::time::Duration::from_secs(900))
        .send()
    {
        Ok(r) => {
            let status = r.status();
            let v: serde_json::Value = r.json().unwrap_or(serde_json::Value::Null);
            let n = v["tokens_predicted"].as_u64().unwrap_or(0);
            eprintln!("generation: status {status}, {n} tokens in {:.1}s", t0.elapsed().as_secs_f64());
        }
        Err(e) => eprintln!("generation failed: {e}"),
    }

    perf::ingest_log(&log);
    let snap = perf::snapshot_json();
    println!("\n===== PERFORMANCE TRACE =====\n{}", snap["report"].as_str().unwrap_or(""));
    println!("===== RESOURCES =====\n{}", serde_json::to_string_pretty(&snap["resources"]).unwrap());

    engine::stop(&state);
    perf::set_enabled(false);
    eprintln!("engine stopped.");
}
