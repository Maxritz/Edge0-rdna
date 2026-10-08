// Measured tuning sweep: launches the real engine under several configurations and
// records prefill/decode throughput straight from llama-server's /completion timings.
// This is the "where can we tune" evidence: each row is one measured config, not a
// guess. Every child is killed before the next config starts.
//
//   cargo run --example tune_sweep
// Env: EDGE0_SMOKE_GGUF (model path), EDGE0_SWEEP_CTX (default 32768).
use edge0_app_lib::paths;
use serde_json::{json, Value};
use std::process::{Child, Command, Stdio};
use std::time::{Duration, Instant};

fn free_port() -> u16 {
    std::net::TcpListener::bind("127.0.0.1:0")
        .and_then(|l| l.local_addr())
        .map(|a| a.port())
        .unwrap_or(8080)
}

fn wait_health(base: &str, child: &mut Child) -> bool {
    let c = reqwest::blocking::Client::builder()
        .timeout(Duration::from_secs(2))
        .build()
        .unwrap();
    let t0 = Instant::now();
    loop {
        if c.get(format!("{base}/health")).send().map(|r| r.status().is_success()).unwrap_or(false) {
            return true;
        }
        if child.try_wait().ok().flatten().is_some() {
            return false; // died (e.g. did not fit / OOM)
        }
        if t0.elapsed().as_secs() > 180 {
            return false;
        }
        std::thread::sleep(Duration::from_millis(500));
    }
}

struct Row {
    n_cpu_moe: u32,
    kv: String,
    prefill: f64,
    decode: f64,
    prompt_n: u64,
    pred_n: u64,
    vram: Option<f64>,
}

fn run_config(bin: &std::path::Path, model: &str, ctx: u32, n_cpu_moe: u32, kv: &str) -> Option<Row> {
    let exe = bin.join("llama-server.exe");
    let port = free_port();
    let base = format!("http://127.0.0.1:{port}");
    let log = std::env::temp_dir().join(format!("edge0-sweep-{n_cpu_moe}-{kv}.log"));
    let out = std::fs::File::create(&log).ok()?;
    let err = out.try_clone().ok()?;

    let mut args: Vec<String> = vec![
        "-m".into(), model.into(),
        "-ngl".into(), "99".into(),
        "--n-cpu-moe".into(), n_cpu_moe.to_string(),
        "--ctx-size".into(), ctx.to_string(),
        "--flash-attn".into(), "auto".into(),
        "--host".into(), "127.0.0.1".into(),
        "--port".into(), port.to_string(),
    ];
    if kv != "f16" {
        args.push("--cache-type-k".into());
        args.push(kv.into());
        args.push("--cache-type-v".into());
        args.push(kv.into());
    }
    let mut child = Command::new(&exe)
        .args(&args)
        .current_dir(bin)
        .stdin(Stdio::null())
        .stdout(Stdio::from(out))
        .stderr(Stdio::from(err))
        .spawn()
        .ok()?;

    let ready = wait_health(&base, &mut child);
    if !ready {
        let _ = child.kill();
        let _ = child.wait();
        eprintln!("  n_cpu_moe={n_cpu_moe} kv={kv}: did not come up (log {})", log.display());
        return None;
    }

    let client = reqwest::blocking::Client::new();
    let prompt = "The quick brown fox jumps over the lazy dog. ".repeat(100);
    let body = json!({ "prompt": prompt, "n_predict": 128, "temperature": 0.0, "cache_prompt": false });
    let v: Value = match client.post(format!("{base}/completion")).json(&body).timeout(Duration::from_secs(900)).send() {
        Ok(r) => r.json().unwrap_or(Value::Null),
        Err(e) => {
            eprintln!("  n_cpu_moe={n_cpu_moe} kv={kv}: request failed {e}");
            let _ = child.kill();
            let _ = child.wait();
            return None;
        }
    };
    let t = &v["timings"];
    let row = Row {
        n_cpu_moe,
        kv: kv.to_string(),
        prefill: t["prompt_per_second"].as_f64().unwrap_or(0.0),
        decode: t["predicted_per_second"].as_f64().unwrap_or(0.0),
        prompt_n: t["prompt_n"].as_u64().unwrap_or(0),
        pred_n: t["predicted_n"].as_u64().unwrap_or(0),
        vram: vram_now(),
    };
    let _ = child.kill();
    let _ = child.wait();
    Some(row)
}

/// VRAM used via the perf resource sampler (real PDH Dedicated Usage).
fn vram_now() -> Option<f64> {
    let snap = edge0_app_lib::perf::snapshot_json();
    snap["resources"]["vram_used_gb"].as_f64()
}

fn main() {
    let bin = paths::bin_dir();
    if !bin.join("llama-server.exe").exists() {
        eprintln!("engine missing at {}", bin.display());
        std::process::exit(1);
    }
    let model = std::env::var("EDGE0_SMOKE_GGUF")
        .unwrap_or_else(|_| r"H:\OLLAMA-Models\GGUF\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf".to_string());
    let ctx: u32 = std::env::var("EDGE0_SWEEP_CTX").ok().and_then(|v| v.parse().ok()).unwrap_or(32768);

    edge0_app_lib::perf::set_enabled(true);

    // EDGE0_SWEEP_CONFIGS="8:f16,12:q8_0" overrides the default list.
    let configs: Vec<(u32, String)> = match std::env::var("EDGE0_SWEEP_CONFIGS") {
        Ok(s) => s
            .split(',')
            .filter_map(|part| {
                let (n, kv) = part.split_once(':')?;
                Some((n.trim().parse().ok()?, kv.trim().to_string()))
            })
            .collect(),
        Err(_) => vec![
            (8, "f16".into()), (12, "f16".into()), (16, "f16".into()),
            (20, "f16".into()), (24, "f16".into()), (31, "f16".into()),
            (12, "q8_0".into()), (16, "q8_0".into()), (24, "q8_0".into()),
        ],
    };

    let mut rows: Vec<Row> = Vec::new();
    for (n, kv) in configs {
        let kv = kv.as_str();
        eprintln!("running n_cpu_moe={n} kv={kv} ctx={ctx} ...");
        if let Some(r) = run_config(&bin, &model, ctx, n, kv) {
            eprintln!(
                "  prefill {:.0} tok/s ({}/{} tok)  decode {:.1} tok/s  vram {}",
                r.prefill, r.prompt_n, r.prompt_n, r.decode,
                r.vram.map(|v| format!("{v:.1} GiB")).unwrap_or_else(|| "n/a".into())
            );
            rows.push(r);
        }
    }

    println!("\n===== TUNING SWEEP (ctx {ctx}) =====");
    println!(
        "{:<11} {:<6} {:>14} {:>14} {:>12}",
        "n_cpu_moe", "kv", "prefill tok/s", "decode tok/s", "vram GiB"
    );
    for r in &rows {
        println!(
            "{:<11} {:<6} {:>14.0} {:>14.1} {:>12}",
            r.n_cpu_moe,
            r.kv,
            r.prefill,
            r.decode,
            r.vram.map(|v| format!("{v:.1}")).unwrap_or_else(|| "n/a".into())
        );
    }
    println!("(prefill measured over {} tokens, decode over {} tokens)", rows.first().map(|r| r.prompt_n).unwrap_or(0), rows.first().map(|r| r.pred_n).unwrap_or(0));

    edge0_app_lib::perf::set_enabled(false);
}
