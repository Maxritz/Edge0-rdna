// TEMP live smoke (untracked helper for manual runs): real doctor + real local-GGUF
// engine load against the HIP engine, via the production code paths only.
use edge0_app_lib::sink::{NoopSink, Sink};
use edge0_app_lib::{doctor, engine};
use std::sync::{Arc, Mutex};

fn find<'a>(r: &'a serde_json::Value, id: &str) -> serde_json::Value {
    r["checks"]
        .as_array()
        .expect("checks array")
        .iter()
        .find(|c| c["id"] == id)
        .cloned()
        .unwrap_or_else(|| panic!("check {id} missing"))
}

#[test]
fn live_doctor_reports_backend_and_gpu() {
    let st: engine::EngineState = Mutex::new(None);
    let r = doctor::run(&st);
    println!("DOCTOR {}\n", serde_json::to_string_pretty(&r).unwrap());
    let backend = find(&r, "backend");
    let gpu = find(&r, "gpu");
    println!("BACKEND verdict={} detail={}", backend["verdict"], backend["detail"]);
    println!("GPU     verdict={} detail={}", gpu["verdict"], gpu["detail"]);
    assert_eq!(backend["verdict"], "pass", "backend check must pass");
    assert!(
        backend["detail"].as_str().unwrap().starts_with("hip"),
        "backend must report hip"
    );
    assert_eq!(gpu["verdict"], "pass", "gpu check must pass");
    assert!(
        gpu["detail"].as_str().unwrap().contains("Radeon"),
        "gpu detail must carry the engine-reported GPU"
    );
}

#[test]
fn live_load_local_gguf() {
    edge0_app_lib::paths::ensure_dirs(None).expect("ensure_dirs (~/.edge0 layout, as app startup does)");
    let path = std::env::var("EDGE0_SMOKE_GGUF")
        .unwrap_or_else(|_| r"H:\OLLAMA-Models\GGUF\Qwen3.5-35B-A3B-UD-Q4_K_XL.gguf".to_string());
    let sink: Arc<dyn Sink> = Arc::new(NoopSink);
    let state: engine::EngineState = Mutex::new(None);
    let v = engine::start_gguf(&sink, &state, &path, 8192).expect("start_gguf");
    println!("ENGINE {}\n", serde_json::to_string_pretty(&v).unwrap());
    println!("LOG PATH: {}", v["log"].as_str().unwrap_or(""));
    engine::stop(&state);
    assert_eq!(engine::status(&state)["running"], serde_json::json!(false), "stop must stop");
}
