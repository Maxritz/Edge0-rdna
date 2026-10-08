// Integration test for the performance tracer: drives a real shell scope, ingests a
// synthetic llama-server log (the exact line shapes --perf/the server prints), and
// renders the sherlock-style component table. Run with --nocapture to see the report.
use edge0_app_lib::perf;
use std::io::Write;

#[test]
fn trace_report_end_to_end() {
    perf::set_enabled(true);
    perf::reset();
    {
        let _s = perf::scope("unit.outer");
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    {
        let _s = perf::scope("unit.inner");
    }

    let dir = std::env::temp_dir();
    let p = dir.join("edge0-perf-test.log");
    let mut f = std::fs::File::create(&p).unwrap();
    writeln!(f, "llama_model_load: load time =  1234.56 ms").unwrap();
    writeln!(f, "slot print_timing: prompt eval time =  11054.38 ms /  2925 tokens (    3.78 ms per token,   264.60 tokens per second)").unwrap();
    writeln!(f, "slot print_timing: eval time =   1760.11 ms /    48 tokens (   37.45 ms per token,    26.70 tokens per second)").unwrap();
    f.flush().unwrap();
    drop(f);

    perf::ingest_log(p.to_str().unwrap());

    // Let the resource sampler produce at least one real sample (primed at 300 ms,
    // 1 s cadence).
    std::thread::sleep(std::time::Duration::from_millis(1400));

    let snap = perf::snapshot_json();
    let report = snap["report"].as_str().unwrap();
    println!("\n----- TRACE REPORT -----\n{report}------------------------");
    println!("----- RESOURCES -----\n{}\n---------------------", serde_json::to_string_pretty(&snap["resources"]).unwrap());

    assert!(report.contains("unit.outer"), "shell scope missing from report");
    assert!(report.contains("engine.prompt-eval"), "engine phase missing from report");
    assert!(report.contains("instrumentation floor"), "floor line missing");
    assert!(snap["components"].as_array().unwrap().len() >= 2);
    assert!(!snap["resources"].is_null(), "resource sample missing");
    assert_eq!(snap["enabled"], serde_json::json!(true));

    perf::set_enabled(false);
}
