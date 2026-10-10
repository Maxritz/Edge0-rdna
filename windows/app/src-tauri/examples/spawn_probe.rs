// Isolate the spawn-method slowdown seen on the gfx1031 box: the SAME llama-server
// with the SAME args runs ~28 tok/s when launched from PowerShell but ~16 tok/s when
// launched by engine::launch. This probe reproduces engine::launch's spawn exactly and
// toggles each non-default factor (CREATE_NO_WINDOW, the kill job, cwd) via EDGE0_MODE.
//
//   set EDGE0_BIN_DIR=<bin> & set EDGE0_SMOKE_GGUF=<model> & set EDGE0_MODE=plain|nowin|job|full|cwd
//   cargo run --release --example spawn_probe
use std::process::{Command, Stdio};

#[cfg(windows)]
use std::os::windows::process::CommandExt;

fn main() {
    let bin = std::env::var("EDGE0_BIN_DIR").expect("EDGE0_BIN_DIR");
    let model = std::env::var("EDGE0_SMOKE_GGUF").expect("EDGE0_SMOKE_GGUF");
    let ncpu = std::env::var("EDGE0_NCMOE").unwrap_or_else(|_| "26".into());
    let ctx = std::env::var("EDGE0_DEMO_CTX").unwrap_or_else(|_| "32768".into());
    let mode = std::env::var("EDGE0_MODE").unwrap_or_else(|_| "full".into());
    let port = 8096;

    let exe = format!("{bin}\\llama-server.exe");
    let log = std::env::temp_dir().join("spawn_probe.log");
    let logfile = std::fs::File::create(&log).expect("log");
    let log2 = logfile.try_clone().expect("log2");

    let mut cmd = Command::new(&exe);
    cmd.args([
        "-m", &model, "-ngl", "99", "--n-cpu-moe", &ncpu, "--ctx-size", &ctx,
        "--flash-attn", "auto", "--no-webui", "--perf", "--port", &port.to_string(),
    ]);
    if mode == "nowin" || mode == "full" {
        #[cfg(windows)]
        cmd.creation_flags(0x0800_0000);
    }
    if mode != "nocwd" {
        cmd.current_dir(&bin);
    }
    cmd.stdout(Stdio::from(logfile)).stderr(Stdio::from(log2));
    let mut child = cmd.spawn().expect("spawn");
    let job = if mode == "job" || mode == "full" { spawn_job(&child) } else { false };

    // wait for health
    let client = reqwest::blocking::Client::new();
    let base = format!("http://127.0.0.1:{port}");
    let mut ready = false;
    for _ in 0..150 {
        std::thread::sleep(std::time::Duration::from_millis(500));
        if let Ok(r) = client.get(format!("{base}/health")).send() {
            if r.status().is_success() {
                ready = true;
                break;
            }
        }
    }
    let body = serde_json::json!({
        "prompt": "Explain in detail how a mixture-of-experts transformer routes each token to its experts.",
        "n_predict": 128, "temperature": 0.2
    });
    let t0 = std::time::Instant::now();
    let r = client.post(format!("{base}/completion")).json(&body).timeout(std::time::Duration::from_secs(600)).send();
    let secs = t0.elapsed().as_secs_f64();
    let toks = r.ok().and_then(|r| r.json::<serde_json::Value>().ok()).and_then(|v| v["tokens_predicted"].as_u64()).unwrap_or(0);
    eprintln!("mode={mode} job={job} ready={ready} tokens={toks} wall={secs:.1}s ({:.2} tok/s wall)", toks as f64 / secs);
    // stop
    let _ = child.kill();
    let _ = child.wait();
    std::thread::sleep(std::time::Duration::from_millis(1500));

    let text = std::fs::read_to_string(&log).unwrap_or_default();
    for line in text.lines() {
        if line.contains("eval time =") || line.contains("prompt eval time =") || line.contains("n_gen =") {
            eprintln!("  {}", line.trim());
        }
    }
}

#[cfg(windows)]
fn spawn_job(_child: &std::process::Child) -> bool {
    use std::os::windows::io::AsRawHandle;
    use windows_sys::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, SetInformationJobObject,
        JobObjectExtendedLimitInformation, JOBOBJECT_EXTENDED_LIMIT_INFORMATION,
        JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
    };
    unsafe {
        let h = CreateJobObjectW(std::ptr::null(), std::ptr::null());
        if h.is_null() {
            return false;
        }
        let mut li: JOBOBJECT_EXTENDED_LIMIT_INFORMATION = std::mem::zeroed();
        li.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
        SetInformationJobObject(h, JobObjectExtendedLimitInformation, &li as *const _ as *const _, std::mem::size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32);
        AssignProcessToJobObject(h, _child.as_raw_handle()) != 0
    }
}

#[cfg(not(windows))]
fn spawn_job(_child: &std::process::Child) -> bool {
    false
}
