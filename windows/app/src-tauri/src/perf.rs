// perf.rs — opt-in performance tracing across everything the model is handled
// through: the shell's own functions (spans), the engine's device phases (parsed
// from the llama-server log), and live system resources (CPU, RAM, GPU util, VRAM,
// ReBAR). Nothing here estimates: every number is a measured value, and a metric
// whose source is absent is reported as unavailable rather than guessed.
//
// Activation: `--profileperf` on the shell command line (or EDGE0_PROFILEPERF=1),
// or the runtime `perf_set` command. When on, engine launches also add `--perf` so
// llama-server emits its internal phase timings, which `ingest_log` folds in.
//
// Win32 access uses plain externs (PDH + advapi32 + kernel32). This mirrors the
// convention in doctor.rs/engine.rs: the windows-sys bindings for these specific
// calls tripped link/resolution issues here, and a direct extern is all we need.
use serde_json::{json, Value};
use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

// --------------------------------------------------------------------------- globals

static ENABLED: AtomicBool = AtomicBool::new(false);
static PERF: OnceLock<Mutex<Perf>> = OnceLock::new();
static SAMPLER_ON: AtomicBool = AtomicBool::new(false);
static SAMPLER: Mutex<Option<std::thread::JoinHandle<()>>> = Mutex::new(None);

fn perf() -> &'static Mutex<Perf> {
    PERF.get_or_init(|| Mutex::new(Perf::new()))
}

pub fn enabled() -> bool {
    ENABLED.load(Ordering::Relaxed)
}

fn now_epoch() -> u64 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0)
}

struct Perf {
    started_at: Instant,
    started_epoch: u64,
    spans: BTreeMap<String, SpanAgg>,
    phases: BTreeMap<String, PhaseAgg>,
    idle: BTreeMap<String, u64>,
    span_last_end: BTreeMap<String, u64>,
    resources: Vec<ResourceSample>,
    last_res: Option<ResourceSample>,
    engine_log: Option<String>,
    floor: Option<Floor>,
}

impl Perf {
    fn new() -> Self {
        Perf {
            started_at: Instant::now(),
            started_epoch: now_epoch(),
            spans: BTreeMap::new(),
            phases: BTreeMap::new(),
            idle: BTreeMap::new(),
            span_last_end: BTreeMap::new(),
            resources: Vec::new(),
            last_res: None,
            engine_log: None,
            floor: None,
        }
    }
}

#[derive(Default, Clone)]
struct SpanAgg {
    ops: u64,
    host_us: u64,
    host_us_max: u64,
}

#[derive(Default, Clone)]
struct PhaseAgg {
    ops: u64,
    dev_us: u64,
}

#[derive(Clone, Copy)]
struct Floor {
    n: u64,
    host_us_each: f64,
}

/// Cross-process/device resource sample. Fields are Option so an absent counter is
/// reported as null, never coerced to zero.
#[derive(Clone, Default)]
struct ResourceSample {
    t_ms: u64,
    cpu_pct: Option<f64>,
    ram_used_gb: Option<f64>,
    ram_total_gb: Option<f64>,
    gpu_util_pct: Option<f64>,
    vram_used_gb: Option<f64>,
    vram_total_gb: Option<f64>,
    rebar: Option<(String, u32, String)>, // (state, raw value, source)
}

// --------------------------------------------------------------------------- spans

/// RAII scope timer. A no-op when tracing is off, so instrumented call sites carry
/// no measurable cost until `--profileperf` is set.
pub struct Scope {
    name: String,
    t0: Option<Instant>,
    on: bool,
}

pub fn scope(name: &str) -> Scope {
    if enabled() {
        Scope {
            name: name.to_string(),
            t0: Some(Instant::now()),
            on: true,
        }
    } else {
        Scope {
            name: String::new(),
            t0: None,
            on: false,
        }
    }
}

impl Drop for Scope {
    fn drop(&mut self) {
        if self.on {
            if let Some(t0) = self.t0 {
                record_span(&self.name, t0.elapsed());
            }
        }
    }
}

fn record_span(name: &str, dur: Duration) {
    let us = dur.as_micros() as u64;
    let p = perf();
    let mut g = p.lock().unwrap();
    let end_ns = g.started_at.elapsed().as_nanos() as u64;
    let dur_ns = dur.as_nanos() as u64;
    let start_ns = end_ns.saturating_sub(dur_ns);
    // idle = wall gap between the end of the previous call of THIS component and the
    // start of this one ("time spent elsewhere between successive calls"). Per
    // component so parallel threads and unrelated components never pollute it.
    if let Some(prev) = g.span_last_end.get(name).copied() {
        if start_ns > prev {
            *g.idle.entry(name.to_string()).or_insert(0) += (start_ns - prev) / 1000; // ns -> us
        }
    }
    g.span_last_end.insert(name.to_string(), end_ns);
    let a = g.spans.entry(name.to_string()).or_default();
    a.ops += 1;
    a.host_us += us;
    if us > a.host_us_max {
        a.host_us_max = us;
    }
}

/// Record one engine-side phase measurement (device time, already in microseconds).
pub fn phase(name: &str, dev_us: u64, count: u64) {
    if !enabled() {
        return;
    }
    let mut g = perf().lock().unwrap();
    let a = g.phases.entry(name.to_string()).or_default();
    a.ops += count.max(1);
    a.dev_us += dev_us;
}

// --------------------------------------------------------------------------- floor

/// Instrumentation floor: time N empty scopes through the same begin/end path, so a
/// reported per-op cost can be read against the measurement's own cost (the example
/// "instrumentation floor" line).
fn measure_floor() -> Floor {
    const N: u64 = 256;
    let saved = enabled();
    ENABLED.store(true, Ordering::Relaxed);
    let t0 = Instant::now();
    for i in 0..N {
        let s = scope(&format!("__floor_{i}"));
        std::hint::black_box(&s);
    }
    let each = t0.elapsed().as_secs_f64() * 1e6 / N as f64;
    ENABLED.store(saved, Ordering::Relaxed);
    // Drop the floor scopes from the trace so they do not pollute the report.
    {
        let mut g = perf().lock().unwrap();
        g.spans.retain(|k, _| !k.starts_with("__floor_"));
    }
    Floor { n: N, host_us_each: each }
}

// --------------------------------------------------------------------------- enable

pub fn set_enabled(on: bool) {
    if on {
        ENABLED.store(true, Ordering::Relaxed);
        start_sampler();
        if perf().lock().unwrap().floor.is_none() {
            let f = measure_floor();
            perf().lock().unwrap().floor = Some(f);
        }
    } else {
        ENABLED.store(false, Ordering::Relaxed);
        stop_sampler();
    }
}

pub fn reset() {
    {
        let mut g = perf().lock().unwrap();
        *g = Perf::new();
    }
    if enabled() {
        let f = measure_floor();
        perf().lock().unwrap().floor = Some(f);
    }
}

fn start_sampler() {
    if SAMPLER_ON.swap(true, Ordering::SeqCst) {
        return;
    }
    let h = std::thread::Builder::new()
        .name("edge0-perf-sampler".into())
        .spawn(|| {
            let mut pdh = pdh::Sampler::new();
            // Prime rate counters, then sample on a 1 s cadence.
            if let Some(s) = &mut pdh {
                s.collect();
            }
            std::thread::sleep(Duration::from_millis(300));
            while SAMPLER_ON.load(Ordering::Relaxed) {
                let res = sample_now(&mut pdh);
                {
                    let mut g = perf().lock().unwrap();
                    if let Some(r) = res {
                        g.last_res = Some(r.clone());
                        g.resources.push(r);
                        if g.resources.len() > 3600 {
                            g.resources.remove(0);
                        }
                    }
                }
                std::thread::sleep(Duration::from_millis(1000));
            }
        })
        .ok();
    *SAMPLER.lock().unwrap() = h;
}

fn stop_sampler() {
    if !SAMPLER_ON.swap(false, Ordering::SeqCst) {
        return;
    }
    if let Some(h) = SAMPLER.lock().unwrap().take() {
        let _ = h.join();
    }
}

/// One resource sample: refresh PDH rate counters, then read CPU/GPU/VRAM; RAM and
/// ReBAR come from kernel32/registry. `pdh` may be None (counters unavailable).
fn sample_now(pdh: &mut Option<pdh::Sampler>) -> Option<ResourceSample> {
    let mut s = ResourceSample { t_ms: perf().lock().unwrap().started_at.elapsed().as_millis() as u64, ..Default::default() };
    s.ram_used_gb = ram_used_gb();
    s.ram_total_gb = ram_total_gb();
    s.rebar = rebar_status();
    if let Some(p) = pdh {
        p.collect();
        s.cpu_pct = p.cpu_pct();
        s.gpu_util_pct = p.gpu_util_pct();
        s.vram_used_gb = p.vram_used_gb();
    }
    s.vram_total_gb = engine_vram_total_gb();
    Some(s)
}

/// VRAM total as the engine reports it (llama-server --list-devices). Cached for the
/// process so a sampling cadence never shells out repeatedly.
fn engine_vram_total_gb() -> Option<f64> {
    static TOTAL: OnceLock<Option<f64>> = OnceLock::new();
    *TOTAL.get_or_init(|| {
        crate::gguf::detect_gpu()
            .ok()
            .flatten()
            .map(|d| d.total_mib as f64 / 1024.0)
    })
}

// --------------------------------------------------------------------------- engine log

/// Fold llama-server's internal phase timings into the trace. Recognised lines are
/// the ones `--perf` / the server's timing printer emit, e.g.:
///   "... load time =  1234.56 ms"
///   "... prompt eval time =  5678.90 ms / 2925 tokens (...)"
///   "... eval time =  1760.11 ms / 48 tokens (...)"
#[derive(Default, Debug, PartialEq)]
struct LogPhases {
    load_us: Option<u64>,
    prompt_us: u64,
    prompt_tok: u64,
    decode_us: u64,
    decode_tok: u64,
}

/// Parse llama-server's timing lines. Kept pure (no globals) so it is directly testable.
fn parse_log(text: &str) -> LogPhases {
    let mut out = LogPhases::default();
    for line in text.lines() {
        let l = line.trim();
        if let Some(rest) = l.split("load time =").nth(1) {
            if let Some(v) = first_f64(rest) {
                out.load_us = Some((v * 1000.0) as u64);
            }
        } else if let Some(rest) = l.split("prompt eval time =").nth(1) {
            if let Some(v) = first_f64(rest) {
                out.prompt_us += (v * 1000.0) as u64;
                out.prompt_tok += tokens_on(l);
            }
        } else if let Some(rest) = l.split("eval time =").nth(1) {
            // "eval time" also matches "prompt eval time"; the prompt branch above
            // already consumed those, so only decode lines reach here.
            if let Some(v) = first_f64(rest) {
                out.decode_us += (v * 1000.0) as u64;
                out.decode_tok += tokens_on(l);
            }
        }
    }
    out
}

pub fn ingest_log(path: &str) {
    if !enabled() {
        return;
    }
    let text = match std::fs::read_to_string(path) {
        Ok(t) => t,
        Err(_) => return,
    };
    let ph = parse_log(&text);
    let mut g = perf().lock().unwrap();
    g.engine_log = Some(path.to_string());
    if let Some(us) = ph.load_us {
        let a = g.phases.entry("engine.load".into()).or_default();
        a.ops = 1;
        a.dev_us = us;
    }
    if ph.prompt_tok > 0 {
        let a = g.phases.entry("engine.prompt-eval".into()).or_default();
        a.ops = ph.prompt_tok;
        a.dev_us = ph.prompt_us;
    }
    if ph.decode_tok > 0 {
        let a = g.phases.entry("engine.decode".into()).or_default();
        a.ops = ph.decode_tok;
        a.dev_us = ph.decode_us;
    }
}

fn first_f64(s: &str) -> Option<f64> {
    let t = s.trim_start();
    let end = t.find(|c: char| !(c.is_ascii_digit() || c == '.' || c == 'e' || c == 'E' || c == '-' || c == '+')).unwrap_or(t.len());
    t[..end].parse::<f64>().ok()
}

/// Token count on a timing line: the "N tokens" immediately after the first "/".
fn tokens_on(line: &str) -> u64 {
    // e.g. "prompt eval time =  11054.38 ms /  2925 tokens ( ... )"
    for seg in line.split('/') {
        let seg = seg.trim();
        let mut it = seg.split_whitespace();
        if let (Some(n), Some(unit)) = (it.next(), it.next()) {
            if unit.starts_with("token") {
                if let Ok(v) = n.parse::<u64>() {
                    return v;
                }
            }
        }
    }
    1
}

// --------------------------------------------------------------------------- report

/// Render the trace as the sherlock-style component table plus a resource summary.
pub fn report_text() -> String {
    let g = perf().lock().unwrap();
    let total_dev: u64 = g.phases.values().map(|p| p.dev_us).sum();
    let mut names: Vec<String> = g.spans.keys().chain(g.phases.keys()).cloned().collect();
    names.sort();
    names.dedup();

    let mut out = String::new();
    out.push_str(&format!(
        "{:<24} {:<7} {:>6} {:>8} {:>13} {:>12} {:>12}\n",
        "component", "scope", "ops", "%dev", "dev us", "idle us", "host us"
    ));
    let mut rows: Vec<(String, String)> = Vec::new();
    for name in &names {
        let sp = g.spans.get(name);
        let ph = g.phases.get(name);
        let scope = match (sp.is_some(), ph.is_some()) {
            (true, true) => "both",
            (true, false) => "app",
            (false, true) => "engine",
            _ => continue,
        };
        let ops = ph.map(|p| p.ops).unwrap_or(0).max(sp.map(|s| s.ops).unwrap_or(0));
        let dev_us = ph.map(|p| p.dev_us).unwrap_or(0) as f64 / ops.max(1) as f64;
        let host_us = sp.map(|s| s.host_us).unwrap_or(0) as f64 / ops.max(1) as f64;
        let idle_us = *g.idle.get(name).unwrap_or(&0) as f64 / ops.max(1) as f64;
        let pct = if total_dev > 0 { ph.map(|p| p.dev_us).unwrap_or(0) as f64 / total_dev as f64 * 100.0 } else { 0.0 };
        let dev_col = if ph.is_some() { format!("{dev_us:.2}") } else { "—".into() };
        let pct_col = if ph.is_some() { format!("{pct:.1}%") } else { "—".into() };
        let row = format!(
            "{:<24} {:<7} {:>6} {:>8} {:>13} {:>12} {:>12}\n",
            name, scope, ops, pct_col, dev_col, format!("{idle_us:.2}"), format!("{host_us:.2}")
        );
        rows.push((name.clone(), row));
    }
    for (_, r) in rows {
        out.push_str(&r);
    }
    if let Some(f) = g.floor {
        out.push_str(&format!(
            "instrumentation floor, {} empty ops timed through the same begin/end path: {:.2} us host each.\n",
            f.n, f.host_us_each
        ));
    }
    if let Some(r) = &g.last_res {
        out.push_str(&format!(
            "resources: cpu {}  ram {}  gpu {}  vram {}  rebar {}\n",
            opt_pct(r.cpu_pct),
            opt_gb(r.ram_used_gb, r.ram_total_gb),
            opt_pct(r.gpu_util_pct),
            opt_gb(r.vram_used_gb, r.vram_total_gb),
            r.rebar.as_ref().map(|(s, _, _)| s.clone()).unwrap_or_else(|| "unavailable".into()),
        ));
    }
    out
}

fn opt_pct(v: Option<f64>) -> String {
    v.map(|x| format!("{x:.1}%")).unwrap_or_else(|| "n/a".into())
}

fn opt_gb(used: Option<f64>, total: Option<f64>) -> String {
    match (used, total) {
        (Some(u), Some(t)) => format!("{u:.1}/{t:.1} GiB"),
        (Some(u), None) => format!("{u:.1} GiB"),
        _ => "n/a".into(),
    }
}

pub fn snapshot_json() -> Value {
    let g = perf().lock().unwrap();
    let total_dev: u64 = g.phases.values().map(|p| p.dev_us).sum();
    let mut names: Vec<String> = g.spans.keys().chain(g.phases.keys()).cloned().collect();
    names.sort();
    names.dedup();
    let components: Vec<Value> = names
        .iter()
        .filter_map(|name| {
            let sp = g.spans.get(name);
            let ph = g.phases.get(name);
            let scope = match (sp.is_some(), ph.is_some()) {
                (true, true) => "both",
                (true, false) => "app",
                (false, true) => "engine",
                _ => return None,
            };
            let ops = ph.map(|p| p.ops).unwrap_or(0).max(sp.map(|s| s.ops).unwrap_or(0));
            Some(json!({
                "name": name,
                "scope": scope,
                "ops": ops,
                "pct_dev": if total_dev > 0 { ph.map(|p| p.dev_us).unwrap_or(0) as f64 / total_dev as f64 * 100.0 } else { 0.0 },
                "dev_us": ph.map(|p| p.dev_us as f64 / ops.max(1) as f64).unwrap_or(0.0),
                "idle_us": *g.idle.get(name).unwrap_or(&0) as f64 / ops.max(1) as f64,
                "host_us": sp.map(|s| s.host_us as f64 / ops.max(1) as f64).unwrap_or(0.0),
                "host_us_max": sp.map(|s| s.host_us_max).unwrap_or(0),
            }))
        })
        .collect();
    let res = res_last_json(&g);
    json!({
        "enabled": enabled(),
        "started_at": g.started_epoch,
        "engine_log": g.engine_log,
        "components": components,
        "resources": res,
        "resource_samples": g.resources.len(),
        "floor": g.floor.map(|f| json!({ "n": f.n, "host_us_each": f.host_us_each })),
        "report": report_text_unlocked(&g),
    })
}

fn res_last_json(g: &Perf) -> Value {
    match &g.last_res {
        Some(r) => json!({
            "t_ms": r.t_ms,
            "cpu_pct": r.cpu_pct,
            "ram_used_gb": r.ram_used_gb,
            "ram_total_gb": r.ram_total_gb,
            "gpu_util_pct": r.gpu_util_pct,
            "vram_used_gb": r.vram_used_gb,
            "vram_total_gb": r.vram_total_gb,
            "rebar": r.rebar.as_ref().map(|(s, v, src)| json!({ "state": s, "value": v, "source": src })),
        }),
        None => Value::Null,
    }
}

/// Same table as report_text but on an already-locked guard (used by snapshot_json to
/// avoid re-locking the non-reentrant mutex).
fn report_text_unlocked(g: &Perf) -> String {
    let total_dev: u64 = g.phases.values().map(|p| p.dev_us).sum();
    let mut names: Vec<String> = g.spans.keys().chain(g.phases.keys()).cloned().collect();
    names.sort();
    names.dedup();
    let mut out = format!(
        "{:<24} {:<7} {:>6} {:>8} {:>13} {:>12} {:>12}\n",
        "component", "scope", "ops", "%dev", "dev us", "idle us", "host us"
    );
    for name in &names {
        let sp = g.spans.get(name);
        let ph = g.phases.get(name);
        let scope = match (sp.is_some(), ph.is_some()) {
            (true, true) => "both",
            (true, false) => "app",
            (false, true) => "engine",
            _ => continue,
        };
        let ops = ph.map(|p| p.ops).unwrap_or(0).max(sp.map(|s| s.ops).unwrap_or(0));
        let dev_us = ph.map(|p| p.dev_us).unwrap_or(0) as f64 / ops.max(1) as f64;
        let host_us = sp.map(|s| s.host_us).unwrap_or(0) as f64 / ops.max(1) as f64;
        let idle_us = *g.idle.get(name).unwrap_or(&0) as f64 / ops.max(1) as f64;
        let pct = if total_dev > 0 { ph.map(|p| p.dev_us).unwrap_or(0) as f64 / total_dev as f64 * 100.0 } else { 0.0 };
        let dev_col = if ph.is_some() { format!("{dev_us:.2}") } else { "—".into() };
        let pct_col = if ph.is_some() { format!("{pct:.1}%") } else { "—".into() };
        out.push_str(&format!(
            "{:<24} {:<7} {:>6} {:>8} {:>13} {:>12} {:>12}\n",
            name, scope, ops, pct_col, dev_col, format!("{idle_us:.2}"), format!("{host_us:.2}")
        ));
    }
    if let Some(f) = g.floor {
        out.push_str(&format!(
            "instrumentation floor, {} empty ops timed through the same begin/end path: {:.2} us host each.\n",
            f.n, f.host_us_each
        ));
    }
    out
}

// --------------------------------------------------------------------------- win32: RAM / registry

#[repr(C)]
struct MemStatusEx {
    dw_length: u32,
    memory_load: u32,
    total_phys: u64,
    avail_phys: u64,
    total_pagefile: u64,
    avail_pagefile: u64,
    total_virtual: u64,
    avail_virtual: u64,
    avail_ext_virtual: u64,
}

#[link(name = "kernel32")]
extern "system" {
    fn GlobalMemoryStatusEx(buf: *mut MemStatusEx) -> i32;
}

fn mem_status() -> Option<MemStatusEx> {
    unsafe {
        let mut m: MemStatusEx = std::mem::zeroed();
        m.dw_length = std::mem::size_of::<MemStatusEx>() as u32;
        if GlobalMemoryStatusEx(&mut m) != 0 {
            Some(m)
        } else {
            None
        }
    }
}

fn ram_total_gb() -> Option<f64> {
    mem_status().map(|m| m.total_phys as f64 / 1_073_741_824.0)
}

fn ram_used_gb() -> Option<f64> {
    mem_status().map(|m| (m.total_phys.saturating_sub(m.avail_phys)) as f64 / 1_073_741_824.0)
}

const HKEY_LOCAL_MACHINE: isize = 0x8000_0002u32 as isize;
const KEY_READ: u32 = 0x2_0019;
const REG_DWORD: u32 = 4;

#[link(name = "advapi32")]
extern "system" {
    fn RegOpenKeyExW(hkey: isize, subkey: *const u16, opt: u32, sam: u32, out: *mut isize) -> i32;
    fn RegQueryValueExW(hkey: isize, name: *const u16, res: *mut u32, ty: *mut u32, data: *mut u8, len: *mut u32) -> i32;
    fn RegCloseKey(hkey: isize) -> i32;
}

fn wstr(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

/// ReBAR state from the AMD kernel-mode driver's control value. On AMD Windows
/// drivers, KMD_RebarControlMode = 1 means Resizable BAR is enabled, 0 disabled.
/// Absent key/value (e.g. non-AMD or an older driver) reports None, never a guess.
fn rebar_status() -> Option<(String, u32, String)> {
    let sub = "SYSTEM\\CurrentControlSet\\Control\\Class\\{4d36e968-e325-11ce-bfc1-08002be10318}\\0000";
    let name = "KMD_RebarControlMode";
    unsafe {
        let mut hk: isize = 0;
        if RegOpenKeyExW(HKEY_LOCAL_MACHINE, wstr(sub).as_ptr(), 0, KEY_READ, &mut hk) != 0 {
            return None;
        }
        let mut ty: u32 = 0;
        let mut data: u32 = 0;
        let mut len: u32 = 4;
        let rc = RegQueryValueExW(hk, wstr(name).as_ptr(), std::ptr::null_mut(), &mut ty, &mut data as *mut u32 as *mut u8, &mut len);
        RegCloseKey(hk);
        if rc != 0 || ty != REG_DWORD {
            return None;
        }
        let state = match data {
            1 => "on",
            0 => "off",
            _ => "unknown",
        };
        Some((state.to_string(), data, format!("HKLM\\{sub}\\{name}")))
    }
}

// --------------------------------------------------------------------------- win32: PDH counters

mod pdh {
    type Handle = *mut core::ffi::c_void;

    const PDH_FMT_DOUBLE: u32 = 0x0000_0200;
    const PDH_MORE_DATA: u32 = 0x8000_07D2;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct FmtValue {
        status: u32,
        _pad: u32,
        value: u64,
    }
    impl FmtValue {
        fn as_f64(&self) -> f64 {
            f64::from_bits(self.value)
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct FmtItem {
        name: *const u16,
        value: FmtValue,
    }

    #[link(name = "pdh")]
    extern "system" {
        fn PdhOpenQueryW(src: *const u16, user: usize, out: *mut Handle) -> u32;
        fn PdhAddEnglishCounterW(q: Handle, path: *const u16, user: usize, out: *mut Handle) -> u32;
        fn PdhCollectQueryData(q: Handle) -> u32;
        fn PdhGetFormattedCounterValue(c: Handle, fmt: u32, ty: *mut u32, v: *mut FmtValue) -> u32;
        fn PdhGetFormattedCounterArrayW(c: Handle, fmt: u32, size: *mut u32, count: *mut u32, buf: *mut FmtItem) -> u32;
        fn PdhCloseQuery(q: Handle) -> u32;
    }

    fn wstr(s: &str) -> Vec<u16> {
        s.encode_utf16().chain(std::iter::once(0)).collect()
    }

    /// Persistent PDH query for the three rate counters. Rate counters need two
    /// collects separated in time; the caller primes and then reads on a cadence.
    pub struct Sampler {
        q: Handle,
        cpu: Handle,
        gpu: Handle,
        vram: Handle,
    }

    // The handles are owned by one sampler thread; sharing the query pointer across
    // threads is not done (each thread creates its own), so the raw pointers are only
    // ever used on their owning thread.
    unsafe impl Send for Sampler {}

    impl Sampler {
        pub fn new() -> Option<Sampler> {
            unsafe {
                let mut q: Handle = std::ptr::null_mut();
                if PdhOpenQueryW(std::ptr::null(), 0, &mut q) != 0 {
                    return None;
                }
                let mut cpu: Handle = std::ptr::null_mut();
                let mut gpu: Handle = std::ptr::null_mut();
                let mut vram: Handle = std::ptr::null_mut();
                let ok_cpu = PdhAddEnglishCounterW(q, wstr("\\Processor Information(_Total)\\% Processor Time").as_ptr(), 0, &mut cpu) == 0;
                let ok_gpu = PdhAddEnglishCounterW(q, wstr("\\GPU Engine(*)\\Utilization Percentage").as_ptr(), 0, &mut gpu) == 0;
                let ok_vram = PdhAddEnglishCounterW(q, wstr("\\GPU Adapter Memory(*)\\Dedicated Usage").as_ptr(), 0, &mut vram) == 0;
                if !ok_cpu && !ok_gpu && !ok_vram {
                    PdhCloseQuery(q);
                    return None;
                }
                Some(Sampler { q, cpu, gpu, vram })
            }
        }

        pub fn collect(&mut self) {
            unsafe {
                PdhCollectQueryData(self.q);
            }
        }

        pub fn cpu_pct(&self) -> Option<f64> {
            if self.cpu.is_null() {
                return None;
            }
            unsafe {
                let mut v = FmtValue { status: 0, _pad: 0, value: 0 };
                if PdhGetFormattedCounterValue(self.cpu, PDH_FMT_DOUBLE, std::ptr::null_mut(), &mut v) == 0 {
                    Some(v.as_f64())
                } else {
                    None
                }
            }
        }

        pub fn gpu_util_pct(&self) -> Option<f64> {
            let items = self.read_array(self.gpu)?;
            // Sum of per-process/per-engine instances, clamped to 100 per the counter's
            // own semantic is wrong; the honest reading is the busiest single engine.
            items.iter().map(|(_, v)| *v).fold(None, |acc, v| Some(acc.map_or(v, |a: f64| a.max(v))))
        }

        pub fn vram_used_gb(&self) -> Option<f64> {
            let items = self.read_array(self.vram)?;
            let bytes: f64 = items.iter().map(|(_, v)| *v).sum();
            if bytes <= 0.0 {
                None
            } else {
                Some(bytes / 1_073_741_824.0)
            }
        }

        fn read_array(&self, c: Handle) -> Option<Vec<(String, f64)>> {
            if c.is_null() {
                return None;
            }
            unsafe {
                let mut size: u32 = 0;
                let mut count: u32 = 0;
                let rc = PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE, &mut size, &mut count, std::ptr::null_mut());
                if rc != PDH_MORE_DATA || size == 0 || count == 0 {
                    return None;
                }
                let mut buf = vec![0u8; size as usize];
                let rc = PdhGetFormattedCounterArrayW(c, PDH_FMT_DOUBLE, &mut size, &mut count, buf.as_mut_ptr() as *mut FmtItem);
                if rc != 0 {
                    return None;
                }
                let items = std::slice::from_raw_parts(buf.as_ptr() as *const FmtItem, count as usize);
                let mut out = Vec::with_capacity(items.len());
                for it in items {
                    if it.value.status != 0 {
                        continue;
                    }
                    let name = if it.name.is_null() { String::new() } else { ptr_to_string(it.name) };
                    out.push((name, it.value.as_f64()));
                }
                Some(out)
            }
        }
    }

    unsafe fn ptr_to_string(p: *const u16) -> String {
        let mut len = 0usize;
        while *p.add(len) != 0 {
            len += 1;
            if len > 4096 {
                break;
            }
        }
        String::from_utf16_lossy(std::slice::from_raw_parts(p, len))
    }
}

// --------------------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scope_noop_when_disabled() {
        set_enabled(false);
        {
            let _s = scope("disabled.span");
        }
        let g = perf().lock().unwrap();
        assert!(!g.spans.contains_key("disabled.span"), "disabled scope must not record");
    }

    #[test]
    fn floor_is_positive_and_clean() {
        let f = measure_floor();
        assert_eq!(f.n, 256);
        assert!(f.host_us_each > 0.0, "floor must be > 0");
        let g = perf().lock().unwrap();
        assert!(!g.spans.keys().any(|k| k.starts_with("__floor")), "floor scopes must be removed");
    }

    #[test]
    fn ram_reads_a_plausible_total() {
        let t = ram_total_gb().expect("GlobalMemoryStatusEx");
        assert!(t > 1.0, "total RAM {t} GiB implausible");
    }

    #[test]
    fn rebar_does_not_panic() {
        let _ = rebar_status();
    }

    #[test]
    fn parse_log_reads_phases() {
        let text = "llama_model_load: load time =  1234.56 ms\n\
                    slot print_timing: prompt eval time =  11054.38 ms /  2925 tokens ( 3.78 ms per token)\n\
                    slot print_timing: eval time =   1760.11 ms /    48 tokens ( 37.45 ms per token)\n";
        let p = parse_log(text);
        assert_eq!(p.load_us, Some(1_234_560));
        assert_eq!(p.prompt_tok, 2925);
        assert_eq!(p.prompt_us, 11_054_380);
        assert_eq!(p.decode_tok, 48);
        assert_eq!(p.decode_us, 1_760_110);
    }

    #[test]
    fn pdh_sampler_reads_cpu_when_available() {
        let mut s = pdh::Sampler::new();
        if let Some(s) = &mut s {
            s.collect();
            std::thread::sleep(Duration::from_millis(300));
            s.collect();
            if let Some(c) = s.cpu_pct() {
                assert!((0.0..=100.5).contains(&c), "cpu pct out of range: {c}");
            }
        }
    }

    #[test]
    fn report_has_header_and_floor() {
        let mut g = perf().lock().unwrap();
        g.spans.entry("unit.span".into()).or_default().ops = 3;
        g.spans.get_mut("unit.span").unwrap().host_us = 300;
        g.floor = Some(Floor { n: 256, host_us_each: 0.5 });
        let t = report_text_unlocked(&g);
        assert!(t.contains("component"));
        assert!(t.contains("unit.span"));
        assert!(t.contains("instrumentation floor"));
    }
}
