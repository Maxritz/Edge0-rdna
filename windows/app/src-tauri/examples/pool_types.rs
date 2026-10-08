// pool_types.rs — validates the pool's tt_bytes fix against the real ggml library.
// Loads ggml-hip.dll and reads ggml_type_size / ggml_blck_size for the expert types
// used by the target models, then computes per-expert bytes for their exact shapes.
// This is the fast check of the fix's arithmetic (the resolver requires the computed
// per-expert stride to equal the engine's nb02 exactly).
//
//   set PATH=G:\ROCM10RT-gfx1201\bin;%PATH%
//   cargo run --example pool_types
use std::ffi::{c_void, CString};

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryA(name: *const u8) -> *mut c_void;
    fn GetProcAddress(h: *mut c_void, name: *const u8) -> *mut c_void;
}

type TypeSizeFn = unsafe extern "C" fn(i32) -> usize;
type BlckSizeFn = unsafe extern "C" fn(i32) -> i64;

fn main() {
    #[cfg(not(windows))]
    { eprintln!("windows-only"); }

    #[cfg(windows)]
    unsafe {
        let dll = std::env::var("EDGE0_GGML_DLL").unwrap_or_else(|_|
            r"C:\Users\rr\OneDrive\Desktop\Edge0-rdna\wt\win\build-hip\bin\ggml-base.dll".to_string());
        let c = CString::new(dll.clone()).unwrap();
        let h = LoadLibraryA(c.as_ptr() as *const u8);
        if h.is_null() {
            eprintln!("FAIL: LoadLibraryA({dll}) null");
            std::process::exit(1);
        }
        let sym = |n: &str| -> *mut c_void {
            let c = CString::new(n).unwrap();
            GetProcAddress(h, c.as_ptr() as *const u8)
        };
        let ts_p = sym("ggml_type_size");
        let bs_p = sym("ggml_blck_size");
        if ts_p.is_null() || bs_p.is_null() {
            eprintln!("FAIL: ggml_type_size / ggml_blck_size not exported by {dll}");
            std::process::exit(1);
        }
        let type_size: TypeSizeFn = std::mem::transmute(ts_p);
        let blck_size: BlckSizeFn = std::mem::transmute(bs_p);

        // Known-good ggml values (bytes per block / elements per block).
        let expect: &[(i32, &str, usize, i64)] = &[
            (0,  "F32",    4,  1),
            (1,  "F16",    2,  1),
            (2,  "BF16",   18, 32),
            (3,  "Q4_1",   20, 32),
            (8,  "Q8_0",   34, 32),
            (10, "Q2_K",   84, 256),
            (12, "Q4_K",   144, 256),
            (13, "Q5_K",   176, 256),
            (14, "Q6_K",   210, 256),
            (16, "IQ2_XXS",66, 256),
            (39, "MXFP4",  17, 32),
            (40, "NVFP4",  36, 64),
        ];

        println!("{:>4} {:<8} {:>6} {:>6} {:>10} {:>10}  {}", "type", "name", "size", "blck", "bits/wt", "match", "old tt_bytes");
        let mut all_ok = true;
        for &(t, name, es, bs) in expect {
            let gs = type_size(t);
            let gb = blck_size(t);
            let bpw = gs as f64 * 8.0 / gb as f64;
            let ok = gs == es && gb == bs;
            if !ok { all_ok = false; }
            // old hand table: only F32/F16/BF16/Q4_1 returned nonzero
            let old = match t { 0 => "ok", 1 => "ok", 2 => "ok", 3 => "ok", _ => "SKIPPED (0)" };
            println!("{:>4} {:<8} {:>6} {:>6} {:>10.3} {:>10}  {}", t, name, gs, gb, bpw,
                     if ok { "OK" } else { "MISMATCH" }, old);
        }

        // Per-expert bytes for the target models' expert shapes (contiguous 3D tensor).
        // per_expert = type_size * (prod(dims) / blck) / n_experts ; must be an exact integer.
        let shapes: &[(i32, &str, &[u64])] = &[
            (39, "35B gate/up  [2048,512,256]",  &[2048, 512, 256]),
            (39, "35B down      [512,2048,256]", &[512, 2048, 256]),
            (14, "35B Q6_K down [512,2048,256]", &[512, 2048, 256]),
            (39, "gpt-oss gate  [2880,2880,128]", &[2880, 2880, 128]),
            (16, "DSV4 gate/up  [4096,2048,256]", &[4096, 2048, 256]),
            (10, "DSV4 down     [2048,4096,256]", &[2048, 4096, 256]),
            (12, "Qwen3.8 gate  [4096,2048,288]", &[4096, 2048, 288]),
        ];
        println!("\n{:>4} {:<30} {:>12} {:>12} {:>8}", "type", "shape", "tensor MiB", "per-expert B", "exact?");
        for &(t, label, dims) in shapes {
            let es = type_size(t) as u64;
            let bs = blck_size(t) as u64;
            let n: u64 = dims.iter().product();
            let nexps = dims[2];
            if n % bs != 0 {
                println!("{:>4} {:<30} {:>12} {:>12} {:>8}", t, label, "-", "-", "DIVISOR!");
                all_ok = false;
                continue;
            }
            let total = es * (n / bs);
            let per = total / nexps;
            let exact = total % nexps == 0;
            if !exact { all_ok = false; }
            println!("{:>4} {:<30} {:>12.3} {:>12} {:>8}", t, label,
                     total as f64 / (1024.0 * 1024.0), per, if exact { "yes" } else { "NO" });
        }

        println!("\nPOOL TYPE-SIZE TEST: {}", if all_ok { "PASS" } else { "FAIL" });
        if !all_ok { std::process::exit(1); }
    }
}
