// split_gemm.rs — split-precision FP32 GEMM (tilelang WMMA, gfx1201) tested from Rust.
// Validates fp32 accuracy (split: hi+lo fp16, 3 tensor-core MMAs) vs a plain fp16 GEMM,
// and benchmarks TFLOPS. A/B target: ggml ROCm0 f32 GEMM 12.44 TF, f16 96 TF
// (test-backend-ops, m=4096 n=512 k=14336).
//
//   set PATH=G:\ROCM10RT-gfx1201\bin;%PATH%
//   cargo run --example split_gemm --release
use std::ffi::{c_void, CString};
use std::time::Instant;

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryA(name: *const u8) -> *mut c_void;
    fn GetProcAddress(h: *mut c_void, name: *const u8) -> *mut c_void;
}

type GemmFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, i32, i32, i32) -> i32;

fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff { return s | 0x7c00; }
    let mut e = e - 127 + 15;
    if e <= 0 {
        if e < -10 { return s; }
        let mm = m | 0x80_0000;
        let sh = (14 - e) as u32;
        return s | (mm >> sh) as u16;
    }
    if e >= 0x1f { return s | 0x7c00; }
    s | ((e as u16) << 10) | ((m >> 13) as u16)
}
fn f16_to_f32(h: u16) -> f32 {
    let s = ((h & 0x8000) as u32) << 16;
    let e = ((h >> 10) & 0x1f) as u32;
    let m = (h & 0x3ff) as u32;
    let bits = if e == 0 {
        if m == 0 { s } else { let mut e = 127 - 15 + 1i32; let mut mm = m; while mm & 0x400 == 0 { mm <<= 1; e -= 1; } s | ((e as u32) << 23) | ((mm & 0x3ff) << 13) }
    } else if e == 0x1f { s | 0x7f80_0000 | (m << 13) } else { s | ((e + 127 - 15) << 23) | (m << 13) };
    f32::from_bits(bits)
}

fn main() {
    #[cfg(not(windows))] { eprintln!("windows-only"); return; }
    #[cfg(windows)]
    unsafe {
        let dll = std::env::var("EDGE0_SPLIT_DLL")
            .unwrap_or_else(|_| r"C:\Users\rr\AppData\Local\Temp\opencode\e0_split_gemm.dll".to_string());
        let c = CString::new(dll.clone()).unwrap();
        let h = LoadLibraryA(c.as_ptr() as *const u8);
        if h.is_null() { eprintln!("FAIL: LoadLibraryA({dll}) null (amdhip64_7.dll on PATH?)"); std::process::exit(1); }
        let sym = |n: &str| -> *mut c_void { let c = CString::new(n).unwrap(); GetProcAddress(h, c.as_ptr() as *const u8) };
        let p = sym("e0_split_gemm_f32");
        if p.is_null() { eprintln!("FAIL: e0_split_gemm_f32 not found"); std::process::exit(1); }
        let gemm: GemmFn = std::mem::transmute(p);

        let m = 512; let n = 512; let k = 512; // small for the f64 reference; %128/%32 ok

        let mut st = 0x12345678u32;
        let mut nxt = || { st ^= st << 13; st ^= st >> 17; st ^= st << 5; (st as f32 / u32::MAX as f32) * 2.0 - 1.0 };
        let a: Vec<f32> = (0..m * k).map(|_| nxt()).collect();
        let b: Vec<f32> = (0..k * n).map(|_| nxt()).collect();
        let mut c: Vec<f32> = vec![0.0; m * n];

        let rc = gemm(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), m as i32, n as i32, k as i32);
        if rc != 0 { eprintln!("FAIL: e0_split_gemm_f32 rc={rc}"); std::process::exit(1); }

        // f64 reference + plain-fp16 simulation.
        let mut split_abs = 0f64; let mut split_rel = 0f64;
        let mut f16_abs = 0f64;
        let a16: Vec<u16> = a.iter().map(|&x| f32_to_f16(x)).collect();
        let b16: Vec<u16> = b.iter().map(|&x| f32_to_f16(x)).collect();
        for i in 0..m {
            for j in 0..n {
                let mut ref_acc = 0f64; let mut f16_acc = 0f64;
                for kk in 0..k {
                    ref_acc += a[i * k + kk] as f64 * b[kk * n + j] as f64;
                    f16_acc += f16_to_f32(a16[i * k + kk]) as f64 * f16_to_f32(b16[kk * n + j]) as f64;
                }
                let got = c[i * n + j] as f64;
                let d = (got - ref_acc).abs();
                if d > split_abs { split_abs = d; }
                let r = if ref_acc.abs() > 1e-3 { d / ref_acc.abs() } else { d };
                if r > split_rel { split_rel = r; }
                let df = (f16_acc - ref_acc).abs();
                if df > f16_abs { f16_abs = df; }
            }
        }

        println!("split-precision fp32 GEMM  {m}x{n}x{k}");
        println!("  split  max abs err vs f64: {split_abs:.3e}   max rel: {split_rel:.3e}");
        println!("  plain  max abs err vs f64: {f16_abs:.3e}  (fp16 baseline)");
        let acc_ok = split_abs < 1e-3 && split_abs < f16_abs * 0.1;

        // Benchmark at a GEMM-sized shape (prefill-like), grid-aligned.
        let (bm, bn, bk) = (1024usize, 1024usize, 2048usize);
        let mut sa = 0x9e3779b9u32;
        let mut n2 = || { sa ^= sa << 13; sa ^= sa >> 17; sa ^= sa << 5; (sa as f32 / u32::MAX as f32) - 0.5 };
        let ab: Vec<f32> = (0..bm * bk).map(|_| n2()).collect();
        let bb: Vec<f32> = (0..bk * bn).map(|_| n2()).collect();
        let mut cb: Vec<f32> = vec![0.0; bm * bn];
        let _ = gemm(ab.as_ptr(), bb.as_ptr(), cb.as_mut_ptr(), bm as i32, bn as i32, bk as i32);
        let reps = 20;
        let t0 = Instant::now();
        for _ in 0..reps {
            let _ = gemm(ab.as_ptr(), bb.as_ptr(), cb.as_mut_ptr(), bm as i32, bn as i32, bk as i32);
        }
        let dt = t0.elapsed().as_secs_f64() / reps as f64;
        let tf = 2.0 * (bm * bn * bk) as f64 / dt / 1e12;
        println!("  bench {bm}x{bn}x{bk} (incl H2D+D2H): {:.3} ms  ~{tf:.1} TFLOP/s", dt * 1e3);
        println!("  A/B ggml ROCm0: f32 12.4 TF, f16 96.3 TF");
        println!("SPLIT GEMM: {}", if acc_ok { "PASS (fp32-accurate)" } else { "CHECK ACCURACY" });
        if !acc_ok { std::process::exit(1); }
    }
}
