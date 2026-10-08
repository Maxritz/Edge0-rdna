// tilelang_bridge.rs — end-to-end test of the ggml-external "custom-op bridge" idea:
// a tilelang-generated HIP kernel wrapped in a C ABI (e0_matmul_f16), loaded at runtime
// from Rust via LoadLibraryA/GetProcAddress (same raw-Win32 pattern as perf.rs, no crates).
//
// The kernel is the tilelang quickstart matmul (f16 in, f32 accumulate, relu) compiled for
// gfx1201. This proves the FFI half: Rust can drive a tilelang kernel. It does NOT prove the
// ggml graph integration (that is C++: op enum + dispatch case + graph-builder wiring).
//
//   set PATH=G:\ROCM10RT-gfx1201\bin;%PATH%
//   cargo run --example tilelang_bridge
// Env: EDGE0_TILELANG_DLL overrides the DLL path.

use std::ffi::{c_void, CString};
use std::time::Instant;

#[cfg(windows)]
#[link(name = "kernel32")]
extern "system" {
    fn LoadLibraryA(name: *const u8) -> *mut c_void;
    fn GetProcAddress(h: *mut c_void, name: *const u8) -> *mut c_void;
}

type MatmulFn = unsafe extern "C" fn(*const c_void, *const c_void, *mut c_void, i32, i32, i32) -> i32;
type NameFn = unsafe extern "C" fn() -> *const i8;

fn f32_to_f16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xff) as i32;
    let mant = bits & 0x007f_ffff;
    if exp == 0xff {
        return sign | 0x7c00 | if mant != 0 { 0x200 } else { 0 };
    }
    let mut e = exp - 127 + 15;
    if e <= 0 {
        if e < -10 {
            return sign;
        }
        let m = mant | 0x0080_0000;
        let shift = (14 - e) as u32;
        let mut half = (m >> shift) as u16;
        let rem = m & ((1u32 << shift) - 1);
        let mid = 1u32 << (shift - 1);
        if rem > mid || (rem == mid && (half & 1) == 1) {
            half += 1;
        }
        return sign | half;
    }
    if e >= 0x1f {
        return sign | 0x7c00;
    }
    let mut m = (mant >> 13) as u16;
    let rem = mant & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (m & 1) == 1) {
        m += 1;
        if m == 0x400 {
            m = 0;
            e += 1;
            if e >= 0x1f {
                return sign | 0x7c00;
            }
        }
    }
    sign | ((e as u16) << 10) | m
}

fn f16_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let mant = (h & 0x3ff) as u32;
    let bits = if exp == 0 {
        if mant == 0 {
            sign
        } else {
            let mut e = 127 - 15 + 1i32;
            let mut m = mant;
            while (m & 0x400) == 0 {
                m <<= 1;
                e -= 1;
            }
            m &= 0x3ff;
            sign | ((e as u32) << 23) | (m << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (mant << 13)
    } else {
        sign | ((exp + 127 - 15) << 23) | (mant << 13)
    };
    f32::from_bits(bits)
}

fn main() {
    #[cfg(not(windows))]
    {
        eprintln!("tilelang_bridge is Windows-only (LoadLibraryA)");
        return;
    }

    #[cfg(windows)]
    unsafe {
        let dll = std::env::var("EDGE0_TILELANG_DLL")
            .unwrap_or_else(|_| r"C:\Users\rr\AppData\Local\Temp\opencode\e0_tilelang_bridge.dll".to_string());
        let cpath = CString::new(dll.clone()).unwrap();
        let handle = LoadLibraryA(cpath.as_ptr() as *const u8);
        if handle.is_null() {
            eprintln!("FAIL: LoadLibraryA({dll}) returned null (is amdhip64_7.dll on PATH?)");
            std::process::exit(1);
        }

        let sym = |n: &str| -> *mut c_void {
            let c = CString::new(n).unwrap();
            GetProcAddress(handle, c.as_ptr() as *const u8)
        };
        let matmul_p = sym("e0_matmul_f16");
        let name_p = sym("e0_hip_device_name");
        if matmul_p.is_null() {
            eprintln!("FAIL: e0_matmul_f16 not found in {dll}");
            std::process::exit(1);
        }
        let matmul: MatmulFn = std::mem::transmute(matmul_p);
        if !name_p.is_null() {
            let name: NameFn = std::mem::transmute(name_p);
            let p = name();
            if !p.is_null() {
                let mut s = String::new();
                let mut i = 0isize;
                while *p.offset(i) != 0 {
                    s.push(*p.offset(i) as u8 as char);
                    i += 1;
                }
                println!("HIP device reported by bridge: {s}");
            }
        }

        let (m, n, k) = (1024usize, 1024usize, 1024usize);

        // Exact f16-representable inputs so the reference and the kernel read identical values.
        let table = [-2.0f32, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 0.25];
        let mut state = 2463534242u32;
        let mut next = || {
            state ^= state << 13;
            state ^= state >> 17;
            state ^= state << 5;
            table[(state % table.len() as u32) as usize]
        };
        let a_f: Vec<f32> = (0..m * k).map(|_| next()).collect();
        let b_f: Vec<f32> = (0..k * n).map(|_| next()).collect();
        let a_h: Vec<u16> = a_f.iter().map(|&x| f32_to_f16(x)).collect();
        let b_h: Vec<u16> = b_f.iter().map(|&x| f32_to_f16(x)).collect();
        let mut c_h: Vec<u16> = vec![0u16; m * n];

        let rc = matmul(
            a_h.as_ptr() as *const c_void,
            b_h.as_ptr() as *const c_void,
            c_h.as_mut_ptr() as *mut c_void,
            m as i32,
            n as i32,
            k as i32,
        );
        if rc != 0 {
            eprintln!("FAIL: e0_matmul_f16 returned {rc}");
            std::process::exit(1);
        }

        // Reference: relu(A@B) in f32 over the same f16 values.
        let mut max_abs = 0.0f32;
        let mut max_rel = 0.0f32;
        let mut nonzero = 0usize;
        for i in 0..m {
            for j in 0..n {
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += f16_to_f32(a_h[i * k + kk]) * f16_to_f32(b_h[kk * n + j]);
                }
                let r = acc.max(0.0);
                let got = f16_to_f32(c_h[i * n + j]);
                if got != 0.0 {
                    nonzero += 1;
                }
                let d = (got - r).abs();
                if d > max_abs {
                    max_abs = d;
                }
                let rel = if r.abs() > 1.0 { d / r.abs() } else { d };
                if rel > max_rel {
                    max_rel = rel;
                }
            }
        }

        // Timed re-run (includes H2D + kernel + D2H).
        let t0 = Instant::now();
        let _ = matmul(
            a_h.as_ptr() as *const c_void,
            b_h.as_ptr() as *const c_void,
            c_h.as_mut_ptr() as *mut c_void,
            m as i32,
            n as i32,
            k as i32,
        );
        let dt = t0.elapsed().as_secs_f64();
        let gflops = 2.0 * (m * n * k) as f64 / dt / 1e9;

        println!("shape {m}x{n}x{k}, f16 in / f32 accum / relu, gfx1201 WMMA");
        println!("nonzero outputs: {nonzero} / {}", m * n);
        println!("max abs err: {max_abs:.3e}   max rel err: {max_rel:.3e}");
        println!("call (H2D+kernel+D2H): {:.3} ms  ~{gflops:.1} GFLOP/s", dt * 1e3);

        let ok = nonzero > 0 && max_rel < 2e-2 && max_abs < 1e-1;
        println!("BRIDGE TEST: {}", if ok { "PASS" } else { "FAIL" });
        if !ok {
            std::process::exit(1);
        }
    }
}
