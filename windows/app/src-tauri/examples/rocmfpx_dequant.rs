// rocmfpx_dequant.rs — reference decoders + round-trip tests for the ROCmFPX
// quant formats whose recipes were captured from charlie12345/ROCmFPX
// (ggml/rocmfp4/, ggml/rocmfpx/, ggml/src/ggml-quants.c). Pure-Rust, no crates,
// no GPU. Validates the block layouts, codebooks and bit-packing by
// quantize -> dequantize round trips.
//
//   cargo run --release --example rocmfpx_dequant
//
// Formats: fp2, fp3, fp4, fp4-fast, fp6, fp8 (ROCmFPX) and turbo3/turbo4
// (TurboQuant KV cache, FWHT-rotated). ggml type ids: fp4=100, fp4_fast=101,
// fp6=102, fp8=103, fp3=104, turbo3=105, turbo4=106, fp2=107.

// ---- fp16 <-> fp32 (manual; no half crate) ----
fn f32_to_f16(x: f32) -> u16 {
    let b = x.to_bits();
    let s = ((b >> 16) & 0x8000) as u16;
    let e = ((b >> 23) & 0xff) as i32;
    let m = b & 0x7f_ffff;
    if e == 0xff { return s | 0x7c00 | if m != 0 { 0x200 } else { 0 }; }
    let mut e = e - 127 + 15;
    if e <= 0 {
        if e < -10 { return s; }
        let mm = m | 0x80_0000;
        let sh = (14 - e) as u32;
        let mut h = (mm >> sh) as u16;
        let rem = mm & ((1u32 << sh) - 1);
        let mid = 1u32 << (sh - 1);
        if rem > mid || (rem == mid && (h & 1) == 1) { h += 1; }
        return s | h;
    }
    if e >= 0x1f { return s | 0x7c00; }
    let mut mm = (m >> 13) as u16;
    let rem = m & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (mm & 1) == 1) {
        mm += 1;
        if mm == 0x400 { mm = 0; e += 1; if e >= 0x1f { return s | 0x7c00; } }
    }
    s | ((e as u16) << 10) | mm
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

// ---- UE4M3 scale (finite unsigned E4M3): bytes 0x00..0x7e ----
fn ue4m3_decode(e: u8) -> f32 {
    if e > 0x7e { return 0.0; }
    let exp = (e >> 3) as i32;
    let mant = (e & 7) as f32;
    if exp == 0 { mant * 2f32.powi(-10) } else { (8.0 + mant) * 2f32.powi(exp - 11) }
}
fn ue4m3_nearest(target: f32) -> u8 {
    if !(target > 0.0) || !target.is_finite() { return 0; }
    let mut best = 1u8; let mut bd = (ue4m3_decode(1) - target).abs();
    for e in 1u8..=0x7e {
        let d = (ue4m3_decode(e) - target).abs();
        if d < bd { bd = d; best = e; }
    }
    best
}

// ---- codebooks ----
const FP2_CB: [f32; 4] = [-4.0, -1.0, 1.0, 4.0];
const FP4_CB: [i32; 16] = [0, 1, 2, 3, 4, 6, 8, 10, 0, -1, -2, -3, -4, -6, -8, -10];
const FP3_MAG: [i32; 4] = [0, 1, 2, 4];
const TURBO3_CB: [f32; 8] = [-0.1883972972, -0.1181399059, -0.0665857641, -0.0216044751, 0.0216041461, 0.0665854520, 0.1181396281, 0.1883970748];
const TURBO4_CB: [f32; 16] = [-0.2376389871, -0.1808080141, -0.1417777640, -0.1102646123, -0.0828112376, -0.0577640422, -0.0341540905, -0.0113168380, 0.0112761586, 0.0341139667, 0.0577250301, 0.0827738972, 0.1102295202, 0.1417455465, 0.1807794468, 0.2376153882];

fn fp4_decode(q: u8) -> i32 {
    let q = q & 0x0f;
    let m3 = (q & 0x07) as i32;
    let mag = if m3 <= 4 { m3 } else { 2 * m3 - 4 };
    if q & 0x08 != 0 { -mag } else { mag }
}
fn fp3_decode(code: u8) -> i32 {
    let v = FP3_MAG[(code & 3) as usize];
    if code & 4 != 0 { -v } else { v }
}
fn fp6_decode(code: u8) -> i32 {
    let mag = (code & 31) as i32;
    if code & 32 != 0 { -(if mag == 0 { 32 } else { mag }) } else { mag }
}
fn nearest_f32(cb: &[f32], x: f32) -> u8 {
    let mut bi = 0u8; let mut bd = (cb[0] - x).abs();
    for (i, &c) in cb.iter().enumerate() { let d = (c - x).abs(); if d < bd { bd = d; bi = i as u8; } }
    bi
}
fn nearest_i32(cb: &[i32], x: i32) -> u8 {
    let mut bi = 0u8; let mut bd = (cb[0] - x).abs();
    for (i, &c) in cb.iter().enumerate() { let d = (c - x).abs(); if d < bd { bd = d; bi = i as u8; } }
    bi
}
fn fp3_quant_code(x: f32, inv_scale: f32) -> u8 {
    let a = (x * inv_scale).abs();
    let mag = if a <= 0.5 { 0 } else if a <= 1.5 { 1 } else if a <= 3.0 { 2 } else { 3 };
    if mag == 0 { 0 } else { if x < 0.0 { 4 | mag } else { mag } }
}
fn fp6_quant_code(x: f32, inv_scale: f32) -> u8 {
    let mut q = (x * inv_scale).round() as i32;
    if q > 31 { q = 31; } else if q < -32 { q = -32; }
    if q == 0 { 0 } else if q < 0 { 32u8 | ((-q) as u8 & 31) } else { q as u8 }
}

// ---- pack helpers (verbatim from rocmfpx.c / ggml-quants.c) ----
fn fp3_pack8(dst: &mut [u8], c: &[u8]) {
    dst[0] = (c[0] & 7) | ((c[1] & 7) << 3) | ((c[2] & 3) << 6);
    dst[1] = ((c[2] >> 2) & 1) | ((c[3] & 7) << 1) | ((c[4] & 7) << 4) | ((c[5] & 1) << 7);
    dst[2] = ((c[5] >> 1) & 3) | ((c[6] & 7) << 2) | ((c[7] & 7) << 5);
}
fn fp3_unpack8(src: &[u8], c: &mut [u8]) {
    c[0] = src[0] & 7;
    c[1] = (src[0] >> 3) & 7;
    c[2] = ((src[0] >> 6) & 3) | ((src[1] & 1) << 2);
    c[3] = (src[1] >> 1) & 7;
    c[4] = (src[1] >> 4) & 7;
    c[5] = ((src[1] >> 7) & 1) | ((src[2] & 3) << 1);
    c[6] = (src[2] >> 2) & 7;
    c[7] = (src[2] >> 5) & 7;
}
fn fp6_pack4(dst: &mut [u8], c: &[u8]) {
    dst[0] = (c[0] & 0x3f) | ((c[1] & 0x03) << 6);
    dst[1] = ((c[1] >> 2) & 0x0f) | ((c[2] & 0x0f) << 4);
    dst[2] = ((c[2] >> 4) & 0x03) | ((c[3] & 0x3f) << 2);
}
fn fp6_unpack4(src: &[u8], c: &mut [u8]) {
    c[0] = src[0] & 0x3f;
    c[1] = ((src[0] >> 6) & 0x03) | ((src[1] & 0x0f) << 2);
    c[2] = ((src[1] >> 4) & 0x0f) | ((src[2] & 0x03) << 4);
    c[3] = (src[2] >> 2) & 0x3f;
}

// ---- generic block quantize/dequant (scale chosen as nearest UE4M3 of max/levelmax) ----
// Returns (bytes, scale) for one 32-value block.
fn q_fp2(x: &[f32]) -> (Vec<u8>, f32) {
    let mut out = Vec::new();
    let mut gmax = 0f32;
    for xb in x.chunks_exact(32) {
        let mut blk = vec![0u8; 10];
        for half in 0..2 {
            let xs = &xb[half * 16..half * 16 + 16];
            let maxa = xs.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let e = ue4m3_nearest(if maxa > 0.0 { maxa / 4.0 } else { 0.0 });
            blk[8 + half] = e; gmax = gmax.max(ue4m3_decode(e));
            let s = ue4m3_decode(e);
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            for j in 0..16 {
                let code = nearest_f32(&FP2_CB, xs[j] * inv);
                blk[half * 4 + j / 4] |= code << (2 * (j % 4));
            }
        }
        out.extend_from_slice(&blk);
    }
    (out, gmax)
}
fn d_fp2(b: &[u8], y: &mut [f32]) {
    for (ib, yb) in y.chunks_exact_mut(32).enumerate() {
        let blk = &b[ib * 10..ib * 10 + 10];
        for half in 0..2 {
            let s = ue4m3_decode(blk[8 + half]);
            for j in 0..16 {
                let code = (blk[half * 4 + j / 4] >> (2 * (j % 4))) & 3;
                yb[half * 16 + j] = FP2_CB[code as usize] * s;
            }
        }
    }
}
fn q_fp4(x: &[f32], fast: bool) -> (Vec<u8>, f32) {
    let nscale = if fast { 1 } else { 2 };
    let bytes = if fast { 17 } else { 18 };
    let mut out = Vec::new();
    let mut gmax = 0f32;
    for xb in x.chunks_exact(32) {
        let mut blk = vec![0u8; bytes];
        if fast {
            let maxa = xb.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let e = ue4m3_nearest(if maxa > 0.0 { maxa / 10.0 } else { 0.0 });
            let s = ue4m3_decode(e); gmax = gmax.max(s); blk[16] = e;
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            for j in 0..16 {
                let lo = nearest_i32(&FP4_CB, (xb[j] * inv).round() as i32);
                let hi = nearest_i32(&FP4_CB, (xb[j + 16] * inv).round() as i32);
                blk[j] = lo | (hi << 4);
            }
        } else {
            for half in 0..nscale {
                let xs = &xb[half * 16..half * 16 + 16];
                let maxa = xs.iter().fold(0f32, |a, &v| a.max(v.abs()));
                let e = ue4m3_nearest(if maxa > 0.0 { maxa / 10.0 } else { 0.0 });
                let s = ue4m3_decode(e); gmax = gmax.max(s); blk[16 + half] = e;
                let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
                for j in 0..16 {
                    let code = nearest_i32(&FP4_CB, (xs[j] * inv).round() as i32);
                    blk[j] |= code << (4 * half);
                }
            }
        }
        out.extend_from_slice(&blk);
    }
    (out, gmax)
}
fn d_fp4(b: &[u8], y: &mut [f32], fast: bool) {
    let bytes = if fast { 17 } else { 18 };
    for (ib, yb) in y.chunks_exact_mut(32).enumerate() {
        let blk = &b[ib * bytes..ib * bytes + bytes];
        let d0 = ue4m3_decode(blk[16]);
        let d1 = if fast { d0 } else { ue4m3_decode(blk[17]) };
        for j in 0..16 {
            yb[j] = fp4_decode(blk[j] & 0x0f) as f32 * d0;
            yb[j + 16] = fp4_decode(blk[j] >> 4) as f32 * d1;
        }
    }
}
fn q_fp3(x: &[f32]) -> (Vec<u8>, f32) {
    let mut out = Vec::new();
    let mut gmax = 0f32;
    for xb in x.chunks_exact(32) {
        let mut blk = vec![0u8; 14];
        for half in 0..2 {
            let xs = &xb[half * 16..half * 16 + 16];
            let maxa = xs.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let e = ue4m3_nearest(if maxa > 0.0 { maxa / 4.0 } else { 0.0 });
            blk[12 + half] = e; gmax = gmax.max(ue4m3_decode(e));
            let s = ue4m3_decode(e);
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            let mut codes = [0u8; 16];
            for j in 0..16 { codes[j] = fp3_quant_code(xs[j], inv); }
            let off = half * 6;
            fp3_pack8(&mut blk[off..off + 3], &codes[0..8]);
            fp3_pack8(&mut blk[off + 3..off + 6], &codes[8..16]);
        }
        out.extend_from_slice(&blk);
    }
    (out, gmax)
}
fn d_fp3(b: &[u8], y: &mut [f32]) {
    for (ib, yb) in y.chunks_exact_mut(32).enumerate() {
        let blk = &b[ib * 14..ib * 14 + 14];
        let mut codes = [0u8; 32];
        fp3_unpack8(&blk[0..3], &mut codes[0..8]);
        fp3_unpack8(&blk[3..6], &mut codes[8..16]);
        fp3_unpack8(&blk[6..9], &mut codes[16..24]);
        fp3_unpack8(&blk[9..12], &mut codes[24..32]);
        for half in 0..2 {
            let s = ue4m3_decode(blk[12 + half]);
            for j in 0..16 { yb[half * 16 + j] = fp3_decode(codes[half * 16 + j]) as f32 * s; }
        }
    }
}
fn q_fp6(x: &[f32]) -> (Vec<u8>, f32) {
    let mut out = Vec::new();
    let mut gmax = 0f32;
    for xb in x.chunks_exact(32) {
        let mut blk = vec![0u8; 26];
        for half in 0..2 {
            let xs = &xb[half * 16..half * 16 + 16];
            let maxa = xs.iter().fold(0f32, |a, &v| a.max(v.abs()));
            let e = ue4m3_nearest(if maxa > 0.0 { maxa / 31.0 } else { 0.0 });
            blk[24 + half] = e; gmax = gmax.max(ue4m3_decode(e));
            let s = ue4m3_decode(e);
            let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
            let mut codes = [0u8; 16];
            for j in 0..16 { codes[j] = fp6_quant_code(xs[j], inv); }
            let off = half * 12;
            for g in 0..4 { fp6_pack4(&mut blk[off + g * 3..off + g * 3 + 3], &codes[g * 4..g * 4 + 4]); }
        }
        out.extend_from_slice(&blk);
    }
    (out, gmax)
}
fn d_fp6(b: &[u8], y: &mut [f32]) {
    for (ib, yb) in y.chunks_exact_mut(32).enumerate() {
        let blk = &b[ib * 26..ib * 26 + 26];
        let mut codes = [0u8; 32];
        for g in 0..8 { fp6_unpack4(&blk[g * 3..g * 3 + 3], &mut codes[g * 4..g * 4 + 4]); }
        for half in 0..2 {
            let s = ue4m3_decode(blk[24 + half]);
            for j in 0..16 { yb[half * 16 + j] = fp6_decode(codes[half * 16 + j]) as f32 * s; }
        }
    }
}
fn q_fp8(x: &[f32]) -> (Vec<u8>, f32) {
    let mut out = Vec::new();
    let mut gmax = 0f32;
    for xb in x.chunks_exact(32) {
        let mut blk = vec![0u8; 33];
        let maxa = xb.iter().fold(0f32, |a, &v| a.max(v.abs()));
        let e = ue4m3_nearest(if maxa > 0.0 { maxa / 127.0 } else { 0.0 });
        let s = ue4m3_decode(e); gmax = gmax.max(s); blk[32] = e;
        let inv = if s > 0.0 { 1.0 / s } else { 0.0 };
        for i in 0..32 { let mut q = (xb[i] * inv).round() as i32; if q > 127 { q = 127 } else if q < -127 { q = -127 } blk[i] = q as i8 as u8; }
        out.extend_from_slice(&blk);
    }
    (out, gmax)
}
fn d_fp8(b: &[u8], y: &mut [f32]) {
    for (ib, yb) in y.chunks_exact_mut(32).enumerate() {
        let blk = &b[ib * 33..ib * 33 + 33];
        let s = ue4m3_decode(blk[32]);
        for i in 0..32 { yb[i] = (blk[i] as i8) as f32 * s; }
    }
}

// ---- TurboQuant (FWHT rotated, 128-element chunks) ----
fn fwht(x: &mut [f32]) {
    let n = x.len();
    let mut h = 1;
    while h < n {
        let mut i = 0;
        while i < n {
            for j in i..i + h { let a = x[j]; let b = x[j + h]; x[j] = a + b; x[j + h] = a - b; }
            i += 2 * h;
        }
        h *= 2;
    }
    let s = 1.0 / (n as f32).sqrt();
    for v in x.iter_mut() { *v *= s; }
}
fn turbo_pack3(idx: &[u8]) -> [u8; 12] {
    let mut o = [0u8; 12];
    for i in 0..32 { let bit = i * 3; let by = bit / 8; let sh = bit % 8; o[by] |= (idx[i] & 7) << sh; if sh > 5 && by + 1 < 12 { o[by + 1] |= (idx[i] & 7) >> (8 - sh); } }
    o
}
fn turbo_unpack3(p: &[u8]) -> [u8; 32] {
    let mut o = [0u8; 32];
    for i in 0..32 { let bit = i * 3; let by = bit / 8; let sh = bit % 8; let mut raw = (p[by] as u16) >> sh; if sh > 5 && by + 1 < 12 { raw |= (p[by + 1] as u16) << (8 - sh); } o[i] = (raw & 7) as u8; }
    o
}
fn q_turbo(x: &[f32], bits: usize) -> Vec<u8> {
    let (bsz, blkbytes, cb): (usize, usize, &[f32]) = if bits == 3 { (12, 14, &TURBO3_CB) } else { (16, 18, &TURBO4_CB) };
    let mut out = Vec::new();
    let mut off = 0;
    while off < x.len() {
        let chunk = &x[off..off + 128];
        let norm = chunk.iter().map(|v| v * v).sum::<f32>().sqrt();
        let inv = if norm > 1e-10 { 1.0 / norm } else { 0.0 };
        let mut t: Vec<f32> = chunk.iter().map(|&v| v * inv).collect();
        fwht(&mut t);
        for b in 0..4 {
            let mut idx = [0u8; 32];
            for i in 0..32 { idx[i] = nearest_f32(cb, t[b * 32 + i]); }
            let mut blk = vec![0u8; blkbytes];
            blk[0..2].copy_from_slice(&f32_to_f16(norm).to_le_bytes());
            if bits == 3 { blk[2..14].copy_from_slice(&turbo_pack3(&idx)); } else { for i in 0..16 { blk[2 + i] = (idx[2 * i] & 0xf) | ((idx[2 * i + 1] & 0xf) << 4); } }
            out.extend_from_slice(&blk);
            let _ = bsz;
        }
        off += 128;
    }
    out
}
fn d_turbo(b: &[u8], y: &mut [f32], bits: usize) {
    let blkbytes = if bits == 3 { 14 } else { 18 };
    let cb: &[f32] = if bits == 3 { &TURBO3_CB } else { &TURBO4_CB };
    let nb = b.len() / blkbytes;
    for blk in 0..nb {
        let base = blk * blkbytes;
        let idx: [u8; 32] = if bits == 3 { turbo_unpack3(&b[base + 2..base + 14]) } else { let mut o = [0u8; 32]; for i in 0..16 { o[2 * i] = b[base + 2 + i] & 0xf; o[2 * i + 1] = b[base + 2 + i] >> 4; } o };
        for i in 0..32 { y[blk * 32 + i] = cb[idx[i] as usize]; }
    }
    let mut off = 0;
    while off < y.len() {
        let chunk = 128usize.min(y.len() - off);
        fwht(&mut y[off..off + chunk]);
        let norm = f16_to_f32(u16::from_le_bytes([b[(off / 32) * blkbytes], b[(off / 32) * blkbytes + 1]]));
        for i in 0..chunk { y[off + i] *= norm; }
        off += 128;
    }
}

fn test(name: &str, bytes32: usize, q: impl Fn(&[f32]) -> (Vec<u8>, f32), d: impl Fn(&[u8], &mut [f32]), tol: f32, nblk: usize) -> bool {
    let mut st = 0x1234_5678u32;
    let mut nxt = || { st ^= st << 13; st ^= st >> 17; st ^= st << 5; (st as f32 / u32::MAX as f32) * 2.0 - 1.0 };
    let n = nblk * 32;
    let x: Vec<f32> = (0..n).map(|_| nxt()).collect();
    let (buf, _s) = q(&x);
    let mut y = vec![0f32; n];
    d(&buf, &mut y);
    let mut e = 0f32;
    for i in 0..n { e = e.max((x[i] - y[i]).abs()); }
    let ok = e <= tol && buf.len() == nblk * bytes32;
    println!("  {name:<10} {bytes32:>3} B/32  max_abs_err={e:.5}  {}  {}", if ok { "PASS" } else { "FAIL" }, if buf.len() == nblk * bytes32 { "" } else { "BAD_SIZE" });
    ok
}
fn test_turbo(name: &str, blkbytes: usize, bits: usize, tol: f32) -> bool {
    let mut st = 0x9e37_79b9u32;
    let mut nxt = || { st ^= st << 13; st ^= st >> 17; st ^= st << 5; (st as f32 / u32::MAX as f32) * 2.0 - 1.0 };
    let n = 256;
    let x: Vec<f32> = (0..n).map(|_| nxt()).collect();
    let buf = q_turbo(&x, bits);
    let mut y = vec![0f32; n];
    d_turbo(&buf, &mut y, bits);
    let mut e = 0f32;
    for i in 0..n { e = e.max((x[i] - y[i]).abs()); }
    let ok = e <= tol && buf.len() == n * blkbytes / 32;
    println!("  {name:<10} {blkbytes:>3} B/32  max_abs_err={e:.5}  {}  (FWHT, norm-scaled)", if ok { "PASS" } else { "FAIL" });
    ok
}

fn main() {
    println!("ROCmFPX / TurboQuant reference dequant round-trip (pure CPU)\n");
    let mut ok = true;
    ok &= test("fp2", 10, |x| q_fp2(x), |b, y| d_fp2(b, y), 0.45, 4);
    ok &= test("fp3", 14, |x| q_fp3(x), |b, y| d_fp3(b, y), 0.30, 4);
    ok &= test("fp4", 18, |x| q_fp4(x, false), |b, y| d_fp4(b, y, false), 0.15, 4);
    ok &= test("fp4_fast", 17, |x| q_fp4(x, true), |b, y| d_fp4(b, y, true), 0.16, 4);
    ok &= test("fp6", 26, |x| q_fp6(x), |b, y| d_fp6(b, y), 0.06, 4);
    ok &= test("fp8", 33, |x| q_fp8(x), |b, y| d_fp8(b, y), 0.02, 4);
    ok &= test_turbo("turbo3", 14, 3, 0.45);
    ok &= test_turbo("turbo4", 18, 4, 0.25);
    println!("\nRESULT: {}", if ok { "PASS - all formats decode"} else { "FAIL" });
    if !ok { std::process::exit(1); }
}

