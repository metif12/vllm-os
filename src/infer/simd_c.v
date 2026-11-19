module infer

import os

// C kernel bridge: hand-vectorized matvec / Q8 dot / RMSNorm / fast
// softmax. Scalar C fallbacks compile everywhere; on gcc/clang x86 the
// hot functions use function-target AVX2+FMA codegen, so plain portable
// flags already run full speed (runtime dispatch per call).
//
// Disabling (three independent levers, no rebuild required for 2+3):
//  1. Compile-time: any build WITHOUT -prod links no C code at all —
//     every wrapper below degrades to its pure-V kernel. The
//     `host-minimal` profile uses this (plus -Os) for the smallest,
//     fully-auditable binary.
//  2. Runtime env: VLLM_NO_SIMD=1 forces the pure-V kernels even in
//     -prod binaries (per-machine override, no config edit).
//  3. Config: `backend: cpu | cpu-blocked` never touches SIMD; `auto`
//     picks SIMD only when has_avx2() is true.
//
// Flag hygiene (all measured, see docs/perf.md): custom -cflags APPEND
// to V's defaults (verified via -dump-c-flags + identical binaries), so
// plain `-prod -cc gcc` is already optimal. Never -march=native on V
// code: gcc's SLP vectorizer turns the 4-accumulator loop above into a
// vpermd/vblendps swamp ~2.4x slower than scalar (minimal repro + asm in
// the vlang/v issue). The attributes already cover SIMD.

$if prod {
	#flag -I @VMODROOT/src/infer
	#flag @VMODROOT/src/infer/vllm_kernels.c
	#include "vllm_kernels.h"

	fn C.vllm_has_avx2() int
	fn C.vllm_matvec_f32(w &f32, x &f32, out &f32, rows int, cols int)
	fn C.vllm_q8_dot_f32(qs &i8, x &f32, scale f32, n int) f32
	fn C.vllm_q4_0_dot_f32(qs &u8, x &f32, scale f32, n int) f32
	fn C.vllm_q4_1_dot_f32(qs &u8, x &f32, scale f32, min f32) f32
	fn C.vllm_q4_0_matvec_f32(data &u8, x &f32, out &f32, rows int, cols int)
	fn C.vllm_q4_1_matvec_f32(data &u8, x &f32, out &f32, rows int, cols int)
	fn C.vllm_q6_k_matvec_f32(data &u8, x &f32, out &f32, rows int, cols int)
	fn C.vllm_iq4_nl_dot_f32(qs &u8, x &f32, scale f32, n int) f32
	fn C.vllm_iq4_nl_matvec_f32(data &u8, x &f32, out &f32, rows int, cols int)
	fn C.vllm_q2_0_dot_f32(qs &u8, x &f32, scale f32, n int) f32
	fn C.vllm_q2_0_matvec_f32(data &u8, x &f32, out &f32, rows int, cols int)
	fn C.vllm_rmsnorm_f32(x &f32, w &f32, n int, eps f64)
	fn C.vllm_dot_f32(a &f32, b &f32, n int) f32
	fn C.vllm_softmax_scaled_fast_f32(xs &f32, n int, scale f32)
}

// has_avx2 reports whether the C SIMD fast paths are usable right now:
// -prod binary, non-freestanding, CPU with AVX2+FMA, VLLM_NO_SIMD unset.
pub fn has_avx2() bool {
	// Kill-switch first: checked on every toolchain, no C needed.
	if os.getenv('VLLM_NO_SIMD') == '1' {
		return false
	}
	$if prod {
		$if freestanding {
			return false
		} $else {
			return C.vllm_has_avx2() != 0
		}
	} $else {
		return false
	}
}

// matvec_c computes out = W*x via the C kernel, or the pure-V kernel
// outside -prod / without AVX2 / with VLLM_NO_SIMD=1.
// Same shape contract as matvec.
@[direct_array_access]
pub fn matvec_c(w []f32, x []f32, rows int, cols int, mut out []f32) {
	$if prod {
		if has_avx2() {
			C.vllm_matvec_f32(w.data, x.data, out.data, rows, cols)
			return
		}
	}
	matvec(w, x, rows, cols, mut out)
}

// matvec_c_range computes out[r0..r0+n] = W[r0..r0+n]*x on a row block.
// Same dispatch as matvec_c; the range form exists so worker threads can
// share the caller's buffers (offsets resolve inside, no slice copying).
// Contract: r0 + n <= rows, out.len == rows.
@[direct_array_access]
pub fn matvec_c_range(w []f32, x []f32, rows int, cols int, r0 int, n int, mut out []f32) {
	$if prod {
		if has_avx2() {
			unsafe {
				C.vllm_matvec_f32(&w[r0 * cols], x.data, &out[r0], n, cols)
			}
			return
		}
	}
	ws := w[r0 * cols..(r0 + n) * cols]
	// Shared (not cloned) slice: the caller owns out, this range is ours
	// for the duration of the call — same discipline as the C path above.
	mut outs := unsafe { out[r0..r0 + n] }
	matvec(ws, x, n, cols, mut outs)
}

// q8_0_dot_row_c is the C version of q8_0_dot_row (fused int8 dot * scale).
@[direct_array_access]
pub fn q8_0_dot_row_c(block BlockQ80, x []f32, x_off int) f32 {
	$if prod {
		if has_avx2() {
			return C.vllm_q8_dot_f32(&block.qs[0], &x[x_off], q80_scale(block), qk8_0)
		}
	}
	return q8_0_dot_row(block, x, x_off)
}

// q4_0_dot_row_c is the C version of q4_0_dot_row (fused 4-bit dot *
// scale, no materialized row). Split-half layout, AVX2 nibble-unpack with
// scalar fallback; gated like the Q8 bridge so non-prod stays pure V.
@[direct_array_access]
pub fn q4_0_dot_row_c(block BlockQ40, x []f32, x_off int) f32 {
	$if prod {
		if has_avx2() {
			return C.vllm_q4_0_dot_f32(&block.qs[0], &x[x_off], q40_scale(block), qk4_0)
		}
	}
	return q4_0_dot_row(block, x, x_off)
}

// q4_1_dot_row_c is the C version of q4_1_dot_row (fused unsigned-4-bit
// dot * scale + sum(x) * min, no materialized row). Split-half layout,
// AVX2 nibble-unpack with scalar fallback; gated like the Q4_0 bridge.
@[direct_array_access]
pub fn q4_1_dot_row_c(block BlockQ41, x []f32, x_off int) f32 {
	$if prod {
		if has_avx2() {
			return C.vllm_q4_1_dot_f32(&block.qs[0], &x[x_off], q41_scale(block), q41_min(block))
		}
	}
	return q4_1_dot_row(block, x, x_off)
}

// q4_1_matvec_c computes out = W*x for a Q4_1 row-major matrix via the C
// kernel, or the pure-V kernel outside -prod / without AVX2 / with
// VLLM_NO_SIMD=1. Same shape contract as q4_1_matvec (cols divides 32).
@[direct_array_access]
pub fn q4_1_matvec_c(data []u8, rows int, cols int, x []f32, mut out []f32) {
	$if prod {
		if has_avx2() {
			unsafe {
				C.vllm_q4_1_matvec_f32(data.data, x.data, out.data, rows, cols)
			}
			return
		}
	}
	q4_1_matvec(data, rows, cols, x, mut out)
}

// q4_0_matvec_c computes out = W*x for a Q4_0 row-major matrix via the C
// kernel, or the pure-V kernel outside -prod / without AVX2 / with
// VLLM_NO_SIMD=1. Same shape contract as q4_0_matvec (cols divides 32).
@[direct_array_access]
pub fn q4_0_matvec_c(data []u8, rows int, cols int, x []f32, mut out []f32) {
	$if prod {
		if has_avx2() {
			unsafe {
				C.vllm_q4_0_matvec_f32(data.data, x.data, out.data, rows, cols)
			}
			return
		}
	}
	q4_0_matvec(data, rows, cols, x, mut out)
}

// q6_k_matvec_c computes out = W*x for a Q6_K row-major matrix via the C
// kernel, or the pure-V kernel outside -prod / without AVX2 / with
// VLLM_NO_SIMD=1. Same shape contract as q6_k_matvec (cols divides 256).
@[direct_array_access]
pub fn q6_k_matvec_c(data []u8, rows int, cols int, x []f32, mut out []f32) {
	$if prod {
		if has_avx2() {
			unsafe {
				C.vllm_q6_k_matvec_f32(data.data, x.data, out.data, rows, cols)
			}
			return
		}
	}
	q6_k_matvec(data, rows, cols, x, mut out)
}

// q6_k_matvec_c_range computes out[r0..r0+n] for a Q6_K row-major matrix
// on a row block. Offsets resolve inside (no slice copying on the C
// path); the V fallback borrows slices. Same dispatch as q6_k_matvec_c;
// the range form exists so worker threads can share the caller's buffers
// (offsets resolve inside, no slice copying).
// Contract: cols divides 256, r0 + n <= rows, out.len == rows.
@[direct_array_access]
pub fn q6_k_matvec_c_range(data []u8, rows int, cols int, x []f32, r0 int, n int, mut out []f32) {
	bpr := cols / 256
	$if prod {
		if has_avx2() {
			unsafe {
				C.vllm_q6_k_matvec_f32(&data[r0 * bpr * 210], x.data, &out[r0], n, cols)
			}
			return
		}
	}
	ds := data[r0 * bpr * 210..(r0 + n) * bpr * 210]
	mut outs := unsafe { out[r0..r0 + n] }
	q6_k_matvec(ds, n, cols, x, mut outs)
}

// iq4_nl_dot_row_c is the C/AVX2 version of iq4_nl_dot_row (non-linear
// 4-bit codebook, split-half, fused dot). Falls back to pure V outside -prod
// / without AVX2 / with VLLM_NO_SIMD=1.
@[direct_array_access]
pub fn iq4_nl_dot_row_c(block BlockIQ4NL, x []f32, x_off int) f32 {
	$if prod {
		if has_avx2() {
			return C.vllm_iq4_nl_dot_f32(&block.qs[0], &x[x_off], f16bits_to_f32(block.d), qk4_nl)
		}
	}
	return iq4_nl_dot_row(block, x, x_off)
}

// iq4_nl_matvec_c computes out = W*x for an IQ4_NL row-major matrix via the
// C kernel (AVX2 pshufb codebook lookup), or pure V otherwise.
@[direct_array_access]
pub fn iq4_nl_matvec_c(data []u8, rows int, cols int, x []f32, mut out []f32) {
	$if prod {
		if has_avx2() {
			unsafe {
				C.vllm_iq4_nl_matvec_f32(data.data, x.data, out.data, rows, cols)
			}
			return
		}
	}
	iq4_nl_matvec(data, rows, cols, x, mut out)
}

// iq4_nl_matvec_c_range computes out[r0..r0+n] on a row block (offsets
// resolve inside so workers can share the caller's buffers). Same dispatch
// as iq4_nl_matvec_c; used by the threaded vocab-head path.
@[direct_array_access]
pub fn iq4_nl_matvec_c_range(data []u8, rows int, cols int, x []f32, r0 int, n int, mut out []f32) {
	bpr := cols / qk4_nl
	$if prod {
		if has_avx2() {
			unsafe {
				C.vllm_iq4_nl_matvec_f32(&data[r0 * bpr * iq4_nl_block_bytes], x.data, &out[r0], n,
					cols)
			}
			return
		}
	}
	ds := data[r0 * bpr * iq4_nl_block_bytes..(r0 + n) * bpr * iq4_nl_block_bytes]
	mut outs := unsafe { out[r0..r0 + n] }
	iq4_nl_matvec(ds, n, cols, x, mut outs)
}

// q2_0_dot_row_c is the C/AVX2 version of q2_0_dot_row (signed-scale 2-bit,
// 64 codes per block). Falls back to pure V outside -prod / without AVX2.
@[direct_array_access]
pub fn q2_0_dot_row_c(block BlockQ20, x []f32, x_off int) f32 {
	$if prod {
		if has_avx2() {
			return C.vllm_q2_0_dot_f32(&block.qs[0], &x[x_off], f16bits_to_f32(block.d), qk2_0)
		}
	}
	return q2_0_dot_row(block, x, x_off)
}

// q2_0_matvec_c computes out = W*x for a Q2_0 row-major matrix via the C
// kernel (AVX2 FMA over a local dequant scratch), or pure V otherwise.
@[direct_array_access]
pub fn q2_0_matvec_c(data []u8, rows int, cols int, x []f32, mut out []f32) {
	$if prod {
		if has_avx2() {
			unsafe {
				C.vllm_q2_0_matvec_f32(data.data, x.data, out.data, rows, cols)
			}
			return
		}
	}
	q2_0_matvec(data, rows, cols, x, mut out)
}

// q2_0_matvec_c_range computes out[r0..r0+n] on a row block (offsets resolve
// inside so workers share the caller's buffers). Used by the threaded head.
@[direct_array_access]
pub fn q2_0_matvec_c_range(data []u8, rows int, cols int, x []f32, r0 int, n int, mut out []f32) {
	bpr := cols / qk2_0
	$if prod {
		if has_avx2() {
			unsafe {
				C.vllm_q2_0_matvec_f32(&data[r0 * bpr * q2_0_block_bytes], x.data, &out[r0], n,
					cols)
			}
			return
		}
	}
	ds := data[r0 * bpr * q2_0_block_bytes..(r0 + n) * bpr * q2_0_block_bytes]
	mut outs := unsafe { out[r0..r0 + n] }
	q2_0_matvec(ds, n, cols, x, mut outs)
}

// q2_0_r8_dot_c runs the R8 block dot through the EXISTING int8 C kernel —
// the repack exists precisely so no second math path is needed. Outside
// -prod / without AVX2 it uses the same int8 formula in pure V.
@[direct_array_access]
pub fn q2_0_r8_dot_c(data []u8, blk int, x []f32, x_off int) f32 {
	scale := q2_0_r8_scale(data, blk)
	$if prod {
		if has_avx2() {
			off := blk * q2_0_r8_block_bytes + 2
			return C.vllm_q8_dot_f32(&data[off], &x[x_off], scale, qk2_0)
		}
	}
	return q2_0_r8_dot_block(data, blk, x, x_off)
}

// q2_0_r8_matvec_c computes out = W*x over an R8-repacked Q2_0 matrix.
@[direct_array_access]
pub fn q2_0_r8_matvec_c(data []u8, rows int, cols int, x []f32, mut out []f32) {
	bpr := cols / qk2_0
	for r in 0 .. rows {
		mut s := f32(0)
		for b in 0 .. bpr {
			blk := r * bpr + b
			s += q2_0_r8_dot_c(data, blk, x, b * qk2_0)
		}
		out[r] = s
	}
}

// rmsnorm_c normalizes x with weight w and eps via the C kernel.
@[direct_array_access]
pub fn rmsnorm_c(mut x []f32, w []f32, eps f64) {
	$if prod {
		if has_avx2() {
			C.vllm_rmsnorm_f32(x.data, w.data, x.len, eps)
			return
		}
	}
	rmsnorm_in_place(mut x, w, eps)
}

// softmax_scaled_fast_in_place applies softmax(x * scale) with the
// branchless approximate exp (no libm, SIMD-friendly). Sampling only:
// probabilities carry ~0.5% worst-case error — fine for top-k/p draws,
// never for scoring. Exact path stays softmax_scaled_in_place.
// Outside -prod this IS the exact path (no C linked).
@[direct_array_access]
pub fn softmax_scaled_fast_in_place(mut xs []f32, scale f32) {
	$if prod {
		C.vllm_softmax_scaled_fast_f32(xs.data, xs.len, scale)
	} $else {
		softmax_scaled_in_place(mut xs, scale)
	}
}
