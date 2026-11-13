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
// Flag hygiene: custom -cflags REPLACE V's default flag set, so any
// profile adding -cflags must restate `-O3 -fno-strict-aliasing`
// (bare -O3 measured ~2x slower). Never -march=native: it pessimize
// V's own codegen 3-5x (measured) — the attributes already cover SIMD.

$if prod {
#flag -I @VMODROOT/src/infer
#flag @VMODROOT/src/infer/vllm_kernels.c
#include "vllm_kernels.h"

fn C.vllm_has_avx2() int
fn C.vllm_matvec_f32(w &f32, x &f32, out &f32, rows int, cols int)
fn C.vllm_q8_dot_f32(qs &i8, x &f32, scale f32, n int) f32
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
			return C.vllm_q8_dot_f32(&block.qs[0], &x[x_off], q80_scale(block),
				qk8_0)
		}
	}
	return q8_0_dot_row(block, x, x_off)
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
