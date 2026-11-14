module infer

import math

// Core NN ops for the Qwen3 forward pass. All f32, caller-owned buffers,
// no allocation inside the hot path (bare-metal: arenas, never GC).
// Hot loops carry [direct_array_access]: bounds are guaranteed by the
// caller (shapes validated once at model load, not per token).

// rmsnorm_in_place normalizes x with weight w and eps.
@[direct_array_access]
pub fn rmsnorm_in_place(mut x []f32, w []f32, eps f64) {
	mut ss := f64(0)
	for i in 0 .. x.len {
		v := f64(x[i])
		ss += v * v
	}
	scale := f32(1.0 / math.sqrt(ss / f64(x.len) + eps))
	for i in 0 .. x.len {
		x[i] = x[i] * scale * w[i]
	}
}

// silu_in_place applies x * sigmoid(x) elementwise (SwiGLU gate half).
@[direct_array_access]
pub fn silu_in_place(mut x []f32) {
	for i in 0 .. x.len {
		v := x[i]
		x[i] = v / (1.0 + f32(math.exp(f64(-v))))
	}
}

// rope_in_place applies rotary embedding to q/k head slices in place.
// head_dim must be even; freq base = cfg.rope_theta.
// cos_sin is a precomputed [2 * (head_dim/2)] table for this position:
// caller builds it once per position and reuses it across all heads.
@[direct_array_access]
pub fn rope_in_place(mut q []f32, mut k []f32, cos_sin []f32, head_dim int) {
	half := head_dim / 2
	for h in 0 .. half {
		c := cos_sin[h]
		s := cos_sin[half + h]
		qi := h * 2
		if qi + 1 < q.len {
			a := q[qi]
			b := q[qi + 1]
			q[qi] = a * c - b * s
			q[qi + 1] = a * s + b * c
		}
		ki := h * 2
		if ki + 1 < k.len {
			a := k[ki]
			b := k[ki + 1]
			k[ki] = a * c - b * s
			k[ki + 1] = a * s + b * c
		}
	}
}

// rope_table fills cos_sin for position pos (see rope_in_place).
pub fn rope_table(mut cos_sin []f32, pos int, head_dim int, theta f64) {
	half := head_dim / 2
	for h in 0 .. half {
		freq := 1.0 / math.pow(theta, f64(h * 2) / f64(head_dim))
		ang := f64(pos) * freq
		cos_sin[h] = f32(math.cos(ang))
		cos_sin[half + h] = f32(math.sin(ang))
	}
}

// matvec computes out = W*x where W is row-major [rows x cols] f32.
// Four accumulators keep independent dependency chains for scalar/OoO
// backends. NOTE: do NOT expect the C compiler to autovectorize this
// shape — under AVX2 gcc's SLP pass turns it into a shuffle-heavy mess
// ~2.4x slower than scalar (measured, see docs/perf.md). SIMD comes from
// the C kernel (function-target AVX2 + runtime dispatch), never from
// -march=native on V code.
@[direct_array_access]
pub fn matvec(w []f32, x []f32, rows int, cols int, mut out []f32) {
	for r in 0 .. rows {
		mut s0 := f32(0)
		mut s1 := f32(0)
		mut s2 := f32(0)
		mut s3 := f32(0)
		base := r * cols
		mut c := 0
		for c + 4 <= cols {
			s0 += w[base + c] * x[c]
			s1 += w[base + c + 1] * x[c + 1]
			s2 += w[base + c + 2] * x[c + 2]
			s3 += w[base + c + 3] * x[c + 3]
			c += 4
		}
		for c < cols {
			s0 += w[base + c] * x[c]
			c++
		}
		out[r] = s0 + s1 + s2 + s3
	}
}

// matvec_blocked is the cache-tiled variant for wide layers
// (vocab projection 2048x151936, FFN up/gate). Tile 32 keeps a
// panel of x in registers across row blocks.
@[direct_array_access]
pub fn matvec_blocked(w []f32, x []f32, rows int, cols int, mut out []f32) {
	for r in 0 .. rows {
		out[r] = 0
	}
	tile := 32
	mut c0 := 0
	for c0 < cols {
		c1 := if c0 + tile < cols { c0 + tile } else { cols }
		for r in 0 .. rows {
			mut s := out[r]
			base := r * cols
			for c in c0 .. c1 {
				s += w[base + c] * x[c]
			}
			out[r] = s
		}
		c0 = c1
	}
}
