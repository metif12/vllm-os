module infer

import math

// Core NN ops for the Qwen3 forward pass. All f32, caller-owned buffers,
// no allocation inside the hot path (bare-metal: arenas, never GC).

// rmsnorm_in_place normalizes x with weight w and eps.
pub fn rmsnorm_in_place(mut x []f32, w []f32, eps f64) {
	mut ss := f64(0)
	for v in x {
		ss += f64(v) * f64(v)
	}
	mean := ss / f64(x.len)
	scale := f32(1.0 / math.sqrt(mean + eps))
	for i in 0 .. x.len {
		x[i] = x[i] * scale * w[i]
	}
}

// silu_in_place applies x * sigmoid(x) elementwise (SwiGLU gate half).
pub fn silu_in_place(mut x []f32) {
	for i in 0 .. x.len {
		v := x[i]
		x[i] = v / (1.0 + f32(math.exp(f64(-v))))
	}
}

// rope_applies rotary embedding to q/k heads in place.
// head_dim must be even; freq base = cfg.rope_theta.
pub fn rope_in_place(mut q []f32, mut k []f32, pos int, head_dim int, theta f64) {
	for h in 0 .. head_dim / 2 {
		freq := 1.0 / math.pow(theta, f64(h * 2) / f64(head_dim))
		ang := f64(pos) * freq
		c := f32(math.cos(ang))
		s := f32(math.sin(ang))
		qi := h * 2
		ki := h * 2
		if qi + 1 < q.len {
			a := q[qi]
			b := q[qi + 1]
			q[qi] = a * c - b * s
			q[qi + 1] = a * s + b * c
		}
		if ki + 1 < k.len {
			a := k[ki]
			b := k[ki + 1]
			k[ki] = a * c - b * s
			k[ki + 1] = a * s + b * c
		}
	}
}

// matvec adds W*x into out where W is row-major [rows x cols] f32.
pub fn matvec(w []f32, x []f32, rows int, cols int, mut out []f32) {
	for r in 0 .. rows {
		mut s := f32(0)
		base := r * cols
		for c in 0 .. cols {
			s += w[base + c] * x[c]
		}
		out[r] = s
	}
}
