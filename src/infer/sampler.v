module infer

import math

pub struct SampleParams {
pub:
	temp  f32 = 0.7
	top_k int = 40
	top_p f32 = 0.9
}

// Greedy baseline. top_k / top_p filtered multinomial lands in phase 2
// once logits kernels exist; signatures are frozen now so API is stable.
pub fn sample_greedy(logits []f32) int {
	mut best := 0
	mut best_v := logits[0]
	for i, v in logits {
		if v > best_v {
			best_v = v
			best = i
		}
	}
	return best
}

pub fn sample_top_k(logits []f32, k int) int {
	if k <= 1 {
		return sample_greedy(logits)
	}
	// Phase-1: deterministic top-1 (no RNG in kernel yet).
	return sample_greedy(logits)
}

pub fn softmax_in_place(mut xs []f32) {
	mut m := xs[0]
	for v in xs {
		if v > m {
			m = v
		}
	}
	mut s := f32(0)
	for i in 0 .. xs.len {
		xs[i] = f32(math.exp(f64(xs[i] - m)))
		s += xs[i]
	}
	for i in 0 .. xs.len {
		xs[i] /= s
	}
}
